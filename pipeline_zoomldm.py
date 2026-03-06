"""
Custom diffusers pipeline for ZoomLDM multi-scale image generation.

Usage:
    # Load from original config + checkpoint
    pipe = ZoomLDMPipeline.from_single_file(
        config_path="brca/config.yaml",
        ckpt_path="brca/weights.ckpt",
    )

    # Or from a pre-loaded LatentDiffusion model
    pipe = ZoomLDMPipeline.from_ldm_model(model)

    # Generate images
    images = pipe(
        ssl_features=batch["ssl_feat"],
        magnification=batch["mag"],
        num_inference_steps=50,
        guidance_scale=2.0,
    ).images
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import BaseOutput
from PIL import Image


@dataclass
class ZoomLDMPipelineOutput(BaseOutput):
    """
    Output class for ZoomLDM pipeline.

    Args:
        images: List of PIL images or numpy array of generated images.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class ZoomLDMPipeline(DiffusionPipeline):
    """
    Pipeline for multi-scale image generation with ZoomLDM.

    This pipeline wraps the ZoomLDM model components using the native
    huggingface/diffusers ``DiffusionPipeline`` interface, replacing custom
    samplers with the diffusers ``DDIMScheduler``.

    Args:
        unet: The UNet denoising model (``UNetModel`` from openaimodel).
        vae: The first-stage autoencoder (``VQModelInterface``).
        conditioning_encoder: The conditioning encoder
            (``EmbeddingViT2_5``).
        scheduler: A diffusers noise scheduler (e.g. ``DDIMScheduler``).
        scale_factor: Latent space scaling factor (default: 1.0).
        conditioning_key: Type of conditioning ("crossattn", "concat",
            "hybrid").
    """

    model_cpu_offload_seq = "conditioning_encoder->unet->vae"

    def __init__(
        self,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        conditioning_encoder: torch.nn.Module,
        scheduler: DDIMScheduler,
        scale_factor: float = 1.0,
        conditioning_key: str = "crossattn",
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            conditioning_encoder=conditioning_encoder,
            scheduler=scheduler,
        )
        self.scale_factor = scale_factor
        self.conditioning_key = conditioning_key

    @property
    def device(self) -> torch.device:
        """Return the device of the pipeline's parameters."""
        try:
            return next(self.unet.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @classmethod
    def from_single_file(cls, config_path, ckpt_path, device=None, **kwargs):
        """
        Load a ``ZoomLDMPipeline`` from original ZoomLDM config and
        checkpoint files.

        Args:
            config_path: Path to the YAML config file.
            ckpt_path: Path to the model checkpoint (``.ckpt`` or
                ``.pt``).
            device: Device to load the model onto.

        Returns:
            A ``ZoomLDMPipeline`` instance.

        Example::

            from huggingface_hub import hf_hub_download

            ckpt = hf_hub_download(
                "StonyBrook-CVLab/ZoomLDM", "brca/weights.ckpt"
            )
            cfg = hf_hub_download(
                "StonyBrook-CVLab/ZoomLDM", "brca/config.yaml"
            )
            pipe = ZoomLDMPipeline.from_single_file(cfg, ckpt)
            pipe = pipe.to("cuda")
        """
        from omegaconf import OmegaConf

        from ldm.util import instantiate_from_config

        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        pipe = cls.from_ldm_model(model)

        if device is not None:
            pipe = pipe.to(device)

        return pipe

    @classmethod
    def from_ldm_model(cls, model):
        """
        Create a ``ZoomLDMPipeline`` from an existing ``LatentDiffusion``
        model instance.

        Args:
            model: A ``LatentDiffusion`` model.

        Returns:
            A ``ZoomLDMPipeline`` instance.
        """
        # Apply EMA weights if available
        if hasattr(model, "use_ema") and model.use_ema:
            model.model_ema.copy_to(model.model)

        # Extract components
        unet = model.model.diffusion_model
        vae = model.first_stage_model
        conditioning_encoder = model.cond_stage_model

        # Disable classifier-free dropout in conditioning encoder
        if hasattr(conditioning_encoder, "p_uncond"):
            conditioning_encoder.p_uncond = 0

        # Determine scale_factor
        sf = model.scale_factor
        if isinstance(sf, torch.Tensor):
            sf = sf.item()

        # Create a diffusers DDIMScheduler that matches the original
        # noise schedule.
        #   - The original "linear" beta schedule uses:
        #       betas = linspace(sqrt(start), sqrt(end), T) ** 2
        #     which corresponds to "scaled_linear" in diffusers.
        #   - steps_offset=1 replicates the +1 shift used by the
        #     original DDIM sampler.
        scheduler = DDIMScheduler(
            num_train_timesteps=model.num_timesteps,
            beta_start=model.linear_start,
            beta_end=model.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="epsilon",
            steps_offset=1,
        )

        # Determine the conditioning key
        conditioning_key = "crossattn"
        if hasattr(model, "model") and hasattr(model.model, "conditioning_key"):
            conditioning_key = model.model.conditioning_key or "crossattn"

        return cls(
            unet=unet,
            vae=vae,
            conditioning_encoder=conditioning_encoder,
            scheduler=scheduler,
            scale_factor=sf,
            conditioning_key=conditioning_key,
        )

    def encode_conditioning(self, ssl_features, magnification):
        """
        Encode conditioning inputs through the conditioning encoder.

        Args:
            ssl_features: SSL feature tensors (e.g. UNI or DINO-v2
                embeddings).
            magnification: Integer magnification level tensor.

        Returns:
            Encoded conditioning tensor.
        """
        device = self.device
        cond_dict = {
            self.conditioning_encoder.feat_key: ssl_features,
            self.conditioning_encoder.mag_key: magnification.to(device),
        }

        if hasattr(self.conditioning_encoder, "encode"):
            return self.conditioning_encoder.encode(cond_dict)
        return self.conditioning_encoder(cond_dict)

    def decode_latents(self, latents):
        """
        Decode latent representations to images using the VAE.

        Args:
            latents: Latent tensor from the diffusion process.

        Returns:
            Image tensor in ``[-1, 1]`` range.
        """
        latents = (1.0 / self.scale_factor) * latents
        return self.vae.decode(latents)

    @torch.no_grad()
    def __call__(
        self,
        ssl_features: Union[torch.Tensor, list],
        magnification: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        latent_shape: tuple = (3, 64, 64),
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        """
        Generate images conditioned on SSL features and magnification
        level.

        Args:
            ssl_features: SSL feature tensor(s) for conditioning.
                Shape depends on the magnification level.
            magnification: Integer magnification levels
                (0=20x, 1=10x, 2=5x, 3=2.5x, 4=1.25x).
            num_inference_steps: Number of denoising steps (default: 50).
            guidance_scale: Classifier-free guidance scale (default: 2.0).
            latent_shape: Shape of each latent sample
                (default: ``(3, 64, 64)``).
            generator: Optional random number generator for
                reproducibility.
            latents: Optional pre-initialized latent noise tensor.
            output_type: Output format — ``"pil"``, ``"np"``, or
                ``"pt"`` (default: ``"pil"``).
            return_dict: Whether to return a ``ZoomLDMPipelineOutput``
                or a tuple (default: ``True``).

        Returns:
            ``ZoomLDMPipelineOutput`` with generated images, or a tuple.

        Example::

            pipe = ZoomLDMPipeline.from_single_file(cfg, ckpt)
            pipe = pipe.to("cuda")
            output = pipe(
                ssl_features=batch["ssl_feat"].to("cuda"),
                magnification=batch["mag"].to("cuda"),
                num_inference_steps=50,
                guidance_scale=2.0,
            )
            images = output.images
        """
        device = self.device
        dtype = next(self.unet.parameters()).dtype

        # Determine batch size
        if isinstance(ssl_features, list):
            batch_size = len(ssl_features)
        elif isinstance(ssl_features, torch.Tensor):
            batch_size = ssl_features.shape[0]
        else:
            batch_size = 1

        # 1. Encode conditioning
        cc = self.encode_conditioning(ssl_features, magnification)
        uc = torch.zeros_like(cc)

        # 2. Prepare latents
        if latents is None:
            latents = torch.randn(
                (batch_size, *latent_shape),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        # 3. Set up scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Denoising loop
        for t in self.progress_bar(timesteps):
            latent_model_input = torch.cat([latents, latents])
            t_batch = t.expand(latent_model_input.shape[0])
            cond_input = torch.cat([uc, cc])

            # Predict noise with the UNet
            with torch.amp.autocast(device_type=device.type, enabled=device.type != "cpu"):
                if self.conditioning_key == "crossattn":
                    noise_pred = self.unet(
                        latent_model_input,
                        t_batch,
                        context=cond_input,
                    )
                elif self.conditioning_key == "concat":
                    noise_pred = self.unet(
                        torch.cat(
                            [latent_model_input, cond_input], dim=1
                        ),
                        t_batch,
                    )
                elif self.conditioning_key == "hybrid":
                    raise NotImplementedError(
                        "Hybrid conditioning requires c_concat and "
                        "c_crossattn to be passed separately. Use the "
                        "original LatentDiffusion model for hybrid "
                        "conditioning."
                    )
                else:
                    noise_pred = self.unet(latent_model_input, t_batch)

            # Classifier-free guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # Scheduler step
            latents = self.scheduler.step(
                noise_pred, t, latents, generator=generator
            ).prev_sample

        # 5. Decode latents to images
        images = self.decode_latents(latents)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

        # 6. Convert output format
        if output_type == "pt":
            pass
        elif output_type == "np":
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        elif output_type == "pil":
            images_np = images.cpu().permute(0, 2, 3, 1).float().numpy()
            images = [
                Image.fromarray((img * 255).astype(np.uint8))
                for img in images_np
            ]
        else:
            raise ValueError(
                f"Unknown output_type '{output_type}'. "
                "Use 'pil', 'np', or 'pt'."
            )

        if not return_dict:
            return (images,)

        return ZoomLDMPipelineOutput(images=images)
