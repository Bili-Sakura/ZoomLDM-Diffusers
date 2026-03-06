"""
Tests for the ZoomLDM diffusers custom pipeline.

These tests validate the pipeline structure and inference flow using
lightweight mock components so that no GPU or model weights are needed.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest

from pipeline_zoomldm import ZoomLDMPipeline, ZoomLDMPipelineOutput
from diffusers import DDIMScheduler


# ---------------------------------------------------------------------------
# Minimal mock components that mimic the real model interfaces
# ---------------------------------------------------------------------------


class MockUNet(nn.Module):
    """Mimics UNetModel: forward(x, t, context=None) -> noise prediction."""

    def __init__(self, channels=3):
        super().__init__()
        self.dummy = nn.Linear(1, 1)  # so .parameters() is non-empty

    def forward(self, x, t, context=None, **kwargs):
        # Return tensor of same shape as input (noise prediction)
        return torch.randn_like(x)


class MockVAE(nn.Module):
    """Mimics VQModelInterface: decode(z) -> image."""

    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.dummy = nn.Linear(1, 1)

    def decode(self, z, **kwargs):
        b, c, h, w = z.shape
        # Upsample to image space
        return torch.randn(b, 3, h * self.scale, w * self.scale, device=z.device)


class MockConditioningEncoder(nn.Module):
    """Mimics EmbeddingViT2_5: encode(batch) -> conditioning tensor."""

    def __init__(self, feat_key="ssl_feat", mag_key="mag", hidden=512, seq_len=65):
        super().__init__()
        self.feat_key = feat_key
        self.mag_key = mag_key
        self.hidden = hidden
        self.seq_len = seq_len
        self.p_uncond = 0
        self.dummy = nn.Linear(1, 1)

    def encode(self, batch):
        feat = batch[self.feat_key]
        if isinstance(feat, list):
            bs = len(feat)
        else:
            bs = feat.shape[0]
        device = feat[0].device if isinstance(feat, list) else feat.device
        return torch.randn(bs, self.seq_len, self.hidden, device=device)

    def forward(self, batch):
        return self.encode(batch)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scheduler():
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type="epsilon",
        steps_offset=1,
    )


@pytest.fixture
def pipeline(scheduler):
    return ZoomLDMPipeline(
        unet=MockUNet(),
        vae=MockVAE(),
        conditioning_encoder=MockConditioningEncoder(),
        scheduler=scheduler,
        scale_factor=1.0,
        conditioning_key="crossattn",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestZoomLDMPipelineInit:
    def test_init(self, pipeline):
        assert pipeline.unet is not None
        assert pipeline.vae is not None
        assert pipeline.conditioning_encoder is not None
        assert pipeline.scheduler is not None
        assert pipeline.scale_factor == 1.0
        assert pipeline.conditioning_key == "crossattn"

    def test_registered_modules(self, pipeline):
        """Modules should be accessible as pipeline attributes."""
        assert isinstance(pipeline.unet, MockUNet)
        assert isinstance(pipeline.vae, MockVAE)
        assert isinstance(pipeline.conditioning_encoder, MockConditioningEncoder)
        assert isinstance(pipeline.scheduler, DDIMScheduler)


class TestEncodeConditioning:
    def test_encode(self, pipeline):
        ssl_features = torch.randn(2, 1024, 4, 4)
        magnification = torch.tensor([0, 1])
        cond = pipeline.encode_conditioning(ssl_features, magnification)
        assert cond.shape == (2, 65, 512)

    def test_encode_list_input(self, pipeline):
        ssl_features = [torch.randn(1024, 4, 4), torch.randn(1024, 4, 4)]
        magnification = torch.tensor([0, 1])
        cond = pipeline.encode_conditioning(ssl_features, magnification)
        assert cond.shape[0] == 2


class TestDecodeLatents:
    def test_decode(self, pipeline):
        latents = torch.randn(2, 3, 64, 64)
        images = pipeline.decode_latents(latents)
        assert images.shape == (2, 3, 256, 256)


class TestPipelineCall:
    def test_call_pil_output(self, pipeline):
        ssl_features = torch.randn(2, 1024, 4, 4)
        magnification = torch.tensor([0, 1])

        output = pipeline(
            ssl_features=ssl_features,
            magnification=magnification,
            num_inference_steps=2,
            guidance_scale=2.0,
        )

        assert isinstance(output, ZoomLDMPipelineOutput)
        assert isinstance(output.images, list)
        assert len(output.images) == 2
        from PIL import Image
        assert all(isinstance(img, Image.Image) for img in output.images)

    def test_call_np_output(self, pipeline):
        ssl_features = torch.randn(1, 1024, 4, 4)
        magnification = torch.tensor([0])

        output = pipeline(
            ssl_features=ssl_features,
            magnification=magnification,
            num_inference_steps=2,
            guidance_scale=2.0,
            output_type="np",
        )

        assert isinstance(output.images, np.ndarray)
        assert output.images.shape[0] == 1
        assert output.images.shape[-1] == 3  # HWC

    def test_call_pt_output(self, pipeline):
        ssl_features = torch.randn(1, 1024, 4, 4)
        magnification = torch.tensor([0])

        output = pipeline(
            ssl_features=ssl_features,
            magnification=magnification,
            num_inference_steps=2,
            guidance_scale=2.0,
            output_type="pt",
        )

        assert isinstance(output.images, torch.Tensor)
        assert output.images.shape == (1, 3, 256, 256)

    def test_call_tuple_output(self, pipeline):
        ssl_features = torch.randn(1, 1024, 4, 4)
        magnification = torch.tensor([0])

        result = pipeline(
            ssl_features=ssl_features,
            magnification=magnification,
            num_inference_steps=2,
            return_dict=False,
        )

        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_call_with_generator(self, pipeline):
        ssl_features = torch.randn(1, 1024, 4, 4)
        magnification = torch.tensor([0])
        gen = torch.Generator(device="cpu").manual_seed(42)

        output = pipeline(
            ssl_features=ssl_features,
            magnification=magnification,
            num_inference_steps=2,
            generator=gen,
            output_type="pt",
        )

        assert output.images.shape[0] == 1

    def test_call_with_custom_latents(self, pipeline):
        ssl_features = torch.randn(1, 1024, 4, 4)
        magnification = torch.tensor([0])
        custom_latents = torch.ones(1, 3, 64, 64) * 0.5

        output = pipeline(
            ssl_features=ssl_features,
            magnification=magnification,
            num_inference_steps=2,
            latents=custom_latents,
            output_type="pt",
        )

        assert output.images.shape[0] == 1

    def test_call_invalid_output_type(self, pipeline):
        ssl_features = torch.randn(1, 1024, 4, 4)
        magnification = torch.tensor([0])

        with pytest.raises(ValueError, match="Unknown output_type"):
            pipeline(
                ssl_features=ssl_features,
                magnification=magnification,
                num_inference_steps=2,
                output_type="invalid",
            )


class TestSchedulerConfig:
    def test_scheduler_timesteps(self, scheduler):
        """Verify scheduler timesteps match original DDIM sampler."""
        scheduler.set_timesteps(50)
        timesteps = scheduler.timesteps

        # Original DDIM uses: range(0, 1000, 20) + 1 = 1, 21, 41, ..., 981
        # Reversed: 981, 961, 941, ..., 1
        assert timesteps[0].item() == 981
        assert timesteps[-1].item() == 1
        assert len(timesteps) == 50
