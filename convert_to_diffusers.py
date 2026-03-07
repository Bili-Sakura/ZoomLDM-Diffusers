#!/usr/bin/env python3
"""
Convert ZoomLDM from original checkpoint format to diffusers-style format.

Usage:
    python convert_to_diffusers.py \\
        --repo_id StonyBrook-CVLab/ZoomLDM \\
        --variant brca \\
        --output_dir /root/worksapce/models/BiliSakura/ZoomLDM

Or with local paths:
    python convert_to_diffusers.py \\
        --config_path path/to/config.yaml \\
        --ckpt_path path/to/weights.ckpt \\
        --output_dir /root/worksapce/models/BiliSakura/ZoomLDM
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add project root for imports
_script_dir = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(_script_dir))

from safetensors.torch import save_file as save_safetensors


def save_custom_component(component, config, save_path: Path, safe_serialization: bool = True):
    """Save a custom LDM component (unet, vae, conditioning_encoder) in diffusers format."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save config for reconstruction (OmegaConf -> dict for JSON)
    config_path = save_path / "config.json"
    try:
        if hasattr(config, "to_container"):
            config_to_save = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, dict):
            config_to_save = config
        else:
            config_to_save = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    except Exception:
        config_to_save = OmegaConf.to_container(config, resolve=True)

    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)

    # Save weights
    state_dict = component.state_dict()
    if safe_serialization:
        weights_path = save_path / "diffusion_pytorch_model.safetensors"
        save_safetensors(state_dict, weights_path)
    else:
        weights_path = save_path / "diffusion_pytorch_model.bin"
        torch.save(state_dict, weights_path)


def main():
    parser = argparse.ArgumentParser(description="Convert ZoomLDM to diffusers format")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo_id", type=str, help="HuggingFace repo ID (e.g. StonyBrook-CVLab/ZoomLDM)")
    group.add_argument("--config_path", type=str, help="Path to config.yaml (use with --ckpt_path)")
    parser.add_argument("--ckpt_path", type=str, help="Path to weights.ckpt (use with --config_path)")
    parser.add_argument("--variant", type=str, default="brca", help="Variant: brca, naip, or cdm_brca (when using --repo_id)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for diffusers-format model")
    parser.add_argument("--safe_serialization", action="store_true", default=True, help="Use safetensors (default: True)")
    parser.add_argument("--no_safe_serialization", action="store_true", help="Use PyTorch .bin instead of safetensors")
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading (default: cpu)")
    args = parser.parse_args()

    safe_serialization = not args.no_safe_serialization

    if args.repo_id:
        from huggingface_hub import hf_hub_download
        repo_id = args.repo_id
        variant = args.variant
        config_path = hf_hub_download(repo_id=repo_id, filename=f"{variant}/config.yaml")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=f"{variant}/weights.ckpt")
        print(f"Downloaded config: {config_path}")
        print(f"Downloaded checkpoint: {ckpt_path}")
    else:
        config_path = args.config_path
        ckpt_path = args.ckpt_path
        if not ckpt_path:
            raise ValueError("--ckpt_path required when using --config_path")

    # Patch config: remove unet ckpt_path so we load from main checkpoint
    config = OmegaConf.load(config_path)
    if "model" in config and "params" in config.model:
        unet_cfg = config.model.params.get("unet_config")
        if unet_cfg is not None:
            params = OmegaConf.to_container(unet_cfg.get("params") or {}, resolve=True)
            if isinstance(params, dict):
                params.pop("ckpt_path", None)
                params.pop("ignore_keys", None)
                config.model.params.unet_config.params = OmegaConf.create(params)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            OmegaConf.save(config, f.name)
            config_path = f.name

    from pipeline_zoomldm import ZoomLDMPipeline

    print("Loading ZoomLDM pipeline...")
    # Use device=None to avoid diffusers .to() which expects standard module attributes
    pipe = ZoomLDMPipeline.from_single_file(config_path, ckpt_path, device=None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(config_path)
    model_config = config.model.params

    print(f"Saving to {output_dir}...")

    # 1. Save scheduler (standard diffusers component)
    scheduler_path = output_dir / "scheduler"
    pipe.scheduler.save_pretrained(scheduler_path)
    print(f"  Saved scheduler -> {scheduler_path}")

    # 2. Save unet
    unet_config = model_config.unet_config
    unet_path = output_dir / "unet"
    save_custom_component(pipe.unet, unet_config, unet_path, safe_serialization)
    print(f"  Saved unet -> {unet_path}")

    # 3. Save vae
    vae_config = model_config.first_stage_config
    vae_path = output_dir / "vae"
    save_custom_component(pipe.vae, vae_config, vae_path, safe_serialization)
    print(f"  Saved vae -> {vae_path}")

    # 4. Save conditioning_encoder
    cond_config = model_config.cond_stage_config
    cond_path = output_dir / "conditioning_encoder"
    save_custom_component(pipe.conditioning_encoder, cond_config, cond_path, safe_serialization)
    print(f"  Saved conditioning_encoder -> {cond_path}")

    # 5. Save pipeline config (model_index.json)
    model_index = {
        "_class_name": "ZoomLDMPipeline",
        "_diffusers_version": "0.25.0",
        "conditioning_encoder": ["pipeline_zoomldm", "ZoomLDMPipeline"],
        "scheduler": ["diffusers", "DDIMScheduler"],
        "unet": ["pipeline_zoomldm", "ZoomLDMPipeline"],
        "vae": ["pipeline_zoomldm", "ZoomLDMPipeline"],
        "scale_factor": pipe.scale_factor,
        "conditioning_key": pipe.conditioning_key,
    }
    with open(output_dir / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"  Saved model_index.json")

    # 6. Copy pipeline_zoomldm.py for custom pipeline loading
    pipeline_src = _script_dir / "pipeline_zoomldm.py"
    pipeline_dst = output_dir / "pipeline_zoomldm"
    pipeline_dst.mkdir(exist_ok=True)
    shutil.copy(pipeline_src, pipeline_dst / "pipeline_zoomldm.py")
    # Add __init__.py for package
    (pipeline_dst / "__init__.py").write_text("from .pipeline_zoomldm import ZoomLDMPipeline\n")
    print(f"  Saved pipeline_zoomldm")

    # 7. Save scale_factor and conditioning_key in a small config for from_pretrained
    pipeline_config = {
        "scale_factor": pipe.scale_factor,
        "conditioning_key": pipe.conditioning_key,
    }
    with open(output_dir / "pipeline_config.json", "w") as f:
        json.dump(pipeline_config, f, indent=2)

    print(f"\nDone! Diffusers-format model saved to {output_dir}")
    print("Load with: ZoomLDMPipeline.from_pretrained('<path>')")


if __name__ == "__main__":
    main()
