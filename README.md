# Modular Z-Image Upscale

Tiled image upscaling for Z-Image using [MultiDiffusion](https://arxiv.org/abs/2302.08113) latent-space blending. Built with [Modular Diffusers](https://huggingface.co/blog/modular-diffusers).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/akshan-main/a2fa5c3dc2af23124749966513a7f3c7/modular-zimage-upscale_demo.ipynb)
[![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-Hub-yellow)](https://huggingface.co/akshan-main/modular-zimage-upscale)

## What it does

- Image upscaling at any scale factor using Z-Image
- MultiDiffusion: blends overlapping transformer tile predictions in latent space with cosine weights. No visible seams
- Optional ControlNet conditioning (general-purpose, not tile-specific)
- Progressive upscaling: splits large scale factors into multiple passes
- Auto-strength scaling per pass
- Few-step inference with Z-Image Turbo (4-8 steps)

## Install

```bash
pip install git+https://github.com/huggingface/diffusers.git transformers accelerate safetensors
```

Requires diffusers from main (modular diffusers support).

## Usage

### From HuggingFace Hub

```python
from diffusers import ModularPipeline
import torch

pipe = ModularPipeline.from_pretrained(
    "akshan-main/modular-zimage-upscale",
    trust_remote_code=True,
)
pipe.load_components(torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = ...  # your PIL image

result = pipe(
    prompt="high quality, detailed, sharp",
    image=image,
    scale_factor=2.0,
    num_inference_steps=8,
    generator=torch.Generator("cuda").manual_seed(42),
    output="images",
)
result[0].save("upscaled.png")
```

### 4x upscale

For Z-Image, single-pass 4x is more faithful than progressive multi-pass.

```python
result = pipe(
    prompt="high quality, detailed, sharp",
    image=image,
    scale_factor=4.0,
    progressive=False,
    generator=torch.Generator("cuda").manual_seed(42),
    output="images",
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Input image (PIL) |
| `prompt` | `""` | Text prompt |
| `negative_prompt` | `None` | Negative text prompt |
| `scale_factor` | `2.0` | Scale multiplier |
| `strength` | `0.4` | Denoise strength. Ignored when `auto_strength=True` |
| `num_inference_steps` | `8` | Denoising steps (Z-Image Turbo is fast) |
| `tile_size` | `64` | Tile size in latent pixels |
| `tile_overlap` | `8` | Tile overlap in latent pixels |
| `control_image` | `None` | ControlNet conditioning image (optional) |
| `controlnet_conditioning_scale` | `0.75` | ControlNet strength |
| `progressive` | `True` | Split large upscale factors into multiple passes. `False` often works better for Z-Image |
| `auto_strength` | `True` | Auto-scale strength per pass |
| `generator` | `None` | Torch generator for reproducibility |

## Observations

- Z-Image Turbo converges by 4 steps. More steps don't noticeably change the output
- Strength, ControlNet scale, and negative prompt have subtle effects due to Turbo's distilled nature
- Z-Image's ControlNet Union is general-purpose, not tile-specific. Running without ControlNet can produce results as close or closer to the input
- Progressive mode compounds drift across passes. Single-pass 4x is more faithful
- No visible tile seams at any tile size

## Limitations

- Z-Image Turbo (6B) needs ~22GB VRAM without ControlNet. With ControlNet, use A100 or similar
- ControlNet Union may not improve upscaling faithfulness. Use lower strength instead
- Tiles smaller than 32 latent pixels may produce artifacts
- Very small inputs (below 256px) produce distortion
- Z-Image Turbo's CFG is disabled by default. Negative prompts have limited effect
- Not suitable for text, line art, or pixel art

## Architecture

```
MultiDiffusionUpscaleBlocks (SequentialPipelineBlocks)
  text_encoder    ZImageTextEncoderStep (reused)
  upscale         ZImageUpscaleStep (Lanczos)
  multidiffusion  ZImageMultiDiffusionStep
                  - VAE encode full image
                  - Per timestep: transformer on each latent tile (+optional ControlNet), cosine-weighted blend
                  - VAE decode full latents
```

## Project structure

```
utils_tiling.py              Latent tile planning, cosine weights
input.py                     Upscale step
denoise.py                   MultiDiffusion step, ControlNet integration
modular_blocks.py            Block composition
modular_pipeline.py          Pipeline class
hub_block/                   HuggingFace Hub block (consolidated single file)
```

## References

- [MultiDiffusion](https://arxiv.org/abs/2302.08113) (Bar-Tal et al., 2023)
- [Modular Diffusers](https://huggingface.co/blog/modular-diffusers)
- [Modular Diffusers contribution call](https://github.com/huggingface/diffusers/issues/13295)
