# Modular Z-Image Upscale

Tiled image upscaling for Z-Image using [MultiDiffusion](https://arxiv.org/abs/2302.08113) latent-space blending. Built with [Modular Diffusers](https://huggingface.co/blog/modular-diffusers).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/akshan-main/4965498a6cea46b42dc691318154d0d6/modular-zimage-upscale_demo.ipynb)
[![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-Hub-yellow)](https://huggingface.co/akshan-main/modular-zimage-upscale)

## What it does

- Image upscaling at any scale factor using Z-Image
- MultiDiffusion: blends overlapping transformer tile predictions in latent space with cosine weights. No visible seams
- Optional ControlNet conditioning for faithful upscaling
- Progressive upscaling: automatically splits large scale factors into multiple passes
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
from diffusers import ModularPipelineBlocks
from diffusers.models.controlnets import ZImageControlNetModel
from huggingface_hub import hf_hub_download
import torch

blocks = ModularPipelineBlocks.from_pretrained(
    "akshan-main/modular-zimage-upscale",
    trust_remote_code=True,
)

pipe = blocks.init_pipeline("Tongyi-MAI/Z-Image-Turbo")
pipe.load_components(torch_dtype=torch.bfloat16)

controlnet = ZImageControlNetModel.from_single_file(
    hf_hub_download(
        "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union",
        filename="Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
    ),
    torch_dtype=torch.bfloat16,
)
controlnet = ZImageControlNetModel.from_transformer(controlnet, pipe.transformer)
pipe.update_components(controlnet=controlnet)
pipe.enable_model_cpu_offload()

image = ...  # your PIL image

result = pipe(
    prompt="high quality, detailed, sharp",
    image=image,
    control_image=image,
    controlnet_conditioning_scale=0.75,
    scale_factor=2.0,
    num_inference_steps=8,
    generator=torch.Generator("cuda").manual_seed(42),
    output="images",
)
result[0].save("upscaled.png")
```

### Progressive upscale

```python
result = pipe(
    prompt="high quality, detailed, sharp",
    image=image,
    control_image=image,
    controlnet_conditioning_scale=0.75,
    scale_factor=4.0,
    progressive=True,
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
| `control_image` | `None` | ControlNet conditioning image |
| `controlnet_conditioning_scale` | `0.75` | ControlNet strength |
| `progressive` | `True` | Split large upscale factors into multiple passes |
| `auto_strength` | `True` | Auto-scale strength per pass |
| `generator` | `None` | Torch generator for reproducibility |

## Limitations

- Z-Image Turbo (6B) needs ~22GB VRAM if using without controlnet. If controlnet enabled, use A100 because of higher VRAM
- ControlNet improves faithfulness but is optional
- Tiles smaller than 32 latent pixels may produce artifacts
- Very small inputs produce distortion. Use progressive mode
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
