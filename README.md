# Modular Z-Image Upscale

Tiled image upscaling for Z-Image using [MultiDiffusion](https://arxiv.org/abs/2302.08113) latent-space blending. Built with [Modular Diffusers](https://huggingface.co/blog/modular-diffusers).

Produces seamless upscaled output without tile boundary artifacts. Supports optional ControlNet conditioning, progressive upscaling, and auto-strength.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/akshan-main/4965498a6cea46b42dc691318154d0d6/modular-zimage-upscale_demo.ipynb)
[![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-Hub-yellow)](https://huggingface.co/akshan-main/modular-zimage-upscale)

## Features

- Image upscaling at any scale factor using Z-Image
- Blends overlapping transformer tile predictions in latent space with cosine weights
- Optional ControlNet conditioning for faithful upscaling
- Progressive upscaling for large scale factors with auto-strength per pass
- Seamless output with no tile boundary artifacts
- Few-step inference with Z-Image Turbo (4-8 steps)

## Install

```bash
pip install git+https://github.com/huggingface/diffusers.git transformers accelerate safetensors
```

Requires diffusers from main (modular diffusers support).

## Usage

### From HuggingFace Hub (recommended)

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

### From source

```python
from modular_blocks import MultiDiffusionUpscaleBlocks

blocks = MultiDiffusionUpscaleBlocks()
pipe = blocks.init_pipeline("Tongyi-MAI/Z-Image-Turbo")
pipe.load_components(torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Input image (PIL) |
| `prompt` | `""` | Text prompt |
| `negative_prompt` | `None` | Negative text prompt |
| `scale_factor` | `2.0` | Scale multiplier |
| `strength` | `0.4` | Denoise strength. Lower = closer to input |
| `num_inference_steps` | `8` | Denoising steps |
| `tile_size` | `64` | Tile size in latent pixels |
| `tile_overlap` | `8` | Tile overlap in latent pixels |
| `control_image` | `None` | ControlNet conditioning image |
| `controlnet_conditioning_scale` | `0.75` | ControlNet strength |
| `progressive` | `True` | Split 4x+ into multiple 2x passes |
| `auto_strength` | `True` | Auto-scale strength per pass |
| `generator` | `None` | Torch generator for reproducibility |
| `output_type` | `"pil"` | Output format |

## How it works

1. Input image is upscaled to the target resolution using Lanczos interpolation
2. Upscaled image is encoded to latent space via the Z-Image VAE
3. Noise is added to the latents based on `strength`
4. At each denoising timestep, the transformer runs on overlapping latent tiles with cosine-weighted blending (MultiDiffusion)
5. One scheduler step is taken on the full blended prediction
6. After all timesteps, denoised latents are decoded back to pixel space
7. For upscale factors above 2x with `progressive=True`, steps 1-6 repeat as multiple 2x passes

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

## Repo structure

```
├── hub_block/          # Consolidated single-file for HuggingFace Hub
│   ├── block.py
│   └── modular_config.json
├── tests/
├── utils_tiling.py     # Tile planning + cosine weights
├── input.py            # Lanczos upscale step
├── denoise.py          # MultiDiffusion tiled denoise step
├── modular_blocks.py   # Assembled pipeline blocks
└── __init__.py
```

## Limitations

- Z-Image Turbo (6B) needs ~16GB VRAM. Use `enable_model_cpu_offload()` on T4
- ControlNet is recommended for faithful upscaling. Without it, the model hallucinates new content
- Tiles smaller than 32 latent pixels may produce artifacts
- 4x from very small inputs (below 256px) produces distortion. Use progressive mode
- Not suitable for upscaling text, line art, or pixel art

## Models

- Base: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- ControlNet (optional): [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union)

## References

- [MultiDiffusion](https://arxiv.org/abs/2302.08113) (Bar-Tal et al., 2023)
- [Modular Diffusers](https://huggingface.co/blog/modular-diffusers)
- [Modular Diffusers contribution call](https://github.com/huggingface/diffusers/issues/13295)
