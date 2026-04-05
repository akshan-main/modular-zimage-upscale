"""MultiDiffusion tiled upscaling for Z-Image using Modular Diffusers.

Tiles the latent space, denoises each tile independently per timestep,
blends with cosine-ramp overlap weights, and applies one scheduler step
on the full blended prediction. Zero extra models required.
"""

import math
from dataclasses import dataclass

import PIL.Image
import torch

from diffusers.configuration_utils import FrozenDict
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, ZImageTransformer2DModel
from diffusers.modular_pipelines.modular_pipeline import (
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)
from diffusers.modular_pipelines.z_image.encoders import (
    ZImageTextEncoderStep,
    retrieve_latents,
)
from diffusers.modular_pipelines.z_image.modular_pipeline import ZImageModularPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)


# ============================================================
# Latent tile planning
# ============================================================


@dataclass
class LatentTileSpec:
    """Tile specification in latent space for MultiDiffusion."""

    y: int
    x: int
    h: int
    w: int


def plan_latent_tiles(
    latent_h: int,
    latent_w: int,
    tile_size: int = 64,
    overlap: int = 8,
) -> list[LatentTileSpec]:
    """Plan overlapping tiles in latent space.

    Stride = tile_size - overlap. Final tiles are shifted back to fill
    remaining space so every tile is full-sized.
    """
    stride = tile_size - overlap
    tiles = []

    y = 0
    while y < latent_h:
        h = min(tile_size, latent_h - y)
        if h < tile_size and y > 0:
            y = max(0, latent_h - tile_size)
            h = latent_h - y

        x = 0
        while x < latent_w:
            w = min(tile_size, latent_w - x)
            if w < tile_size and x > 0:
                x = max(0, latent_w - tile_size)
                w = latent_w - x

            tiles.append(LatentTileSpec(y=y, x=x, h=h, w=w))

            if x + w >= latent_w:
                break
            x += stride

        if y + h >= latent_h:
            break
        y += stride

    return tiles


# ============================================================
# Cosine tile weight
# ============================================================


def _make_cosine_tile_weight(
    h: int,
    w: int,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
    is_top: bool = False,
    is_bottom: bool = False,
    is_left: bool = False,
    is_right: bool = False,
) -> torch.Tensor:
    """Create a boundary-aware 2D cosine-ramp weight for MultiDiffusion blending.

    Weight is 1.0 in the centre and smoothly fades at edges that overlap with
    neighbouring tiles. Edges touching the image boundary keep weight = 1.0.
    """

    def _ramp(length, overlap_size, keep_start, keep_end):
        ramp = torch.ones(length, device=device, dtype=dtype)
        if overlap_size > 0 and length > 2 * overlap_size:
            fade = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, overlap_size, device=device, dtype=dtype)))
            if not keep_start:
                ramp[:overlap_size] = fade
            if not keep_end:
                ramp[-overlap_size:] = fade.flip(0)
        return ramp

    w_h = _ramp(h, overlap, keep_start=is_top, keep_end=is_bottom)
    w_w = _ramp(w, overlap, keep_start=is_left, keep_end=is_right)
    return (w_h[:, None] * w_w[None, :]).unsqueeze(0).unsqueeze(0)


# ============================================================
# Upscale step (Lanczos)
# ============================================================


class ZImageUpscaleStep(ModularPipelineBlocks):
    model_name = "z-image"

    @property
    def description(self) -> str:
        return "Upscale input image with Lanczos interpolation"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image", required=True, type_hint=PIL.Image.Image, description="Input image to upscale."),
            InputParam("scale_factor", default=2.0, type_hint=float, description="Upscale factor."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("upscaled_image", type_hint=PIL.Image.Image, description="Upscaled image."),
            OutputParam("height", type_hint=int),
            OutputParam("width", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        image = block_state.image
        scale = block_state.scale_factor

        new_w = int(image.width * scale)
        new_h = int(image.height * scale)

        # Round to multiple of vae_scale_factor_spatial (16)
        sf = components.vae_scale_factor_spatial
        new_w = (new_w // sf) * sf
        new_h = (new_h // sf) * sf

        block_state.upscaled_image = image.resize((new_w, new_h), PIL.Image.LANCZOS)
        block_state.height = new_h
        block_state.width = new_w

        self.set_block_state(state, block_state)
        return components, state


# ============================================================
# MultiDiffusion denoise step
# ============================================================


class ZImageMultiDiffusionStep(ModularPipelineBlocks):
    """MultiDiffusion tiled denoising for Z-Image.

    Outer loop = timesteps, inner loop = tiles.
    Blends noise predictions with cosine-ramp weights, then applies
    a single scheduler step on the full blended prediction.
    """

    model_name = "z-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec("transformer", ZImageTransformer2DModel),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 8 * 2}),
                default_creation_method="from_config",
            ),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 5.0, "enabled": False}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return (
            "MultiDiffusion tiled denoising: encodes the full upscaled image, "
            "denoises with overlapping latent tiles and cosine-weighted blending, "
            "then decodes the result."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("upscaled_image", required=True, type_hint=PIL.Image.Image, description="Upscaled image."),
            InputParam("height", required=True, type_hint=int),
            InputParam("width", required=True, type_hint=int),
            InputParam("prompt_embeds", required=True, description="Text embeddings from text encoder."),
            InputParam("negative_prompt_embeds", description="Negative text embeddings."),
            InputParam("num_inference_steps", default=8, type_hint=int),
            InputParam("strength", default=0.5, type_hint=float, description="Denoising strength (0-1)."),
            InputParam("tile_size", default=64, type_hint=int, description="Tile size in latent pixels."),
            InputParam("tile_overlap", default=8, type_hint=int, description="Tile overlap in latent pixels."),
            InputParam("generator", description="Torch generator for deterministic results."),
            InputParam("output_type", default="pil", type_hint=str),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("images", description="The generated images."),
        ]

    def _run_tile_transformer(
        self,
        components: ZImageModularPipeline,
        tile_latents: torch.Tensor,
        t: torch.Tensor,
        i: int,
        num_inference_steps: int,
        prompt_embeds: list[torch.Tensor],
        negative_prompt_embeds: list[torch.Tensor] | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run transformer on a single tile and return noise prediction."""
        # Prepare latent input: [B, C, H, W] -> [B, C, 1, H, W] -> list of [C, 1, H, W]
        latent_input = tile_latents.unsqueeze(2).to(dtype)
        latent_model_input = list(latent_input.unbind(dim=0))

        # Normalised timestep
        timestep = t.expand(tile_latents.shape[0]).to(dtype)
        timestep = (1000 - timestep) / 1000

        # Setup guider
        guider_inputs = {"cap_feats": (prompt_embeds, negative_prompt_embeds)}
        components.guider.set_state(step=i, num_inference_steps=num_inference_steps, timestep=t)
        guider_state = components.guider.prepare_inputs(guider_inputs)

        for guider_state_batch in guider_state:
            components.guider.prepare_models(components.transformer)

            cond_kwargs = {}
            for k, v in guider_state_batch.as_dict().items():
                if k in guider_inputs:
                    if isinstance(v, torch.Tensor):
                        cond_kwargs[k] = v.to(dtype)
                    elif isinstance(v, list):
                        cond_kwargs[k] = [x.to(dtype) if isinstance(x, torch.Tensor) else x for x in v]
                    else:
                        cond_kwargs[k] = v

            model_out_list = components.transformer(
                x=latent_model_input,
                t=timestep,
                return_dict=False,
                **cond_kwargs,
            )[0]

            noise_pred = torch.stack(model_out_list, dim=0).squeeze(2)
            guider_state_batch.noise_pred = -noise_pred
            components.guider.cleanup_models(components.transformer)

        return components.guider(guider_state)[0]

    @torch.no_grad()
    def __call__(self, components: ZImageModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        vae_dtype = components.vae.dtype
        dtype = components.transformer.dtype

        image = block_state.upscaled_image
        height = block_state.height
        width = block_state.width
        strength = block_state.strength
        num_inference_steps = block_state.num_inference_steps
        tile_size = block_state.tile_size
        tile_overlap = block_state.tile_overlap
        generator = block_state.generator

        # --- VAE encode ---
        image_tensor = components.image_processor.preprocess(image, height=height, width=width)
        image_tensor = image_tensor.to(device=device, dtype=vae_dtype)
        image_latents = retrieve_latents(components.vae.encode(image_tensor), generator=generator)
        image_latents = (image_latents - components.vae.config.shift_factor) * components.vae.config.scaling_factor

        # --- Prepare timesteps with strength ---
        components.scheduler.set_timesteps(num_inference_steps, device=device)
        all_timesteps = components.scheduler.timesteps
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = all_timesteps[t_start:]
        num_inf_steps = len(timesteps)

        if num_inf_steps == 0:
            logger.warning("strength too low — no denoising steps. Returning VAE roundtrip.")
            latents = image_latents
        else:
            # --- Add noise at strength level ---
            noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=image_latents.dtype)
            latents = components.scheduler.add_noise(image_latents, noise, timesteps[:1])

            # --- Plan tiles ---
            latent_h, latent_w = latents.shape[2], latents.shape[3]
            tile_specs = plan_latent_tiles(latent_h, latent_w, tile_size, tile_overlap)
            logger.info(
                f"MultiDiffusion: {len(tile_specs)} tiles, "
                f"latent {latent_w}x{latent_h}, tile {tile_size}, overlap {tile_overlap}"
            )

            # --- MultiDiffusion denoise loop ---
            prompt_embeds = block_state.prompt_embeds
            negative_prompt_embeds = getattr(block_state, "negative_prompt_embeds", None)

            for i, t in enumerate(timesteps):
                noise_pred_accum = torch.zeros_like(latents, dtype=torch.float32)
                weight_accum = torch.zeros(1, 1, latent_h, latent_w, device=device, dtype=torch.float32)

                for tile in tile_specs:
                    tile_latents = latents[:, :, tile.y : tile.y + tile.h, tile.x : tile.x + tile.w].clone()

                    tile_noise_pred = self._run_tile_transformer(
                        components,
                        tile_latents,
                        t,
                        i,
                        num_inf_steps,
                        prompt_embeds,
                        negative_prompt_embeds,
                        dtype,
                    )

                    tile_weight = _make_cosine_tile_weight(
                        tile.h,
                        tile.w,
                        tile_overlap,
                        device,
                        torch.float32,
                        is_top=(tile.y == 0),
                        is_bottom=(tile.y + tile.h >= latent_h),
                        is_left=(tile.x == 0),
                        is_right=(tile.x + tile.w >= latent_w),
                    )

                    noise_pred_accum[:, :, tile.y : tile.y + tile.h, tile.x : tile.x + tile.w] += (
                        tile_noise_pred.to(torch.float32) * tile_weight
                    )
                    weight_accum[:, :, tile.y : tile.y + tile.h, tile.x : tile.x + tile.w] += tile_weight

                blended = noise_pred_accum / weight_accum.clamp(min=1e-6)
                blended = torch.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0).to(latents.dtype)

                latents = components.scheduler.step(blended.float(), t, latents.float(), return_dict=False)[0]
                latents = latents.to(dtype=image_latents.dtype)

        # --- VAE decode ---
        decode_latents = latents.to(vae_dtype)
        decode_latents = decode_latents / components.vae.config.scaling_factor + components.vae.config.shift_factor
        decoded = components.vae.decode(decode_latents, return_dict=False)[0]
        block_state.images = components.image_processor.postprocess(decoded, output_type=block_state.output_type)

        self.set_block_state(state, block_state)
        return components, state


# ============================================================
# Assembled blocks
# ============================================================


class MultiDiffusionUpscaleBlocks(SequentialPipelineBlocks):
    """Modular pipeline blocks for tiled Z-Image upscaling with MultiDiffusion."""

    model_name = "z-image"
    block_classes = [
        ZImageTextEncoderStep,
        ZImageUpscaleStep,
        ZImageMultiDiffusionStep,
    ]
    block_names = ["text_encoder", "upscale", "multidiffusion"]

    @property
    def description(self):
        return (
            "MultiDiffusion upscale pipeline for Z-Image.\n"
            "1. Text encoding (Qwen3)\n"
            "2. Lanczos upscale\n"
            "3. MultiDiffusion tiled denoise + VAE decode"
        )

    @property
    def outputs(self):
        return [OutputParam("images", description="The upscaled images.")]
