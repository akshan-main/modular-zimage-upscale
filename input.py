"""Input and upscale steps for Z-Image MultiDiffusion upscaling."""

import PIL.Image
import torch

from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam
from diffusers.modular_pipelines.z_image.modular_pipeline import ZImageModularPipeline
from diffusers.utils import logging


logger = logging.get_logger(__name__)


class ZImageUpscaleStep(ModularPipelineBlocks):
    """Upscale input image with Lanczos interpolation."""

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
