"""Assembled pipeline blocks for Z-Image MultiDiffusion upscaling."""

from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam
from diffusers.modular_pipelines.z_image.encoders import ZImageTextEncoderStep
from diffusers.utils import logging

from .denoise import ZImageMultiDiffusionStep
from .input import ZImageUpscaleStep


logger = logging.get_logger(__name__)


class MultiDiffusionUpscaleBlocks(SequentialPipelineBlocks):
    """Modular pipeline blocks for tiled Z-Image upscaling with MultiDiffusion.

    Uses latent-space noise prediction blending across overlapping tiles
    for seamless tiled upscaling at any resolution.

    Pipeline:
        [1] text_encoder    – ZImageTextEncoderStep (reused)
        [2] upscale         – ZImageUpscaleStep (Lanczos)
        [3] multidiffusion  – ZImageMultiDiffusionStep (encode → tile denoise → decode)
    """

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
