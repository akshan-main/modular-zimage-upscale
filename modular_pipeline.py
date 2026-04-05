"""ModularPipeline for tiled Z-Image upscaling."""

from diffusers.modular_pipelines.z_image.modular_pipeline import ZImageModularPipeline


class ZImageUpscaleModularPipeline(ZImageModularPipeline):
    """A ModularPipeline for tiled Z-Image upscaling."""

    default_blocks_name = "MultiDiffusionUpscaleBlocks"
