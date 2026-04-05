"""Tile planning and cosine blending weights for MultiDiffusion."""

import math
from dataclasses import dataclass

import torch


@dataclass
class LatentTileSpec:
    """Tile specification in latent space for MultiDiffusion.

    Attributes:
        y: Top edge in latent pixels.
        x: Left edge in latent pixels.
        h: Height in latent pixels.
        w: Width in latent pixels.
    """

    y: int
    x: int
    h: int
    w: int


def validate_tile_params(tile_size: int, overlap: int) -> None:
    """Validate tile parameters."""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")


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
    validate_tile_params(tile_size, overlap)
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


def make_cosine_tile_weight(
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
