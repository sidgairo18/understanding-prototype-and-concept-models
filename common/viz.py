"""Visualization helpers for the *signature artifacts* of each method.

These are the pictures that make a POC's mechanism legible:
  * prototype methods -> a similarity heatmap over an image showing where a prototype
    activates (upsampled from the feature grid to the input resolution);
  * part-discovery methods -> a colored part-segmentation map overlaid on the image.

Kept dependency-light (numpy + matplotlib + torch) and intentionally simple.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common.data.cub import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization on a ``(3, H, W)`` tensor -> ``(H, W, 3)`` uint8 array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (img.detach().cpu() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def upsample_activation(activation: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Bilinearly upsample a ``(h, w)`` feature-grid activation to image ``size`` -> ``(H, W)``.

    Used to turn a prototype's coarse similarity map into an input-resolution heatmap.
    """
    a = activation.detach().cpu().float()[None, None]  # (1, 1, h, w)
    up = F.interpolate(a, size=size, mode="bilinear", align_corners=False)[0, 0]
    up = up - up.min()
    denom = up.max().clamp_min(1e-8)
    return (up / denom).numpy()


def overlay_heatmap(image: torch.Tensor, activation: torch.Tensor):
    """Build an RGB overlay of a prototype activation on an image.

    Returns ``(rgb_image_uint8, heatmap_float)`` ready to hand to matplotlib.
    Plotting/saving is left to the caller (and to each POC's eval script) so this module
    stays free of file-system side effects.
    """
    rgb = denormalize(image)
    heat = upsample_activation(activation, size=rgb.shape[:2])
    return rgb, heat


def colorize_parts(part_maps: torch.Tensor) -> np.ndarray:
    """Turn per-part attention maps ``(K, h, w)`` into a single RGB segmentation ``(h, w, 3)``.

    Each spatial cell is colored by its arg-max part. Background (last channel, by
    convention in SCOPS/PDiscoFormer) renders black. A simple fixed palette is used so the
    same part id keeps the same color across images.
    """
    k = part_maps.shape[0]
    assignment = part_maps.argmax(dim=0).cpu().numpy()  # (h, w) part id per cell
    palette = _palette(k)
    return palette[assignment]


def _palette(k: int) -> np.ndarray:
    """A small deterministic color palette; last entry (background) is black."""
    base = np.array(
        [
            [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
            [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
            [153, 153, 153], [102, 194, 165], [252, 141, 98], [141, 160, 203],
        ],
        dtype=np.uint8,
    )
    colors = np.zeros((k, 3), dtype=np.uint8)
    for i in range(k):
        colors[i] = base[i % len(base)]
    colors[-1] = (0, 0, 0)  # background
    return colors
