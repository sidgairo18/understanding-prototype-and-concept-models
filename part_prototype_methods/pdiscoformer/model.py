"""PDiscoFormer model — a light part head on top of frozen DINO/DINOv2 ViT features.

SKELETON: implemented in the PDiscoFormer POC step. The relaxed constraints live in
`losses.py`; the model just maps ViT patch tokens to per-patch part assignments.

Reference: Aniraj et al., "PDiscoFormer", ECCV 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import PDiscoFormerConfig


class PDiscoFormer(nn.Module):
    """Frozen ViT backbone -> part head -> per-patch (K+1) part assignment maps.

    Shapes (N = h*w patch tokens, C = token dim):
        image            (B, 3, H, W)
        patch tokens     (B, N, C)        from a frozen DINO/DINOv2 ViT
        assignment maps  (B, K+1, h, w)   softmax over parts per patch
    """

    def __init__(self, cfg: PDiscoFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # TODO (POC step): load a frozen DINO/DINOv2 ViT via common.backbones (timm) and
        # expose patch tokens + the (h, w) grid. Then a small head -> K+1 channels.
        self.backbone = None  # set in POC step
        self.part_head: nn.Module = nn.Identity()  # replaced with a real head in POC step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return part assignment maps (B, K+1, h, w), softmax over the part dimension.

        TODO (POC step): ViT tokens -> reshape to grid -> part head -> softmax over parts.
        Also expose pooled per-part token features for classification/orthogonality.
        """
        raise NotImplementedError("Implemented in the PDiscoFormer POC step.")
