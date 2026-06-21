"""SCOPS model — encoder that outputs per-pixel part response maps.

SKELETON: implemented in the SCOPS POC step. The interesting logic is in `losses.py`;
the model itself is a feature encoder + a part-response head.

Reference: Hung et al., "SCOPS: Self-Supervised Co-Part Segmentation", CVPR 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.backbones import build_backbone
from .config import SCOPSConfig


class SCOPSNet(nn.Module):
    """Encoder -> upsample -> (K+1)-channel per-pixel softmax (K parts + background).

    Shapes:
        image           (B, 3, H, W)
        response maps    (B, K+1, H, W)   softmax over the part dimension, per pixel
    """

    def __init__(self, cfg: SCOPSConfig) -> None:
        super().__init__()
        self.cfg = cfg
        backbone, backbone_channels = build_backbone(cfg.backbone, pretrained=True)
        self.backbone = backbone
        # Predict K parts + 1 background channel from features; upsample back to image res.
        self.part_head = nn.Conv2d(backbone_channels, cfg.num_parts + 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return part response maps (B, K+1, H, W), softmax-normalized over the part dim.

        TODO (POC step): run backbone, 1x1 head, bilinear upsample to (H, W), softmax.
        Also expose pooled deep features per part (for the semantic-consistency loss).
        """
        raise NotImplementedError("Implemented in the SCOPS POC step.")
