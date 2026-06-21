"""PIP-Net model — prototype presence head + sparse non-negative scoring sheet.

SKELETON: implemented in the PIP-Net POC step (after ProtoPNet). Keep the self-supervised
alignment and the non-negativity/sparsity constraints front-and-center here.

Reference: Nauta et al., "PIP-Net", CVPR 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.backbones import build_backbone
from .config import PIPNetConfig


class PIPNet(nn.Module):
    """Backbone -> prototype presence maps -> presence vector -> sparse non-neg classifier.

    Shapes (P = num_prototypes, h×w = feature grid):
        image           (B, 3, H, W)
        presence map    (B, P, h, w)   1x1 conv on features, softmax over P per location
        presence g      (B, P)         max-pool over h,w -> "is prototype p present?"
        logits          (B, num_classes)  non-negative sparse linear over g
    """

    def __init__(self, cfg: PIPNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        backbone, backbone_channels = build_backbone(cfg.backbone, pretrained=True)
        self.backbone = backbone
        # 1x1 conv producing per-location scores over P prototypes.
        self.proto_head = nn.Conv2d(backbone_channels, cfg.num_prototypes, kernel_size=1)
        # Non-negative weights enforced at use-time (e.g. via softplus/relu or a projection).
        self.classifier = nn.Linear(cfg.num_prototypes, cfg.num_classes, bias=False)

    def presence(self, x: torch.Tensor) -> torch.Tensor:
        """Return prototype presence vector g ∈ [0,1]^P (max-pool of presence maps).

        TODO (POC step): softmax over prototypes per location, then max-pool over h,w.
        """
        raise NotImplementedError("Implemented in the PIP-Net POC step.")

    def forward(self, x: torch.Tensor):
        """Classify from prototype presence using non-negative classifier weights.

        TODO (POC step): logits = relu(W) @ g  (enforce non-negativity).
        """
        raise NotImplementedError("Implemented in the PIP-Net POC step.")

    def alignment_loss(self, x_view1: torch.Tensor, x_view2: torch.Tensor) -> torch.Tensor:
        """Self-supervised: matching patches across two augmented views should activate the
        same prototype. TODO (POC step): align presence maps under the known view transform.
        """
        raise NotImplementedError("Implemented in the PIP-Net POC step.")
