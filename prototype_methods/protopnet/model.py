"""ProtoPNet model — the prototype layer is the heart of the method.

SKELETON: interfaces and the math are sketched here; the full implementation lands in the
ProtoPNet POC step (see the repo roadmap in CLAUDE.md). Keep this file the *clearest*
explanation of the prototype mechanism in the repo.

Reference: Chen et al., "This Looks Like That", NeurIPS 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.backbones import build_backbone
from .config import ProtoPNetConfig


class ProtoPNet(nn.Module):
    """CNN backbone + add-on convs + prototype layer + linear classifier.

    Shapes (B = batch, D = prototype_dim, m = total prototypes, h×w = feature grid):
        image            (B, 3, H, W)
        features z       (B, D, h, w)      backbone + 1x1 add-on convs, then L2-normalized
        distances        (B, m, h, w)      squared L2 from each patch to each prototype
        similarities     (B, m)            max over h,w of distance->similarity
        logits           (B, num_classes)  linear layer over similarities
    """

    def __init__(self, cfg: ProtoPNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_prototypes = cfg.prototypes_per_class * cfg.num_classes

        backbone, backbone_channels = build_backbone(cfg.backbone, pretrained=True)
        self.backbone = backbone
        # Add-on layers map backbone channels -> prototype_dim (paper uses two 1x1 convs).
        self.add_on = nn.Sequential(
            nn.Conv2d(backbone_channels, cfg.prototype_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(cfg.prototype_dim, cfg.prototype_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        # Prototype vectors, stored as a (m, D, 1, 1) tensor so distances are a conv-like op.
        self.prototypes = nn.Parameter(
            torch.rand(self.num_prototypes, cfg.prototype_dim, 1, 1)
        )
        # Fixed identity map: which class each prototype belongs to (m, num_classes).
        self.register_buffer("prototype_class", self._init_prototype_class_identity())
        # Last layer: similarities -> class logits. Init +1 own-class / -0.5 others.
        self.classifier = nn.Linear(self.num_prototypes, cfg.num_classes, bias=False)

    def _init_prototype_class_identity(self) -> torch.Tensor:
        identity = torch.zeros(self.num_prototypes, self.cfg.num_classes)
        ppc = self.cfg.prototypes_per_class
        for j in range(self.num_prototypes):
            identity[j, j // ppc] = 1.0
        return identity

    # ------------------------------------------------------------------ forward
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone + add-on -> L2-normalized feature grid (B, D, h, w)."""
        z = self.add_on(self.backbone(x))
        return F.normalize(z, dim=1)

    def _squared_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Squared L2 distance between every patch and every prototype -> (B, m, h, w).

        With L2-normalized z and prototypes, ||a-b||^2 = 2 - 2 a·b, so this reduces to a
        1x1 convolution (dot products) — TODO: implement in the POC step.
        """
        raise NotImplementedError("Implemented in the ProtoPNet POC step.")

    @staticmethod
    def _distance_to_similarity(d2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """ProtoPNet's similarity: log((d^2 + 1) / (d^2 + eps)). Monotonic-decreasing in d."""
        return torch.log((d2 + 1) / (d2 + eps))

    def forward(self, x: torch.Tensor):
        """Return (logits, min_distances) — min_distances feed the cluster/sep losses."""
        raise NotImplementedError("Implemented in the ProtoPNet POC step.")

    # ------------------------------------------------------------------ losses
    def cluster_and_separation_costs(self, min_distances: torch.Tensor, labels: torch.Tensor):
        """Cluster cost (patch near a same-class prototype) and separation cost (far from
        other-class prototypes). TODO: implement in the POC step using `prototype_class`.
        """
        raise NotImplementedError("Implemented in the ProtoPNet POC step.")
