"""ProtoPNet model — the prototype layer is the heart of the method.

This file is the clearest explanation of the prototype mechanism in the repo. The flow is:

    image -> backbone f -> add-on convs -> feature grid z  (B, D, h, w)
          -> for each prototype p_j, squared-L2 distance to every patch  (B, m, h, w)
          -> global-min over patches -> per-prototype min distance       (B, m)
          -> distance->similarity                                        (B, m)
          -> linear last layer                                           (B, num_classes)

"This looks like that": a prototype's activation says *how strongly* its learned pattern
is present, and the arg-min location says *where*.

Reference: Chen et al., "This Looks Like That: Deep Learning for Interpretable Image
Recognition", NeurIPS 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.backbones import build_backbone
from .config import ProtoPNetConfig

# Strength of the (negative) connection a prototype has to classes it does NOT belong to,
# used only to initialize the last layer (paper uses -0.5).
_INCORRECT_CLASS_CONNECTION = -0.5


class ProtoPNet(nn.Module):
    """CNN backbone + add-on convs + prototype layer + linear classifier.

    Shapes (B = batch, D = prototype_dim, m = total prototypes, h×w = feature grid):
        image            (B, 3, H, W)
        features z       (B, D, h, w)      backbone + 1x1 add-on convs, sigmoid -> [0,1]
        distances        (B, m, h, w)      squared L2 from each patch to each prototype
        min_distances    (B, m)            global-min over h,w
        similarities     (B, m)            distance -> similarity
        logits           (B, num_classes)  linear layer over similarities
    """

    def __init__(self, cfg: ProtoPNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_prototypes = cfg.prototypes_per_class * cfg.num_classes

        backbone, backbone_channels = build_backbone(cfg.backbone, pretrained=cfg.pretrained)
        self.backbone = backbone

        # Add-on layers map backbone channels -> prototype_dim. The final sigmoid bounds
        # features to [0,1], which bounds the squared L2 distance by `prototype_dim`
        # (used as `max_dist` below). Paper uses two 1x1 convs.
        self.add_on = nn.Sequential(
            nn.Conv2d(backbone_channels, cfg.prototype_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(cfg.prototype_dim, cfg.prototype_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Prototype vectors, stored (m, D, 1, 1) so distances are a 1x1-conv-like op.
        self.prototypes = nn.Parameter(torch.rand(self.num_prototypes, cfg.prototype_dim, 1, 1))

        # A (m, D, 1, 1) all-ones kernel used to sum z^2 over channels via conv2d.
        self.register_buffer("ones", torch.ones_like(self.prototypes))

        # Fixed assignment of each prototype to one class: (m, num_classes), one-hot rows.
        self.register_buffer("prototype_class", self._init_prototype_class_identity())

        # Last layer: similarities -> class logits, no bias. Initialized so a prototype
        # votes +1 for its own class and -0.5 for others.
        self.classifier = nn.Linear(self.num_prototypes, cfg.num_classes, bias=False)
        self._init_last_layer()

    # ------------------------------------------------------------------ init helpers
    def _init_prototype_class_identity(self) -> torch.Tensor:
        """Block assignment: the first `ppc` prototypes belong to class 0, next to class 1..."""
        identity = torch.zeros(self.num_prototypes, self.cfg.num_classes)
        ppc = self.cfg.prototypes_per_class
        for j in range(self.num_prototypes):
            identity[j, j // ppc] = 1.0
        return identity

    def _init_last_layer(self) -> None:
        positive = self.prototype_class.t()                 # (num_classes, m), own-class = 1
        negative = 1.0 - positive
        self.classifier.weight.data.copy_(positive + _INCORRECT_CLASS_CONNECTION * negative)

    # ------------------------------------------------------------------ forward pieces
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone + add-on -> feature grid (B, D, h, w) in [0,1] (no L2 normalization).

        ProtoPNet compares patches to prototypes with *raw* L2 distance in this sigmoid
        feature space (unlike cosine variants such as TesNet), so we do not normalize.
        """
        return self.add_on(self.backbone(x))

    def _l2_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Squared L2 distance between every patch and every prototype -> (B, m, h, w).

        Uses ||z - p||^2 = ||z||^2 - 2 z·p + ||p||^2, each term computed as a 1x1 conv:
          * ||z||^2 summed over channels via the all-ones kernel,
          * z·p as a conv with the prototypes as the kernel,
          * ||p||^2 a per-prototype constant.
        """
        z2_sum = F.conv2d(z**2, self.ones)                            # (B, m, h, w)
        zp = F.conv2d(z, self.prototypes)                             # (B, m, h, w)
        p2 = (self.prototypes**2).sum(dim=(1, 2, 3)).view(1, -1, 1, 1)  # (1, m, 1, 1)
        d2 = z2_sum - 2 * zp + p2
        return F.relu(d2)  # clamp tiny negatives from floating-point error

    def _distance_to_similarity(self, d2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """ProtoPNet's similarity: log((d^2 + 1) / (d^2 + eps)). Monotonic-decreasing in d^2."""
        return torch.log((d2 + 1) / (d2 + eps))

    def forward(self, x: torch.Tensor):
        """Return (logits, min_distances). min_distances (B, m) feed the cluster/sep losses."""
        z = self.features(x)
        distances = self._l2_distances(z)                     # (B, m, h, w)
        b, m, h, w = distances.shape
        # Global min over the spatial grid == "closest patch to this prototype".
        min_distances = -F.max_pool2d(-distances, kernel_size=(h, w)).view(b, m)
        activations = self._distance_to_similarity(min_distances)
        logits = self.classifier(activations)
        return logits, min_distances

    @torch.no_grad()
    def prototype_activation_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Per-prototype similarity heatmap over the feature grid -> (B, m, h, w).

        Used for the "this looks like that" visualization (upsampled to image resolution).
        """
        distances = self._l2_distances(self.features(x))
        return self._distance_to_similarity(distances)

    # ------------------------------------------------------------------ losses
    def cluster_and_separation_costs(self, min_distances: torch.Tensor, labels: torch.Tensor):
        """Cluster cost: each image is close to *some* prototype of its own class.
        Separation cost: each image is far from prototypes of *other* classes.

        Both use the "inverted distance" trick: maximizing (max_dist - d) over a masked set
        is a differentiable surrogate for minimizing d over that set.
        """
        max_dist = float(self.cfg.prototype_dim)                 # upper bound on squared dist
        proto_correct = self.prototype_class[:, labels].t()      # (B, m), 1 if own-class
        proto_wrong = 1.0 - proto_correct

        inverted_correct = torch.max((max_dist - min_distances) * proto_correct, dim=1)[0]
        cluster_cost = torch.mean(max_dist - inverted_correct)

        inverted_wrong = torch.max((max_dist - min_distances) * proto_wrong, dim=1)[0]
        separation_cost = torch.mean(max_dist - inverted_wrong)
        return cluster_cost, separation_cost

    def last_layer_l1(self) -> torch.Tensor:
        """L1 on last-layer weights connecting a prototype to classes it does NOT belong to.

        Keeps explanations sparse: a class should be supported mainly by its own prototypes.
        """
        mask = 1.0 - self.prototype_class.t()                    # (num_classes, m)
        return (self.classifier.weight * mask).abs().sum()

    # ------------------------------------------------------------------ train-mode toggles
    def set_mode(self, mode: str) -> None:
        """Toggle which parameter groups are trainable.

        'warm'  -> add-on + prototypes (backbone frozen, classifier fixed at init)
        'joint' -> backbone + add-on + prototypes (classifier fixed at init)
        'last'  -> classifier only (everything else frozen) — used after a push
        """
        assert mode in {"warm", "joint", "last"}
        for p in self.backbone.parameters():
            p.requires_grad = mode == "joint"
        for p in self.add_on.parameters():
            p.requires_grad = mode in {"warm", "joint"}
        self.prototypes.requires_grad = mode in {"warm", "joint"}
        for p in self.classifier.parameters():
            p.requires_grad = mode == "last"
