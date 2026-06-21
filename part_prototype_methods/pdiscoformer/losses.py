"""PDiscoFormer losses — the *relaxed* part-discovery constraints.

SKELETON: implemented in the PDiscoFormer POC step. The headline contrast with SCOPS is
here: total-variation smoothness + entropy/Gumbel replace the rigid single-blob
concentration prior, allowing multi-modal / spread-out parts.

Reference: Aniraj et al., PDiscoFormer, ECCV 2024.
"""

from __future__ import annotations

import torch


def total_variation_loss(assignment: torch.Tensor) -> torch.Tensor:
    """Spatial smoothness: neighboring patches should share a part.

    Penalize differences between adjacent cells of the assignment maps -> coherent regions
    *without* forcing a single compact blob. TODO (POC step).
    """
    raise NotImplementedError("Implemented in the PDiscoFormer POC step.")


def entropy_loss(assignment: torch.Tensor) -> torch.Tensor:
    """Encourage confident (low-entropy) per-patch part assignments (Gumbel-style).

    TODO (POC step) — balance against TV so parts stay flexible but decisive.
    """
    raise NotImplementedError("Implemented in the PDiscoFormer POC step.")


def equivariance_loss(assignment: torch.Tensor, assignment_warped: torch.Tensor, transform) -> torch.Tensor:
    """Parts transform with the image: parts(T(x)) ≈ T(parts(x)). TODO (POC step)."""
    raise NotImplementedError("Implemented in the PDiscoFormer POC step.")


def orthogonality_loss(part_features: torch.Tensor) -> torch.Tensor:
    """Pooled per-part features should be distinct (decorrelated) so parts don't collapse.

    TODO (POC step).
    """
    raise NotImplementedError("Implemented in the PDiscoFormer POC step.")
