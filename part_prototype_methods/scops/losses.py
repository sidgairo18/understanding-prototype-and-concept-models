"""SCOPS self-supervised losses — the core of the method.

SKELETON: implemented in the SCOPS POC step. Each function takes part response maps
(B, K+1, H, W) and returns a scalar loss. These four terms are what make unlabeled part
discovery work; keep them the clearest part of this POC.

Reference: Hung et al., SCOPS, CVPR 2019.
"""

from __future__ import annotations

import torch


def concentration_loss(response: torch.Tensor) -> torch.Tensor:
    """Each part should be a single compact blob.

    Compute each part's spatial center of mass, then penalize the response-weighted spatial
    variance around that center. TODO (POC step).
    """
    raise NotImplementedError("Implemented in the SCOPS POC step.")


def equivariance_loss(response: torch.Tensor, response_warped: torch.Tensor, transform) -> torch.Tensor:
    """Parts must move with the image: parts(T(x)) ≈ T(parts(x)).

    `response` are parts of x; `response_warped` are parts of T(x); apply T to `response`
    and compare. TODO (POC step) — start with affine, optionally add TPS.
    """
    raise NotImplementedError("Implemented in the SCOPS POC step.")


def semantic_consistency_loss(response: torch.Tensor, features: torch.Tensor, part_basis: torch.Tensor) -> torch.Tensor:
    """A part should look the same across images.

    Pool deep `features` within each part (weighted by `response`) to get part descriptors,
    and tie them to a learned global `part_basis` (dictionary) so part identity is stable
    across the collection. TODO (POC step).
    """
    raise NotImplementedError("Implemented in the SCOPS POC step.")


def background_loss(response: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
    """Parts land on the foreground; background pixels go to the background channel.

    Uses an unsupervised `saliency` map as a soft foreground prior. TODO (POC step).
    """
    raise NotImplementedError("Implemented in the SCOPS POC step.")
