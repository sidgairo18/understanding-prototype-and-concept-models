"""Prototype push / projection — ProtoPNet's interpretability step.

SKELETON: implemented in the ProtoPNet POC step.

After some epochs of training, each prototype vector is replaced by the *nearest latent
patch* among all training images of the prototype's own class. Two consequences:
  1. every prototype becomes an actual image region (visualizable -> "this looks like
     *that*");
  2. the learned similarities now refer to concrete training evidence.

Algorithm (per prototype p_j of class c):
    for every training image x of class c:
        z = model.features(x)                  # (D, h, w)
        find the patch z[:, a, b] minimizing ||z[:, a, b] - p_j||^2
    set p_j <- the global-best patch over all class-c images
    record (image, location, receptive-field box) so the patch can be cropped/visualized.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def push_prototypes(model, dataloader, device: str = "cpu") -> list[dict]:
    """Project every prototype onto its nearest same-class training patch.

    Mutates ``model.prototypes`` in place and returns, per prototype, the metadata needed
    to crop and visualize its source patch (image index, (row, col) on the feature grid,
    and the achieved distance).

    TODO (POC step): iterate the training set, track per-prototype best patches, then copy
    them into ``model.prototypes``.
    """
    raise NotImplementedError("Implemented in the ProtoPNet POC step.")
