"""Prototype push / projection — ProtoPNet's interpretability step.

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
    record (dataset index, grid location, distance) so the patch can be cropped/visualized.

Use an *un-shuffled, un-augmented* loader so the returned dataset indices line up with the
dataset for visualization.

Reference: Chen et al., NeurIPS 2019, Section 3 ("prototype projection").
"""

from __future__ import annotations

import torch


@torch.no_grad()
def push_prototypes(model, dataloader, device: str = "cpu") -> list[dict | None]:
    """Project every prototype onto its nearest same-class training patch (in place).

    Mutates ``model.prototypes`` and returns, per prototype, the metadata needed to crop
    and visualize its source patch::

        {"dataset_index": int, "row": int, "col": int, "distance": float, "grid_hw": (h, w)}

    A prototype whose class never appears in ``dataloader`` keeps its trained value and gets
    ``None`` metadata.
    """
    was_training = model.training
    model.eval()

    m = model.num_prototypes
    d = model.cfg.prototype_dim
    proto_class = model.prototype_class.argmax(dim=1)            # (m,) class id per prototype

    best_dist = torch.full((m,), float("inf"))
    best_vec = model.prototypes.detach().clone().view(m, d).cpu()
    best_meta: list[dict | None] = [None] * m

    seen = 0
    for images, labels in dataloader:
        images = images.to(device)
        z = model.features(images)                              # (B, D, h, w)
        distances = model._l2_distances(z)                      # (B, m, h, w)
        bsz, _, h, w = distances.shape

        for b in range(bsz):
            cls = int(labels[b])
            # Only consider prototypes that belong to this image's class.
            proto_ids = (proto_class == cls).nonzero(as_tuple=True)[0]
            for j in proto_ids.tolist():
                dmap = distances[b, j]                          # (h, w)
                flat = int(torch.argmin(dmap))
                row, col = divmod(flat, w)
                dist = float(dmap[row, col])
                if dist < best_dist[j]:
                    best_dist[j] = dist
                    best_vec[j] = z[b, :, row, col].detach().cpu()
                    best_meta[j] = {
                        "dataset_index": seen + b,
                        "row": row,
                        "col": col,
                        "distance": dist,
                        "grid_hw": (h, w),
                    }
        seen += bsz

    # Write the located patches back into the prototype parameter.
    new_prototypes = best_vec.view(m, d, 1, 1).to(model.prototypes.device)
    model.prototypes.data.copy_(new_prototypes)

    if was_training:
        model.train()
    return best_meta
