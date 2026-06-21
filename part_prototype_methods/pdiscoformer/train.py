"""Train the PDiscoFormer POC on (a subset of) CUB-200-2011.

SKELETON: implemented in the PDiscoFormer POC step. Run from the repo root:

    python -m part_prototype_methods.pdiscoformer.train

Only the part head trains; the DINO ViT backbone stays frozen.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from common.data.cub import CUB200, build_transforms
from .config import PDiscoFormerConfig
from .model import PDiscoFormer
from . import losses


def build_dataloader(cfg: PDiscoFormerConfig):
    train_set = CUB200(
        data_root=cfg.data_root, train=True, num_classes=cfg.num_classes,
        images_per_class=cfg.images_per_class,
        transform=build_transforms(cfg.img_size, train=True),
    )
    return DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.num_workers)


def main(cfg: PDiscoFormerConfig | None = None) -> None:
    cfg = cfg or PDiscoFormerConfig()
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    loader = build_dataloader(cfg)
    model = PDiscoFormer(cfg).to(cfg.device)

    # TODO (POC step):
    #   freeze backbone; optimize only the part head
    #   loss = w_tv*TV + w_ent*entropy + w_eq*equivariance + w_orth*orthogonality (+ classification)
    #   visualize multi-modal part maps via common.viz.colorize_parts (contrast with SCOPS)
    _ = (model, loader, losses)
    raise NotImplementedError("Training loop implemented in the PDiscoFormer POC step.")


if __name__ == "__main__":
    main()
