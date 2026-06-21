"""Train the ProtoPNet POC on (a subset of) CUB-200-2011.

SKELETON: the alternating schedule is outlined; the full loop lands in the POC step.
Run from the repo root:

    python -m prototype_methods.protopnet.train

The schedule alternates: warm/joint SGD (CE + cluster + separation) -> prototype push
(push.py) -> last-layer convex optimization (CE + L1). See README.md.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from common.data.cub import CUB200, build_transforms
from .config import ProtoPNetConfig
from .model import ProtoPNet
from .push import push_prototypes


def build_dataloaders(cfg: ProtoPNetConfig):
    train_set = CUB200(
        data_root=cfg.data_root, train=True, num_classes=cfg.num_classes,
        images_per_class=cfg.images_per_class,
        transform=build_transforms(cfg.img_size, train=True),
    )
    test_set = CUB200(
        data_root=cfg.data_root, train=False, num_classes=cfg.num_classes,
        transform=build_transforms(cfg.img_size, train=False),
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    return train_loader, test_loader


def main(cfg: ProtoPNetConfig | None = None) -> None:
    cfg = cfg or ProtoPNetConfig()
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    train_loader, test_loader = build_dataloaders(cfg)
    model = ProtoPNet(cfg).to(cfg.device)

    # TODO (POC step):
    #   for epoch in range(cfg.epochs):
    #       train one epoch (CE + cluster + separation); warm vs joint by epoch
    #       if epoch % cfg.push_every == 0: push_prototypes(model, train_loader); last-layer opt
    #   evaluate; visualize top prototypes via common.viz
    _ = (model, train_loader, test_loader, push_prototypes)
    raise NotImplementedError("Training loop implemented in the ProtoPNet POC step.")


if __name__ == "__main__":
    main()
