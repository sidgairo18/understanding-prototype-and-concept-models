"""Train the SCOPS POC on (a subset of) CUB-200-2011.

SKELETON: implemented in the SCOPS POC step. Run from the repo root:

    python -m part_prototype_methods.scops.train

Each step: sample a transform T, run the net on x and on T(x), and combine the four losses
from `losses.py`. Part discovery is class-agnostic — labels are unused.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from common.data.cub import CUB200, build_transforms
from .config import SCOPSConfig
from .model import SCOPSNet
from . import losses


def build_dataloader(cfg: SCOPSConfig):
    train_set = CUB200(
        data_root=cfg.data_root, train=True, num_classes=cfg.num_classes,
        images_per_class=cfg.images_per_class,
        transform=build_transforms(cfg.img_size, train=True),
    )
    return DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.num_workers)


def main(cfg: SCOPSConfig | None = None) -> None:
    cfg = cfg or SCOPSConfig()
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    loader = build_dataloader(cfg)
    model = SCOPSNet(cfg).to(cfg.device)

    # TODO (POC step):
    #   for each batch: sample transform T; response = model(x); response_w = model(T(x))
    #     loss = w_c*concentration + w_e*equivariance + w_s*semantic + w_b*background
    #   periodically visualize part maps via common.viz.colorize_parts
    _ = (model, loader, losses)
    raise NotImplementedError("Training loop implemented in the SCOPS POC step.")


if __name__ == "__main__":
    main()
