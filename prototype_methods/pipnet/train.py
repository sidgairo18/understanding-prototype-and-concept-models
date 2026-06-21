"""Train the PIP-Net POC: self-supervised prototype pretraining, then sparse classification.

SKELETON: implemented in the PIP-Net POC step. Run from the repo root:

    python -m prototype_methods.pipnet.train
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from common.data.cub import CUB200, build_transforms
from .config import PIPNetConfig
from .model import PIPNet


def build_dataloaders(cfg: PIPNetConfig):
    # Classification loaders. The self-supervised phase additionally builds two augmented
    # views per image (TODO: a paired-view dataset wrapper in the POC step).
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


def main(cfg: PIPNetConfig | None = None) -> None:
    cfg = cfg or PIPNetConfig()
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    train_loader, test_loader = build_dataloaders(cfg)
    model = PIPNet(cfg).to(cfg.device)

    # TODO (POC step):
    #   phase 1: self-supervised alignment pretraining (no labels)
    #   phase 2: classification with tanh-presence + sparsity on non-negative classifier
    #   evaluate; visualize the few "present" prototypes per image (scoring sheet)
    _ = (model, train_loader, test_loader)
    raise NotImplementedError("Training loop implemented in the PIP-Net POC step.")


if __name__ == "__main__":
    main()
