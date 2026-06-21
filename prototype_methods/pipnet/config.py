"""Hyperparameters for the PIP-Net POC. Tiny defaults; scale up via num_classes/epochs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIPNetConfig:
    # --- data ---
    data_root: str | None = None      # None -> CUB_ROOT env var
    num_classes: int = 5              # tiny default; full CUB = 200
    images_per_class: int | None = None
    img_size: int = 224

    # --- backbone / prototypes ---
    backbone: str = "resnet50"        # paper uses ConvNeXt-tiny; ResNet ok for the POC
    num_prototypes: int = 128         # P: not class-tied; sparsity prunes most per class

    # --- losses ---
    lambda_align: float = 1.0         # self-supervised patch-alignment weight
    lambda_tanh: float = 1.0          # presence (tanh) loss weight
    lambda_sparsity: float = 1e-3     # sparsity on the non-negative classifier

    # --- schedule ---
    pretrain_epochs: int = 5          # self-supervised prototype pretraining (no labels)
    epochs: int = 10                  # classification training
    lr: float = 5e-4
    batch_size: int = 32
    num_workers: int = 4

    # --- misc ---
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "prototype_methods/pipnet/runs"
