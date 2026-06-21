"""Hyperparameters for the PDiscoFormer POC. Tiny defaults; scale via parts/epochs/data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PDiscoFormerConfig:
    # --- data ---
    data_root: str | None = None      # None -> CUB_ROOT env var
    num_classes: int = 10
    images_per_class: int | None = None
    img_size: int = 224               # must match the ViT patch grid

    # --- backbone (frozen self-supervised ViT) ---
    backbone: str = "dino_vits16"     # added to common.backbones when this POC is built
    freeze_backbone: bool = True
    num_parts: int = 8                # K (excludes background)

    # --- loss weights (relaxed constraints) ---
    lambda_tv: float = 1.0            # total-variation smoothness (replaces concentration)
    lambda_entropy: float = 1.0       # entropy / Gumbel assignment
    lambda_equivariance: float = 1.0
    lambda_orthogonality: float = 1.0 # parts distinct
    lambda_classification: float = 1.0

    # --- schedule ---
    epochs: int = 15
    lr: float = 1e-3                  # only the part head trains; backbone frozen
    batch_size: int = 16
    num_workers: int = 4

    # --- misc ---
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "part_prototype_methods/pdiscoformer/runs"
