"""Hyperparameters for the ProtoPNet POC.

Defaults are intentionally *tiny* (a few classes, short schedule) so the POC runs quickly.
Bump `num_classes`/epochs (or set them to the full-CUB values noted inline) to scale up.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProtoPNetConfig:
    # --- data ---
    data_root: str | None = None      # None -> read CUB_ROOT env var
    num_classes: int = 5              # tiny default; full CUB = 200
    images_per_class: int | None = None
    img_size: int = 224
    crop_to_bbox: bool = False        # crop to CUB bbox (paper-faithful; birds cropped)
    strong_aug: bool = False          # stronger online augmentation (rotation/shear/perspective)

    # --- backbone / features ---
    backbone: str = "resnet34"        # from common.backbones; resnet50 in the paper
    pretrained: bool = True           # ImageNet-pretrained backbone (set False for offline smoke tests)
    prototype_dim: int = 128          # D: add-on output channels = prototype dim

    # --- prototype layer ---
    prototypes_per_class: int = 10    # paper default
    # total prototypes = prototypes_per_class * num_classes

    # --- loss weights (paper notation) ---
    lambda_cluster: float = 0.8
    lambda_separation: float = 0.08
    lambda_l1: float = 1e-4           # L1 on cross-class last-layer weights

    # --- optimization / schedule ---
    epochs: int = 10                  # full run is ~1000+ iters/epoch over many epochs
    warm_epochs: int = 5              # train add-on + prototypes before joint
    push_every: int = 5              # run prototype push/projection every N epochs
    last_layer_iters: int = 20        # convex last-layer optimization steps after each push
    lr: float = 1e-3
    lr_step_size: int = 5             # joint-phase StepLR: decay every N joint epochs (paper uses StepLR)
    lr_gamma: float = 0.1             # joint-phase StepLR decay factor
    batch_size: int = 32
    num_workers: int = 4

    # --- checkpointing / resume ---
    # None -> start fresh; "auto" -> resume <out_dir>/<ckpt_name> if present (chaining-friendly);
    # or an explicit checkpoint path. Settable on the CLI via --resume [PATH].
    resume: str | None = None
    ckpt_name: str = "ckpt_last.pt"   # rolling 'latest' checkpoint filename under out_dir
    best_ckpt_name: str = "ckpt_best.pt"  # best-test-accuracy checkpoint filename under out_dir

    # --- experiment tracking (wandb; a no-op unless enabled via config/--wandb) ---
    wandb: bool = False
    wandb_project: str = "understanding-protos"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"        # online | offline | disabled

    # --- misc ---
    seed: int = 0
    device: str = "cuda"              # falls back to cpu in train.py if unavailable
    out_dir: str = "prototype_methods/protopnet/runs"

    # --- distributed (only used when launched via torchrun; see common/distributed.py) ---
    sync_batchnorm: bool = False      # convert backbone BN -> SyncBatchNorm (CUDA+DDP only)
