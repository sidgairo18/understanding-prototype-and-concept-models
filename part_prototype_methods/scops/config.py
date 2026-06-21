"""Hyperparameters for the SCOPS POC. Tiny defaults; scale via num_classes/parts/epochs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SCOPSConfig:
    # --- data ---
    data_root: str | None = None      # None -> CUB_ROOT env var
    num_classes: int = 10             # part discovery is class-agnostic; this just sizes data
    images_per_class: int | None = None
    img_size: int = 128               # part-discovery POCs use smaller inputs

    # --- model ---
    backbone: str = "resnet34"        # encoder for features / response maps
    num_parts: int = 4                # K (excludes background); paper uses up to ~8

    # --- loss weights ---
    lambda_concentration: float = 1.0
    lambda_equivariance: float = 10.0
    lambda_semantic: float = 1.0
    lambda_background: float = 1.0

    # --- equivariance transform ---
    use_tps: bool = False             # start with affine-only; enable TPS warp later

    # --- schedule ---
    epochs: int = 15
    lr: float = 1e-3
    batch_size: int = 16
    num_workers: int = 4

    # --- misc ---
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "part_prototype_methods/scops/runs"
