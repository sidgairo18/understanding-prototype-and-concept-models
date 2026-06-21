"""Dataset loaders shared across POCs (currently CUB-200-2011)."""

from common.data.cub import CUB200, build_transforms

__all__ = ["CUB200", "build_transforms"]
