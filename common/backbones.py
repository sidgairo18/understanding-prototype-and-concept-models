"""Shared feature-extractor backbones.

POCs follow a *hybrid* rule (see CLAUDE.md): we write the paper's signature mechanism from
scratch but reuse stock, pretrained backbones. This module centralizes backbone creation
so no POC reimplements a ResNet or wires up DINO weights by hand.

A "backbone" here is a convolutional/transformer feature extractor that maps an input
image ``(B, 3, H, W)`` to a dense feature map ``(B, C, h, w)`` — the grid of *latent
patches* that prototype and part-discovery methods operate on.

Implemented:
  * ``resnet50`` / ``resnet34`` — torchvision, ImageNet-pretrained (ProtoPNet's default
    backbone family).

Planned (filled in when the corresponding POC is built):
  * ``convnext`` — for PIP-Net.
  * ``dino_vits16`` / ``dinov2`` — self-supervised ViT features for PDiscoFormer / SCOPS.

Each builder returns ``(backbone_module, out_channels)`` so downstream code knows the
prototype/feature dimension without hardcoding it.
"""

from __future__ import annotations

import torch.nn as nn


def build_resnet(name: str = "resnet50", pretrained: bool = True) -> tuple[nn.Module, int]:
    """A torchvision ResNet truncated to its last conv feature map (before pooling/fc).

    Returns the convolutional trunk and its output channel count (2048 for resnet50,
    512 for resnet34). Input 224x224 yields a 7x7 feature grid.
    """
    from torchvision import models

    factory = {"resnet50": models.resnet50, "resnet34": models.resnet34}
    if name not in factory:
        raise ValueError(f"Unknown resnet variant: {name}")
    weights = "IMAGENET1K_V1" if pretrained else None
    net = factory[name](weights=weights)
    # Drop the global average pool and classifier; keep the conv feature extractor.
    trunk = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool,
        net.layer1, net.layer2, net.layer3, net.layer4,
    )
    out_channels = 2048 if name == "resnet50" else 512
    return trunk, out_channels


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Dispatch to a backbone family by name.

    POCs call this so the backbone choice lives in their `config.py`. Extend the dispatch
    as new POCs need ConvNeXt / DINO; keep the ``(module, out_channels)`` contract.
    """
    if name.startswith("resnet"):
        return build_resnet(name, pretrained)
    raise NotImplementedError(
        f"Backbone '{name}' not implemented yet — add it here when the POC that needs it "
        "is built (e.g. convnext for PIP-Net, dino/dinov2 for PDiscoFormer)."
    )
