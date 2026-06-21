# PDiscoFormer — Relaxing Part Discovery Constraints with ViTs (Aniraj et al., ECCV 2024)

**Role in this repo:** a *latest*-generation part-discovery method, contrasting with SCOPS.

> Paper PDF: `papers/part_prototype_methods/PDiscoFormer- Relaxing Part Discovery Constraints with Vision Transformers.pdf`
> Predecessor (also in `papers/`): PDiscoNet (BMVC 2023).

## Why it exists (vs. SCOPS / PDiscoNet)

Earlier part-discovery methods bake in strong priors — most notably **geometric
concentration** (each part is one compact Gaussian-like blob). That hurts parts that are
naturally **spread out or multi-modal** (e.g. "wings" appear on both sides; "spots" are
scattered). PDiscoFormer makes two moves:

1. **Strong self-supervised ViT features.** It builds on a frozen **DINO / DINOv2** ViT
   backbone, whose patch tokens already encode rich, semantically-clustered object
   structure — a much better starting point than ImageNet-supervised CNN features.
2. **Relaxed constraints.** It replaces the rigid single-blob concentration prior with a
   **total-variation (spatial smoothness)** term plus an **entropy / Gumbel** mechanism,
   allowing parts to be **multi-modal and non-compact** while still being spatially
   coherent. This yields better, more flexible parts on fine-grained data.

## Architecture

```
image ──▶ frozen DINO/DINOv2 ViT ──▶ patch tokens  ∈ R^{N×C}   (N = h·w patches)
                                          │
                                          ▼   lightweight part head (attention / 1x1)
                                  part assignment maps  A ∈ R^{(K+1)×h×w}
                                  (K parts + background, softmax over parts per patch)
```

A small trainable head on top of frozen ViT features produces per-patch part assignments;
pooled per-part token features support a discriminative/classification signal.

## Losses / constraints (the heart of the method)

- **Total variation (smoothness)** — neighboring patches should share a part → spatially
  coherent regions, *without* forcing a single blob (the SCOPS concentration relaxation).
- **Entropy / Gumbel assignment** — encourages confident but flexible (optionally
  multi-modal) part assignments.
- **Equivariance** — parts transform with the image under geometric transforms (kept from
  the earlier line of work).
- **Foreground / background** + **part presence/orthogonality** terms so parts are distinct
  and land on the object.
- A **classification** objective (fine-grained label) ties parts to discriminative regions.

## What this POC implements from scratch

- The **part head** on frozen DINO features → per-patch `(K+1)` assignment (`model.py`).
- The **relaxed constraints** in `losses.py`: total-variation smoothness, entropy/Gumbel
  assignment, equivariance, foreground/orthogonality.
- A **visualization** of multi-modal part maps overlaid on birds, to *contrast with SCOPS*
  (compact-blob parts vs. relaxed, possibly spread-out parts).

Reused (hybrid rule): the **DINO/DINOv2 ViT** loaded via `common/backbones.py` (added when
this POC is built, e.g. through `timm`), CUB loader from `common/data/cub.py`.

## How to run (once implemented)

```bash
export CUB_ROOT=/path/to/CUB_200_2011
python -m part_prototype_methods.pdiscoformer.train
```

**Signature artifact:** part-segmentation overlays on birds where parts may be spread out /
multi-modal, shown side-by-side conceptually with the SCOPS POC's compact parts.

## POC simplifications (default tiny config)

- Frozen small DINO ViT, small `K`, abbreviated training, a subset of the full loss set to
  start (TV + entropy + equivariance), classification optional. Goal: show the *relaxed
  constraint + ViT-feature* idea, not paper-level numbers. Deviations logged in `notes.md`.
