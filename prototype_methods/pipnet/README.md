# PIP-Net — Patch-based Intuitive Prototypes (Nauta et al., CVPR 2023)

**Role in this repo:** a *latest*-generation prototype method, contrasting with ProtoPNet.

> Paper PDF: `papers/prototype_methods/PIP-Net- Patch-Based Intuitive Prototypes for Interpretable Image Classification.pdf`
> (+ supplementary PDF in the same folder)

## Why it exists (vs. ProtoPNet)

ProtoPNet prototypes are class-tied, dense, and can suffer a *semantic gap* — the model's
notion of similarity doesn't always match a human's. PIP-Net targets three things:

1. **Human-aligned prototypes.** Prototypes are learned with a **self-supervised** patch
   alignment objective so that visually similar patches map to the same prototype —
   *before* any class label is involved. This narrows the "this looks like that, but I
   don't see why" gap.
2. **Sparsity.** A typical image activates only a *handful* of prototypes, and each class
   uses few prototypes — the explanation is short by construction.
3. **Interpretable scoring sheet.** Classification is a **sparse linear** sum of prototype
   *presence* scores with **non-negative** weights, so reasoning reads like a scoring
   sheet: "prototype 7 present (+2.1 toward Cardinal), prototype 31 present (+1.4) → ...".

## Architecture

```
two augmented views of an image
        │
        ▼  CNN backbone f (e.g. ConvNeXt-tiny) + 1x1 conv -> prototype presence maps
   p(view) ∈ R^{P×h×w}, softmax over the P prototypes per location
        │
        ├── self-supervised alignment: matching patches in the two views should
        │   activate the *same* prototype  (an InfoNCE-style / alignment loss)
        │
        ▼  max-pool over h×w -> prototype presence vector  g ∈ [0,1]^P
        │
        ▼  sparse, non-negative linear classifier  ->  class logits
```

- **Prototype presence** `g_p ∈ [0,1]`: "is prototype *p* present anywhere in the image?"
  (max-pool of the presence map). Unlike ProtoPNet, prototypes are **not** pre-assigned to
  classes — the classifier learns which prototypes matter for which class.
- **Non-negative classifier**: weights `w ≥ 0` with strong sparsity, so a prototype can
  only ever be *positive evidence* for a class — no "this is a Cardinal because it's *not*
  a Jay" reasoning.

## Training

- **Self-supervised pretraining** of the prototypes via the alignment loss on augmented
  view pairs (no labels) — encourages consistent, human-aligned prototypes.
- **Classification training** with a tanh-based presence loss + a sparsity-inducing
  objective on the non-negative linear layer. Many prototypes end up with zero weight to
  every class and are effectively pruned.

## What this POC implements from scratch

- The **prototype presence head** (1×1 conv → per-location softmax over prototypes →
  max-pool presence) in `model.py`.
- The **self-supervised alignment loss** on two augmented views (`model.py` / loss).
- The **sparse non-negative scoring-sheet classifier** and its sparsity objective.
- A **visualization** of which few prototypes fire for an image and where.

Reused (hybrid rule): backbone from `common/backbones.py` (a ConvNeXt/ResNet variant; see
`notes.md`), CUB loader from `common/data/cub.py`, two-view augmentation built on the
shared transforms.

## How to run (once implemented)

```bash
export CUB_ROOT=/path/to/CUB_200_2011
python -m prototype_methods.pipnet.train
```

**Signature artifact:** for a few test images, the *small set* of prototypes that are
"present", each with its activation region and its (non-negative) contribution to the
predicted class — the scoring sheet.

## POC simplifications (default tiny config)

- Few classes, modest prototype count, abbreviated self-supervised phase.
- We prioritize showing the *self-supervised alignment + sparse scoring sheet* mechanism
  over matching paper accuracy. Deviations logged in `notes.md`.
