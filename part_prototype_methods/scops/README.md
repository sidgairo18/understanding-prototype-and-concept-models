# SCOPS — Self-Supervised Co-Part Segmentation (Hung et al., CVPR 2019)

**Role in this repo:** the *first* important part-discovery method — discovers object parts
**without any part annotations**.

> Paper PDF: `papers/part_prototype_methods/SCOPS- Self-Supervised Co-Part Segmentation.pdf`

## The idea in one paragraph

Given a collection of images of the same object category (e.g. birds), SCOPS learns to
segment each image into `K` **parts** (head, body, wing, ...) that are **consistent across
the whole collection** — part 3 means "head" in every image — using only self-supervision.
There are no ground-truth part masks. Instead, four losses encode what a "good part"
should be, and a network that minimizes them discovers semantically meaningful parts.

## Architecture

```
image ──▶ encoder (e.g. VGG/ResNet, ImageNet features) ──▶ part response maps
                                                            R ∈ R^{(K+1)×H×W}
                                          (softmax over K parts + 1 background, per pixel)
```

The output is a per-pixel soft assignment to `K` parts plus background. Everything is
driven by the loss, not labels.

## The four losses (this is the heart of SCOPS)

1. **Geometric concentration** — each part should be a *single compact blob*, not scattered
   pixels. Penalizes the spatial variance of each part's response around its center of mass.
2. **Equivariance** — apply a known spatial transform `T` (affine + TPS warp) to the image;
   the predicted parts must transform the same way: `parts(T(x)) ≈ T(parts(x))`. Forces
   parts to track object geometry rather than fixed image locations.
3. **Semantic consistency** — a part should look the same across images. SCOPS pools deep
   features within each part to form a part feature, and (via a learned global **part
   basis** / dictionary) encourages each part's appearance to be consistent across the
   dataset — so "part 3" has a stable meaning collection-wide.
4. **Background / saliency constraint** — uses an unsupervised **saliency map** so parts
   land on the *foreground object*, and background pixels are assigned to the background
   channel.

A reconstruction-style term ties the pooled part features back to the image features.

## What this POC implements from scratch

- The **part response head** (encoder → `(K+1)`-way per-pixel softmax) in `model.py`.
- The four losses in `losses.py`:
  - concentration (center-of-mass + spatial variance),
  - equivariance under a sampled affine/TPS transform,
  - semantic consistency via pooled part features,
  - background/saliency term.
- A **visualization** that overlays the discovered `K`-part segmentation on birds
  (`common/viz.colorize_parts`).

Reused (hybrid rule): encoder/backbone from `common/backbones.py`, CUB loader from
`common/data/cub.py` (with `return_bbox`/saliency as needed).

## How to run (once implemented)

```bash
export CUB_ROOT=/path/to/CUB_200_2011
python -m part_prototype_methods.scops.train
```

**Signature artifact:** colored part-segmentation maps for several birds, where the *same
color = same semantic part* across different images.

## POC simplifications (default tiny config)

- Small `K` (e.g. 4 parts), a simplified saliency proxy, affine-only equivariance to start
  (TPS optional). Goal: show the four-loss self-supervised part-discovery mechanism, not
  paper-level segmentation quality. Deviations logged in `notes.md`.
