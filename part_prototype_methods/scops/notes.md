# SCOPS POC — implementation notes

Fill in during the SCOPS implementation step.

## Deviations from the paper (planned)
- Smaller `K` (4 parts) and smaller input resolution for speed.
- Saliency: paper uses an off-the-shelf unsupervised saliency network; the POC may start
  with a simple proxy (e.g. center prior or a lightweight saliency model) — record choice.
- Equivariance: start affine-only; TPS warp is optional (`use_tps`).

## Key mechanism reminders
- Response maps are a per-pixel softmax over `K parts + 1 background`.
- Concentration uses the response-weighted center of mass and spatial variance.
- Equivariance compares `T(parts(x))` to `parts(T(x))` — be careful applying T to the maps
  consistently (same grid_sample as the image).
- Semantic consistency needs pooled part features tied to a learned global part basis.

## TODO
- [ ] `forward` (+ pooled part features) in model.py
- [ ] Four losses in losses.py
- [ ] Transform sampler (affine, optional TPS) + saliency source
- [ ] Training loop in train.py
- [ ] Part-segmentation overlay visualization
