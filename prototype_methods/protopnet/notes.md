# ProtoPNet POC — implementation notes

Status: **implemented & verified** on a synthetic learnable dataset (the real-CUB run is a
one-liner once `CUB_ROOT` points at the downloaded data).

## Deviations from the paper (as built)
- **Backbone**: default `resnet34` (paper uses `resnet50`/VGG/DenseNet). Configurable via
  `config.backbone`.
- **Schedule**: simplified to warm → joint, with a prototype push at scheduled epochs and
  always on the final epoch; after each push the last layer is optimized with Adam for
  `last_layer_iters` mini-epochs (the paper solves a convex last-layer problem — Adam is a
  close, simpler stand-in). Cluster/separation/L1 weights follow paper defaults.
- **Tiny defaults**: few classes, 10 prototypes/class, short schedule — for fast iteration,
  not paper accuracy.
- **Prototype visualization**: we overlay each prototype's *upsampled activation heatmap*
  on its source training image rather than cropping the exact receptive-field bounding box.
  Same intent ("this looks like that"), simpler implementation.

## Faithful-to-paper choices (kept on purpose)
- Sigmoid add-on features in [0,1] + **raw squared-L2** distance (not cosine/normalized);
  `max_dist = prototype_dim` follows from the [0,1] bound.
- Similarity = `log((d²+1)/(d²+ε))`, ε=1e-4.
- Cluster cost (close to a same-class prototype) and **negative** separation cost (far from
  other-class prototypes) via the inverted-distance trick.
- Last layer init +1 (own class) / −0.5 (others); **L1 only on cross-class** weights; last
  layer trained only in the post-push phase.
- Push uses an **un-augmented, un-shuffled** loader so indices map back to real images.

## Gotchas confirmed
- After a push, accuracy can dip until the last layer is re-optimized (seen in the smoke
  run) — that re-optimization step is essential.
- Keep ε small but nonzero; it sets the similarity ceiling as d²→0.

## Verified (synthetic smoke test)
- Forward shapes; loss + cluster cost decrease; test acc → 1.0 on a learnable toy set.
- Push mutates all prototypes and returns per-prototype source metadata.
- "This looks like that" figures render (test heatmap ↔ source-patch heatmap).

## TODO (when scaling to real CUB)
- [ ] Run on full/few-class CUB with `pretrained=True` and report the signature figure.
- [ ] Optional: exact receptive-field bbox crops for prototypes (vs. upsampled heatmap).
