# ProtoPNet POC — implementation notes

Running log of decisions, deviations from the paper, and gotchas. Fill in during the POC
implementation step.

## Deviations from the paper (planned)
- Smaller backbone (`resnet34`) and fewer classes by default for speed.
- Shortened alternating schedule (fewer epochs between pushes).

## Gotchas to watch
- Similarity is `log((d²+1)/(d²+ε))` — keep ε small but nonzero; it controls the
  similarity ceiling at d→0.
- After a push, the last layer must be re-optimized or accuracy drops.
- L1 sparsity is applied only to **cross-class** last-layer weights (own-class init = +1).
- Prototype receptive field: to crop the source patch, map the feature-grid (row, col)
  back through the backbone's receptive field, or upsample the activation (POC uses the
  latter via `common/viz.py`).

## TODO
- [ ] Implement `_squared_distances`, `forward`, `cluster_and_separation_costs` in model.py
- [ ] Implement `push_prototypes` in push.py
- [ ] Implement the alternating training loop in train.py
- [ ] Add a prototype-activation visualization (the "this looks like that" figure)
