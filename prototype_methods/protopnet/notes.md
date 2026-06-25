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

## Add-on layers: why `ReLU` then `Sigmoid`

The add-on is `Conv1x1 → ReLU → Conv1x1 → Sigmoid` (`model.py`). The two activations do
different jobs:

- **`ReLU` (middle)** — plain nonlinearity. The 1×1 convs act per spatial location, so the
  add-on is a tiny per-pixel 2-layer MLP; without a nonlinearity between them the two convs
  collapse into one linear map. Nothing subtle.
- **`Sigmoid` (final)** — *bounds* every feature channel to (0,1), so each latent patch lives
  in the unit hypercube `[0,1]^D` (`D = prototype_dim`). This is the load-bearing choice:
  1. It caps the squared L2 distance: with patch and prototype in `[0,1]^D`, `‖z−p‖² ≤ D`.
     That ceiling **is** `max_dist = prototype_dim`, which the cluster/separation costs need
     for the inverted-distance trick `max(max_dist − d)` to be a valid surrogate for `min d`.
     An unbounded final activation (e.g. ReLU) would make `max_dist` meaningless.
  2. Keeps distances — and the log-similarity — well-conditioned, and keeps prototypes (init
     in `[0,1]` via `torch.rand`) on the same scale as patches from step one.
  3. Realizes the probabilistic interpretation's latent domain `Ω = [0,1]^{H₁×W₁×D}`
     (supplement S2; see `notes_extended.md` §3.6).

We keep **raw L2 in this sigmoid space (no L2-normalization)** — cosine variants (e.g. TesNet)
normalize instead and would drop the sigmoid + change `max_dist`.

### Verified against the author's code (`cfchen-duke/ProtoPNet`, `master`)

Checked our choices against the original implementation on 2026-06-26:

| Claim | Author's code | Match |
|---|---|---|
| Default add-on = `Conv→ReLU→Conv→Sigmoid` | `model.py` else-branch is exactly this; bottleneck branch also ends in `Sigmoid`, intermediate acts are `ReLU` | ✅ |
| Distance = `‖x‖²−2·x·p+‖p‖²`, ReLU-clamped | `_l2_convolution`: `F.relu(x2_patch_sum − 2*xp + p2)` | ✅ |
| Similarity = `log((d²+1)/(d²+ε))`, ε=1e-4 | `distance_2_similarity` 'log' branch; `self.epsilon = 1e-4` | ✅ |
| `max_dist = prototype_dim` | `train_and_test.py`: `prototype_shape[1]*[2]*[3]` = `D*1*1` | ✅ |
| Cluster/sep via `max(max_dist − min_dist)` masked by class | `train_and_test.py` cluster_cost / separation_cost — identical form | ✅ |
| No L2-normalization before distances | confirmed absent | ✅ |

## Frozen-backbone BatchNorm (intentional improvement over the reference)

`requires_grad=False` freezes a BN layer's affine params (γ, β) but **not** its running-stat
buffers — `running_mean`/`running_var` update on every forward in `train()` mode regardless
of `requires_grad`. So a "frozen" backbone left in `train()` keeps overwriting its pretrained
stats with tiny-subset stats (and normalizes with batch stats). The original ProtoPNet has
exactly this behavior during warm-up (it only sets `requires_grad=False`).

We override `ProtoPNet.train()` to force any frozen BN **inside the backbone** back to eval.
Design choices that make it bulletproof:
- **Single source of truth**: BN mode is re-derived from `requires_grad` (set by `set_mode`),
  not a separate flag that could desync.
- **At the `train()` chokepoint**: every `model.train()` (any call site) re-asserts it, so it
  can't be silently undone.
- **Scoped to `self.backbone`**: BN elsewhere (e.g. a future trainable add-on BN) is untouched.
- **Per-BN**: partial freezing works — only frozen blocks' BN go to eval.

This diverges from the reference *on purpose*; it matters most for permanently-frozen
backbones (e.g. a DINO/ViT feature-extractor variant) where the pretrained stats should be
preserved.

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
