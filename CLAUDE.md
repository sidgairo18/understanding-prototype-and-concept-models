# CLAUDE.md

Guidance for working in this repository. Read this first.

## What this project is

An **educational** repository for deeply understanding interpretable-by-design vision
models. For each salient paper we keep (a) a short written summary and (b) a
**proof-of-concept (POC) implementation** that builds the paper's *core mechanism* from
scratch so the idea is legible in code, not just in prose.

Three families are in scope, studied in order:

1. **Prototype methods** — classify by comparing image patches to learned class
   prototypes ("this looks like that"). *Active focus.*
2. **Part-prototype / part-discovery methods** — discover semantically consistent object
   parts (e.g. bird head, wing, leg) without part annotations. *Active focus.*
3. **Concept models (SAEs, etc.)** — sparse-autoencoder-style concept extraction.
   *Deferred — future work, no code yet.*

The goal is **mechanistic understanding, not benchmark accuracy**. POCs are designed to
run on a tiny data subset in minutes, with config to scale up to full CUB-200-2011 for
anyone who wants to push numbers.

## Repository structure

```
CLAUDE.md                      # this file
README.md                      # short public-facing overview
requirements.txt               # Python dependencies
papers/                        # source PDFs (untracked, git-ignored, kept locally)
  prototype_methods/           #   11 prototype-method papers
  part_prototype_methods/      #   3 part-discovery papers
common/                        # shared, paper-agnostic code (reuse this!)
  data/cub.py                  #   CUB-200-2011 dataset + subset/config + transforms
  backbones.py                 #   torchvision ResNet / ViT and DINO feature extractors
  viz.py                       #   prototype-activation & part-map visualization helpers
prototype_methods/
  protopnet/                   # POC: ProtoPNet (first, Chen et al. 2019)
  pipnet/                      # POC: PIP-Net (latest, CVPR 2023)
part_prototype_methods/
  scops/                       # POC: SCOPS (first, CVPR 2019)
  pdiscoformer/                # POC: PDiscoFormer (latest, ECCV 2024)
```

Each POC folder follows the same layout:

```
README.md      # paper summary + how this POC maps to the paper's equations/figures
notes.md       # implementation gotchas, deviations from the paper, TODOs
config.py      # dataclass of hyperparameters (tiny-default; scalable)
model.py       # the core mechanism, written from scratch and heavily commented
train.py       # training/eval loop, runnable as `python -m <pkg>.train`
losses.py      # (part-discovery POCs) the discovery loss terms
push.py        # (ProtoPNet) prototype projection onto nearest training patches
```

## Paper roadmap

| # | Family | Paper | Role | POC folder | Status |
|---|--------|-------|------|------------|--------|
| 1 | Prototype | This Looks Like That (ProtoPNet), Chen et al. NeurIPS 2019 | first | `prototype_methods/protopnet/` | done |
| 2 | Prototype | PIP-Net, Nauta et al. CVPR 2023 | latest | `prototype_methods/pipnet/` | planned |
| 3 | Part-prototype | SCOPS: Self-Supervised Co-Part Segmentation, Hung et al. CVPR 2019 | first | `part_prototype_methods/scops/` | planned |
| 4 | Part-prototype | PDiscoFormer, Aniraj et al. ECCV 2024 | latest | `part_prototype_methods/pdiscoformer/` | planned |

Other papers present in `papers/` (ProtoTree, ProtoPShare, TesNet, Deformable ProtoPNet,
ProtoPool, ProtoViT, ProtoPNeXt, PDiscoNet) are read/summarized but **not** slated for a
POC unless we later decide to add one. Update the **Status** column as POCs progress
(`planned` → `in-progress` → `done`).

## POC conventions

- **Hybrid implementation.** Write the *signature mechanism* from scratch and comment it
  densely (the prototype layer + push, the part-discovery losses, etc.). **Reuse** stock
  components for everything else: torchvision/DINO backbones and the shared CUB loader in
  `common/`. Do not reinvent backbones or data pipelines per POC.
- **Reuse `common/` first.** Before writing data/backbone/viz code in a POC, check
  `common/`. Add to `common/` when something is genuinely shared; keep paper-specific
  logic inside the POC folder.
- **Readable over fast.** Favor clear, paper-faithful code over micro-optimized code.
  When the POC deviates from the paper (smaller backbone, fewer prototypes, simplified
  loss), record it in that POC's `notes.md`.
- **Configurable scale.** Every POC reads hyperparameters from its `config.py` and
  defaults to a *tiny* setting (a handful of CUB classes, few epochs) that runs on
  CPU/modest GPU. The same config scales to full CUB-200-2011.
- **Runnable entrypoints.** Train via module path from the repo root, e.g.
  `python -m prototype_methods.protopnet.train`. Each POC's README documents its exact
  command and the signature artifact it produces.

## Shared training infrastructure (wire into every POC's `train.py`)

ProtoPNet is the **reference implementation** for the cross-cutting machinery below. Each
*new* POC's `train.py` should wire in the same pieces (copy the pattern from
`prototype_methods/protopnet/`) so all POCs train, scale, checkpoint, and run on SLURM the
same way. Keep the paper's signature mechanism the star; this is the boilerplate around it.

1. **Distributed (DDP) via `common/distributed.py`.** Single-process by default; DDP engages
   only under `torchrun`. In `main()`: `ctx = init_distributed(...)`, set `cfg.device =
   ctx.device`; shard with `DistributedSampler` (call `set_epoch` each epoch); `wrap_ddp`;
   use `unwrap_model()` to reach custom methods; gate prints/saves on `is_main_process()`;
   reduce metrics with `all_reduce_sum`; `cleanup()` at the end. Backbone-with-frozen-BN POCs
   should override `Module.train()` to force frozen BN back to `eval()` (see ProtoPNet
   `model.py` / `notes.md`).
2. **Checkpointing + `--resume` (per-POC `checkpoint.py`).** Mirror
   `prototype_methods/protopnet/checkpoint.py`: write a rolling `ckpt_last.pt` **and** a
   best-metric `ckpt_best.pt` at each epoch end, **atomically** (temp file + `os.replace`)
   so a walltime kill can't corrupt them; **rank 0 saves, every rank loads**. The checkpoint
   must hold *all* state needed to continue bit-faithfully: model `state_dict`, **every**
   optimizer, the LR scheduler, epoch, RNG, running `best_acc`, and any POC-specific state
   (e.g. ProtoPNet's `push_meta`). On resume, **validate** that the checkpoint's
   architecture-defining config matches before loading (hard error on mismatch), restore
   model weights *before* the DDP wrap, then restore optimizer/scheduler/RNG/epoch and
   continue at `epoch+1`. Load with `weights_only=False` (checkpoints hold non-tensor data).
3. **`--resume` CLI.** Expose `main(cfg)` plus a small `_cli_config()` (argparse) with at
   least `--resume [PATH]` (bare = auto-detect `<out_dir>/<ckpt_name>`), `--out-dir`,
   `--data-root`, `--epochs`, `--num-classes`. Same argv reaches every `torchrun` rank.
4. **SLURM launchers (`scripts/slurm/`).** `train_poc.sh <poc>` maps the POC name to its
   train module and submits via `_sbatch_submit.sh` (partitions `gpu16,gpu17,gpu20,gpu22,gpu24`;
   conda env `proto-concept`; `torchrun` for multi-GPU). **Job chaining** (`CHAIN_JOBS=N` →
   array `-a 1-N%1`) auto-appends `--resume`, so a new POC gets chaining *for free* once its
   `train.py` implements the `--resume` hook above. Add the POC to the `case` in
   `train_poc.sh` if it isn't already.
5. **Experiment tracking via `common/wandb_logger.py`.** Construct `WandbLogger(cfg, ctx)`
   (rank-0-only; a **no-op** unless `cfg.wandb` / `--wandb`, so tiny/smoke runs need neither
   the dependency nor a login). Log per-epoch scalars with `logger.log({...}, step=epoch)`,
   the signature artifact with `logger.log_figure(name, fig)`, then `logger.finish()`. Add
   the `wandb*` fields to the POC's `config.py` and the `--wandb[/-project/-run-name/-mode]`
   CLI flags (copy ProtoPNet). The logger reads settings with `getattr` defaults, so it
   works even if a POC config defines only some fields — and covers eval scripts too.

When a POC genuinely needs a new shared capability, add it to `common/` (or the shared
launcher) rather than copying it per POC.

## Environment & setup

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__)"
```

PyTorch is the framework. A GPU is optional for the tiny default configs and recommended
for full-CUB runs.

## Dataset setup (CUB-200-2011)

All four POCs share **CUB-200-2011** (Caltech-UCSD Birds, 200 classes, 11,788 images).

1. Download `CUB_200_2011.tgz` from the official source and extract it.
2. Point the loader at it via the `CUB_ROOT` env var or the `data_root` field in a POC's
   `config.py`.
3. `common/data/cub.py` exposes a subset switch — `num_classes` and `images_per_class`
   (plus the official train/test split) — so the *default* run uses only a few classes
   for fast iteration, and the *full* run uses all 200.

Datasets and the `papers/` PDFs are **git-ignored** — never commit them.

## How to run (once POCs are implemented)

```bash
# Prototype methods
python -m prototype_methods.protopnet.train      # ProtoPNet
python -m prototype_methods.pipnet.train         # PIP-Net

# Part-prototype methods
python -m part_prototype_methods.scops.train     # SCOPS
python -m part_prototype_methods.pdiscoformer.train  # PDiscoFormer
```

Each `train.py` accepts overrides for the tiny-vs-full switch (see each POC's README).

## Glossary

- **Prototype** — a learned vector in feature space representing a recurring, class-linked
  visual pattern (a patch concept). Classification compares image patches to prototypes.
- **Prototype layer** — computes similarity between every spatial location of a feature
  map and each prototype; the max similarity is that prototype's activation.
- **Push / projection (ProtoPNet)** — periodically replace each prototype with the nearest
  real training-image patch, so every prototype is visualizable as an actual image region.
- **Latent patch** — one spatial cell of the CNN/ViT feature map, with a receptive field
  back in the input image.
- **Part-response / attention map** — per-part spatial heatmap indicating where a
  discovered part is present in the image.
- **Part-discovery losses** — concentration (each part is spatially compact), equivariance
  (parts move with the image under transforms), semantic consistency (a part means the
  same thing across images), plus background/entropy/total-variation regularizers.

## Notes for Claude (working agreements)

- Keep the **core mechanism the star**: it should read like the paper. Push boilerplate
  into `common/` or clearly-marked helpers.
- Implement and verify **one POC at a time**, in roadmap order, before moving on.
- After each POC, run the tiny-subset config end-to-end and produce its signature artifact
  (prototype activations for ProtoPNet/PIP-Net; part-segmentation overlays for
  SCOPS/PDiscoFormer) as the verification step.
- Do not commit `papers/` PDFs, datasets, checkpoints, or other large/binary artifacts.
- When behavior deviates from the paper, write it down in that POC's `notes.md`.
- Concept models / SAEs are out of scope until the prototype and part-prototype POCs land.
```
