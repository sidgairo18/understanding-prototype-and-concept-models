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
| 1 | Prototype | This Looks Like That (ProtoPNet), Chen et al. NeurIPS 2019 | first | `prototype_methods/protopnet/` | planned |
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
