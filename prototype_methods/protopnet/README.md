# ProtoPNet — *This Looks Like That* (Chen et al., NeurIPS 2019)

**Role in this repo:** the *first* important prototype method — the foundation the whole
family builds on.

**Status:** implemented & verified end-to-end (`model.py` + `push.py` + `train.py`). The
core mechanism (prototype layer, cluster/separation losses, push, last-layer schedule, and
the "this looks like that" visualization) is complete; defaults are tiny for fast runs.
See `notes.md` for deviations.

> Paper PDF: `papers/prototype_methods/This Looks Like That- Deep Learning for Interpretable Image Recognition.pdf`

## The idea in one paragraph

ProtoPNet classifies an image by checking *"which class prototypes does this image look
like, and where?"*. The network learns a fixed set of **prototypes** — vectors in CNN
feature space, each tied to one class. For a test image, the model finds, for every
prototype, the image patch most similar to it, and turns that similarity into evidence for
the prototype's class. Because each prototype is *projected* onto a real training patch,
its meaning is human-visible ("this looks like *that* part of a training bird"), giving a
self-explaining classification: `0.9 × (looks like this wing) + 0.8 × (looks like this
head) + ... → class`.

## Architecture

```
image ──▶ CNN backbone f  ──▶ feature map z ∈ R^{D×h×w}   (a grid of latent patches)
                                  │
                                  ▼
              prototype layer g_p:  for each prototype p_j (∈ R^D),
                 sim_j = max over the h×w patches of  similarity(patch, p_j)
                                  │
                                  ▼
              fully-connected h:  logits = W · [sim_1, ..., sim_m]
```

- **Backbone** `f`: ImageNet-pretrained CNN (ResNet/VGG/DenseNet) truncated to its last
  conv feature map, plus two 1×1 "add-on" conv layers. Output `D`-dim feature per spatial
  cell (D = prototype dim, e.g. 128).
- **Prototype layer** `g_p`: `m` prototypes, **`m_k` per class** (paper uses 10/class).
  Similarity is derived from squared L2 distance:
  `sim = log((d² + 1) / (d² + ε))`, so smaller distance ⇒ larger similarity. Each
  prototype's activation is the **max** similarity over all `h×w` patches (≈ "is this
  pattern present anywhere?") together with *where* the max occurred.
- **Classifier** `h`: a single linear layer from the `m` similarities to class logits,
  initialized so a prototype votes +1 for its own class and −0.5 for others.

## Training recipe (3 alternating stages)

1. **Warm / joint SGD** — train add-on layers + prototypes (and optionally backbone) with
   cross-entropy plus two structure terms:
   - **Cluster cost**: pull each training patch close to *some* prototype of its own class
     (every image should strongly match at least one same-class prototype).
   - **Separation cost**: push patches *away* from prototypes of other classes.
2. **Prototype push / projection** (`push.py`): replace each prototype vector with the
   *nearest latent patch from a training image of its class*. After this, every prototype
   literally **is** a real image patch and can be visualized.
3. **Last-layer convex optimization**: with backbone+prototypes frozen, fine-tune only
   `h` (with L1 sparsity on cross-class connections) to sharpen accuracy while keeping the
   explanation sparse.

Repeat 1→2→3. The push step is what makes ProtoPNet *interpretable* rather than just
prototype-regularized.

## What this POC implements from scratch

- The **prototype layer**: L2-distance-to-similarity, per-class prototype assignment,
  max-pooling over patches → activations (`model.py`).
- The **cluster + separation losses** (`model.py` / training loss).
- The **push/projection** step (`push.py`) — the core interpretability move.
- A small **visualization** of where each prototype activates on a test image, using
  `common/viz.py`.

Reused (hybrid rule): the **ResNet backbone** from `common/backbones.py`, the **CUB
loader** from `common/data/cub.py`.

## How to run (once implemented)

```bash
export CUB_ROOT=/path/to/CUB_200_2011
python -m prototype_methods.protopnet.train          # tiny default: few classes, few epochs
```

**Signature artifact:** for a handful of test images, a heatmap per top prototype showing
the patch it matches, plus the prototype's source training patch — the literal *"this
looks like that"* figure.

## POC simplifications (default tiny config)

- Few CUB classes, 10 prototypes/class, ResNet truncated backbone, short schedule.
- Goal is a faithful, legible mechanism — not the paper's reported accuracy. Deviations
  are logged in `notes.md`.
