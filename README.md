# understanding-prototype-and-concept-models

Understanding prototype and concept models for interpretable image recognition — paper
summaries and **from-scratch proof-of-concept (POC) code** for the core mechanisms.

This is a study repo. For each salient paper we keep a short summary plus a small,
heavily-commented implementation that runs on a tiny data subset in minutes (and scales up
to full CUB-200-2011 if you want real numbers). The emphasis is on *understanding the
mechanism*, not reproducing benchmark accuracy.

See [CLAUDE.md](CLAUDE.md) for the full project guide (conventions, structure, setup).

## Families studied

1. **Prototype methods** — "this looks like that": classify by matching image patches to
   learned class prototypes. *Active.*
2. **Part-prototype / part-discovery methods** — discover semantically consistent object
   parts without part labels. *Active.*
3. **Concept models (SAEs)** — sparse-autoencoder concept extraction. *Planned / deferred.*

## POC roadmap

| Family | Paper | Role | Folder |
|--------|-------|------|--------|
| Prototype | ProtoPNet — *This Looks Like That* (Chen et al., 2019) | first | [`prototype_methods/protopnet/`](prototype_methods/protopnet/) |
| Prototype | PIP-Net (Nauta et al., CVPR 2023) | latest | [`prototype_methods/pipnet/`](prototype_methods/pipnet/) |
| Part-prototype | SCOPS (Hung et al., CVPR 2019) | first | [`part_prototype_methods/scops/`](part_prototype_methods/scops/) |
| Part-prototype | PDiscoFormer (Aniraj et al., ECCV 2024) | latest | [`part_prototype_methods/pdiscoformer/`](part_prototype_methods/pdiscoformer/) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download CUB-200-2011 and set CUB_ROOT (see CLAUDE.md), then e.g.:
python -m prototype_methods.protopnet.train
```

## Repository layout

- `common/` — shared dataset loader (CUB), backbones, and visualization helpers.
- `prototype_methods/`, `part_prototype_methods/` — one folder per POC.
- `papers/` — source PDFs (kept locally, not committed).

## License

MIT — see [LICENSE](LICENSE).
