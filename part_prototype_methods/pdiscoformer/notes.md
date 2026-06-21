# PDiscoFormer POC — implementation notes

Fill in during the PDiscoFormer implementation step.

## Deviations from the paper (planned)
- Smaller frozen DINO ViT (e.g. `dino_vits16`) and small `K`.
- Start with a subset of losses (TV + entropy + equivariance); add orthogonality and the
  classification head incrementally.

## Key mechanism reminders
- The backbone is **frozen**; only the part head trains — much cheaper than SCOPS.
- The headline idea vs. SCOPS: **relax** the single-blob concentration prior →
  total-variation smoothness + entropy/Gumbel → parts may be multi-modal / spread out.
- Patch grid must be consistent: img_size / ViT patch size = (h, w) of assignment maps.
- Build `common.backbones` support for DINO/DINOv2 (via `timm`) as part of this step.

## TODO
- [ ] Add DINO/DINOv2 loader to common/backbones.py (timm)
- [ ] `forward` (+ pooled part features) in model.py
- [ ] TV, entropy, equivariance, orthogonality in losses.py
- [ ] Training loop (frozen backbone) in train.py
- [ ] Multi-modal part visualization (contrast with SCOPS compact parts)
