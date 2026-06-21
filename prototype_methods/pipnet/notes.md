# PIP-Net POC — implementation notes

Fill in during the PIP-Net implementation step.

## Deviations from the paper (planned)
- Backbone: paper uses ConvNeXt-tiny; the POC may start with ResNet-50 from
  `common/backbones.py` and add ConvNeXt later (via `timm`).
- Shortened self-supervised pretraining.

## Key mechanism reminders
- Prototypes are **not** class-tied; the classifier learns the prototype→class mapping.
- Classifier weights are **non-negative** (positive-evidence-only reasoning).
- Sparsity is central: most prototypes should end with zero weight to every class.
- Alignment loss needs the *known geometric relation* between the two augmented views to
  match corresponding patches.

## TODO
- [ ] Paired two-view augmentation dataset wrapper
- [ ] `presence`, `forward`, `alignment_loss` in model.py
- [ ] Non-negativity + sparsity objective on the classifier
- [ ] Two-phase training loop in train.py
- [ ] Scoring-sheet visualization (few present prototypes + contributions)
