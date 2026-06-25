"""Train the ProtoPNet POC on (a subset of) CUB-200-2011.

Run from the repo root::

    export CUB_ROOT=/path/to/CUB_200_2011
    python -m prototype_methods.protopnet.train

The schedule alternates (paper Section 3):
  1. warm  : train add-on + prototypes (backbone frozen)        — epochs < warm_epochs
  2. joint : train backbone + add-on + prototypes               — epochs >= warm_epochs
  3. push  : project prototypes onto nearest training patches   — at scheduled epochs
  4. last  : convex-ish optimization of the last layer (CE+L1)  — right after each push

The functions here (`train_one_epoch`, `evaluate`, `compute_loss`, `run_training`) are kept
loader-agnostic so they can be driven by any DataLoader (the synthetic smoke test reuses
them). `main` wires up the CUB loaders and the final "this looks like that" visualization.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.data.cub import CUB200, build_transforms
from .config import ProtoPNetConfig
from .model import ProtoPNet
from .push import push_prototypes


# ---------------------------------------------------------------------------- loss / steps
def compute_loss(model: ProtoPNet, logits, min_distances, labels, cfg: ProtoPNetConfig):
    """Cross-entropy + cluster + (−)separation + L1, the full ProtoPNet objective.

    Separation enters with a *minus* sign: minimizing the loss pushes each image *away*
    from other-class prototypes. The cluster/separation terms are constant w.r.t. the last
    layer, so the same objective is reused during the last-layer-only phase.
    """
    ce = F.cross_entropy(logits, labels)
    cluster, separation = model.cluster_and_separation_costs(min_distances, labels)
    l1 = model.last_layer_l1()
    loss = (
        ce
        + cfg.lambda_cluster * cluster
        - cfg.lambda_separation * separation
        + cfg.lambda_l1 * l1
    )
    stats = {
        "loss": float(loss),
        "ce": float(ce),
        "cluster": float(cluster),
        "sep": float(separation),
        "l1": float(l1),
    }
    return loss, stats


def train_one_epoch(model, loader, optimizer, cfg) -> dict:
    model.train()
    totals, n = {}, 0
    for images, labels in loader:
        images, labels = images.to(cfg.device), labels.to(cfg.device)
        logits, min_distances = model(images)
        loss, stats = compute_loss(model, logits, min_distances, labels, cfg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v * images.size(0)
        n += images.size(0)
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, cfg) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(cfg.device), labels.to(cfg.device)
        logits, _ = model(images)
        correct += int((logits.argmax(1) == labels).sum())
        total += labels.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------- schedule
def should_push(epoch: int, cfg: ProtoPNetConfig) -> bool:
    """Push after warm-up at every `push_every` epochs, and always on the final epoch."""
    if epoch < cfg.warm_epochs:
        return False
    if epoch == cfg.epochs - 1:
        return True
    return (epoch - cfg.warm_epochs) % cfg.push_every == 0


def _build_optimizers(model: ProtoPNet, cfg: ProtoPNetConfig):
    warm = torch.optim.Adam(
        [{"params": model.add_on.parameters()}, {"params": [model.prototypes]}], lr=cfg.lr
    )
    joint = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": cfg.lr * 0.1},
            {"params": model.add_on.parameters()},
            {"params": [model.prototypes]},
        ],
        lr=cfg.lr,
    )
    last = torch.optim.Adam(model.classifier.parameters(), lr=cfg.lr * 0.1)
    return warm, joint, last


def run_training(cfg, model, train_loader, test_loader, push_loader) -> list:
    """Run the full alternating schedule. Returns the metadata from the final push."""
    warm_opt, joint_opt, last_opt = _build_optimizers(model, cfg)
    push_meta: list = [None] * model.num_prototypes

    for epoch in range(cfg.epochs):
        if epoch < cfg.warm_epochs:
            model.set_mode("warm")
            stats = train_one_epoch(model, train_loader, warm_opt, cfg)
            phase = "warm"
        else:
            model.set_mode("joint")
            stats = train_one_epoch(model, train_loader, joint_opt, cfg)
            phase = "joint"

        acc = evaluate(model, test_loader, cfg)
        print(f"[epoch {epoch:02d}] {phase:5s} loss={stats['loss']:.3f} "
              f"ce={stats['ce']:.3f} clst={stats['cluster']:.3f} sep={stats['sep']:.3f} "
              f"| test_acc={acc:.3f}")

        if should_push(epoch, cfg):
            push_meta = push_prototypes(model, push_loader, cfg.device)
            acc_after = evaluate(model, test_loader, cfg)
            print(f"           push -> test_acc={acc_after:.3f}; optimizing last layer...")
            model.set_mode("last")
            for _ in range(cfg.last_layer_iters):
                train_one_epoch(model, train_loader, last_opt, cfg)
            print(f"           last-layer done -> test_acc={evaluate(model, test_loader, cfg):.3f}")

    return push_meta


# ---------------------------------------------------------------------------- data
def build_dataloaders(cfg: ProtoPNetConfig):
    """CUB train/test loaders plus a clean (un-shuffled, un-augmented) push loader.

    The push loader shares index order with `push_set`, so push metadata indexes back into
    real training images for visualization.
    """
    train_set = CUB200(
        data_root=cfg.data_root, train=True, num_classes=cfg.num_classes,
        images_per_class=cfg.images_per_class,
        transform=build_transforms(cfg.img_size, train=True),
    )
    test_set = CUB200(
        data_root=cfg.data_root, train=False, num_classes=cfg.num_classes,
        transform=build_transforms(cfg.img_size, train=False),
    )
    push_set = CUB200(  # same train images, but normalize-only transform + stable order
        data_root=cfg.data_root, train=True, num_classes=cfg.num_classes,
        images_per_class=cfg.images_per_class,
        transform=build_transforms(cfg.img_size, train=False),
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    push_loader = DataLoader(push_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    return train_loader, test_loader, push_loader, test_set, push_set


# ---------------------------------------------------------------------------- visualization
@torch.no_grad()
def visualize_prototypes(cfg, model, test_set, push_set, push_meta, n_images=4, top_k=3):
    """Save the signature "this looks like that" figures.

    For a few test images: predict the class, take the top-K most-activated prototypes of
    that class, and for each show (left) the test image with the prototype's activation
    heatmap and (right) the prototype's *source* training patch (the image it was pushed
    onto) with its activation heatmap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import viz

    os.makedirs(cfg.out_dir, exist_ok=True)
    model.eval()
    ppc = cfg.prototypes_per_class

    n_images = min(n_images, len(test_set))
    for i in range(n_images):
        image, label = test_set[i]
        x = image.unsqueeze(0).to(cfg.device)
        logits, _ = model(x)
        pred = int(logits.argmax(1))
        acts = model.prototype_activation_maps(x)[0]            # (m, h, w)
        per_proto_score = acts.amax(dim=(1, 2))                 # (m,) max similarity

        # Top-K prototypes belonging to the predicted class.
        class_proto_ids = list(range(pred * ppc, pred * ppc + ppc))
        ranked = sorted(class_proto_ids, key=lambda j: float(per_proto_score[j]), reverse=True)
        chosen = ranked[:top_k]

        fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 3 * len(chosen)))
        if len(chosen) == 1:
            axes = axes.reshape(1, 2)
        for r, j in enumerate(chosen):
            test_rgb, test_heat = viz.overlay_heatmap(image, acts[j])
            axes[r, 0].imshow(test_rgb)
            axes[r, 0].imshow(test_heat, cmap="jet", alpha=0.5)
            axes[r, 0].set_title(f"test img (pred={pred}, gt={label})\nproto {j} act={per_proto_score[j]:.2f}")
            axes[r, 0].axis("off")

            meta = push_meta[j] if j < len(push_meta) else None
            if meta is not None:
                src_img, _ = push_set[meta["dataset_index"]]
                src_act = model.prototype_activation_maps(src_img.unsqueeze(0).to(cfg.device))[0, j]
                src_rgb, src_heat = viz.overlay_heatmap(src_img, src_act)
                axes[r, 1].imshow(src_rgb)
                axes[r, 1].imshow(src_heat, cmap="jet", alpha=0.5)
                axes[r, 1].set_title(f"...looks like THAT\n(source train patch, proto {j})")
            else:
                axes[r, 1].text(0.5, 0.5, "prototype not pushed", ha="center", va="center")
            axes[r, 1].axis("off")

        fig.tight_layout()
        out = os.path.join(cfg.out_dir, f"this_looks_like_that_{i:02d}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"saved {out}")


# ---------------------------------------------------------------------------- entrypoint
def main(cfg: ProtoPNetConfig | None = None) -> None:
    cfg = cfg or ProtoPNetConfig()
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    train_loader, test_loader, push_loader, test_set, push_set = build_dataloaders(cfg)
    model = ProtoPNet(cfg).to(cfg.device)
    print(f"ProtoPNet: {model.num_prototypes} prototypes "
          f"({cfg.prototypes_per_class}/class × {cfg.num_classes} classes), device={cfg.device}")

    push_meta = run_training(cfg, model, train_loader, test_loader, push_loader)
    visualize_prototypes(cfg, model, test_set, push_set, push_meta)
    print("done.")


if __name__ == "__main__":
    main()
