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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from common import distributed as dist_utils
from common.data.cub import CUB200, build_transforms
from . import checkpoint
from .config import ProtoPNetConfig
from .model import ProtoPNet
from .push import push_prototypes


# ---------------------------------------------------------------------------- loss / steps
def compute_loss(core: ProtoPNet, logits, min_distances, labels, cfg: ProtoPNetConfig):
    """Cross-entropy + cluster + (−)separation + L1, the full ProtoPNet objective.

    Separation enters with a *minus* sign: minimizing the loss pushes each image *away*
    from other-class prototypes. The cluster/separation terms are constant w.r.t. the last
    layer, so the same objective is reused during the last-layer-only phase.

    Takes the *unwrapped* model (``core``): the custom cost methods live on ProtoPNet, not
    on the DDP wrapper. Every parameter these terms touch (prototypes via ``min_distances``,
    classifier via ``logits``) is reachable from the DDP forward outputs, so DDP reduces
    their gradients correctly.
    """
    ce = F.cross_entropy(logits, labels)
    cluster, separation = core.cluster_and_separation_costs(min_distances, labels)
    l1 = core.last_layer_l1()
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


def _global_means(totals: dict, n: int, cfg) -> dict:
    """Average per-sample stats across *all* ranks (so rank-0's log is the global mean).

    Packs the running sums + sample count into one tensor and does a single all-reduce.
    No-op arithmetic when not distributed.
    """
    keys = list(totals.keys())
    packed = torch.tensor([totals[k] for k in keys] + [float(n)], device=cfg.device)
    dist_utils.all_reduce_sum(packed)
    denom = max(packed[-1].item(), 1.0)
    return {k: packed[i].item() / denom for i, k in enumerate(keys)}


def train_one_epoch(model, loader, optimizer, cfg) -> dict:
    model.train()                                   # DDP recurses -> ProtoPNet.train() (frozen-BN guard)
    core = dist_utils.unwrap_model(model)
    totals, n = {}, 0
    for images, labels in loader:
        images, labels = images.to(cfg.device), labels.to(cfg.device)
        logits, min_distances = model(images)       # through DDP -> gradients are all-reduced
        loss, stats = compute_loss(core, logits, min_distances, labels, cfg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v * images.size(0)
        n += images.size(0)
    return _global_means(totals, n, cfg)


@torch.no_grad()
def evaluate(model, loader, cfg) -> float:
    model.eval()
    correct = torch.zeros((), device=cfg.device)
    total = torch.zeros((), device=cfg.device)
    for images, labels in loader:
        images, labels = images.to(cfg.device), labels.to(cfg.device)
        logits, _ = model(images)
        correct += (logits.argmax(1) == labels).sum()
        total += labels.size(0)
    # Sum local shard counts into a global accuracy (no-op when single-process).
    dist_utils.all_reduce_sum(correct)
    dist_utils.all_reduce_sum(total)
    return float(correct.item() / max(total.item(), 1.0))


# ---------------------------------------------------------------------------- schedule
def should_push(epoch: int, cfg: ProtoPNetConfig) -> bool:
    """Push after warm-up at every `push_every` epochs, and always on the final epoch."""
    if epoch < cfg.warm_epochs:
        return False
    if epoch == cfg.epochs - 1:
        return True
    return (epoch - cfg.warm_epochs) % cfg.push_every == 0


def _build_optimizers(core: ProtoPNet, cfg: ProtoPNetConfig):
    warm = torch.optim.Adam(
        [{"params": core.add_on.parameters()}, {"params": [core.prototypes]}], lr=cfg.lr
    )
    joint = torch.optim.Adam(
        [
            {"params": core.backbone.parameters(), "lr": cfg.lr * 0.1},
            {"params": core.add_on.parameters()},
            {"params": [core.prototypes]},
        ],
        lr=cfg.lr,
    )
    last = torch.optim.Adam(core.classifier.parameters(), lr=cfg.lr * 0.1)
    return warm, joint, last


def _distributed_push(core, push_loader, cfg):
    """Push under DDP: run the projection on rank 0 only (it owns the full, un-sharded push
    loader), then **broadcast** the updated prototypes to every rank so all replicas stay
    bit-identical. The push metadata is only needed on rank 0 (for visualization).

    Rationale: the push mutates ``prototypes`` *outside* the optimizer, so DDP's gradient
    sync can't keep it consistent — an explicit broadcast does. Doing the scan on one rank
    avoids redundant work; the broadcast is cheap (a handful of vectors). A fully sharded
    distributed-argmin push is a possible future optimization (see notes.md).
    """
    push_meta = [None] * core.num_prototypes
    if dist_utils.is_main_process():
        push_meta = push_prototypes(core, push_loader, cfg.device)
    dist_utils.barrier()
    dist_utils.broadcast(core.prototypes.data, src=0)   # sync prototypes to all ranks
    return push_meta


def run_training(cfg, model, train_loader, test_loader, push_loader,
                 train_sampler=None, ckpt=None) -> list:
    """Run the full alternating schedule. Returns the metadata from the final push.

    ``model`` may be a DDP wrapper; ``core`` is always the underlying ProtoPNet for the
    custom methods (``set_mode``, ``num_prototypes``, push). Only rank 0 logs and writes
    checkpoints. If ``ckpt`` is given (from :func:`checkpoint.maybe_load`), the optimizers,
    LR scheduler, RNG, epoch and push metadata are restored and training continues from the
    next epoch. A rolling checkpoint is written at the end of every epoch.
    """
    core = dist_utils.unwrap_model(model)
    main = dist_utils.is_main_process()
    warm_opt, joint_opt, last_opt = _build_optimizers(core, cfg)
    # Joint-phase LR schedule (paper uses StepLR on the joint optimizer). Saved/restored so
    # decay continues correctly across a resume.
    joint_sched = torch.optim.lr_scheduler.StepLR(
        joint_opt, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    optimizers = {"warm": warm_opt, "joint": joint_opt, "last": last_opt}
    push_meta: list = [None] * core.num_prototypes

    start_epoch = 0
    if ckpt is not None:
        start_epoch, restored_meta = checkpoint.restore_train_state(
            optimizers, joint_sched, ckpt, cfg)
        if restored_meta is not None:
            push_meta = restored_meta
    if start_epoch >= cfg.epochs:
        if main:
            print(f"[resume] checkpoint already trained through epoch {start_epoch - 1} "
                  f">= {cfg.epochs - 1}; nothing to train.")
        return push_meta

    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)          # reshuffle shards each epoch (DDP)

        if epoch < cfg.warm_epochs:
            core.set_mode("warm")
            stats = train_one_epoch(model, train_loader, warm_opt, cfg)
            phase = "warm"
        else:
            core.set_mode("joint")
            stats = train_one_epoch(model, train_loader, joint_opt, cfg)
            joint_sched.step()                      # decay joint LR per the schedule
            phase = "joint"

        acc = evaluate(model, test_loader, cfg)
        if main:
            print(f"[epoch {epoch:02d}] {phase:5s} loss={stats['loss']:.3f} "
                  f"ce={stats['ce']:.3f} clst={stats['cluster']:.3f} sep={stats['sep']:.3f} "
                  f"lr={joint_opt.param_groups[0]['lr']:.1e} | test_acc={acc:.3f}")

        if should_push(epoch, cfg):
            push_meta = _distributed_push(core, push_loader, cfg)
            acc_after = evaluate(model, test_loader, cfg)
            if main:
                print(f"           push -> test_acc={acc_after:.3f}; optimizing last layer...")
            core.set_mode("last")
            for _ in range(cfg.last_layer_iters):
                train_one_epoch(model, train_loader, last_opt, cfg)
            if main:
                print(f"           last-layer done -> test_acc={evaluate(model, test_loader, cfg):.3f}")

        # Rolling checkpoint at end of epoch (rank 0; atomic) — enables --resume / chaining.
        checkpoint.save_if_main(cfg, core, optimizers, joint_sched, epoch, push_meta)

    return push_meta


# ---------------------------------------------------------------------------- data
def build_dataloaders(cfg: ProtoPNetConfig, ctx: dist_utils.DistContext | None = None):
    """CUB train/test loaders plus a clean (un-shuffled, un-augmented) push loader.

    Under DDP the train/test sets are sharded with ``DistributedSampler`` (the returned
    ``train_sampler`` gets ``set_epoch`` each epoch). The **push loader is never sharded** —
    the push runs on rank 0 over the full set (see ``_distributed_push``), and its index
    order matches ``push_set`` so push metadata maps back to real images for visualization.
    """
    distributed = ctx is not None and ctx.distributed

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

    if distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        test_sampler = DistributedSampler(test_set, shuffle=False)
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=train_sampler,
                                  num_workers=cfg.num_workers)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, sampler=test_sampler,
                                 num_workers=cfg.num_workers)
    else:
        train_sampler = None
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers)
    push_loader = DataLoader(push_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    return train_loader, test_loader, push_loader, test_set, push_set, train_sampler


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

    # Single-process by default; engages DDP only under torchrun (WORLD_SIZE > 1).
    ctx = dist_utils.init_distributed(prefer_device=cfg.device)
    cfg.device = ctx.device
    torch.manual_seed(cfg.seed)                     # DDP also broadcasts params at wrap time

    train_loader, test_loader, push_loader, test_set, push_set, train_sampler = \
        build_dataloaders(cfg, ctx)

    model = ProtoPNet(cfg).to(cfg.device)

    # Resume: load + validate the checkpoint (every rank), then restore MODEL weights BEFORE
    # SyncBN/DDP wrap so DDP's construction-time broadcast carries the resumed weights. The
    # optimizer / scheduler / epoch / RNG are restored later inside run_training.
    ckpt = checkpoint.maybe_load(cfg)
    if ckpt is not None:
        checkpoint.load_model_state(dist_utils.unwrap_model(model), ckpt)

    if ctx.distributed and cfg.sync_batchnorm and ctx.device.startswith("cuda"):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)   # cross-GPU batch stats in joint phase
    model = dist_utils.wrap_ddp(model, ctx)

    if dist_utils.is_main_process():
        core = dist_utils.unwrap_model(model)
        print(f"ProtoPNet: {core.num_prototypes} prototypes "
              f"({cfg.prototypes_per_class}/class × {cfg.num_classes} classes), "
              f"device={cfg.device}, world_size={ctx.world_size}")

    push_meta = run_training(cfg, model, train_loader, test_loader, push_loader,
                             train_sampler, ckpt=ckpt)

    if dist_utils.is_main_process():                # visualize once, on rank 0
        visualize_prototypes(cfg, dist_utils.unwrap_model(model), test_set, push_set, push_meta)
        print("done.")

    dist_utils.cleanup()


def _cli_config() -> ProtoPNetConfig:
    """Build a config from CLI flags. Only the commonly-overridden knobs are exposed;
    everything else stays at the `config.py` defaults. Same argv on every torchrun rank.
    """
    import argparse

    p = argparse.ArgumentParser(description="Train the ProtoPNet POC on (a subset of) CUB.")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="Resume training. Bare --resume auto-detects <out_dir>/<ckpt_name>; "
                        "or pass an explicit checkpoint path.")
    p.add_argument("--out-dir", default=None, help="Override cfg.out_dir (checkpoints + figures).")
    p.add_argument("--data-root", default=None, help="CUB root (else CUB_ROOT env var).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    args = p.parse_args()

    cfg = ProtoPNetConfig()
    if args.resume is not None:
        cfg.resume = args.resume
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.num_classes is not None:
        cfg.num_classes = args.num_classes
    return cfg


if __name__ == "__main__":
    main(_cli_config())
