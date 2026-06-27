"""Checkpoint save / resume for the ProtoPNet POC.

A run keeps a single *rolling* checkpoint (``ckpt_last.pt`` under ``out_dir``), rewritten at
the end of every epoch. It is written **atomically** (temp file + ``os.replace``) so a
walltime kill — e.g. when a chained SLURM slot is pre-empted mid-write — can never leave a
half-written, unloadable checkpoint behind.

The checkpoint captures *everything* needed to continue the alternating schedule faithfully:

  * **model**      — backbone + add-on + prototypes + classifier + buffers (``state_dict``);
  * **optimizers** — all three Adam optimizers (warm / joint / last), each with its momentum;
  * **scheduler**  — the joint-phase ``StepLR`` (so LR decay resumes at the right step);
  * **epoch**      — last completed epoch (resume continues at ``epoch + 1``);
  * **push_meta**  — the latest prototype-projection metadata (for the final visualization);
  * **rng**        — torch (and CUDA) RNG state, so augmentation/shuffling continue stream-faithfully.

Resume **validates** that the checkpoint's architecture matches the current config before
loading anything, so a stale/mismatched checkpoint fails loudly instead of silently
corrupting a run.

Under DDP, only rank 0 writes; every rank loads (each restores its own replica/device).
"""

from __future__ import annotations

import os

import torch

from common import distributed as dist_utils

# Bump if the checkpoint dict layout changes incompatibly.
CKPT_FORMAT = 1

# Config fields that define parameter *shapes* — these MUST match to resume.
_ARCH_KEYS = ("num_classes", "prototypes_per_class", "prototype_dim", "backbone")
# Fields worth a warning if they differ, but which don't change parameter shapes.
_WARN_KEYS = ("img_size", "images_per_class")


# --------------------------------------------------------------------------- paths / state
def latest_path(cfg) -> str:
    """Path of the rolling 'latest' checkpoint for this run."""
    return os.path.join(cfg.out_dir, cfg.ckpt_name)


def best_path(cfg) -> str:
    """Path of the 'best test accuracy so far' checkpoint for this run."""
    return os.path.join(cfg.out_dir, cfg.best_ckpt_name)


def _arch_snapshot(cfg) -> dict:
    return {k: getattr(cfg, k) for k in (_ARCH_KEYS + _WARN_KEYS)}


def build_state(core, optimizers: dict, scheduler, epoch: int, push_meta, cfg,
                best_acc: float, acc: float) -> dict:
    """Assemble the full checkpoint dict from the live training objects."""
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "format": CKPT_FORMAT,
        "epoch": epoch,
        "acc": acc,                 # this epoch's end-of-epoch test accuracy
        "best_acc": best_acc,       # running best test accuracy (for best-ckpt tracking)
        "model": core.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "scheduler": scheduler.state_dict(),
        "push_meta": push_meta,
        "rng": {"torch": torch.get_rng_state(), "cuda": cuda_rng},
        "config": _arch_snapshot(cfg),
    }


def atomic_save(state: dict, path: str) -> None:
    """Write ``state`` to ``path`` atomically (temp file + rename) so it is never half-written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def save_if_main(cfg, core, optimizers: dict, scheduler, epoch: int, push_meta,
                 best_acc: float, acc: float, is_best: bool) -> None:
    """On rank 0: write the rolling checkpoint, and (if this is the best so far) also the
    best checkpoint — built once, written to both paths. Barrier so ranks stay in lockstep.
    """
    if dist_utils.is_main_process():
        state = build_state(core, optimizers, scheduler, epoch, push_meta, cfg, best_acc, acc)
        atomic_save(state, latest_path(cfg))
        if is_best:
            atomic_save(state, best_path(cfg))
    dist_utils.barrier()


# --------------------------------------------------------------------------- load / restore
def _check_compatible(ckpt: dict, cfg) -> None:
    """Raise if the checkpoint's architecture-defining config differs from ``cfg``."""
    saved = ckpt.get("config", {})
    mismatched = {k: (saved.get(k), getattr(cfg, k))
                  for k in _ARCH_KEYS if saved.get(k) != getattr(cfg, k)}
    if mismatched:
        diffs = ", ".join(f"{k}: ckpt={v[0]!r} vs cfg={v[1]!r}" for k, v in mismatched.items())
        raise ValueError(
            f"Refusing to resume: checkpoint architecture does not match config ({diffs}). "
            "Use a matching config, or start fresh (drop --resume / delete the checkpoint).")
    if dist_utils.is_main_process():
        for k in _WARN_KEYS:
            if saved.get(k) != getattr(cfg, k):
                print(f"[resume][warn] {k} differs (ckpt={saved.get(k)!r} vs cfg={getattr(cfg, k)!r}); "
                      "continuing — it does not affect parameter shapes.")


def maybe_load(cfg):
    """Resolve ``cfg.resume`` and load+validate a checkpoint, or return ``None`` (start fresh).

    ``cfg.resume``:
      * ``None``   — no resume (returns None);
      * ``"auto"`` — load ``<out_dir>/<ckpt_name>`` if it exists, else start fresh (chaining-friendly);
      * a path     — load that explicit checkpoint (error if missing).
    """
    spec = getattr(cfg, "resume", None)
    if not spec:
        return None

    if spec == "auto":
        path = latest_path(cfg)
        if not os.path.isfile(path):
            if dist_utils.is_main_process():
                print(f"[resume] no checkpoint at {path}; starting fresh.")
            return None
    else:
        path = spec
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--resume checkpoint not found: {path}")

    # weights_only=False: our checkpoint holds non-tensor objects (config, push_meta, RNG).
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    if ckpt.get("format") != CKPT_FORMAT:
        raise ValueError(
            f"Unsupported checkpoint format {ckpt.get('format')!r} (expected {CKPT_FORMAT}).")
    _check_compatible(ckpt, cfg)
    if dist_utils.is_main_process():
        print(f"[resume] loaded {path} (trained through epoch {ckpt['epoch']}); "
              f"continuing at epoch {ckpt['epoch'] + 1}.")
    return ckpt


def load_model_state(core, ckpt) -> None:
    """Strictly load model weights + buffers; surface any key mismatch clearly."""
    incompatible = core.load_state_dict(ckpt["model"], strict=False)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint model keys do not match the model: missing={missing}, "
            f"unexpected={unexpected}.")


def restore_train_state(optimizers: dict, scheduler, ckpt, cfg) -> tuple[int, list | None, float]:
    """Restore the three optimizers, the scheduler, and RNG.

    Returns ``(start_epoch, push_meta, best_acc)``. ``best_acc`` lets best-checkpoint tracking
    continue across a resume so a worse post-resume epoch never overwrites the best checkpoint.
    """
    saved_opts = ckpt["optimizers"]
    for name, opt in optimizers.items():
        if name not in saved_opts:
            raise KeyError(f"Checkpoint is missing optimizer state for '{name}'.")
        opt.load_state_dict(saved_opts[name])
    scheduler.load_state_dict(ckpt["scheduler"])
    _restore_rng(ckpt.get("rng"))
    return ckpt["epoch"] + 1, ckpt.get("push_meta"), ckpt.get("best_acc", -1.0)


def _restore_rng(rng) -> None:
    """Restore torch (and CUDA) RNG state; skip CUDA restore if the GPU count changed."""
    if not rng:
        return
    torch.set_rng_state(rng["torch"].cpu())  # set_rng_state wants a CPU ByteTensor
    cuda = rng.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        cuda = [c.cpu() for c in cuda]
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)
        elif dist_utils.is_main_process():
            print(f"[resume][warn] checkpoint has {len(cuda)} CUDA RNG state(s) but "
                  f"{torch.cuda.device_count()} GPU(s) present; skipping CUDA RNG restore.")
