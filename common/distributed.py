"""Minimal Distributed Data Parallel (DDP) helpers, shared by every POC.

Design goal: **the single-process path is unchanged.** If a script is launched normally
(no ``torchrun``), :func:`init_distributed` reports a non-distributed context and every
helper here degrades to a rank-0 / no-op path — so the tiny CPU/GPU default still runs
exactly as before. DDP only engages when launched under ``torchrun``, which sets the
``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` environment variables, e.g.::

    torchrun --nproc_per_node=4 -m prototype_methods.protopnet.train

These helpers are intentionally tiny and framework-faithful so the *training code* stays
legible: the POC train loops call ``is_main_process()``, ``all_reduce_sum()``,
``unwrap_model()`` etc. rather than sprinkling ``torch.distributed`` calls everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistContext:
    """Describes this process's role. ``distributed=False`` for a normal single run."""

    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: str


def init_distributed(prefer_device: str = "cuda") -> DistContext:
    """Initialize the process group iff launched under ``torchrun`` (``WORLD_SIZE > 1``).

    Picks the NCCL backend on CUDA and Gloo on CPU, so the *same* code runs (and can be
    tested) on multi-GPU or on CPU. Resolves the per-process device with a CPU fallback.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        # Single process: no DDP. Resolve the device with the usual CPU fallback.
        use_cuda = prefer_device.startswith("cuda") and torch.cuda.is_available()
        return DistContext(False, 0, 1, 0, prefer_device if use_cuda else "cpu")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    use_cuda = prefer_device.startswith("cuda") and torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if use_cuda else "gloo",
                            rank=rank, world_size=world_size)
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return DistContext(True, rank, world_size, local_rank, device)


def wrap_ddp(model: torch.nn.Module, ctx: DistContext) -> torch.nn.Module:
    """Wrap ``model`` in DDP when distributed, else return it unchanged.

    ``find_unused_parameters=True`` because ProtoPNet-style schedules freeze different
    parameter subsets per phase (warm / joint / last); it makes the reducer robust to the
    trainable set changing between iterations. Harmless (small) overhead for the legibility.
    """
    if not ctx.distributed:
        return model
    kwargs = {
        "find_unused_parameters": True,
        # We manage BatchNorm explicitly: DDP broadcasts buffers once at construction (so all
        # ranks start with identical pretrained running stats), and a frozen backbone keeps
        # them in eval so they never drift. So we don't need DDP to re-broadcast buffers every
        # forward — it only adds per-step comms (and trips a gloo coalesced-broadcast bug). For
        # *synchronized* trainable BN across GPUs, use cfg.sync_batchnorm (SyncBatchNorm), which
        # is the correct tool rather than broadcast_buffers' cruder rank-0-wins broadcast.
        "broadcast_buffers": False,
    }
    if ctx.device.startswith("cuda"):
        kwargs["device_ids"] = [ctx.local_rank]
        kwargs["output_device"] = ctx.local_rank
    return DDP(model, **kwargs)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, unwrapping DDP if present.

    Use this whenever you need a *custom* method/attribute (``set_mode``, ``num_prototypes``,
    ``cluster_and_separation_costs`` …): DDP does not proxy arbitrary attribute access.
    """
    return model.module if isinstance(model, DDP) else model


# --------------------------------------------------------------------- collective helpers
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    """True on rank 0 (and always true when not distributed). Gate prints / saves on this."""
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce across ranks (no-op when not distributed)."""
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """In-place broadcast of ``tensor`` from ``src`` to all ranks (no-op when not dist)."""
    if is_dist():
        dist.broadcast(tensor, src=src)
    return tensor


def cleanup() -> None:
    if is_dist():
        dist.destroy_process_group()
