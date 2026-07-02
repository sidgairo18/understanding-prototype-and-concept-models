"""Shared Weights & Biases logging, reused by every POC (training and eval).

Design goals (mirror `common/distributed.py`): **the default path is a no-op.** If wandb is
disabled (default), not installed, or this is not rank 0, every method is a cheap no-op — so
tiny/CPU/smoke runs need neither the dependency nor a login. wandb only activates when a
config asks for it (``cfg.wandb=True`` / ``--wandb``) on the main process.

Usage in a POC ``train.py``::

    from common.wandb_logger import WandbLogger
    logger = WandbLogger(cfg, ctx)                 # ctx from init_distributed (rank gate)
    ...
    logger.log({"train/loss": ..., "test/acc": ...}, step=epoch)
    logger.log_figure("this_looks_like_that", fig, step=epoch)   # matplotlib Figure
    logger.finish()

The logger reads its settings with ``getattr(cfg, ...)`` defaults, so a POC config that only
defines *some* of the wandb fields still works — every new POC inherits this for free.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

from common import distributed as dist_utils


def _cfg_to_dict(cfg) -> dict:
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


class WandbLogger:
    """Thin wandb wrapper that is a no-op unless enabled on the main process.

    Args:
        cfg: a POC config (dataclass) read for wandb settings and logged as the run config.
             Fields (all optional, with defaults): ``wandb`` (bool, default False),
             ``wandb_project`` (str), ``wandb_entity`` (str|None), ``wandb_run_name`` (str|None),
             ``wandb_mode`` ("online"|"offline"|"disabled").
        ctx: optional DistContext; logging is gated to rank 0 (``is_main_process()``).
        enabled: force-enable/disable, overriding ``cfg.wandb`` (handy for eval scripts).
    """

    def __init__(self, cfg=None, ctx=None, enabled: bool | None = None) -> None:
        want = getattr(cfg, "wandb", False) if enabled is None else enabled
        self.active = bool(want) and dist_utils.is_main_process()
        self.run = None
        self._wandb = None
        if not self.active:
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] requested but not installed (`pip install wandb`); disabling logging.")
            self.active = False
            return
        self._wandb = wandb
        self.run = wandb.init(
            project=getattr(cfg, "wandb_project", "understanding-protos"),
            entity=getattr(cfg, "wandb_entity", None),
            name=getattr(cfg, "wandb_run_name", None),
            mode=getattr(cfg, "wandb_mode", "online"),
            config=_cfg_to_dict(cfg),
        )

    # ---------------------------------------------------------------- logging API (no-ops if inactive)
    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.active:
            self._wandb.log(metrics, step=step)

    def log_figure(self, key: str, figure, step: int | None = None) -> None:
        """Log a matplotlib Figure (the POC's signature artifact) as a wandb image."""
        if self.active:
            self._wandb.log({key: self._wandb.Image(figure)}, step=step)

    def log_summary(self, summary: dict) -> None:
        """Set run-level summary values (e.g. final/best accuracy)."""
        if self.active:
            self.run.summary.update(summary)

    def finish(self) -> None:
        if self.active and self.run is not None:
            self.run.finish()
