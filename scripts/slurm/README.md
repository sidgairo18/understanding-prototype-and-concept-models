# SLURM launchers

Submit POC training to the cluster. Single-node jobs; multi-GPU runs go through
`torchrun` (DDP engages automatically — see [`common/distributed.py`](../../common/distributed.py)).

## Files

| File | Purpose |
|---|---|
| `train_poc.sh` | Front door. `train_poc.sh <poc>` resolves the module + job name and submits. |
| `_sbatch_submit.sh` | Reusable submitter: builds the `#SBATCH` header, activates the `proto-concept` conda env, and runs `python -m` / `torchrun`. |
| `bad_nodes.txt` | *(optional, create if needed)* one node name per line to exclude. |

## Quick start

```bash
# From the repo root, after activating any shell with sbatch available:
CUB_ROOT=/path/to/CUB_200_2011 bash scripts/slurm/train_poc.sh protopnet
```

POC names: `protopnet` (implemented), `pipnet`, `scops`, `pdiscoformer` (scaffolded —
will run once their POC step lands).

## Partitions

Default partition list (job runs on whichever is free first):

```
gpu16,gpu17,gpu20,gpu22,gpu24
```

Override with `PART`, e.g. a single partition or a different mix:

```bash
PART=gpu24 CUB_ROOT=/data/CUB_200_2011 bash scripts/slurm/train_poc.sh protopnet
PART=gpu17,gpu20 ... bash scripts/slurm/train_poc.sh protopnet
```

## Knobs (env vars)

| Var | Default | Meaning |
|---|---|---|
| `CUB_ROOT` | *(required)* | Path to the extracted `CUB_200_2011` dataset. |
| `PART` | `gpu16,gpu17,gpu20,gpu22,gpu24` | Partition(s) to request. |
| `GPUS` | `1` | GPUs on the node. `>1` → `torchrun --nproc_per_node=$GPUS` (DDP). |
| `CPUS` | `8` | `--cpus-per-task`. |
| `MEM` | `48GB` | `--mem`. |
| `TIME_LIMIT` | `08:00:00` | Walltime. |
| `JN` | `<poc>` | Job name (also the log file prefix). |
| `ENV_NAME` | `proto-concept` | Conda env to activate. |
| `CONDA_BASE` | miniforge that holds the env | Conda install to source. |
| `EXCLUDE` | *(unset)* | Comma-separated nodes to exclude; falls back to `bad_nodes.txt`. |
| `EXTRA` | *(empty)* | Extra args appended to the launch command. |
| `CHAIN_JOBS` | `1` | Number of sequential job slots (job chaining). `>1` → SLURM array `-a 1-N%1`. |
| `DEPENDENCY` | *(unset)* | Start only after an existing job, e.g. `afterany:123456`. |

Examples:

```bash
# 4-GPU DDP run on gpu24, 24h:
GPUS=4 PART=gpu24 TIME_LIMIT=24:00:00 CUB_ROOT=/data/CUB_200_2011 \
    bash scripts/slurm/train_poc.sh protopnet

# Exclude a flaky node:
EXCLUDE=gpu24-01 CUB_ROOT=/data/CUB_200_2011 bash scripts/slurm/train_poc.sh protopnet
```

## Job chaining

A long run can outlive a single walltime window by chaining several job slots that run
back-to-back. Set `CHAIN_JOBS=N` to submit a SLURM array `-a 1-N%1` — N tasks throttled
to **one at a time**, so slot 2 starts when slot 1 ends (walltime, finish, or crash):

```bash
# 4 sequential 8h slots (32h of wall budget) on the default partitions:
CHAIN_JOBS=4 CUB_ROOT=/data/CUB_200_2011 bash scripts/slurm/train_poc.sh protopnet
```

You can also chain onto an *existing* job with `DEPENDENCY` (any sbatch dependency spec):

```bash
DEPENDENCY=afterany:123456 CUB_ROOT=/data/CUB_200_2011 \
    bash scripts/slurm/train_poc.sh protopnet
```

Each slot writes its own log: `runs/slurm_logs/<JN>-<arrayid>_<slot>.out`. Cancel the
whole chain with `scancel <arrayid>`.

**Resume is wired (ProtoPNet).** For a chained run the launcher auto-appends `--resume`, so
each slot continues from the previous one's checkpoint instead of restarting. ProtoPNet
writes a rolling, atomically-written `ckpt_last.pt` (under `out_dir`) at the end of every
epoch — capturing the model, all three optimizers (warm/joint/last), the joint LR schedule,
the epoch, the latest push metadata, and RNG state — and resumes from it on the next slot.
The first slot starts fresh (no checkpoint yet). All chained slots share the same `out_dir`,
so keep `out_dir` fixed across the chain (the default is per-POC, so this is automatic).

To get the same crash-recovery on a **single** (non-chained) job, add `EXTRA="--resume"`.
The other POCs (pipnet/scops/pdiscoformer) gain chaining once they implement the same
`--resume` hook.

## Logs

Stdout/stderr → `runs/slurm_logs/<JN>-<jobid>.out` (single job) or
`runs/slurm_logs/<JN>-<arrayid>_<slot>.out` (chained). The submitter prints the exact
`squeue` / `tail -f` commands after submission.

## Note on hyperparameters

These launchers control **compute** (partition, GPUs, time). The **model**
hyperparameters (classes, epochs, prototypes, batch size) live in each POC's
`config.py`, which defaults to a *tiny* fast run. To scale to full CUB-200, edit the
POC's `config.py` (e.g. `num_classes=200`, more `epochs`) — or ask for an env/CLI
override to be wired into `train.py` so it can be set per-submission.
