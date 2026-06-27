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

Examples:

```bash
# 4-GPU DDP run on gpu24, 24h:
GPUS=4 PART=gpu24 TIME_LIMIT=24:00:00 CUB_ROOT=/data/CUB_200_2011 \
    bash scripts/slurm/train_poc.sh protopnet

# Exclude a flaky node:
EXCLUDE=gpu24-01 CUB_ROOT=/data/CUB_200_2011 bash scripts/slurm/train_poc.sh protopnet
```

## Logs

Stdout/stderr → `runs/slurm_logs/<JN>-<jobid>.out`. The submitter prints the exact
`squeue` / `tail -f` commands after submission.

## Note on hyperparameters

These launchers control **compute** (partition, GPUs, time). The **model**
hyperparameters (classes, epochs, prototypes, batch size) live in each POC's
`config.py`, which defaults to a *tiny* fast run. To scale to full CUB-200, edit the
POC's `config.py` (e.g. `num_classes=200`, more `epochs`) — or ask for an env/CLI
override to be wired into `train.py` so it can be set per-submission.
