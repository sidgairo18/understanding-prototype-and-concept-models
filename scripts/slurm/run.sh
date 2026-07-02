#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# General-purpose SLURM launcher for the repo: pick a POC + a named preset config
# and submit. Presets bundle the *training* args (classes/epochs/batch/out-dir/wandb)
# and sensible *compute* defaults; every knob stays overridable via env.
#
# Usage:
#   bash scripts/slurm/run.sh <poc> [preset]
#   ssh slurm-submit "cd <repo> && bash scripts/slurm/run.sh protopnet cub_small"
#
#   <poc>    : protopnet | pipnet | scops | pdiscoformer
#   [preset] : named config (default: cub_small). See the case block below.
#
# Compute defaults (override via env): PART=gpu22, GPUS=4 (DDP via torchrun),
#   TIME_LIMIT=02:59:00, CUB_ROOT=<your CUB path>. Per-preset: CHAIN_JOBS, batch size.
#   Any of PART/GPUS/TIME_LIMIT/CHAIN_JOBS/CUB_ROOT/JN can be overridden on the CLI env.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

POC="${1:-protopnet}"
PRESET="${2:-cub_small}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- shared compute defaults (env-overridable) ---
export CUB_ROOT="${CUB_ROOT:-/BS/generative_modelling_for_image_understanding_2/nobackup/data/CUB_200_2011}"
export PART="${PART:-gpu22}"          # gpu22 nodes have 8 GPUs (a40/a100); we take GPUS of them
export GPUS="${GPUS:-4}"             # >1 -> torchrun DDP (single node)
export TIME_LIMIT="${TIME_LIMIT:-02:59:00}"   # < 03:00:00 per cluster policy

WANDB_ARGS="--wandb --wandb-project understanding-protos"

case "${POC}:${PRESET}" in
  # --- ProtoPNet ---
  protopnet:smoke)       # smallest: pipeline sanity check
    export JN="${JN:-protopnet-smoke}"; export CHAIN_JOBS="${CHAIN_JOBS:-1}"
    export EXTRA="--num-classes 5 --epochs 5 --batch-size 16 \
--out-dir prototype_methods/protopnet/runs/smoke ${WANDB_ARGS} --wandb-run-name ${JN}" ;;
  protopnet:cub_small)   # small subset -> signature figure on real birds
    export JN="${JN:-protopnet-cub-small}"; export CHAIN_JOBS="${CHAIN_JOBS:-1}"
    export EXTRA="--num-classes 10 --epochs 20 --batch-size 32 \
--out-dir prototype_methods/protopnet/runs/cub_small ${WANDB_ARGS} --wandb-run-name ${JN}" ;;
  protopnet:cub_full)    # full CUB-200 baseline (chained across walltime slots)
    export JN="${JN:-protopnet-cub-full}"; export CHAIN_JOBS="${CHAIN_JOBS:-2}"
    export CPUS="${CPUS:-16}"; export MEM="${MEM:-64GB}"
    export EXTRA="--num-classes 200 --epochs 100 --batch-size 20 --lr-step-size 30 \
--out-dir prototype_methods/protopnet/runs/cub_full ${WANDB_ARGS} --wandb-run-name ${JN}" ;;
  protopnet:cub_full_paper)   # paper-faithful: bbox-crop + strong ONLINE aug + push-every-10
    export JN="${JN:-protopnet-cub-full-paper}"; export CHAIN_JOBS="${CHAIN_JOBS:-3}"
    export CPUS="${CPUS:-16}"; export MEM="${MEM:-64GB}"
    export EXTRA="--num-classes 200 --epochs 120 --batch-size 20 \
--warm-epochs 10 --push-every 10 --lr-step-size 30 --crop-bbox --strong-aug \
--out-dir prototype_methods/protopnet/runs/cub_full_paper ${WANDB_ARGS} --wandb-run-name ${JN}" ;;
  protopnet:cub_full_repro)   # paper repro: OFFLINE ~15x augmented (cropped) train set + longer schedule
    export JN="${JN:-protopnet-cub-full-repro}"; export CHAIN_JOBS="${CHAIN_JOBS:-3}"
    export CPUS="${CPUS:-16}"; export MEM="${MEM:-64GB}"
    AUG_DIR="${AUG_DIR:-/BS/generative_modelling_for_image_understanding_2/nobackup/data/cub200_crop_aug_train}"
    export EXTRA="--num-classes 200 --epochs 40 --batch-size 20 \
--warm-epochs 2 --push-every 5 --lr-step-size 15 --crop-bbox --aug-train-dir ${AUG_DIR} \
--out-dir prototype_methods/protopnet/runs/cub_full_repro ${WANDB_ARGS} --wandb-run-name ${JN}" ;;

  *)
    echo "Unknown POC:preset '${POC}:${PRESET}'." >&2
    echo "Available presets: protopnet:{smoke,cub_small,cub_full,cub_full_paper,cub_full_repro}" >&2
    echo "(pipnet/scops/pdiscoformer presets: add here once those POCs are implemented.)" >&2
    exit 1 ;;
esac

echo "== run.sh: ${POC} / ${PRESET} =="
echo "   PART=${PART} GPUS=${GPUS} TIME_LIMIT=${TIME_LIMIT} CHAIN_JOBS=${CHAIN_JOBS} JN=${JN}"
echo "   CUB_ROOT=${CUB_ROOT}"
echo "   EXTRA=${EXTRA}"
bash "${HERE}/train_poc.sh" "${POC}"
