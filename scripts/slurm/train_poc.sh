#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit a SLURM training job for one of the repo's POCs.
#
# Usage:
#   CUB_ROOT=/path/to/CUB_200_2011 bash scripts/slurm/train_poc.sh protopnet
#   GPUS=2 TIME_LIMIT=12:00:00 CUB_ROOT=/data/CUB_200_2011 \
#       bash scripts/slurm/train_poc.sh protopnet
#
# POC name -> training module. Only ProtoPNet is implemented today; the others are
# scaffolded and will run via the same launcher once their POC step lands.
#
# All submission knobs (PART, GPUS, CPUS, MEM, TIME_LIMIT, ENV_NAME, EXCLUDE, EXTRA)
# are documented in _sbatch_submit.sh and the README. Default partitions:
#   gpu16,gpu17,gpu20,gpu22,gpu24
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

POC="${1:-protopnet}"
case "${POC}" in
    protopnet)    MODULE="prototype_methods.protopnet.train" ;;
    pipnet)       MODULE="prototype_methods.pipnet.train" ;;
    scops)        MODULE="part_prototype_methods.scops.train" ;;
    pdiscoformer) MODULE="part_prototype_methods.pdiscoformer.train" ;;
    *) echo "Unknown POC '${POC}'. Choose: protopnet | pipnet | scops | pdiscoformer" >&2; exit 1 ;;
esac

export MODULE
export JN="${JN:-${POC}}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_sbatch_submit.sh"
