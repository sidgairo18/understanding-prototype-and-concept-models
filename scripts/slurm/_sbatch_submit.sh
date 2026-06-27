#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Reusable SLURM submitter for the prototype / part-prototype POCs.
#
# Submits a single-node training job. For >1 GPU it launches via `torchrun`
# (DDP — the POC train loops engage DDP automatically under torchrun, see
# common/distributed.py); for 1 GPU it runs plain `python -m`.
#
# Callers (e.g. train_poc.sh) set MODULE and JN, then `source` this file. You can
# also use it directly:
#   MODULE=prototype_methods.protopnet.train CUB_ROOT=/data/CUB_200_2011 \
#     bash scripts/slurm/_sbatch_submit.sh
#
# Required:
#   MODULE      python module to run, e.g. prototype_methods.protopnet.train
#   CUB_ROOT    path to the extracted CUB_200_2011 dataset
#
# Knobs (env-overridable; defaults in []):
#   JN          job name                                   [poc-train]
#   PART        partition list (comma = any of)            [gpu16,gpu17,gpu20,gpu22,gpu24]
#   GPUS        GPUs on the node (>1 -> torchrun DDP)       [1]
#   CPUS        cpus-per-task                               [8]
#   MEM         memory                                      [48GB]
#   TIME_LIMIT  walltime HH:MM:SS                           [08:00:00]
#   ENV_NAME    conda env to activate                       [proto-concept]
#   CONDA_BASE  conda install that holds ENV_NAME           [miniforge3 used to create it]
#   EXCLUDE     nodes to exclude (csv); else bad_nodes.txt if present
#   EXTRA       extra args appended to the launch command
#   CHAIN_JOBS  number of sequential slots (job chaining)   [1]
#   DEPENDENCY  start only after an existing job, e.g.      [unset]
#               afterany:123456 (chain onto a prior run)
#
# Job chaining (CHAIN_JOBS > 1): submits a SLURM array `-a 1-N%1`, i.e. N tasks that
# run strictly one-at-a-time. When a slot hits the walltime it is killed and the next
# slot starts, so a long run survives short walltime windows. CAVEAT: each slot just
# re-runs the training command, so it only *continues* (rather than restarts) if
# train.py resumes from a checkpoint — NOT wired yet (see README "Job chaining").
# Default CHAIN_JOBS=1 → a single ordinary job (no behavior change).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${MODULE:?MODULE not set (e.g. prototype_methods.protopnet.train)}"

# Repo root = two levels up from this script (scripts/slurm/_sbatch_submit.sh).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"

JN="${JN:-poc-train}"
PART="${PART:-gpu16,gpu17,gpu20,gpu22,gpu24}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-48GB}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
ENV_NAME="${ENV_NAME:-proto-concept}"
CONDA_BASE="${CONDA_BASE:-/BS/dnn_interpretablity_robustness_representation_learning_2/work/libs/miniforge3}"
EXTRA="${EXTRA:-}"
CHAIN_JOBS="${CHAIN_JOBS:-1}"
DEPENDENCY="${DEPENDENCY:-}"
CUB_ROOT="${CUB_ROOT:?set CUB_ROOT=/path/to/CUB_200_2011 before submitting}"

LOGDIR="${REPO}/runs/slurm_logs"
mkdir -p "${LOGDIR}"

# Optional node exclusion: explicit EXCLUDE wins; else fall back to bad_nodes.txt.
BAD_NODES_FILE="${HERE}/bad_nodes.txt"
if [[ -z "${EXCLUDE:-}" && -f "${BAD_NODES_FILE}" ]]; then
    EXCLUDE="$(grep -Ev '^\s*(#|$)' "${BAD_NODES_FILE}" | tr '\n' ',' | sed 's/,$//')"
fi
EXCLUDE_LINE=""
[[ -n "${EXCLUDE:-}" ]] && EXCLUDE_LINE="#SBATCH --exclude=${EXCLUDE}"

# Job chaining: `-a 1-N%1` is an array of N tasks throttled to one running at a time,
# so the slots run back-to-back (a chain). %j (single job) -> %A_%a (array) for logs so
# each slot gets its own file. DEPENDENCY chains this whole run after an external job.
ARRAY_LINE=""
LOG_PATTERN="%j"
if [[ "${CHAIN_JOBS}" -gt 1 ]]; then
    ARRAY_LINE="#SBATCH -a 1-${CHAIN_JOBS}%1"
    LOG_PATTERN="%A_%a"
fi
DEP_LINE=""
[[ -n "${DEPENDENCY}" ]] && DEP_LINE="#SBATCH --dependency=${DEPENDENCY}"

# Launch command: torchrun (single-node, standalone rendezvous) for multi-GPU DDP,
# else plain python for one GPU.
if [[ "${GPUS}" -gt 1 ]]; then
    LAUNCH="torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS} -m ${MODULE} ${EXTRA}"
else
    LAUNCH="python -u -m ${MODULE} ${EXTRA}"
fi

echo "Submitting '${JN}': module=${MODULE} part=${PART} gpus=${GPUS} time=${TIME_LIMIT} mem=${MEM} chain=${CHAIN_JOBS}"
[[ -n "${EXCLUDE:-}" ]] && echo "  excluding nodes: ${EXCLUDE}"
[[ -n "${DEPENDENCY}" ]] && echo "  dependency: ${DEPENDENCY}"

JOB_ID=$(sbatch --parsable << EOT
#!/bin/bash
#SBATCH --job-name=${JN}
#SBATCH --partition=${PART}
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPUS}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM}
#SBATCH -o ${LOGDIR}/${JN}-${LOG_PATTERN}.out
${EXCLUDE_LINE}
${ARRAY_LINE}
${DEP_LINE}

echo "Start: \$(date)  Node: \$(hostname)  Job: \${SLURM_JOB_ID}  Slot: \${SLURM_ARRAY_TASK_ID:-n/a}/\${SLURM_ARRAY_TASK_COUNT:-1}  Partition: \${SLURM_JOB_PARTITION}"
nvidia-smi -L || true

# --- activate conda env (use the install that holds ${ENV_NAME}) ---
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# --- runtime env ---
export CUB_ROOT="${CUB_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}

cd ${REPO}
echo "Running: ${LAUNCH}"
${LAUNCH}
echo "Done: \$(date)"
EOT
)
if [[ "${CHAIN_JOBS}" -gt 1 ]]; then
    echo "Submitted chained array ${JOB_ID} (${CHAIN_JOBS} slots, 1 at a time)"
    echo "Logs:   ${LOGDIR}/${JN}-${JOB_ID}_*.out   (one per slot)"
    echo "Watch:  squeue -j ${JOB_ID}    Cancel all:  scancel ${JOB_ID}"
else
    echo "Submitted job ${JOB_ID}  ->  ${LOGDIR}/${JN}-${JOB_ID}.out"
    echo "Watch:  squeue -j ${JOB_ID}    Tail:  tail -f ${LOGDIR}/${JN}-${JOB_ID}.out"
fi
