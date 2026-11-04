#!/usr/bin/env bash
# Sourced at the start of every job in Fep/jobs/ (never run directly).
# Slurm copies the job script elsewhere, so we find the repository from the
# working directory: Fep/submit.sh starts the job in the repository root.
set -euo pipefail
if [ ! -f scripts/_common.sh ]; then
    echo "ERROR: the job must start in the repository root." >&2
    echo "Submit it with: bash Fep/submit.sh <job> ... (or sbatch from the root)" >&2
    exit 1
fi
echo "=== job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), $(date)"
echo "    partition ${SLURM_JOB_PARTITION:-?}, ${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-?}} CPUs, GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "    data -> $(readlink -f data 2>/dev/null || echo 'data/ (not linked: run Fep/setup_fep.sh)')"
echo "    command: $*"
# shellcheck disable=SC1091
source scripts/_common.sh
