#!/usr/bin/env bash
#SBATCH --job-name=centralised
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
# ---------------------------------------------------------------------------
# Centralised baseline (all clients' data together), as experiments/centralised.py.
#   bash Fep/submit.sh centralised --partitions data/partitions/dirichlet --epochs 10
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh centralised.py "$@"
python experiments/centralised.py "$@"
