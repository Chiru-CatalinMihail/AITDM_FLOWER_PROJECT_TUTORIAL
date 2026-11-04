#!/usr/bin/env bash
#SBATCH --job-name=simulation
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
# ---------------------------------------------------------------------------
# One federated run (server + all clients) with Flower's Simulation Runtime:
# the same as scripts/run_simulation.sh on your laptop, same arguments.
#   bash Fep/submit.sh simulation data/partitions/dirichlet
#   bash Fep/submit.sh simulation data/partitions/dirichlet "num-server-rounds=10 strategy='fedprox'"
#   bash Fep/submit.sh -p ml --gres=gpu:1 simulation data/partitions/dirichlet
# Results: outputs/sim-<partition-name>-<timestamp>/ (i.e. ~/AITDM_Project/outputs).
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh run_simulation.sh "$@"
bash scripts/run_simulation.sh "$@"
