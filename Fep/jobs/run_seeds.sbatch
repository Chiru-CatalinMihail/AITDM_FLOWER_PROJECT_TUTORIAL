#!/usr/bin/env bash
#SBATCH --job-name=run_seeds
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
# ---------------------------------------------------------------------------
# One experiment with seeds 1, 2, 3 (mean +- std), as experiments/run_seeds.py.
#   bash Fep/submit.sh run_seeds --name fedavg --partitions data/partitions/dirichlet \
#       --config "num-server-rounds=10"
# Results: outputs/experiments/<name>/. Finished seeds are skipped, so a job
# that hit its time limit can simply be submitted again.
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh run_seeds.py "$@"
python experiments/run_seeds.py "$@"
