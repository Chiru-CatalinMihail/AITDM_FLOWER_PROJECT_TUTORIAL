#!/usr/bin/env bash
#SBATCH --job-name=command
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# ---------------------------------------------------------------------------
# Any other command of the tutorials, run on a compute node with .venv active.
#   bash Fep/submit.sh command python explainability/gradcam.py --run outputs/<run>
#   bash Fep/submit.sh command python data_splitting/split_data.py --dataset dermamnist \
#       --num-clients 5 --strategy dirichlet --alpha 0.3
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh "$@"
"$@"
