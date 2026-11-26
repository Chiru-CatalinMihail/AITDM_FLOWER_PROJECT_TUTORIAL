#!/usr/bin/env bash
#SBATCH --job-name=strategy_trajectories
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00
# ---------------------------------------------------------------------------
# Ablation of the server-weight trajectories of FedAvg / FedProx / FedAvgM /
# FedAdam / FedYogi (3 seeds, 1000 rounds x 100 local steps), all 15 runs in
# parallel (FedAvg takes 2 CPUs: it also computes the counterfactuals), then the figure.
#   bash Fep/submit.sh strategy_trajectories
#   bash Fep/submit.sh strategy_trajectories --rounds 500 --local-steps 200
# Results: outputs/strategy_trajectories/ (~3 GB). Finished runs are skipped,
# so a job that hit its time limit can simply be submitted again.
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh strategy_trajectories.py "$@"
python experiments/strategy_trajectories.py all "$@"
python experiments/plot_strategy_trajectories.py
