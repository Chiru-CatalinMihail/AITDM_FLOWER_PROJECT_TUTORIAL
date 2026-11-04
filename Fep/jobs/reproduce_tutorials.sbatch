#!/usr/bin/env bash
#SBATCH --job-name=reproduce_tutorials
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
# ---------------------------------------------------------------------------
# Re-creates every number of the tutorials (experiments/reproduce_tutorials.sh).
#   bash Fep/submit.sh reproduce_tutorials          # everything
#   bash Fep/submit.sh reproduce_tutorials t4       # only tutorial 4
# Finished seeds are skipped: if the job hits its time limit, submit it again.
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh reproduce_tutorials.sh "$@"
bash experiments/reproduce_tutorials.sh "$@"
