#!/usr/bin/env bash
#SBATCH --job-name=strategy_trajectories_base
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
# ---------------------------------------------------------------------------
# One-step counterfactuals along another base strategy (3 seeds in parallel),
# then the figures. Needs the real runs of strategy_trajectories.sbatch.
#   bash Fep/submit.sh strategy_trajectories_base fedavgm fedadam fedyogi
#     (base FedAvgM; at every round, the FedAdam / FedYogi step from its model)
# The base runs are redone with the extra outputs (same seeds: same trajectory).
# ---------------------------------------------------------------------------
source Fep/jobs/_job_common.sh strategy_trajectories.py "$@"
BASE="$1"
shift
OUT=outputs/strategy_trajectories
for seed in 1 2 3; do
    # keep the first run's trajectory, to check that the rerun is identical
    [ -f "$OUT/$BASE/seed$seed/traj.npy" ] && cp "$OUT/$BASE/seed$seed/traj.npy" "$OUT/$BASE/seed$seed/traj_first.npy"
    python experiments/strategy_trajectories.py run --strategy "$BASE" --seed "$seed" \
        --counterfactuals "$@" > "outputs/strategy_trajectories/logs/${BASE}_cf_seed${seed}.log" 2>&1 &
done
wait
tail -n 2 "$OUT"/logs/"${BASE}"_cf_seed*.log
python - "$OUT/$BASE" <<'PY'
import sys, numpy as np, pathlib
for d in sorted(pathlib.Path(sys.argv[1]).glob("seed*")):
    if (d / "traj_first.npy").exists():
        diff = np.abs(np.load(d / "traj.npy") - np.load(d / "traj_first.npy")).max()
        print(f"{d.name}: rerun vs first run, max |diff| = {diff}")
        (d / "traj_first.npy").unlink()
PY
python experiments/plot_strategy_trajectories.py --base "$BASE"
