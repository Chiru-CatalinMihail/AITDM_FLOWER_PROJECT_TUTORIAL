"""Figures for experiments/strategy_trajectories.py: the server's weights over
the rounds, compressed from R^P to 2 coordinates with ONE shared PCA, and time t
as the third axis.

    python experiments/plot_strategy_trajectories.py                 # steps from FedAvg
    python experiments/plot_strategy_trajectories.py --base fedavgm  # steps from FedAvgM

Writes docs/images/strategy_trajectories_{absolute,relative}[_<base>].png and
outputs/strategy_trajectories/projection_<base>.json (the input of
experiments/strategy_trajectories_explorer.py, the interactive version).

Every curve is the mean over the seeds, taken in weight space (the seeds share
the initial model, see --init-seed). Dark: real runs. Light, same hue: the
one-step counterfactuals computed along the base run ("what this strategy would
do from the base's model at round t").

Two views:
* absolute: PCA of all curves together, centred on their mean. Shows where the
  strategies go; the counterfactuals sit (almost) on top of the base, because they
  are one step away from it.
* relative to the base: the same, after subtracting the base's model of the same
  round (the base is the line 0). Shows HOW each strategy deviates from it,
  for the real runs (accumulated over many rounds) and for a single step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ["fedavg", "fedprox", "fedavgm", "fedadam", "fedyogi"]
LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedavgm": "FedAvgM",
          "fedadam": "FedAdam", "fedyogi": "FedYogi"}
# (dark, light) pairs from matplotlib's tab20
_T20 = plt.get_cmap("tab20").colors
COLORS = {"fedavg": (_T20[0], _T20[1]), "fedprox": (_T20[4], _T20[5]),
          "fedavgm": (_T20[6], _T20[7]), "fedadam": (_T20[8], _T20[9]),
          "fedyogi": (_T20[2], _T20[3])}


def seed_mean(paths: list[Path]) -> np.ndarray:
    acc = None
    for p in paths:
        a = np.load(p, mmap_mode="r")
        acc = np.array(a, dtype=np.float64) if acc is None else acc + a
    return (acc / len(paths)).astype(np.float32)


def load_curves(out: Path, base: str) -> tuple[dict, dict, np.ndarray, list[int]]:
    seeds = sorted(int(p.name[4:]) for p in (out / base).glob("seed*")
                   if (p / "metrics.json").exists())
    if not seeds:
        raise SystemExit(f"No finished runs in {out}. Run strategy_trajectories.py first.")
    settings = json.loads((out / base / f"seed{seeds[0]}" / "metrics.json").read_text())["settings"]
    real = {}
    for s in STRATEGIES:
        paths = [out / s / f"seed{k}" / "traj.npy" for k in seeds]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"Missing runs: {missing}")
        real[s] = seed_mean(paths)
    names = [c for c in STRATEGIES
             if all((out / base / f"seed{k}" / f"cf_{c}.npy").exists() for k in seeds)]
    if not names:
        raise SystemExit(f"No counterfactuals (cf_*.npy) in {out / base}")
    cf = {c: seed_mean([out / base / f"seed{k}" / f"cf_{c}.npy" for k in seeds])
          for c in names}
    rounds = np.arange(real[base].shape[0]) * int(settings["save_every"])
    return real, cf, rounds, seeds


def pca(blocks: list[np.ndarray], k: int, center: bool):
    """One basis for all blocks (rows = points). Returns projections, explained ratio."""
    X = torch.from_numpy(np.concatenate(blocks))
    mu = X.mean(0, keepdim=True) if center else torch.zeros(1, X.shape[1])
    Xc = X - mu
    _, S, V = torch.pca_lowrank(Xc, q=k + 4, center=False, niter=6)
    explained = (S[:k] ** 2 / (Xc ** 2).sum()).tolist()
    proj = [((torch.from_numpy(b) - mu) @ V[:, :k]).numpy() for b in blocks]
    return proj, explained


def figure(proj_real, proj_cf, rounds, explained, title, path: Path, base: str, norms=None):
    ncols = 3 if norms is None else 4
    fig = plt.figure(figsize=(6 * ncols, 5.2))
    ax3 = fig.add_subplot(1, ncols, 1, projection="3d")
    ax1, ax2 = fig.add_subplot(1, ncols, 2), fig.add_subplot(1, ncols, 3)
    for s, c in proj_real.items():
        kw = dict(color=COLORS[s][0], lw=1.8, label=LABELS[s])
        ax3.plot(rounds, c[:, 0], c[:, 1], **kw)
        ax1.plot(rounds, c[:, 0], **kw)
        ax2.plot(rounds, c[:, 1], **kw)
    for s, c in proj_cf.items():
        kw = dict(color=COLORS[s][1], lw=1.4, ls="--",
                  label=f"{LABELS[s]} step from {LABELS[base]}")
        # point 0 is w_0 (no step yet)
        ax3.plot(rounds[1:], c[1:, 0], c[1:, 1], **kw)
        ax1.plot(rounds[1:], c[1:, 0], **kw)
        ax2.plot(rounds[1:], c[1:, 1], **kw)
    ax3.set_xlabel("round t")
    ax3.set_ylabel(f"PC1 ({explained[0]:.0%})")
    ax3.set_zlabel(f"PC2 ({explained[1]:.0%})")
    ax1.set_xlabel("round t")
    ax1.set_ylabel(f"PC1 ({explained[0]:.0%} of variance)")
    ax2.set_xlabel("round t")
    ax2.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)")
    if norms is not None:
        ax4 = fig.add_subplot(1, ncols, 4)
        for (s, kind), v in norms.items():
            dark, light = COLORS[s]
            # from round 1 on: at t = 0 every distance is 0 (log scale)
            if kind == "real":
                ax4.plot(rounds[1:], v[1:], color=dark, lw=1.8)
            else:
                ax4.plot(rounds[1:], v[1:], color=light, lw=1.4, ls="--")
        ax4.set_yscale("log")
        ax4.set_xlabel("round t")
        ax4.set_ylabel(rf"$\|w - w^{{\mathrm{{{LABELS[base]}}}}}_t\|_2$ (all P weights)")
        ax4.set_title("Full distance, no projection")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/strategy_trajectories")
    parser.add_argument("--images", type=Path, default=REPO_ROOT / "docs/images")
    parser.add_argument("--base", choices=STRATEGIES, default="fedavg",
                        help="the run along which the one-step counterfactuals were computed")
    args = parser.parse_args()
    base = args.base

    real, cf, rounds, seeds = load_curves(args.out, base)
    cf_names = list(cf)
    n_seeds = f"mean of {len(seeds)} seeds"
    step = f"dashed = one step from {LABELS[base]}"
    suffix = "" if base == "fedavg" else f"_{base}"
    export = {"base": base, "rounds": rounds.tolist(), "seeds": seeds,
              "counterfactuals": cf_names}

    # --- absolute ---------------------------------------------------------
    blocks = [real[s] for s in STRATEGIES] + [cf[c] for c in cf_names]
    proj, expl = pca(blocks, 2, center=True)
    p_real = dict(zip(STRATEGIES, proj[:len(STRATEGIES)]))
    p_cf = dict(zip(cf_names, proj[len(STRATEGIES):]))
    figure(p_real, p_cf, rounds, expl,
           f"Server weights, one shared PCA ({n_seeds}); {step}",
           args.images / f"strategy_trajectories_absolute{suffix}.png", base)
    export["absolute"] = {"explained": expl,
                          "real": {s: v.tolist() for s, v in p_real.items()},
                          "counterfactual": {s: v.tolist() for s, v in p_cf.items()}}

    # --- relative to the base ---------------------------------------------
    ref = real[base]
    rel_real = {s: real[s] - ref for s in STRATEGIES}
    rel_cf = {c: cf[c] - ref for c in cf_names}
    others = [s for s in STRATEGIES if s != base]
    proj, expl = pca([rel_real[s] for s in others] + [rel_cf[c] for c in cf_names],
                     2, center=False)  # origin = the base
    r_real = {base: np.zeros((len(rounds), 2), np.float32),
              **dict(zip(others, proj[:len(others)]))}
    r_real = {s: r_real[s] for s in STRATEGIES}
    r_cf = dict(zip(cf_names, proj[len(others):]))
    norms = {**{(s, "real"): np.linalg.norm(rel_real[s], axis=1) for s in others},
             **{(c, "cf"): np.linalg.norm(rel_cf[c], axis=1) for c in cf_names}}
    figure(r_real, r_cf, rounds, expl,
           f"Server weights minus {LABELS[base]}'s of the same round ({n_seeds}); {step}",
           args.images / f"strategy_trajectories_relative{suffix}.png", base, norms=norms)
    export["relative"] = {"explained": expl,
                          "real": {s: v.tolist() for s, v in r_real.items()},
                          "counterfactual": {s: v.tolist() for s, v in r_cf.items()},
                          "distance": {f"{s}:{k}": v.tolist() for (s, k), v in norms.items()}}

    # validation curves and final test numbers, for context
    metrics = {s: [json.loads((args.out / s / f"seed{k}" / "metrics.json").read_text())
                   for k in seeds] for s in STRATEGIES}
    export["val_rounds"] = [p["round"] for p in metrics[base][0]["val_curve"]]
    export["val_accuracy"] = {s: np.mean([[p["accuracy"] for p in m["val_curve"]] for m in ms],
                                         axis=0).tolist() for s, ms in metrics.items()}
    export["test"] = {s: [m["test"] for m in ms] for s, ms in metrics.items()}
    path = args.out / f"projection_{base}.json"
    path.write_text(json.dumps(export))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
