"""Ablation: how the server's weights move under FedAvg, FedProx, FedAvgM,
FedAdam and FedYogi, with everything else matched.

Two kinds of trajectories are recorded for every seed:

1. **Real runs.** One federated run per strategy. Every strategy sees the same
   initial model, the same mini-batches (in the same order) and the same dropout
   masks; only the aggregation differs. Budget: --rounds x --local-steps SGD
   steps per client (default 1000 x 100 = 100k).
2. **One-step counterfactuals along a base run** (by default FedAvg, with all
   the others; any base with --counterfactuals). At every round t, starting from
   the base's global model w_t, we compute the model each other strategy WOULD
   produce in that round, save it, and throw it away (the base continues with
   its own w_{t+1}):
     * FedProx:  the clients train again from w_t, on the same batches, with the
                 proximal term; the server averages them.
     * FedAvgM / FedAdam / FedYogi: applied to the base's client models of this
                 round. Their state (momentum, m_t, v_t) is accumulated along
                 the base's updates (a "shadow" optimiser), otherwise their first
                 step would be trivial (FedAvgM's first step IS FedAvg).

The server rules copy flwr.serverapp.strategy (Flower 1.x) line by line, on one
flat weight vector, so no Flower/Ray process is needed (and runs are ~100x
cheaper than flwr simulations). Client training is task.train() of the FL app.

    # everything (15 runs, in parallel), locally or inside a Slurm job:
    python experiments/strategy_trajectories.py all --partitions data/partitions/dirichlet
    # one run:
    python experiments/strategy_trajectories.py run --partitions data/partitions/dirichlet \\
        --strategy fedadam --seed 1
    # FedAdam / FedYogi steps along FedAvgM:
    python experiments/strategy_trajectories.py run --strategy fedavgm --seed 1 \
        --counterfactuals fedadam fedyogi
    # then the figure:
    python experiments/plot_strategy_trajectories.py              # base FedAvg
    python experiments/plot_strategy_trajectories.py --base fedavgm

Output: outputs/strategy_trajectories/<strategy>/seed<s>/
    traj.npy         (rounds/save_every + 1, P) float32 global weights w_0, w_s, w_2s ...
    cf_<name>.npy    (base runs only) the counterfactual w'_{t+1} for the same rounds
    metrics.json     settings, pooled-validation curve, final test metrics
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "fl_app"))

from aitdm_fl.task import (  # noqa: E402
    build_model,
    derive_seed,
    evaluate,
    load_npz_arrays,
    num_classes_of_dataset,
    train,
)

STRATEGIES = ["fedavg", "fedprox", "fedavgm", "fedadam", "fedyogi"]
COUNTERFACTUALS = ["fedprox", "fedavgm", "fedadam", "fedyogi"]
DEFAULT_OUT = REPO_ROOT / "outputs" / "strategy_trajectories"


# ---------------------------------------------------------------------------
# Server-side rules (flat float32 vectors), as in flwr.serverapp.strategy
# ---------------------------------------------------------------------------
class Server:
    """step(w, avg, server_round) -> new global weights. Keeps its own state."""

    def __init__(self, name: str, momentum: float, server_lr: float,
                 beta_1: float = 0.9, beta_2: float = 0.99, tau: float = 1e-3):
        self.name, self.momentum, self.eta = name, momentum, server_lr
        self.beta_1, self.beta_2, self.tau = beta_1, beta_2, tau
        self.buf = self.m = self.v = None

    def step(self, w: torch.Tensor, avg: torch.Tensor, server_round: int) -> torch.Tensor:
        if self.name in ("fedavg", "fedprox"):
            return avg.clone()
        if self.name == "fedavgm":
            # FedAvgM: pseudo-gradient old - new, momentum, SGD with lr 1 (Flower default)
            pg = w - avg
            self.buf = pg.clone() if self.buf is None else self.momentum * self.buf + pg
            return w - 1.0 * self.buf
        # FedOpt family
        delta = avg - w
        if self.m is None:
            self.m, self.v = torch.zeros_like(w), torch.zeros_like(w)
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * delta
        if self.name == "fedadam":
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * delta**2
            eta = (self.eta * np.sqrt(1 - self.beta_2 ** (server_round + 1.0))
                   / (1 - self.beta_1 ** (server_round + 1.0)))
        elif self.name == "fedyogi":
            self.v = self.v - (1 - self.beta_2) * delta**2 * torch.sign(self.v - delta**2)
            eta = self.eta
        else:
            raise ValueError(self.name)
        return w + eta * self.m / (torch.sqrt(self.v) + self.tau)


# ---------------------------------------------------------------------------
# Data and clients
# ---------------------------------------------------------------------------
class Client:
    """One client's training data (uint8 kept as is, converted per batch) and an
    endless stream of shuffled mini-batches that depends only on (seed, client),
    never on the strategy: every strategy sees exactly the same batches."""

    def __init__(self, path: Path, cid: int, seed: int, batch_size: int):
        data = np.load(path)
        self.x = torch.from_numpy(data["x_train"])  # (N, H, W, C) uint8
        self.y = torch.from_numpy(data["y_train"].reshape(-1).astype(np.int64))
        self.n = len(self.y)
        self.cid, self.batch_size = cid, batch_size
        self.gen = torch.Generator().manual_seed(derive_seed(seed, "batches", cid))
        self.perm, self.pos = torch.randperm(self.n, generator=self.gen), 0

    def _indices(self) -> torch.Tensor:
        if self.pos + self.batch_size > self.n:  # new epoch
            self.perm, self.pos = torch.randperm(self.n, generator=self.gen), 0
        idx = self.perm[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return idx

    def batches(self, k: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out = []
        for _ in range(k):
            idx = self._indices()
            x = self.x[idx].float().div_(255.0)
            x = x.permute(0, 3, 1, 2) if x.ndim == 4 else x[:, None]
            out.append((x.contiguous(), self.y[idx]))
        return out


def get_flat(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([v.detach().reshape(-1).float().cpu() for v in model.state_dict().values()])


def set_flat(model: torch.nn.Module, flat: torch.Tensor) -> None:
    state, i = model.state_dict(), 0
    for k, v in state.items():
        state[k] = flat[i:i + v.numel()].view_as(v).to(v.dtype)
        i += v.numel()
    model.load_state_dict(state)


def local_train(model, w: torch.Tensor, batches, lr: float, mu: float, seed: int) -> torch.Tensor:
    """task.train() of the FL app from global weights w, on the given batches."""
    set_flat(model, w)
    torch.manual_seed(seed)  # same dropout masks for every strategy
    train(model, batches, epochs=1, lr=lr, device=torch.device("cpu"), proximal_mu=mu)
    return get_flat(model)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run(args) -> None:
    torch.set_num_threads(args.threads)
    partitions = args.partitions.resolve()
    meta = json.loads((partitions / "meta.json").read_text())
    num_clients = int(meta["num_clients"])  # stale client_<i>.npz beyond this are ignored
    num_classes = num_classes_of_dataset(str(partitions))
    out = args.out / args.strategy / f"seed{args.seed}"
    if args.counterfactuals is None:  # default: FedAvg is the base of all the others
        args.counterfactuals = COUNTERFACTUALS if args.strategy == "fedavg" else []
    cf_names = [c for c in args.counterfactuals if c != args.strategy]
    done = (out / "metrics.json").exists() and all(
        (out / f"cf_{c}.npy").exists() for c in cf_names)
    if done and not args.force:
        print(f"{out} done already, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)

    clients = [Client(partitions / f"client_{c}.npz", c, args.seed, args.batch_size)
               for c in range(num_clients)]
    weights = torch.tensor([c.n for c in clients], dtype=torch.float32)
    weights /= weights.sum()  # FedAvg weights n_k / n

    x_test, y_test = load_npz_arrays(partitions / "test.npz", "test")
    model_shape = tuple(x_test.shape[1:])
    init_seed = args.seed if args.init_seed < 0 else args.init_seed
    torch.manual_seed(init_seed)
    model = build_model(model_shape, num_classes)
    w = get_flat(model)

    val = [load_npz_arrays(partitions / f"client_{c}.npz", "val") for c in range(num_clients)]
    xv, yv = np.concatenate([v[0] for v in val]), np.concatenate([v[1] for v in val])

    def loader(x, y, bs=1024):
        return [(torch.from_numpy(x[i:i + bs]), torch.from_numpy(y[i:i + bs]))
                for i in range(0, len(y), bs)]

    val_loader, test_loader = loader(xv, yv), loader(x_test, y_test)

    def evaluate_flat(flat, data):
        set_flat(model, flat)
        return evaluate(model, data, torch.device("cpu"))

    mu = args.mu if args.strategy == "fedprox" else 0.0
    server = Server(args.strategy, args.momentum, args.server_lr)
    shadows = {name: Server(name, args.momentum, args.server_lr) for name in cf_names}

    n_saved = args.rounds // args.save_every + 1
    traj = np.lib.format.open_memmap(out / "traj.npy.part", mode="w+", dtype=np.float32,
                                     shape=(n_saved, w.numel()))
    traj[0] = w.numpy()
    # cf[name][j] = counterfactual w'_{t+1} computed at round t+1 = j * save_every
    cfs = {name: np.lib.format.open_memmap(out / f"cf_{name}.npy.part", mode="w+",
                                           dtype=np.float32, shape=(n_saved, w.numel()))
           for name in shadows}
    for arr in cfs.values():
        arr[0] = w.numpy()

    curve, t0 = [], time.time()
    for rnd in range(1, args.rounds + 1):
        batches = [c.batches(args.local_steps) for c in clients]  # same for all strategies
        seeds = [derive_seed(args.seed, "dropout", c.cid, rnd) for c in clients]
        local = [local_train(model, w, b, args.lr, mu, s) for b, s in zip(batches, seeds)]
        avg = sum(wk * lw for wk, lw in zip(weights, local))
        save = rnd % args.save_every == 0

        for name, shadow in shadows.items():
            if name == "fedprox":
                prox = [local_train(model, w, b, args.lr, args.mu, s)
                        for b, s in zip(batches, seeds)]
                w_cf = sum(wk * lw for wk, lw in zip(weights, prox))
            else:
                w_cf = shadow.step(w, avg, rnd)  # state follows FedAvg's updates
            if save:
                cfs[name][rnd // args.save_every] = w_cf.numpy()

        w = server.step(w, avg, rnd)
        if save:
            traj[rnd // args.save_every] = w.numpy()
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            m = evaluate_flat(w, val_loader)
            curve.append({"round": rnd, **m})
            el = time.time() - t0
            print(f"[{args.strategy} seed{args.seed}] round {rnd}/{args.rounds} "
                  f"val_acc={m['accuracy']:.4f} val_loss={m['loss']:.4f} "
                  f"({el / rnd:.2f}s/round, ETA {el / rnd * (args.rounds - rnd) / 60:.0f} min)",
                  flush=True)

    traj.flush()
    del traj
    os.replace(out / "traj.npy.part", out / "traj.npy")
    for name in list(cfs):
        cfs[name].flush()
        del cfs[name]
        os.replace(out / f"cf_{name}.npy.part", out / f"cf_{name}.npy")
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                if k != "func"}
    (out / "metrics.json").write_text(json.dumps({
        "settings": {**settings, "init_seed_used": init_seed, "num_clients": num_clients,
                     "client_weights": weights.tolist(), "num_params": w.numel()},
        "val_curve": curve,
        "test": evaluate_flat(w, test_loader),  # once, at the end
        "seconds": time.time() - t0,
    }, indent=2))
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# All runs in parallel
# ---------------------------------------------------------------------------
def run_all(args, passthrough: list[str]) -> None:
    cpus = args.cpus or len(os.sched_getaffinity(0))
    jobs = [(s, st) for s in args.seeds for st in STRATEGIES]
    # FedAvg does ~2x the work (FedProx counterfactual): give it 2 threads
    threads = {st: (2 if st == "fedavg" else 1) for st in STRATEGIES}
    print(f"{len(jobs)} runs on {cpus} CPUs", flush=True)
    log_dir = args.out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running, pending, failed = [], list(jobs), []
    while pending or running:
        used = sum(threads[st] for _, _, st in running)
        while pending and used + threads[pending[0][1]] <= max(cpus, 2):
            seed, st = pending.pop(0)
            cmd = [sys.executable, __file__, "run", "--strategy", st, "--seed", str(seed),
                   "--threads", str(threads[st]), *passthrough]
            log = open(log_dir / f"{st}_seed{seed}.log", "w")
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT),
                            seed, st))
            used += threads[st]
        time.sleep(5)
        for p in [r for r in running if r[0].poll() is not None]:
            running.remove(p)
            status = "ok" if p[0].returncode == 0 else f"FAILED ({p[0].returncode})"
            if p[0].returncode != 0:
                failed.append(p[1:])
            print(f"{p[2]} seed{p[1]}: {status}", flush=True)
    if failed:
        raise SystemExit(f"Failed runs: {failed}; see {log_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--partitions", type=Path, default=REPO_ROOT / "data/partitions/dirichlet")
    common.add_argument("--out", type=Path, default=DEFAULT_OUT)
    common.add_argument("--rounds", type=int, default=1000, help="server rounds T")
    common.add_argument("--local-steps", type=int, default=100,
                        help="SGD steps per client per round (budget = rounds x local-steps)")
    common.add_argument("--batch-size", type=int, default=32)
    common.add_argument("--lr", type=float, default=1e-3, help="client Adam learning rate")
    common.add_argument("--mu", type=float, default=0.1, help="FedProx proximal-mu")
    common.add_argument("--momentum", type=float, default=0.9, help="FedAvgM server-momentum")
    common.add_argument("--server-lr", type=float, default=0.01, help="FedAdam/FedYogi eta")
    common.add_argument("--init-seed", type=int, default=0,
                        help="seed of the initial model, shared by all seeds so that the "
                             "trajectories can be averaged in weight space (-1 = use --seed)")
    common.add_argument("--save-every", type=int, default=2, help="save every k-th round")
    common.add_argument("--eval-every", type=int, default=25)
    common.add_argument("--force", action="store_true", help="re-run finished runs")

    p_run = sub.add_parser("run", parents=[common], help="one strategy, one seed")
    p_run.add_argument("--strategy", choices=STRATEGIES, required=True)
    p_run.add_argument("--seed", type=int, default=1)
    p_run.add_argument("--threads", type=int, default=1)
    p_run.add_argument("--counterfactuals", nargs="*", choices=COUNTERFACTUALS + ["fedavg"],
                       help="one-step counterfactuals computed from THIS run's model every "
                            "round (default: all others for fedavg, none otherwise; "
                            "give the flag alone for none)"),

    p_all = sub.add_parser("all", parents=[common], help="all strategies x seeds, in parallel")
    p_all.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p_all.add_argument("--cpus", type=int, default=0, help="0 = all CPUs this process may use")

    args = parser.parse_args()
    if args.mode == "run":
        run(args)
    else:
        # forward the common options (resolved values) to every run
        passthrough = []
        for action in common._actions:
            if action.dest == "help":
                continue
            value, flag = getattr(args, action.dest), action.option_strings[0]
            if isinstance(value, bool):
                passthrough += [flag] if value else []
            else:
                passthrough += [flag, str(value)]
        run_all(args, passthrough)


if __name__ == "__main__":
    main()
