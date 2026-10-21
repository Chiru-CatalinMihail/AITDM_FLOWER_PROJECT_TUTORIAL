# AITDM – Federated Learning Starter Kit

Starter code for the **AI for Trustworthy Decision Making** project. It gives you:

1. **A reproducible environment** with one command (Flower, PyTorch, MONAI, ...).
2. **A working federated learning app** (Flower + PyTorch + MONAI) you can run either
   with a single `flwr run` command or "for real" with one terminal per server / client.
3. **A tool to split any dataset in a biased (non-IID) way**, which simulates what
   real federated data looks like (different hospitals, devices, users, annotators).
4. **Trust mechanisms ready to use:** 9 aggregation strategies (robust, fair, adaptive),
   simulated attackers, differential privacy with ε reporting, Grad-CAM and feature importance.
5. **Two data tracks:** medical images (MedMNIST) and Human Activity Recognition (skeleton CSVs).
   Both use the same scripts and the same Flower app.

Everything is tested with **Flower 1.41**, PyTorch 2.14 and MONAI 1.6 on Python 3.12.

---

## Quick start (5 commands)

```bash
# 1. Create .venv and install everything (drop --cpu if you have an NVIDIA GPU)
bash scripts/setup_env.sh --cpu
source .venv/bin/activate

# 2. Download a medical dataset (DermaMNIST, skin lesions) and split it into
#    3 biased clients
python data_splitting/split_data.py --dataset dermamnist --num-clients 3 --strategy dirichlet --alpha 0.3

# 3a. Run server + 3 clients with ONE command (simulation)
bash scripts/run_simulation.sh

# 3b. ...or run them as separate processes (deployment), all from one terminal
bash scripts/run_deployment_local.sh

# 3c. ...or open one window per component, and re-run pieces after the first run
bash scripts/run_deployment_terminals.sh
```

You can also start every component by hand in its own terminal; see
[docs/02_running_flower.md](docs/02_running_flower.md).

**HAR instead of images?** Put the CSVs from Moodle in `data/raw/har/` and replace step 2 with:

```bash
python data_splitting/split_data.py --input data/raw/har/train_client*.csv \
    --label-col Action --sample-col IDSample --feature-cols '^J' \
    --strategy group --group-col Camera --num-clients 3 --out data/partitions/har_camera
bash scripts/run_simulation.sh data/partitions/har_camera
```

See [docs/06_har.md](docs/06_har.md).

---

## Tutorials

| # | Tutorial | What you learn |
|---|----------|----------------|
| 1 | [Environment setup](docs/01_environment_setup.md) | venv, `requirements.txt`, CPU vs GPU PyTorch, Windows/WSL2 |
| 2 | [Running Flower](docs/02_running_flower.md) | Simulation (`flwr run`) vs. deployment (SuperLink + SuperNodes in separate terminals) |
| 3 | [Biased data splits](docs/03_biased_data_splits.md) | IID, label skew (Dirichlet), pathological, quantity skew, natural (per-subject) splits, feature skew, label noise |
| 4 | [Aggregation strategies](docs/04_aggregation_strategies.md) | FedAvg, FedProx, FedAvgM, FedAdam/Yogi, FedMedian, FedTrimmedAvg, Krum, q-FedAvg, differential privacy; attack experiments |
| 5 | [Grad-CAM](docs/05_gradcam.md) | Explaining predictions, choosing the layer, sanity checks, what to report, and the iterative process: from errors to a better model (worked example) |
| 6 | [Human Activity Recognition](docs/06_har.md) | The HAR skeleton data end to end: splitting, training, trust experiments, feature importance |

---

## Repository structure

```text
AITDM-FL-Starter/
├── README.md                    <- you are here
├── requirements.txt             <- pinned dependencies (Flower, PyTorch, MONAI, ...)
├── scripts/
│   ├── setup_env.sh             <- creates .venv and installs everything
│   ├── run_simulation.sh        <- server + all clients, one command (Simulation Runtime)
│   ├── run_deployment_local.sh  <- SuperLink + SuperNodes + run, all in one terminal
│   ├── run_deployment_terminals.sh <- one window per component, re-run any of them from a menu
│   ├── _terminal_loop.sh        <- (used by run_deployment_terminals.sh) the menu of each window
│   ├── start_superlink.sh       <- terminal 1: the server side
│   ├── start_supernode.sh       <- terminals 2..N: one client each
│   ├── submit_run.sh            <- last terminal: submit the app (flwr run)
│   └── stop_flower.sh           <- kill all Flower processes (ports in use, stuck runs)
├── data_splitting/
│   ├── split_data.py            <- CLI: dataset (MedMNIST / .npz / .csv) -> biased client partitions + plots
│   └── partitioning.py          <- the splitting algorithms (reusable functions)
├── explainability/
│   ├── error_analysis.py        <- where does the model fail? per class / client, hardest examples
│   ├── gradcam.py               <- Grad-CAM heat maps (image models)
│   └── feature_importance.py    <- permutation importance (feature models, e.g. HAR)
├── fl_app/                      <- the Flower app (what `flwr run` executes)
│   ├── pyproject.toml           <- app metadata + run configuration (rounds, lr, strategy, ...)
│   └── aitdm_fl/
│       ├── task.py              <- models (MONAI CNN / MLP), data loading, train / evaluate (no Flower here)
│       ├── client_app.py        <- ClientApp: local training & evaluation (+ optional attacker)
│       └── server_app.py        <- ServerApp: aggregation strategy, DP, global evaluation
├── docs/                        <- the tutorials + figures
├── data/        (git-ignored)   <- raw downloads and generated partitions
├── outputs/     (git-ignored)   <- final_model.pt + metrics.json of every run
└── logs/        (git-ignored)   <- component logs of run_deployment_local.sh
```

---

## How this maps to the project stages

| Project requirement | Where to start |
|---|---|
| *Stage 1 – simulate 3–5 clients with non-IID splits* | `data_splitting/split_data.py` ([tutorial 3](docs/03_biased_data_splits.md)) |
| *Stage 1 – minimal federated experiment (FedAvg)* | `scripts/run_simulation.sh` ([tutorial 2](docs/02_running_flower.md)) |
| *Stage 1 – centralised vs. federated comparison* | Reuse `fl_app/aitdm_fl/task.py` (`build_model`, `train`, `evaluate`) in a centralised script on the same `test.npz` ([example](docs/06_har.md#centralised-baseline)) |
| *Stage 2 – different aggregation types* | `--run-config "strategy='krum'"` etc. ([tutorial 4](docs/04_aggregation_strategies.md)) |
| *Stage 2 – differential privacy* | `--run-config "dp-noise-multiplier=1.0"`, ε is logged and saved ([tutorial 4](docs/04_aggregation_strategies.md#5-privacy-central-differential-privacy)) |
| *Stage 2 – robustness / data poisoning* | `malicious-clients='4'` (model poisoning) or `split_data.py --label-noise` (data poisoning) + robust strategies |
| *Stage 2 – fairness* | `strategy='qfedavg'`, per-client evaluation, quantity-skew splits |
| *Improving the model iteratively* | `explainability/error_analysis.py` + Grad-CAM + `loss-weighting='balanced'` ([worked example](docs/05_gradcam.md#the-iterative-process-from-explanation-to-a-better-model)) |
| *Stage 2 – interpretability* | `python explainability/gradcam.py --run outputs/<run>` ([tutorial 5](docs/05_gradcam.md)); HAR: `explainability/feature_importance.py` ([tutorial 6](docs/06_har.md#5-explain-it)) |
| *Team work – cross-evaluation* | Each member gets a partition (`client_<i>.npz`); everyone evaluates on the shared `test.npz` |

---

## Useful links

* Flower docs: <https://flower.ai/docs/framework/>
  * [Quickstart PyTorch](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
  * [Run simulations](https://flower.ai/docs/framework/how-to-run-simulations.html)
  * [Run with the Deployment Engine](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)
  * [Flower Datasets partitioners](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html)
* MONAI: <https://docs.monai.io/>
* MedMNIST (datasets used in the examples): <https://medmnist.com/>
* Federated learning survey: <https://arxiv.org/abs/1907.09693>
* DP book (Dwork & Roth): <https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf>
