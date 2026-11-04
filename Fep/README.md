# Running the starter kit on fep (Slurm)

fep (`fep.grid.pub.ro`) is the university cluster.
You log in to a shared **login node** and ask **Slurm** to run your experiments on a compute node, as *jobs*.
The code is exactly the one you use on your laptop; this folder only adds the fep setup and the job scripts.

| # | Guide | What you learn |
|---|---|---|
| 1 | [Environment setup on fep](01_environment_setup.md) | clone, `.venv`, data and results in `~/AITDM_Project`, a first test job |
| 2 | [Running Flower with Slurm](02_running_on_slurm.md) | jobs, partitions (CPU / GPU), simulation and deployment jobs, seeds, writing your own job |

Then continue with the tutorials shared by every system: [3. Biased data splits](../docs/03_biased_data_splits.md), [4. Aggregation strategies](../docs/04_aggregation_strategies.md), [5. Grad-CAM](../docs/05_gradcam.md), [6. HAR](../docs/06_har.md).
Their commands become jobs: `bash scripts/run_simulation.sh ARGS` → `bash Fep/submit.sh simulation ARGS`, `python experiments/run_seeds.py ARGS` → `bash Fep/submit.sh run_seeds ARGS`, any other command → `bash Fep/submit.sh command <command>`.

## Quick start

```bash
ssh <your.username>@fep.grid.pub.ro
git clone https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL.git
cd AITDM_FLOWER_PROJECT_TUTORIAL
bash Fep/setup_fep.sh                     # once: .venv (about 15 min), links to ~/AITDM_Project
bash Fep/submit.sh smoke_test             # about 3 min on a compute node; the log must end with SMOKE TEST PASSED

source .venv/bin/activate
python data_splitting/split_data.py --dataset dermamnist --num-clients 3 --strategy dirichlet --alpha 0.3
bash Fep/submit.sh simulation             # server + 3 clients, one job
squeue --me                               # is it running?
```

## Where everything is

```text
~/AITDM_FLOWER_PROJECT_TUTORIAL/          <- your clone (code + .venv); can be deleted and re-cloned
├── data     -> ~/AITDM_Project/data       (links created by setup_fep.sh)
├── outputs  -> ~/AITDM_Project/outputs
├── logs     -> ~/AITDM_Project/logs
└── Fep/
    ├── setup_fep.sh                      <- the one-time setup
    ├── submit.sh                         <- bash Fep/submit.sh [sbatch options] JOB [arguments]
    ├── my_partitions.sh                  <- which partitions (queues) your account may use
    └── jobs/                             <- smoke_test, simulation, deployment, run_seeds, centralised,
                                             reproduce_tutorials, command (one .sbatch file each)
~/AITDM_Project/                          <- your data and results, kept between clones
├── data/                                 <- raw datasets and partitions
├── outputs/                              <- one folder per run: metrics.json, final_model.pt
├── logs/<jobid>/                         <- SuperLink / SuperNode logs of deployment jobs
└── slurm/<job>-<jobid>.out               <- the log of every job
```

## What was tested on fep

On 4 November 2026, with Flower 1.41, PyTorch 2.14.1 (CUDA 13) and Python 3.12:

| Test | Partition | Result |
|---|---|---|
| `smoke_test` | haswell (CPU) | passed, 3 min |
| `smoke_test` with `--gres=gpu:1` | ucsx (A100), dgxa100 (A100) | passed, `CUDA available: True` |
| `simulation`, DermaMNIST, 3 clients (Dirichlet α = 0.3), 5 rounds | haswell, 8 CPUs | 85 s, test accuracy 0.716 |
| the same, on one A100 | GPU | 60 s, test accuracy 0.715 |
| two `simulation` jobs at the same time on the same node | haswell | both correct, separate SuperLinks and results |
| `deployment` with `-N 1 --gres=gpu:1`, 3 clients, 3 rounds | dgxa100 | 3 min, every ClientApp on the A100, test accuracy 0.711 |
| `deployment`, 3 clients, 3 rounds, on 1 node and on 2 nodes (connected by IP) | haswell | 2 min each, the same test accuracy 0.711 on both |
| `scripts/run_deployment_local.sh` inside a job, next to a `simulation` job on the same node | haswell | both correct (the job's own ports, not 8000) |
| `run_seeds`, 3 seeds × 10 rounds, FedAvg and FedAvg + `loss-weighting='balanced'`, plus `centralised` | haswell, dgxa100 | 7 min per experiment on CPU, 6 min on an A100; the numbers below |

The numbers match those of [tutorial 2](../Local/Linux/02_running_flower.md#centralised-baseline-and-seeds) computed on a laptop, within the spread over seeds:

```text
| Experiment                | Val acc.      | Val bal. acc. | Val macro-F1  | Worst client val acc. |
|---------------------------|---------------|---------------|---------------|-----------------------|
| central (10 epochs)       | 0.726 ± 0.012 | 0.455 ± 0.012 | 0.428 ± 0.047 | 0.591 ± 0.013         |
| fedavg (CPU, haswell)     | 0.717 ± 0.003 | 0.296 ± 0.041 | 0.305 ± 0.036 | 0.470 ± 0.040         |
| fedavg (GPU, dgxa100)     | 0.716 ± 0.006 | 0.288 ± 0.049 | 0.298 ± 0.049 | 0.468 ± 0.040         |
| fedavg-balanced (CPU)     | 0.647 ± 0.053 | 0.428 ± 0.034 | 0.375 ± 0.056 | 0.448 ± 0.011         |
```

Not tested yet: the `xl` partition (its nodes were down; their P100 GPUs are too old for the CUDA 13 PyTorch anyway) and the interactive tmux windows inside `salloc`.
