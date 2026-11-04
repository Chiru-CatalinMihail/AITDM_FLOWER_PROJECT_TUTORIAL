# 1. Environment setup on fep

This is the fep version of [the local setup tutorial](../Local/Linux/01_environment_setup.md).
The code is the same, only where things live and how you start them change:

| | On your laptop | On fep |
|---|---|---|
| Where you type commands | a terminal on your machine | the **login node** `fep.grid.pub.ro` (over SSH) |
| Where the training runs | your machine | a **compute node**, through a Slurm job |
| Data and results | `data/` and `outputs/` in the clone | `~/AITDM_Project/data` and `~/AITDM_Project/outputs` (linked into the clone) |
| How you start a run | `bash scripts/run_simulation.sh ...` | `bash Fep/submit.sh simulation ...` (same arguments) |

> Never train on the login node.
> It is shared by everybody, and long processes there are killed.
> Installing packages, downloading data and editing files are fine there.

## 0. Connect

```bash
ssh <your.username>@fep.grid.pub.ro
```

Your home folder is shared between the login node and all compute nodes, so whatever you create there is visible to your jobs.
Students have **50 GB** in their home folder: the environment takes about 6 GB, DermaMNIST a few MB.
Check your usage with `du -sh ~/AITDM_Project ~/AITDM_FLOWER_PROJECT_TUTORIAL`.

## 1. Clone and install (once, on the login node)

```bash
git clone https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL.git
cd AITDM_FLOWER_PROJECT_TUTORIAL
bash Fep/setup_fep.sh
```

The script:

1. creates your project folder `~/AITDM_Project/` with `data/`, `outputs/`, `logs/` and `slurm/` in it,
2. replaces `data/`, `outputs/` and `logs/` of the clone with links to those folders,
3. creates `.venv/` with [`scripts/setup_env.sh`](../scripts/setup_env.sh), exactly as on a laptop (Python 3.12 is already installed on fep).

Because of step 2, **every command of the tutorials works unchanged** (`data/partitions/...`, `outputs/...`), but the files are stored in `~/AITDM_Project`.
You can delete or re-clone the repository without losing your data and results.
Choose another folder with `AITDM_PROJECT=/some/path bash Fep/setup_fep.sh` (and set the same variable when you submit jobs).

The installation takes about 15 minutes and downloads about 6 GB: PyTorch with CUDA, so the same environment runs on the CPU and the GPU partitions.
`bash Fep/setup_fep.sh --cpu` installs the much smaller CPU-only PyTorch, if you will only use CPU partitions.

At the end it prints `CUDA available: False`: that is expected, the login node has no GPU.

```text
=== Environment ready ===
flwr        1.41.0
torch       2.14.1+cu130  (CUDA available: False)
```

## 2. Check it on a compute node

```bash
bash Fep/submit.sh smoke_test
```

```text
Submitted job 293017
Log:    /export/home/.../AITDM_Project/slurm/smoke_test-293017.out
Follow: tail -f /export/home/.../AITDM_Project/slurm/smoke_test-293017.out     Cancel: scancel 293017
```

The smoke test trains one federated round on synthetic data, runs the explanation scripts and a centralised baseline, in about 3 minutes.
The end of the log must say:

```text
torch 2.14.1+cu130 | CUDA available: False
>>> images: split, 1 round, explanations, centralised
>>> HAR-like CSV: split, 1 round, feature importance
>>> SMOKE TEST PASSED
```

On a GPU partition (see [which ones you can use](02_running_on_slurm.md#1-partitions-which-queue-can-i-use)) the first line says `CUDA available: True`:

```bash
bash Fep/submit.sh -p dgxa100 --gres=gpu:1 smoke_test
```

## 3. Get the data

Downloads are fine on the login node (they are small), and so is splitting a small dataset like DermaMNIST:

```bash
source .venv/bin/activate
python data_splitting/split_data.py --dataset dermamnist --num-clients 3 --strategy dirichlet --alpha 0.3
ls ~/AITDM_Project/data/partitions/dirichlet          # client_0.npz ... test.npz meta.json
```

For larger datasets (MedMNIST at 128 or 224 px, the full NTU archive), run the same command as a job instead, so it does not slow down the login node for everybody:

```bash
bash Fep/submit.sh command python data_splitting/split_data.py --dataset dermamnist --size 128 --num-clients 3 --strategy dirichlet --alpha 0.3 --out data/partitions/derma128_dir0.3
```

**HAR data:** copy the zip from Moodle to fep from your own machine, then install it on fep:

```bash
scp ~/Downloads/Demo_HAR.zip <your.username>@fep.grid.pub.ro:~/AITDM_Project/      # on your laptop
python data_splitting/get_data.py --har ~/AITDM_Project/Demo_HAR.zip               # on fep
```

The rest is as in [tutorial 6](../docs/06_har.md).

## Editing the code

The files you will change are the same as on a laptop ([list](../Local/Linux/01_environment_setup.md#where-the-code-you-will-edit-lives)).
Edit them on fep with `nano`/`vim`, or, more comfortably, with VS Code's *Remote - SSH* extension connected to `fep.grid.pub.ro`.
A job uses the files as they are **when it starts**: if you edit `task.py` while a job waits in the queue, the job runs the new version.

You can also develop on your laptop and only run the long experiments on fep: push your changes to your own repository and `git pull` on fep.
Never commit `data/` (it is git-ignored, also as a link).

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: no .venv yet` | Run `bash Fep/setup_fep.sh` on the login node first |
| `Disk quota exceeded` during the install | You are over 50 GB: `du -sh ~/* ~/.cache` and delete old things. `rm -rf ~/.cache/pip` frees about 3 GB after the install |
| The smoke test log says `CUDA available: False` on a GPU partition | You forgot `--gres=gpu:1`: without it Slurm gives you no GPU, even on a GPU node |
| `Invalid account or account/partition combination specified` | Your account may not use that partition: `bash Fep/my_partitions.sh` lists those you can, see [partitions](02_running_on_slurm.md#1-partitions-which-queue-can-i-use) |
| `data/` is an ordinary folder, not a link | You created it before running `setup_fep.sh`. Run the script again: it moves the content to `~/AITDM_Project/data` and creates the link |
