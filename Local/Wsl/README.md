# Running the starter kit on Windows (WSL2)

On Windows, everything runs inside **WSL2** (Windows Subsystem for Linux): a real Ubuntu
next to Windows. The code, the scripts and the tutorials are exactly the ones of the
[Linux guides](../Linux/README.md). This page lists only what is different in WSL, and
how to set it up.

Tested end to end on WSL2 (kernel 6.18), **Ubuntu 20.04**, CPU only (16 cores, 15 GB RAM given to WSL):
setup, the unit and smoke tests, simulation, the three deployment modes, Grad-CAM / error analysis,
and `experiments/reproduce_tutorials.sh` for tutorials 3, 4 and 5 (36 experiments × 3 seeds). The numbers
match the tutorials, up to tiny floating-point differences (≤ 0.003). Tutorial 6 (HAR)
was run on synthetic data in the NTU format, because the course data is not public.

| Step | Time on that machine |
|---|---|
| `bash scripts/setup_env.sh --cpu` | 2 min (2.2 GB in `.venv/`) |
| `bash tests/smoke_test.sh` | 1.5 min |
| `bash scripts/run_simulation.sh` (DermaMNIST, 3 clients, 5 rounds) | 1 min |
| `bash scripts/run_deployment_local.sh` (2 rounds) | 1.5 min |
| `bash experiments/reproduce_tutorials.sh` (t3, t4, t5) | 2 h |

---

## 1. Install WSL2 and Ubuntu

In **PowerShell as administrator**:

```powershell
wsl --install            # installs WSL2 + the latest Ubuntu LTS; reboot afterwards
wsl --list --verbose     # VERSION must be 2, not 1
```

Open the **Ubuntu** app from the Start menu and choose a user name and password (this is
the password `sudo` asks for later).

Already have WSL? Run `wsl --update` in PowerShell, and `wsl --set-version Ubuntu 2` if the distribution
still uses WSL 1.

## 2. Clone the repository in the Linux file system

```bash
cd ~                     # /home/<you>, NOT /mnt/c/Users/...
git clone https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL.git
cd AITDM_FLOWER_PROJECT_TUTORIAL
```

* **Do not work under `/mnt/c/...`** (your Windows disk). Reading thousands of files there
  (the venv, the datasets) is many times slower, and Ray (Flower's simulation engine) can
  have problems with its sockets there.
* Clone **from WSL**, not with Git for Windows. If you did clone it with Git for Windows, the
  repository's `.gitattributes` keeps the scripts in Unix (LF) line endings. Without it, the
  scripts would fail with `$'\r': command not found`.
* To see the files in Windows Explorer, run `explorer.exe .` in the repository folder, or open
  `\\wsl.localhost\Ubuntu\home\<you>` in Explorer.
  In VS Code, install the *WSL* extension and run `code .` from WSL.

## 3. Python 3.11+ (the main WSL difference)

Flower needs **Python 3.11 – 3.13**. Check what your Ubuntu ships with:

```bash
python3 --version
```

| Ubuntu | `python3` | What to do |
|---|---|---|
| 24.04 (current `wsl --install`) | 3.12 | nothing: `sudo apt install python3.12-venv`, then go to step 4 |
| 22.04 | 3.10 | too old: install uv (below) |
| 20.04 | 3.8 | too old: install uv (below) |

**uv** installs a recent Python in your home folder, with **no `sudo`** and without
touching the system Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env          # or open a new terminal
```

That is all: `scripts/setup_env.sh` notices that `python3` is too old and gets Python 3.12
from uv by itself (`>>> python3 is too old; getting Python 3.12 from uv`). It also uses
`python3.13` / `python3.12` / `python3.11` if one of them is already installed.

<details><summary>Alternative with sudo (deadsnakes)</summary>

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update && sudo apt install python3.12 python3.12-venv
PYTHON=python3.12 bash scripts/setup_env.sh --cpu
```
</details>

## 4. Create the environment

```bash
bash scripts/setup_env.sh --cpu
source .venv/bin/activate
bash tests/smoke_test.sh           # optional: about 1.5 minutes, ends with SMOKE TEST PASSED
```

Then follow the Linux guides from [1. Environment setup](../Linux/01_environment_setup.md#using-the-environment)
on, and [2. Running Flower](../Linux/02_running_flower.md). All their commands work unchanged.

### NVIDIA GPU (optional, not tested yet)

WSL2 can use the GPU of the Windows machine:

1. Install the normal NVIDIA driver **on Windows**. Never install a Linux NVIDIA driver inside WSL.
2. In WSL, `nvidia-smi` must show your GPU.
3. Create the venv **without** `--cpu`: `bash scripts/setup_env.sh` (if a CPU venv exists, delete
   `.venv/` first). The setup prints `CUDA available: True`.

Without an NVIDIA GPU, always use `--cpu`: the default PyTorch build adds several GB of CUDA
libraries that WSL cannot use.

---

## 5. What is different when running

### Windows for each component (`run_deployment_terminals.sh`)

The script looks for a graphical terminal first, then falls back to **tmux**:

| Your WSL | What happens | Install |
|---|---|---|
| Windows 11 / recent Windows 10 (WSLg, a graphical desktop for Linux apps) | real windows, if a terminal program is installed | `sudo apt install xterm` |
| no `sudo`, or no WSLg | **tmux**: all windows inside your current terminal (tested) | usually preinstalled; else `sudo apt install tmux` |

Force one with `FL_TERMINAL=tmux` or `FL_TERMINAL=xterm`. In tmux:

* `Ctrl+b n` / `Ctrl+b p` next / previous window, `Ctrl+b w` list;
* `Ctrl+b d` detach (everything keeps running), `tmux attach -t aitdm-fl` to come back;
* the menus wait for a whole line: type the letter (`m`, `e`, `q`, ...) **and press Enter**.
* Closing tmux does not stop Flower: run `bash scripts/stop_flower.sh` (and
  `tmux kill-session -t aitdm-fl`).

`run_simulation.sh`, `run_deployment_local.sh` and the "one terminal per component" steps of
[tutorial 2](../Linux/02_running_flower.md#b-deployment-every-component-in-its-own-terminal)
need nothing extra: open several Ubuntu tabs in Windows Terminal (`Ctrl+Shift+T`).

### Opening HTML pages and figures

The plots (`label_distribution.png`, Grad-CAM, ...) and the [Dirichlet explorer](../../dirichlet_explorer.html)
open with Windows programs:

```bash
explorer.exe dirichlet_explorer.html             # in the Windows browser
explorer.exe data/partitions/dirichlet/label_distribution.png
explorer.exe outputs                             # the folder in Windows Explorer
```

`explorer.exe` returns exit code 1 even when it worked; ignore it.
`wslview <file>` (package `wslu`) does the same.

### Memory and CPU

WSL2 gets half of the Windows RAM and all CPU cores by default. That was enough for every
tutorial experiment (tested with 15 GB). With many simulated clients,
or larger images (`--size 64` / `128`), Ray can run out of memory (`OutOfMemoryError`, workers
killed). Then give WSL more RAM in `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=12GB
processors=8
```

and restart it from PowerShell with `wsl --shutdown`.

Long runs (`reproduce_tutorials.sh`, `run_seeds.py`): closing the Ubuntu window stops the run
started in it, and WSL shuts down shortly after its last terminal closes. Keep the window open.
Windows sleep also pauses WSL.

### Localhost

The SuperLink at `127.0.0.1:8000` is reachable from WSL only. That is all the scripts need.
For several machines ([tutorial 2](../Linux/02_running_flower.md#running-on-several-machines)),
WSL's default NAT networking hides your WSL from the other computers. Use
`networkingMode=mirrored` under `[wsl2]` in `.wslconfig` (Windows 11), and allow port 8000 in
the Windows firewall.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: Python 3.11-3.13 required, but python3 is older ...` | Install uv ([step 3](#3-python-311-the-main-wsl-difference)) and re-run `bash scripts/setup_env.sh --cpu` |
| `$'\r': command not found` / `bad interpreter: /bin/bash^M` | The scripts have Windows line endings (cloned with Git for Windows). Clone again, from WSL, into `~` |
| Everything is very slow | The repository is under `/mnt/c/...`: clone it into `~` ([step 2](#2-clone-the-repository-in-the-linux-file-system)) |
| `run_deployment_terminals.sh` opens no window | No graphical terminal installed: it uses tmux in the current terminal (`tmux attach -t aitdm-fl`), or install `xterm` |
| A run stops when I close the terminal | Closing the window stops it. Keep the window open during long runs |
| Ray: `OutOfMemoryError` / workers killed | Give WSL more memory in `.wslconfig` ([above](#memory-and-cpu)) |
| `CUDA available: False` with an NVIDIA GPU | Windows driver installed? `nvidia-smi` works in WSL? venv created without `--cpu`? ([GPU](#nvidia-gpu-optional-not-tested-yet)) |

Everything else: see the troubleshooting of [tutorial 1](../Linux/01_environment_setup.md#troubleshooting),
[tutorial 2](../Linux/02_running_flower.md#troubleshooting) and the [main README](../../README.md#common-troubleshooting).
