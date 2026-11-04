# Running the starter kit on Windows (WSL2)

> This guide is not written yet: WSL2 still has to be tested end to end.
> Until then, install WSL2 and follow the [Linux guides](../Linux/README.md) inside it.

Install WSL2: open PowerShell as administrator, run `wsl --install`, reboot, open the "Ubuntu" app.
Inside Ubuntu, clone the repository **in the Linux file system** (e.g. `~/AITDM_FLOWER_PROJECT_TUTORIAL`), not under `/mnt/c/...`, which is much slower.
Then follow [1. Environment setup](../Linux/01_environment_setup.md) and [2. Running Flower](../Linux/02_running_flower.md).

Known differences, to be confirmed:

* `run_deployment_terminals.sh` opens windows only if WSL has a graphical terminal (`sudo apt install xterm`), otherwise it uses tmux (`sudo apt install tmux`).
* Open `dirichlet_explorer.html` with the Windows browser: `explorer.exe dirichlet_explorer.html` from the repository folder.
