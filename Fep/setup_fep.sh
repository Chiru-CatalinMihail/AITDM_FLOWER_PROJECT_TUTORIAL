#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time setup on fep (run it on the login node, from the repository root).
#
#   bash Fep/setup_fep.sh            # PyTorch with CUDA: runs on CPU and GPU nodes
#   bash Fep/setup_fep.sh --cpu      # CPU-only PyTorch (smaller, CPU partitions only)
#
# 1. Creates your project folder (default ~/AITDM_Project, change it with
#    AITDM_PROJECT=/other/path) with data/, outputs/, logs/ and slurm/.
# 2. Links data/, outputs/ and logs/ of this clone to it, so every command of
#    the tutorials works unchanged and your data and results survive if you
#    delete or re-clone the repository.
# 3. Creates .venv with scripts/setup_env.sh (if it does not exist yet).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AITDM_PROJECT="${AITDM_PROJECT:-$HOME/AITDM_Project}"

case "${1:-}" in
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
esac

echo ">>> Project folder: $AITDM_PROJECT"
mkdir -p "$AITDM_PROJECT"/{data,outputs,logs,slurm}

for name in data outputs logs; do
    link="$REPO_ROOT/$name"
    target="$AITDM_PROJECT/$name"
    if [ -L "$link" ]; then
        ln -sfn "$target" "$link"
    elif [ -d "$link" ]; then
        # A real folder from an earlier run: move its content, then link
        echo ">>> Moving existing $name/ to $target"
        cp -a "$link/." "$target/"
        rm -rf "$link"
        ln -s "$target" "$link"
    else
        ln -s "$target" "$link"
    fi
    echo "    $name/ -> $target"
done

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo ">>> .venv exists already (delete it to reinstall)"
else
    bash "$REPO_ROOT/scripts/setup_env.sh" "$@"
fi

cat <<EOF

=== fep setup done ===
Data and results live in $AITDM_PROJECT.
Next, check that everything works on a compute node (about 5 minutes):
    bash Fep/submit.sh smoke_test
    squeue --me
EOF
