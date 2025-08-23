
#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="live-mvp-cuda"

# Make sure we run in the repo root (where pyproject.toml lives)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Ensure conda exists; if not, install Mambaforge lightweight
if ! command -v conda >/dev/null 2>&1; then
  echo "[info] conda not found; installing Mambaforge in $HOME/mambaforge ..."
  curl -L -o /tmp/mambaforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
  bash /tmp/mambaforge.sh -b -p "$HOME/mambaforge"
  eval "$("$HOME/mambaforge/bin/conda" shell.bash hook)"
  conda init bash
  echo "[info] Restart your shell if conda command is not recognized."
fi

# Load conda shell functions
eval "$(conda shell.bash hook)"

echo "[info] Creating conda env ${ENV_NAME} ..."
conda create -y -n "${ENV_NAME}" python=3.11 pip

echo "[info] Installing JAX (CUDA12) + optax via pip ..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
# JAX CUDA wheels (Linux only)
conda run -n "${ENV_NAME}" python -m pip install -U "jax[cuda12]" optax

echo "[info] Installing live_mvp (editable) ..."
conda run -n "${ENV_NAME}" python -m pip install -e .

echo "[info] Verifying JAX detects the GPU ..."
conda run -n "${ENV_NAME}" python scripts/verify_jax.py

cat <<EOF

[OK] Environment '${ENV_NAME}' is ready.

Activate it with:
  conda activate ${ENV_NAME}

Run the demo:
  python -m live_mvp.train_live

In VS Code:
  1) Install the "WSL" extension.
  2) Open this folder in WSL (Remote: WSL).
  3) Select interpreter: ${ENV_NAME}

EOF
