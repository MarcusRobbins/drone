#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate live-mvp-jax
python -m pip install -U pytest numpy
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
pytest
