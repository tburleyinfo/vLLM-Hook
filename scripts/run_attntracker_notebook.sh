#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
METAL_REPO_ROOT="${VLLM_METAL_REPO:-$HOOK_REPO_ROOT/../vllm-metal}"
VENV_PATH="$METAL_REPO_ROOT/.venv-vllm-metal"
NOTEBOOK_PATH="$HOOK_REPO_ROOT/notebooks/demo_attntracker.ipynb"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Error: missing environment at $VENV_PATH"
  echo "Run scripts/setup_attntracker_metal_env.sh first."
  exit 1
fi

if [[ ! -f "$NOTEBOOK_PATH" ]]; then
  echo "Error: notebook not found at $NOTEBOOK_PATH"
  exit 1
fi

source "$VENV_PATH/bin/activate"
cd "$HOOK_REPO_ROOT/notebooks"
jupyter lab demo_attntracker.ipynb
