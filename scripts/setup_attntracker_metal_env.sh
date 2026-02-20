#!/usr/bin/env bash

set -euo pipefail

KERNEL_NAME="${1:-vllm-metal-attntracker}"
KERNEL_DISPLAY="${2:-vllm-metal-attntracker}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
METAL_REPO_ROOT="${VLLM_METAL_REPO:-$HOOK_REPO_ROOT/../vllm-metal}"
VENV_PATH="$METAL_REPO_ROOT/.venv-vllm-metal"

if [[ ! -f "$METAL_REPO_ROOT/install.sh" ]]; then
  echo "Error: vllm-metal install script not found at:"
  echo "  $METAL_REPO_ROOT/install.sh"
  echo "Set VLLM_METAL_REPO to override the location."
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Error: This setup requires Apple Silicon macOS (Darwin arm64)."
  exit 1
fi

echo "Installing vllm-metal into: $VENV_PATH"
pushd "$METAL_REPO_ROOT" >/dev/null
bash "./install.sh"
popd >/dev/null

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Error: expected virtual environment missing at $VENV_PATH"
  exit 1
fi

source "$VENV_PATH/bin/activate"

echo "Installing vLLM-Hook requirements into $VENV_PATH"
uv pip install -r "$HOOK_REPO_ROOT/requirement.txt"
uv pip install -e "$HOOK_REPO_ROOT/vllm_hook_plugins"
uv pip install ipykernel jupyterlab

python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY" --force

echo ""
echo "Setup complete."
echo "Environment : $VENV_PATH"
echo "Kernel name : $KERNEL_NAME"
echo ""
echo "Run the notebook with:"
echo "  source \"$VENV_PATH/bin/activate\""
echo "  cd \"$HOOK_REPO_ROOT/notebooks\""
echo "  jupyter lab demo_attntracker.ipynb"
echo ""
echo "Then select kernel: $KERNEL_DISPLAY"
