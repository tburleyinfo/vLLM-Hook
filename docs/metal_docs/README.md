# Metal Attention Tracker (Local Dev)

This guide explains how to set up and run `demo_attntracker.py` on Apple Silicon using `vllm-metal`, with support for local changes in sibling repositories.

## Repositories Expected

This layout is assumed:

- `{path-to-directory}/vLLM-Hook`
- `{path-to-directory}/vllm-metal`
- `{path-to-directory}/mlx-lm`

## 1) Use the vllm-metal Python Environment

```bash
source {path-to-directory}/vllm-metal/.venv-vllm-metal/bin/activate
python -V
which python
```

## 2) Install Local Editable Repos (No Dependency Re-resolution)

Use `--no-deps` to avoid resolver conflicts (notably around `transformers`).

```bash
python -m pip uninstall -y mlx-lm vllm-metal vllm-hook-plugins
python -m pip install -e {path-to-directory}/mlx-lm --no-deps
python -m pip install -e {path-to-directory}/vllm-metal --no-deps
python -m pip install -e {path-to-directory}/vLLM-Hook/vllm_hook_plugins --no-deps
```

Verify editable wiring:

```bash
python -m pip list --editable | rg "mlx-lm|vllm-metal|vllm-hook-plugins"
```

## 3) Verify MLX-LM Callback Support Is Active

```bash
python - <<'PY'
import inspect
import mlx_lm
from mlx_lm.models import qwen2
print("mlx_lm:", mlx_lm.__file__)
print("qwen2:", qwen2.__file__)
print("Model.__call__:", inspect.signature(qwen2.Model.__call__))
PY
```

Expected:

- `mlx_lm` and `qwen2` resolve to local repo paths under `{path-to-directory}/mlx-lm/...`
- `Model.__call__` includes `qk_capture_callback`

## 4) Run Demo Attention Tracker

From `vLLM-Hook` root:

```bash
cd {path-to-directory}/vLLM-Hook

ATTNTRACKER_MAX_MODEL_LEN=512 \
PYTHONPATH={path-to-directory}/vllm-metal \
{path-to-directory}/vllm-metal/.venv-vllm-metal/bin/python \
examples/demo_attntracker.py
```

Optional overrides:

```bash
export ATTNTRACKER_MODEL=Qwen/Qwen2-1.5B-Instruct
export ATTNTRACKER_GPU_MEM_UTIL=0.7
```

## 5) What Successful Hooking Looks Like on Metal

You should see install/observation logs similar to:

- `Using model-runner native Q/K capture path for layers: [...]`
- `Installed X hooks on layers: ['model.layers.6.self_attn.attn', ...]`
- `Observed Q/K capture on layers: ['model.layers.6.self_attn.attn', ...]`

Important:

- The worker is configured to **fail fast** if hooks/capture are not installed.
- The runner also validates runtime observations and raises if requested layers were not actually captured.

## Local-Changes Note (Metal)

When working on local branches in `vllm-metal` or `mlx-lm`, always reinstall editable packages in the active `vllm-metal` venv after switching branches or pulling new commits:

```bash
python -m pip install -e {path-to-directory}/mlx-lm --no-deps
python -m pip install -e {path-to-directory}/vllm-metal --no-deps
python -m pip install -e {path-to-directory}/vLLM-Hook/vllm_hook_plugins --no-deps
```

If behavior looks stale, do a full uninstall/reinstall sequence (Section 2), then restart the Python process/kernel.
