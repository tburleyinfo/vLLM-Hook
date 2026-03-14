# Metal Attention Tracker (Local Dev)

This guide explains how to run the attention-tracker demo on Apple Silicon
using the current local-dev stack:

- `mlx`
- `mlx-lm`
- `vllm-metal`
- `vLLM-Hook`

It is written for this workspace layout:

- `/Users/timothyburley/opensource/mlx`
- `/Users/timothyburley/opensource/mlx-lm`
- `/Users/timothyburley/opensource/vllm-metal`
- `/Users/timothyburley/opensource/vLLM-Hook`

The Python environment of record is:

- `/Users/timothyburley/opensource/vllm-metal/.venv-vllm-metal`

## 1) Activate the vllm-metal Environment

```bash
source /Users/timothyburley/opensource/vllm-metal/.venv-vllm-metal/bin/activate
python -V
which python
```

## 2) Choose an MLX Install Strategy

There are two valid paths:

- Path A: local editable `mlx`
- Path B: keep the existing wheel-installed `mlx` and only use local
  `mlx-lm`, `vllm-metal`, and `vLLM-Hook`

### Path A: local editable `mlx`

This requires Apple’s Metal toolchain. If it is missing, editable build fails
with:

- `cannot execute tool 'metal' due to missing Metal Toolchain`
- `use: xcodebuild -downloadComponent MetalToolchain`

You can verify that state directly with:

```bash
xcrun -f metal
xcrun metal -v
```

If `xcrun metal -v` prints the missing-toolchain error, editable `mlx` builds
will continue to fail until the component is installed.

Install the toolchain first:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Then install the full local stack with `--no-deps`:

```bash
python -m pip uninstall -y mlx mlx-lm vllm-metal vllm-hook-plugins

python -m pip install -e /Users/timothyburley/opensource/mlx --no-deps
python -m pip install -e /Users/timothyburley/opensource/mlx-lm --no-deps
python -m pip install -e /Users/timothyburley/opensource/vllm-metal --no-deps
python -m pip install -e /Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins --no-deps
```

Then retry the editable `mlx` install:

```bash
python -m pip install -e /Users/timothyburley/opensource/mlx --no-deps
```

### Path B: keep the existing `mlx` wheel

If you do not want to rebuild `mlx`, leave the installed `mlx` package alone
and only reinstall the repos above it:

```bash
python -m pip uninstall -y mlx-lm vllm-metal vllm-hook-plugins

python -m pip install -e /Users/timothyburley/opensource/mlx-lm --no-deps
python -m pip install -e /Users/timothyburley/opensource/vllm-metal --no-deps
python -m pip install -e /Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins --no-deps
```

Verify editable wiring:

```bash
python -m pip list --editable | rg "mlx|mlx-lm|vllm-metal|vllm-hook-plugins"
```

## 3) Verify the Current Hook Path Is Active

This stack no longer relies on the old callback-only MLX path. The expected
current behavior is:

- `mlx.nn.Module` exposes `register_forward_hook`
- `mlx-lm` exposes `model.layers.<i>.self_attn.attn`
- `vllm-metal` and `vllm_hook_plugins` resolve to local repo paths

Run:

```bash
python - <<'PY'
import mlx
import mlx.nn as nn
import vllm_metal
import vllm_hook_plugins
from mlx_lm.models import qwen2

print("mlx:", mlx.__file__)
print("vllm_metal:", vllm_metal.__file__)
print("vllm_hook_plugins:", vllm_hook_plugins.__file__)
print("register_forward_hook:", hasattr(nn.Linear(1, 1), "register_forward_hook"))

args = qwen2.ModelArgs(
    model_type="qwen2",
    hidden_size=32,
    num_hidden_layers=2,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=4,
    rms_norm_eps=1e-5,
    vocab_size=128,
)
model = qwen2.Model(args)
print("has model.layers.0.self_attn.attn:",
      "model.layers.0.self_attn.attn" in dict(model.named_modules()))
PY
```

Expected:

- `vllm_metal` and `vllm_hook_plugins` resolve to local repo paths
- `mlx` resolves either to the local repo path or to the installed wheel path,
  depending on which install path you chose above
- `register_forward_hook: True`
- `has model.layers.0.self_attn.attn: True`

## 4) Run the Demo Script

From the `vLLM-Hook` repo root:

```bash
cd /Users/timothyburley/opensource/vLLM-Hook
```

The current script is still hardcoded for:

- `cache_dir = "/dccstor/pyrite/irene/"`
- `model = "ibm-granite/granite-3.1-8b-instruct"`
- `gpu_memory_utilization = 0.7`
- `max_model_len = 2048`

So run it directly only if those defaults make sense in your environment:

```bash
/Users/timothyburley/opensource/vllm-metal/.venv-vllm-metal/bin/python \
examples/demo_attntracker.py
```

If you want a lighter local run, edit [demo_attntracker.py](/Users/timothyburley/opensource/vLLM-Hook/examples/demo_attntracker.py) before running:

- switch the model to `Qwen/Qwen2-1.5B-Instruct`
- lower `max_model_len`
- point `cache_dir` at a local writable path

## 5) Run the Notebook

Install the venv as a Jupyter kernel if needed:

```bash
python -m ipykernel install --user \
  --name vllm-metal \
  --display-name "vllm-metal"
```

Launch Jupyter from the same venv:

```bash
cd /Users/timothyburley/opensource/vLLM-Hook
jupyter lab notebooks/demo_attntracker.ipynb
```

Use the `vllm-metal` kernel for the notebook.

## 6) What Successful Hooking Looks Like on Metal Now

You should see logs shaped like:

- `Installed X hooks on layers: ['model.layers.6.self_attn.attn', ...]`
- `Hooks installed successfully`

And then the analysis phase should complete without a missing-`qk.pt` failure.

Conceptually, the current Metal path is:

- MLX provides PyTorch-style forward hooks
- `mlx-lm` exposes a stable `self_attn.attn` hook target
- `vllm-metal` preserves that target even under paged attention
- the metal worker writes the same `qk.pt` artifact schema expected by `vLLM-Hook`

## 7) Local-Changes Workflow

When you switch branches or pull new commits in any of these repos:

- `/Users/timothyburley/opensource/mlx-lm`
- `/Users/timothyburley/opensource/vllm-metal`
- `/Users/timothyburley/opensource/vLLM-Hook`

reinstall the editable packages into the active venv:

```bash
python -m pip install -e /Users/timothyburley/opensource/mlx-lm --no-deps
python -m pip install -e /Users/timothyburley/opensource/vllm-metal --no-deps
python -m pip install -e /Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins --no-deps
```

If you are also iterating on `mlx` itself, reinstall that too:

```bash
python -m pip install -e /Users/timothyburley/opensource/mlx --no-deps
```

If behavior looks stale:

1. Uninstall the editable packages again.
2. Reinstall them in the same order as the install path you chose in Section 2.
3. Restart the Python process or notebook kernel.
