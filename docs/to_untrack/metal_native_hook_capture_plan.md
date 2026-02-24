# Metal Native Hook Capture Plan

## Objective

Fix both failure modes in Metal attention tracking:

1. `Installed 0 metal hooks on layers: []` followed by a misleading success message.
2. `FileNotFoundError: No Q/K cache artifacts found ...` at analyze time.

## Root Cause Summary

- The Metal worker was trying to use PyTorch hook APIs (`named_modules`, `register_forward_hook`) against MLX models.
- MLX models generally do not expose that PyTorch module/hook surface.
- The worker treated the hook install path as successful even when zero hooks were installed.
- No Q/K artifacts were written, so analyzer failed late with `FileNotFoundError`.

## Implementation Plan (Lowest Necessary Abstraction)

### 1. Capture at the model runner boundary in `vllm-metal`

- Add Metal-native Q/K capture in `vllm_metal/v1/model_runner.py`.
- Parse and cache hook env metadata (`VLLM_HOOK_*`).
- Add a forward wrapper that passes `qk_capture_callback` to MLX models when supported.
- Convert captured MLX arrays to CPU PyTorch tensors using `mlx_to_torch`.
- Persist artifacts in the same schema used by existing analyzer:
  - `${VLLM_HOOK_DIR}/${run_id}/tp_rank_${VLLM_TP_RANK}/qk.pt`
  - payload keys: `config`, `qk_cache`.

Checkpoint:

```text
[ ] model_runner accepts qk capture callback path
[ ] qk.pt is written under run_id/tp_rank_*/qk.pt
[ ] q/k tensor shapes match analyzer assumptions
```

### 2. Expose callback in `mlx-lm` only where needed

- Thread optional `qk_capture_callback` and `layer_idx` through:
  - `mlx_lm/models/qwen2.py::Attention.__call__`
  - `mlx_lm/models/qwen2.py::TransformerBlock.__call__`
  - `mlx_lm/models/qwen2.py::Qwen2Model.__call__`
  - `mlx_lm/models/qwen2.py::Model.__call__`
- Invoke callback right after rope/cache-updated `queries`/`keys` are available.

Checkpoint:

```text
[ ] qwen2 forward path accepts qk_capture_callback without changing default behavior
[ ] callback sees per-layer Q/K tensors during prefill/decode
```

### 3. Make Metal worker status truthful in `vLLM-Hook`

- Update `probe_hookqk_worker_metal.py` to:
  - Prefer runner-native callback mode when available.
  - Use PyTorch hook mode only as fallback.
  - Raise/fail early when neither mode is active.
  - Never print success for zero-hook/no-op installs.

Checkpoint:

```text
[ ] no "success" message when installed hooks == 0
[ ] worker reports native mode explicitly when active
[ ] invalid/empty VLLM_HOOK_LAYER_HEADS fails early with clear error
```

### 4. End-to-end validation

- Run the demo with Metal worker class and verify:
  - native mode activation message
  - non-empty `qk.pt` artifacts
  - analyzer completes without `FileNotFoundError`

Checkpoint:

```text
[ ] demo_attntracker runs through llm.analyze()
[ ] qk artifacts present for latest run_id
[ ] no late-stage cache artifact FileNotFoundError
```

## Install/Reinstall Commands During Test Cycle

Use these whenever switching local branches or changing local package wiring.

### A. Activate vllm-metal env

```bash
source /Users/timothyburley/opensource/vllm-metal/.venv-vllm-metal/bin/activate
python -V
which python
```

### B. Reinstall local `mlx-lm` into the active env (editable)

```bash
pip uninstall -y mlx-lm
pip install -e /Users/timothyburley/opensource/mlx-lm
pip show mlx-lm
pip list --editable | rg mlx-lm
```

### C. Reinstall local `vllm-metal` into the active env (editable)

```bash
pip uninstall -y vllm-metal
pip install -e /Users/timothyburley/opensource/vllm-metal
pip show vllm-metal
pip list --editable | rg vllm-metal
```

### D. Optional: force reinstall if stale metadata is suspected

```bash
pip install --force-reinstall -e /Users/timothyburley/opensource/mlx-lm
pip install --force-reinstall -e /Users/timothyburley/opensource/vllm-metal
```

### E. Sanity import path checks

```bash
python - <<'PY'
import mlx_lm, vllm_metal
print("mlx_lm:", mlx_lm.__file__)
print("vllm_metal:", vllm_metal.__file__)
PY
```

## Runtime Validation Command

```bash
ATTNTRACKER_MAX_MODEL_LEN=512 \
PYTHONPATH=/Users/timothyburley/opensource/vllm-metal \
/Users/timothyburley/opensource/vllm-metal/.venv-vllm-metal/bin/python \
/Users/timothyburley/opensource/vLLM-Hook/examples/demo_attntracker.py
```

Expected signal:

```text
Using model-runner native Q/K capture path for layers: [...]
Metal hooks installed successfully (native capture, layers=[...])
```

## Notes

- Constraint observed for this implementation: replaced logic should be commented rather than removed where practical.
- If future models do not support `qk_capture_callback`, add the same callback threading in the corresponding `mlx-lm` model file(s) used by that architecture.
