# Cross-Repo Changelog: Metal Native Hook Capture

This document summarizes coordinated changes across:

- `https://github.com/tburleyinfo/vLLM-Hook`
- `https://github.com/tburleyinfo/vllm-metal`
- `https://github.com/tburleyinfo/mlx-lm`

## 1) vLLM-Hook (Primary for this copy)

### Hook worker behavior and install gating

- Updated `vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py`.
- Added strict install gating:
  - Hook install failure now raises and aborts model startup.
  - Zero-hook installs are treated as failure.
  - Empty `VLLM_HOOK_LAYER_HEADS` is treated as failure.
- Added mode-aware install flow:
  - Prefer native model-runner capture path when callback support is present.
  - Fall back to PyTorch-style hooks only when module APIs are present.
- Improved runtime detection:
  - Added direct signature probe for model callback support to avoid early fallback mistakes.
- Added config fallback for layer/head mapping in Metal worker:
  - If `VLLM_HOOK_LAYER_HEADS` is missing, worker reads `VLLM_HOOK_CONFIG`.
  - Parses `params.important_heads` from the config JSON.
  - Raises explicit error if neither source resolves hook layers.
- Updated success logging to match expected format:
  - `Installed X hooks on layers: [...]`
  - Canonical layer naming for native mode: `model.layers.<i>.self_attn.attn`.

### Notebook and dependency workflow (Metal)

- Updated `notebooks/demo_attntracker.ipynb` top setup flow:
  - Auto-detect backend (`metal` vs `vllm`).
  - Metal path uses local editable installs with `--no-deps` for:
    - `vllm_hook_plugins`
    - local `vllm-metal`
    - local `mlx-lm`
  - Added clean reinstall toggle (`CLEAN_REINSTALL`).
  - Added explicit verification cell for local wiring and callback signature.
- Split notebook variants for clarity:
  - Copied Metal workflow to `notebooks/demo_attntracker_metal.ipynb`.
  - Reverted `notebooks/demo_attntracker.ipynb` to committed baseline (`HEAD`).
- Updated model/config defaults for low-memory devices:
  - Default model set to `Qwen/Qwen2-1.5B-Instruct` (override via `ATTNTRACKER_MODEL`).
  - Config path selection now maps by model and resolves robustly from repo-root-aware paths.

### New/updated docs and requirements

- Added `docs/metal_docs/README.md`:
  - Metal setup/run guide, local-edit workflow, expected hook logs.
- Updated root `README.md`:
  - Added link to `docs/metal_docs/README.md` under Attention Tracker usage.
- Added `requirement_metal.txt`:
  - Base metal-oriented requirements.
- Added `requirement_metal_local.txt`:
  - Overlay with editable local repo references (`../vllm-metal`, `../mlx-lm`).
- Updated `vllm_hook_plugins/hook_llm.py`:
  - Exports `VLLM_HOOK_CONFIG` when `config_file` is provided, enabling worker-side fallback config parsing.

### Additional workspace state in this repo

The working tree also contains unrelated pre-existing modifications/deletions outside this changelog scope (for example old docs removals and `.gitignore` changes). Keep those separate when committing if needed.

## 2) vllm-metal

### Metal-native Q/K capture integration

- Updated `vllm_metal/v1/model_runner.py`.
- Added hook env/config parsing and state:
  - `VLLM_HOOK_FLAG`, `VLLM_HOOK_DIR`, `VLLM_RUN_ID`, `VLLM_HOOKQ_MODE`, `VLLM_TP_RANK`, `VLLM_HOOK_LAYER_HEADS`.
- Added callback support detection at model-load time:
  - `_detect_qk_capture_callback_support()` via `inspect.signature`.
- Added forward wrapper:
  - `_forward_model(...)` passes `qk_capture_callback` when supported.
  - Graceful fallback if callback unsupported.
- Rewired inference paths to use `_forward_model(...)` and preserved original calls as comments where replaced.
- Added Q/K capture pipeline:
  - `_capture_qk(...)` for callback ingestion.
  - `mlx` -> `torch(cpu)` conversion.
  - Artifact caching and flush to `qk.pt`.

### Runtime-observed layer validation

- Added canonical layer naming helpers.
- Added observed runtime layer tracking per run:
  - observed layer set
  - per-layer callback call counts
- Added flush-time hard checks:
  - raise if capture enabled but no runtime callbacks observed.
  - raise if requested layers are missing from observed runtime layers.
- Added logs:
  - `Observed Q/K capture on layers: [...]`
  - per-layer call counts.

## 3) mlx-lm

### Qwen2 callback extension points

- Updated `mlx_lm/models/qwen2.py` to thread capture callback through model stack:
  - `Attention.__call__`
  - `TransformerBlock.__call__`
  - `Qwen2Model.__call__`
  - `Model.__call__`
- Callback signature extended to include canonical layer name:
  - `Callable[[int, str, mx.array, mx.array], None]`
- Callback invoked from attention with concrete runtime layer identity:
  - `model.layers.<i>.self_attn.attn`

### Local test scaffold note

- `mlx_lm/models/cache.py` contains a commented local test injection snippet around `make_prompt_cache` (non-functional unless uncommented).

## Verification Summary (Cross-Repo)

- Metal worker now fails fast when hooks/capture are not successfully installed.
- Native Metal capture now provides both:
  - intended target layer list at install time, and
  - observed runtime layer list from actual callback hits.
- End-to-end correctness depends on local editable wiring being active in the executing environment.
