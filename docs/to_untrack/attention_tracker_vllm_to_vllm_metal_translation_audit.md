# Attention Tracker Translation Audit: vLLM -> vLLM-Metal (MLX)

## Scope

This document summarizes a hook-parity audit across:

- Upstream `vllm` package in the local virtualenv:  
  `/Users/timothyburley/opensource/vllm-metal/.venv-vllm-metal/lib/python3.12/site-packages/vllm/`
- Metal plugin implementation:  
  `/Users/timothyburley/opensource/vllm-metal/vllm_metal/`
- Hook integration layer:  
  `/Users/timothyburley/opensource/vLLM-Hook/vllm_hook_plugins/`

The local `vllm` tree contains `1707` files. The audit focuses on functionality required for `vllm-hook` (attention tracking + activation steering), not every vLLM feature.

## Executive Summary

`vllm-hook` is currently blocked on Metal primarily because vLLM GPU integrations depend on PyTorch module hooks and forward-context machinery that `vllm-metal` does not currently replicate.

This is not mainly a "bug in one worker file"; it is a backend-integration gap:

1. No PyTorch module traversal/hook surface on MLX models (`named_modules`, `register_forward_hook`).
2. No `set_forward_context(...)` usage in `MetalModelRunner.execute_model(...)`, so `get_forward_context()`-based hook logic has no data source.
3. No built attention metadata equivalent in Metal runner (`seq_lens`, `slot_mapping`, per-layer metadata map).
4. No explicit Q/K capture API in Metal runner/model path.
5. Layer naming and config schema differ from current hook assumptions.

## Core Parity Matrix (Hook-Critical)

| Capability | vLLM (GPU) | vLLM-Metal (current) | Gap Severity |
|---|---|---|---|
| Worker customization entrypoint | `worker_cls` supported | Supported | Low |
| Access underlying model | `self.model_runner.model` | `self.model_runner.model` | Low |
| Model config object access | `model.config` | `self.model_runner.model_args` (dict) | Medium |
| Module iteration | `model.named_modules()` | No equivalent on MLX model object | Critical |
| Forward hook registration | `module.register_forward_hook(...)` | No equivalent on MLX model object | Critical |
| Forward context set around model call | `set_forward_context(...)` used in GPU model runner | No usage in `vllm_metal.v1.model_runner` | Critical |
| Per-layer attention metadata map | Built in GPU runner (`layer_name -> metadata`) | Not built in Metal runner | Critical |
| Q/K extraction path for hooks | Hook input tuple from torch modules | No equivalent capture interface | Critical |
| Activation steering injection point | Forward hook on layer module output | No equivalent layer interception API | Critical |

## Evidence (Key File Anchors)

### 1) Forward context exists upstream and is used by GPU runner

- `vllm/forward_context.py:272` defines `set_forward_context(...)`
- `vllm/forward_context.py:221` defines `get_forward_context()`
- `vllm/v1/worker/gpu_model_runner.py:3279` wraps forward with `set_forward_context(...)`
- `vllm/v1/worker/gpu_model_runner.py:1566` builds attention metadata
- `vllm/v1/worker/gpu_model_runner.py:1746` maps metadata by `layer_name`

### 2) Metal runner currently does not set forward context or build attn metadata

- `vllm_metal/v1/model_runner.py:1085` `execute_model(...)` entry
- `vllm_metal/v1/model_runner.py:876` direct model call in prefill
- `vllm_metal/v1/model_runner.py:954` direct model call in batched decode
- `vllm_metal/v1/model_runner.py` has no `set_forward_context(...)` usage

### 3) Hook workers still assume PyTorch module hooks

- GPU worker:
  - `vllm_hook_plugins/workers/probe_hookqk_worker.py:142` uses `model.named_modules()`
  - `vllm_hook_plugins/workers/probe_hookqk_worker.py:148` uses `register_forward_hook`
- Metal worker (current):
  - `vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py:134`
  - `vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py:140`
  - `vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py:79`
  - `vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py:81`

### 4) Config contract mismatch

- Metal runner stores normalized config in dict:
  - `vllm_metal/v1/model_runner.py:465`
  - `vllm_metal/v1/model_runner.py:554`
- Metal worker currently still reads `cfg = model.config`:
  - `vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py:61`

### 5) Layer naming mismatch

- Metal KV cache spec names: `layers.{i}.self_attn`:
  - `vllm_metal/v1/model_runner.py:639`
- Hook regex expects names like `model.layers.{i}.self_attn.attn`:
  - `vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py:15`

## Missing Pieces Required for Full vllm-hook Functionality on Metal

### A) Replace PyTorch hook dependence with Metal-native interception

Required:
- Add a hook/callback manager in `MetalModelRunner` (or wrapper) that can:
  - Enable/disable capture per request/run ID
  - Capture Q/K tensors during prefill/decode
  - Capture/modify activations for steering at target layers

Rationale:
- MLX model objects do not provide PyTorch-style module hook APIs.

### B) Add a Metal forward context equivalent

Required:
- Implement a context object in Metal path containing at least:
  - per-request `seq_lens` (or equivalent token boundaries)
  - mapping from generated data to logical layer names
  - request/run identifiers needed by artifact writer
- Enter/exit this context around each model forward in `execute_model`.

Rationale:
- Existing hook logic depends on `get_forward_context().attn_metadata`.

### C) Build attention metadata in Metal runner (or compatible substitute)

Required:
- Construct metadata needed by analyzers and by Q/K chunk slicing logic.
- Ensure batch/token boundary semantics match analyzer expectations.

Rationale:
- GPU path relies on metadata built in `_build_attention_metadata(...)`; Metal path currently has none.

### D) Introduce a stable layer naming contract for Metal artifacts

Required:
- Choose one naming format and enforce it end-to-end:
  - either GPU-like names (`model.layers.{i}.self_attn.attn`)
  - or Metal names (`layers.{i}.self_attn`) + analyzer regex update

Rationale:
- Current names/patterns diverge and will silently miss layers.

### E) Correct Metal worker config access

Required:
- Replace `model.config` reads with `self.model_runner.model_args` in Metal workers.

Rationale:
- Current Metal worker uses a GPU-style config path that does not match runner contract.

### F) Add a first-class capture API instead of file-flag polling inside module hooks

Required:
- Move run-ID/hook-enable checks into runner-level capture path.
- Keep output artifact format backward-compatible with existing analyzers (`qk.pt` structure).

Rationale:
- Polling via filesystem inside per-layer forward hooks is a GPU-era implementation detail.

## Recommended Translation Targets (vLLM -> vLLM-Metal)

### 1) `vllm/forward_context.py` -> `vllm_metal/v1/forward_context_metal.py` (new)

Create a minimal Metal context model mirroring only fields needed by hook analyzers.

### 2) `vllm/v1/worker/gpu_model_runner.py::_build_attention_metadata(...)` -> Metal metadata builder (new)

Not a line-by-line port. Build only required metadata:

- `seq_lens`
- logical layer map
- request token partition data

### 3) Hooking surface translation

GPU:
- `named_modules()` + `register_forward_hook(...)`

Metal:
- explicit callback points inside `MetalModelRunner` prefill/decode code paths.

## Implementation Order (Minimal Path to Working Attention Tracker)

1. Implement runner-level Q/K capture callbacks in `vllm_metal/v1/model_runner.py`.
2. Add Metal forward-context object + context manager around model calls.
3. Write artifacts in existing `qk.pt` format expected by current analyzers.
4. Update `ProbeHookQKWorkerMetal` to use runner APIs; remove PyTorch hook assumptions.
5. Normalize layer names and update regex/analyzer if needed.
6. Apply same pattern for activation steering (`SteerHookActWorkerMetal`).

## Non-Blocking / Out-of-Scope for Hook Enablement

- Full parity for all vLLM features in the 1707-file tree.
- LoRA support on Metal (`vllm_metal/v1/worker.py:350` currently returns unsupported).
- Cascade/sparse/MLA attention support (`vllm_metal/platform.py:230`, `:252`, `:254`).

## Bottom Line

To make `vllm-hook` fully functional on `vllm-metal`, you do **not** need to re-implement all of vLLM.  
You need a Metal-native replacement for three GPU assumptions:

1. Module hooks,
2. Forward context + attention metadata plumbing,
3. Q/K and activation interception points.

These are integration-layer gaps between the current vLLM hook architecture and MLX-backed execution in `vllm-metal`.
