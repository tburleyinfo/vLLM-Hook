# vLLM-Metal translation

## Implemented

- Added the `metal` optional dependency profile without changing the existing
  CPU/CUDA profiles, pins, or installation recipes.
- Added a scoped Q/K interception primitive for the paged vLLM-Metal SDPA
  path. It observes post-RoPE MLX Q/K arrays, converts them to the PyTorch
  tensor representation expected by the existing Q/K analyzer, and restores
  patched target callables after normal or exceptional exit.
- Kept generation in vLLM/vLLM-Metal; the adapter does not invoke standalone
  `mlx_lm` generation or change forward outputs.
- Added a Metal Attention Tracker usage note, a unit test for the existing
  analyzer probe contract and hook restoration, and discovery notes tracing
  the relevant source and target code.

## How to run

Install the optional target profile with `uv sync --extra metal --group test`.
For vLLM-Metal, configure the plugin with `VLLM_PLUGINS=metal`; see
`docs/metal_attntracker.md` for the target environment.

The Q/K adapter currently requires vLLM-Metal paged SDPA attention enabled
and generation in the current Python execution context. Therefore it is not
usable under the requested `VLLM_METAL_USE_PAGED_ATTENTION=0` environment, nor
is the notebook's existing
`MiaLLM(worker_name="capture_qk", analyzer_name="attn_tracker")` flow yet
connected end-to-end.

## Checks performed

- Inspected the selected notebook, model configuration, MIA public API,
  Q/K worker, analyzer and hook installation path.
- Inspected the supplied vLLM-Metal platform, worker, model lifecycle,
  model runner, paged attention wrapper, SDPA implementation and platform
  hook notes.
- Added but did not execute `tests/use_cases/test_metal_attention_capture.py`.
  This environment exposed file inspection/editing tools only; no test runner
  or target runtime command could be invoked here.

## Remaining implementation work

- Integrate with vLLM-Metal's worker/runner lifecycle so the existing
  `MiaLLM` and server APIs can arm, route, retrieve and clean up captures by
  request ID while preserving batching, chunked prefill, cached-prefix K
  reconstruction, cancellation, errors and artifact persistence.
- Implement Q/K observation on the requested non-paged Metal path; current
  adapter depends on a wrapper bypassed when `VLLM_METAL_USE_PAGED_ATTENTION=0`.
- Implement MIA's public activation-steering behavior on MLX layer residuals
  with the existing steering vectors, phase/position semantics and request
  lifecycle. No Metal steering implementation is provided by this change.
- Add and run target-runtime tests for generation, capture, analyzer parity,
  batching, lifecycle cleanup, and steering on Apple Silicon.
- Verify compatible vLLM-Metal package/version and dependency resolution on
  Apple Silicon; versions could not be established from the supplied source
  tree or runtime checks.
