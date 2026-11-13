# vLLM-Metal translation: Attention Tracker

## Implemented

- Added an opt-in `metal` dependency profile (`vllm-metal`) alongside the existing CPU and CUDA profiles. The existing pins and installation paths are unchanged.
- Added `mia.backends.metal.capture_qk` / `MetalQKCapture`. When vLLM-Metal paged SDPA is enabled, it observes the target SDPA `prepare_sdpa_qkv` seam, captures post-RoPE Q/K arrays, exports MLX bfloat16 safely through float32, and returns the existing analyzer's `{"config": ..., "qk_cache": ...}` probe contract.
- Added a temporary MLX decoder-layer steering adapter using the existing `vector_path`, `method`, `coefficient`, and `optimal_layer` configuration fields. It restores wrapped layer classes and instance markers when closed.
- Added focused steering lifecycle tests. Existing public MIA APIs and the CPU/CUDA implementations are unchanged.

## Selected model and how to run

The selected model config sets `hookq_mode` to `last_token`, which the new Q/K adapter accepts. However, the supplied target environment sets `VLLM_METAL_USE_PAGED_ATTENTION=0`; that selects the non-paged MLX attention path, which does not call the adapter's current SDPA capture seam. The adapter fails explicitly in this environment rather than returning an empty capture. A direct invocation for a future paged-path integration would have this form:

```python
from mia.backends.metal import capture_qk

with capture_qk(metal_model_runner.model, hookq_mode="last_token") as capture:
    output = vllm_llm.generate(prompt, sampling_params)
probes = capture.probes()
scores = mia_llm.analyze(
    analyzer_spec={"input_range": input_range, "attn_func": "sum_normalize"},
    probes=probes,
)
```

For the target process settings, launch with:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python your_attention_tracker_driver.py
```

This is **not yet an end-to-end runnable Attention Tracker path**: the target environment uses the non-paged execution path; this repository does not have a supported capture seam for that path, a way to retrieve the Metal worker's loaded MLX model in `MiaLLM`, a way to enter capture around worker execution, or request-to-probe correlation. The snippet documents a future paged-path adapter boundary, not a runnable driver. The current notebook's `MiaLLM.generate()` therefore remains CPU/CUDA-only for Q/K capture. The `MiaClient`/`vllm serve` path is not connected either.

## Remaining work

- Integrate the MLX adapter into the vLLM-Metal worker and `MiaLLM`/`MiaClient` request lifecycles, preserving enable/disable, batching, cancellation, per-request probes, and disk delivery behavior. In particular, collect only the request's prefill Q/K for this config and attach the resulting probes to the corresponding existing `RequestOutput`.
- Steering is currently an explicit model-scoped helper, not a MIA worker/API integration. It only supports `phase="both"` and `positions="all_tokens"` and cannot route distinct steering configurations to requests in a batch. Implement routing through Metal's request metadata before claiming existing steering behavior is preserved.
- Implement full `all_tokens` Q/K capture, including reconstructing cached prefix keys, and test chunked prefill, decode, multi-request batches, and cleanup. The present adapter intentionally rejects `all_tokens` rather than emit incomplete key histories.
- Implement capture on the selected non-paged MLX attention path, or establish an allowed target setting with paged SDPA that preserves the requested behavior.
- Add real Apple Silicon integration tests using vLLM generation and the selected Qwen2 model; verify scores, request isolation, unchanged generated output, and lifecycle cleanup. The repository's supplied target source does not declare a compatible vLLM-Metal version or an installation recipe, so the new dependency profile is unpinned pending that compatibility declaration.

## Checks performed

- Inspected the Attention Tracker notebook, selected model config, existing Q/K worker/analyzer/API, and the declared vLLM-Metal SDPA wrapper, Q/K preparation, model walker, and worker/model-runner lifecycle sources.
- Added `tests/use_cases/test_attntracker_metal.py`; tests were not run because this environment exposes no command-execution tool and no Apple Silicon runtime was supplied.
- No model generation or runtime correctness validation was performed.
