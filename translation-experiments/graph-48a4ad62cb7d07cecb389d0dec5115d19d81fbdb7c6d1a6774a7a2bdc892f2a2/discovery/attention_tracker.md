# Attention Tracker Metal discovery

## Selected repository path

`notebooks/demo_attntracker_colab.ipynb` configures `MiaLLM(worker_name="capture_qk", analyzer_name="attn_tracker")`, generates prompt completions, obtains `output[0].probes`, and calls `MiaLLM.analyze`. The selected model config requests last-token Q and the complete prompt K prefix for the listed important heads. The analyzer recomputes attention from Q/K using the model head counts and attention multiplier, then scores the configured instruction/data token ranges.

## Source contracts inspected

- `mia/llm.py`: preserves MIA's public generation arguments, puts `output_qk`, `hookq_mode`, and other request settings in `SamplingParams.extra_args`, attaches returned probes, and dispatches analyzer calls.
- `mia/workers/qk_capture_worker.py`: the established Q/K artifact schema is per request and attention layer, with query and full-prefix key tensors plus `layer_num` and `hookq_mode`.
- `mia/artifacts.py` and `mia/analyzers/attention_tracker_analyzer.py`: probe and analyzer tensor contracts used unchanged.
- `platform_reference/source/vllm_metal/v1/worker.py`, `v1/model_runner.py`: Metal uses its own vLLM v1 worker/runner; with paged attention disabled it calls the MLX model directly from `_prefill_single`, `_sequential_decode`, and `_batched_decode`. Request IDs and `SamplingParams` remain available there.
- `platform_reference/source/vllm_metal/platform.py`: the plugin selects `vllm_metal.v1.worker.MetalWorker`; the selected Metal runtime requires `VLLM_USE_V2_MODEL_RUNNER=0`. MIA's existing V2-only guard is now bypassed only for the Metal plugin paired with its Q/K worker, and the resolved runner validation permits that same exact combination.
- `platform_reference/source/mlx_lm/models/granite.py`: Granite attention computes Q/K projections, applies the same RoPE module to Q then K, and updates its cache before the native attention operation.
- `platform_reference/source/mlx_lm/models/cache.py`: `update_and_fetch` returns the full updated key prefix for the non-paged cache path.
- `platform_reference/source/vllm_metal/attention/impls/sdpa.py`, `attention/impls/sdpa_wrapper.py`, and `attention/patching.py`: paged Metal dispatch prepares post-RoPE Q/K separately; however the selected target environment explicitly disables that path.

## Implemented seam

New `mia.backends.metal.qk_capture` wraps compatible Granite attention calls without replacing generation. A request-scoped context is installed around the Metal runner's existing prefill/decode forwards. The backend observes post-RoPE Q/K and the updated KV prefix, converts only the requested probe arrays to CPU tensors, and stores them in the existing worker/artifact representation. The standard analyzer and request/probe delivery path are reused.

The selected path is intentionally limited to the requested non-paged Granite use case. CPU/CUDA worker paths and existing public MIA APIs are not replaced.
