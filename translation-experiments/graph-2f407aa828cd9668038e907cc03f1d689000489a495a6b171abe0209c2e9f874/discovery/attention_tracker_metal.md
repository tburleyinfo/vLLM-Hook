# Attention Tracker / vLLM-Metal discovery

## Source behavior slice

- `examples/demo_attntracker.py` exercises `MiaLLM.generate(...)`, consumes
  `RequestOutput.probes`, and calls `MiaLLM.analyze(...)`.
- `mia/workers/qk_capture_worker.py` owns request-scoped Q/K capture and
  retrieval through `get_captured_states`.
- `mia/core/delivery/delivered_probes.py:qk_probes` shapes captured Q/K into the
  existing probe contract.
- `mia/analyzers/attention_tracker_analyzer.py` unpacks Q/K, repeats grouped
  key heads as needed, applies the configured attention scale and softmax, and
  emits `{"score": [...]}`.
- `model_configs/attention_tracker/Qwen2-1.5B-Instruct.json` selects layers
  10-19 and `hookq_mode: last_token`. The prompt-range recipe in the example
  depends on the Qwen2 chat template offsets.

## Target source facts

- `vllm_metal/v1/model_lifecycle.py` loads generation models with `mlx_lm` or
  `mlx_vlm`; model parameters and attention computation are MLX-owned.
- `vllm_metal/v1/worker.py` selects `MetalModelRunner`, not vLLM's normal CUDA
  worker runner.
- The supplied `mlx_lm/models/qwen2.py` attention exposes `q_proj`, `k_proj`,
  `rope`, `n_heads`, and `n_kv_heads`. The supplied Metal SDPA wrapper uses
  callable-module replacement and delegates to the original attention when
  there is no paged-attention context.
- In the supplied `MetalModelRunner`, non-paged `_prefill_single` has the
  request id, token ids, and sampling parameters at the point it calls the
  loaded model. This is the request-scoping seam used by
  `mia/backends/metal.py`.
- `platform_reference/notes/hook_primitives.md` establishes callable-module
  replacement as a possible correspondence and requires MLX-side dtype casts
  before NumPy conversion. The bridge follows that cast rule when creating
  CPU Torch probe tensors.

## Implemented slice and boundaries

The bridge wraps each standard MLX decoder `self_attn`, computes its post-RoPE
Q/K during a capture-enabled prefill, converts captures to the existing Torch
artifact shape, and delegates the actual attention call unchanged. The model
generation path itself remains vLLM-Metal. The implementation is enabled by
the existing worker's `install_hooks` RPC and feeds the existing QK worker
retrieval path.

The selected target environment disables paged attention and the selected
model's prompt capture is prefill-only. Paged attention, decode behavior,
durable capture, and runtime validation require further work. The separate
steering bridge applies the repository's existing vector/method/layer/phase/
position settings to MLX decoder outputs on the non-paged runner. The supplied
target sources do not provide a declared package version or install recipe, so
no target release pin is inferred here.
