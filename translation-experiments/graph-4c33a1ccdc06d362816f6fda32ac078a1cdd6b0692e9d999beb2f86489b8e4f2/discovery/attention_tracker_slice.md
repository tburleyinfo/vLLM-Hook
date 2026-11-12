# Attention Tracker migration discovery

## Repository-side feature slice

- Public generation/analysis API: `mia/llm.py` (`MiaLLM.generate`,
  `MiaLLM.analyze`); plugin registration: `mia/__init__.py`.
- Capture source/schema: `mia/workers/qk_capture_worker.py`,
  `mia/artifacts.py` (`unpack_qk`), and the probe collector in
  `mia/core/delivery/delivered_probes.py`.
- Consumer contract: `mia/analyzers/attention_tracker_analyzer.py` reads
  `config` plus per-layer `qk_cache` entries with `q`, `k_all`, and `layer_num`.
- Selected usage: `examples/demo_attntracker.py`; Qwen2 head selection:
  `model_configs/attention_tracker/Qwen2-1.5B-Instruct.json`.
- Existing CUDA integration is built around torch modules, CUDA graph custom
  ops, worker forward contexts, and a vLLM GPU Worker. Those assumptions do not
  describe the target MLX model path.

## Target-side feature slice

Supplied source references inspected:

- `platform_reference/source/vllm_metal/v1/model_runner.py` and
  `v1/model_lifecycle.py`: vLLM generation is routed through Metal's model
  runner and model lifecycle, with the loaded MLX model.
- `attention/impls/sdpa.py`: `prepare_sdpa_qkv` produces post-normalization,
  post-RoPE Q/K arrays; `scaled_dot_product_attention` is the ordinary MLX
  attention boundary.
- `attention/impls/sdpa_wrapper.py` and `attention/patching.py`: the paged
  path replaces model attention modules with wrappers. The target environment
  explicitly selects non-paged attention, so the repository integration
  observes ordinary attention module SDPA calls instead.
- `pytorch_backend/tensor_bridge.py`: `mlx_to_torch(..., device="cpu")`
  provides target-supported conversion for captured arrays.
- `attention/context.py`: paged request boundaries are represented by
  `cu_seqlens`, but the configured non-paged path does not expose this metadata
  to the ordinary model attention call.

## Implementation boundary

The adapter watches target attention `scaled_dot_product_attention` calls
while `MiaLLM.generate()` is active and leaves the target's original SDPA
function responsible for the actual computation. Output Q/K tensors use the
existing analyzer's CPU tensor contract. This slice does not establish full
request-to-probe correspondence for batched or chunked prefill, and worker
RPC/lifecycle behavior requires Apple Silicon runtime verification.
