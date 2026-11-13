# Attention Tracker Metal discovery

## Selected source behavior

- `examples/demo_attntracker.py` constructs `MiaLLM` with `worker_name="capture_qk"`,
  `analyzer_name="attn_tracker"`, and the Granite 3.1 8B config.
- Each test case is sent as a separate `generate` call, with the same prompt
  converted to the model's chat template and with `hookq_mode="last_token"`.
- `AttntrackerAnalyzer` consumes a `config` map plus per-layer Q/K entries from
  `qk_cache`; raw queries and keys are converted to per-head attention by the
  unchanged analyzer.
- The target runtime specifies `VLLM_METAL_USE_PAGED_ATTENTION=0`.

## Target observations

- `vllm_metal.v1.worker.MetalWorker` owns a `MetalModelRunner`; its model
  lifecycle loads the MLX model, not a PyTorch model exposing vLLM CUDA
  attention modules.
- The supplied Granite MLX model computes Q/K in `mlx_lm.models.granite`
  `Attention.__call__`; its imported `scaled_dot_product_attention` receives
  the rotated queries and keys. This is the capture seam for the selected
  non-paged path.
- The Metal paged path instead wraps layer attention in
  `SDPAPagedAttentionWrapper` and computes Q/K in
  `vllm_metal.attention.impls.sdpa.prepare_sdpa_qkv`; the provided environment
  disables that path.
- MIA's CUDA graph ops, Triton capture, worker mixins and GPU aperture are not
  compatible with the Metal MLX model lifecycle and are left unchanged.

## Implementation boundary

New `mia.backends.metal.attntracker` instruments Granite's imported SDPA
function and layer calls, captures the first call for each configured layer
during generation, and exports the existing analyzer payload. The inspected
non-paged runner performs new-request prefills sequentially, so the session
uses that ordering for input batches. Since capture is global to the current
worker process and request attribution is not available at the MLX attention
seam, the implementation does not claim correct concurrent, chunked, paged,
or server/client capture.
