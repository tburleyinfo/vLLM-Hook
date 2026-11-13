# Attention Tracker translation discovery

## Repository contract

- The notebook calls `MiaLLM(model, worker_name="capture_qk",
  analyzer_name="attn_tracker", config_file=...)`.
- `MiaLLM.generate` arms each request with Q/K capture and returns
  `RequestOutput` objects; for a multi-prompt call, `outputs[0].probes` is the
  aggregate probe payload.
- The selected model config requests `hookq_mode="last_token"` and specifies
  important attention heads on layers 10, 11, 12, 13, 14, 15, 17, 18 and 19.
- `AttntrackerAnalyzer` accepts `{"config": ..., "qk_cache": ...}` and
  calculates head attention from the last query and all keys. It then scores
  the configured instruction/data token ranges.
- The disk API stores/loads Q/K artifacts under `<hook_dir>/<run_id>/`.

## Target-library correspondence

- Supplied `vllm_metal.v1.model_lifecycle` loads MLX-LM models and assigns
  them to `MetalModelRunner.model`; the runner uses `mlx_lm` cache and
  forward APIs while vLLM owns scheduling and sampling.
- Supplied `vllm_metal.pytorch_backend.tensor_bridge` exports MLX arrays via
  an array boundary. Supplied platform notes require casting MLX bfloat16 to
  float32 before NumPy conversion.
- Supplied `mlx_lm.models.qwen2` computes Q/K, applies RoPE and cache updates,
  then calls its `scaled_dot_product_attention` function with post-RoPE Q/K.
  This is the capture seam used by the new module. The Q/K arrays are
  transposed/flattened to the same dimensions consumed by the existing
  analyzer.
- The target Qwen2 block forms `h = x + attention_output` and calls
  `post_attention_layernorm(h)` before its MLP. The steering bridge modifies
  this residual input before the original normalization call.
- The supplied runner's non-paged path includes `_prefill_single`, while its
  paged path packs multiple requests into unified tokens. The specified
  environment disables paged attention and multiprocessing, which is the
  intended target configuration for this bridge.

## Known gaps

- The target bridge only instruments `mlx_lm.models.qwen2` and single-request
  forward batches. It must be validated against exact target package versions
  and runner scheduling.
- Hidden-state hooks and models outside Qwen2 are not bridged.
- Runtime and artifact round-trip tests could not be run in this tool
  environment.
