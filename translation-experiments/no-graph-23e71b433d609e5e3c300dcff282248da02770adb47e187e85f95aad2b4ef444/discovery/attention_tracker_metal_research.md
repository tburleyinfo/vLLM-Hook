# Attention Tracker / vLLM-Metal discovery

## Repository contract

- Selected config: `model_configs/attention_tracker/Mistral-7B-Instruct-v0.3.json`.
  It requests selected layers/heads and `hookq_mode=last_token`.
- `MiaLLM` passes the existing `capture_qk` worker class through vLLM's worker extension
  mechanism, sets per-request `SamplingParams.extra_args`, attaches `probes` to returned
  request outputs, and delegates scoring to `AttntrackerAnalyzer`.
- The analyzer accepts either raw `q`/`k_all` tensors or score tensors. Raw probes must
  include `config` fields `num_attention_heads`, `num_key_value_heads`, `head_dim` and
  `attention_multiplier`; each layer record uses the layer number, query tensor and
  full-key tensor.
- Existing CUDA capture operates on PyTorch/vLLM runner internals and CUDA graph operators;
  that path is not a suitable Metal implementation.

## Target source correspondence

- Supplied target source versions identify `vllm_metal.__version__` as 0.2.0 and
  `mlx_lm.__version__` as 0.31.3; repository vLLM remains pinned at 0.29.0.
- The supplied `vllm_metal.v1.model_runner.MetalModelRunner` performs vLLM v1 request
  scheduling, prefill/decode, and uses MLX model forwards.
- `ModelLifecycle` loads the model with `mlx_lm`, and the provided MLX Llama/Mistral
  attention computes projected Q/K, applies RoPE and updates its cache before calling
  `scaled_dot_product_attention`.
- The provided tensor bridge represents MLX↔PyTorch conversion via CPU-compatible buffers;
  platform notes require casting MLX bfloat16 arrays to float32 before NumPy conversion.
- vLLM-Metal's non-paged runner has `_prefill_single`; its attention path receives Q and
  full cached K at the scaled-dot-product-attention call. A scoped observer can therefore
  capture real target-model tensors without replacing vLLM-Metal generation.
- The selected Metal environment explicitly uses the non-paged MLX path and disables V1
  engine multiprocessing, which is needed by the worker extension's in-process capture
  state.

## Scope decisions / gaps

- Implemented the selected prompt-prefill raw-Q/K Attention Tracker path only.
- Did not port steering or hidden-state capture; those involve distinct insertion points and
  are not implied by the selected use case.
- Generation mode, model-family compatibility, multi-request interleaving, delivery
  lifecycle, and numerical behavior still require execution against the actual target
  libraries and Apple Silicon.
