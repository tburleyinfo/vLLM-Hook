# Attention Tracker / vLLM-Metal discovery

## Selected path

- `notebooks/demo_attntracker.ipynb` uses `MiaLLM.generate`, request extra
  arguments for Q/K capture, `RequestOutput.probes`, and
  `MiaLLM.analyze`.
- `model_configs/attention_tracker/Mistral-7B-Instruct-v0.3.json` selects
  the model's 25 important `(layer, head)` pairs and `hookq_mode=last_token`.
- `mia/workers/qk_capture_worker.py` establishes the existing payload contract:
  a `qk_cache` keyed by attention module name, `layer_num`, query rows, full
  key prefixes, and the model's head dimensions / attention multiplier.
  Its score-capture path also defines the per-head score payload retained by
  the Metal adapter.
- `mia/analyzers/attention_tracker_analyzer.py` consumes that payload to build
  head attention and returns `{"score": [...]}`. No analyzer changes are needed.

## Target source findings

- The supplied `vllm_metal/v1/model_runner.py` loads and executes `mlx_lm`
  models. The non-paged path uses the MLX model's own prompt KV cache and direct
  model calls; the selected environment explicitly sets
  `VLLM_METAL_USE_PAGED_ATTENTION=0`.
- The supplied `mlx_lm/models/llama.py` Mistral-compatible `Attention.__call__`
  computes Q/K projections, applies RoPE and updates cache before SDPA. It does
  not return Q/K. Its cache exposes the cached keys and offset.
- In the paged path, `vllm_metal/attention/impls/sdpa.py` has a much cleaner
  post-RoPE Q/K boundary in `prepare_sdpa_qkv`; the provided environment does
  not select that path.
- `vllm_metal/pytorch_backend/tensor_bridge.py` and
  `platform_reference/notes/hook_primitives.md` document MLX/PyTorch bridges;
  MIA's CPU analyzer consumes CPU PyTorch tensors, so the new capture adapter
  exports MLX Q/K through float32 NumPy arrays before creating torch tensors.
- Target library source exposes model/load/attention lifecycle but does not
  declare package versions or a dependency installation recipe. Accordingly,
  the new optional `metal` extra is unpinned and remains separate from CPU/CUDA.

## Scope decisions

The new Q/K adapter is limited to the selected non-paged Mistral Attention
Tracker flow. It recomputes Q/K projections solely for instrumentation, rotates
them using the model's own RoPE, includes prior cache keys, and stores them in
the current QK worker payload format. Generation still goes through the vLLM
LLM API and vLLM-Metal; it does not call `mlx_lm.generate` or standalone MLX
generation. The adapter does not change model parameters, cache contents,
generation return types, analyzer contracts, or the worker's retrieval method.

It does not implement hidden-state capture or activation steering for vLLM-Metal,
the vLLM-Metal paged-attention capture path, or general MLX architectures. Those
are remaining work, not treated as unsupported-success paths.
