# Attention Tracker migration discovery

## Source behavior slice

- `model_configs/attention_tracker/Mistral-7B-Instruct-v0.3.json` selects
  25 layer/head pairs and `hookq_mode=last_token`.
- `examples/demo_attntracker.py` and the selected Colab notebook render system
  and user messages using the tokenizer chat template, generate through
  `MiaLLM`, then call `analyze` with `input_range` and `sum_normalize`.
- `mia/llm.py` delegates generation to `vllm.LLM`, supplies `output_qk` and
  `hookq_mode` in sampling extra args, and attaches delivered probes.
- `mia/workers/qk_capture_worker.py` stores request-scoped Q/K, returns the
  existing compressed payload and removes request state after delivery.
- `mia/analyzers/attention_tracker_analyzer.py` reconstructs selected-head
  attention from flattened Q/K and emits `{"score": [...]}`.
- Source relationship slice: `target_relationships/source/files/mia/core/_plugin.py.json`,
  `.../mia/workers/qk_capture_worker.py.json`, and
  `.../mia/analyzers/attention_tracker_analyzer.py.json`.

## Target behavior slice

- `vllm_metal/v1/model_runner.py` uses MLX-loaded models and a non-paged MLX
  cache path when `VLLM_METAL_USE_PAGED_ATTENTION=0`; its `MetalModelRunner`
  performs prefill and decode through the model while retaining vLLM request
  and output handling.
- `mlx_lm/models/llama.py` (the Mistral architecture implementation) computes
  Q/K, applies `rope`, and updates/fetches keys from its MLX KV cache.
- `vllm_metal/attention/impls/sdpa_wrapper.py` delegates to the original
  attention when there is no paged-attention context. Its paged path instead
  prepares Q/K through `sdpa_forward`.
- `vllm_metal/pytorch_backend/tensor_bridge.py` and
  `platform_reference/notes/hook_primitives.md` document MLX/PyTorch array
  conversion; bfloat16 must be cast within MLX before NumPy conversion.
- Target relationship slice:
  `target_relationships/files/vllm_metal/v1/model_runner.py.json`,
  `.../vllm_metal/attention/impls/sdpa_wrapper.py.json`,
  `.../mlx_lm/models/llama.py.json`, and `.../mlx_lm/models/cache.py.json`.

## Adapter boundary

The new Q/K adapter is installed by the existing `capture_qk` worker when the
live runner belongs to `vllm_metal.v1`. It wraps Mistral's RoPE-based attention
callable, retains its original model computation and caches, and captures
post-RoPE Q/K in MIA's request bucket so the existing worker serialization,
delivery, and analyzer remain the public path. No custom generation loop is
introduced.

The delivered platform variables select the non-paged path. Current adapter
support is limited to that path and single-sequence request execution.
