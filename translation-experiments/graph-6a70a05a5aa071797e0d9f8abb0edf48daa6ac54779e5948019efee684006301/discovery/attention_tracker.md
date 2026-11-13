# Attention Tracker discovery

## Repository path

- `notebooks/demo_attntracker.ipynb` demonstrates MIA Q/K capture followed by
  `MiaLLM.analyze(..., probes=...)` and the `attn_tracker` analyzer.
- `model_configs/attention_tracker/Meta-Llama-3-8B-Instruct.json` selects
  important attention heads and requests last-token Q/K capture.
- Existing probe schema is `{"config": ..., "qk_cache": ...}`. Entries carry
  per-request Q tensors, all K rows, and `layer_num`; analyzer reconstruction
  uses configured attention dimensions and the important query heads.

## Target correspondence inspected

- vLLM-Metal's `attention/impls/sdpa.py` prepares post-normalization,
  post-RoPE query and key arrays.
- Its `attention/impls/sdpa_wrapper.py` dispatches this path only when paged
  attention is active.
- With the supplied `VLLM_METAL_USE_PAGED_ATTENTION=0` target setting,
  `mlx_lm.models.llama.Attention` passes post-RoPE Q and cache-resolved K into
  `scaled_dot_product_attention`.
- `vllm_metal.v1.model_runner.MetalModelRunner` performs single-request
  prefill and sequential or batched decode in-process when
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`. Those boundaries provide the request IDs
  needed to preserve the standard per-request probe shape.
- `platform_reference/notes/hook_primitives.md` describes callable-module
  wrapping as a possible correspondence, but the implementation instead wraps
  the specific existing Llama SDPA and runner call boundaries.

## Scope decisions

The translation implements the attention-tracker Q/K path needed by this
configuration, rather than translating the notebook collection. Existing
attention-tracker analysis remains the source of scoring behavior. Other MIA
workers and paged/server capture are intentionally not claimed as implemented.
