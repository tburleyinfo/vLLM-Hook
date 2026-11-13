# Attention Tracker vLLM-Metal discovery

## Repository behavior slice

- Public entry: `MiaLLM` builds a vLLM `LLM`, applies per-request `output_qk`
  and `hookq_mode` extra arguments, returns probes, and invokes registered
  analyzers (`mia/llm.py`).
- Worker contract: `QKCaptureWorker` stores per-request, per-layer entries
  containing `q`, `k_all`, `layer_num`, and `hookq_mode`; its existing
  `get_captured_states` serializes those entries for the ordinary MIA delivery
  path (`mia/workers/qk_capture_worker.py`).
- Analysis: `AttntrackerAnalyzer` uses the captured Q/K, config head counts,
  head dimension and attention scale to reconstruct attention and compute the
  configured range score (`mia/analyzers/attention_tracker_analyzer.py`).
- Selected config: Granite 3.1 2B quantized W4A16; seven important heads;
  Q/K capture mode `last_token`
  (`model_configs/attention_tracker/granite-3.1-2b-instruct-quantized.w4a16.json`).

## Target source findings

- `vllm_metal.v1.model_runner.MetalModelRunner` performs vLLM request handling
  and delegates model forwards to its loaded MLX model. Non-paged execution
  calls `_prefill_single`, `_sequential_decode`, and `_batched_decode`; request
  IDs and each request's `SamplingParams` remain available at those boundaries.
- `mlx_lm.models.granite.Attention.__call__` computes Q/K, applies RoPE and
  updates the cache before invoking its module-level
  `scaled_dot_product_attention`. Observing that call provides post-RoPE Q and
  the full available K prefix without substituting model generation or changing
  the model's attention result.
- MLX arrays, not Torch tensors, drive model execution. Captured arrays are
  materialized with an MLX-side float32 cast before NumPy/Torch conversion,
  consistent with `platform_reference/notes/hook_primitives.md`.
- The target source versions identify vLLM-Metal 0.2.0 and mlx-lm 0.31.3;
  MLX's package version is not declared in the supplied source tree.

## Implemented target correspondence

The new `mia.backends.vllm_metal.attention_tracker` backend patches Granite's
attention observation point and scopes it to the request boundaries. It
records the existing MIA Q/K entry shape and honors layer selection,
`hooks_on`, `hookq_mode`, and save-to-disk routing. Existing public APIs and
the analyzer remain unchanged. The backend requires non-paged Metal attention,
as selected by `VLLM_METAL_USE_PAGED_ATTENTION=0`.

## Remaining work

- Run the focused contract tests and a live Apple Silicon generation against
  the selected quantized checkpoint; neither was executable in this
  read/edit-only discovery environment.
- Confirm target dependency resolver compatibility on the actual Apple Silicon
  Python environment; the supplied library source tree contains no MLX package
  version or packaging manifest.
- Implement and verify the separate activation-steering and hidden-state
  capture paths on vLLM-Metal before claiming broader worker parity.
- Implement and verify the optional precomputed `qk_capture="score"` format;
  this adapter currently accepts the selected raw-Q/K Attention Tracker mode.
