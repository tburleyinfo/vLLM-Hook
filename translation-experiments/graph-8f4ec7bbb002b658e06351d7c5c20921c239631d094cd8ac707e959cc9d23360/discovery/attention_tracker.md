# Attention Tracker migration discovery

## Selected source behavior

- `notebooks/demo_attntracker_colab.ipynb` exercises `MiaLLM`, Q/K capture and
  the `attn_tracker` analyzer. The runnable repository counterpart is
  `examples/demo_attntracker.py`.
- The selected model config is
  `model_configs/attention_tracker/granite-3.1-8b-instruct.json`. It chooses
  41 layer/head pairs and asks for `hookq_mode: last_token`.
- `MiaLLM.generate()` forwards generation to `vllm.LLM.generate()`, attaches
  `output_qk`/`hookq_mode` using existing `SamplingParams.extra_args`, and
  returns probes on `RequestOutput`. `MiaLLM.analyze(probes=...)` calls
  `AttntrackerAnalyzer.analyze()` and preserves the `{"score": [...]}` result
  contract.
- `QKCaptureWorker` uses `install_hooks`, request-scoped `_captured_states`,
  `get_captured_states`, `clear_captured_states` and `flush_disk`; the Metal
  adapter integrates those existing lifecycle points rather than changing
  the public API.

## Target source contracts inspected

- `vllm_metal.v1.model_lifecycle.ModelLifecycle` loads the target model through
  `mlx_lm.load`.
- Supplied `mlx_lm.models.granite.Attention` calculates Q/K, applies RoPE and
  updates the cache inside its callable method. Its layer is available via
  `model.model.layers[i].self_attn`.
- `vllm_metal.v1.model_runner.MetalModelRunner` uses `_prefill_single` and
  the non-paged decode path when paged attention is disabled. The prefill
  method carries both the vLLM request ID and its `SamplingParams`.
- `SDPAPagedAttentionWrapper` and `walk_and_wrap` establish the callable
  module wrapping pattern; the target's model/cache operations use MLX.
- `vllm_metal.__init__` exposes plugin registration and macOS spawn defaults.
  The selected environment explicitly disables paged attention and v1
  multiprocessing.

## Chosen implementation and boundaries

- Q/K probes are captured after original attention executes, using the
  Granite attention projections/RoPE and the updated per-request cache. This
  does not alter model outputs or cache lifecycle.
- MLX arrays are evaluated and converted to float32 before NumPy/PyTorch
  conversion; this follows the supplied array-interoperability note.
- Captures are stored in the repository's pre-existing eager worker format,
  preserving response and disk artifact handling.
- Metal steering wraps decoder layer callables and mirrors the existing
  worker's `add_vector`, `adjust_rs`, phase and position behavior.
- The target use case sets `VLLM_METAL_USE_PAGED_ATTENTION=0`; paged-path
  capture/steering and actual native runtime verification remain work items.
