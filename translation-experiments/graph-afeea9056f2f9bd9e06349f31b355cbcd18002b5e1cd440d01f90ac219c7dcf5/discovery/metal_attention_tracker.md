# Metal Attention Tracker discovery

## Scope traces

- Example entry point: `examples/demo_attntracker.py`; it creates `MiaLLM`
  with `worker_name="capture_qk"` and `analyzer_name="attn_tracker"`, calls
  `generate`, then analyzes `out[0].probes`.
- Selected config: `model_configs/attention_tracker/Qwen2-1.5B-Instruct.json`.
  It identifies `Qwen/Qwen2-1.5B-Instruct`, important heads, and
  `hookq_mode=last_token`.
- Probe analysis: `mia/analyzers/attention_tracker_analyzer.py` consumes
  per-layer Q/K and model configuration, computes attention, and sums or
  normalizes attention over the instruction/data ranges.
- Source capture: `QKCaptureWorker` in
  `mia/workers/qk_capture_worker.py` installs PyTorch module forward hooks
  after requiring the V2 runner; graph capture uses `torch.ops.mia` and CUDA
  dispatch in `mia/core/hooks/ops.py`.

## Target source facts

- The supplied target graph indexes `mlx_lm.models.qwen2.Attention.__call__`
  and records its lexical call to `mlx_lm.models.base.scaled_dot_product_attention`.
  The source graph indexes `QKCaptureWorker.install_hooks`, `_prepend_cached_keys`,
  and the Q/K marshalling functions as the capture/lifecycle surface.
- The target runner graph shows `MetalModelRunner.target_forward` and its model
  adapter delegation; the worker graph exposes `MetalWorker.execute_model`
  and `MetalWorker.sample_tokens`. The worker has no indexed MIA extension
  installer or PyTorch module-hook method.
- `vllm_metal.v1.worker.MetalWorker` owns a `MetalModelRunner`; its model
  lifecycle loads the MLX model through `mlx_lm.load`.
- `MetalModelRunner` uses an adapter's `target_forward` for ordinary model
  evaluation. The inspected runner model and target Qwen2 model are MLX
  modules/arrays, not PyTorch modules/tensors.
- Target `mlx_lm.models.qwen2.Attention.__call__` computes queries/keys,
  applies RoPE and then passes them to `scaled_dot_product_attention`.
  Therefore post-RoPE Q/K can be observed at the SDPA call boundary.
- The target Metal worker does not show the source CUDA worker's
  `model_runner.prepare_inputs`, PyTorch `register_forward_hook`, or
  `torch.ops.mia` graph-dispatch integration.
- With paged attention disabled as required by the environment, capture must
  also account for the target's non-paged MLX cache and generation lifecycle;
  source code inspection alone does not establish request-to-array association.

## Implemented primitive and remaining boundary

`mia.backends.metal_qk.MetalQKCapture` converts a single request's post-RoPE
arrays into the existing QK analyzer probe format. Conversion casts to MLX
float32 before `np.asarray`, following `platform_reference/notes/hook_primitives.md`.
It keeps the first per-layer Q/K prefix to match last-token prompt capture.
The helper is not yet connected to model calls or generation output delivery.
See `TRANSLATION.md` for remaining implementation work and actual-check status.
