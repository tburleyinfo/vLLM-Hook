# Attention Tracker / vLLM-Metal discovery

## Repository path

- `notebooks/demo_attntracker.ipynb` is an old v0 notebook and does not use this
  checkout's current public interface.
- `examples/demo_attntracker.py` exercises `MiaLLM.generate()` with
  `worker_name="capture_qk"`, `analyzer_name="attn_tracker"`, and passes
  `out[0].probes` to `MiaLLM.analyze()`.
- `model_configs/attention_tracker/Mistral-7B-Instruct-v0.3.json` specifies
  important attention heads and `hookq_mode="last_token"`.
- `mia/workers/qk_capture_worker.py` owns eager-QK probe serialization and
  `mia/analyzers/attention_tracker_analyzer.py` computes the score from those
  probes. `mia/llm.py` preserves their public lifecycle and request-extra-args
  contract.
- Existing hook installation assumes a vLLM V2 PyTorch model runner and
  PyTorch/CUDA attention modules. `mia/core/runner.py` confirms the runner
  restriction; `mia/core/hooks/ops.py` registers CUDA-dispatch custom operators.

## Target source evidence

- `platform_reference/source/vllm_metal/v1/worker.py` constructs a
  `MetalModelRunner`; `model_lifecycle.py` loads the generation model through
  `mlx_lm.load`.
- `platform_reference/source/vllm_metal/v1/model_runner.py` runs actual model
  forward calls through its loaded model and MLX prompt cache. With the supplied
  environment's paged-attention flag off, it does not use vLLM's PyTorch Q/K
  modules.
- `platform_reference/source/mlx_lm/models/llama.py` implements Mistral-style
  split projections, RoPE, cache update, then calls the imported
  `scaled_dot_product_attention(queries, keys, ...)`. That call receives the
  post-RoPE query and full cached key sequence required by the existing
  analyzer. `platform_reference/source/mlx_lm/utils.py` maps the `mistral`
  model type to the Llama implementation. The actual model-loader resolution
  for the selected checkpoint still requires runtime confirmation.
- `platform_reference/source/vllm_metal/attention/patching.py` demonstrates
  callable-module replacement and its requirement to preserve delegation.
  The new Metal observer delegates the target SDPA call unchanged and restores
  patched callables on worker shutdown.
- `platform_reference/notes/hook_primitives.md` describes this general
  replacement correspondence; it is not itself evidence that an arbitrary
  internal tensor is observable. The SDPA call-site is the application-specific
  observation point identified from target source.
- Supplied source and target relationship graphs describe static syntax only;
  they do not establish runtime dispatch or equivalence. The implementation
  follows the direct repository path above and target source call sites rather
  than treating graph relationships as runtime proof. The per-file graph
  indexes were queried for the repository's `install_hooks`/`generate` symbols
  and the target runner/attention symbols to locate this static source slice.

## Implemented slice

`mia/core/hooks/metal_qk.py` adapts the existing Q/K worker's in-memory and
disk-state payload shape. It observes only requested layers/requests, captures
post-RoPE Q and full K, and lets the existing `get_captured_states`,
probe attachment, `MiaLLM.analyze`, and `AttntrackerAnalyzer` process the
payload. The integration dispatches to this module only when the worker runner
comes from vLLM-Metal; CPU/CUDA installation paths remain unchanged.

`examples/demo_attntracker_metal.py` uses the selected Mistral config and
existing public API. The existing notebook collection remains out of scope.

## Not validated here

The candidate environment exposes file inspection/editing tools but no command
execution tool. The added test module and example have therefore not been
executed, and no Apple Silicon runtime or model generation result has been
verified. The generated discovery result records this remaining verification
work.
