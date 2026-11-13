# vLLM-Metal translation

## Implemented for the selected use case

- Added a Metal Q/K capture backend for the target library's `mlx_lm` Granite
  layout. It observes the actual post-RoPE query and key arrays in the loaded
  model and does not replace vLLM generation with standalone MLX generation.
- Routed the existing `capture_qk` worker's install, retrieval, and abort-cleanup
  APIs to this backend when its runner is vLLM-Metal. Captures use the existing
  `qk_cache` / `config` probe contract and therefore feed the existing
  `AttntrackerAnalyzer` and `MiaLLM.analyze()` path.
- Added `examples/demo_attntracker_metal.py` as runnable use-case notes/code.
  In a fresh Apple Silicon environment, install this package's Metal profile
  and run it from the repository root:

  ```sh
  pip install -e ".[metal]"
  python examples/demo_attntracker_metal.py
  ```

  It selects the Metal plugin, uses non-paged attention, disables vLLM V1
  multiprocessing as requested, generates through `MiaLLM`, then analyzes the
  captured probes before shutting down the engine. The configured Granite model
  may require substantial unified memory and a model download.
- Added a `metal` optional dependency profile using the target source versions
  while retaining the existing CPU/CUDA profiles and base dependency pins.
- Added a focused backend test for probe contract serialization and paged-path
  refusal.

## Checks performed

- Inspected the repository's Q/K worker, `MiaLLM` generation and analyzer,
  Granite Attention Tracker configuration, notebook flow, and relevant vLLM-Metal
  model runner, model adapter, worker, Granite `mlx_lm` model, and target hook
  correspondence reference.
- Did not execute tests or start a model: this environment provides file
  inspection/editing tools only, and the Apple Silicon runtime/dependencies are
  not available for verification here.

## Remaining implementation work

- Activation steering and hidden-state capture are not implemented for
  vLLM-Metal. The repository's existing steering/hidden-state workers depend on
  vLLM's PyTorch V2 runner and CUDA-oriented module hooks; no Metal-specific
  equivalent has been added.
- Metal Q/K currently supports the inspected Granite model layout on the
  non-paged execution path. It rejects paged attention, `save_to_disk`, score
  capture, and other model layouts rather than silently returning incomplete
  captures. Validate the target's MLX callable wrapping and request lifecycle on
  an Apple Silicon runtime, and extend/retest for batched/scheduled execution
  and disk artifact delivery before claiming broader API parity.
- Add runtime parity tests for request batching, prefill/decode mode selection,
  abort/preemption cleanup, and disk artifact delivery.
