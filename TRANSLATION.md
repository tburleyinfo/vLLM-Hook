# vLLM-Metal translation

## Scope and implementation status

This draft targets the Attention Tracker use case and its
`Qwen2-1.5B-Instruct.json` model configuration. The existing public analyzer
contract accepts `probes={"qk_cache": ..., "config": ...}` and returns one
`score` for each supplied input range. A new MLX Q/K conversion module is
implemented in `mia.backends.metal_qk`: it accepts post-RoPE Q/K arrays,
converts MLX bfloat16-compatible arrays to float32 before NumPy conversion,
and produces the existing probe structure using CPU torch tensors.

**Migration status: incomplete.** The Q/K primitive is not yet installed in
vLLM-Metal's model runner, and `MiaLLM.generate()` does not yet route
`capture_qk` requests to it. Do not treat capture generation or a completed
vLLM-Metal model process as evidence that Attention Tracker works. The current
CUDA worker, steering, hidden-state, graph capture, and delivery paths have
not been migrated to Metal. No existing CPU/CUDA implementation or dependency
profile was removed.

## Intended run

The intended environment and example command are documented in
[`docs/use_cases/attntracker_metal.md`](docs/use_cases/attntracker_metal.md).
The command is not runnable for MIA capture until the target-runner integration
listed below is complete. `MiaLLM`, `SamplingParams`, the configured model and
the analyzer remain the intended public entry points.

## Checks performed

- Inspected the repository's QK capture worker, graph custom-op registration,
  analyzer, `MiaLLM` generation/probe lifecycle and the selected model config.
- Inspected the supplied vLLM-Metal worker, model runner, model lifecycle,
  adapter, Qwen2 MLX attention implementation and attention patching code.
- Added focused unit tests for the new Q/K conversion and probe format. They
  have **not** been executed in this environment.
- No Apple Silicon runtime, model generation, end-to-end capture, analyzer
  score, behavioral parity, or lifecycle checks were performed.

## Remaining implementation work

1. Add a Metal-native worker/model-runner integration that observes Qwen2's
   post-RoPE query and full-prefix key arrays during actual vLLM-Metal
   `LLM.generate()` calls, while preserving request association, prompt
   boundaries, cleanup and concurrent-request behavior.
2. Wire the Metal capture to `MiaLLM` and attach probes using the existing
   output contract; validate both in-memory analysis and save-to-disk behavior.
3. Run the new unit tests and Apple-Silicon end-to-end tests, including
   lifecycle and output parity checks.
4. Implement and validate any other backend capabilities required by the
   existing public APIs (including steering) before claiming repository-wide
   Metal parity.
