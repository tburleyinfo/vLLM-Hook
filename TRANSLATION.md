# vLLM-Metal translation: Attention Tracker

## Implemented

- Added the `metal` optional dependency profile without changing the existing
  CPU/CUDA profiles, base pins, or installation paths.
- Added a Metal-specific Q/K observer for the MLX SDPA boundary. It captures
  actual Q/K values from vLLM-Metal model execution (not a separate MLX
  generation path), returns them in the existing `probes["config"]` and
  `probes["qk_cache"]` format, and attaches them to `MiaLLM.generate()` output.
- Kept analysis on the existing `AttntrackerAnalyzer` and `MiaLLM.analyze()`
  APIs.
- Added unit tests for the Q/K probe contract and runnable Attention Tracker
  usage notes in `examples/README_vllm_metal_attention.md`.

## Checks performed

- Added tests; they are runnable with
  `pytest -q tests/use_cases/test_vllm_metal_attention.py`.
- Tests were not executed in this environment.
- Inspected the repository's Q/K worker, analyzer, probe schema, example and
  supplied vLLM-Metal attention/model-runner sources.
- No Apple Silicon runtime or vLLM-Metal generation test was run in this
  environment.

## Remaining implementation work

- Verify worker-extension RPC installation and end-to-end capture against the
  declared vLLM-Metal runtime on Apple Silicon. In particular, verify real
  batching/prompt ordering and model lifecycle integration.
- Implement correct association/reconstruction for chunked prefill, since
  segmented prefill needs request identifiers and complete prefix keys.
- Integrate persistent (`save_to_disk`) delivery and the served `MiaClient`
  path with Metal worker lifecycle.
- Port Metal-compatible steering, hidden-state capture, and other worker
  modes; they remain on their existing CPU/CUDA implementations.
- Validate the shipped Granite prompt offsets, the selected Qwen2 configuration,
  analyzer scores, and public lifecycle behavior on actual Metal hardware.
