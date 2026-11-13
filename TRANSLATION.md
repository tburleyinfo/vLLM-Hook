# vLLM-Metal translation

## Implemented

- Added the optional `metal` dependency profile alongside the existing CPU and
  CUDA profiles.
- Added a vLLM-Metal worker hook that instruments the loaded MLX Granite
  Attention module without replacing its forward result.
- Added in-memory Q/K capture in the same `config` / `qk_cache` probe contract
  consumed by MIA's existing Attention Tracker analyzer.
- Added offline `save_to_disk=True` output in MIA's existing per-rank
  `qk.pt` artifact layout.
- Wired the selected `MiaLLM(worker_name="capture_qk")` path to the vLLM-Metal
  worker and capture session. Actual model generation remains
  `vllm.LLM.generate`. Requests in a batch are attributed by the sequential
  non-paged prefill order of the supplied Metal runner implementation.
- Added a focused probe-contract test and runnable Granite example notes in
  `docs/use_cases/attntracker_metal.md`.

## Checks performed

- Inspected the repository's `MiaLLM`, Q/K worker, analyzer and selected Granite
  config, plus the supplied vLLM-Metal worker, model runner, Granite model and
  attention implementation sources.
- Added unit tests for MLX-to-CPU Q/K conversion and the established probe
  structure. Tests were not executed in this environment.
- No vLLM-Metal model process or Apple Silicon runtime check was performed.

## Remaining implementation work

- Verify plugin registration and worker RPC integration against the installed
  vLLM-Metal version on Apple Silicon; validate real generation and Attention
  Tracker scores end to end.
- Support exact per-request attribution for concurrent and chunked-prefill
  requests and server/client capture; validate repeated disk-run lifecycle.
- Implement hidden-state capture and activation steering on the Metal model
  path while preserving their existing public APIs.
- Confirm target dependency versions and establish a tested installation
  profile. Supplied target sources do not declare distribution versions.
- Add behavioral/lifecycle tests on real Granite generation and test cleanup
  and repeated engine lifecycles.

The migration is incomplete; unimplemented behavior is not treated as a
successful no-op or as an unsupported-operation implementation.
