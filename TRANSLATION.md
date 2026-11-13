# vLLM-Metal translation

## Implemented capabilities

- Added an opt-in `metal` dependency profile alongside the existing CPU and
  CUDA profiles.
- Added a vLLM-Metal worker adapter for the non-paged MLX path. Q/K capture
  wraps the model's callable attention modules, captures request-scoped
  post-RoPE queries and keys, and uses the existing MIA worker result,
  analyzer, and disk artifact contracts.
- Added Metal activation steering at callable decoder-layer boundaries with
  the existing `add_vector` and `adjust_rs` modes and phase/position
  selection.
- Made the analyzer select the final query row when callers provide
  all-token Q captures; Attention Tracker remains a last-query analysis and
  previously could not reshape an all-token query matrix.
- Preserved the existing `MiaLLM`, worker and analyzer public interfaces and
  left CPU/CUDA implementations in place.
- Added runnable Attention Tracker instructions for
  `notebooks/demo_attntracker_colab.ipynb`'s selected model configuration in
  [docs/use_cases/attention_tracker_metal.md](docs/use_cases/attention_tracker_metal.md).

## Checks performed

- Inspected the selected Granite model configuration and notebook use-case
  calls, repository worker/analyzer/probe-delivery code, supplied relationship
  slices, and target vLLM-Metal's Granite, model runner, worker, attention
  wrapper, and model lifecycle sources.
- Added unit tests for MLX-to-CPU Q/K conversion, existing probe-contract
  analysis, separation of in-memory versus disk capture buckets, and Metal
  steering selection/modes.
- The Metal extra pins `mlx-lm` to the inspected target source version 0.31.3;
  MLX itself remains unpinned because no MLX package metadata was supplied.
- No test suite or Apple Silicon runtime was executed in this environment.

## Remaining work

- Run the targeted tests and Attention Tracker demo on an Apple Silicon host
  with the declared vLLM-Metal dependencies; verify model loading, generated
  output, per-request probe delivery and cleanup.
- Exercise the new Metal steering adapter against an actual supported model
  and steering vector, including both batched and sequential decode.
- The adapter currently targets the target library's non-paged model path;
  Q/K and steering capture with vLLM-Metal paged attention remains to be
  implemented and tested.
