# vLLM-Metal translation

## Implemented capabilities

- Added an Apple Silicon `metal` dependency profile alongside the existing CPU
  and CUDA profiles. The target source versions found were vLLM-Metal 0.2.0
  and mlx-lm 0.31.3; MLX itself had no declared version in the supplied source.
- Added a Metal Q/K adapter behind the existing `capture_qk` worker. It observes
  post-RoPE Granite Q/K at MLX SDPA, associates captures with vLLM-Metal request
  IDs, and stores them in MIA's existing per-request cache contract for the
  unchanged probe delivery and Attention Tracker analysis APIs.
- Added focused contract tests and a runnable selected-use-case script, with
  setup instructions in `docs/use_cases/attntracker_vllm_metal.md`.
- vLLM-Metal generation remains the vLLM serving path; standalone MLX generation
  is not used.

## How to run

On Apple Silicon, install `python -m pip install -e '.[metal]'`, then follow
`docs/use_cases/attntracker_vllm_metal.md` to set the required runtime variables
and run `python examples/demo_attntracker_metal.py`. The script selects
`model_configs/attention_tracker/granite-3.1-2b-instruct-quantized.w4a16.json`.

## Checks performed

- Inspected the repository's Attention Tracker config, notebook, demo, worker,
  analyzer, MIA request/delivery APIs and dependency profiles.
- Inspected the supplied vLLM-Metal model runner, worker, model adapter and
  lifecycle sources, plus mlx-lm's Granite attention implementation and
  platform notes.
- Added tests for per-request Q/K cache shape, last-token selection, and
  layer/phase filtering. Tests, dependency resolution, and live generation were
  not run: this session exposes read/search/edit tools only and has no command
  runner or Apple Silicon runtime.

## Remaining implementation work

- Run the focused tests and end-to-end Attention Tracker example on Apple
  Silicon, including model loading, probe delivery and score calculation.
- Verify the added dependency profile resolves with the actual supported MLX,
  PyTorch and vLLM-Metal wheels.
- Port and test activation steering and hidden-state capture on vLLM-Metal.
  Those existing worker behaviors are not implemented by this selected
  Attention Tracker migration.
- The new adapter is scoped to the inspected Granite model with non-paged
  attention (`VLLM_METAL_USE_PAGED_ATTENTION=0`). Other MLX model families and
  Metal's paged-attention path remain to be implemented and validated. Raw Q/K
  capture is implemented; the optional `qk_capture="score"` representation
  remains to be implemented for this backend.
