# vLLM-Metal translation

## Implemented

- Added a separate `metal` optional-dependency profile for `mlx`, `mlx-lm`, and
  `vllm-metal`; existing CPU/CUDA extras and dependency pins are unchanged.
- Added an MLX Mistral Q/K capture adapter, connected to the existing
  `capture_qk` worker installation path. It reads request capture flags and
  capture modes, captures model RoPE’d queries and full cached key prefixes,
  retains `save_to_disk` routing, supports the existing per-head score mode,
  and emits the same `qk_cache` data expected by existing `MiaLLM` delivery
  and `AttntrackerAnalyzer`.
- Added MLX float32-export, request/layer routing, raw-QK shape, and per-head
  score-contract tests.
- Added a runnable selected-use-case script: `examples/demo_attntracker_metal.py`.
- Added source-derived notes at `translation_discovery/metal_attention_tracker.md`.

The adapter is scoped to the supplied Mistral model on the non-paged MLX path.
The instrumented path evaluates the Mistral projections a second time to see
the internal Q/K values, but delegates the original attention call unchanged.
Inference remains real vLLM generation through vLLM-Metal.

## Run

On Apple Silicon, provision this project with its Metal profile and the
vLLM-Metal source/package version matching your vLLM installation:

```sh
pip install -e '.[metal]'
VLLM_PLUGINS=metal \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
MIA_ALLOW_CUDAGRAPH=0 \
python examples/demo_attntracker_metal.py
```

The script uses the repository's
`model_configs/attention_tracker/Mistral-7B-Instruct-v0.3.json`. Override
`MIA_DEMO_MODEL` or `MIA_CONFIG_FILE` to select another compatible model/config.
Model download access and enough unified memory for Mistral-7B are required.

## Checks actually performed

- Inspected the notebook, model config, existing QK worker, analyzer, probe
  delivery, and declared vLLM-Metal / MLX source files and platform note.
- Added focused tests for capture request/layer filtering, existing raw-QK and
  score payload shapes, and MLX bfloat16, float16 and float32 conversion.
- Tests and a real Apple Silicon inference run were **not executed** in this
  environment. Runtime behavior, target package compatibility and generated
  scores have not been verified.

## Remaining implementation work

- Add vLLM-Metal activation-steering and hidden-state capture adapters through
  the existing worker/public APIs.
- Implement and verify the paged-attention capture path and broader MLX model
  coverage.
- Run the new tests and end-to-end Attention Tracker on Apple Silicon using the
  target vLLM-Metal environment; verify capture delivery, lifecycle cleanup,
  analysis output and numerical behavior.
- Establish compatible, declared vLLM-Metal/MLX dependency versions when the
  target library publishes or supplies version metadata.
