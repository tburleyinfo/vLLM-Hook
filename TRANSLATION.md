# vLLM-Metal translation: Attention Tracker

## Implemented

- Added the `metal` optional dependency profile in `pyproject.toml` alongside,
  not in place of, the existing CPU and CUDA profiles.
- Added a Metal Q/K observer integrated through the existing
  `capture_qk` worker. It wraps the MLX attention call used by vLLM-Metal's
  non-paged Mistral path, delegates generation to the original target call,
  captures post-RoPE Q and full cached K for requested layers, and stores them
  in the existing worker payload format. Request-scoped context prevents
  unrequested generation from being captured. Patched callables are restored
  when the worker shuts down.
- The existing `MiaLLM.generate()` / `MiaLLM.analyze()` APIs and
  `AttntrackerAnalyzer` are unchanged; capture only activates when requested by
  the existing worker extras. The CPU/CUDA QK installation path remains
  unchanged.
- Added a Metal-focused capture contract test and runnable
  `examples/demo_attntracker_metal.py` usage notes for the selected Mistral
  config in `docs/use_cases/attntracker_vllm_metal.md`.
- Added source-path findings in
  `translation_discovery/attention_tracker_metal_research.md`.

## Run

On Apple Silicon, install the Metal profile and run the focused example:

```bash
uv sync --extra metal --group test
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python examples/demo_attntracker_metal.py
```

The script selects
`model_configs/attention_tracker/Mistral-7B-Instruct-v0.3.json` and uses
`mistralai/Mistral-7B-Instruct-v0.3` by default. `MIA_DEMO_MODEL` and
`MIA_CONFIG_FILE` can override these defaults together. The old v0 notebook at
`notebooks/demo_attntracker.ipynb` is not a migrated entry point; the rest of
the notebook collection is outside this translation's scope.

## Checks performed

- Inspected the selected notebook, Mistral Attention Tracker config, current
  example/API, QK worker payload path, analyzer, and target vLLM-Metal worker,
  runner, model lifecycle, MLX-LM Llama attention, and hook correspondence
  sources.
- Added `tests/use_cases/test_attntracker_metal.py` to cover requested-layer
  capture, probe tensor shapes, output delegation, no-capture requests, and
  patch restoration.
- No tests, type checks, package resolution, model loading, or Apple Silicon
  generation were run: this session only provides file inspection and edit
  tools, not command execution.

## Remaining work

- Run the added test and the example on a supported Apple Silicon/vLLM-Metal
  environment; confirm worker-extension dispatch and end-to-end probe delivery
  against the installed target versions. The target library sources are
  reference-only and do not establish that this runtime path matches the
  inspected source.
- Validate the resolved `mlx` / `mlx-lm` versions with the declared
  `vllm-metal==0.2.0` and `vllm==0.29.0` dependencies, then add compatible
  version pins if needed.
- Extend Metal instrumentation for steering and other capture workers if those
  public APIs are in scope beyond Attention Tracker. This translation implements
  only the Q/K capture path required by the selected use case; no steering
  backend is claimed.
