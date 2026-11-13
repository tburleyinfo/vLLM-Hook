# vLLM-Metal translation: Attention Tracker

## Implemented

- Added a vLLM-Metal worker integration for the existing `MiaLLM` public API. It
  retains real vLLM generation via `MetalWorker` and `MetalModelRunner`; there
  is no standalone MLX generation path.
- For the non-paged Metal runner, wraps the loaded MLX model's split-projection
  attention calls without replacing their return values. Prompt Q/K are
  recomputed from the same projections and RoPE, converted from MLX arrays via
  an MLX-side float32 cast, and routed by request to the existing `qk_cache`
  probe contract. Request capture, delivery, cleanup, `save_to_disk`, and
  `AttntrackerAnalyzer` use the repository's existing APIs.
- Added selector/shape tests and a runnable Granite 3.1 2B Attention Tracker
  example at `examples/demo_attntracker_metal.py`.
- Added the `metal` optional dependency profile; existing CPU/CUDA pins and
  installation profiles are unchanged.

## Run on Apple Silicon

Install the project's target runtime and MIA in an Apple Silicon Python
environment using a Python version supported by both vLLM-Metal and MIA
(MIA requires Python 3.11-3.14), including the `metal` extra:

```sh
python -m pip install -e ".[metal]"
```

Before starting Python, select the Metal plugin and the use-case's non-paged
path:

```sh
export VLLM_PLUGINS=metal
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
python examples/demo_attntracker_metal.py
```

The script loads
`RedHatAI/granite-3.1-2b-instruct-quantized.w4a16` and
`model_configs/attention_tracker/granite-3.1-2b-instruct-quantized.w4a16.json`,
generates the notebook's benign and injection-style examples, and analyzes the
captured prompt Q/K with the selected important heads. Model availability,
checkpoint access, and sufficient unified memory are required.

## Checks performed

- Inspected the notebook's selected model, Q/K mode, important heads, input
  ranges, generation, and analysis calls.
- Inspected the declared vLLM-Metal worker, model lifecycle, Granite MLX model,
  attention projections/RoPE, non-paged prefill flow, and AWQ loader.
- Added unit tests for requested-layer selection and the Q/K probe tensor
  shapes. Tests and a real Apple Silicon generation run were not executed in
  this environment.

## Remaining work

- Implement Metal hidden-state capture and activation steering through their
  existing public APIs; this translation currently wires only the Q/K
  Attention Tracker path.
- Add Metal capture for paged attention and decode/both `hooks_on` modes. The
  implemented Q/K capture path explicitly requires prefill-only capture with
  `VLLM_METAL_USE_PAGED_ATTENTION=0`.
- Run the new tests and the example on Apple Silicon with the declared
  vLLM-Metal release and verify the outputs and lifecycle end to end.
- Confirm target dependency distribution names and compatible versions; the
  supplied target source tree does not include package metadata, so the new
  `metal` extra is not version-pinned or installation-verified.
- Confirm target dependency distribution names and compatible versions; the
  supplied target source tree does not include package metadata, so the new
  `metal` extra is not version-pinned or installation-verified.
