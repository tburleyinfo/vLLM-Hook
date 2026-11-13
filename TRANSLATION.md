# vLLM-Metal translation

## Implemented

- Added an optional `metal` dependency profile while keeping the existing CPU/CUDA profiles and their pins intact.
- Added a vLLM-Metal Q/K capture worker selected through the existing `capture_qk` worker registration when `VLLM_PLUGINS=metal`.
- For non-paged Mistral-family inference, the worker observes the real MLX queries and cached keys at the model's scaled-dot-product-attention call. Conversion to the existing MIA torch probe contract happens after capture; generation remains vLLM-Metal generation.
- The existing `MiaLLM.generate()` and `MiaLLM.analyze()` APIs and Attention Tracker analyzer are used without changing their signatures. A Metal-specific runnable demo and unit tests cover the probe shape and analyzer score calculation.
- MIA's CUDA-graph selection is disabled for the Metal profile without forcing vLLM-Metal's own eager mode.

## Run the selected use case

Use an Apple Silicon Mac with a compatible vLLM-Metal/vLLM environment, model access for
`mistralai/Mistral-7B-Instruct-v0.3`, and this repository installed with its Metal profile.
The profile pins the supplied target source versions (`vllm-metal` 0.2.0 and `mlx-lm` 0.31.3);
vLLM remains at the repository's existing 0.29.0 pin.

```bash
python -m pip install -e '.[metal]'
VLLM_PLUGINS=metal \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
python examples/demo_attntracker_metal.py
```

The script uses `model_configs/attention_tracker/Mistral-7B-Instruct-v0.3.json`, the
Mistral chat template, and the same instruction/data token-range recipe as the repository's
Attention Tracker demo. `MIA_DEMO_MODEL` and `MIA_CONFIG_FILE` can override its defaults.

## Checks performed

- Inspected the repository's Attention Tracker config, analyzer, `MiaLLM` API, Q/K delivery
  contract, plugin worker path, and the supplied vLLM-Metal runner, model lifecycle, tensor
  bridge, paged context and MLX Mistral/Llama attention source.
- Added unit tests for the emitted raw-Q/K data shape and its use by the existing analyzer.
- No tests, model generation, or runtime checks were executed in this environment.

## Remaining implementation work

- Metal activation steering and hidden-state capture are not implemented. Existing CPU/CUDA
  implementations and their APIs remain unchanged.
- The Metal capture path is limited to one non-paged Mistral-family request at a time during
  prompt prefill. Batched requests, decode/both capture modes, paged attention, other MLX
  model families, disk artifact delivery, server-mode delivery, and tensor parallelism need
  backend implementations and behavioral tests before they can be considered supported.
- The underlying vLLM-Metal/vLLM version compatibility and a complete Apple Silicon runtime
  pass remain unverified. In particular, capture assumes the target runner calls its
  non-paged `_prefill_single` path and the Mistral attention implementation exposes the
  `scaled_dot_product_attention` global described by the inspected sources.
