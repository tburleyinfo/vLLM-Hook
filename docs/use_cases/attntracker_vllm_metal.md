# Attention Tracker with vLLM-Metal

This note uses the existing `MiaLLM` / `AttntrackerAnalyzer` API with the
repository's Granite 3.1 2B W4A16 attention-tracker model configuration. It
does not translate the rest of the notebook collection.

On Apple Silicon, install the Metal profile alongside the repository package:

```bash
python -m pip install -e '.[metal]'
```

The target runtime settings are:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
export VLLM_PLUGINS=metal
python examples/demo_attntracker_metal.py
```

The runnable script explicitly registers MIA's general plugin and worker/
analyzer registry because `VLLM_PLUGINS=metal` selects the vLLM-Metal plugin.
It uses the configured important heads, captures post-RoPE Q/K from the MLX
Granite attention computation during prefill, and passes the returned probes
to the existing attention-tracker analyzer. Generation remains the vLLM engine
served by vLLM-Metal; it does not call `mlx_lm.generate`.

The analyzer's `input_range` is tied to the Granite 3.1 chat-template layout
used by the notebook: token positions `[3, 3 + instruction_len)` for the
instruction and `[-5 - data_len, -5)` for the data. A different chat template
requires recalculating those ranges before interpreting scores.
