# Attention Tracker with vLLM-Metal

This focused example uses the existing `MiaLLM` API, runs model generation
through vLLM with the `metal` platform plugin, captures Q/K at the MLX attention
call, and uses the existing `attn_tracker` analyzer:

```bash
uv sync --extra metal --group test
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python examples/demo_attntracker_metal.py
```

Run on an Apple Silicon Mac with enough unified memory for
`mistralai/Mistral-7B-Instruct-v0.3`. The script defaults to
`model_configs/attention_tracker/Mistral-7B-Instruct-v0.3.json`; use
`MIA_DEMO_MODEL` and `MIA_CONFIG_FILE` together to select another model and
matching config. The script sets the supplied target environment defaults
before importing vLLM. It also uses `enforce_eager=True` to select the
non-CUDA-graph path.

The original notebook collection is not migrated by this scoped example. The
notebook at `notebooks/demo_attntracker.ipynb` uses the old v0 API and should
not be used as the vLLM-Metal entry point; run the script above for this use
case.
