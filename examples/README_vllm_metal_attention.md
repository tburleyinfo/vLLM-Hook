# Attention Tracker with vLLM-Metal

Run on an Apple Silicon Mac with a compatible vLLM-Metal environment. The
repository's CPU and CUDA extras remain unchanged; select the new `metal`
extra when installing this project.

```bash
uv sync --extra metal --group test
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
export VLLM_PLUGINS=metal
export MIA_DEMO_MODEL=Qwen/Qwen2-1.5B-Instruct
python examples/demo_attntracker.py
```

The demo still calls `MiaLLM.generate()` for real vLLM generation and
`MiaLLM.analyze()` for scoring. The Metal Q/K bridge observes the MLX
`scaled_dot_product_attention` call and supplies the same `qk_cache` probe
shape consumed by `AttntrackerAnalyzer`. Keep `VLLM_ENABLE_V1_MULTIPROCESSING=0`
so the in-process capture collector can observe the model execution.

This path is intended for ordinary full-prompt SDPA prefill. Chunked prompt
prefill, non-SDPA attention families, `save_to_disk`, served `MiaClient`,
steering, and other capture workers are not translated yet. The shipped
example's default Granite checkpoint and model-specific token offsets should
not be assumed to have been validated on Metal; the Qwen2 example config is
provided as an alternate model selection.
