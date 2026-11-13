# Attention Tracker on vLLM-Metal

This note covers the Granite 3.1 8B Attention Tracker example only. The new
Metal path uses vLLM generation and captures Q/K at Granite's MLX attention
call boundary; it does not substitute direct `mlx_lm` generation.

On an Apple Silicon machine, install the Metal profile and this repository:

```bash
python -m pip install -e ".[metal]"
```

Run the existing example from the repository root with vLLM-Metal selected:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
MIA_DEMO_MODEL=ibm-granite/granite-3.1-8b-instruct \
python examples/demo_attntracker.py
```

The existing model configuration is
`model_configs/attention_tracker/granite-3.1-8b-instruct.json`. The example
uses one prompt per `MiaLLM.generate` call, requests in-memory Q/K capture,
and runs the unchanged `AttntrackerAnalyzer`.

The Metal implementation currently captures the first configured attention
invocation for each Granite layer per prompt in non-paged generation calls.
`MiaLLM.generate(..., save_to_disk=True)` also writes the existing ranked
`qk.pt` artifact shape for offline analysis. It does not yet support reliable
request attribution during concurrent or chunked prefill, paged Metal
attention, server/client capture, hidden-state capture, or steering. These
paths must not be interpreted as supported by this usage note.
