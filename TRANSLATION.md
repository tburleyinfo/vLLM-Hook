# vLLM-Metal translation: Attention Tracker

## Implemented

- Added an optional `metal` dependency profile alongside the existing CPU/CUDA profiles; `requirement.txt` and existing pins remain unchanged.
- Added a Metal-specific Q/K capture adapter for the selected `Qwen/Qwen2-1.5B-Instruct` path. It observes post-RoPE Q/K arrays at the model's existing SDPA call, preserves the original SDPA invocation and return value, and collects capture only for requests that ask for `output_qk`.
- Routed in-memory probes, request cleanup, and disk artifacts through the existing worker RPC names and Q/K payload/artifact shapes. `MiaLLM` now explicitly registers its plugin when the configured vLLM plugin list selects only the platform plugin.
- Added a runnable Apple Silicon entry point at `examples/demo_attntracker_metal.py` and focused contract tests at `tests/backends/test_metal_qk_capture.py`.
- The existing `AttntrackerAnalyzer`, model config, and notebook files are unchanged.

## Run

Use an Apple Silicon environment with vLLM-Metal installed and the project installed with its Metal extra:

```sh
pip install -e '.[metal]'
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
MIA_ALLOW_CUDAGRAPH=0 \
python examples/demo_attntracker_metal.py
```

The demo uses the repository's `model_configs/attention_tracker/Qwen2-1.5B-Instruct.json`, real `MiaLLM`/vLLM generation, its configured important heads, and the same prompt-token range recipe as the selected notebook. The model weights/tokenizer must be accessible to vLLM-Metal. The adapter is intentionally restricted to the split-projection Qwen2 attention interface and fails during hook installation if that interface is unavailable.

## Checks performed

- Inspected the selected notebook, model configuration, Q/K worker and analyzer path, and the declared vLLM-Metal Qwen2/SDPA and worker/lifecycle sources.
- Added tests for layer selection, MLX/NumPy-to-CPU tensor conversion, compressed probe contract shape, and request-state cleanup.
- No tests or Apple Silicon runtime checks were executed in this environment; the supplied tool interface did not provide a test runner or shell.

## Remaining work

- Run the focused tests and end-to-end demo on Apple Silicon with the declared vLLM-Metal runtime. The target sources are reference-only here, and dependency versions/runtime compatibility cannot be verified from this tree alone.
- This selected Attention Tracker translation does not implement activation steering or hidden-state capture on Metal; those are outside the selected use case and must not be treated as supported by this backend.
- Metal score-only Q/K capture and the paged-attention Q/K path are also not implemented. The selected environment uses raw Q/K and disables paged attention; other requests are refused rather than silently returning empty capture.
- Validate `save_to_disk` artifact lifecycle and batched/decode capture on the target runtime, in addition to the demo's in-memory prefill capture.
