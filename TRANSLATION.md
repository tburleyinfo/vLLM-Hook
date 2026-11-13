# vLLM-Metal translation: Attention Tracker

## Implemented

- Added the `metal` optional dependency profile without replacing or changing the existing CPU/CUDA profiles or pins.
- `MiaLLM(worker_name="capture_qk", analyzer_name="attn_tracker", ...)` selects the Metal capture bridge when `VLLM_PLUGINS` includes `metal`. It does not pass the CUDA worker extension to the Metal engine.
- The bridge observes Q/K prepared by vLLM-Metal's MLX SDPA implementation and reads full prompt keys from that layer's paged MLX KV cache after the ordinary attention forward has completed. It converts the captured arrays to CPU PyTorch tensors in the existing `config`/`qk_cache` probe contract. Generation remains vLLM generation through vLLM-Metal.
- Existing `generate(...).probes`, `analyze(probes=...)`, and `save_to_disk`/`analyze(run_id=...)` routes are connected. The standard analyzer still computes Attention Tracker scores.
- Added a runnable Granite 3.1 8B example following the notebook's two-prompt flow and contract/analyzer tests.

The bridge captures each prompt's last-token query from prompt prefill. The example follows the notebook's two-prompt flow and disables chunked prefill and prefix caching so those lifecycle paths do not alter which query is captured.

## Run

On an Apple Silicon Mac with a compatible vLLM-Metal environment:

```sh
uv sync --extra metal --group test
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python examples/demo_attntracker_metal.py
```

The example uses `model_configs/attention_tracker/granite-3.1-8b-instruct.json` and the Granite chat-template token-range convention used in `notebooks/demo_attntracker_colab.ipynb`. Model weights must be available to the vLLM-Metal loader.

## Checks performed

- Inspected the notebook's Attention Tracker prompt/template flow, the selected model configuration, the existing Q/K worker, analyzer, probe formats, and `MiaLLM` public API.
- Inspected supplied vLLM-Metal sources for MLX SDPA Q/K preparation, Metal wrapper dispatch, paged KV cache layout, generation lifecycle, and plugin initialization.
- Added a unit test that checks the generated probe contract by running it through the existing Attention Tracker analyzer. Added a negative test for incomplete configured-layer capture.
- Tests and model generation were **not run**: this environment provides no terminal/test runner and no Apple Silicon vLLM-Metal runtime. The dependency profile and runtime bridge therefore remain unverified.

## Remaining work

- Validate installation resolution for `vllm-metal==0.2.0` and its vLLM/MLX requirements on a supported Apple Silicon setup; the supplied target source tree did not include its packaging metadata or install recipe.
- Run the new unit tests and the example on Apple Silicon. Verify Granite model loading, matching layer indices, paged-cache request mapping, MLX evaluation/conversion, in-memory and disk delivery, and numerical score parity against the existing CPU/CUDA path.
- Verify multi-prompt ordering, chunked-prefill, and concurrent generation capture semantics. The example uses two prompts in one generation call with unchunked prefill; this scheduling path has not yet been run on the target.
- Metal steering is not part of this Attention Tracker use case and has not been ported. The existing CPU/CUDA steering implementation remains unchanged.
