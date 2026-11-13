# vLLM-Metal translation: Attention Tracker

## Implemented

- Added a separate `metal` dependency profile in `pyproject.toml`; the CPU and CUDA profiles and their existing pins remain unchanged.
- Added MLX Llama Q/K capture primitives in `mia.backends.metal.attention_tracker`. The inspected capture point receives RoPE-applied queries and keys, including the cached prefix. Captures are request-scoped, retain the last query row and full key prefix, and use the `config` / `qk_cache` contract consumed by `AttntrackerAnalyzer`.
- Added a vLLM-Metal runner bridge for the non-paged prefill path. It reads the existing `output_qk` request args, invokes the original runner forward unchanged, and returns captured probes by request ID. The MIA plugin uses this bridge for in-memory `MiaLLM.generate()` and the existing Q/K disk format when `save_to_disk=True`.
- Added chat-template character-offset token range calculation and enabled the Meta-Llama-3-8B-Instruct path in `examples/demo_attntracker.py`.
- Added unit tests for the Q/K probe shape and contents, public analyzer consumption, invalid heads, request-scoped runner capture and template-derived token offsets.

## Usage notes for the selected use case

The selected config is `model_configs/attention_tracker/Meta-Llama-3-8B-Instruct.json`. The example's model and config can be selected with:

```bash
VLLM_PLUGINS=metal \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
MIA_DEMO_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
MIA_CONFIG_FILE=model_configs/attention_tracker/Meta-Llama-3-8B-Instruct.json \
python examples/demo_attntracker.py
```

The model is gated and requires the normal Hugging Face access setup. On Apple Silicon install the intended profile with `pip install -e '.[metal]'`.

The runner bridge targets the supplied vLLM-Metal Llama model and the requested non-paged, single-process configuration. The command above follows the existing example API; first-use dependency installation, plugin discovery, gated model access, and the complete generated-data result still need validation in an Apple Silicon runtime.

## Checks performed

- Inspected the repository's Attention Tracker config, analyzer, `MiaLLM` API, Q/K worker and current hook installation paths.
- Inspected the supplied vLLM-Metal Llama attention, model lifecycle, worker and model-runner sources and the supplied hook-primitives note.
- Added focused tests for probe shape/content, invalid heads, request-scoped runner capture and template-derived token offsets, but did not run them: this environment provides file-inspection/edit tools only, not a Python test runner.
- No dependency installation, Apple Silicon model execution or runtime parity check was performed.

## Remaining implementation work

- Run the end-to-end Attention Tracker example and behavioral tests with the declared target dependencies on Apple Silicon; validate plugin discovery, request matching, real vLLM generation, capture values, analyzer outputs, disk artifact loading and engine shutdown.

The current CPU/CUDA hook implementations remain in place. This translation is incomplete; the Metal capture primitives do not replace or simulate generation.
