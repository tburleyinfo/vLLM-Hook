# Attention Tracker on vLLM-Metal

This note uses the existing `MiaLLM` API and the Granite 3.1 8B Attention
Tracker configuration. It captures the configured heads' post-RoPE Q/K on the
non-paged vLLM-Metal model path and runs the existing Attention Tracker
analyzer. Generation is performed by vLLM with the Metal plugin; it does not
use standalone MLX generation.

On Apple Silicon, provision a Python environment with the repository's Metal
extra and install the project:

```bash
uv sync --extra metal --group test
```

Set the plugin environment and run the existing demo from the repository root:

```bash
export VLLM_PLUGINS=metal
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
export MIA_DEMO_MODEL=ibm-granite/granite-3.1-8b-instruct
export MIA_CONFIG_FILE=model_configs/attention_tracker/granite-3.1-8b-instruct.json
export MIA_DEMO_ENFORCE_EAGER=1
python examples/demo_attntracker.py
```

The demo emits generated text and one focus score for each of the original and
injection-containing inputs, followed by their difference. Downloading model
weights requires the normal model-cache access available on the machine. The
8B model needs sufficient unified memory; use an appropriately sized Apple
Silicon system.

## Scope and implementation notes

- The Metal Q/K worker attaches wrappers at the existing callable attention
  modules and scopes capture to each non-paged prefill request. It stores Q/K
  rows as CPU PyTorch tensors in the repository's current probe/artifact
  contract, so `MiaLLM.generate`, `MiaLLM.analyze`, and existing disk flush
  behavior continue to be the public surfaces.
- The Metal steering worker wraps decoder layer callables and applies the
  existing `add_vector` or `adjust_rs` steering semantics to configured
  residual rows. CPU/CUDA worker paths are unchanged.
- The Metal adapters intentionally reject paged attention at install time:
  the target environment for this use case sets `VLLM_METAL_USE_PAGED_ATTENTION=0`.
- Runtime verification on an Apple Silicon host is still needed for actual
  vLLM-Metal model loading, probe delivery/cleanup and steering behavior. The
  unit tests cover the probe shape/analysis contract, not the native Metal
  runtime.
