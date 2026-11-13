# vLLM-Metal translation

## Implemented

- Added a separate `metal` dependency profile with vLLM-Metal 0.2.0, leaving
  the CPU and CUDA profiles, pins, and install paths unchanged.
- Added `mia.core.backends.vllm_metal.MetalQKTap`, a scoped observer for
  vLLM-Metal's actual MLX SDPA Q/K preparation point. It delegates to the
  original target function and returns its result unchanged; observer errors
  are surfaced and the original function is restored on close.
- Added unit tests for observation scope, layer mapping, return preservation,
  failure validation and lifecycle restoration.
- Added runnable setup/smoke-check notes for the selected use case below.

## Run

On Apple Silicon, provision the vLLM-Metal distribution and its compatible
vLLM/MLX dependencies per the vLLM-Metal release instructions, then install
this repository with the Metal extra:

```bash
python -m pip install -e '.[metal]'
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
MIA_DEMO_MODEL=RedHatAI/granite-3.1-2b-instruct-quantized.w4a16 \
python examples/demo_attntracker.py
```

The command is provided as the use-case entry point but **the Attention Tracker
capture path is not yet integrated with vLLM-Metal**, so it will not produce
valid tracker probes until the remaining work below is finished. A model-only
vLLM-Metal smoke check can be run with:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python -c 'from vllm import LLM, SamplingParams; m=LLM("RedHatAI/granite-3.1-2b-instruct-quantized.w4a16", max_model_len=2048); print(m.generate(["Say hello."], SamplingParams(max_tokens=8))[0].outputs[0].text)'
```

The inspected Metal loader supports AWQ and rejects GPTQ. The selected
checkpoint's quantization metadata was not available in this repository, so
model loading must be verified on the target machine before treating that
checkpoint as runnable.

## Checks performed

- Inspected the repository's Attention Tracker entry point, model config, QK
  worker, analyzers, `MiaLLM` public API, and delivery contracts.
- Inspected the declared vLLM-Metal plugin registration, worker, model
  lifecycle, Granite MLX model, SDPA wrapper, Q/K preparation helper, paged
  attention context and AWQ loader.
- Added unit tests under `tests/metal/`; no test command or Metal runtime was
  executed in this environment.

Run the new unit tests with:

```bash
pytest tests/metal -q
```

## Remaining implementation work

- Connect Metal worker/runner initialization and per-request scheduler context
  to the existing `capture_qk` API and cleanup lifecycle.
- Implement last-token Q selection and full cached K-prefix reconstruction
  from vLLM-Metal's paged KV cache, while retaining request boundaries.
- Convert MLX arrays safely into the existing probe/artifact contract, deliver
  probes to `RequestOutput.probes`/disk, and preserve batching/abort semantics.
- Verify the selected checkpoint's quantization method and real model loading.
- Run backend unit tests and Apple-Silicon end-to-end generation/capture/
  analysis tests. Real vLLM generation alone does not establish capture
  correctness.
