# vLLM-Metal translation

## Implemented

- Added the `metal` optional dependency profile without changing the existing
  CPU/CUDA profiles or their pins.
- Added an opt-in Q/K capture path for in-process vLLM-Metal generation using
  the model's real MLX Llama attention calls. It observes post-RoPE queries and
  the key tensor passed to SDPA, then delivers standard `MiaLLM` probes
  (`config` and `qk_cache`) to the existing analyzer. The existing analyzer,
  output shape, and `MiaLLM` generation API are reused.
- Supports standard non-paged Llama attention on vLLM-Metal for capture,
  in-memory analysis, and `save_to_disk` analysis. The attention observer is
  inactive outside an armed `MiaLLM.generate()` call.
- Added `examples/demo_attntracker_metal.py` for the supplied
  `Meta-Llama-3-8B-Instruct` attention-tracker configuration. It derives
  instruction/data token spans from that model's chat template instead of
  assuming another model's token offsets.

## Run

Install the existing project together with its Metal profile in an
Apple-Silicon environment that has access to the gated Llama checkpoint:

```sh
python -m pip install -e ".[metal]"
```

From the repository root, run:

```sh
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python examples/demo_attntracker_metal.py
```

The script generates both benign and injection-style examples through
`MiaLLM`/vLLM-Metal, captures Q/K during real model generation, and prints the
existing Attention Tracker scores. The new targeted test is
`tests/backends/test_vllm_metal_qk_capture.py`.

## Checks performed

- Inspected the selected notebook, model configuration, existing `MiaLLM`
  worker/analyzer flow, Q/K probe/artifact formats, and the declared
  vLLM-Metal Llama attention, model lifecycle, and request-runner sources.
- Added unit tests for the emitted probe contract, analysis compatibility,
  batched requests, and sequential decode request association.
- No test runner, Apple-Silicon host, model load, package installation, or
  generation check was available/performed in this environment. Runtime
  compatibility with the installed vLLM-Metal release therefore remains
  unverified.

## Remaining work

- Metal hidden-state capture and activation steering are not implemented;
  their existing CPU/CUDA worker paths remain unchanged.
- This capture path covers in-process, non-paged Llama attention only. Paged
  Metal attention and server/remote-worker capture delivery need separate
  request-lifecycle integration.
- Run the new tests and selected example on the target Apple-Silicon
  environment, and verify tokenizer/model access, runtime scheduling, and
  probe delivery against the installed target-library release.
