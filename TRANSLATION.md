# vLLM-Metal translation

## Implemented

- Added a separate `metal` dependency extra with vLLM-Metal, MLX, and mlx-lm;
  the CPU and CUDA extras, their pins, and their installation paths remain.
- Added a vLLM-Metal Q/K capture path for the non-paged MLX runner. It wraps
  Qwen-style `self_attn` callables without replacing model outputs, scopes
  capture to the active `_prefill_single` request, honors the existing
  `output_qk`, `hookq_mode`, and prefill hook settings, and stores CPU Torch
  tensors in the existing worker result format. The existing `MiaLLM`,
  worker RPC, probe delivery, and Attention Tracker analyzer APIs are retained.
- Added non-paged MLX activation steering through the existing `steer` worker.
  It applies the configured vector/method/layer/phase/position to decoder-layer
  outputs and handles single-request prefill plus batched and sequential decode.
- Added a runnable Qwen2-1.5B Attention Tracker example and focused capture
  contract tests.

## Run the selected use case

Use an Apple Silicon host with a vLLM-Metal-compatible vLLM installation. The
repository's target-library source snapshot does not declare release pins;
install the `metal` extra alongside the corresponding vLLM-Metal runtime:

```sh
pip install -e ".[metal]"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
export VLLM_PLUGINS=metal
python examples/demo_attntracker_metal.py
```

The example uses
`model_configs/attention_tracker/Qwen2-1.5B-Instruct.json`, real vLLM
generation via `MiaLLM`, captured Q/K probes, and the repository's existing
Attention Tracker analyzer. It requires the model to be available to the
configured model downloader/cache. Use `save_to_disk=False` (the example's
default) for this bridge.

## Checks performed

- Inspected the repository's QK worker, request/probe delivery path, analyzer,
  selected model configuration, and Attention Tracker demo.
- Inspected the supplied vLLM-Metal model lifecycle, worker, Qwen2 MLX model,
  SDPA wrapper, runner prefill flow, and hook correspondence notes.
- Added unit tests for request/layer filtering, Q/K tensor layout, and
  steering configuration. They
  were **not run**: this environment provides no shell/test runner, and no
  Apple Silicon vLLM-Metal runtime was installed or exercised.

## Remaining work

This translation is incomplete. Metal Q/K capture currently covers only
non-paged Qwen-style prefill. Decode/batched and chunked Q/K capture,
paged-attention capture, and Metal durable/on-disk capture have not been
ported. Those paths must preserve the existing per-request semantics before
the backend can be considered behaviorally complete. The new steering bridge,
dependency extra, and end-to-end example also need validation against a real
compatible Apple Silicon environment.
