# vLLM-Metal translation: Attention Tracker

## Implemented

- `MiaLLM(worker_name="capture_qk", analyzer_name="attn_tracker", ...)` uses the
  existing public API and real vLLM generation. With `VLLM_PLUGINS=metal`, it
  avoids installing the CUDA/CPU MIA worker extension and captures Qwen2
  prefill Q/K from the MLX attention path used by vLLM-Metal.
- Captured probes use MIA's existing `{"config": ..., "qk_cache": ...}` contract,
  including the configured important heads, last-token query mode, and full
  prefill keys. Existing `AttntrackerAnalyzer` computes the score unchanged.
- In-memory batches and `save_to_disk=True` use the existing probe and run
  artifact APIs. The Metal capture path requires full, unchunked Qwen2 prefill;
  it raises a delivery error if configured layers or request rows were not
  captured rather than returning success-shaped empty probes.
- vLLM-Metal dependencies are available as a separate optional `metal` profile;
  CPU/CUDA profiles and their existing dependency pins remain present.

## Run on Apple Silicon

Install the Metal profile in an Apple Silicon Python environment, then install
this repository in editable mode:

```bash
uv sync --extra metal --group test
```

Set the target platform options before starting Python, then run the
Attention-Tracker usage example:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
export VLLM_PLUGINS=metal
python examples/demo_attntracker_metal.py
```

The example uses
`model_configs/attention_tracker/Qwen2-1.5B-Instruct.json`, generates one
prompt per call through vLLM-Metal, and analyzes the captured Q/K probes with
the repository's existing analyzer.

## Checks and validation status

- Added a focused automated test for probe tensor shapes, metadata and the
  existing analyzer's score result using synthetic Q/K tensors, plus a test
  that verifies missing prefill-layer capture is an error. These tests were
  not run in this workspace.
- The new runtime path and installation recipe were not executed on Apple
  Silicon; the supplied target source was inspected for integration points.

## Remaining work

- The selected Attention Tracker use case is implemented for the source-inspected
  `mlx_lm.models.qwen2` path. Other MIA workers (hidden-state capture and
  activation steering), other MLX model families, paged-attention capture,
  chunked prefill, and multi-process Metal execution are outside this selected
  use case and are not translated by this change.
- Validate the runnable example on an Apple Silicon host with the declared
  target dependencies, including prompt token ranges and the installed
  vLLM/vLLM-Metal version combination.
