# vLLM-Metal translation

## Implemented

- Added a Metal Q/K capture adapter and connected it to the existing
  `capture_qk` worker. It wraps vLLM-Metal's MLX attention modules, observes
  post-RoPE Q/K while delegating the original attention call, and stores the
  captured tensors in MIA's existing per-request Q/K format.
- MIA now selects eager hook mode when `VLLM_PLUGINS=metal`, because its
  existing CUDA-graph capture implementation cannot execute on the MLX runner.
  An explicit `MIA_ALLOW_CUDAGRAPH=1` on Metal is refused rather than silently
  enabling an unsupported CUDA path.
- Capture delivery, `MiaLLM.generate`, and `AttntrackerAnalyzer` continue to
  use the existing public API and payload shapes. The analyzer remains the
  existing PyTorch implementation and operates on captured CPU tensors.
- Added a `metal` optional dependency profile without removing or changing the
  existing CPU/CUDA profiles.
- Added `examples/demo_attntracker_metal.py` for the supplied Mistral model
  configuration.

## Run

On Apple Silicon, use Python 3.11–3.14 and install the new profile with
`pip install -e '.[metal]'`. If the Mistral checkpoint is gated, configure
Hugging Face access before starting. From the repository root, run:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python examples/demo_attntracker_metal.py
```

The adapter needs the non-paged MLX cache path and uses `max_num_seqs=1` for
the current capture implementation. Generation is still executed by the real
vLLM engine backed by vLLM-Metal; it is not replaced with direct `mlx_lm`
generation.

## Checks and remaining work

- Inspected the source and target relationship indexes and traced the
  Attention Tracker request through `MiaLLM`, `QKCaptureWorker`,
  `AttntrackerAnalyzer`, the Metal runner, its attention wrapper, Mistral's
  RoPE/cache flow, and the Metal tensor bridge.
- Added unit tests for capture tensor layout, key-cache slicing, layer selection,
  and preserving the original attention result. Tests were not run in this
  environment.
- No Apple Silicon runtime, generated Mistral Q/K artifacts, analyzer parity,
  end-to-end output comparison, or performance check was available here.
- Metal activation steering and hidden-state capture are not implemented.
  Paged-attention Q/K capture, batched decode capture, disk artifact capture,
  and models beyond the inspected Mistral RoPE/cache layout also remain
  unimplemented or unverified. These paths must not be treated as supported.
