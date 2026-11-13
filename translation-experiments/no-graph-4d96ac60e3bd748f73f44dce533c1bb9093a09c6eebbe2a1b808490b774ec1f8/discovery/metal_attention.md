# Attention Tracker / vLLM-Metal source findings

## Repository contract

- The selected model config requests the Q/K capture worker, last-token query
  mode, and seven configured layer/head pairs.
- `MiaLLM` adds `output_qk` and `hookq_mode` to `SamplingParams.extra_args`,
  attaches captured probes to generated outputs, and invokes
  `AttntrackerAnalyzer` through the public `analyze(probes=...)` API.
- The analyzer requires a per-layer `qk_cache`, the head-count/head-dimension/
  attention-multiplier config, and input ranges matching the rendered prompt.
- The Colab example uses Granite ranges `(3, 3 + instruction_len)` and
  `(-5 - data_len, -5)`.

## Target source evidence

- `vllm_metal.v1.worker.MetalWorker` constructs `MetalModelRunner`; the worker
  does not expose vLLM's GPU/CPU runner API used by MIA's CUDA worker mixins.
- `MetalModelRunner._handle_new_requests` calls `_prefill_single` per request
  when its non-paged path is selected. Its `_prefill_single` creates a fresh
  MLX prompt cache and invokes the loaded model for the prompt.
- The supplied Granite MLX attention module exposes `q_proj`, `k_proj`, head
  counts, `rope`, and its cache's `offset`/`keys`. Its Q/K values are projected,
  reshaped, RoPE-transformed, then consumed by attention.
- The supplied vLLM-Metal attention patcher uses callable wrappers and preserves
  inner module objects. The new implementation uses a guarded class-level
  callable wrapper and does not change model outputs.
- vLLM-Metal's AWQ loader owns AWQ checkpoint normalization and MLX loading.
  Generation remains through the vLLM-Metal runner.
  The supplied target source tree does not include package metadata or a
  dependency manifest, so the new optional dependency distribution names and
  compatible release pins cannot be confirmed from the provided reference.

## Implemented boundary

The current implementation uses the repository's existing `MiaLLM`, worker
extension, compressed probe RPC, disk artifact, and analyzer contracts for
prompt-prefill Q/K capture with paged attention disabled. Other capture modes,
hidden-state capture, and steering remain unfinished; see `TRANSLATION.md`.
