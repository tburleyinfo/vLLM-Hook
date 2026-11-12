# Attention Tracker / vLLM-Metal discovery

## Repository contract found

- `MiaLLM(model, worker_name="capture_qk", analyzer_name="attn_tracker", ...)`
  uses the existing plugin worker lifecycle. Its `generate()` passes real
  prompts and `SamplingParams` to `vllm.LLM.generate()`, then returns probe
  data on each request output. `analyze(probes=...)` uses the public analyzer.
- The analyzer accepts `{"config": ..., "qk_cache": ...}`. Q is a flattened
  last-query head vector; K is a sequence of flattened KV-head rows. It either
  derives attention from Q/K or consumes precomputed per-head scores.
- The existing Q/K worker expects PyTorch attention modules, vLLM forward
  context, V1 runner request metadata and CUDA graph/device paths. Those are
  not interchangeable with vLLM-Metal's MLX attention runtime.

## Target source correspondence

- `vllm_metal.v1.worker.MetalWorker` owns a `MetalModelRunner` and returns
  real models through its worker interface.
- `vllm_metal.attention.impls.sdpa_wrapper.SDPAPagedAttentionWrapper` invokes
  `sdpa_forward()` and identifies the model layer.
- `vllm_metal.attention.impls.sdpa.prepare_sdpa_qkv()` returns post-norm,
  post-RoPE MLX Q/K arrays shaped `[batch, heads, tokens, head_dim]`.
- `vllm_metal.attention.context.PagedAttentionContext.cu_seqlens` supplies
  packed request token boundaries. `vllm_metal.v1.model_runner` prepares this
  context for paged forward passes and clears it after forward execution.
- Metal's generation model is loaded through `mlx_lm` but called by the
  vLLM-Metal runner; the translation does not invoke standalone MLX generation.

## Remaining gap

The new backend module implements the SDPA observation and packed Q/K tensor
contract, but it is not connected to the existing `MiaLLM` worker extension or
per-request delivery. In particular, the current attention context has no
request IDs, and no validated mapping has been added from packed segments to
MIA's request-id lifecycle. The target reference platform does not include
source for vLLM's worker-extension composition or engine scheduling internals,
so changing that path without runtime evidence risks breaking existing API
behavior or cross-request capture isolation. No Apple Silicon Metal runtime is
available for integration validation here.
