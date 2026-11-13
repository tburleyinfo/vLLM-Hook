# Attention Tracker / vLLM-Metal discovery

## Repository contract

The selected model config is `model_configs/attention_tracker/Meta-Llama-3-8B-Instruct.json`.
Its `important_heads` are layer/head pairs and its `hookq_mode` is `last_token`.
`MiaLLM(worker_name="capture_qk", analyzer_name="attn_tracker")` supplies the existing
request extra arguments, returns generation outputs, and expects the captured payload at
`output.probes`. `AttntrackerAnalyzer` consumes a `config` plus `qk_cache`; each layer entry
contains list-valued `q` and `k_all` tensors and a `layer_num`. The analyzer computes attention
from the final query against each key and applies the supplied instruction/data token ranges.

The current Q/K worker and graph path depend on vLLM's PyTorch V2 runner, GPU worker-extension
mixins, CUDA graph custom ops, request metadata, and the CUDA KV-cache layout. The direct path
also registers PyTorch `CUDA` dispatch ops and installs PyTorch forward hooks. None of those
surfaces are supplied by the target runner.

## Target source findings

- The target selects `vllm_metal.v1.worker.MetalWorker` and `MetalModelRunner`. Its model
  lifecycle loads the actual generation model using `mlx_lm.load`; the Llama model has
  `model.layers[*].self_attn.{q_proj,k_proj,rope}`.
- With paged attention enabled, `SDPAPagedAttentionWrapper` replaces the attention module and
  `prepare_sdpa_qkv` exposes Q/K after RoPE. The requested environment disables this path
  (`VLLM_METAL_USE_PAGED_ATTENTION=0`), so the supplied target source does not install that
  wrapper for this run.
- In the non-paged Llama path, the original mlx_lm attention projects Q/K, reshapes them,
  applies RoPE and updates its model cache. The new adapter wraps these original projection
  callables, retains their outputs, applies the same RoPE and converts capture arrays to the
  standard MIA probe tensor format. It does not substitute generation.
- The target source offers no documented model-forward hook, request-extra-args callback,
  capture delivery API, worker-extension integration, or steering callback. Its general hook
  note supports wrapping/replacing a callable module with restoration, but does not bridge
  scheduling, request metadata, the MIA RPC delivery path, or lifecycle teardown.
- Supplied source versions: `vllm_metal` 0.2.0 and `mlx_lm` 0.31.3; vLLM is pinned by this
  repository at 0.29.0. No target package manifest or target-side test/runtime profile was
  supplied.

## Implementation boundary

`mia.backends.metal.attention_tracker.MetalAttentionCapture` is a backend implementation
primitive: it can instrument a loaded mlx_lm Llama-family model and produce analyzer-compatible
Q/K probes when its `capture_scope` surrounds the relevant model forward. `close()` restores
the original Q/K projection objects. The source inspection does not establish a safe bridge
from this primitive to the `MiaLLM`/MetalModelRunner lifecycle or to request-scoped generation
and output delivery. Therefore the selected use case is not complete and the current CUDA
worker path remains unchanged.

## Remaining work

1. Add request-scoped capture enablement to `MetalModelRunner` while preserving scheduler,
   prefix-cache, streaming and cleanup behavior, and wire it to `MiaLLM.generate` and the
   existing per-request probe delivery contract.
2. Verify projection wrapping, RoPE offsets, multiple requests, exceptions and restoration on
   an Apple Silicon runtime with the declared target packages. The current unit test only
   validates the emitted probe shape and analyzer hand-off.
3. Implement and verify vLLM-Metal activation steering through the existing `steer` API; the
   supplied target sources do not expose an equivalent integration point.
