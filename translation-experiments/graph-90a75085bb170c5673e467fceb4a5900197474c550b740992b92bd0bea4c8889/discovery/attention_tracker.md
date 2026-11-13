# Attention Tracker / vLLM-Metal discovery

## Repository path

`examples/demo_attntracker.py` exercises the stable public sequence:
`MiaLLM(worker_name="capture_qk", analyzer_name="attn_tracker")`, generate with
vLLM `SamplingParams`, then analyze the `probes` attached to the returned
`RequestOutput`. The tracker configuration supplies important heads and
`hookq_mode: last_token`. `AttntrackerAnalyzer` expects per-layer Q/K tensors,
the model head counts and head dimension, and scores the last captured query
against supplied instruction/data token ranges.

The existing QK worker is a vLLM CUDA worker extension. Its hooks consume
PyTorch attention-module inputs, read vLLM forward-context request metadata,
route per-request rows, restore cached keys, and hand request-indexed artifacts
to the existing delivery API. The graph implementation additionally registers
custom ops on the CUDA dispatch key and allocates CUDA buffers.

## Target path and boundary

The declared vLLM-Metal source registers a Metal platform and selects
`vllm_metal.v1.worker.MetalWorker`; its runner loads MLX models. Granite's
MLX implementation uses `layers[*].self_attn`. The target's
`SDPAPagedAttentionWrapper` delegates to `sdpa_forward`, which calls
`prepare_sdpa_qkv`; that helper returns post-RoPE Q and K arrays, shaped
`[batch, heads, tokens, head_dim]`. This supplies an actual observation point
without changing attention math. `PagedAttentionContext` has packed sequence
boundaries and cache metadata, but it does not contain request IDs. The target
Metal worker/runner has no vLLM GPU-worker extension lifecycle matching MIA's
existing `worker_extension_cls` contract.

The new `MetalQKTap` wraps the declared Q/K helper, delegates its original
implementation unchanged, and provides an explicit scoped observer. It is a
tested low-level primitive, not a complete capture implementation. MIA's
request routing, last-prompt-token selection, key-prefix reconstruction,
probe schema/delivery, and analyzer lifecycle are not connected to this tap.
There is no supported way in the inspected target source to associate these
Q/K observations with the MIA request output contract without extending the
Metal runner lifecycle.

The vLLM-Metal AWQ loader owns AWQ checkpoints and explicitly refuses GPTQ.
The selected Granite checkpoint's quantization method must be checked against
that policy at runtime; the repository's JSON model config does not identify
the checkpoint's `quantization_config`.
