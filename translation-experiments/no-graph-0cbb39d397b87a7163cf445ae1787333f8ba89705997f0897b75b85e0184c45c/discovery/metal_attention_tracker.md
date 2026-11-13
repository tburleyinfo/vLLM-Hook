# Attention Tracker / vLLM-Metal discovery

## Relevant repository contract

- `MiaLLM.generate()` attaches capture intent through request `extra_args`, then exposes in-memory probe dictionaries on the output; `MiaLLM.analyze()` passes that dictionary to the configured analyzer.
- `AttntrackerAnalyzer` consumes a `qk_cache` mapping with per-layer query/key rows, `layer_num`, and a config containing query/KV head counts, head size, and attention multiplier.
- `model_configs/attention_tracker/Qwen2-1.5B-Instruct.json` selects fourteen heads and specifies `hookq_mode: last_token`.
- The existing Q/K worker relies on PyTorch attention hooks, vLLM forward metadata, request IDs, and cached-prefix-key reconstruction. Those are not available through the target's MLX attention wrapper interface.

## Target source correspondence

- `vllm_metal.attention.impls.sdpa.prepare_sdpa_qkv()` returns post-normalization, post-RoPE Q/K arrays in `[batch, heads, tokens, head_dim]` layout. This is the closest target seam for an attention capture.
- `SDPAPagedAttentionWrapper` calls `sdpa_forward()` when paged attention context is present and delegates to the original module otherwise.
- Target attention modules are replaced by `walk_and_wrap()` during runtime installation; `find_layers()` and `find_attn_attr()` provide target-side model traversal. Layer wrappers preserve the wrapped module at `_inner`.
- The target worker loads an MLX model into `MetalModelRunner.model`; its attention context carries sequence boundaries, cache slots, and offsets, but does not expose MIA request IDs or per-request capture settings.
- The target SDPA operation receives only newly projected K rows; the active paged KV cache contains prior keys, so a naïve Q/K capture does not satisfy the existing full-prefix K contract on decode.
- The requested target environment sets `VLLM_METAL_USE_PAGED_ATTENTION=0`. The SDPA wrapper delegates to the original attention module when there is no paged context, and the runner's non-paged cache path does not call `prepare_sdpa_qkv`; the new hook seam is therefore not active in the specified environment.

## Scope decisions and gaps

- The new adapter observes `prepare_sdpa_qkv`, exports arrays by casting to MLX float32 before NumPy conversion, and emits the analyzer-compatible shape for `last_token` prefill capture.
- The adapter does not wire itself into a model-runner load or generation lifecycle; callers must already hold the loaded model and invoke capture in the same execution context. For this reason the selected notebook is not yet runnable end-to-end on Metal.
- The adapter explicitly rejects `VLLM_METAL_USE_PAGED_ATTENTION=0` rather than report a successful empty capture; implementing capture on the specified fallback path remains necessary.
- Decode-key reconstruction, full-token mode, request routing, server delivery, and MIA steering lifecycle integration remain unfinished. The new steering helper only supports an explicit model scope with all-token/both-phase behavior.
