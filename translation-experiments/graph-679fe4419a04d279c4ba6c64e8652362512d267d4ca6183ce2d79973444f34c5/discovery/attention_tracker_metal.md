# Attention Tracker / vLLM-Metal discovery

## Selected behavior slice

Source graph (`target_relationships/source/files/`) and code inspection trace the notebook path:

`examples/demo_attntracker.py` → `MiaLLM.generate()` → `SamplingParams.extra_args` (`output_qk`, `hookq_mode`, disk flags) → `QKCaptureWorker.install_hooks()` → request/layer Q/K state → `get_captured_states()`/`flush_disk()` worker RPC → compressed `{config, qk_cache}` payload → `AttntrackerAnalyzer.analyze()`.

The analyzer requires post-RoPE Q/K with the configured attention head geometry. The selected JSON requests 14 layers' configured important heads using `hookq_mode: last_token`; the analyzer reselects those heads from full Q/K and computes the normalized instruction-attention score.

## Target relationship slice

Target graph and source inspection show:

- `mlx_lm.models.qwen2.Attention.__call__` produces Q/K, applies RoPE and cache update, then calls the module-global `scaled_dot_product_attention`. This is the observable point where post-RoPE Q/K and full cached K are available on the non-paged execution path.
- `MetalModelRunner._prefill_single`, `_batched_decode`, and `_sequential_decode` execute generation and retain each request's sampling parameters. `MetalWorker` delegates worker calls to this runner.
- The Metal paged wrapper dispatches through `SDPAPagedAttentionWrapper.__call__` → `sdpa_forward` → `prepare_sdpa_qkv`; that is a different path from Qwen2's original SDPA global.
- The requested environment disables paged attention and v1 multiprocessing. The adapter therefore wraps the Qwen2 class and its SDPA module-global, scopes observation to the target runner's request methods, and leaves original attention execution intact. It intentionally does not claim coverage for the paged path.
- Existing `_plugin._patched_llm_generate` retrieves captured states via worker RPC and expects compressed pickle payloads; the Metal worker bridge exposes the same RPC names. The in-memory tensor schema matches `unpack_qk()` and analyzer expectations.

## Inferences and limits

The static graphs document lexical structure and calls, not runtime equivalence. In particular, request scheduling, lazy MLX materialization, worker RPC dispatch, output artifact handling, and actual vLLM plugin loading require Apple Silicon runtime verification. Metal steering, hidden-state capture, QK score-only capture and paged-attention capture are not implemented by this selected-use-case adapter.
