# Attention Tracker / vLLM-Metal discovery

## Repository contract

`notebooks/demo_attntracker_colab.ipynb` constructs `MiaLLM` with
`worker_name="capture_qk"` and `analyzer_name="attn_tracker"`, requests Q/K
capture for the selected layers, and supplies captured data through
`outputs[0].probes` to `MiaLLM.analyze(analyzer_spec=..., probes=...)`.
The analyzer consumes `{"config": ..., "qk_cache": ...}`. Each layer entry
contains per-prompt `q` and `k_all` tensors plus `layer_num`; the tracker uses
the configured per-layer heads and the analyzer spec's input token ranges.

`model_configs/attention_tracker/granite-3.1-8b-instruct.json` selects
41 important heads spanning layers 6 through 19 and sets `hookq_mode` to
`last_token`. The Colab notebook's Granite prompt range convention is
`((3, 3 + instruction_token_count), (-5 - data_token_count, -5))`.

The current `QKCaptureWorker` and `install_qk_hosts` paths expect vLLM PyTorch
attention modules, CUDA device tensors, a vLLM worker extension, CUDA graph
hooks, and CUDA KV-cache metadata. `AttntrackerAnalyzer` itself is not
CUDA-dependent once it receives ordinary CPU PyTorch tensors.

## Target contract

The supplied `vllm_metal` source loads the generation model through
`mlx_lm.load`; `MetalModelRunner` uses MLX arrays, and the MHA runtime patches
MLX attention objects with `SDPAPagedAttentionWrapper`. Its SDPA path calls
`prepare_sdpa_qkv`, which returns post-norm/post-RoPE Q/K arrays shaped as
`(batch, heads, tokens, head_dim)`. `MetalPagedKVCache.key_caches[layer]`
stores per-layer MLX keys as `(blocks, block_size, num_kv_heads, head_dim)`.
The SDPA forward writes keys to that cache before returning.

These are real target SDPA intermediates and the target's actual paged cache;
the bridge does not replace vLLM generation with `mlx_lm.stream_generate`.
The supplied platform note requires casting MLX bfloat16 arrays to float32
before conversion to NumPy.

## Integration boundary and caveats

The new bridge wraps the target SDPA preparation/dispatch symbols and scopes
capture to an active `MiaLLM.generate` call. It observes prepared queries and
reads full keys from the post-forward Metal cache, then returns the established
MIA probe shape. Integration into `MiaLLM` omits `worker_extension_cls` only
for `capture_qk` when the `metal` plugin is selected; all other backend paths
retain their original worker extension.

`ctx` sequence boundaries and block tables are used to map the packed request
to the paged key cache. That mapping, the target version's function signatures,
and callback context propagation have only been inspected from the provided
source; none could be exercised here. The example follows the notebook's
two-prompt call and disables chunked prefill and prefix caching. Batch ordering
and concurrent requests need Apple Silicon runtime tests before their capture
guarantees can be claimed. Other features, including Metal steering, remain
unimplemented.
