# Attention Tracker behavior and target inspection

## Repository path

`MiaLLM.generate` adds `output_qk` and `hookq_mode` sampling metadata for the
`capture_qk` worker. The Q/K worker captures post-RoPE Q/K; `qk_probes` and
`_merge_probes` produce the public probe structure. `AttntrackerAnalyzer`
reconstructs selected-head attention from Q/K and aggregates instruction/data
token ranges. The selected config specifies `hookq_mode: last_token` and
important Qwen2-1.5B attention heads. The existing example's chat messages
and token-range recipe are reused for the corresponding model.

## vLLM-Metal target path

The supplied target source identifies vLLM-Metal 0.2.0 and mlx-lm 0.31.3.
`MetalWorker` constructs `MetalModelRunner`; `ModelLifecycle.load` assigns the
MLX model to the runner; `MetalModelRunner` performs actual vLLM scheduled
generation through its MLX model execution. The Qwen2 source computes Q/K,
applies RoPE and updates its cache before calling module-level
`scaled_dot_product_attention(queries, keys, values, cache, scale, mask)`.
Wrapping that call observes the post-RoPE prompt Q/K while delegating the
original Metal computation. The wrapper is active only during the MIA generation
scope, and the `VLLM_ENABLE_V1_MULTIPROCESSING=0` target setting keeps the
capture collection in-process.

## Deliberate scope boundary

The vLLM-Metal runner/model objects do not implement the vLLM PyTorch V2 worker
runner interface required by existing CUDA/CPU worker hooks. Therefore the
Qwen2 Attention Tracker path uses a narrowly scoped MLX attention wrapper and
the unchanged public MIA generation/analyzer APIs; it does not pretend the
existing CUDA graph aperture, PyTorch hooks, hidden-state capture or steering
hooks work on Metal.
