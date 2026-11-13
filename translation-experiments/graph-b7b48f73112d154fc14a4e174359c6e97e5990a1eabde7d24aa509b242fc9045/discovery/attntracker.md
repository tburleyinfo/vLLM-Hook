# Attention Tracker migration discovery

## Selected flow

The selected notebook configures `MiaLLM` with `worker_name="capture_qk"` and
`analyzer_name="attn_tracker"`. `MiaLLM.generate` injects `output_qk` and
`hookq_mode` into vLLM `SamplingParams.extra_args`; the worker captures Q/K,
returns a `probes` dictionary and `AttntrackerAnalyzer` reconstructs
attention-weight mass for the instruction/data token ranges. The model config
selects heads `(5,18)`, `(7,12)`, `(9,29)`, and `(17,2)` with last-token Q/K
capture.

The existing worker installs PyTorch forward hooks on vLLM runner attention
modules, accesses vLLM forward context/request metadata, relies on CUDA
custom-op registrations and graph/aperture routing, and serializes raw
PyTorch Q/K into the established probe/artifact contract.

## Target source findings

- `vllm_metal.v1.worker.MetalWorker` constructs `MetalModelRunner`.
- `ModelLifecycle` loads model code and weights through `mlx_lm.load`; generation
  still enters vLLM's engine/worker and uses MetalModelRunner, rather than
  standalone MLX generation.
- The paged SDPA path replaces MLX attention modules with
  `SDPAPagedAttentionWrapper`. `sdpa_forward` calls `prepare_sdpa_qkv`, which
  returns post-RoPE Q/K arrays in MLX, so this is a possible exact observation
  point when that path is active.
- The target platform identifies PyTorch's device as CPU/MPS and its custom
  dispatch key as CPU; it is not a CUDA-compatible backend. The existing
  CUDA ops, CUDA graph capture and V2 GPU runner assumptions do not transfer.
- `VLLM_METAL_USE_PAGED_ATTENTION=0` routes around the paged SDPA wrapper, so
  the new interception primitive cannot observe Q/K under the requested target
  environment.
- vLLM-Metal's attention context contains packed sequence boundaries, but
  supporting chunked prefill, cached-prefix K reconstruction, per-request
  capture routing, request IDs, artifact delivery and full request lifecycle
  needs an explicit runner integration.

## Derived relationship slice

Source path: `MiaLLM.generate` → vLLM `LLM.generate` → MIA plugin extra-args
and worker installation → `QKCaptureWorker` eager hook or `install_qk_hosts`
graph path → Q/K probes → `AttntrackerAnalyzer.compute_attention_from_qk` →
`attn2score`.

Target path: vLLM plugin selects `MetalPlatform` → `MetalWorker` →
`MetalModelRunner` → `ModelLifecycle`/MLX model → optional
`SDPAPagedAttentionWrapper` → `sdpa_forward` →
`prepare_sdpa_qkv` → MLX Q/K → Metal attention kernel.

The source and target paths meet only at a possible attention observation
point; the worker lifecycle, request routing, capture return path and device
semantics are not equivalent today.
