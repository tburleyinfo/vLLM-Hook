# Backend discovery notes

## Repository integration

- `MiaLLM` constructs `vllm.LLM`, sends worker configuration through `SamplingParams.extra_args`, and returns `RequestOutput` objects whose instrumentation output is attached as `probes`.
- Existing CPU/CUDA instrumentation is implemented as vLLM worker extensions. Those extensions assume vLLM runner hooks, Torch modules/tensors, runner request batches, and worker RPC methods.
- Capture delivery is broader than observing a layer result: it includes per-request routing, probes shape/metadata, optional disk artifacts, analyzer input, and lifecycle cleanup. The API server path and offline LLM path both participate.

## Target source observations

- `vllm_metal.platform.MetalPlatform.check_and_update_config()` selects `vllm_metal.v1.worker.MetalWorker` when the worker class is automatic.
- `MetalWorker.init_device()` constructs `MetalModelRunner`; `MetalModelRunner.load_model()` delegates to `ModelLifecycle`, which loads generation models through `mlx_lm` or `mlx_vlm`.
- The model runner executes model forwards through its MLX `_forward_model` callable and preserves the usual vLLM `execute_model()` / `sample_tokens()` handoff.
- MLX neural modules are callable modules. Their decoder layer collections and forward calls differ from MIA's current Torch hooks. Q/K tensors in supplied MLX model implementations are local variables within attention forwards; wrapping the outer layer does not observe post-RoPE Q/K.
- `vllm_metal.v1.model_adapter.DefaultModelAdapter` contains target-forward helpers for particular operations such as optional Gemma4 target hidden-state collection, but it is not a general MIA capture, steering, or artifact-delivery interface.
- The platform defaults and environment listed in the task select non-paged MLX execution and automatic unified-memory sizing.

## Port status

`mia/core/hooks/metal.py` provides MLX callable wrappers, request-row scope primitives, hidden-state extraction, and steering operations. No supplied source contract defines how to map MIA's worker-request delivery lifecycle into Metal's scheduler. Therefore the primitives are not installed by the public APIs yet; current integration does not claim hidden-state delivery, Q/K capture, or effective public-API steering on Metal.

## Evidence used

Inspected `mia/llm.py`, `mia/core/_plugin.py`, `mia/core/hooks/{install,install_hs,install_steer,ops,hosts,registry}.py`, the three MIA workers, `pyproject.toml`, `setup.py`, and supplied target sources under `platform_reference/source/vllm_metal/` and `platform_reference/source/mlx/`.
