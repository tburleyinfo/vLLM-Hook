# vLLM-Metal translation

## Implemented

- Added an additive `metal` dependency profile in `pyproject.toml`; the existing CPU and CUDA profiles, dependency pins, `setup.py` install requirements, and plugin entry points remain unchanged. The profile includes vLLM-Metal 0.2.0 and its MLX generation/model packages.
- Added MLX callable-module hook primitives in `mia/core/hooks/metal.py`. The module discovers common MLX decoder-layer paths, offers a per-forward request-row scope, captures selected layer outputs as CPU tensors, applies the existing activation-steering configuration and vector formats to MLX outputs, and releases runtime references with `close()`.
- Added unit coverage for layer discovery, layer selection, wrapper behavior while unarmed, request-scope lifecycle, and cleanup.

## Run

On an Apple Silicon Mac with a compatible Python environment:

```sh
uv sync --extra metal
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(model="mlx-community/Qwen2.5-0.5B-Instruct")
result = llm.generate(
    ["Write one sentence about the ocean."],
    SamplingParams(max_tokens=32, temperature=0),
)
print(result[0].outputs[0].text)
PY
```

Generation remains vLLM generation: vLLM-Metal selects its vLLM worker/model runner, which delegates model execution to MLX. This example is a backend smoke-use pattern, not a claim that the instrumentation path is complete. The new hook primitives require an active `MetalHookRuntime.request_scope(...)`; the repository does not yet connect that scope to vLLM-Metal's per-request forward lifecycle or attach collected probes to `RequestOutput`.

## Checks performed

- Inspected MIA's public generation, capture, steering, analysis, worker-install, and dependency surfaces.
- Inspected the supplied vLLM-Metal platform, worker, model runner, lifecycle, and adapter sources, plus MLX module and model-callable sources.
- Added tests in `tests/test_metal_hooks.py`. They were not executed in this environment.
- No Apple Silicon runtime, MLX installation, or vLLM-Metal generation check was available here.

## Remaining implementation work

- Connect `MetalHookRuntime` to the actual vLLM-Metal runner lifecycle and request scheduling. The current public APIs do not pass a `MetalRequest` scope to the added hook primitives.
- Route captured hidden-state tensors through the existing artifact/delivery contract so `MiaLLM.generate()`, `MiaClient`, `analyze()`, and save-to-disk runs observe the same shapes, metadata, and cleanup behavior as CPU/CUDA.
- Implement post-RoPE Q/K capture. MLX model forwards compute Q/K as local intermediate values; wrapping only decoder-block callables does not expose them. No Q/K support is claimed.
- Verify steering numerical equivalence, token/phase routing, batched and chunked prefill behavior, cache/lifecycle handling, supported model families, and served-generation behavior on Apple Silicon.

These gaps are incomplete work, not implemented capabilities. Existing CPU/CUDA implementations remain intact.
