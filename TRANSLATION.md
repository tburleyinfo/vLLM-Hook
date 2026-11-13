# vLLM-Metal translation

## Implemented

- Added the `metal` optional dependency profile without changing the existing
  CPU/CUDA dependencies or installation paths.
- `MiaLLM` keeps the public API and uses real `vllm.LLM.generate` calls on
  vLLM-Metal. For `capture_qk`, a scoped Qwen2 backend observes the post-RoPE
  query/key arrays at the MLX attention call used by vLLM-Metal. It exports
  CPU tensors in the existing MIA probe shape; `AttntrackerAnalyzer` remains
  the analysis implementation.
- `save_to_disk=True` writes the existing `qk.pt` artifact contract, so the
  existing disk-analysis route remains available.
- The same backend applies existing `steer` configs to the Qwen2 residual
  stream immediately before `post_attention_layernorm`. It supports the
  existing `add_vector` and `adjust_rs` methods, layer selection, phase and
  position options.
- Q/K capture supports both raw-Q/K and selected-head score payloads.
- Instrumentation is installed only around the `generate` call and restored
  in `finally`; calls without hooks retain normal vLLM-Metal model execution.
- Added numeric layout tests and a runnable Attention Tracker example.

The bridge is specific to the MLX Qwen2 model implementation because that is
the selected model and the model source supplied for vLLM-Metal uses
`mlx_lm.models.qwen2.Attention`. It does not replace the target model's
generation path with standalone MLX generation.

## Run Attention Tracker

Use an Apple Silicon environment with compatible vLLM, vLLM-Metal, MLX and
MLX-LM packages installed. From the repository root, install the Metal extra
and test group:

```sh
uv sync --extra metal --group test
```

Run the selected model with the target configuration:

```sh
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python examples/demo_attntracker_metal.py
```

The script registers MIA's existing plugins, loads
`model_configs/attention_tracker/Qwen2-1.5B-Instruct.json`, captures the two
prompts' selected heads, and prints the analyzer's per-prompt `score`. The
same capture can be used from the existing API:

```python
outputs = llm.generate(prompts, sampling_params, use_hook=True)
result = llm.analyze(
    probes=outputs[0].probes,
    analyzer_spec={"input_range": ranges, "attn_func": "sum_normalize"},
)
```

## Checks and remaining work

- Inspected the repository's `MiaLLM`, Q/K worker, analyzer, selected model
  config and notebook; inspected vLLM-Metal's model lifecycle, runner, MLX
  tensor bridge and supplied MLX-LM Qwen2 implementation.
- Added `tests/use_cases/test_attntracker_metal_backend.py` for Q/K layout
  conversion. **Tests and an Apple Silicon/vLLM-Metal end-to-end run were not
  executed in this environment.**
- The backend currently targets Qwen2, the selected model family. Hidden-state
  capture and broader model-family support remain unimplemented. Multiprompt
  generation depends on the non-paged runner's
  sequential prefill behavior; verify request-to-probe ordering against the
  target runtime before treating it as validated.
- Verify capture, disk round-trip, steering output behavior, and cleanup on
  the target Apple Silicon stack before declaring runtime parity.
