# vLLM-Metal translation

## Implemented for the selected use case

- Added the optional `metal` dependency profile with the supplied target library version. CPU and CUDA profiles, pins, install sources, and the existing public `MiaLLM` API remain intact.
- Added Metal Q/K capture for Granite's non-paged MLX attention path. It observes post-RoPE Q/K and the full updated key cache while delegating attention and token generation to the original model call.
- Request scoping uses each request's existing `SamplingParams.extra_args`; probe tensors are stored in the repository's existing Q/K entry format and delivered through the current MIA worker path.
- The existing `AttntrackerAnalyzer`, `MiaLLM.generate`, and `MiaLLM.analyze` behavior remains the analysis path. Added a unit-test module for tensor shape/layout and probe-entry contracts.

## Run on Apple Silicon

Use an Apple Silicon Mac with the vLLM-Metal plugin and MIA installed in the same environment. Install this repository with its Metal profile and test tools:

```sh
uv sync --extra metal --group test
```

Run the selected Granite Attention Tracker example from the repository root:

```sh
python examples/demo_attntracker_metal.py
```

The script selects `VLLM_PLUGINS=metal` and `VLLM_USE_V2_MODEL_RUNNER=0` (required by the Metal worker), disables vLLM engine multiprocessing and Metal paged attention, and enables eager execution because Metal does not use MIA's CUDA graph hooks. It uses the supplied Granite W4A16 model configuration, performs real `MiaLLM.generate` inference, checks that Q/K probes were returned, and analyzes the instruction/data token ranges. The first run requires the model files to be available to the configured Hugging Face model loader.

The focused tests are:

```sh
pytest -q tests/backends/metal/test_qk_capture.py
```

## Checks and remaining work

The source inspection and contracts are recorded in `translation_discovery/attention_tracker.md`. No tests or model run were executed in this environment: it does not provide a shell/test runner or an Apple Silicon vLLM-Metal runtime. Consequently, the W4A16 model-load path and end-to-end worker-extension lifecycle remain unverified here.

This translation is incomplete. Metal steering through the existing `SteerWorker` API is not implemented. Q/K capture is implemented for the selected non-paged Granite path; paged Metal capture and other MLX model architectures are not implemented. The Attention Tracker's expected runtime behavior must be verified on Apple Silicon before treating the port as validated.
