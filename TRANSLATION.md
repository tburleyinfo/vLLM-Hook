# vLLM-Metal translation: Attention Tracker

## Implemented

- Added an optional `metal` dependency profile alongside the existing CPU and
  CUDA profiles; those profiles and their pins remain unchanged.
- Added `mia.backends.vllm_metal.attention_tracker`, a target-specific Q/K
  observation primitive for vLLM-Metal's real SDPA runtime. It observes
  post-RoPE MLX Q/K arrays, converts them with an MLX-first float32 cast, and
  splits packed requests into the same flattened query/key tensor contract
  consumed by MIA's existing analyzer.
- Added contract tests for request segmentation, captured tensor shapes and
  malformed input rejection.
- Existing `MiaLLM`, analyzer APIs and CPU/CUDA capture paths are unchanged.
  Generation in the target design remains vLLM generation via vLLM-Metal.
- The low-level Metal observer deliberately records only fresh, full prefill
  Q/K. Decode and continuation-prefill passes cannot supply complete key
  prefixes at this boundary and are not represented as complete captures.

## Running checks and usage

Run the backend contract tests in the repository's existing test environment:

```sh
python -m pytest tests/use_cases/test_metal_attntracker.py
```

On Apple Silicon, install the Metal profile alongside the repository:

```sh
python -m pip install -e '.[metal]'
```

Set the target runtime environment before launching the engine:

```sh
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
export VLLM_PLUGINS=metal
```

The existing end-to-end Attention Tracker entry point is
`python examples/demo_attntracker.py`; it uses `MiaLLM` and the established
analyzer API. **End-to-end Metal capture is not wired yet**, so do not expect
that demo to capture successfully on Metal. The tests above exercise only the
new tensor-contract helper, not a Metal engine or model.

## Checks actually performed

- Inspected the repository's capture worker, analyzer, `MiaLLM` lifecycle and
  Attention Tracker demo.
- Inspected the supplied vLLM-Metal worker, model runner, paged-attention
  wrapper, SDPA preparation and attention context sources.
- Added tests; test execution has not been performed in this environment.

## Remaining implementation work

- Connect the target capture primitive to `MiaLLM`'s existing worker extension
  and probe delivery lifecycle.
- Associate packed Q/K captures with request IDs, including scheduled batches,
  prefill chunking, cancellation and cache reuse, without leaking data between
  requests.
- Implement steering on Metal only if it is required by a selected MIA use case;
  Attention Tracker itself has no steering behavior, and existing steering
  remains unchanged on its current backends.
- Run the end-to-end Attention Tracker demo and API/lifecycle parity checks on
  Apple Silicon with vLLM-Metal installed.
- Verify model-specific attention coverage (including non-SDPA attention),
  dtype handling, memory/performance behavior and cleanup on engine teardown.

The existing CUDA/CPU implementation is retained. The work above remains
incomplete; capture and analysis on the Metal backend must not be considered
supported until these integrations are implemented and tested.
