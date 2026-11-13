# vLLM-Metal translation

**Status: incomplete.** CPU/CUDA files, APIs, dependency profiles and backend installation
paths remain in place. A separate `metal` dependency extra and an MLX Llama Q/K capture adapter
were added. The adapter wraps the existing Q/K projections, applies the model's RoPE, exports
an MIA-compatible probe payload, and restores original projections on `close()`. This is a
capture primitive only; it is not connected to `MiaLLM` or the Metal worker lifecycle.

## Attention Tracker usage and status

The selected config is
`model_configs/attention_tracker/Meta-Llama-3-8B-Instruct.json`. Install the target profile on
Apple Silicon from the repository root with:

```bash
python -m pip install -e '.[metal]'
```

The target runtime settings for this task are:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
export VLLM_PLUGINS=metal
```

Do not treat the following as a complete use-case invocation yet: although
`MetalAttentionCapture(model, layer_to_heads)` can create the existing Q/K probe payload for
`AttntrackerAnalyzer`, the inspected vLLM-Metal source provides no supported point at which
the adapter can obtain the scheduled request's extra args, bracket the correct model forward,
and return the payload through `MiaLLM.generate`'s normal `probes` lifecycle. The target's
non-paged Llama generation itself remains owned by vLLM-Metal; standalone MLX generation is
not used as a fallback. End-to-end runnable Attention Tracker usage requires the integration
listed below and has not been claimed.

The existing analyzer hand-off is:

```python
stats = llm.analyze(
    analyzer_spec={"input_range": ranges, "attn_func": "sum_normalize"},
    probes=outputs[0].probes,
)
```

The selected model requires gated-model access and sufficient Apple unified memory. Target
versions available from the inspected source are vLLM-Metal 0.2.0 and mlx-lm 0.31.3; `mlx` is
an unpinned dependency of this added profile because no corresponding source version pin was
provided.

## Checks actually performed

- Inspected the repository's Attention Tracker config, notebook, Q/K worker, analyzer, public
  `MiaLLM` API and delivery path.
- Inspected the supplied vLLM-Metal platform, worker, model runner/lifecycle, SDPA wrapper,
  Llama model source and hook-primitives note.
- Added `tests/metal/test_attention_tracker_backend.py` to check the adapter's expected Q/K
  probe schema and its compatibility with the existing analyzer.
- No tests, builds, package installation, model generation or Apple Silicon runtime checks
  were performed in this environment.

## Remaining implementation work

- Connect request metadata, capture scope, model lifecycle and Q/K probe delivery to the
  existing `MiaLLM` and server APIs on vLLM-Metal without changing request behavior or
  teardown semantics.
- Implement activation steering through the existing `steer` API on the target backend.
- Exercise lifecycle, request routing, capture values/RoPE positions, analysis, generation and
  cleanup against an Apple Silicon installation of the declared target libraries.

Detailed source findings are in
[`translation_discovery/attention_tracker-metal.md`](translation_discovery/attention_tracker-metal.md).
