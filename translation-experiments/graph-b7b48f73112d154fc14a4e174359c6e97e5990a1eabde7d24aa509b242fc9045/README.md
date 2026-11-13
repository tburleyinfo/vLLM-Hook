# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Connect Metal Q/K capture to the existing MiaLLM and server worker lifecycle, including per-request routing, cached-prefix reconstruction, chunked prefill, artifact delivery, and cleanup.
- Capture Q/K on the requested non-paged vLLM-Metal path (VLLM_METAL_USE_PAGED_ATTENTION=0).
- Implement existing public activation-steering behavior on MLX residual streams.
- Validate target dependency resolution and run unit/integration tests on Apple Silicon.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
