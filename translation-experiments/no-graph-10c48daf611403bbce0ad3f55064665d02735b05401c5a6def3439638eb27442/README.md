# Translation experiment (no-graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Connect Metal Q/K capture to the existing MiaLLM worker-extension and probe-delivery lifecycle.
- Implement and verify per-request association for packed, chunked, cached, batched and cancelled requests.
- Run end-to-end Attention Tracker API and lifecycle checks on Apple Silicon using vLLM-Metal.
- Validate supported model attention variants, dtype conversion, resource behavior and engine teardown.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
