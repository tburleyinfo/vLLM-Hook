# Translation experiment (no-graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Run focused tests and live Attention Tracker generation on Apple Silicon with the selected quantized checkpoint.
- Verify dependency resolution for the Metal extra on the target Apple Silicon environment.
- Implement and verify activation steering, hidden-state capture and optional precomputed Q/K score capture on vLLM-Metal.
- Extend and verify capture support for other MLX model families and the Metal paged-attention path.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
