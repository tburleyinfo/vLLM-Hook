# Translation experiment (no-graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Run the vLLM-Metal dependency resolution, tests, and Granite 3.1 8B example on a supported Apple Silicon environment.
- Verify Metal capture mapping and parity for request scheduling, cache behavior, delivery lifecycle, and Attention Tracker scores.
- Implement and validate Metal steering through the existing public API.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
