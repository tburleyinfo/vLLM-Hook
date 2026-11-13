# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Connect Metal worker/runner lifecycle and scheduler request identity to MIA capture_qk and cleanup.
- Reconstruct each request's full cached K prefix and apply existing last-token capture semantics.
- Deliver MLX-derived probes through the existing RequestOutput/disk artifact contracts and preserve batching/abort behavior.
- Validate selected checkpoint quantization and run Apple-Silicon end-to-end generation, capture, and analysis tests.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
