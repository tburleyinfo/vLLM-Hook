# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Implement and preserve Q/K capture semantics for decode, batched/chunked, paged-attention, and durable/on-disk capture paths.
- Run the added tests and validate the Metal Q/K capture and steering bridges, dependency extra, and end-to-end Attention Tracker flow on a compatible Apple Silicon vLLM-Metal environment.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
