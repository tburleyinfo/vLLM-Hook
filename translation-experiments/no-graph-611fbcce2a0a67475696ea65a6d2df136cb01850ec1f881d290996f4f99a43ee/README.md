# Translation experiment (no-graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Run the new Q/K layout tests and an end-to-end Apple Silicon vLLM-Metal Attention Tracker test, including multiprompt ordering, disk artifact analysis, steering output, and hook cleanup.
- Implement and validate hidden-state capture and non-Qwen2 model support if those repository APIs are required on vLLM-Metal.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
