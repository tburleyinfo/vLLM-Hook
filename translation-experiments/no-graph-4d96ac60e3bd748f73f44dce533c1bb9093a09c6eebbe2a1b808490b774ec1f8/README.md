# Translation experiment (no-graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Implement vLLM-Metal hidden-state capture through the existing public API.
- Implement vLLM-Metal activation steering through the existing public API.
- Support Q/K capture with paged attention and decode/both hooks_on modes.
- Run the added tests and selected use case on Apple Silicon with vLLM-Metal.
- Confirm vLLM-Metal dependency distribution names and compatible versions, then pin and verify the metal extra.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
