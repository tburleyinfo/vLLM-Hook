# Translation experiment (no-graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Wire request-scoped Q/K capture and probe delivery through MiaLLM and the vLLM-Metal worker/model-runner lifecycle.
- Implement activation steering through the existing steer API on vLLM-Metal.
- Run the Attention Tracker integration and lifecycle tests on Apple Silicon with the target dependencies.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
