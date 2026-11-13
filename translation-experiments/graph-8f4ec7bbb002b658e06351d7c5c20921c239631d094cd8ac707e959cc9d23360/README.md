# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Run the added targeted tests and Attention Tracker use case on Apple Silicon with vLLM-Metal.
- Verify Metal Q/K probe delivery, analysis, disk persistence, and cleanup at runtime.
- Verify Metal activation steering on a supported model for batched and sequential decode.
- Implement and test Q/K capture and steering when vLLM-Metal paged attention is enabled.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
