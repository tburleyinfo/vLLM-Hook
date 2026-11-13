# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Implement Metal activation steering through the existing SteerWorker API.
- Validate vLLM-Metal worker-extension installation, the Granite W4A16 model load, Q/K probe delivery, and analyzer scores on Apple Silicon.
- Implement and validate paged-attention Q/K capture and support for additional MLX model architectures if they are required beyond the selected non-paged Granite use case.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
