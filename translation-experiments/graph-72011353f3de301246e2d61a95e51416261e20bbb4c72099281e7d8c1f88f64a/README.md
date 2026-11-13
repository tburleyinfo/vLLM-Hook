# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Implement Metal activation steering and hidden-state capture through the existing worker APIs.
- Implement and verify Q/K capture for paged attention, batched decode, disk delivery, and model layouts beyond the selected Mistral case.
- Run the new unit tests and an Apple Silicon vLLM-Metal end-to-end Attention Tracker run; verify probe contracts, analyzer scores, lifecycle cleanup, and generation parity.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
