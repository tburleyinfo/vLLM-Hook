# Translation experiment (no-graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Implement and behaviorally test Metal activation steering and hidden-state capture through the existing public APIs.
- Implement and test Metal batch, decode/both-phase capture, disk delivery, server-mode delivery, and paged-attention lifecycle behavior.
- Run the selected Mistral Attention Tracker end to end on a compatible Apple Silicon vLLM-Metal runtime and verify numerical/output contracts.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
