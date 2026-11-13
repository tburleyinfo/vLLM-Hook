# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Run focused Metal backend tests and the Attention Tracker demo on Apple Silicon with vLLM-Metal.
- Validate save_to_disk artifact lifecycle and batched/decode capture on the target runtime.
- Implement and test Metal activation steering, hidden-state capture, score-only Q/K capture, and paged-attention Q/K capture before claiming those repository capabilities are translated.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
