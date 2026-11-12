# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Verify Metal worker-extension RPC installation and end-to-end Q/K capture on Apple Silicon.
- Preserve exact request association for batching and implement chunked-prefill key reconstruction.
- Integrate Metal save-to-disk and served MiaClient delivery.
- Port and validate Metal steering, hidden-state capture, and remaining workers.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
