# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Run the added Metal capture/analyzer test and the selected Attention Tracker example on a supported Apple Silicon vLLM-Metal runtime; verify worker-extension dispatch and end-to-end probe delivery.
- Validate compatible mlx and mlx-lm versions against vllm-metal==0.2.0 and vllm==0.29.0, then pin them if required.
- Implement and validate Metal steering and non-QK capture workers if those existing public APIs are required beyond the selected Attention Tracker use case.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
