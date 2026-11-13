# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Install the MLX Q/K capture primitive in the vLLM-Metal Qwen2 runner and associate captured arrays with the correct generation requests.
- Wire Metal captures through MiaLLM.generate and preserve in-memory and disk probe lifecycle behavior.
- Run unit and Apple-Silicon end-to-end tests for capture, analyzer output, lifecycle, and behavioral parity.
- Implement and validate other requested Metal backend capabilities, including steering, before claiming repository-wide parity.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
