# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Verify vLLM-Metal worker RPC/plugin integration and real Granite Attention Tracker generation on Apple Silicon.
- Implement and validate exact request attribution for concurrent and chunked-prefill requests and server/client capture.
- Validate repeated offline disk-run lifecycle and implement server/client artifacts.
- Implement hidden-state capture and activation steering through the existing public APIs on vLLM-Metal.
- Pin and validate the target dependency profile and add real-runtime behavioral and lifecycle tests.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
