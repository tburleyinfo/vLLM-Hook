# Translation experiment (no-graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Implement Q/K capture on the requested VLLM_METAL_USE_PAGED_ATTENTION=0 MLX execution path.
- Wire MLX Q/K capture through vLLM-Metal worker and MiaLLM/MiaClient request lifecycles, including in-memory probes and disk delivery.
- Implement cached-prefix key reconstruction, full-token capture, chunked prefill, decode, and batched request isolation.
- Integrate request-routed activation steering with MIA's existing worker/API and phase/position semantics.
- Run Apple Silicon end-to-end generation tests for Qwen2-1.5B-Instruct and validate output parity, scores, and lifecycle cleanup.
- Declare a compatible vLLM-Metal version and installation recipe before pinning the new dependency profile.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
