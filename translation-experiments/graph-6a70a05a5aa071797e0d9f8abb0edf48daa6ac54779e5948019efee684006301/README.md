# Translation experiment (graph)

Generation: **pass**. Deliverables: **incomplete**. Validation: **skipped**.

Remaining work:
- Implement and validate Metal hidden-state capture and activation steering through the existing public APIs.
- Integrate Q/K capture with vLLM-Metal paged attention and server/remote-worker lifecycles.
- Run the added tests and selected Attention Tracker example on the target Apple-Silicon environment.

Generation success does not establish runtime correctness. Skipped validation is not a passing functional test.

Implementation and usage notes are in the PR files. experiment.json records the original patch hash and results. discovery/ contains translator-generated research documents, when present.
