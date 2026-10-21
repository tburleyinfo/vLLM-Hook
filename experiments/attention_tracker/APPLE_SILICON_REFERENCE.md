# IBM Apple Silicon branch: infrastructure extraction map

Inspected and remotely verified on October 13, 2026. Branch `apple-silicon`, commit
`0bb64d85ccad159e7d46990006be28117cd3df23`.

A complete source snapshot and SHA-256 index are stored under
`experiment-runs/attention-tracker/upstream-apple-silicon-reference/`.
This is a held-out research reference, not input to either translation candidate.
No upstream code has been installed or copied into the shared translation package.

## Reusable mechanisms

- `metal/_hook_plugin.py:31`: selects `worker_extension_cls` before engine configuration finishes. This puts custom methods inside the worker; it does not depend on the driver directly owning the model.
- `metal/hook_llm_metal.py:472`: encode capture builds a hook-enabled engine, invokes `collective_rpc("install_hooks")`, performs generation and checks artifacts. Cleanup shuts down that engine. Ordinary output generation can use a separate base engine.
- `workers/metal/probe_hidden_states_worker_metal.py:28`: a small MLX module wrapper delegates to the original callable, observes its output and returns the output unchanged.
- The same worker at line 78 accesses `self.model_runner.model`, traverses MLX `named_modules()` and installs wrappers. It also has indexed layer lookup using backend `find_layers`.
- `workers/metal/probe_hookqk_worker_metal.py:40`: a wrapper supports pre/post observation. Installation replaces a parent's child and retains the original for restoration at line 652.
- `workers/metal/probe_hidden_states_worker_metal.py:317`: collection is run-scoped and disk-backed. `get_captured_states` flushes and returns `None`; it does not return a per-request tensor payload. `flush_disk` and `_flush_run_cache` persist artifacts.
- `metal/run_utils_metal.py`: loads and merges stored artifacts. General artifact plumbing is reusable; Q/K normalization is task-specific.

## Keep out of the first supplied infrastructure package

- Q/K/V reconstruction in `probe_hookqk_worker_metal.py:385`.
- Positional transforms, KV-cache reconstruction, head selection and Q/K/V formatting at lines 265–430.
- Completed attention analyzer and task notebook.
- Hidden-state/steering task logic if extracting their generic wrapper machinery.

The upstream package intertwines infrastructure and capability implementations.
Extract narrowly and test a generic worker observation rather than providing whole
worker files to the translator.

## Compatibility details that must be tested

1. MLX exposes `named_modules()` here. PyTorch-style `register_forward_hook` is replaced by callable wrappers. Do not equate traversal with hook support.
2. `HookLLMMetal` defaults `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `VLLM_METAL_USE_PAGED_ATTENTION=0` (line 143). The notebook guide documents this configuration. Existing environment values can override these defaults.
3. Q/K installation unwraps an attention module's `_inner` when present (line 519), and has `find_layers` / `find_attn_attr` fallback (line 613 onward). A missing direct projection attribute does not prove that no capture point exists.
4. This is the older `HookLLMMetal` API. It is reference infrastructure, not a replacement for the current MIA source contract. Its saved `qkv_cache` is not the evaluator's current `qk_cache` contract.
5. Its disk delivery is keyed by run IDs. Request ordering and isolation must be established for the current fixture rather than assumed.
6. The current pilot failures have not yet been reproduced against this configuration. Paged attention and `_inner` are diagnostic leads, not proven causes.
7. Source inspection establishes design, not successful execution on the installed backend or numerical parity.

## Next implementation boundary

Build a small current-version worker extension with install/observe/collect/restore
methods. Use an MLX wrapper derived from the generic pattern. Run real generation
and return a generic call observation, without tracker Q/K logic. Verify restoration
and output preservation. Freeze only that tested infrastructure for both conditions.
Record any non-paged-attention configuration explicitly as an experimental setting.

## Primary sources

All links are pinned to the inspected commit:

- [Worker registration](https://github.com/IBM/vLLM-Hook/blob/0bb64d85ccad159e7d46990006be28117cd3df23/vllm_hook_plugins/vllm_hook_plugins/metal/_hook_plugin.py#L31)
- [Engine orchestration](https://github.com/IBM/vLLM-Hook/blob/0bb64d85ccad159e7d46990006be28117cd3df23/vllm_hook_plugins/vllm_hook_plugins/metal/hook_llm_metal.py#L472)
- [Generic observation pattern in hidden-state worker](https://github.com/IBM/vLLM-Hook/blob/0bb64d85ccad159e7d46990006be28117cd3df23/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hidden_states_worker_metal.py#L28)
- [Attention wrapper and held-out capture implementation](https://github.com/IBM/vLLM-Hook/blob/0bb64d85ccad159e7d46990006be28117cd3df23/vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py#L40)
- [Metal notebook setup and backend configuration](https://github.com/IBM/vLLM-Hook/blob/0bb64d85ccad159e7d46990006be28117cd3df23/notebooks/metal/README.md)
