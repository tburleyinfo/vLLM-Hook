# MLR-15 / MLR-40 dependency map

Source: immutable MIA commit `3c4f6874cd007ca39fd76c40b8c481b573798cf3`.
The graph is code-grounded but awaits human validation. The experiment must not
describe it as human-validated until review is recorded against its hash.

| Dependency | Source evidence | Metal disposition | Required evidence |
| --- | --- | --- | --- |
| Example/API | `examples/demo_attntracker.py`: `MiaLLM`, `capture_qk`, `attn_tracker` | Adapted API needed | Executable generate → capture → analyze, then notebook |
| Worker registration | `mia/__init__.py`, `mia/registry.py`, `mia/llm.py` | Missing compatible integration | Backend worker installs hooks during actual generation |
| Request lifecycle | `mia/core/runner.py`: `require_v2_runner`, `install_request_arg_stash`, `step_view` | Blocked for direct reuse | Source explicitly accepts vLLM CPU/GPU V2 runners; installed Metal uses `vllm_metal.v1.model_runner` |
| Capture point | `mia/workers/qk_capture_worker.py`: `install_hooks`, `qkv_hook`, `register_forward_hook` | Adaptation required | Post-positional-transform Q/K, selected layer/head, request identity and prefill position |
| Cache history | Same worker: `_prepend_cached_keys`, `key_cache_from_layer_kv`, `append_k_prefix` | Adaptation required | Metal paged-cache layout and complete prompt keys; prefix reuse remains a later check |
| Probe delivery | `mia/llm.py`: `_merge_probes`; `mia/artifacts.py`: `unpack_qk` | Adaptation required | One request entry per prompt, no reordered or mixed histories |
| Scoring | `mia/analyzers/attention_tracker_analyzer.py`: `compute_attention_from_qk`, `attn2score` | Reuse candidate, unverified | GQA head expansion, scale, softmax and normalized instruction score |
| Heads/configuration | `model_configs/attention_tracker/*`, CPU fixture | Adapted diagnostic config | Layer 0/head 0 on both backends; this is a capture/parity experiment, not validated injection-detection quality |
| Token spans | Example and CPU fixture | Reuse protocol | Identical prompt token IDs; half-open instruction/data spans derived from tokenizer offsets |
| Metal architecture | Installed `vllm_metal/v1/worker.py`, `v1/model_runner.py`, `paged_attention_backend/mha.py`, `metal_kernel_backend/paged_attention.py` | Available backend, capture unverified | Backend file hashes and installed distribution versions preserved per run |

MLR-48 establishes server generation only. MLR-50's hidden-state pass is not an
Attention Tracker pass. MLR-53 provides the existing atomic CPU fixture. MLR-7
requires structural and numerical comparison but currently supplies no numeric
tolerance values: the manifest's values are provisional and need review.

The experiment fixes source code, target starting tree, translator model/effort,
instructions and evaluator across conditions. A two-condition pilot is not
statistical evidence of improvement. Preserve failures and human corrections.
