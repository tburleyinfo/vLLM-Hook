# Attention Tracker Metal discovery

## Repository path

- `MiaLLM` is the public offline API (`mia/llm.py`). It injects worker behavior into `SamplingParams.extra_args`, calls the real `vllm.LLM.generate`, and exposes in-memory probes on request outputs or disk-backed analysis through `analyze`.
- Attention Tracker analysis (`mia/analyzers/attention_tracker_analyzer.py`) consumes `probes["config"]` with `num_attention_heads`, `num_key_value_heads`, `head_dim`, and `attention_multiplier`, plus per-layer `q` and `k_all` tensors in `probes["qk_cache"]`.
- The existing Q/K worker (`mia/workers/qk_capture_worker.py`) and install path (`mia/core/hooks/install.py`) expect vLLM's GPU worker, PyTorch attention modules, CUDA graph ops, and the CUDA QK aperture. These do not describe the selected Metal execution path.
- The selected model's config identifies four significant heads and requests last-token Q capture.
- The Attention Tracker example previously had no Meta-Llama-3 token-range recipe. It now derives ranges using offsets from the model's rendered chat template rather than assuming offsets for a different model's template.

## Supplied target-source findings

- `platform_reference/source/vllm_metal/v1/model_lifecycle.py` loads generation models with `mlx_lm.load` and installs the model/tokenizer on the Metal runner.
- `platform_reference/source/vllm_metal/v1/worker.py` defines a dedicated `MetalWorker`, not the upstream CUDA worker patched by MIA's current `patch_worker_load_model`.
- With paged attention disabled, `platform_reference/source/vllm_metal/v1/model_runner.py` performs model forwards through the MLX model. The selected `mlx_lm/models/llama.py` attention computes projected, RoPE-applied query/key arrays and calls `scaled_dot_product_attention(queries, keys, values, ...)`; the key array includes cached keys after `cache.update_and_fetch`.
- The supplied `hook_primitives.md` describes general hook primitive considerations, but does not provide an MIA-to-vLLM-Metal request-lifecycle bridge.
- The supplied hook note records that MLX bfloat16 cannot cross the NumPy boundary directly. The Metal backend casts bfloat16 arrays to MLX float32 before NumPy conversion and then stores the exact representable values in CPU bfloat16 tensors for the existing analyzer path.
- The target source reports vLLM-Metal version `0.2.0` in `vllm_metal/__init__.py`; no dependency manifest or end-to-end MIA integration recipe was supplied.

## Implemented boundary

The new `mia.backends.metal.attention_tracker` module captures the observed MLX Q/K call inputs under an explicit request scope and creates the analyzer's probe shape. Its runner adapter consumes `output_qk` from the inspected runner's `SamplingParams` during non-paged prefill, leaves the model forward intact, and returns captures by request ID. MIA's plugin path attaches these to generated outputs, and `MiaLLM` persists them in the existing `qk.pt` layout. These connections are source-based and have not yet been exercised in a Metal runtime; no end-to-end correctness claim is made.
