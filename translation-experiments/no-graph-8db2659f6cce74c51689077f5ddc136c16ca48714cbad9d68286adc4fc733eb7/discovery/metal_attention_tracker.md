# Discovery: Attention Tracker on vLLM-Metal

## Source behavior

- `model_configs/attention_tracker/granite-3.1-8b-instruct.json` selects
  important attention heads and `hookq_mode: last_token`.
- `MiaLLM` sends `output_qk` through the established worker extension contract,
  receives request probes from worker `get_captured_states`, and passes those
  probes to `AttntrackerAnalyzer`.
- The analyzer expects per-layer Q/K vectors with
  `num_attention_heads`, `num_key_value_heads`, `head_dim`, and
  `attention_multiplier` in the config.

## Target correspondence

- `vllm_metal.v1.model_runner.MetalModelRunner` uses the real vLLM engine
  scheduler but executes model forwards through `mlx_lm`.
- Target `mlx_lm.models.granite.Attention.__call__` computes queries and keys,
  invokes one RoPE callable for both, then passes those results into MLX cache
  and scaled dot-product attention.
- Supplied `hook_primitives.md` permits callable-module replacement as a
  possible observation approach, while requiring delegation and lifecycle
  preservation. The added backend wraps Granite's RoPE callable and retains its
  original delegate.
- The target mlx_lm bfloat16 conversion note requires casting to float32 before
  crossing to NumPy; `_mlx_to_torch` follows this rule.

## Scope decisions

- The supplied run configuration uses the Metal non-paged path. Capture hooks
  are scoped to the target runner's prefill and decode forward methods; paged
  attention is explicitly rejected.
- Capture is integrated into the existing worker API and returns existing Q/K
  probe keys; analyzer APIs remain unchanged.
- Steering, hidden states, and full parity with other MIA modes remain
  unimplemented. See `TRANSLATION.md`.
