"""Prompt Q/K capture for the non-paged vLLM-Metal runner."""
from __future__ import annotations

import weakref
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


_WRAPPED_ATTENTION_TYPES: dict[type, Any] = {}
_WORKER_REF_ATTR = "_mia_metal_worker_ref"
_LAYER_ATTR = "_mia_metal_layer_num"


def requested_layers(output_qk: Any, num_layers: int) -> set[int]:
    """Resolve the public output_qk selector to valid model layer indices."""
    if output_qk is True:
        return set(range(num_layers))
    if isinstance(output_qk, dict):
        layers = {int(layer) for layer in output_qk}
    elif isinstance(output_qk, (list, tuple)):
        layers = {int(layer) for layer in output_qk}
    else:
        layers = set()
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(f"output_qk contains layer(s) outside [0, {num_layers}): {invalid}")
    return layers


def stack_qk_capture(q_rows: list[torch.Tensor], k_rows: list[torch.Tensor],
                     mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the existing probe tensor shapes from per-forward Q and prefix-K rows."""
    if not q_rows or not k_rows or len(q_rows) != len(k_rows):
        raise ValueError("Q/K capture requires matching, non-empty query and key rows")
    if mode == "last_token":
        q = torch.stack([row.reshape(-1) for row in q_rows], dim=0)
    elif mode == "all_tokens":
        q = pad_sequence(q_rows, batch_first=True)
    else:
        raise ValueError(f"hookq_mode must be 'last_token' or 'all_tokens', got {mode!r}")
    return q, pad_sequence(k_rows, batch_first=True)


def _mlx_to_torch(value) -> torch.Tensor:
    import mlx.core as mx
    import numpy as np

    mx.eval(value)
    numpy_dtype = mx.float16 if value.dtype == mx.float16 else mx.float32
    return torch.from_numpy(np.asarray(value.astype(numpy_dtype))).contiguous()


def _captured_qk(attention, hidden, cache, layer_num: int):
    import mlx.core as mx

    q_heads = int(getattr(attention, "n_heads",
                          getattr(attention, "num_attention_heads", 0)))
    kv_heads = int(getattr(attention, "n_kv_heads",
                            getattr(attention, "num_key_value_heads", 0)))
    if not q_heads or not kv_heads:
        raise RuntimeError(f"Metal attention layer {layer_num} does not expose Q/K head counts")

    q_proj = getattr(attention, "q_proj", None)
    k_proj = getattr(attention, "k_proj", None)
    rope = getattr(attention, "rope", None) or getattr(attention, "rotary_emb", None)
    if q_proj is None or k_proj is None or rope is None:
        raise RuntimeError(
            f"Metal attention layer {layer_num} must expose q_proj, k_proj, and RoPE "
            "to capture post-RoPE Q/K"
        )

    batch, seq_len, _ = hidden.shape
    q = q_proj(hidden).reshape(batch, seq_len, q_heads, -1)
    k = k_proj(hidden).reshape(batch, seq_len, kv_heads, -1)
    if hasattr(attention, "q_norm"):
        q = attention.q_norm(q)
    if hasattr(attention, "k_norm"):
        k = attention.k_norm(k)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)

    cache_offset = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0
    offset = cache_offset - seq_len if cache_offset >= seq_len else 0
    if cache is not None and cache_offset < seq_len:
        raise RuntimeError(
            f"Metal Q/K capture found an inconsistent KV-cache offset "
            f"({cache_offset} for {seq_len} query tokens)"
        )
    q = rope(q, offset=offset)
    k = rope(k, offset=offset)
    q_rows = q.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

    if cache is not None and getattr(cache, "keys", None) is not None:
        full_k = cache.keys[..., :cache_offset, :]
    else:
        full_k = k
    k_rows = full_k.transpose(0, 2, 1, 3).reshape(batch, full_k.shape[2], -1)

    return [
        (_mlx_to_torch(q_rows[row]), _mlx_to_torch(k_rows[row]))
        for row in range(batch)
    ]


def _wrap_attention_type(attention_type: type) -> None:
    if attention_type in _WRAPPED_ATTENTION_TYPES:
        return
    original_call = attention_type.__call__
    _WRAPPED_ATTENTION_TYPES[attention_type] = original_call

    def wrapped(attention, *args, **kwargs):
        output = original_call(attention, *args, **kwargs)
        worker_ref = getattr(attention, _WORKER_REF_ATTR, None)
        worker = worker_ref() if worker_ref is not None else None
        active = getattr(worker, "_mia_metal_active", None) if worker is not None else None
        if active is None:
            return output
        hidden = args[0] if args else kwargs.get("x")
        cache = args[2] if len(args) > 2 else kwargs.get("cache")
        if hidden is None:
            raise RuntimeError("Metal attention call did not provide its hidden-state input")
        layer_num = int(getattr(attention, _LAYER_ATTR))
        if layer_num not in active["layers"]:
            return output
        captured = _captured_qk(attention, hidden, cache, layer_num)
        if len(captured) != 1:
            raise RuntimeError(
                "The non-paged Metal prefill hook expected one request per forward call"
            )
        q, k = captured[0]
        worker._record_metal_qk(active, layer_num, q, k)
        return output

    attention_type.__call__ = wrapped


def install_prompt_qk_capture(worker) -> None:
    """Attach request-aware Q/K capture to each supported MLX attention module."""
    runner = worker.model_runner
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("Cannot install Metal Q/K capture before the model is loaded")

    from vllm_metal.attention.patching import find_layers, find_attn_attr

    layers = find_layers(model)
    if not layers:
        raise RuntimeError("vLLM-Metal model contains no transformer layers")

    worker._mia_metal_module_names = {}
    matched = 0
    for layer_num, layer in enumerate(layers):
        attn_attr = find_attn_attr(layer)
        if attn_attr is None:
            continue
        attention = getattr(layer, attn_attr)
        if not (hasattr(attention, "q_proj") and hasattr(attention, "k_proj")):
            continue
        object.__setattr__(attention, _LAYER_ATTR, layer_num)
        object.__setattr__(attention, _WORKER_REF_ATTR, weakref.ref(worker))
        _wrap_attention_type(type(attention))
        worker._mia_metal_module_names[layer_num] = f"model.layers.{layer_num}.{attn_attr}"
        matched += 1

    if not matched:
        raise RuntimeError("No split-projection MLX attention layers matched for Q/K capture")
    worker._mia_metal_num_layers = len(layers)
    args = getattr(model, "args", None)
    model_args = args if args is not None else getattr(model, "config", None)
    if model_args is None:
        raise RuntimeError("Cannot resolve model dimensions for Metal Q/K capture")
    num_heads = int(getattr(model_args, "num_attention_heads"))
    num_kv_heads = int(getattr(model_args, "num_key_value_heads", num_heads))
    hidden_size = int(getattr(model_args, "hidden_size"))
    head_dim = int(getattr(model_args, "head_dim", hidden_size // num_heads))
    worker._mia_metal_qk_config = {
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "attention_multiplier": float(
            getattr(model_args, "attention_multiplier", 1.0 / (head_dim ** 0.5))
        ),
    }


def install_request_capture_context(worker) -> None:
    """Route non-paged prompt forwards through MIA's request-specific capture state."""
    runner = worker.model_runner
    if getattr(runner, "_mia_metal_prefill_wrapped", False):
        return
    original_prefill = runner._prefill_single

    def prefill_single(req_id, token_ids, sampling_params, generator=None):
        extra = getattr(sampling_params, "extra_args", None) or {}
        output_qk = extra.get("output_qk")
        if output_qk is None:
            return original_prefill(req_id, token_ids, sampling_params, generator)
        hooks_on = extra.get("hooks_on", "prefill")
        if hooks_on != "prefill":
            raise RuntimeError(
                "vLLM-Metal Q/K capture currently supports hooks_on='prefill'; "
                f"received {hooks_on!r}"
            )
        mode = extra.get("hookq_mode", "all_tokens")
        if mode not in ("last_token", "all_tokens"):
            raise ValueError(f"Unsupported hookq_mode {mode!r}")
        layers = requested_layers(output_qk, worker._mia_metal_num_layers)
        worker._mia_metal_active = {
            "req_id": str(req_id),
            "extra": extra,
            "mode": mode,
            "layers": layers,
        }
        try:
            result = original_prefill(req_id, token_ids, sampling_params, generator)
            target = (worker._mia_metal_disk_states if extra.get("save_to_disk")
                      else worker._mia_metal_states)
            captured = target.get(str(req_id), {})
            captured_layers = {int(entry["layer_num"]) for entry in captured.values()}
            missing = sorted(layers - captured_layers)
            if missing:
                raise RuntimeError(
                    f"Metal Q/K capture missed requested layer(s) {missing} for "
                    f"request {req_id!r}"
                )
            return result
        finally:
            worker._mia_metal_active = None

    runner._prefill_single = prefill_single
    runner._mia_metal_prefill_wrapped = True
