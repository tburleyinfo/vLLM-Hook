"""Q/K capture from vLLM-Metal's non-paged MLX Mistral attention path."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import torch

from mia.workers.qk_capture_worker import (
    _resolve_score_heads,
    append_k_prefix,
    compute_head_scores,
    new_qk_entry,
)


@dataclass(frozen=True)
class _RequestCapture:
    req_id: str
    extra: dict
    phase: str
    token_count: int


@dataclass
class _ActiveRequests:
    contexts: list[_RequestCapture | None]
    sequential: bool = False
    max_layer: int = -1
    current: int = 0
    last_layer: int = -1


_ACTIVE: dict[int, _ActiveRequests] = {}
_WRAPPED_CLASSES: dict[type, Any] = {}


def _capture_requested(context: _RequestCapture | None, layer: int) -> bool:
    if context is None:
        return False
    extra = context.extra
    output = extra.get("output_qk")
    if output is None:
        return False
    hooks_on = extra.get("hooks_on", "prefill")
    if hooks_on not in ("both", context.phase):
        return False
    if isinstance(output, dict):
        return any(int(item) == layer for item in output)
    if isinstance(output, (tuple, list)):
        return layer in {int(item) for item in output}
    return True


def _mlx_float32_numpy(array, mx) -> np.ndarray:
    """Convert MLX tensors through float32, including MLX bfloat16 arrays."""
    mx.eval(array)
    return np.asarray(mx.contiguous(array).astype(mx.float32))


def _request_extra(params) -> dict:
    return dict(getattr(params, "extra_args", None) or {})


def _request(req_id: str, params, phase: str, token_count: int) -> _RequestCapture | None:
    extra = _request_extra(params)
    if extra.get("output_qk") is None:
        return None
    hooks_on = extra.get("hooks_on", "prefill")
    if hooks_on not in ("prefill", "decode", "both"):
        raise ValueError(f"unsupported hooks_on value for Metal capture: {hooks_on!r}")
    mode = extra.get("hookq_mode", "all_tokens")
    if mode not in ("all_tokens", "last_token"):
        raise ValueError(f"unsupported hookq_mode for Metal capture: {mode!r}")
    if extra.get("qk_capture", "qk") not in ("qk", "score"):
        raise ValueError(
            f"unsupported qk_capture value for Metal capture: {extra['qk_capture']!r}"
        )
    return _RequestCapture(str(req_id), extra, phase, int(token_count))


def _store_qk(worker, layer: int, q, k, context: _RequestCapture, mx) -> None:
    extra = context.extra
    mode = extra.get("hookq_mode", "all_tokens")
    q_rows = q[-1, :] if mode == "last_token" else q
    k_tensor = torch.from_numpy(_mlx_float32_numpy(k, mx).copy())
    bucket = worker._disk_states if extra.get("save_to_disk") else worker._captured_states
    module_name = f"model.layers.{layer}.self_attn.attn"
    layer_states = bucket.setdefault(context.req_id, {})
    if extra.get("qk_capture", "qk") == "score":
        heads = _resolve_score_heads(
            extra.get("output_qk"),
            layer,
            int(extra.get("score_head", getattr(worker, "_score_head_default", 0))),
        )
        q_tensor = torch.from_numpy(_mlx_float32_numpy(q_rows, mx).copy())
        if q_tensor.ndim == 1:
            q_tensor = q_tensor.unsqueeze(0)
        score = compute_head_scores(
            q_tensor,
            k_tensor,
            heads,
            worker._conf,
            getattr(worker, "_score_dtype", torch.float16),
        )
        entry = layer_states.setdefault(
            module_name,
            {
                "scores": [],
                "heads": heads,
                "layer_num": layer,
                "hookq_mode": mode,
                "capture": "score",
            },
        )
        entry["scores"].append(score)
        return

    q_tensor = torch.from_numpy(_mlx_float32_numpy(q_rows, mx).copy())
    entry = layer_states.setdefault(
        module_name, new_qk_entry(layer, mode)
    )
    entry["q"].append(q_tensor)
    append_k_prefix(entry, k_tensor)


def _capture_layer(worker, module, x, cache, layer: int, contexts, mx) -> None:
    if not any(_capture_requested(context, layer) for context in contexts):
        return
    if x.shape[0] != len(contexts):
        raise RuntimeError(
            f"Metal Q/K capture layer {layer} received batch size {x.shape[0]} "
            f"for {len(contexts)} request contexts"
        )
    n_heads = int(module.n_heads)
    n_kv_heads = int(module.n_kv_heads)
    head_dim = int(module.head_dim)
    queries = module.q_proj(x).reshape(
        x.shape[0], x.shape[1], n_heads, head_dim
    ).transpose(0, 2, 1, 3)
    keys = module.k_proj(x).reshape(
        x.shape[0], x.shape[1], n_kv_heads, head_dim
    ).transpose(0, 2, 1, 3)
    if hasattr(module, "q_norm"):
        queries = module.q_norm(queries.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    if hasattr(module, "k_norm"):
        keys = module.k_norm(keys.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)

    offset = int(getattr(cache, "offset", 0)) if cache is not None else 0
    queries = module.rope(queries, offset=offset)
    keys = module.rope(keys, offset=offset)

    prior_keys = getattr(cache, "keys", None) if cache is not None else None
    for row, context in enumerate(contexts):
        if not _capture_requested(context, layer):
            continue
        q_row = queries[row].transpose(1, 0, 2).reshape(x.shape[1], -1)
        current_k = keys[row : row + 1]
        if prior_keys is not None and offset:
            prefix = prior_keys[row : row + 1, :, :offset, :]
            complete_k = mx.concatenate((prefix, current_k), axis=2)
        else:
            complete_k = current_k
        expected = context.token_count
        if expected > complete_k.shape[2]:
            raise RuntimeError(
                f"Metal Q/K capture layer {layer} has {complete_k.shape[2]} key rows "
                f"for request {context.req_id!r}, expected {expected}"
            )
        k_row = complete_k[0, :, :expected, :].transpose(1, 0, 2).reshape(
            expected, -1
        )
        _store_qk(worker, layer, q_row, k_row, context, mx)


def _wrap_attention_class(cls: type) -> None:
    if cls in _WRAPPED_CLASSES:
        return
    original = cls.__call__

    @wraps(original)
    def wrapped(module, *args, **kwargs):
        active = _ACTIVE.get(id(module))
        if active is not None:
            if active.sequential:
                layer = module._mia_qk_layer
                if layer <= active.last_layer:
                    if layer != 0 or active.last_layer != active.max_layer:
                        raise RuntimeError(
                            "Metal sequential decode attention layers ran out of order"
                        )
                    active.current += 1
                if active.current >= len(active.contexts):
                    raise RuntimeError(
                        "Metal sequential decode produced more forwards than request contexts"
                    )
                active.last_layer = layer
                contexts = [active.contexts[active.current]]
            else:
                contexts = active.contexts
            x = args[0] if args else kwargs.get("x")
            cache = kwargs.get("cache")
            if cache is None and len(args) > 2:
                cache = args[2]
            if x is None:
                raise TypeError("Mistral attention call did not provide its hidden states")
            _capture_layer(
                module._mia_qk_worker,
                module,
                x,
                cache,
                module._mia_qk_layer,
                contexts,
                module._mia_mx,
            )
        return original(module, *args, **kwargs)

    _WRAPPED_CLASSES[cls] = original
    cls.__call__ = wrapped


@contextmanager
def _active_requests(
    contexts: list[_RequestCapture | None], modules, *, sequential: bool = False
):
    active = _ActiveRequests(
        contexts=contexts,
        sequential=sequential,
        max_layer=len(modules) - 1,
    )
    previous = []
    for module in modules:
        previous.append((module, _ACTIVE.get(id(module))))
        _ACTIVE[id(module)] = active
    try:
        yield active
    finally:
        for module, prior in previous:
            if prior is None:
                _ACTIVE.pop(id(module), None)
            else:
                _ACTIVE[id(module)] = prior


def install_qk_capture(worker) -> None:
    """Install request-scoped Q/K capture on a vLLM-Metal Mistral runner."""
    runner = worker.model_runner
    model = getattr(runner, "model", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        raise RuntimeError(
            "Metal Attention Tracker requires an mlx_lm Mistral model with "
            "model.layers; refusing to install an inactive capture hook"
        )

    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("vLLM-Metal Q/K capture requires mlx") from exc

    args = getattr(model, "args", None)
    num_heads = int(getattr(args, "num_attention_heads"))
    num_kv_heads = int(getattr(args, "num_key_value_heads", num_heads))
    hidden_size = int(getattr(args, "hidden_size"))
    head_dim = int(getattr(args, "head_dim", None) or hidden_size // num_heads)
    worker._conf = {
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "attention_multiplier": float(head_dim**-0.5),
    }
    worker._captured_states = {}
    worker._disk_states = {}
    worker._should_capture = True
    worker._qk_module_names = {}
    worker.hookq_mode = "all_tokens"
    matched = []
    for layer_num, layer in enumerate(layers):
        module = getattr(layer, "self_attn", None)
        if module is None or not all(
            hasattr(module, name) for name in ("q_proj", "k_proj", "rope")
        ):
            raise RuntimeError(
                f"Metal Attention Tracker cannot capture layer {layer_num}: "
                "expected Mistral q_proj, k_proj and rope"
            )
        object.__setattr__(module, "_mia_qk_worker", worker)
        object.__setattr__(module, "_mia_qk_layer", layer_num)
        object.__setattr__(module, "_mia_mx", mx)
        _wrap_attention_class(type(module))
        matched.append(module)
        worker._qk_module_names[layer_num] = (
            f"model.layers.{layer_num}.self_attn.attn"
        )

    original_prefill = runner._prefill_single

    @wraps(original_prefill)
    def prefill(req_id, token_ids, sampling_params, *args, **kwargs):
        capture = _request(req_id, sampling_params, "prefill", len(token_ids))
        with _active_requests([capture], matched):
            return original_prefill(req_id, token_ids, sampling_params, *args, **kwargs)

    runner._prefill_single = prefill

    original_sequential = runner._sequential_decode

    @wraps(original_sequential)
    def sequential_decode(decode_reqs, *args, **kwargs):
        contexts = [
            _request(req_id, state.sampling_params, "decode", len(state.token_ids))
            for req_id, state in decode_reqs
        ]
        with _active_requests(contexts, matched, sequential=True) as active:
            result = original_sequential(decode_reqs, *args, **kwargs)
            if contexts and (
                active.current != len(contexts) - 1
                or active.last_layer != active.max_layer
            ):
                raise RuntimeError(
                    "Metal sequential decode did not complete each request's attention pass"
                )
            return result

    runner._sequential_decode = sequential_decode

    original_batched = runner._batched_decode

    @wraps(original_batched)
    def batched_decode(decode_reqs, *args, **kwargs):
        contexts = [
            _request(req_id, state.sampling_params, "decode", len(state.token_ids))
            for req_id, state in decode_reqs
        ]
        with _active_requests(contexts, matched):
            result = original_batched(decode_reqs, *args, **kwargs)
            return result

    runner._batched_decode = batched_decode
