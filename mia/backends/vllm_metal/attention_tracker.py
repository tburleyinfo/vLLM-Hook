"""Request-scoped Q/K capture for vLLM-Metal's MLX Granite attention path."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import sys
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class _RequestCapture:
    req_id: str
    extra: dict
    prefill: bool
    key_length: int


@dataclass(frozen=True)
class _CaptureBatch:
    worker: Any
    requests: tuple[_RequestCapture, ...]


_ACTIVE_BATCH: ContextVar[_CaptureBatch | None] = ContextVar(
    "mia_metal_qk_batch", default=None
)
_ACTIVE_LAYER: ContextVar[int | None] = ContextVar("mia_metal_qk_layer", default=None)
_PATCHED_ATTENTION_CLASSES: set[type] = set()
_PATCHED_SDPA_MODULES: set[str] = set()


def _extra_args(sampling_params: Any) -> dict:
    extra = getattr(sampling_params, "extra_args", None)
    return extra if isinstance(extra, dict) else {}


@contextmanager
def _capture_scope(worker, requests: tuple[_RequestCapture, ...]):
    token = _ACTIVE_BATCH.set(_CaptureBatch(worker, requests))
    try:
        yield
    finally:
        _ACTIVE_BATCH.reset(token)


def _requested_layer(extra: dict, layer_num: int) -> bool:
    output = extra.get("output_qk")
    if output is None:
        return False
    if output is True:
        return True
    if isinstance(output, dict):
        return any(int(layer) == layer_num for layer in output)
    if isinstance(output, (list, tuple, set)):
        return layer_num in {int(layer) for layer in output}
    raise TypeError(
        f"output_qk must be a mapping, sequence, True, or None; "
        f"got {type(output).__name__}"
    )


def _request_enabled(request: _RequestCapture) -> bool:
    if request.extra.get("output_qk") is None:
        return False
    hooks_on = request.extra.get("hooks_on", "prefill")
    if hooks_on not in ("prefill", "decode", "both"):
        raise ValueError(f"unsupported hooks_on value for Metal capture: {hooks_on!r}")
    return hooks_on == "both" or (hooks_on == "prefill" and request.prefill) or (
        hooks_on == "decode" and not request.prefill
    )


def _validate_request(extra: dict) -> None:
    if extra.get("output_qk") is None:
        return
    capture = extra.get("qk_capture", "qk")
    if capture != "qk":
        raise ValueError(
            f"vLLM-Metal's Granite adapter captures raw Q/K, not "
            f"qk_capture={capture!r}"
        )
    mode = extra.get("hookq_mode", "all_tokens")
    if mode not in ("all_tokens", "last_token"):
        raise ValueError(f"unsupported hookq_mode for Metal capture: {mode!r}")
    hooks_on = extra.get("hooks_on", "prefill")
    if hooks_on not in ("prefill", "decode", "both"):
        raise ValueError(f"unsupported hooks_on value for Metal capture: {hooks_on!r}")


def _record_qk(
    worker,
    request: _RequestCapture,
    layer_num: int,
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
) -> None:
    """Append one target-format Q/K observation to MIA's established cache entry."""
    extra = request.extra
    mode = extra.get("hookq_mode", getattr(worker, "hookq_mode", "all_tokens"))
    if mode not in ("all_tokens", "last_token"):
        raise ValueError(f"unsupported hookq_mode for Metal capture: {mode!r}")

    q_rows = q_rows[-1:] if mode == "last_token" else q_rows
    if q_rows.ndim != 2 or k_rows.ndim != 2:
        raise ValueError("Metal Q/K capture expects flattened [tokens, head_width] tensors")

    bucket = worker._disk_states if extra.get("save_to_disk") else worker._captured_states
    layer_states = bucket.setdefault(request.req_id, {})
    module_name = f"model.layers.{layer_num}.self_attn"
    if module_name not in layer_states:
        from mia.workers.qk_capture_worker import new_qk_entry

        layer_states[module_name] = new_qk_entry(layer_num, mode)
    entry = layer_states[module_name]
    if entry["hookq_mode"] != mode:
        raise ValueError(f"request {request.req_id!r} changed hookq_mode during capture")

    from mia.workers.qk_capture_worker import append_k_prefix

    entry["q"].append(q_rows.contiguous())
    append_k_prefix(entry, k_rows.contiguous())


def _mlx_rows_to_torch(array, mx) -> torch.Tensor:
    """Materialize MLX arrays safely, including bfloat16 source arrays."""
    converted = array.astype(mx.float32)
    mx.eval(converted)
    return torch.from_numpy(np.array(np.asarray(converted), copy=True))


def _capture_attention_arrays(queries, keys, mx) -> None:
    batch = _ACTIVE_BATCH.get()
    layer_num = _ACTIVE_LAYER.get()
    if batch is None or layer_num is None:
        return

    selected = [
        (index, request)
        for index, request in enumerate(batch.requests)
        if _request_enabled(request) and _requested_layer(request.extra, layer_num)
    ]
    if not selected:
        return
    if len(queries.shape) != 4 or len(keys.shape) != 4:
        raise ValueError(
            "Metal Granite attention Q/K must have [batch, heads, tokens, head_dim] shape"
        )
    if queries.shape[0] != len(batch.requests) or keys.shape[0] != len(batch.requests):
        raise ValueError("Metal attention batch size does not match request metadata")

    query_rows = _mlx_rows_to_torch(
        queries.transpose(0, 2, 1, 3).reshape(
            queries.shape[0], queries.shape[2], -1
        ),
        mx,
    )
    key_rows = _mlx_rows_to_torch(
        keys.transpose(0, 2, 1, 3).reshape(keys.shape[0], keys.shape[2], -1),
        mx,
    )

    for index, request in selected:
        q = query_rows[index]
        k = key_rows[index, : request.key_length]
        if k.shape[0] != request.key_length:
            raise ValueError(
                f"Metal attention returned {k.shape[0]} key rows for request "
                f"{request.req_id!r}; expected {request.key_length}"
            )
        _record_qk(batch.worker, request, layer_num, q, k)


def _install_attention_wrappers(model) -> int:
    layers = getattr(model, "layers", None)
    if layers is None:
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
    if not layers:
        raise TypeError("vLLM-Metal model exposes no MLX transformer layers")

    patched_layers = 0
    for layer_num, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        cls = type(attention)
        module = sys.modules.get(cls.__module__)
        sdpa = getattr(module, "scaled_dot_product_attention", None)
        if module is None or not callable(sdpa):
            continue

        if cls not in _PATCHED_ATTENTION_CLASSES:
            original_call = cls.__call__

            def wrapped_call(self, *args, __original=original_call, **kwargs):
                layer_token = _ACTIVE_LAYER.set(
                    int(getattr(self, "_mia_metal_layer_num", -1))
                )
                try:
                    return __original(self, *args, **kwargs)
                finally:
                    _ACTIVE_LAYER.reset(layer_token)

            cls.__call__ = wrapped_call
            _PATCHED_ATTENTION_CLASSES.add(cls)

        attention._mia_metal_layer_num = int(layer_num)
        module_name = cls.__module__
        if module_name not in _PATCHED_SDPA_MODULES:
            original_sdpa = sdpa

            def wrapped_sdpa(
                queries, keys, values, *args, __original=original_sdpa, **kwargs
            ):
                if _ACTIVE_BATCH.get() is not None:
                    import mlx.core as mx

                    _capture_attention_arrays(queries, keys, mx)
                return __original(queries, keys, values, *args, **kwargs)

            module.scaled_dot_product_attention = wrapped_sdpa
            _PATCHED_SDPA_MODULES.add(module_name)
        patched_layers += 1

    if patched_layers != len(layers):
        raise TypeError(
            f"vLLM-Metal Granite capture found {patched_layers} attention modules "
            f"for {len(layers)} model layers"
        )
    return patched_layers


def _make_model_config(worker) -> dict:
    args = getattr(worker.model_runner, "model_args", {}) or {}
    model = worker.model_runner.model
    model_args = getattr(model, "args", None)

    def value(key, *aliases):
        for source in (args, model_args):
            for name in (key, *aliases):
                if isinstance(source, dict) and source.get(name) is not None:
                    return source[name]
                found = getattr(source, name, None)
                if found is not None:
                    return found
        return None

    num_heads = value("num_attention_heads", "n_heads")
    num_kv_heads = value("num_key_value_heads", "n_kv_heads") or num_heads
    hidden = value("hidden_size", "dim")
    head_dim = value("head_dim") or (
        int(hidden) // int(num_heads) if hidden is not None and num_heads else None
    )
    if not num_heads or not num_kv_heads or not hidden or not head_dim:
        raise ValueError("vLLM-Metal model is missing Granite attention dimensions")
    attention_multiplier = value("attention_multiplier", "scale")
    if attention_multiplier is None:
        attention_multiplier = 1.0 / math.sqrt(int(head_dim))
    return {
        "num_attention_heads": int(num_heads),
        "num_key_value_heads": int(num_kv_heads),
        "hidden_size": int(hidden),
        "head_dim": int(head_dim),
        "attention_multiplier": float(attention_multiplier),
    }


def install_metal_qk_capture(worker) -> None:
    """Install request-scoped Q/K observation on vLLM-Metal's MLX model."""
    runner = worker.model_runner
    if getattr(getattr(runner, "metal_config", None), "use_paged_attention", False):
        raise RuntimeError(
            "MIA's vLLM-Metal Q/K adapter currently requires "
            "VLLM_METAL_USE_PAGED_ATTENTION=0"
        )
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("vLLM-Metal model must be loaded before Q/K capture installs")

    if not hasattr(worker, "_captured_states") or worker._captured_states is None:
        worker._captured_states = {}
    if not hasattr(worker, "_disk_states") or worker._disk_states is None:
        worker._disk_states = {}
    worker._conf = _make_model_config(worker)
    worker.hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        raise TypeError("vLLM-Metal model exposes no MLX transformer layers")
    worker._qk_module_names = {
        layer_num: f"model.layers.{layer_num}.self_attn"
        for layer_num in range(len(layers))
    }
    worker._should_capture = True
    worker._tp_rank = int(getattr(worker, "rank", 0))
    from mia.core.delivery.writer_process import init_writer_process

    init_writer_process(worker)
    _install_attention_wrappers(model)

    if getattr(runner, "_mia_metal_qk_installed", False):
        return

    original_prefill = runner._prefill_single

    def wrapped_prefill(*args, **kwargs):
        req_id = args[0] if args else kwargs["req_id"]
        token_ids = args[1] if len(args) > 1 else kwargs["token_ids"]
        sampling_params = args[2] if len(args) > 2 else kwargs["sampling_params"]
        extra = _extra_args(sampling_params)
        _validate_request(extra)
        request = _RequestCapture(
            str(req_id), extra, True, len(token_ids)
        )
        with _capture_scope(worker, (request,)):
            return original_prefill(*args, **kwargs)

    runner._prefill_single = wrapped_prefill

    def wrap_decode(original):
        def wrapped(decode_reqs, *args, **kwargs):
            request_items = []
            for req_id, state in decode_reqs:
                extra = _extra_args(state.sampling_params)
                _validate_request(extra)
                request_items.append(
                    _RequestCapture(
                        str(req_id), extra, False, len(state.token_ids)
                    )
                )
            requests = tuple(request_items)
            with _capture_scope(worker, requests):
                return original(decode_reqs, *args, **kwargs)

        return wrapped

    runner._sequential_decode = wrap_decode(runner._sequential_decode)
    runner._batched_decode = wrap_decode(runner._batched_decode)
    runner._mia_metal_qk_installed = True
