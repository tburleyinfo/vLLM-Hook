"""Request-scoped Q/K capture for the MLX attention path in vLLM-Metal."""
from __future__ import annotations

import contextlib
import contextvars
import sys
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import torch

from mia.errors import MiaConfigurationError


@dataclass
class _CaptureCall:
    worker: Any
    request_ids: list[str]
    sampling_params: list[Any]
    phase: str


_ACTIVE_CALL: contextvars.ContextVar[_CaptureCall | None] = contextvars.ContextVar(
    "mia_metal_qk_call", default=None
)
_ACTIVE_LAYER: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mia_metal_qk_layer", default=None
)
_PATCH_LOCK = threading.RLock()
_ATTENTION_LAYERS: dict[int, int] = {}
_ATTENTION_PATCHES: dict[type, tuple[Any, Any, int]] = {}
_SDPA_PATCHES: dict[ModuleType, tuple[Any, Any, int]] = {}
_INSTALLS: dict[int, dict[str, Any]] = {}


def _mlx_float32_numpy(array: Any) -> np.ndarray:
    """Export an MLX array without asking NumPy to interpret MLX bfloat16."""
    if isinstance(array, np.ndarray):
        return array.astype(np.float32, copy=False)

    import mlx.core as mx

    mx.eval(array)
    return np.asarray(array.astype(mx.float32))


def _as_torch_rows(array: Any) -> torch.Tensor:
    return torch.from_numpy(_mlx_float32_numpy(array).copy())


def _requested_layers(worker: Any, params: Any) -> tuple[dict, str, str, set[int] | None]:
    extra = getattr(params, "extra_args", None) or {}
    output = extra.get("output_qk")
    if output is None:
        return extra, "", "", None
    layer_filter = (
        {int(layer) for layer in output}
        if isinstance(output, dict)
        else ({int(layer) for layer in output} if isinstance(output, (list, tuple)) else None)
    )
    mode = str(extra.get("hookq_mode", getattr(worker, "hookq_mode", "all_tokens")))
    hooks_on = extra.get("hooks_on", getattr(worker, "_default_hooks_on", "prefill"))
    return extra, mode, hooks_on, layer_filter


def _request_wants_layer(call: _CaptureCall, layer_num: int, params: Any) -> bool:
    extra, mode, hooks_on, layer_filter = _requested_layers(call.worker, params)
    return bool(
        mode
        and hooks_on in ("both", call.phase)
        and (layer_filter is None or layer_num in layer_filter)
        and extra.get("output_qk") is not None
    )


def _record_attention(query: Any, key: Any) -> None:
    call = _ACTIVE_CALL.get()
    layer_num = _ACTIVE_LAYER.get()
    if call is None or layer_num is None or not any(
        _request_wants_layer(call, layer_num, params) for params in call.sampling_params
    ):
        return

    query_np = _mlx_float32_numpy(query)
    key_np = _mlx_float32_numpy(key)
    if query_np.ndim != 4 or key_np.ndim != 4:
        raise MiaConfigurationError(
            "vLLM-Metal Q/K capture expected attention tensors with shape "
            "[batch, heads, sequence, head_dim]."
        )
    if query_np.shape[0] != len(call.request_ids) or key_np.shape[0] != len(call.request_ids):
        raise MiaConfigurationError(
            "vLLM-Metal Q/K capture batch does not match the active request batch."
        )

    module_name = f"model.layers.{layer_num}.self_attn.attn"
    for batch_idx, (req_id, params) in enumerate(zip(call.request_ids, call.sampling_params)):
        extra, mode, hooks_on, layer_filter = _requested_layers(call.worker, params)
        if not mode:
            continue
        if layer_filter is not None and layer_num not in layer_filter:
            continue
        if hooks_on not in ("both", call.phase):
            continue

        q_rows = _as_torch_rows(query_np[batch_idx].transpose(1, 0, 2).reshape(
            query_np.shape[2], -1))
        if mode == "last_token":
            q_rows = q_rows[-1:]
        elif mode != "all_tokens":
            raise MiaConfigurationError(f"Unknown hookq_mode for Metal capture: {mode!r}")
        k_rows = _as_torch_rows(key_np[batch_idx].transpose(1, 0, 2).reshape(
            key_np.shape[2], -1))

        bucket_name = "_disk_states" if extra.get("save_to_disk") else "_captured_states"
        bucket = getattr(call.worker, bucket_name)
        layer_states = bucket.setdefault(str(req_id), {})
        entry = layer_states.setdefault(
            module_name,
            {"q": [], "k_all": [], "layer_num": layer_num, "hookq_mode": mode,
             "k_prefix_ends": []},
        )
        entry["q"].append(q_rows)
        previous_end = entry["k_prefix_ends"][-1] if entry["k_prefix_ends"] else 0
        if k_rows.shape[0] < previous_end:
            raise MiaConfigurationError(
                f"Metal Q/K capture key length shrank at layer {layer_num}: "
                f"{k_rows.shape[0]} < {previous_end}."
            )
        entry["k_all"].append(k_rows[previous_end:].clone() if previous_end else k_rows)
        entry["k_prefix_ends"].append(int(k_rows.shape[0]))


def _register_attention_class(attention: Any, layer_num: int) -> None:
    cls = type(attention)
    with _PATCH_LOCK:
        _ATTENTION_LAYERS[id(attention)] = int(layer_num)
        existing = _ATTENTION_PATCHES.get(cls)
        if existing is not None:
            original, wrapped, count = existing
            _ATTENTION_PATCHES[cls] = (original, wrapped, count + 1)
            return

        original = cls.__call__

        def wrapped(self, *args, **kwargs):
            active_layer = _ATTENTION_LAYERS.get(id(self))
            token = _ACTIVE_LAYER.set(active_layer)
            try:
                return original(self, *args, **kwargs)
            finally:
                _ACTIVE_LAYER.reset(token)

        cls.__call__ = wrapped
        _ATTENTION_PATCHES[cls] = (original, wrapped, 1)

        module = sys.modules.get(cls.__module__)
        sdpa = getattr(module, "scaled_dot_product_attention", None)
        if module is None or not callable(sdpa):
            cls.__call__ = original
            _ATTENTION_PATCHES.pop(cls, None)
            _ATTENTION_LAYERS.pop(id(attention), None)
            raise MiaConfigurationError(
                f"Cannot install Metal Q/K capture for {cls.__module__}.{cls.__name__}: "
                "its module does not expose scaled_dot_product_attention."
            )
        patch = _SDPA_PATCHES.get(module)
        if patch is None:
            def wrapped_sdpa(*args, **kwargs):
                result = sdpa(*args, **kwargs)
                if len(args) >= 2:
                    _record_attention(args[0], args[1])
                else:
                    query = kwargs.get("queries", kwargs.get("query"))
                    key = kwargs.get("keys", kwargs.get("key"))
                    if query is not None and key is not None:
                        _record_attention(query, key)
                return result

            setattr(module, "scaled_dot_product_attention", wrapped_sdpa)
            _SDPA_PATCHES[module] = (sdpa, wrapped_sdpa, 1)
        else:
            _SDPA_PATCHES[module] = (patch[0], patch[1], patch[2] + 1)


@contextlib.contextmanager
def _request_capture(worker: Any, request_ids: list[str], params: list[Any], phase: str):
    token = _ACTIVE_CALL.set(_CaptureCall(worker, request_ids, params, phase))
    try:
        yield
    finally:
        _ACTIVE_CALL.reset(token)


def install_metal_qk_hooks(worker: Any) -> None:
    """Attach the existing capture worker contract to a vLLM-Metal model runner."""
    runner = worker.model_runner
    model = getattr(runner, "model", None)
    layers = getattr(model, "layers", None)
    if model is None or not isinstance(layers, (list, tuple)) or not layers:
        raise MiaConfigurationError(
            "Metal Q/K capture requires a loaded MLX model exposing a non-empty layers list."
        )
    if not hasattr(runner, "_prefill_single") or not hasattr(runner, "_batched_decode"):
        raise MiaConfigurationError(
            "Metal Q/K capture requires vLLM-Metal's prefill and batched-decode lifecycle."
        )

    original_prefill = runner._prefill_single
    original_decode = runner._batched_decode
    registered: list[Any] = []
    try:
        for layer_num, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                continue
            _register_attention_class(attention, layer_num)
            registered.append(attention)
        if not registered:
            raise MiaConfigurationError("No self-attention layers found for Metal Q/K capture.")

        args = getattr(runner, "model_args", {})
        heads = int(args["num_attention_heads"])
        kv_heads = int(args.get("num_key_value_heads", heads))
        head_dim = int(args.get("head_dim") or (int(args["hidden_size"]) // heads))
        worker._conf = {
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "hidden_size": int(args.get("hidden_size", heads * head_dim)),
            "head_dim": head_dim,
            "attention_multiplier": float(args.get("attention_multiplier", head_dim ** -0.5)),
        }
        worker._captured_states = {}
        worker._disk_states = {}
        worker.hookq_mode = "all_tokens"
        worker._hooks_installed = True

        def prefill(req_id, token_ids, sampling_params, *args_, **kwargs):
            with _request_capture(worker, [str(req_id)], [sampling_params], "prefill"):
                return original_prefill(req_id, token_ids, sampling_params, *args_, **kwargs)

        def batched_decode(decode_reqs, *args_, **kwargs):
            request_ids = [str(req_id) for req_id, _ in decode_reqs]
            params = [state.sampling_params for _, state in decode_reqs]
            with _request_capture(worker, request_ids, params, "decode"):
                return original_decode(decode_reqs, *args_, **kwargs)

        runner._prefill_single = prefill
        runner._batched_decode = batched_decode
        _INSTALLS[id(worker)] = {
            "runner": runner,
            "original_prefill": original_prefill,
            "original_decode": original_decode,
            "registered": registered,
        }
        original_shutdown = getattr(worker, "shutdown", None)
        if callable(original_shutdown):
            def shutdown(*args_, **kwargs):
                try:
                    return original_shutdown(*args_, **kwargs)
                finally:
                    uninstall_metal_qk_hooks(worker)
            worker.shutdown = shutdown
    except Exception:
        for attention in registered:
            _unregister_attention(attention)
        raise


def _unregister_attention(attention: Any) -> None:
    cls = type(attention)
    with _PATCH_LOCK:
        _ATTENTION_LAYERS.pop(id(attention), None)
        patch = _ATTENTION_PATCHES.get(cls)
        if patch is None:
            return
        original, wrapped, count = patch
        if count > 1:
            _ATTENTION_PATCHES[cls] = (original, wrapped, count - 1)
            return
        if cls.__call__ is wrapped:
            cls.__call__ = original
        _ATTENTION_PATCHES.pop(cls, None)
        module = sys.modules.get(cls.__module__)
        sdpa_patch = _SDPA_PATCHES.get(module) if module is not None else None
        if sdpa_patch is not None:
            sdpa_original, sdpa_wrapped, sdpa_count = sdpa_patch
            if sdpa_count > 1:
                _SDPA_PATCHES[module] = (sdpa_original, sdpa_wrapped, sdpa_count - 1)
            else:
                if getattr(module, "scaled_dot_product_attention", None) is sdpa_wrapped:
                    setattr(module, "scaled_dot_product_attention", sdpa_original)
                _SDPA_PATCHES.pop(module, None)


def uninstall_metal_qk_hooks(worker: Any) -> None:
    """Restore model and runner callables installed by :func:`install_metal_qk_hooks`."""
    install = _INSTALLS.pop(id(worker), None)
    if install is None:
        return
    runner = install["runner"]
    runner._prefill_single = install["original_prefill"]
    runner._batched_decode = install["original_decode"]
    for attention in install["registered"]:
        _unregister_attention(attention)
    if "shutdown" in getattr(worker, "__dict__", {}):
        del worker.shutdown
