"""Q/K capture for Mistral-family MLX models served through vLLM-Metal."""
from __future__ import annotations

import atexit
import pickle
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
import zstandard as zstd

_PATCH_LOCK = threading.RLock()
_PATCHED_GLOBALS: dict[int, tuple[dict, Any, Any]] = {}
_ACTIVE = threading.local()
_COMPRESSOR = zstd.ZstdCompressor(level=1)


@dataclass
class _CaptureRequest:
    request_id: str
    requested_layers: set[int] | None
    hookq_mode: str
    qk_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    next_layer: int = 0


def _mlx_to_torch(value) -> torch.Tensor:
    import mlx.core as mx
    import numpy as np

    array = value.astype(mx.float32)
    mx.eval(array)
    return torch.from_numpy(np.array(array, copy=True))


def _stack_if_uniform(tensors: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    if tensors and all(tensor.shape == tensors[0].shape for tensor in tensors[1:]):
        return torch.stack(tensors)
    return tensors


def build_qk_payload(
    config: dict, captured: dict[int, dict], module_names: dict[int, str]
) -> dict:
    """Build the existing MIA raw-Q/K probe contract from per-layer tensors."""
    cache = {}
    for layer_num in sorted(captured):
        record = captured[layer_num]
        queries = record.get("q", ())
        keys = record.get("k_all", ())
        if not queries or not keys:
            continue
        name = module_names[layer_num]
        cache[name] = {
            "q": _stack_if_uniform(list(queries)),
            "k_all": _stack_if_uniform(list(keys)),
            "layer_num": int(layer_num),
            "hookq_mode": record.get("hookq_mode", "last_token"),
        }
    return {"config": dict(config), "qk_cache": cache}


def _capture_sdpa(original):
    def wrapped(queries, keys, values, *args, **kwargs):
        request = getattr(_ACTIVE, "request", None)
        if request is not None:
            layer_num = request.next_layer
            request.next_layer += 1
            if request.requested_layers is None or layer_num in request.requested_layers:
                q = _mlx_to_torch(queries).detach().cpu()
                k = _mlx_to_torch(keys).detach().cpu()
                if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1:
                    raise RuntimeError(
                        "Mistral Q/K capture expected one request with [batch, heads, "
                        "sequence, head_dim] attention tensors; got "
                        f"Q={tuple(q.shape)}, K={tuple(k.shape)}."
                    )
                q_rows = q[0].transpose(0, 1).reshape(q.shape[2], -1)
                k_rows = k[0].transpose(0, 1).reshape(k.shape[2], -1)
                if request.hookq_mode == "last_token":
                    q_rows = q_rows[-1:]
                elif request.hookq_mode != "all_tokens":
                    raise ValueError(
                        f"hookq_mode={request.hookq_mode!r} must be 'last_token' or "
                        "'all_tokens' for vLLM-Metal Q/K capture."
                    )
                record = request.qk_cache.setdefault(
                    layer_num, {"q": [], "k_all": [], "hookq_mode": request.hookq_mode}
                )
                record["q"].append(q_rows)
                record["k_all"].append(k_rows)
        return original(queries, keys, values, *args, **kwargs)

    wrapped._mia_metal_capture = True
    return wrapped


def _install_sdpa_wrappers(attention_modules: list) -> None:
    with _PATCH_LOCK:
        for module in attention_modules:
            method = getattr(type(module), "__call__", None)
            namespace = getattr(method, "__globals__", None)
            if not isinstance(namespace, dict):
                raise RuntimeError(
                    f"Cannot locate the MLX attention function globals for {type(module)!r}."
                )
            original = namespace.get("scaled_dot_product_attention")
            if not callable(original):
                raise RuntimeError(
                    f"{type(module).__module__} does not expose the expected "
                    "scaled_dot_product_attention call."
                )
            key = id(namespace)
            if key in _PATCHED_GLOBALS:
                continue
            if getattr(original, "_mia_metal_capture", False):
                continue
            wrapped = _capture_sdpa(original)
            namespace["scaled_dot_product_attention"] = wrapped
            _PATCHED_GLOBALS[key] = (namespace, original, wrapped)


def _restore_sdpa_wrappers() -> None:
    with _PATCH_LOCK:
        for namespace, original, wrapped in _PATCHED_GLOBALS.values():
            if namespace.get("scaled_dot_product_attention") is wrapped:
                namespace["scaled_dot_product_attention"] = original
        _PATCHED_GLOBALS.clear()


atexit.register(_restore_sdpa_wrappers)


def _model_layers(runner) -> list:
    model = getattr(runner, "_forward_model", None)
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError(
            "The loaded vLLM-Metal model does not expose the MLX layers expected "
            "for Attention Tracker capture."
        )
    attention = [getattr(layer, "self_attn", None) for layer in layers]
    if not attention or any(module is None for module in attention):
        raise RuntimeError(
            "The loaded model is not a supported Mistral-family model with "
            "self_attn layers."
        )
    return attention


def _requested_layers(output_spec) -> set[int] | None:
    if output_spec is None or output_spec is True:
        return None
    if isinstance(output_spec, dict):
        return {int(layer) for layer in output_spec}
    if isinstance(output_spec, (list, tuple, set)):
        return {int(layer) for layer in output_spec}
    raise ValueError(
        f"output_qk must be a layer map or sequence, not {type(output_spec).__name__}."
    )


def _install_worker_capture(worker) -> None:
    runner = getattr(worker, "model_runner", None)
    if runner is None or not type(runner).__module__.startswith("vllm_metal."):
        raise RuntimeError("MetalQKCaptureWorker requires vLLM-Metal's model runner.")
    parallel_config = getattr(
        getattr(runner, "vllm_config", None), "parallel_config", None
    )
    if int(getattr(parallel_config, "tensor_parallel_size", 1) or 1) != 1:
        raise RuntimeError(
            "Attention Tracker Q/K capture on vLLM-Metal requires tensor parallel size 1."
        )
    if getattr(runner, "_mia_metal_qk_installed", False):
        return

    attentions = _model_layers(runner)
    _install_sdpa_wrappers(attentions)
    original_prefill = runner._prefill_single
    module_names = {i: f"model.layers.{i}.self_attn" for i in range(len(attentions))}
    model_args = getattr(runner, "model_args", {})
    num_heads = int(model_args.get("num_attention_heads") or attentions[0].n_heads)
    num_kv_heads = int(model_args.get("num_key_value_heads") or attentions[0].n_kv_heads)
    head_dim = int(model_args.get("head_dim") or attentions[0].head_dim)
    config = {
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "attention_multiplier": float(getattr(attentions[0], "scale", head_dim ** -0.5)),
    }

    def prefill_with_capture(req_id, token_ids, sampling_params, *args, **kwargs):
        extra = getattr(sampling_params, "extra_args", None) or {}
        output_spec = extra.get("output_qk")
        if output_spec is None:
            return original_prefill(req_id, token_ids, sampling_params, *args, **kwargs)
        hookq_mode = extra.get("hookq_mode", "all_tokens")
        request = _CaptureRequest(
            request_id=str(req_id),
            requested_layers=_requested_layers(output_spec),
            hookq_mode=hookq_mode,
        )
        previous = getattr(_ACTIVE, "request", None)
        _ACTIVE.request = request
        try:
            result = original_prefill(req_id, token_ids, sampling_params, *args, **kwargs)
        finally:
            _ACTIVE.request = previous
        worker._mia_metal_qk_captures[str(req_id)] = request
        return result

    runner._prefill_single = prefill_with_capture
    runner._mia_metal_qk_installed = True
    worker._mia_metal_qk_captures = {}
    worker._mia_metal_qk_config = config
    worker._mia_metal_qk_module_names = module_names


class MetalQKCaptureWorker:
    """Worker extension providing MIA's existing capture_qk RPC contract on Metal."""

    def install_hooks(self) -> None:
        _install_worker_capture(self)

    def mia_delivery_info(self) -> dict:
        return {"per_request": False}

    def get_captured_states(self, request_id: str) -> bytes:
        captured = getattr(self, "_mia_metal_qk_captures", {}).pop(str(request_id), None)
        if captured is None or not captured.qk_cache:
            payload = {
                "mia_error": (
                    f"No Mistral Q/K rows were captured for request {request_id!r}."
                )
            }
        else:
            payload = build_qk_payload(
                self._mia_metal_qk_config,
                captured.qk_cache,
                self._mia_metal_qk_module_names,
            )
        return _COMPRESSOR.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
