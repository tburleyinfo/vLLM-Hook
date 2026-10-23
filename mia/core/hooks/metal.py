"""MLX-module hook primitives for vLLM-Metal's model runner."""
from __future__ import annotations

import contextlib
import contextvars
import re
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import torch

from mia.workers.steer_worker import (
    _load_steering_vector,
    _resolve_steer_config,
    _steer_targets_layer,
)


_LAYER_PATTERNS = (
    re.compile(r"^model\.layers\.(\d+)$"),
    re.compile(r"^language_model\.model\.layers\.(\d+)$"),
    re.compile(r"^transformer\.h\.(\d+)$"),
    re.compile(r"^model\.decoder\.layers\.(\d+)$"),
)
_ACTIVE_SCOPE: contextvars.ContextVar[tuple | None] = contextvars.ContextVar(
    "mia_metal_hook_scope", default=None
)
_WRAPPED_CLASSES: dict[type, Callable[..., object]] = {}
_CLASS_INSTANCES: dict[type, int] = {}


@dataclass(frozen=True)
class MetalRequest:
    """Request row mapping for one vLLM-Metal forward."""

    request_id: str
    extra_args: dict
    start: int
    end: int
    is_prefill: bool
    is_final_prefill: bool = True


def _layer_number(name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.match(name)
        if match:
            return int(match.group(1))
    return None


def _mlx_to_torch(value) -> torch.Tensor:
    import mlx.core as mx

    mx.eval(value)
    return torch.from_numpy(np.asarray(value.astype(mx.float32)).copy())


def _selected_layers(spec, layer_num: int) -> bool:
    if spec is True:
        return True
    if isinstance(spec, dict):
        spec = set(spec)
    if isinstance(spec, (list, tuple, set)):
        try:
            return layer_num + 1 in {int(layer) for layer in spec}
        except (TypeError, ValueError):
            return False
    return False


def _capture(runtime, requests, layer_num: int, output) -> None:
    if not hasattr(output, "shape") or len(output.shape) < 2:
        return
    rows = output.reshape((-1, output.shape[-1]))
    n_rows = int(rows.shape[0])
    for request in requests:
        spec = request.extra_args.get("output_hidden_states")
        if spec is None or not _selected_layers(spec, layer_num):
            continue
        hooks_on = request.extra_args.get("hooks_on", "prefill")
        if hooks_on not in ("both", "prefill" if request.is_prefill else "decode"):
            continue
        start, end = max(0, request.start), min(n_rows, request.end)
        if end <= start:
            continue
        if request.extra_args.get("hs_mode", "last_token") == "last_token":
            if request.is_prefill and not request.is_final_prefill:
                continue
            start = end - 1
        runtime.probes.setdefault(request.request_id, {}).setdefault(
            "hs_cache", {}
        )[layer_num + 1] = {
            "hidden_states": [_mlx_to_torch(rows[start:end])],
        }


def _steer(runtime, requests, layer_num: int, output):
    import mlx.core as mx

    if not hasattr(output, "shape") or len(output.shape) < 2:
        return output
    rows = output.reshape((-1, output.shape[-1]))
    n_rows, hidden = int(rows.shape[0]), int(rows.shape[-1])
    plans = []
    for request in requests:
        raw = request.extra_args.get("steer")
        cfg = _resolve_steer_config(raw, runtime.steer_config_path)
        if cfg is None or not _steer_targets_layer(cfg.get("optimal_layer", -1), layer_num):
            continue
        phase = cfg.get("phase", "both")
        if phase not in ("both", "prefill" if request.is_prefill else "decode"):
            continue
        positions = cfg.get("positions")
        if positions is None:
            positions = "all_tokens" if cfg.get("apply_at_all_positions", True) else "last_token"
        if positions not in ("all_tokens", "last_token"):
            raise ValueError(f"Invalid steering.positions={positions!r}")
        start, end = max(0, request.start), min(n_rows, request.end)
        if end <= start:
            continue
        if positions == "last_token":
            if request.is_prefill and not request.is_final_prefill:
                continue
            start = end - 1
        plans.append((cfg, start, end))
    if not plans:
        return output

    for cfg, _start, _end in plans:
        data = runtime.vector_cache.get(cfg["vector_path"])
        if data is None:
            data = _load_steering_vector(cfg["vector_path"])
            runtime.vector_cache[cfg["vector_path"]] = data
        vector = mx.array(np.asarray(data["dir"], dtype=np.float32), dtype=rows.dtype)
        if int(vector.shape[-1]) != hidden:
            raise ValueError(
                f"Steering vector width {vector.shape[-1]} does not match "
                f"model hidden size {hidden}"
            )
        method = cfg.get("method", "adjust_rs")
        if method == "add_vector":
            coefficient = float(cfg.get("coefficient", 0))
            delta = mx.broadcast_to(vector * coefficient, rows.shape)
        elif method == "adjust_rs":
            if "avg_proj" not in data:
                raise KeyError(
                    f"Steering vector {cfg['vector_path']!r} has no avg_proj"
                )
            projection = mx.sum(rows * vector, axis=-1)
            coefficient = mx.array(
                np.asarray(data["avg_proj"]), dtype=rows.dtype
            ) - projection
            delta = mx.expand_dims(coefficient, -1) * vector
        else:
            raise ValueError(f"Unknown steering method: {method}")
        row_mask = np.zeros(n_rows, dtype=np.bool_)
        row_mask[_start:_end] = True
        mask = mx.array(row_mask)
        rows = mx.where(mx.expand_dims(mask, -1), rows + delta, rows)
    return rows.reshape(output.shape)


class MetalHookRuntime:
    """Attach capture and steering wrappers to a loaded MLX model.

    Forward request boundaries are supplied with :meth:`request_scope`; model
    generation itself remains owned by vLLM-Metal.
    """

    def __init__(self, model, steer_config_path: str | None = None):
        self.model = model
        self.steer_config_path = steer_config_path
        self.vector_cache: dict[str, dict] = {}
        self.probes: dict[str, dict] = {}
        self._modules = []
        self._install()

    @contextlib.contextmanager
    def request_scope(
        self, requests: tuple[MetalRequest, ...] | list[MetalRequest]
    ) -> Iterator[None]:
        token = _ACTIVE_SCOPE.set((self, tuple(requests)))
        try:
            yield
        finally:
            _ACTIVE_SCOPE.reset(token)

    def take_probes(self, request_id: str) -> dict | None:
        """Remove and return captured data for one request."""
        return self.probes.pop(str(request_id), None)

    def _install(self) -> None:
        for name, module in self.model.named_modules():
            layer_num = _layer_number(name)
            if layer_num is None:
                continue
            setattr(module, "_mia_metal_runtime", self)
            setattr(module, "_mia_metal_layer_num", layer_num)
            self._modules.append(module)
            cls = type(module)
            _CLASS_INSTANCES[cls] = _CLASS_INSTANCES.get(cls, 0) + 1
            if cls in _WRAPPED_CLASSES:
                continue
            original = cls.__call__
            _WRAPPED_CLASSES[cls] = original

            def wrapped(instance, *args, __original=original, **kwargs):
                result = __original(instance, *args, **kwargs)
                active = _ACTIVE_SCOPE.get()
                runtime = getattr(instance, "_mia_metal_runtime", None)
                if active is None or runtime is None or active[0] is not runtime:
                    return result
                layer = instance._mia_metal_layer_num
                requests = active[1]
                if isinstance(result, tuple):
                    target = result[-1]
                    _capture(runtime, requests, layer, target)
                    steered = _steer(runtime, requests, layer, target)
                    return (*result[:-1], steered)
                _capture(runtime, requests, layer, result)
                return _steer(runtime, requests, layer, result)

            cls.__call__ = wrapped

    def close(self) -> None:
        """Detach runtime state from the model without changing model weights."""
        for module in self._modules:
            if getattr(module, "_mia_metal_runtime", None) is self:
                delattr(module, "_mia_metal_runtime")
                delattr(module, "_mia_metal_layer_num")
            cls = type(module)
            remaining = _CLASS_INSTANCES.get(cls, 0) - 1
            if remaining <= 0:
                original = _WRAPPED_CLASSES.pop(cls, None)
                if original is not None:
                    cls.__call__ = original
                _CLASS_INSTANCES.pop(cls, None)
            else:
                _CLASS_INSTANCES[cls] = remaining
        self._modules.clear()
        self.vector_cache.clear()
