"""Qwen2 Q/K capture for vLLM-Metal's MLX model execution path."""
from __future__ import annotations

import functools
import importlib
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch

from mia.errors import MiaConfigurationError, MiaDeliveryError

_CAPTURE_LOCK = threading.RLock()
_ACTIVE_CAPTURE: "MetalQKCapture | None" = None
_ACTIVE_LAYER: int | None = None
_LIFECYCLE_PATCHED = False
_PATCHED_CLASSES: set[type] = set()
_PATCHED_SDPA_MODULES: set[str] = set()


def metal_plugin_selected() -> bool:
    """Whether vLLM-Metal is explicitly selected in the current environment."""
    return "metal" in {
        item.strip() for item in os.environ.get("VLLM_PLUGINS", "").split(",")
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    import mlx.core as mx

    return np.asarray(value.astype(mx.float32)).copy()


class MetalQKCapture:
    """One synchronous generation's prefill Q/K buffers in the existing probes schema."""

    def __init__(self, layer_to_heads: dict[int, list], hookq_mode: str) -> None:
        self.layer_to_heads = {int(k): list(v) for k, v in layer_to_heads.items()}
        self.hookq_mode = hookq_mode
        self.layers = set(self.layer_to_heads)
        self._current_layer: int | None = None
        self._last_layer: int | None = None
        self._seen_layers: set[int] = set()
        self._batches: list[dict[int, list[tuple[torch.Tensor, torch.Tensor]]]] = []
        self._current: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._config: dict[str, int | float] | None = None

    def begin_layer(self, layer_num: int) -> None:
        """Track model layer order to separate successive model forwards."""
        layer_num = int(layer_num)
        if self._last_layer is not None and layer_num <= self._last_layer:
            if self._current:
                self._batches.append(self._current)
            self._current = {}
            self._seen_layers.clear()
        self._last_layer = layer_num
        self._current_layer = layer_num

    def record(self, queries: Any, keys: Any, scale: float) -> None:
        """Record one layer's prefill tensors after RoPE and cache update."""
        layer_num = self._current_layer
        if layer_num not in self.layers or layer_num in self._seen_layers:
            return

        q = _to_numpy(queries)
        k = _to_numpy(keys)
        if q.ndim != 4 or k.ndim != 4:
            raise MiaDeliveryError(
                f"vLLM-Metal Q/K capture expected [batch, heads, seq, dim], got "
                f"Q={q.shape}, K={k.shape}"
            )
        if q.shape[0] != k.shape[0]:
            raise MiaDeliveryError(
                f"vLLM-Metal Q/K capture batch mismatch: Q={q.shape}, K={k.shape}"
            )
        # The tracker consumes prompt prefill only. Decode has Q length 1 and
        # K length greater than 1; chunked prefill cannot provide full prompt Q.
        if q.shape[2] != k.shape[2]:
            return

        captured = []
        for batch_idx in range(q.shape[0]):
            q_rows = q[batch_idx].transpose(1, 0, 2).reshape(q.shape[2], -1)
            k_rows = k[batch_idx].transpose(1, 0, 2).reshape(k.shape[2], -1)
            if self.hookq_mode == "last_token":
                q_rows = q_rows[-1:]
            captured.append(
                (
                    torch.from_numpy(np.ascontiguousarray(q_rows)),
                    torch.from_numpy(np.ascontiguousarray(k_rows)),
                )
            )
        self._current[layer_num] = captured
        self._seen_layers.add(layer_num)

        if self._config is None:
            self._config = {
                "num_attention_heads": int(q.shape[1]),
                "num_key_value_heads": int(k.shape[1]),
                "head_dim": int(q.shape[3]),
                "attention_multiplier": float(scale),
            }
        elif (
            self._config["num_attention_heads"] != int(q.shape[1])
            or self._config["num_key_value_heads"] != int(k.shape[1])
            or self._config["head_dim"] != int(q.shape[3])
        ):
            raise MiaDeliveryError("vLLM-Metal Q/K dimensions changed during generation")

    def probes(self, request_count: int) -> list[dict]:
        """Return one ``{config, qk_cache}`` mapping per generated request."""
        batches = [*self._batches]
        if self._current:
            batches.append(self._current)
        rows: list[dict[int, list[tuple[torch.Tensor, torch.Tensor]]]] = [
            {} for _ in range(request_count)
        ]
        batch_cursor = 0
        for batch in batches:
            batch_size = max((len(v) for v in batch.values()), default=0)
            if batch_cursor + batch_size > request_count:
                raise MiaDeliveryError(
                    "vLLM-Metal Q/K capture observed more prefill rows than generated requests"
                )
            for layer_num, values in batch.items():
                if len(values) != batch_size:
                    raise MiaDeliveryError(
                        f"vLLM-Metal Q/K capture layer {layer_num} has inconsistent batch rows"
                    )
                for idx, value in enumerate(values):
                    rows[batch_cursor + idx][layer_num] = value
            batch_cursor += batch_size

        if batch_cursor != request_count or self._config is None:
            raise MiaDeliveryError(
                f"vLLM-Metal Q/K capture produced {batch_cursor} prompt rows for "
                f"{request_count} generated requests"
            )
        for request_idx, request_layers in enumerate(rows):
            missing = self.layers - request_layers.keys()
            if missing:
                raise MiaDeliveryError(
                    f"vLLM-Metal Q/K capture missed configured layers "
                    f"{sorted(missing)} for request {request_idx}; full, unchunked prefill "
                    "is required"
                )

        output = []
        for request_layers in rows:
            qk_cache = {}
            for layer_num in sorted(request_layers):
                q_tensor, k_tensor = request_layers[layer_num]
                module_name = f"model.layers.{layer_num}.self_attn.attn"
                qk_cache[module_name] = {
                    "q": [q_tensor],
                    "k_all": [k_tensor],
                    "layer_num": layer_num,
                    "hookq_mode": self.hookq_mode,
                }
            output.append({"config": dict(self._config), "qk_cache": qk_cache})
        return output


@contextmanager
def capture_generation(layer_to_heads: dict[int, list], hookq_mode: str) -> Iterator[MetalQKCapture]:
    """Activate capture across the in-process vLLM-Metal generation call."""
    global _ACTIVE_CAPTURE
    with _CAPTURE_LOCK:
        if _ACTIVE_CAPTURE is not None:
            raise MiaConfigurationError("nested vLLM-Metal MIA capture is not supported")
        capture = MetalQKCapture(layer_to_heads, hookq_mode)
        _ACTIVE_CAPTURE = capture
        try:
            yield capture
        finally:
            _ACTIVE_CAPTURE = None


def _install_qwen2_capture(model: Any) -> None:
    layers_container = getattr(model, "model", model)
    layers = getattr(layers_container, "layers", None)
    if layers is None:
        raise MiaConfigurationError(
            f"vLLM-Metal Attention Tracker expected Qwen2 model.layers on "
            f"{type(model).__module__}.{type(model).__name__}"
        )

    matched = 0
    for layer_num, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        cls = type(attention)
        module_name = cls.__module__
        if not module_name.endswith(".qwen2"):
            continue
        setattr(attention, "_mia_metal_layer_num", layer_num)
        if cls in _PATCHED_CLASSES:
            matched += 1
            continue
        original_call = cls.__call__

        @functools.wraps(original_call)
        def wrapped_call(self, *args, __original=original_call, **kwargs):
            global _ACTIVE_LAYER
            previous = _ACTIVE_LAYER
            layer_id = getattr(self, "_mia_metal_layer_num", None)
            if _ACTIVE_CAPTURE is not None and layer_id is not None:
                _ACTIVE_CAPTURE.begin_layer(layer_id)
                _ACTIVE_LAYER = layer_id
            try:
                return __original(self, *args, **kwargs)
            finally:
                _ACTIVE_LAYER = previous

        cls.__call__ = wrapped_call
        _PATCHED_CLASSES.add(cls)
        matched += 1

        if module_name not in _PATCHED_SDPA_MODULES:
            module = importlib.import_module(module_name)
            original_sdpa = module.scaled_dot_product_attention

            @functools.wraps(original_sdpa)
            def wrapped_sdpa(*args, __original=original_sdpa, **kwargs):
                if _ACTIVE_CAPTURE is not None and _ACTIVE_LAYER is not None:
                    queries = args[0] if args else kwargs.get("queries")
                    keys = args[1] if len(args) > 1 else kwargs.get("keys")
                    scale = args[4] if len(args) > 4 else kwargs.get("scale")
                    if queries is None or keys is None or scale is None:
                        raise MiaDeliveryError(
                            "Qwen2 SDPA call did not expose query, key, and scale arguments"
                        )
                    _ACTIVE_CAPTURE.record(queries, keys, float(scale))
                return __original(*args, **kwargs)

            module.scaled_dot_product_attention = wrapped_sdpa
            _PATCHED_SDPA_MODULES.add(module_name)

    if matched == 0:
        raise MiaConfigurationError(
            "vLLM-Metal Attention Tracker could not find Qwen2 attention modules; "
            "the configured model must use mlx_lm.models.qwen2"
        )


def install_lifecycle_capture() -> None:
    """Install capture wrappers after vLLM-Metal has loaded its MLX model."""
    global _LIFECYCLE_PATCHED
    if _LIFECYCLE_PATCHED:
        return
    lifecycle = importlib.import_module("vllm_metal.v1.model_lifecycle")
    original_load = lifecycle.ModelLifecycle.load

    @functools.wraps(original_load)
    def wrapped_load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        _install_qwen2_capture(self._runner.model)
        return result

    lifecycle.ModelLifecycle.load = wrapped_load
    _LIFECYCLE_PATCHED = True
