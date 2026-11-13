"""Q/K capture primitives for MLX Llama attention executed by vLLM-Metal.

The capture point is the existing MLX scaled-dot-product attention call: its
queries and keys are already RoPE-applied, and the key tensor includes cached
prefix tokens. No generation is performed by this module.
"""
from __future__ import annotations

import contextvars
import functools
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch


@dataclass
class AttentionTrackerCapture:
    """Request-local Attention Tracker captures in the repository's probe format."""

    layer_to_heads: dict[int, list[int]]
    hookq_mode: str = "last_token"
    qk_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.hookq_mode != "last_token":
            raise ValueError(
                "The vLLM-Metal Attention Tracker backend currently captures "
                "hookq_mode='last_token' only"
            )

    def record(self, layer_num: int, queries: Any, keys: Any, attention: Any) -> None:
        """Store the last query row and full cached K prefix for one layer."""
        if layer_num not in self.layer_to_heads:
            return

        q = _as_numpy(queries)
        k = _as_numpy(keys)
        if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1:
            raise ValueError(
                "Metal Attention Tracker expects single-request Q/K arrays shaped "
                "(1, heads, tokens, head_dim); received "
                f"Q={q.shape}, K={k.shape}"
            )
        if q.shape[1] < max(self.layer_to_heads[layer_num], default=-1) + 1:
            raise ValueError(f"Configured query head is absent in layer {layer_num}")

        q_rows = q[0, :, -1, :].reshape(1, -1)
        k_rows = k[0].transpose(1, 0, 2).reshape(k.shape[2], -1)
        scale = float(getattr(attention, "scale"))
        heads_q = int(q.shape[1])
        heads_kv = int(k.shape[1])
        head_dim = int(q.shape[-1])

        if self.config is None:
            self.config = {
                "num_attention_heads": heads_q,
                "num_key_value_heads": heads_kv,
                "head_dim": head_dim,
                "attention_multiplier": scale,
            }
        elif self.config != {
            "num_attention_heads": heads_q,
            "num_key_value_heads": heads_kv,
            "head_dim": head_dim,
            "attention_multiplier": scale,
        }:
            raise ValueError("Attention dimensions differ between captured layers")

        name = f"model.layers.{layer_num}.self_attn"
        self.qk_cache[name] = {
            "layer_num": layer_num,
            "heads": list(self.layer_to_heads[layer_num]),
            "hookq_mode": self.hookq_mode,
            "q": [_as_torch(q_rows, queries)],
            "k_all": [_as_torch(k_rows, keys)],
        }

    def probes(self) -> dict[str, Any]:
        """Return the public analyzer input contract used by ``MiaLLM.analyze``."""
        if not self.qk_cache or self.config is None:
            raise RuntimeError("No Attention Tracker Q/K rows were captured")
        return {"config": dict(self.config), "qk_cache": dict(self.qk_cache)}


_CURRENT_CAPTURE: contextvars.ContextVar[AttentionTrackerCapture | None] = (
    contextvars.ContextVar("mia_metal_attention_tracker_capture", default=None)
)
_CURRENT_LAYER: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mia_metal_attention_tracker_layer", default=None
)
_CURRENT_ATTENTION: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "mia_metal_attention_tracker_attention", default=None
)
_PATCHED_ATTENTION_CLASSES: set[type] = set()
_PATCHED_ATTENTION_FUNCTIONS: set[tuple[str, str]] = set()
_PATCHED_RUNNER_CLASSES: set[type] = set()
_RUNNER_CAPTURES: dict[str, dict[str, Any]] = {}
_RUNNER_CAPTURES_LOCK = threading.Lock()


def _as_numpy(value: Any) -> np.ndarray:
    if "bfloat16" in str(getattr(value, "dtype", "")).lower():
        import mlx.core as mx

        return np.asarray(value.astype(mx.float32)).copy()
    return np.asarray(value).copy()


def _as_torch(array: np.ndarray, source: Any) -> torch.Tensor:
    if "bfloat16" in str(getattr(source, "dtype", "")).lower():
        return torch.from_numpy(array.astype(np.float32, copy=False)).to(torch.bfloat16)
    return torch.from_numpy(array)


@contextmanager
def capture_scope(capture: AttentionTrackerCapture) -> Iterator[AttentionTrackerCapture]:
    """Set a capture target around a model forward without changing its signature."""
    capture_token = _CURRENT_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _CURRENT_CAPTURE.reset(capture_token)


def _patch_attention_function(module: Any) -> None:
    function = getattr(module, "scaled_dot_product_attention", None)
    if not callable(function):
        return
    key = (module.__name__, "scaled_dot_product_attention")
    if key in _PATCHED_ATTENTION_FUNCTIONS:
        return

    @functools.wraps(function)
    def wrapped(queries, keys, values, *args, **kwargs):
        capture = _CURRENT_CAPTURE.get()
        layer_num = _CURRENT_LAYER.get()
        attention = _CURRENT_ATTENTION.get()
        if capture is not None and layer_num is not None and attention is not None:
            capture.record(layer_num, queries, keys, attention)
        return function(queries, keys, values, *args, **kwargs)

    module.scaled_dot_product_attention = wrapped
    _PATCHED_ATTENTION_FUNCTIONS.add(key)


def install_llama_attention_capture(model: Any) -> int:
    """Instrument an already-loaded mlx-lm Llama model for request-scoped Q/K capture.

    Returns the number of attention layers instrumented and fails explicitly if
    the model layout or upstream attention function is not the inspected Llama
    contract.
    """
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise TypeError(f"Cannot find Llama transformer layers in {type(model).__name__}")

    instrumented = 0
    for layer_num, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        module = sys.modules.get(type(attention).__module__)
        if module is None or not callable(
            getattr(module, "scaled_dot_product_attention", None)
        ):
            raise TypeError(
                f"{type(attention).__module__} does not expose the expected "
                "scaled_dot_product_attention capture point"
            )
        _patch_attention_function(module)

        cls = type(attention)
        if cls not in _PATCHED_ATTENTION_CLASSES:
            original_call = cls.__call__

            @functools.wraps(original_call)
            def wrapped_call(self, *args, __original=original_call, **kwargs):
                capture = _CURRENT_CAPTURE.get()
                layer = getattr(self, "_mia_metal_layer_num", None)
                if capture is None or layer is None:
                    return __original(self, *args, **kwargs)
                token = _CURRENT_LAYER.set(int(layer))
                attention_token = _CURRENT_ATTENTION.set(self)
                try:
                    return __original(self, *args, **kwargs)
                finally:
                    _CURRENT_ATTENTION.reset(attention_token)
                    _CURRENT_LAYER.reset(token)

            cls.__call__ = wrapped_call
            _PATCHED_ATTENTION_CLASSES.add(cls)
        object.__setattr__(attention, "_mia_metal_layer_num", layer_num)
        instrumented += 1

    if not instrumented:
        raise TypeError("No Llama self_attn layers found to instrument")
    return instrumented


def install_vllm_metal_runner_capture(runner_class: type) -> None:
    """Bridge the selected Q/K request arguments into Metal prefill capture.

    The runner receives vLLM's real ``SamplingParams`` and runs the model's
    real MLX forward pass. Results are retained by request ID for the caller to
    attach to ``RequestOutput`` through MIA's normal delivery path.
    """
    if runner_class in _PATCHED_RUNNER_CLASSES:
        return
    original_prefill = runner_class._prefill_single

    @functools.wraps(original_prefill)
    def prefill_single(self, req_id, token_ids, sampling_params, *args, **kwargs):
        extra = getattr(sampling_params, "extra_args", None) or {}
        output_qk = extra.get("output_qk")
        if not isinstance(output_qk, dict):
            return original_prefill(
                self, req_id, token_ids, sampling_params, *args, **kwargs
            )
        if extra.get("hooks_on", "prefill") != "prefill":
            raise NotImplementedError(
                "The vLLM-Metal Attention Tracker bridge currently captures "
                "prefill attention only"
            )
        install_llama_attention_capture(self.model)
        selected_layers = {
            int(layer): [int(head) for head in heads]
            for layer, heads in output_qk.items()
            if heads
        }
        capture = AttentionTrackerCapture(
            layer_to_heads=selected_layers,
            hookq_mode=extra.get("hookq_mode", "last_token"),
        )
        with capture_scope(capture):
            result = original_prefill(
                self, req_id, token_ids, sampling_params, *args, **kwargs
            )
        if selected_layers:
            probes = capture.probes()
            with _RUNNER_CAPTURES_LOCK:
                _RUNNER_CAPTURES[str(req_id)] = probes
        return result

    runner_class._prefill_single = prefill_single
    _PATCHED_RUNNER_CLASSES.add(runner_class)


def take_runner_capture(request_id: str) -> dict[str, Any] | None:
    """Remove and return a completed capture produced for ``request_id``."""
    with _RUNNER_CAPTURES_LOCK:
        return _RUNNER_CAPTURES.pop(str(request_id), None)
