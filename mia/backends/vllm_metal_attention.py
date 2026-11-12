"""Attention Tracker capture bridge for vLLM-Metal's MLX attention path."""
from __future__ import annotations

import contextlib
import threading
from typing import Any

import torch


_PATCH_LOCK = threading.RLock()
_GENERATE_LOCK = threading.RLock()
_ACTIVE_CAPTURE = None
_ACTIVE_LAYER = threading.local()
_PATCHED_CLASSES: set[type] = set()
_PATCHED_GLOBALS: set[int] = set()
_INSTANCE_LAYERS: dict[int, tuple[int, Any]] = {}
_ORIGINAL_CALLS: dict[type, Any] = {}


class AttentionCapture:
    """Collect per-prompt last-query Q and full K tensors in the public probe schema."""

    def __init__(self, prompt_count: int) -> None:
        self.prompt_count = int(prompt_count)
        self._layers: dict[int, dict[str, Any]] = {}

    def record(self, layer_num: int, q: torch.Tensor, k: torch.Tensor, scale: float) -> None:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(f"Metal attention Q/K must be [batch, heads, seq, dim], got "
                             f"{tuple(q.shape)} and {tuple(k.shape)}")

        layer = self._layers.setdefault(
            int(layer_num),
            {"q": [], "k_all": [], "scale": float(scale), "q_heads": int(q.shape[1]),
             "kv_heads": int(k.shape[1]), "head_dim": int(q.shape[3])},
        )
        remaining = self.prompt_count - len(layer["q"])
        if remaining <= 0:
            return
        n = min(remaining, int(q.shape[0]), int(k.shape[0]))
        for i in range(n):
            query = q[i, :, -1, :].reshape(-1).detach().cpu().contiguous()
            keys = k[i].permute(1, 0, 2).reshape(k.shape[2], -1).detach().cpu().contiguous()
            layer["q"].append(query)
            layer["k_all"].append(keys)

    def probes(self, layer_names: dict[int, str]) -> dict:
        if not self._layers:
            raise RuntimeError("vLLM-Metal Attention Tracker did not observe any attention Q/K")
        incomplete = [
            layer for layer in layer_names
            if layer not in self._layers or len(self._layers[layer]["q"]) != self.prompt_count
        ]
        if incomplete:
            raise RuntimeError(
                "vLLM-Metal Attention Tracker could not capture every prompt for layer(s) "
                f"{incomplete}; ensure each prompt is processed as a full prefill."
            )

        qk_cache = {}
        q_heads = kv_heads = head_dim = scale = None
        for layer_num, name in layer_names.items():
            item = self._layers[layer_num]
            q_heads = item["q_heads"]
            kv_heads = item["kv_heads"]
            head_dim = item["head_dim"]
            scale = item["scale"]
            qk_cache[name] = {
                "q": item["q"],
                "k_all": item["k_all"],
                "layer_num": layer_num,
                "hookq_mode": "last_token",
            }
        return {
            "config": {
                "num_attention_heads": q_heads,
                "num_key_value_heads": kv_heads,
                "head_dim": head_dim,
                "attention_multiplier": scale,
            },
            "qk_cache": qk_cache,
        }


def _mlx_array_to_cpu_tensor(value) -> torch.Tensor:
    from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

    return mlx_to_torch(value, device="cpu").detach().contiguous()


def _wrap_sdpa(original):
    def observe(*args, **kwargs):
        active = _ACTIVE_CAPTURE
        layer_num = getattr(_ACTIVE_LAYER, "layer_num", None)
        if active is not None and layer_num is not None:
            q = args[0] if args else kwargs.get("queries")
            k = args[1] if len(args) > 1 else kwargs.get("keys")
            scale = args[4] if len(args) > 4 else kwargs.get("scale")
            if q is None or k is None or scale is None:
                raise RuntimeError(
                    "Unrecognized MLX scaled_dot_product_attention call signature"
                )
            active.record(int(layer_num), _mlx_array_to_cpu_tensor(q),
                          _mlx_array_to_cpu_tensor(k), float(scale))
        return original(*args, **kwargs)

    return observe


def _patch_attention_module(layer_num: int, module) -> None:
    cls = type(module)
    with _PATCH_LOCK:
        _INSTANCE_LAYERS[id(module)] = (layer_num, module)
        if cls not in _PATCHED_CLASSES:
            original_call = cls.__call__
            _ORIGINAL_CALLS[cls] = original_call

            def wrapped_call(self, *args, **kwargs):
                entry = _INSTANCE_LAYERS.get(id(self))
                if entry is None or entry[1] is not self:
                    return original_call(self, *args, **kwargs)
                previous = getattr(_ACTIVE_LAYER, "layer_num", None)
                _ACTIVE_LAYER.layer_num = entry[0]
                try:
                    return original_call(self, *args, **kwargs)
                finally:
                    if previous is None:
                        try:
                            del _ACTIVE_LAYER.layer_num
                        except AttributeError:
                            pass
                    else:
                        _ACTIVE_LAYER.layer_num = previous

            cls.__call__ = wrapped_call
            _PATCHED_CLASSES.add(cls)

        namespace = getattr(_ORIGINAL_CALLS[cls], "__globals__", {})
        original_sdpa = namespace.get("scaled_dot_product_attention")
        if callable(original_sdpa) and id(namespace) not in _PATCHED_GLOBALS:
            namespace["scaled_dot_product_attention"] = _wrap_sdpa(original_sdpa)
            _PATCHED_GLOBALS.add(id(namespace))


def install(model) -> dict[int, str]:
    """Patch the loaded Metal model's ordinary MLX SDPA calls without changing their results."""
    from vllm_metal.attention.patching import find_attn_attr, find_layers

    layers = find_layers(model)
    names = {}
    for layer_num, layer in enumerate(layers):
        attr = find_attn_attr(layer)
        if attr is None:
            continue
        attention = getattr(layer, attr)
        if not all(hasattr(attention, name) for name in ("q_proj", "k_proj")):
            continue
        _patch_attention_module(layer_num, attention)
        names[layer_num] = f"model.layers.{layer_num}.self_attn.attn"
    if not names:
        raise RuntimeError("No compatible SDPA attention modules found in the vLLM-Metal model")
    return names


@contextlib.contextmanager
def capture(prompt_count: int):
    """Enable capture around a normal vLLM generation call."""
    global _ACTIVE_CAPTURE
    collector = AttentionCapture(prompt_count)
    with _GENERATE_LOCK:
        previous = _ACTIVE_CAPTURE
        _ACTIVE_CAPTURE = collector
        try:
            yield collector
        finally:
            _ACTIVE_CAPTURE = previous
