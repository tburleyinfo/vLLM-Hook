"""Q/K observation primitives for vLLM-Metal's MLX SDPA implementation.

The tap observes the post-normalization, post-RoPE arrays returned by
``prepare_sdpa_qkv``. It does not own request routing or artifact delivery.
"""
from __future__ import annotations

import contextvars
import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any


QKObserver = Callable[[int, Any, Any, Any], None]


class MetalQKTap:
    """Temporarily observe per-layer SDPA Q/K arrays without changing outputs.

    ``observer`` receives ``(layer_index, paged_context, queries, keys)``.
    Arrays are MLX arrays shaped ``[batch, heads, tokens, head_dim]``. The tap
    is inactive outside :meth:`capture` scopes and restores the target function
    on :meth:`close`.
    """

    def __init__(self, model: Any, observer: QKObserver) -> None:
        if not callable(observer):
            raise TypeError("observer must be callable")

        patching = importlib.import_module("vllm_metal.attention.patching")
        sdpa = importlib.import_module("vllm_metal.attention.impls.sdpa")
        self._sdpa: ModuleType = sdpa
        self._original = sdpa.prepare_sdpa_qkv
        self._observer = observer
        self._active: contextvars.ContextVar[bool] = contextvars.ContextVar(
            f"mia_metal_qk_active_{id(self)}", default=False
        )
        self._layer_by_attention = self._find_attention_layers(
            model, patching.find_layers
        )
        if not self._layer_by_attention:
            raise ValueError("No SDPA attention layers found in the loaded model")
        if getattr(self._original, "_mia_metal_qk_tap", False):
            raise RuntimeError("vLLM-Metal SDPA Q/K observation is already patched")

        def observed_prepare(*args, **kwargs):
            result = self._original(*args, **kwargs)
            if self._active.get():
                inner = args[0] if args else kwargs["inner"]
                ctx = args[2] if len(args) > 2 else kwargs["ctx"]
                layer = self._layer_by_attention.get(id(inner))
                if layer is not None:
                    queries, keys = result[0], result[1]
                    self._observer(layer, ctx, queries, keys)
            return result

        observed_prepare._mia_metal_qk_tap = True
        self._patched = observed_prepare
        sdpa.prepare_sdpa_qkv = observed_prepare
        self._closed = False

    @staticmethod
    def _find_attention_layers(model: Any, find_layers: Callable) -> dict[int, int]:
        result: dict[int, int] = {}
        for layer_idx, layer in enumerate(find_layers(model)):
            for name in ("self_attn", "linear_attn", "attention"):
                attention = getattr(layer, name, None)
                if attention is not None:
                    inner = getattr(attention, "_inner", attention)
                    result[id(inner)] = layer_idx
                    result[id(attention)] = layer_idx
                    break
        return result

    def capture(self):
        """Return a context manager enabling observation in the current context."""
        if self._closed:
            raise RuntimeError("cannot capture with a closed MetalQKTap")
        return _CaptureScope(self._active)

    def close(self) -> None:
        """Restore vLLM-Metal's original Q/K preparation function."""
        if self._closed:
            return
        if self._sdpa.prepare_sdpa_qkv is not self._patched:
            raise RuntimeError(
                "vLLM-Metal prepare_sdpa_qkv changed while MetalQKTap was installed"
            )
        self._sdpa.prepare_sdpa_qkv = self._original
        self._closed = True

    def __enter__(self) -> "MetalQKTap":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class _CaptureScope:
    def __init__(self, active: contextvars.ContextVar[bool]) -> None:
        self._active = active
        self._token = None

    def __enter__(self):
        self._token = self._active.set(True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._token is not None:
            self._active.reset(self._token)
