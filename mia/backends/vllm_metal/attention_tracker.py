"""Q/K capture primitives for the vLLM-Metal SDPA execution path.

This module observes Q/K after vLLM-Metal's RoPE preparation. Generation
continues to run through the vLLM engine and its Metal attention runtime.
"""
from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


_ACTIVE_CAPTURE: contextvars.ContextVar["MetalQKCapture | None"] = (
    contextvars.ContextVar("mia_metal_qk_capture", default=None)
)
_PATCH_LOCK = threading.RLock()
_PATCH_STATE: tuple[Any, Any, Any, Any] | None = None


@dataclass(frozen=True)
class QKCapture:
    """One packed-sequence capture in MIA's established Q/K tensor contract."""

    layer_num: int
    q: torch.Tensor
    k_all: torch.Tensor


def _mlx_float32_numpy(value: Any) -> np.ndarray:
    """Materialize an MLX array in a NumPy-compatible dtype."""
    import mlx.core as mx

    value = value.astype(mx.float32)
    mx.eval(value)
    return np.asarray(value).copy()


def split_packed_qk(
    layer_num: int,
    queries: np.ndarray,
    keys: np.ndarray,
    cu_seqlens: list[int] | tuple[int, ...],
) -> list[QKCapture]:
    """Split SDPA's ``[1, heads, tokens, dim]`` arrays by packed request.

    The query contract matches MIA's ``last_token`` mode (flattened head
    vectors); key rows match ``unpack_qk`` (flattened KV-head vectors).
    """
    q = np.asarray(queries)
    k = np.asarray(keys)
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("Metal Q/K capture expects singleton-batch [1, heads, tokens, dim] arrays")
    if len(cu_seqlens) < 2 or cu_seqlens[0] != 0:
        raise ValueError("cu_seqlens must start at zero and contain request boundaries")
    if q.shape[2] != k.shape[2] or cu_seqlens[-1] != q.shape[2]:
        raise ValueError("cu_seqlens must cover the packed Q/K token dimension")
    if any(right <= left for left, right in zip(cu_seqlens, cu_seqlens[1:])):
        raise ValueError("cu_seqlens request boundaries must be strictly increasing")

    result = []
    for start, end in zip(cu_seqlens, cu_seqlens[1:]):
        q_last = q[0, :, end - 1, :].reshape(-1).copy()
        k_rows = k[0, :, start:end, :].transpose(1, 0, 2).reshape(end - start, -1).copy()
        result.append(
            QKCapture(
                layer_num=int(layer_num),
                q=torch.from_numpy(q_last),
                k_all=torch.from_numpy(k_rows),
            )
        )
    return result


class MetalQKCapture:
    """Temporarily observe Q/K at vLLM-Metal's SDPA projection boundary.

    Use one capture scope per generation call. Only full, uncached prefill
    passes are recorded; decode and continuation-prefill passes are omitted
    because their complete key prefixes are not available at this boundary.
    Call ``captures`` after vLLM generation returns; the returned tensors are
    CPU tensors and independent of the Metal runtime's KV cache.
    """

    def __init__(self) -> None:
        self._captures: list[QKCapture] = []
        self._lock = threading.Lock()
        self._token = None

    def __enter__(self) -> "MetalQKCapture":
        if self._token is not None:
            raise RuntimeError("MetalQKCapture scopes cannot be re-entered")
        self._install_patch()
        self._token = _ACTIVE_CAPTURE.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        if self._token is not None:
            _ACTIVE_CAPTURE.reset(self._token)
            self._token = None

    @property
    def captures(self) -> list[QKCapture]:
        with self._lock:
            return list(self._captures)

    def _record(self, layer_num: int, queries: Any, keys: Any, ctx: Any) -> None:
        boundaries = getattr(ctx, "cu_seqlens", None)
        if boundaries is None:
            raise RuntimeError("vLLM-Metal attention context has no cu_seqlens")
        if int(getattr(ctx, "num_decode_requests", 0)) > 0:
            return
        if any(int(offset) > 0 for offset in (getattr(ctx, "offsets", None) or ())):
            return
        q_np = _mlx_float32_numpy(queries)
        k_np = _mlx_float32_numpy(keys)
        captures = split_packed_qk(layer_num, q_np, k_np, boundaries)
        with self._lock:
            self._captures.extend(captures)

    @staticmethod
    def _install_patch() -> None:
        global _PATCH_STATE
        with _PATCH_LOCK:
            if _PATCH_STATE is not None:
                return
            try:
                from vllm_metal.attention import context as metal_context
                from vllm_metal.attention.impls import sdpa, sdpa_wrapper
            except ImportError as exc:
                raise RuntimeError(
                    "Metal Q/K capture requires the vLLM-Metal SDPA runtime"
                ) from exc

            original_prepare = sdpa.prepare_sdpa_qkv
            original_forward = sdpa_wrapper.sdpa_forward
            layer_context: contextvars.ContextVar[int | None] = contextvars.ContextVar(
                "mia_metal_qk_layer", default=None
            )

            def prepare(*args, **kwargs):
                prepared = original_prepare(*args, **kwargs)
                capture = _ACTIVE_CAPTURE.get()
                layer_num = layer_context.get()
                ctx = metal_context.get_context()
                if capture is not None and layer_num is not None and ctx is not None:
                    capture._record(layer_num, prepared[0], prepared[1], ctx)
                return prepared

            def forward(inner, x, ctx, kv_cache, layer_idx, *args, **kwargs):
                token = layer_context.set(int(layer_idx))
                try:
                    return original_forward(
                        inner, x, ctx, kv_cache, layer_idx, *args, **kwargs
                    )
                finally:
                    layer_context.reset(token)

            sdpa.prepare_sdpa_qkv = prepare
            sdpa_wrapper.sdpa_forward = forward
            _PATCH_STATE = (sdpa, sdpa_wrapper, original_prepare, original_forward)

    @staticmethod
    def restore_patch() -> None:
        """Restore the shared SDPA functions when the backend is unloaded."""
        global _PATCH_STATE
        with _PATCH_LOCK:
            if _PATCH_STATE is None:
                return
            sdpa, sdpa_wrapper, prepare, forward = _PATCH_STATE
            sdpa.prepare_sdpa_qkv = prepare
            sdpa_wrapper.sdpa_forward = forward
            _PATCH_STATE = None
