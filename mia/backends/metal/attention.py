"""Q/K capture primitives for vLLM-Metal's MLX attention path.

The adapter observes the post-RoPE Q/K arrays produced by vLLM-Metal's own
SDPA preparation. It never runs generation or attention independently.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from typing import Any, Iterator

import numpy as np
import torch


_ACTIVE_CAPTURE: contextvars.ContextVar[MetalQKCapture | None] = contextvars.ContextVar(
    "mia_metal_qk_capture", default=None
)
_ACTIVE_LAYER: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mia_metal_attention_layer", default=None
)
_PATCH_LOCK = threading.RLock()


def _mlx_array_to_torch(value: Any) -> torch.Tensor:
    import mlx.core as mx

    mx.eval(value)
    return torch.from_numpy(np.asarray(value.astype(mx.float32)).copy())


class MetalQKCapture:
    """Collect per-layer, per-prefill Q/K arrays in MIA's probe contract."""

    def __init__(self, config: dict[str, Any], *, hookq_mode: str = "last_token"):
        if hookq_mode not in ("last_token", "all_tokens"):
            raise ValueError(f"unsupported hookq_mode {hookq_mode!r}")
        self.config = dict(config)
        self.hookq_mode = hookq_mode
        self.qk_cache: dict[str, dict[str, Any]] = {}

    def record(self, layer_num: int, queries: Any, keys: Any, context: Any) -> None:
        """Record one MLX SDPA preparation result without changing its outputs."""
        offsets = getattr(context, "cu_seqlens", None)
        if not offsets or len(offsets) < 2:
            raise RuntimeError("vLLM-Metal attention context has no cu_seqlens")

        boundaries = [int(value) for value in offsets]
        decode_count = int(getattr(context, "num_decode_requests", 0))
        layer_name = f"layer.{int(layer_num)}"
        entry = self.qk_cache.setdefault(
            layer_name,
            {"q": [], "k_all": [], "layer_num": int(layer_num), "hookq_mode": self.hookq_mode},
        )

        q_flat = queries[0].transpose(1, 0, 2)
        k_flat = keys[0].transpose(1, 0, 2)
        for request_idx in range(decode_count, len(boundaries) - 1):
            start, end = boundaries[request_idx : request_idx + 2]
            if end <= start:
                continue
            q_rows = q_flat[start:end]
            q_selected = q_rows[-1:] if self.hookq_mode == "last_token" else q_rows
            entry["q"].append(_mlx_array_to_torch(q_selected.reshape(q_selected.shape[0], -1)))
            entry["k_all"].append(
                _mlx_array_to_torch(k_flat[start:end].reshape(end - start, -1))
            )

    def probes(self) -> dict[str, Any]:
        """Return the public analyzer input shape consumed by ``MiaLLM.analyze``."""
        return {"config": dict(self.config), "qk_cache": self.qk_cache}


@contextlib.contextmanager
def capture_vllm_metal_attention(capture: MetalQKCapture) -> Iterator[MetalQKCapture]:
    """Temporarily intercept vLLM-Metal Q/K preparation in the current process.

    This context manager is a backend primitive, not an automatic MIA worker
    integration. The target model still executes through vLLM-Metal.
    """
    import vllm_metal.attention.impls.sdpa as sdpa
    import vllm_metal.attention.impls.sdpa_wrapper as sdpa_wrapper

    with _PATCH_LOCK:
        original_prepare = sdpa.prepare_sdpa_qkv
        wrapper_class = sdpa_wrapper.SDPAPagedAttentionWrapper
        original_call = wrapper_class.__call__

        def capture_prepare(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            active = _ACTIVE_CAPTURE.get()
            layer_num = _ACTIVE_LAYER.get()
            if active is not None and layer_num is not None:
                active.record(layer_num, result[0], result[1], args[2])
            return result

        def capture_call(self, *args, **kwargs):
            token = _ACTIVE_LAYER.set(int(self._mk_layer_idx))
            try:
                return original_call(self, *args, **kwargs)
            finally:
                _ACTIVE_LAYER.reset(token)

        sdpa.prepare_sdpa_qkv = capture_prepare
        wrapper_class.__call__ = capture_call
        active_token = _ACTIVE_CAPTURE.set(capture)
        try:
            yield capture
        finally:
            _ACTIVE_CAPTURE.reset(active_token)
            wrapper_class.__call__ = original_call
            sdpa.prepare_sdpa_qkv = original_prepare
