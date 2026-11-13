"""MLX Q/K capture primitives for the vLLM-Metal Qwen2 attention path.

The target's Qwen2 implementation calls scaled_dot_product_attention with
post-RoPE queries and keys. A Metal integration can pass those arrays here
without changing the model computation.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch


_ACTIVE_CAPTURE: ContextVar["MetalQKCapture | None"] = ContextVar(
    "mia_metal_qk_capture", default=None
)


def _mlx_array_to_numpy(value) -> np.ndarray:
    """Materialize an MLX array as float32 before crossing into NumPy."""
    try:
        import mlx.core as mx
    except ImportError:
        return np.asarray(value, dtype=np.float32)
    return np.asarray(value.astype(mx.float32))


@dataclass
class MetalQKCapture:
    """Collect single-request Q/K probes in the existing analyzer contract.

    ``record`` expects post-RoPE arrays shaped ``[batch, heads, tokens, dim]``.
    For the Attention Tracker notebook, the last query row and the complete
    key prefix are retained once per layer, matching ``hookq_mode=last_token``.
    """

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_multiplier: float
    _layers: dict[str, dict] = field(default_factory=dict)

    def record(self, layer_num: int, queries, keys) -> None:
        """Record one layer's post-RoPE queries and full key prefix."""
        layer_num = int(layer_num)
        layer_name = f"model.layers.{layer_num}.self_attn.attn"
        if layer_name in self._layers:
            return

        q = _mlx_array_to_numpy(queries)
        k = _mlx_array_to_numpy(keys)
        expected_heads = (self.num_attention_heads, self.num_key_value_heads)
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(
                f"Metal Q/K capture expects rank-4 [batch, heads, tokens, dim] "
                f"arrays; got q={q.shape}, k={k.shape}"
            )
        if q.shape[0] != 1 or k.shape[0] != 1:
            raise ValueError(
                "Metal Q/K capture currently requires a single active request; "
                f"got batch sizes q={q.shape[0]}, k={k.shape[0]}"
            )
        if q.shape[1] != expected_heads[0] or k.shape[1] != expected_heads[1]:
            raise ValueError(
                f"Metal Q/K head counts do not match model configuration "
                f"{expected_heads}: q={q.shape}, k={k.shape}"
            )
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(
                f"Metal Q/K head dimension must be {self.head_dim}; "
                f"got q={q.shape[-1]}, k={k.shape[-1]}"
            )
        if q.shape[2] == 0 or k.shape[2] == 0:
            raise ValueError("Metal Q/K capture cannot record empty token dimensions")

        q_rows = q[0].transpose(1, 0, 2).reshape(
            q.shape[2], self.num_attention_heads * self.head_dim
        )
        k_rows = k[0].transpose(1, 0, 2).reshape(
            k.shape[2], self.num_key_value_heads * self.head_dim
        )
        self._layers[layer_name] = {
            "q": [torch.from_numpy(np.ascontiguousarray(q_rows[-1])).clone()],
            "k_all": [torch.from_numpy(np.ascontiguousarray(k_rows)).clone()],
            "layer_num": layer_num,
            "hookq_mode": "last_token",
        }

    def to_probes(self) -> dict:
        """Return ``probes`` in the shape consumed by MiaLLM and AttntrackerAnalyzer."""
        return {
            "qk_cache": dict(self._layers),
            "config": {
                "num_attention_heads": int(self.num_attention_heads),
                "num_key_value_heads": int(self.num_key_value_heads),
                "head_dim": int(self.head_dim),
                "attention_multiplier": float(self.attention_multiplier),
            },
        }


@contextmanager
def capture_session(capture: MetalQKCapture) -> Iterator[MetalQKCapture]:
    """Make a capture available to an attention adapter on the current context."""
    token: Token[MetalQKCapture | None] = _ACTIVE_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _ACTIVE_CAPTURE.reset(token)


def record_active_qk(layer_num: int, queries, keys) -> bool:
    """Record Q/K into the current session; return False when capture is unarmed."""
    capture = _ACTIVE_CAPTURE.get()
    if capture is None:
        return False
    capture.record(layer_num, queries, keys)
    return True
