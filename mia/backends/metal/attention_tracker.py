"""MLX projection instrumentation for Llama-family Attention Tracker capture.

The adapter records post-RoPE Q/K from the actual mlx_lm model. It deliberately
does not generate tokens: generation remains owned by vLLM-Metal.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

import torch


class MetalAttentionCapture:
    """Capture Attention Tracker Q/K from a loaded mlx_lm Llama model.

    ``install`` wraps the model's existing Q/K projections. Call
    ``capture_scope`` around the vLLM-Metal model forward that belongs to a
    request, then pass ``to_probes()`` to MIA's existing analyzer.
    """

    def __init__(self, model: Any, layer_to_heads: dict[int, list[int]]) -> None:
        import mlx.core as mx

        self._mx = mx
        args = getattr(model, "args", None)
        if args is None:
            raise TypeError("MetalAttentionCapture requires an mlx_lm model with .args")
        self.config = {
            "num_attention_heads": int(args.num_attention_heads),
            "num_key_value_heads": int(
                getattr(args, "num_key_value_heads", args.num_attention_heads)
            ),
            "head_dim": int(
                getattr(args, "head_dim", None)
                or args.hidden_size // args.num_attention_heads
            ),
            "attention_multiplier": float(
                (getattr(args, "head_dim", None)
                 or args.hidden_size // args.num_attention_heads) ** -0.5
            ),
        }
        self.layer_to_heads = {
            int(layer): [int(head) for head in heads]
            for layer, heads in layer_to_heads.items()
        }
        self._active = threading.local()
        self._captured: dict[int, dict[str, torch.Tensor]] = {}
        self._originals: list[tuple[Any, str, Any]] = []
        try:
            self._install(model)
        except Exception:
            self.close()
            raise

    def _install(self, model: Any) -> None:
        import mlx.nn as nn

        class ProjectionCapture(nn.Module):
            def __init__(self, inner: Any, on_output) -> None:
                super().__init__()
                object.__setattr__(self, "_inner", inner)
                object.__setattr__(self, "_on_output", on_output)

            def __call__(self, *args, **kwargs):
                output = self._inner(*args, **kwargs)
                self._on_output(output)
                return output

        root = getattr(model, "language_model", model)
        layers_container = getattr(root, "model", root)
        layers = getattr(layers_container, "layers", None)
        if layers is None:
            raise TypeError(
                f"Cannot locate mlx_lm transformer layers in {type(model).__name__}"
            )

        for layer_num, layer in enumerate(layers):
            if self.layer_to_heads and layer_num not in self.layer_to_heads:
                continue
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                continue
            for projection_name, heads in (
                ("q_proj", self.config["num_attention_heads"]),
                ("k_proj", self.config["num_key_value_heads"]),
            ):
                projection = getattr(attention, projection_name, None)
                if projection is None:
                    raise TypeError(
                        f"Layer {layer_num} attention lacks {projection_name}"
                    )

                def on_output(output, *, ln=layer_num, attn=attention,
                              name=projection_name, n_heads=heads):
                    self._capture_projection(ln, attn, name, n_heads, output)

                self._originals.append((attention, projection_name, projection))
                setattr(
                    attention,
                    projection_name,
                    ProjectionCapture(projection, on_output),
                )

    @contextmanager
    def capture_scope(self, request_id: str, position_offset: int = 0) -> Iterator[None]:
        """Enable capture during one model forward; restore nested state on exit."""
        previous = getattr(self._active, "request", None)
        request = {
            "request_id": str(request_id),
            "position_offset": int(position_offset),
        }
        self._active.request = request
        try:
            yield
        finally:
            self._active.request = previous

    def _capture_projection(
        self, layer_num: int, attention: Any, name: str, heads: int, output: Any
    ) -> None:
        request = getattr(self._active, "request", None)
        if request is None:
            return
        if name not in ("q_proj", "k_proj"):
            return

        batch, seq_len, _ = output.shape
        if batch != 1:
            raise ValueError(
                "MetalAttentionCapture currently requires one request per forward; "
                f"got batch size {batch}"
            )
        head_dim = self.config["head_dim"]
        shaped = output.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
        rope = getattr(attention, "rope", None)
        if rope is None:
            raise TypeError(
                f"Layer {layer_num} attention does not expose mlx_lm's RoPE module"
            )
        rotated = rope(shaped, offset=request["position_offset"])
        if name == "q_proj":
            rotated = rotated[:, :, -1:, :]
        flattened = rotated.transpose(0, 2, 1, 3).reshape(
            batch, -1, heads * head_dim
        )[0]
        value = self._to_torch(flattened)
        entry = self._captured.setdefault(layer_num, {})
        entry["q" if name == "q_proj" else "k_all"] = value

    def _to_torch(self, value: Any) -> torch.Tensor:
        import numpy as np

        compatible = self._mx.contiguous(value.astype(self._mx.float32))
        self._mx.eval(compatible)
        return torch.from_numpy(np.asarray(compatible).copy())

    def to_probes(self) -> dict:
        """Return the standard MIA Q/K probe payload for the active capture."""
        qk_cache = {}
        for layer_num, tensors in sorted(self._captured.items()):
            if "q" not in tensors or "k_all" not in tensors:
                raise RuntimeError(
                    f"Incomplete Q/K capture for layer {layer_num}; "
                    "both projections must run in the same model forward"
                )
            qk_cache[str(layer_num)] = {
                "q": [tensors["q"]],
                "k_all": [tensors["k_all"]],
                "layer_num": layer_num,
                "hookq_mode": "last_token",
            }
        missing_layers = sorted(set(self.layer_to_heads) - self._captured.keys())
        if missing_layers:
            raise RuntimeError(
                f"No Q/K data was captured for configured layers {missing_layers}"
            )
        if not qk_cache:
            raise RuntimeError("No Q/K data was captured in the active request scope")
        return {"config": dict(self.config), "qk_cache": qk_cache}

    def clear(self) -> None:
        """Clear captured tensors between requests."""
        self._captured.clear()

    def close(self) -> None:
        """Restore every original projection; repeated calls are safe."""
        while self._originals:
            owner, name, original = self._originals.pop()
            setattr(owner, name, original)
