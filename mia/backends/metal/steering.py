"""MLX decoder-layer activation steering for vLLM-Metal models."""
from __future__ import annotations

from typing import Any


class MetalSteeringHandle:
    """Owns temporary decoder-layer wrappers and restores them on close."""

    def __init__(self, model: Any, config: dict[str, Any]) -> None:
        import mlx.core as mx
        from vllm_metal.attention.patching import find_layers

        self._mx = mx
        self._layers = find_layers(model)
        self._config = dict(config)
        self._originals: dict[type, Any] = {}
        self._installed: list[Any] = []
        self._closed = False
        phase = self._config.get("phase", "both")
        positions = self._config.get(
            "positions",
            "all_tokens" if self._config.get("apply_at_all_positions", True)
            else "last_token",
        )
        if phase != "both" or positions != "all_tokens":
            raise NotImplementedError(
                "The Metal steering adapter currently supports phase='both' and "
                "positions='all_tokens' only."
            )

        vector_path = self._config.get("vector_path")
        if not vector_path:
            raise ValueError("Metal steering requires steering.vector_path")
        import torch

        raw = torch.load(vector_path, map_location="cpu", weights_only=False)
        direction = raw.get("dir")
        if direction is None:
            raise ValueError(f"Steering vector at {vector_path!r} has no 'dir' entry")
        self._direction = mx.array(direction.detach().cpu().numpy() if torch.is_tensor(direction)
                                   else direction)
        self._average = raw.get("avg_proj")
        if self._average is not None and torch.is_tensor(self._average):
            self._average = float(self._average.item())
        self._average = None if self._average is None else float(self._average)

        target_layers = self._config.get("optimal_layer", "all")
        if target_layers == "all":
            selected = range(len(self._layers))
        else:
            raw_layers = target_layers if isinstance(target_layers, (list, tuple)) else [target_layers]
            selected = sorted({int(index) for index in raw_layers
                               if 0 <= int(index) < len(self._layers)})
        if not selected:
            raise ValueError("steering.optimal_layer selects no decoder layers")
        try:
            for layer_idx in selected:
                layer = self._layers[layer_idx]
                self._install_layer(type(layer), layer_idx, layer)
        except Exception:
            self.close()
            raise

    def _install_layer(self, layer_type: type, layer_idx: int, layer: Any) -> None:
        if layer_type not in self._originals:
            original = layer_type.__call__
            self._originals[layer_type] = original
            owner = self

            def wrapped(instance, *args, **kwargs):
                output = original(instance, *args, **kwargs)
                if getattr(instance, "_mia_metal_steer_owner", None) is not owner:
                    return output
                if isinstance(output, tuple):
                    if len(output) < 2:
                        return output
                    values = list(output)
                    values[1] = owner._steer(values[1])
                    return tuple(values)
                return owner._steer(output)

            layer_type.__call__ = wrapped
        object.__setattr__(layer, "_mia_metal_steer_owner", self)
        self._installed.append(layer)

    def _steer(self, rows):
        mx = self._mx
        direction = self._direction.astype(rows.dtype)
        method = self._config.get("method", "adjust_rs")
        if method == "add_vector":
            coefficient = float(self._config.get("coefficient", 0))
            return rows + coefficient * direction
        if method == "adjust_rs":
            if self._average is None:
                raise ValueError("adjust_rs steering vector must contain 'avg_proj'")
            projection = (rows * direction).sum(axis=-1, keepdims=True)
            return rows + (self._average - projection) * direction
        raise ValueError(f"Unknown steering method: {method}")

    def close(self) -> None:
        if self._closed:
            return
        for layer in self._installed:
            if getattr(layer, "_mia_metal_steer_owner", None) is self:
                object.__delattr__(layer, "_mia_metal_steer_owner")
        for layer_type, original in self._originals.items():
            layer_type.__call__ = original
        self._installed.clear()
        self._originals.clear()
        self._closed = True

    def __enter__(self) -> "MetalSteeringHandle":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def install_activation_steering(model: Any, config: dict[str, Any]) -> MetalSteeringHandle:
    """Install temporary activation steering using the existing steering config."""
    return MetalSteeringHandle(model, config)
