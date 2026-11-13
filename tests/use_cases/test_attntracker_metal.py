"""Focused lifecycle checks for the vLLM-Metal steering adapter."""
import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("vllm_metal.attention.patching")

from mia.backends.metal.steering import install_activation_steering


class _Layer(nn.Module):
    def __call__(self, x):
        return x


class _Model:
    def __init__(self):
        self.model = type("_Decoder", (), {"layers": [_Layer(), _Layer()]})()


def test_additive_steering_restores_layer_call(tmp_path):
    vector_path = tmp_path / "direction.pt"
    torch.save({"dir": torch.tensor([1.0, 0.0])}, vector_path)
    model = _Model()
    layer = model.model.layers[0]
    original = layer(mx.array([[2.0, 3.0]]))

    with install_activation_steering(
        model,
        {
            "vector_path": str(vector_path),
            "method": "add_vector",
            "coefficient": 0.5,
            "optimal_layer": [0],
        },
    ):
        steered = mx.eval(layer(mx.array([[2.0, 3.0]])))
        np.testing.assert_allclose(np.asarray(steered), [[2.5, 3.0]])

    restored = mx.eval(layer(mx.array([[2.0, 3.0]])))
    np.testing.assert_allclose(np.asarray(restored), np.asarray(mx.eval(original)))


def test_invalid_steering_method_fails_at_application(tmp_path):
    vector_path = tmp_path / "direction.pt"
    torch.save({"dir": torch.tensor([1.0, 0.0])}, vector_path)
    model = _Model()
    handle = install_activation_steering(
        model, {"vector_path": str(vector_path), "method": "unknown", "optimal_layer": 0}
    )
    try:
        with pytest.raises(ValueError, match="Unknown steering method"):
            model.model.layers[0](mx.array([[2.0, 3.0]]))
    finally:
        handle.close()
