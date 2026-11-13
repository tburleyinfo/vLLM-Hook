"""Contract tests for the Metal Attention Tracker request filter."""
import numpy as np
import pytest

pytest.importorskip("vllm")

from mia.backends.metal import (
    _capture_active,
    _qk_capture,
    _selected,
    _steer_layer_selected,
    _steer_modes,
)


@pytest.mark.parametrize(
    ("spec", "layer_num", "expected"),
    [
        (True, 10, True),
        ({10: [6], 11: [0]}, 10, True),
        ({"10": [6]}, 10, True),
        ([10, 11], 11, True),
        ({10: [6]}, 9, False),
        (None, 10, False),
    ],
)
def test_selected_qk_layers(spec, layer_num, expected):
    assert _selected(spec, layer_num) is expected


def test_qk_capture_defaults_to_prefill_only():
    state = {"extra": {"output_qk": {10: [6]}}}
    assert _capture_active(state, 10)
    state["extra"]["hooks_on"] = "decode"
    assert not _capture_active(state, 10)


def test_qk_capture_filter_excludes_non_prefill_phase():
    state = {"extra": {"output_qk": [10], "hooks_on": "both"}}
    assert not _capture_active(state, 10)


def test_steering_uses_existing_layer_and_phase_position_contract():
    assert _steer_layer_selected([10, 11], 11)
    assert not _steer_layer_selected([10, 11], 12)
    assert _steer_modes({"phase": "decode", "positions": "last_token"}) == (
        "decode",
        "last_token",
    )


class _Array:
    def __init__(self, values):
        self.values = values
        self.shape = values.shape

    def __getitem__(self, key):
        return _Array(self.values[key])

    def reshape(self, *shape):
        return _Array(self.values.reshape(*shape))

    def transpose(self, *axes):
        return _Array(self.values.transpose(*axes))

    def astype(self, dtype):
        return _Array(self.values.astype(dtype))

    def __array__(self, dtype=None):
        return self.values.astype(dtype) if dtype is not None else self.values


def test_qk_capture_keeps_qwen_head_layout_for_existing_analyzer():
    class _Projection:
        def __init__(self, width=None):
            self.width = width

        def __call__(self, x):
            values = x.values if self.width is None else x.values[..., : self.width]
            return _Array(values)

    class _Attention:
        n_heads = 2
        n_kv_heads = 1
        q_proj = _Projection()
        k_proj = _Projection(width=2)

        @staticmethod
        def rope(x, offset=0):
            return x

    class _Worker:
        _captured_states = {}

    class _MX:
        float32 = "float32"

        @staticmethod
        def eval(*arrays):
            del arrays

    state = {
        "worker": _Worker(),
        "req_id": "metal-request",
        "extra": {"output_qk": {10: [6]}},
        "mode": "last_token",
    }
    _qk_capture(
        state,
        10,
        "model.layers.10.self_attn",
        _Attention(),
        _Array(np.arange(8, dtype="float32").reshape(1, 2, 4)),
        None,
        _MX(),
    )
    entry = state["worker"]._captured_states["metal-request"][
        "model.layers.10.self_attn"
    ]
    assert entry["q"][0].shape == (4,)
    assert entry["k_all"][0].shape == (2, 2)
