from types import SimpleNamespace

import numpy as np
import torch

from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from mia.backends.vllm_metal import (
    _apply_steering,
    _qk_capture,
    _to_cpu_tensor,
    record_qk_rows,
)


class _FakeMetalArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def astype(self, _dtype):
        return self.value.astype(np.float32)


class _FakeMX:
    float32 = np.float32

    @staticmethod
    def eval(_value):
        return None

    @staticmethod
    def sum(value, axis=None, keepdims=False):
        return np.sum(value, axis=axis, keepdims=keepdims)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)


class _FakeAttention:
    @staticmethod
    def q_proj(value):
        return value

    @staticmethod
    def k_proj(value):
        return value

    @staticmethod
    def rope(value, offset=0):
        return value + offset


def test_metal_array_conversion_uses_float32_before_numpy():
    converted = _to_cpu_tensor(_FakeMetalArray([[1, 2]]), _FakeMX)

    assert converted.dtype == torch.float16
    assert converted.device.type == "cpu"
    assert converted.tolist() == [[1.0, 2.0]]


def test_metal_qk_capture_uses_prefill_rope_offset_and_updated_cache():
    worker = SimpleNamespace(
        _conf={
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
        },
        _captured_states={},
        _disk_states={},
        _mia_metal_active={
            "kind": "qk",
            "contexts": [
                {
                    "request_id": "request-1",
                    "layers": {0},
                    "mode": "last_token",
                    "hooks_on": "prefill",
                    "phase": "prefill",
                    "cache_offset": 0,
                    "seq_len": 2,
                    "extra": {},
                }
            ],
        },
    )
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    k = x.reshape(1, 2, 2, 2).transpose(0, 2, 1, 3)
    cache = SimpleNamespace(keys=k, offset=2)

    _qk_capture(worker, 0, _FakeAttention(), x, cache, 0, _FakeMX)

    entry = worker._captured_states["request-1"]["model.layers.0.self_attn.attn"]
    assert torch.equal(entry["q"][0], torch.from_numpy(x[0, -1]).half())
    assert torch.equal(
        entry["k_all"][0], torch.from_numpy(x[0].astype(np.float16))
    )


def test_metal_qk_capture_uses_existing_probe_contract():
    worker = SimpleNamespace(_captured_states={}, _disk_states={})
    q_rows = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    k_rows = torch.tensor([[1.0, 0.0, 0.0, 1.0],
                           [0.0, 1.0, 1.0, 0.0],
                           [1.0, 1.0, 0.0, 0.0]])

    record_qk_rows(worker, "request-1", 0, "model.layers.0.self_attn.attn",
                   q_rows, k_rows, "last_token", {})

    entry = worker._captured_states["request-1"]["model.layers.0.self_attn.attn"]
    assert len(entry["q"]) == 1
    assert torch.equal(entry["q"][0], q_rows[-1])
    assert len(entry["k_all"]) == 1
    assert torch.equal(entry["k_all"][0], k_rows)
    assert entry["layer_num"] == 0
    assert entry["hookq_mode"] == "last_token"

    analyzer = AttntrackerAnalyzer("/tmp", {0: [1]})
    result = analyzer.analyze(
        analyzer_spec={
            "input_range": [((0, 1), (1, 3))],
            "attn_func": "sum_normalize",
        },
        probes={
            "config": {
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
                "attention_multiplier": 0.5,
            },
            "qk_cache": worker._captured_states["request-1"],
        },
    )
    assert list(result) == ["score"]
    assert len(result["score"]) == 1
    assert 0.0 <= result["score"][0] <= 1.0


def test_metal_all_token_capture_analyzes_last_query_row():
    worker = SimpleNamespace(_captured_states={}, _disk_states={})
    q_rows = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    k_rows = torch.tensor([[1.0, 0.0, 0.0, 1.0],
                           [0.0, 1.0, 1.0, 0.0]])
    record_qk_rows(worker, "request-1", 0, "layer", q_rows, k_rows, "all_tokens", {})
    analyzer = AttntrackerAnalyzer("/tmp", {0: [1]})

    result = analyzer.analyze(
        analyzer_spec={
            "input_range": [((0, 1), (1, 2))],
            "attn_func": "sum_normalize",
        },
        probes={
            "config": {
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
                "attention_multiplier": 0.5,
            },
            "qk_cache": worker._captured_states["request-1"],
        },
    )

    assert len(result["score"]) == 1
    assert 0.0 <= result["score"][0] <= 1.0


def test_metal_qk_capture_keeps_disk_and_rpc_lifecycles_separate():
    worker = SimpleNamespace(_captured_states={}, _disk_states={})
    q_rows = torch.ones((2, 4))
    k_rows = torch.ones((2, 4))

    record_qk_rows(worker, "rpc", 0, "layer", q_rows, k_rows, "last_token", {})
    record_qk_rows(
        worker, "disk", 0, "layer", q_rows, k_rows, "last_token",
        {"save_to_disk": True},
    )

    assert "rpc" in worker._captured_states
    assert "disk" not in worker._captured_states
    assert "disk" in worker._disk_states
    assert "rpc" not in worker._disk_states


def test_metal_steering_respects_layer_and_position_selection():
    worker = SimpleNamespace(
        _mia_metal_active={
            "kind": "steer",
            "contexts": [
                {
                    "steer": {
                        "vector_path": "vector",
                        "method": "add_vector",
                        "coefficient": 2,
                        "optimal_layer": 0,
                        "phase": "prefill",
                        "positions": "last_token",
                    },
                    "is_prefill": True,
                }
            ],
        },
        _mia_metal_steer_vectors={
            "vector": (np.array([1.0, 0.0]), None),
        },
    )
    hidden = np.zeros((1, 2, 2), dtype=np.float32)

    steered = _apply_steering(worker, 0, hidden, None, _FakeMX)
    untouched = _apply_steering(worker, 1, hidden, None, _FakeMX)

    np.testing.assert_array_equal(steered, [[[0.0, 0.0], [2.0, 0.0]]])
    np.testing.assert_array_equal(untouched, hidden)


def test_metal_adjust_rs_steering_matches_existing_projection_rule():
    worker = SimpleNamespace(
        _mia_metal_active={
            "kind": "steer",
            "contexts": [
                {
                    "steer": {
                        "vector_path": "vector",
                        "method": "adjust_rs",
                        "optimal_layer": 0,
                        "phase": "both",
                        "positions": "all_tokens",
                    },
                    "is_prefill": False,
                }
            ],
        },
        _mia_metal_steer_vectors={
            "vector": (np.array([1.0, 0.0]), np.array(0.5)),
        },
    )
    hidden = np.array([[[1.0, 2.0]]], dtype=np.float32)

    steered = _apply_steering(worker, 0, hidden, None, _FakeMX)

    np.testing.assert_array_equal(steered, [[[0.5, 2.0]]])


def test_metal_invalid_steering_modes_warn_and_leave_output_unchanged(capsys):
    worker = SimpleNamespace(
        _mia_metal_active={
            "kind": "steer",
            "contexts": [
                {
                    "steer": {
                        "optimal_layer": 0,
                        "phase": "invalid",
                    },
                    "is_prefill": True,
                }
            ],
        },
        _mia_metal_steer_vectors={},
    )
    hidden = np.ones((1, 1, 2), dtype=np.float32)

    steered = _apply_steering(worker, 0, hidden, None, _FakeMX)

    np.testing.assert_array_equal(steered, hidden)
    assert worker._bad_cfg_warned == 1
    assert "ignoring request with invalid steer config" in capsys.readouterr().out
