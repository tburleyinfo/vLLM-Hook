import pickle
import sys
import types

import numpy as np
import pytest
import zstandard as zstd

from mia.backends.metal_qk import MetalQKCapture


class _Attention:
    def __init__(self):
        self.rope = lambda value: value


class _Layer:
    def __init__(self):
        self.self_attn = _Attention()


class _Args:
    num_attention_heads = 2
    num_key_value_heads = 1
    hidden_size = 4
    attention_multiplier = 0.5


class _Model:
    layers = [_Layer()]
    args = _Args()


class _Runner:
    _paged_attention_runtime = None

    def __init__(self):
        self.model = _Model()

    def _prefill_single(self, *args, **kwargs):
        return None

    def _sequential_decode(self, *args, **kwargs):
        return None

    def _batched_decode(self, *args, **kwargs):
        return None


def _mlx_stub(monkeypatch):
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.float32 = np.float32
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def test_granite_capture_emits_existing_qk_probe_contract(monkeypatch):
    _mlx_stub(monkeypatch)
    capture = MetalQKCapture(_Runner(), worker=None)
    params = types.SimpleNamespace(
        extra_args={"output_qk": True, "hookq_mode": "last_token"}
    )

    with capture._request_scope(["req-1"], [params], is_prefill=True):
        query = np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)
        key = np.arange(6, dtype=np.float32).reshape(1, 1, 3, 2)
        capture._observe_rope(0, query)
        capture._observe_rope(0, key)

    payload = pickle.loads(
        zstd.ZstdDecompressor().decompress(capture.get_captured_states("req-1"))
    )
    entry = payload["qk_cache"]["model.layers.0.self_attn"]
    assert payload["config"] == {
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "hidden_size": 4,
        "head_dim": 2,
        "attention_multiplier": 0.5,
    }
    assert entry["q"].shape == (1, 4)
    assert entry["k_all"][0].shape == (3, 2)
    assert entry["layer_num"] == 0
    assert capture.get_captured_states("req-1") is None
    capture.close()


def test_paged_attention_is_rejected_without_changing_runner_behavior(monkeypatch):
    _mlx_stub(monkeypatch)

    class PagedRunner(_Runner):
        _paged_attention_runtime = object()

    with pytest.raises(RuntimeError, match="VLLM_METAL_USE_PAGED_ATTENTION=0"):
        MetalQKCapture(PagedRunner(), worker=None)
