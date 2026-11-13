"""Contract checks for the MLX-side Q/K capture adapter."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("mlx")

import mlx.core as mx

from mia.backends.metal.qk_capture import (
    _RequestCapture,
    _capture_requested,
    _mlx_float32_numpy,
    _store_qk,
)


def test_capture_respects_worker_request_and_layer_contract():
    prefill = _RequestCapture(
        req_id="request-1",
        extra={"output_qk": {3: [4]}, "hooks_on": "prefill"},
        phase="prefill",
        token_count=12,
    )
    assert _capture_requested(prefill, 3)
    assert not _capture_requested(prefill, 2)

    decode_only = _RequestCapture(
        req_id="request-2",
        extra={"output_qk": True, "hooks_on": "decode"},
        phase="prefill",
        token_count=8,
    )
    assert not _capture_requested(decode_only, 3)

    decode = _RequestCapture(
        req_id="request-2",
        extra={"output_qk": True, "hooks_on": "decode"},
        phase="decode",
        token_count=9,
    )
    assert _capture_requested(decode, 3)


def test_last_token_payload_keeps_the_existing_vector_shape():
    class NumpyMLX:
        float32 = np.float32

        @staticmethod
        def eval(_array):
            return None

        @staticmethod
        def contiguous(array):
            return np.ascontiguousarray(array)

    worker = SimpleNamespace(_captured_states={}, _disk_states={})
    context = _RequestCapture(
        req_id="request-1",
        extra={"output_qk": {2: [4]}, "hookq_mode": "last_token"},
        phase="prefill",
        token_count=3,
    )
    q = np.arange(12, dtype=np.float32).reshape(3, 4)
    k = np.arange(24, dtype=np.float32).reshape(3, 8)

    _store_qk(worker, 2, q, k, context, NumpyMLX)
    entry = worker._captured_states["request-1"]["model.layers.2.self_attn.attn"]
    assert entry["q"][0].shape == (4,)
    assert entry["k_all"][0].shape == (3, 8)


def test_score_capture_keeps_the_existing_per_head_score_contract():
    class NumpyMLX:
        float32 = np.float32

        @staticmethod
        def eval(_array):
            return None

        @staticmethod
        def contiguous(array):
            return np.ascontiguousarray(array)

    worker = SimpleNamespace(
        _captured_states={},
        _disk_states={},
        _conf={
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "attention_multiplier": 2**-0.5,
        },
        _score_head_default=0,
        _score_dtype=torch.float16,
    )
    context = _RequestCapture(
        req_id="request-1",
        extra={
            "output_qk": {2: [1]},
            "hookq_mode": "last_token",
            "qk_capture": "score",
        },
        phase="prefill",
        token_count=3,
    )
    q = np.arange(4, dtype=np.float32).reshape(1, 4)
    k = np.arange(12, dtype=np.float32).reshape(3, 4)

    _store_qk(worker, 2, q, k, context, NumpyMLX)
    entry = worker._captured_states["request-1"]["model.layers.2.self_attn.attn"]
    assert entry["capture"] == "score"
    assert entry["heads"] == [1]
    assert entry["scores"][0].shape == (1, 1, 3)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_mlx_array_export_uses_numpy_compatible_dtype(dtype):
    values = mx.array([[1.0, 2.0]], dtype=dtype)
    exported = _mlx_float32_numpy(values, mx)
    assert exported.dtype.name == "float32"
    assert exported.tolist() == [[1.0, 2.0]]
