import numpy as np
import pytest
import torch

from mia.backends.metal_qk import (
    MetalQKCapture,
    capture_session,
    record_active_qk,
)


def test_capture_produces_existing_qk_probe_contract():
    capture = MetalQKCapture(
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        attention_multiplier=2**-0.5,
    )
    q = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    k = np.arange(12, dtype=np.float32).reshape(1, 1, 6, 2)

    with capture_session(capture):
        assert record_active_qk(3, q, k)
    probes = capture.to_probes()

    entry = probes["qk_cache"]["model.layers.3.self_attn.attn"]
    assert entry["hookq_mode"] == "last_token"
    assert entry["q"][0].shape == (4,)
    assert entry["k_all"][0].shape == (6, 2)
    assert torch.equal(entry["q"][0], torch.tensor([2.0, 3.0, 6.0, 7.0]))
    assert probes["config"]["num_attention_heads"] == 2
    assert probes["config"]["num_key_value_heads"] == 1
    assert not record_active_qk(3, q, k)


def test_capture_keeps_first_prefix_for_each_layer():
    capture = MetalQKCapture(1, 1, 2, 2**-0.5)
    first_q = np.ones((1, 1, 2, 2), dtype=np.float32)
    first_k = np.ones((1, 1, 2, 2), dtype=np.float32)
    later_k = np.full((1, 1, 3, 2), 9, dtype=np.float32)

    capture.record(0, first_q, first_k)
    capture.record(0, first_q, later_k)

    entry = capture.to_probes()["qk_cache"]["model.layers.0.self_attn.attn"]
    assert entry["k_all"][0].shape == (2, 2)
    assert torch.equal(entry["k_all"][0], torch.ones((2, 2)))


@pytest.mark.parametrize(
    "q,k",
    [
        (np.zeros((2, 2, 1, 2)), np.zeros((2, 1, 1, 2))),
        (np.zeros((1, 3, 1, 2)), np.zeros((1, 1, 1, 2))),
        (np.zeros((1, 2, 1, 3)), np.zeros((1, 1, 1, 2))),
        (np.zeros((1, 2, 1, 2)), np.zeros((1, 1, 1, 3))),
    ],
)
def test_capture_rejects_mismatched_target_arrays(q, k):
    capture = MetalQKCapture(2, 1, 2, 2**-0.5)
    with pytest.raises(ValueError):
        capture.record(0, q, k)
