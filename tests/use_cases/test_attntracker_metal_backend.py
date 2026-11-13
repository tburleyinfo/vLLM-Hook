"""Focused tests for the vLLM-Metal Q/K adapter's MIA data contract."""
import pytest
import torch

pytest.importorskip("vllm")

from mia.backends.vllm_metal.attention_tracker import (
    _RequestCapture,
    _record_qk,
    _requested_layer,
    _request_enabled,
)


class _Worker:
    def __init__(self):
        self._captured_states = {}
        self._disk_states = {}
        self.hookq_mode = "all_tokens"


def test_record_qk_uses_existing_qk_cache_contract():
    worker = _Worker()
    request = _RequestCapture(
        "req-1",
        {"output_qk": {"10": [24]}, "hookq_mode": "last_token"},
        True,
        3,
    )
    q = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    k = torch.arange(48, dtype=torch.float32).reshape(6, 8)

    _record_qk(worker, request, 10, q, k)
    _record_qk(worker, request, 10, q, torch.cat((k, k[-1:]), dim=0))

    entry = worker._captured_states["req-1"]["model.layers.10.self_attn"]
    assert entry["layer_num"] == 10
    assert entry["hookq_mode"] == "last_token"
    assert torch.equal(entry["q"][0], q[-1:])
    assert torch.equal(entry["q"][1], q[-1:])
    assert torch.equal(entry["k_all"][0], k)
    assert torch.equal(entry["k_all"][1], k[-1:])


def test_capture_selection_preserves_layer_and_hook_phase_filters():
    request = _RequestCapture("req-2", {"output_qk": {"4": [1]}}, True, 2)

    assert _requested_layer(request.extra, 4)
    assert not _requested_layer(request.extra, 3)
    assert _request_enabled(request)
    assert not _request_enabled(
        _RequestCapture(
            "req-3", {"output_qk": True, "hooks_on": "decode"}, True, 2
        )
    )
