import pickle

import numpy as np
import pytest
import torch
import zstandard

pytest.importorskip("vllm")

from mia.backends.metal.qk_capture import (
    _array_to_tensor,
    _capture_spec,
    clear_captured_states,
    get_captured_states,
)
from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from mia.errors import MiaConfigurationError


def test_capture_spec_respects_layer_selection():
    assert _capture_spec({"output_qk": {"10": [6]}}, 10)
    assert not _capture_spec({"output_qk": {"10": [6]}}, 9)
    assert _capture_spec({"output_qk": [10, 11]}, 11)
    assert _capture_spec({"output_qk": True}, 0)
    assert not _capture_spec({}, 0)
    with pytest.raises(MiaConfigurationError, match="raw Q/K"):
        _capture_spec({"output_qk": True, "qk_capture": "score"}, 0)


def test_array_conversion_preserves_float_values_on_cpu():
    values = np.array([[1.25, -2.0]], dtype=np.float16)
    tensor = _array_to_tensor(values)
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, torch.tensor([[1.25, -2.0]]))


def test_captured_states_use_existing_compressed_probe_contract():
    runner = type("Runner", (), {})()
    runner._mia_metal_qk_states = {
        "config": {
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "hidden_size": 4,
            "head_dim": 2,
            "attention_multiplier": 2**-0.5,
        },
        "memory": {
            "request-12345678": {
                "model.layers.10.self_attn.attn": {
                    "q": [torch.tensor([1.0, 0.0, 0.5, 0.5])],
                    "k_all": [torch.tensor([[1.0, 0.0], [0.0, 1.0]])],
                    "layer_num": 10,
                    "hookq_mode": "last_token",
                }
            }
        },
        "disk": {},
    }

    encoded = get_captured_states(runner, "request")
    assert encoded is not None
    payload = pickle.loads(zstandard.ZstdDecompressor().decompress(encoded))
    entry = payload["qk_cache"]["model.layers.10.self_attn.attn"]
    assert payload["config"]["num_attention_heads"] == 2
    assert entry["layer_num"] == 10
    assert entry["hookq_mode"] == "last_token"
    assert len(entry["q"]) == len(entry["k_all"]) == 1
    result = AttntrackerAnalyzer("", {10: [0, 1]}).analyze(
        analyzer_spec={
            "input_range": ((0, 1), (1, 2)),
            "attn_func": "sum_normalize",
        },
        probes=payload,
    )
    assert len(result["score"]) == 1
    assert 0.0 <= result["score"][0] <= 1.0
    assert get_captured_states(runner, "request") is None


def test_clear_captured_states_cleans_memory_and_disk():
    runner = type("Runner", (), {})()
    runner._mia_metal_qk_states = {
        "config": {},
        "memory": {"request-12345678": {}},
        "disk": {"request-12345678": {}},
    }
    runner._mia_metal_disk_requests = {"request-12345678": True}

    clear_captured_states(runner, "request")

    assert runner._mia_metal_qk_states["memory"] == {}
    assert runner._mia_metal_qk_states["disk"] == {}
    assert runner._mia_metal_disk_requests == {}
