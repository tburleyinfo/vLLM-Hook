import pytest
import torch

pytest.importorskip("vllm")

from mia.backends.vllm_metal.qk_capture import MetalQKCapture
from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer


def test_capture_uses_mia_qk_probe_contract_and_analyzes():
    config = {
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "hidden_size": 4,
        "head_dim": 2,
        "attention_multiplier": 2**-0.5,
    }
    capture = MetalQKCapture(config, selected_layers={0})
    queries = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])

    with capture.requests(["request-1"]):
        capture.record(0, queries, keys)
    probes = capture.probes("request-1")

    entry = probes["qk_cache"]["model.layers.0.self_attn"]
    assert entry["q"][0].shape == (4,)
    assert entry["k_all"][0].shape == (3, 2)
    analyzer = AttntrackerAnalyzer("", {0: [0, 1]})
    result = analyzer.analyze(
        analyzer_spec={
            "input_range": [((0, 1), (1, 3))],
            "attn_func": "sum_normalize",
        },
        probes=probes,
    )
    assert len(result["score"]) == 1
    assert 0.0 <= result["score"][0] <= 1.0


def test_batched_capture_returns_individual_request_probes():
    config = {
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "hidden_size": 2,
        "head_dim": 2,
        "attention_multiplier": 2**-0.5,
    }
    capture = MetalQKCapture(config, selected_layers={0})
    queries = torch.ones((2, 1, 1, 2))
    keys = torch.ones((2, 1, 3, 2))

    with capture.requests(["request-a", "request-b"]):
        capture.record(0, queries, keys)

    request_a = capture.probes("request-a")
    request_b = capture.probes("request-b")
    assert request_a["qk_cache"]["model.layers.0.self_attn"]["q"][0].shape == (2,)
    assert request_b["qk_cache"]["model.layers.0.self_attn"]["k_all"][0].shape == (3, 2)


def test_sequential_capture_maps_each_forward_to_its_request():
    config = {
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "hidden_size": 2,
        "head_dim": 2,
        "attention_multiplier": 2**-0.5,
    }
    capture = MetalQKCapture(config)
    q = torch.ones((1, 1, 1, 2))
    k = torch.ones((1, 1, 2, 2))

    with capture.requests(["request-a", "request-b"], sequential=True):
        for _ in range(2):
            capture.record(0, q, k)

    assert capture.probes("request-a") is not None
    assert capture.probes("request-b") is not None
