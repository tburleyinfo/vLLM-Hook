import math

import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("zstandard")

from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from mia.core.metal.attntracker import build_qk_payload


def test_metal_qk_payload_preserves_attention_tracker_contract(tmp_path):
    config = {
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "attention_multiplier": 1.0,
    }
    query = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    probes = build_qk_payload(
        config,
        {0: {"q": [query], "k_all": [keys], "hookq_mode": "last_token"}},
        {0: "model.layers.0.self_attn"},
    )

    assert probes["config"] == config
    entry = probes["qk_cache"]["model.layers.0.self_attn"]
    assert entry["layer_num"] == 0
    assert entry["hookq_mode"] == "last_token"
    assert entry["q"].shape == (1, 2)
    assert entry["k_all"].shape == (1, 2, 2)

    analyzer = AttntrackerAnalyzer(str(tmp_path), {0: [0]})
    result = analyzer.analyze(
        analyzer_spec={
            "input_range": ((0, 1), (1, 2)),
            "attn_func": "sum_normalize",
        },
        probes=probes,
    )
    assert len(result["score"]) == 1
    assert math.isclose(result["score"][0], 1 / (1 + math.exp(-1)), rel_tol=1e-5)


def test_metal_payload_keeps_each_layer_and_pass():
    payload = build_qk_payload(
        {
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "attention_multiplier": 1.0,
        },
        {
            1: {
                "q": [torch.tensor([[2.0]]), torch.tensor([[3.0]])],
                "k_all": [torch.tensor([[1.0]]), torch.tensor([[1.0], [2.0]])],
                "hookq_mode": "last_token",
            }
        },
        {1: "model.layers.1.self_attn"},
    )

    entry = payload["qk_cache"]["model.layers.1.self_attn"]
    assert torch.equal(entry["q"], torch.tensor([[[2.0]], [[3.0]]]))
    assert isinstance(entry["k_all"], list)
    assert torch.equal(entry["k_all"][0], torch.tensor([[1.0]]))
    assert torch.equal(entry["k_all"][1], torch.tensor([[1.0], [2.0]]))
