import pytest
import torch

pytest.importorskip("vllm")

from mia.backends.metal.attention_tracker import MetalAttentionCapture
from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer


def test_probe_payload_preserves_mia_qk_contract(tmp_path):
    capture = MetalAttentionCapture.__new__(MetalAttentionCapture)
    capture.config = {
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "attention_multiplier": 2**-0.5,
    }
    capture._captured = {
        5: {
            "q": torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
            "k_all": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
            ),
        }
    }

    probes = capture.to_probes()
    analyzer = AttntrackerAnalyzer(str(tmp_path), {5: [0, 1]})
    result = analyzer.analyze(
        analyzer_spec={
            "input_range": [((0, 2), (2, 4))],
            "attn_func": "sum_normalize",
        },
        probes=probes,
    )

    assert probes["qk_cache"]["5"]["hookq_mode"] == "last_token"
    assert probes["qk_cache"]["5"]["q"][0].shape == (1, 4)
    assert probes["qk_cache"]["5"]["k_all"][0].shape == (4, 2)
    assert len(result["score"]) == 1
    assert torch.isfinite(torch.tensor(result["score"])).all()
