import pytest
import torch

from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from mia.backends.vllm_metal.qk_capture import MetalQKCaptureSession, write_qk_run


def _session():
    session = MetalQKCaptureSession(layer_to_heads={0: [0]}, prompt_count=1)
    session.config = {
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "hidden_size": 4,
        "attention_multiplier": 2**-0.5,
    }
    session.captures = {
        0: {
            0: {
                "q": torch.tensor([1.0, 0.0, 0.0, 1.0]),
                "k_all": torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
                ),
                "layer_num": 0,
                "hookq_mode": "last_token",
            }
        }
    }
    return session


def test_metal_probe_shape_matches_existing_attention_tracker_contract():
    probes = _session().to_probes()

    entry = probes["qk_cache"]["model.layers.0.self_attn"]
    assert probes["config"]["num_attention_heads"] == 2
    assert entry["q"][0].shape == (4,)
    assert entry["k_all"][0].shape == (3, 2)
    assert entry["layer_num"] == 0

    analyzer = AttntrackerAnalyzer(hook_dir="", layer_to_heads={0: [0]})
    result = analyzer.analyze(
        analyzer_spec={
            "input_range": [((0, 1), (1, 3))],
            "attn_func": "sum_normalize",
        },
        probes=probes,
    )
    assert list(result) == ["score"]
    assert len(result["score"]) == 1
    assert 0.0 <= result["score"][0] <= 1.0


def test_metal_probe_raises_when_any_configured_layer_was_not_captured():
    session = _session()
    session.layer_to_heads[1] = [0]

    with pytest.raises(RuntimeError, match="configured Attention Tracker layers"):
        session.to_probes()


def test_metal_probe_preserves_prompt_order_for_batched_analyzer_inputs():
    session = _session()
    session.prompt_count = 2
    session.captures[0][1] = {
        "q": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "k_all": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "layer_num": 0,
        "hookq_mode": "last_token",
    }

    probes = session.to_probes()
    entry = probes["qk_cache"]["model.layers.0.self_attn"]
    assert len(entry["q"]) == 2
    assert torch.equal(entry["q"][1], session.captures[0][1]["q"])


def test_metal_disk_artifact_uses_existing_analyzer_route(tmp_path):
    probes = _session().to_probes()
    write_qk_run(str(tmp_path), "metal-test", probes)

    analyzer = AttntrackerAnalyzer(
        hook_dir=str(tmp_path),
        layer_to_heads={0: [0]},
    )
    result = analyzer.analyze(
        analyzer_spec={
            "input_range": [((0, 1), (1, 3))],
            "attn_func": "sum_normalize",
        },
        run_id="metal-test",
    )
    assert len(result["score"]) == 1
