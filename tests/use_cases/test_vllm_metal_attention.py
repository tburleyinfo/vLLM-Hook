import pytest
import torch

from mia.backends.vllm_metal_attention import AttentionCapture


def test_attention_capture_keeps_probe_tensor_contract():
    capture = AttentionCapture(prompt_count=2)
    queries = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    keys = torch.arange(2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4)

    capture.record(4, queries, keys, 0.5)
    probes = capture.probes({4: "model.layers.4.self_attn.attn"})

    entry = probes["qk_cache"]["model.layers.4.self_attn.attn"]
    assert entry["layer_num"] == 4
    assert entry["hookq_mode"] == "last_token"
    assert len(entry["q"]) == len(entry["k_all"]) == 2
    assert entry["q"][0].shape == (8,)
    torch.testing.assert_close(entry["q"][0], queries[0, :, -1, :].reshape(-1))
    assert entry["k_all"][0].shape == (3, 4)
    assert probes["config"] == {
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "attention_multiplier": 0.5,
    }


def test_attention_capture_refuses_incomplete_prompt_layer_set():
    capture = AttentionCapture(prompt_count=2)
    capture.record(0, torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4), 0.5)

    with pytest.raises(RuntimeError, match="full prefill"):
        capture.probes({0: "model.layers.0.self_attn.attn"})


def test_attention_capture_requires_sdpa_qk_shape():
    capture = AttentionCapture(prompt_count=1)

    with pytest.raises(ValueError, match="batch, heads, seq, dim"):
        capture.record(0, torch.ones(2, 4), torch.ones(2, 4), 0.5)
