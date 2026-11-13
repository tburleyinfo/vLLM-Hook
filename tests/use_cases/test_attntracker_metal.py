from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from mia.backends.metal.attention_tracker import (  # noqa: E402
    AttentionTrackerCapture,
    capture_scope,
    install_vllm_metal_runner_capture,
    take_runner_capture,
)
from mia.backends.metal.prompt_ranges import chat_template_token_range  # noqa: E402
from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer  # noqa: E402


def scaled_dot_product_attention(queries, keys, values):
    return values


def test_metal_qk_capture_matches_attention_tracker_probe_contract():
    capture = AttentionTrackerCapture({5: [18], 7: [12]})
    queries = np.ones((1, 32, 1, 128), dtype=np.float16)
    keys = np.ones((1, 8, 7, 128), dtype=np.float16)
    attention = SimpleNamespace(scale=128**-0.5)

    with capture_scope(capture):
        capture.record(5, queries, keys, attention)

    probes = capture.probes()
    assert probes["config"] == {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "attention_multiplier": 128**-0.5,
    }
    entry = probes["qk_cache"]["model.layers.5.self_attn"]
    assert entry["layer_num"] == 5
    assert entry["heads"] == [18]
    assert entry["q"][0].shape == (1, 32 * 128)
    assert entry["k_all"][0].shape == (7, 8 * 128)
    assert entry["q"][0].dtype == torch.float16


def test_metal_qk_capture_rejects_missing_configured_head():
    capture = AttentionTrackerCapture({0: [8]})
    with pytest.raises(ValueError, match="Configured query head"):
        capture.record(
            0,
            np.ones((1, 4, 1, 8), dtype=np.float32),
            np.ones((1, 2, 1, 8), dtype=np.float32),
            SimpleNamespace(scale=0.5),
        )


def test_metal_capture_is_consumable_by_public_attention_tracker_analyzer(tmp_path):
    capture = AttentionTrackerCapture({0: [0]})
    capture.record(
        0,
        np.ones((1, 2, 1, 2), dtype=np.float32),
        np.ones((1, 1, 4, 2), dtype=np.float32),
        SimpleNamespace(scale=1.0),
    )
    analyzer = AttntrackerAnalyzer(str(tmp_path), {0: [0]})

    result = analyzer.analyze(
        analyzer_spec={
            "input_range": ((0, 2), (2, 4)),
            "attn_func": "sum_normalize",
        },
        probes=capture.probes(),
    )

    assert result["score"] == pytest.approx([0.5])


def test_chat_template_range_uses_character_offsets():
    class OffsetTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return (
                f"<system>{messages[0]['content']}</system>"
                f"<user>{messages[1]['content']}</user>"
            )

        def __call__(self, text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "Data: untrusted text"},
    ]
    assert chat_template_token_range(OffsetTokenizer(), messages, "untrusted text") == (
        len("<system>instructions</system><user>Data: "),
        len("<system>instructions</system><user>Data: untrusted text"),
    )


def test_runner_bridge_captures_requested_prefill_and_keeps_real_forward():
    class Attention:
        scale = 0.125

        def __call__(self):
            q = np.ones((1, 4, 1, 2), dtype=np.float32)
            k = np.ones((1, 2, 3, 2), dtype=np.float32)
            return scaled_dot_product_attention(q, k, k)

    class Model:
        def __init__(self):
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(self_attn=Attention())]
            )

        def __call__(self):
            return self.model.layers[0].self_attn()

    class Runner:
        def load_model(self):
            self.model = Model()

        def _prefill_single(self, req_id, token_ids, sampling_params):
            return self.model()

    install_vllm_metal_runner_capture(Runner)
    runner = Runner()
    runner.load_model()
    output = runner._prefill_single(
        "request-1",
        [1, 2, 3],
        SimpleNamespace(extra_args={"output_qk": {0: [1]}}),
    )

    assert output.shape == (1, 2, 3, 2)
    probes = take_runner_capture("request-1")
    assert probes["qk_cache"]["model.layers.0.self_attn"]["k_all"][0].shape == (3, 4)
