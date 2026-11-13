"""Contract tests for the MLX attention observer used by vLLM-Metal."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from mia.core.hooks.metal_qk import install_metal_qk_hooks, uninstall_metal_qk_hooks
from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer


def scaled_dot_product_attention(query, key):
    return query


class FakeAttention:
    def __call__(self, query, key):
        return scaled_dot_product_attention(query, key)


class FakeLayer:
    def __init__(self):
        self.self_attn = FakeAttention()


class FakeModel:
    def __init__(self):
        self.layers = [FakeLayer(), FakeLayer()]


class FakeRunner:
    def __init__(self):
        self.model = FakeModel()
        self.model_args = {
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "hidden_size": 4,
        }

    def _prefill_single(self, req_id, token_ids, sampling_params):
        query = np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)
        key = np.arange(6, dtype=np.float32).reshape(1, 1, 3, 2)
        results = [layer.self_attn(query, key) for layer in self.model.layers]
        return req_id, results

    def _batched_decode(self, decode_reqs):
        return decode_reqs


def test_metal_qk_capture_keeps_attention_output_and_probe_contract():
    worker = SimpleNamespace(
        model_runner=FakeRunner(),
        _hooks_installed=False,
        _default_hooks_on="prefill",
    )
    params = SimpleNamespace(extra_args={
        "output_qk": {"0": [0]},
        "hookq_mode": "last_token",
    })
    runner = worker.model_runner
    original_prefill = runner._prefill_single
    original_sdpa = scaled_dot_product_attention

    install_metal_qk_hooks(worker)
    result = runner._prefill_single("request-1", [1, 2, 3], params)

    assert result[0] == "request-1"
    assert result[1][0].shape == (1, 2, 3, 2)
    assert worker._captured_states["request-1"]["model.layers.0.self_attn.attn"][
        "q"
    ][0].shape == (1, 4)
    expected_query = np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)[0]
    expected_q_values = expected_query.transpose(1, 0, 2).reshape(3, 4)
    expected_q = torch.from_numpy(expected_q_values[-1:].copy())
    assert torch.equal(
        worker._captured_states["request-1"]["model.layers.0.self_attn.attn"][
            "q"
        ][0],
        expected_q,
    )
    assert worker._captured_states["request-1"]["model.layers.0.self_attn.attn"][
        "k_all"
    ][0].shape == (3, 2)
    assert "model.layers.1.self_attn.attn" not in worker._captured_states["request-1"]
    assert worker._conf["attention_multiplier"] == pytest.approx(2 ** -0.5)

    entry = worker._captured_states["request-1"]["model.layers.0.self_attn.attn"]
    stats = AttntrackerAnalyzer("", {0: [0]}).analyze(
        analyzer_spec={
            "input_range": ((0, 1), (1, 3)),
            "attn_func": "sum_normalize",
        },
        probes={
            "config": worker._conf,
            "qk_cache": {
                "model.layers.0.self_attn.attn": {
                    "q": entry["q"],
                    "k_all": entry["k_all"],
                    "layer_num": 0,
                }
            },
        },
    )
    assert len(stats["score"]) == 1
    assert np.isfinite(stats["score"][0])

    uninstall_metal_qk_hooks(worker)
    assert runner._prefill_single == original_prefill
    assert scaled_dot_product_attention is original_sdpa


def test_metal_qk_capture_is_inactive_without_output_request():
    worker = SimpleNamespace(
        model_runner=FakeRunner(),
        _hooks_installed=False,
        _default_hooks_on="prefill",
    )
    params = SimpleNamespace(extra_args={})
    runner = worker.model_runner

    install_metal_qk_hooks(worker)
    runner._prefill_single("request-2", [1, 2, 3], params)

    assert worker._captured_states == {}
    uninstall_metal_qk_hooks(worker)
