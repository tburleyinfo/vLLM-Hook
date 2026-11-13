import importlib
import sys
import types
import uuid

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from mia.backends import vllm_metal
from mia.backends.vllm_metal import MetalQKCapture
from mia.errors import MiaDeliveryError


def test_qwen2_wrapper_records_qk_and_preserves_original_call(monkeypatch):
    module_name = f"mia_test_{uuid.uuid4().hex}.qwen2"
    module = types.ModuleType(module_name)
    sentinel = object()
    module.scaled_dot_product_attention = lambda *args, **kwargs: sentinel
    monkeypatch.setitem(sys.modules, module_name, module)

    class Attention:
        def __call__(self, queries, keys):
            return importlib.import_module(module_name).scaled_dot_product_attention(
                queries, keys, None, None, 0.5, None
            )

    Attention.__module__ = module_name
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            layers=[types.SimpleNamespace(self_attn=Attention())]
        )
    )
    vllm_metal._install_qwen2_capture(model)
    queries = np.ones((1, 2, 3, 2), dtype=np.float32)
    keys = np.ones((1, 1, 3, 2), dtype=np.float32)

    with vllm_metal.capture_generation({0: [0]}, "last_token") as capture:
        assert model.model.layers[0].self_attn(queries, keys) is sentinel
    probes = capture.probes(1)[0]
    assert probes["qk_cache"]["model.layers.0.self_attn.attn"]["k_all"][0].shape == (3, 2)


def test_metal_qk_capture_matches_attention_tracker_probe_contract(tmp_path):
    capture = MetalQKCapture({1: [0], 2: [1]}, "last_token")
    queries = np.arange(1 * 2 * 3 * 2, dtype=np.float32).reshape(1, 2, 3, 2)
    keys = np.arange(1 * 1 * 3 * 2, dtype=np.float32).reshape(1, 1, 3, 2)

    capture.begin_layer(0)
    capture.begin_layer(1)
    capture.record(queries, keys, 0.5)
    capture.begin_layer(2)
    capture.record(queries, keys, 0.5)
    (probes,) = capture.probes(1)

    assert probes["config"] == {
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "attention_multiplier": 0.5,
    }
    entry = probes["qk_cache"]["model.layers.1.self_attn.attn"]
    assert entry["layer_num"] == 1
    assert entry["q"][0].shape == (1, 4)
    assert entry["k_all"][0].shape == (3, 2)
    assert entry["q"][0].dtype == torch.float32

    analyzer = AttntrackerAnalyzer(str(tmp_path), {1: [0], 2: [1]})
    result = analyzer.analyze(
        analyzer_spec={
            "input_range": [((0, 1), (1, 3))],
            "attn_func": "sum_normalize",
        },
        probes=probes,
    )
    assert len(result["score"]) == 1
    assert 0.0 <= result["score"][0] <= 1.0


def test_metal_qk_capture_refuses_missing_prefill_layers():
    capture = MetalQKCapture({1: [0], 2: [0]}, "last_token")
    queries = np.ones((1, 2, 3, 2), dtype=np.float32)
    keys = np.ones((1, 1, 3, 2), dtype=np.float32)
    capture.begin_layer(0)
    capture.begin_layer(1)
    capture.record(queries, keys, 0.5)
    with pytest.raises(MiaDeliveryError, match="missed configured layers"):
        capture.probes(1)
