from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mia.core._plugin import resolve_capture_mode
from mia.core.metal.qk_capture import (
    MetalQKCapture,
    _Request,
    _cached_keys,
    _flat_heads,
    _layer_is_selected,
)
from mia.errors import MiaConfigurationError


class _Rope:
    def __call__(self, value, **kwargs):
        return value


class _Attention:
    n_heads = 2
    n_kv_heads = 1

    def __init__(self):
        self.rope = _Rope()

    def __call__(self, x, mask=None, cache=None):
        query = np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)
        key = np.arange(6, dtype=np.float32).reshape(1, 1, 3, 2)
        self.rope(query)
        self.rope(key)
        cache.keys = key
        cache.offset = 3
        return x


def test_capture_records_rope_qk_and_preserves_attention_result():
    attention = _Attention()
    runner = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]),
        model_args={
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "hidden_size": 4,
            "head_dim": 2,
        },
        _paged_attention_runtime=None,
        _prefill_single=lambda *args, **kwargs: None,
    )
    worker = SimpleNamespace(model_runner=runner)
    backend = MetalQKCapture(worker)
    backend.install()

    request = backend._request.set(
        _Request(
            "request-1",
            {"output_qk": {0: [1]}, "hookq_mode": "last_token", "hooks_on": "prefill"},
        )
    )
    try:
        cache = SimpleNamespace(keys=None, offset=0)
        tokens = np.zeros((1, 3, 4), dtype=np.float32)
        output = runner.model.layers[0].self_attn(tokens, cache=cache)
    finally:
        backend._request.reset(request)

    assert output is tokens
    entry = worker._captured_states["request-1"]["layers.0.self_attn"]
    assert entry["q"][0].shape == (4,)
    assert entry["k_all"][0].shape == (3, 2)
    assert entry["k_prefix_ends"] == [3]


def test_mlx_head_rows_are_flattened_in_token_major_order():
    value = np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)
    rows = _flat_heads(value)
    assert torch.equal(rows, torch.from_numpy(value[0].transpose(1, 0, 2).reshape(3, 4)))


def test_cache_key_capture_respects_current_offset():
    keys = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    cache = SimpleNamespace(keys=keys, offset=2)
    assert _cached_keys(cache, keys).shape == (1, 1, 2, 4)


def test_layer_selection_supports_existing_output_qk_shapes():
    assert _layer_is_selected({4: [2], "7": [1]}, 7)
    assert _layer_is_selected([3, 5], 5)
    assert not _layer_is_selected({4: [2]}, 3)


def test_metal_engine_uses_eager_mia_hooks_without_changing_cpu_cuda_default():
    metal = resolve_capture_mode(
        SimpleNamespace(enforce_eager=False, enable_prefix_caching=None),
        {"VLLM_PLUGINS": "cuda, metal"},
        kind="qk",
    )
    default = resolve_capture_mode(
        SimpleNamespace(enforce_eager=False, enable_prefix_caching=None),
        {},
        kind="qk",
    )
    assert not metal.graph and metal.engine_eager
    assert metal.prefix_caching_off
    assert default.graph and not default.engine_eager


def test_metal_rejects_explicit_cuda_graph_capture():
    with pytest.raises(MiaConfigurationError, match="cannot run on vLLM-Metal"):
        resolve_capture_mode(
            SimpleNamespace(enforce_eager=False, enable_prefix_caching=None),
            {"VLLM_PLUGINS": "metal", "MIA_ALLOW_CUDAGRAPH": "1"},
            kind="qk",
        )
