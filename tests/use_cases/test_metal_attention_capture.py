import pytest
import torch

pytest.importorskip("mlx.core")
pytest.importorskip("vllm_metal")

from mia.backends.metal.attention import MetalQKCapture, capture_vllm_metal_attention
from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer


def test_metal_qk_probe_contract_and_analyzer(tmp_path):
    import mlx.core as mx

    capture = MetalQKCapture(
        {
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "attention_multiplier": 0.5,
        }
    )

    class Context:
        cu_seqlens = [0, 3]
        num_decode_requests = 0

    queries = mx.array([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]])
    keys = mx.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]])

    with capture_vllm_metal_attention(capture):
        capture.record(0, queries, keys, Context())

    probes = capture.probes()
    analyzer = AttntrackerAnalyzer(str(tmp_path), {0: [0]})
    result = analyzer.analyze(
        analyzer_spec={"input_range": ((0, 1), (1, 3)), "attn_func": "sum_normalize"},
        probes=probes,
    )

    assert probes["qk_cache"]["layer.0"]["q"][0].shape == (1, 4)
    assert probes["qk_cache"]["layer.0"]["k_all"][0].shape == (3, 2)
    assert 0.0 < result["score"][0] < 1.0


def test_interceptor_restores_target_callables():
    import vllm_metal.attention.impls.sdpa as sdpa
    import vllm_metal.attention.impls.sdpa_wrapper as sdpa_wrapper

    capture = MetalQKCapture({})
    prepare = sdpa.prepare_sdpa_qkv
    call = sdpa_wrapper.SDPAPagedAttentionWrapper.__call__

    with pytest.raises(RuntimeError, match="sentinel"):
        with capture_vllm_metal_attention(capture):
            assert sdpa.prepare_sdpa_qkv is not prepare
            assert sdpa_wrapper.SDPAPagedAttentionWrapper.__call__ is not call
            raise RuntimeError("sentinel")

    assert sdpa.prepare_sdpa_qkv is prepare
    assert sdpa_wrapper.SDPAPagedAttentionWrapper.__call__ is call
