from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mia.backends.metal.qk_capture import _CaptureRequest, format_qk_capture
from mia.core._plugin import validate_v2_runner_selected
from mia.errors import MiaConfigurationError


def test_last_token_capture_flattens_query_heads_and_preserves_all_keys():
    queries = np.arange(1 * 4 * 3 * 2, dtype=np.float32).reshape(1, 4, 3, 2)
    keys = np.arange(1 * 2 * 3 * 2, dtype=np.float32).reshape(1, 2, 3, 2)

    q, k = format_qk_capture(queries, keys, batch_index=0, mode="last_token")

    assert q.shape == (8,)
    assert torch.equal(q, torch.from_numpy(queries[0, :, -1, :].reshape(-1)))
    assert k.shape == (3, 4)
    assert torch.equal(k, torch.from_numpy(keys[0].transpose(1, 0, 2).reshape(3, 4)))


def test_all_token_capture_preserves_query_sequence_and_gqa_key_width():
    queries = np.arange(1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2)
    keys = np.arange(1 * 2 * 2 * 2, dtype=np.float32).reshape(1, 2, 2, 2)

    q, k = format_qk_capture(queries, keys, batch_index=0, mode="all_tokens")

    assert q.shape == (2, 8)
    assert k.shape == (2, 4)


def test_capture_preserves_numpy_float16_dtype():
    queries = np.ones((1, 2, 1, 2), dtype=np.float16)
    keys = np.ones((1, 2, 1, 2), dtype=np.float16)

    q, k = format_qk_capture(queries, keys, batch_index=0, mode="last_token")

    assert q.dtype == torch.float16
    assert k.dtype == torch.float16


def test_capture_request_stores_existing_qk_probe_entry_contract():
    worker = SimpleNamespace(
        _captured_states={},
        _disk_states={},
        _artifact_tag=None,
        _artifact_gran=None,
        _artifact_gsize=128,
    )
    request = _CaptureRequest(
        worker,
        "request-1",
        {"output_qk": {3: [0]}, "hookq_mode": "last_token"},
        batch_index=0,
        phase="prefill",
    )
    queries = np.ones((1, 4, 2, 2), dtype=np.float32)
    keys = np.ones((1, 2, 2, 2), dtype=np.float32)

    request.record(3, "model.layers.3.self_attn.attn", queries, keys)

    entry = worker._captured_states["request-1"]["model.layers.3.self_attn.attn"]
    assert entry["layer_num"] == 3
    assert entry["hookq_mode"] == "last_token"
    assert entry["q"][0].shape == (8,)
    assert entry["k_prefix_ends"] == [2]
    assert entry["k_all"][0].shape == (2, 4)


def test_vllm_metal_qk_runner_is_accepted_when_v2_is_unavailable():
    config = SimpleNamespace(
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(
            worker_cls="vllm_metal.v1.worker.MetalWorker",
            worker_extension_cls="mia.workers.qk_capture_worker.QKCaptureWorker",
        ),
    )

    validate_v2_runner_selected(config)


def test_non_metal_v1_runner_remains_rejected():
    config = SimpleNamespace(
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(
            worker_cls="vllm.worker.worker.Worker",
            worker_extension_cls="mia.workers.qk_capture_worker.QKCaptureWorker",
        ),
    )

    with pytest.raises(RuntimeError, match="requires vLLM's V2 model runner"):
        validate_v2_runner_selected(config)


@pytest.mark.parametrize(
    ("queries", "keys", "batch_index", "mode", "error"),
    [
        (np.zeros((2, 1, 1)), np.zeros((2, 1, 1)), 0, "last_token", ValueError),
        (np.zeros((1, 1, 1, 2)), np.zeros((1, 1, 1, 2)), 1, "last_token", IndexError),
        (
            np.zeros((1, 1, 1, 2)),
            np.zeros((1, 1, 1, 2)),
            0,
            "unknown",
            MiaConfigurationError,
        ),
    ],
)
def test_capture_rejects_invalid_shapes_indices_and_modes(
    queries, keys, batch_index, mode, error
):
    with pytest.raises(error):
        format_qk_capture(queries, keys, batch_index, mode)
