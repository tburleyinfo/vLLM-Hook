"""Shape and layout contract for the vLLM-Metal Qwen2 Q/K bridge."""
import numpy as np
import torch

from mia.backends.vllm_metal import _attention_scores, _qk_tensors


def test_qk_tensors_keep_last_query_and_flatten_kv_heads():
    q = np.arange(1 * 4 * 3 * 2, dtype=np.float32).reshape(1, 4, 3, 2)
    k = np.arange(1 * 2 * 5 * 2, dtype=np.float32).reshape(1, 2, 5, 2)

    q_tensor, k_tensor = _qk_tensors(q, k)

    assert q_tensor.shape == (1, 8)
    assert k_tensor.shape == (1, 5, 4)
    torch.testing.assert_close(q_tensor[0], torch.from_numpy(q[0, :, -1, :].reshape(-1)))
    torch.testing.assert_close(
        k_tensor[0],
        torch.from_numpy(k.transpose(0, 2, 1, 3).reshape(1, 5, 4)[0]),
    )


def test_qk_tensors_reject_non_attention_layouts():
    with np.testing.assert_raises(ValueError):
        _qk_tensors(np.zeros((1, 4, 2)), np.zeros((1, 2, 5, 2)))


def test_qk_tensors_preserve_all_query_tokens_when_requested():
    q = np.arange(1 * 4 * 3 * 2, dtype=np.float32).reshape(1, 4, 3, 2)
    k = np.zeros((1, 2, 5, 2), dtype=np.float32)

    q_tensor, _ = _qk_tensors(q, k, hookq_mode="all_tokens")

    assert q_tensor.shape == (1, 3, 8)
    torch.testing.assert_close(
        q_tensor[0],
        torch.from_numpy(q.transpose(0, 2, 1, 3).reshape(1, 3, 8)[0]),
    )


def test_attention_scores_match_selected_gqa_heads():
    q = np.ones((1, 4, 1, 2), dtype=np.float32)
    k = np.ones((1, 2, 3, 2), dtype=np.float32)

    scores = _attention_scores(q, k, [1, 3], scale=0.5)

    assert scores.shape == (2, 1, 3)
    torch.testing.assert_close(scores, torch.full((2, 1, 3), 1 / 3))
