"""Contract checks for the vLLM-Metal Attention Tracker capture primitives."""
import numpy as np
import pytest
import torch

from mia.backends.vllm_metal.attention_tracker import split_packed_qk


def test_split_packed_qk_preserves_last_query_and_each_request_keys():
    queries = np.arange(1 * 2 * 5 * 3, dtype=np.float32).reshape(1, 2, 5, 3)
    keys = (100 + np.arange(1 * 1 * 5 * 3, dtype=np.float32)).reshape(1, 1, 5, 3)

    captures = split_packed_qk(4, queries, keys, [0, 2, 5])

    assert len(captures) == 2
    assert [entry.layer_num for entry in captures] == [4, 4]
    assert captures[0].q.shape == (6,)
    assert captures[0].k_all.shape == (2, 3)
    assert torch.equal(captures[0].q, torch.from_numpy(queries[0, :, 1, :].reshape(-1)))
    assert torch.equal(
        captures[0].k_all,
        torch.from_numpy(keys[0, :, :2, :].transpose(1, 0, 2).reshape(2, 3)),
    )
    assert torch.equal(captures[1].q, torch.from_numpy(queries[0, :, 4, :].reshape(-1)))
    assert torch.equal(
        captures[1].k_all,
        torch.from_numpy(keys[0, :, 2:5, :].transpose(1, 0, 2).reshape(3, 3)),
    )


@pytest.mark.parametrize(
    ("queries", "keys", "cu_seqlens"),
    [
        (np.zeros((2, 2, 2, 3)), np.zeros((2, 1, 2, 3)), [0, 2]),
        (np.zeros((1, 2, 2, 3)), np.zeros((1, 1, 2, 3)), [1, 2]),
        (np.zeros((1, 2, 2, 3)), np.zeros((1, 1, 2, 3)), [0, 1, 1, 2]),
        (np.zeros((1, 2, 2, 3)), np.zeros((1, 1, 2, 3)), [0, 1]),
    ],
)
def test_split_packed_qk_rejects_invalid_shapes_or_boundaries(queries, keys, cu_seqlens):
    with pytest.raises(ValueError):
        split_packed_qk(0, queries, keys, cu_seqlens)
