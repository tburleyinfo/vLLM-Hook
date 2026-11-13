import pytest
import torch

pytest.importorskip("vllm")

from mia.backends.vllm_metal.qk_capture import requested_layers, stack_qk_capture


def test_requested_layers_accepts_public_attention_tracker_selectors():
    assert requested_layers({10: [24], 13: [20, 22]}, 40) == {10, 13}
    assert requested_layers([2, 5], 8) == {2, 5}
    assert requested_layers(True, 3) == {0, 1, 2}


def test_requested_layers_rejects_out_of_range_layer_ids():
    with pytest.raises(ValueError, match="outside"):
        requested_layers({40: [1]}, 40)


def test_stack_last_token_q_and_full_prompt_k_shapes():
    q, k = stack_qk_capture(
        [torch.ones(8), torch.full((8,), 2.0)],
        [torch.ones(4, 6), torch.full((7, 6), 2.0)],
        "last_token",
    )
    assert tuple(q.shape) == (2, 8)
    assert tuple(k.shape) == (2, 7, 6)
    assert torch.equal(k[0, :4], torch.ones(4, 6))
    assert torch.count_nonzero(k[0, 4:]) == 0


def test_stack_all_token_q_preserves_per_pass_lengths():
    q, k = stack_qk_capture(
        [torch.ones(3, 8), torch.full((1, 8), 2.0)],
        [torch.ones(3, 6), torch.full((1, 6), 2.0)],
        "all_tokens",
    )
    assert tuple(q.shape) == (2, 3, 8)
    assert tuple(k.shape) == (2, 3, 6)
    assert torch.count_nonzero(q[1, 1:]) == 0
