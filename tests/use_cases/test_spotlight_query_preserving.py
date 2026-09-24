import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


if "vllm" not in sys.modules:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.SamplingParams = object
    sys.modules["vllm"] = vllm_stub

UTILS_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_hook_plugins"
    / "vllm_hook_plugins"
    / "utils"
    / "spotlight"
    / "utils.py"
)
spec = spec_from_file_location("spotlight_utils_under_test", UTILS_PATH)
spotlight_utils = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(spotlight_utils)

compute_query_preserving_spotlight_bias = (
    spotlight_utils.compute_query_preserving_spotlight_bias
)
compute_spotlight_bias = spotlight_utils.compute_spotlight_bias


def _canonical_reference(logits, span_ranges, target_proportion):
    attn_weights = F.softmax(logits, dim=-1)
    modified_weights = attn_weights.clone()

    for batch_idx, ranges in enumerate(span_ranges):
        if not ranges:
            continue

        union_mask = torch.zeros(
            modified_weights.size(-1),
            device=modified_weights.device,
            dtype=modified_weights.dtype,
        )
        for start, end in ranges:
            union_mask[start:end] = 1.0
        union_mask = union_mask.view(1, 1, -1)

        current_proportion = (
            modified_weights[batch_idx] * union_mask
        ).sum() / modified_weights[batch_idx].sum()

        if current_proportion < target_proportion:
            bias_value = torch.log(
                torch.tensor(
                    target_proportion / current_proportion,
                    device=modified_weights.device,
                    dtype=torch.float32,
                )
            )
            bias_mask = union_mask * bias_value
            attn_logits = logits[batch_idx].float()
            attn_logits = attn_logits + bias_mask
            modified_weights[batch_idx] = F.softmax(
                attn_logits, dim=-1, dtype=torch.float32
            ).to(modified_weights.dtype)

    return modified_weights


def _causal_logits(heads=2, q_len=4):
    logits = torch.full((1, heads, q_len, q_len), -0.2, dtype=torch.float32)
    mask = torch.triu(torch.full((q_len, q_len), float("-inf")), diagonal=1)
    return logits + mask.view(1, 1, q_len, q_len)


def test_canonical_spotlight_matches_reference_copy():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5, 5)
    span_ranges = [[(1, 3)], [(0, 1), (4, 5)]]

    actual = compute_spotlight_bias(logits, span_ranges, target_proportion=0.4)
    expected = _canonical_reference(logits, span_ranges, target_proportion=0.4)

    assert torch.equal(actual, expected)


def test_query_preserving_returns_same_external_attention_shape_and_normalization():
    logits = _causal_logits(heads=3, q_len=5)

    modified, diagnostics = compute_query_preserving_spotlight_bias(
        logits,
        [[(1, 3)]],
        target_proportion=0.7,
        return_diagnostics=True,
    )

    assert modified.shape == logits.shape
    assert diagnostics[0]["psi_current"].shape == (5,)
    assert diagnostics[0]["bias_value"].shape == (5,)
    assert diagnostics[0]["should_steer"].shape == (5,)
    assert diagnostics[0]["causally_reachable"].shape == (5,)
    assert torch.allclose(
        modified.sum(dim=-1),
        torch.ones_like(modified.sum(dim=-1)),
        atol=1e-6,
    )


def test_query_preserving_skips_causally_unreachable_highlighted_span():
    logits = _causal_logits(heads=2, q_len=4)

    modified, diagnostics = compute_query_preserving_spotlight_bias(
        logits,
        [[(2, 4)]],
        target_proportion=0.8,
        return_diagnostics=True,
    )

    diag = diagnostics[0]
    assert diag["psi_current"][0].item() == pytest.approx(0.0)
    assert diag["causally_reachable"][0].item() is False
    assert diag["should_steer"][0].item() is False
    assert diag["bias_value"][0].item() == pytest.approx(0.0)
    assert torch.equal(modified[:, :, 0, :], F.softmax(logits, dim=-1)[:, :, 0, :])
    assert torch.any(diag["should_steer"][2:])


def test_query_preserving_broadcasts_bias_across_heads_but_not_queries():
    logits = _causal_logits(heads=2, q_len=4)

    _modified, diagnostics = compute_query_preserving_spotlight_bias(
        logits,
        [[(1, 2)]],
        target_proportion=0.9,
        return_diagnostics=True,
    )

    bias = diagnostics[0]["bias_value"]
    assert bias.shape == (4,)
    assert bias[0].item() == pytest.approx(0.0)
    assert bias[1].item() > 0
    assert not torch.allclose(bias[1], bias[3])
