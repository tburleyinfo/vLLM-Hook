"""Query/head-preserving Spotlight tensor utilities.

These helpers back the MLR-30/32 aggregation-granularity ablation and are kept
separate from canonical Spotlight utilities to preserve provenance.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F


def compute_query_preserving_spotlight_bias(
    logits: torch.Tensor,
    span_ranges: List[List[Tuple[int, int]]],
    target_proportion: float,
    eps: float = 1e-12,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
    """Apply the query/head-preserving Spotlight variant.

    This is a novel aggregation-granularity ablation, not canonical Spotlight.
    It preserves both attention head and query-position dimensions in the
    internal controller: psi_current and bias_value are [heads, q_len] per
    batch item. Only key positions are reduced when measuring highlighted mass.

    Args:
        logits: Raw attention logits, shape [batch, num_heads, q_len, k_len].
        span_ranges: List of span-range lists, one per batch item.
        target_proportion: Target proportion of attention on span.
        eps: Numerical floor used only where highlighted mass is positive.
        return_diagnostics: When true, return query/head-resolved diagnostics.

    Returns:
        Modified attention weights with the same shape as logits. If
        return_diagnostics is true, returns (weights, diagnostics).
    """
    if logits.ndim != 4:
        raise ValueError(
            f"Expected logits with shape [batch, heads, q_len, k_len], got {tuple(logits.shape)}"
        )

    attn_weights = F.softmax(logits, dim=-1)
    modified_weights = attn_weights.clone()
    diagnostics: List[Dict[str, torch.Tensor]] = []

    batch_size, num_heads, q_len, k_len = logits.shape
    if len(span_ranges) != batch_size:
        raise ValueError(
            f"Expected {batch_size} span-range entries, got {len(span_ranges)}"
        )

    for batch_idx, ranges in enumerate(span_ranges):
        if not ranges:
            if return_diagnostics:
                zeros = torch.zeros(
                    (num_heads, q_len), device=logits.device, dtype=torch.float32
                )
                diagnostics.append({
                    "psi_current": zeros,
                    "bias_value": zeros,
                    "should_steer": torch.zeros(
                        (num_heads, q_len), device=logits.device, dtype=torch.bool
                    ),
                    "causally_reachable": torch.zeros(
                        (num_heads, q_len), device=logits.device, dtype=torch.bool
                    ),
                })
            continue

        union_mask = torch.zeros(
            k_len,
            device=modified_weights.device,
            dtype=modified_weights.dtype,
        )
        for start, end in ranges:
            safe_start = max(0, min(int(start), k_len))
            safe_end = max(safe_start, min(int(end), k_len))
            union_mask[safe_start:safe_end] = 1.0
        union_mask = union_mask.view(1, 1, k_len)  # [1, 1, K]

        highlighted_mass = (
            modified_weights[batch_idx] * union_mask
        ).sum(dim=2)  # [H, Q], summed over K only
        total_mass = modified_weights[batch_idx].sum(dim=2).clamp_min(eps)  # [H, Q]
        psi_current = highlighted_mass / total_mass

        causally_reachable = highlighted_mass > 0
        should_steer = causally_reachable & (psi_current < target_proportion)
        safe_psi = psi_current.clamp_min(eps)
        bias_value = torch.where(
            should_steer,
            torch.log(
                torch.as_tensor(
                    target_proportion,
                    device=modified_weights.device,
                    dtype=torch.float32,
                ) / safe_psi.float()
            ),
            torch.zeros_like(psi_current, dtype=torch.float32),
        )

        if torch.any(should_steer):
            bias_mask = (
                bias_value.view(num_heads, q_len, 1).to(logits.dtype)
                * union_mask.to(logits.dtype)
            )  # [H, Q, K], no head/query aggregation
            attn_logits = logits[batch_idx].float() + bias_mask.float()
            modified_weights[batch_idx] = F.softmax(
                attn_logits, dim=-1, dtype=torch.float32
            ).to(modified_weights.dtype)

        if return_diagnostics:
            diagnostics.append({
                "psi_current": psi_current.detach(),
                "bias_value": bias_value.detach(),
                "should_steer": should_steer.detach(),
                "causally_reachable": causally_reachable.detach(),
            })

    if return_diagnostics:
        return modified_weights, diagnostics

    return modified_weights


def generate_with_query_preserving_spotlight(
    llm,
    prompts,
    emph_strings,
    alpha: float = 0.2,
    sampling_params=None,
    **kwargs
):
    """Generate with the query/head-preserving Spotlight worker.

    Configure HookLLM with worker_name="probe_spotlight_query_preserving".
    The SamplingParams payload stays compatible with canonical Spotlight while
    the worker and controller implementation remain separate.
    """
    from vllm_hook_plugins.utils.spotlight.utils import generate_with_spotlight

    return generate_with_spotlight(
        llm=llm,
        prompts=prompts,
        emph_strings=emph_strings,
        alpha=alpha,
        sampling_params=sampling_params,
        **kwargs,
    )
