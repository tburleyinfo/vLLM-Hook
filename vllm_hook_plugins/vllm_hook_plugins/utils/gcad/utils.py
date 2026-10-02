"""Utilities for Gated Cropped Attention-Delta steering.

The worker consumes saved vectors with this shape:

{
    "layers": {
        12: {
            "delta": Tensor[hidden_size],
            "mean_sys_key": Tensor[num_heads, head_dim],
            "d_bar": scalar,
        },
        ...
    }
}

The tensors can also use string layer keys because JSON-adjacent metadata often
round-trips dictionary keys as strings.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import torch


def load_gcad_vectors(vector_path: str) -> Dict[str, Any]:
    """Load a GCAD vector artifact from disk."""
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"GCAD vector artifact not found at: {vector_path}")
    return torch.load(vector_path, weights_only=False)


def normalize_gcad_layers(raw: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Return layer-indexed GCAD entries with normalized key types."""
    if "layers" not in raw or not isinstance(raw["layers"], dict):
        raise ValueError("GCAD vector artifact must contain a 'layers' dict")

    layers: Dict[int, Dict[str, Any]] = {}
    for layer_key, entry in raw["layers"].items():
        layer_num = int(layer_key)
        if "delta" not in entry:
            raise ValueError(f"GCAD layer {layer_num} is missing 'delta'")
        layers[layer_num] = dict(entry)
    return layers


def compute_gcad_gate(
    query: torch.Tensor,
    mean_sys_key: torch.Tensor,
    d_bar: torch.Tensor | float,
    c_base: float,
    sharpness: float,
) -> torch.Tensor:
    """Compute the token gate from prompt-key compatibility.

    Args:
        query: Query tensor shaped [heads, q_len, head_dim].
        mean_sys_key: Mean system-prompt key shaped [heads, head_dim].
        d_bar: Extraction-time mean compatibility baseline.
        c_base: Nominal steering strength.
        sharpness: Sigmoid sharpness.

    Returns:
        Gate tensor shaped [q_len].
    """
    if query.dim() != 3:
        raise ValueError("query must have shape [heads, q_len, head_dim]")
    if mean_sys_key.dim() != 2:
        raise ValueError("mean_sys_key must have shape [heads, head_dim]")
    if query.size(0) != mean_sys_key.size(0):
        raise ValueError("query and mean_sys_key must have the same head count")
    if query.size(-1) != mean_sys_key.size(-1):
        raise ValueError("query and mean_sys_key must have the same head_dim")

    scale = query.size(-1) ** -0.5
    compat = (query * mean_sys_key[:, None, :]).sum(dim=-1) * scale
    compat = compat.mean(dim=0)
    baseline = torch.as_tensor(d_bar, device=query.device, dtype=query.dtype)
    return 2.0 * float(c_base) * torch.sigmoid(float(sharpness) * (compat - baseline))
