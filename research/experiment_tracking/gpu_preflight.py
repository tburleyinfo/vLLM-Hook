"""GPU memory preflight helpers for research notebooks."""

from __future__ import annotations


def required_vram_bytes(total_bytes: int, gpu_memory_utilization: float) -> int:
    """Return vLLM's configured startup reservation target."""

    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in the interval (0, 1].")
    return int(total_bytes * gpu_memory_utilization)


def has_required_vram(
    *,
    free_bytes: int,
    total_bytes: int,
    gpu_memory_utilization: float,
) -> bool:
    """Whether current free VRAM satisfies the configured vLLM reservation."""

    return free_bytes >= required_vram_bytes(total_bytes, gpu_memory_utilization)


def format_gib(num_bytes: int) -> str:
    """Format bytes as GiB for notebook diagnostics."""

    return f"{num_bytes / 1024**3:.2f} GiB"

