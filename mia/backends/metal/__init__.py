"""MIA hooks for vLLM-Metal's MLX-backed models."""

from mia.backends.metal.attention import MetalQKCapture, capture_qk
from mia.backends.metal.steering import install_activation_steering

__all__ = ["MetalQKCapture", "capture_qk", "install_activation_steering"]
