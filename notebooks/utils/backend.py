from __future__ import annotations

import importlib.util
import os
import platform
import sys


SUPPORTED_BACKENDS = ("vllm", "mlx")


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def in_colab() -> bool:
    if "google.colab" in sys.modules:
        return True
    return "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def has_mlx() -> bool:
    if not is_apple_silicon():
        return False
    return _has_module("mlx") and _has_module("vllm_mlx")


def select_backend() -> str:
    override = os.getenv("VLLM_HOOK_BACKEND", "").strip().lower()
    if override in SUPPORTED_BACKENDS:
        return override
    if has_mlx():
        return "mlx"
    return "vllm"


def get_hook_llm(feature: str = "this notebook"):
    backend = select_backend()
    if backend != "vllm":
        raise RuntimeError(
            f"{feature} requires the vLLM backend. MLX is detected but not supported for hooks yet."
        )
    from vllm_hook_plugins import HookLLM

    return HookLLM
