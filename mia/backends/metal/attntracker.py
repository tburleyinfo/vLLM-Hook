"""In-memory Attention Tracker capture for vLLM-Metal's MLX Granite model."""
from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import torch


_METAL_PLUGIN_NAME = "metal"
_PATCHED = False
_ACTIVE_SESSION = None
_SESSION_LOCK = threading.RLock()
_LAYER_BY_ID: dict[int, int] = {}
_CURRENT_LAYER = threading.local()
_ORIGINAL_CALL = None
_ORIGINAL_SDPA = None
_INSTALLED_WORKER = None


def metal_plugin_enabled() -> bool:
    """Whether vLLM was configured to load its Metal platform plugin."""
    return _METAL_PLUGIN_NAME in {
        item.strip() for item in os.environ.get("VLLM_PLUGINS", "").split(",")
    }


def _mlx_arrays_to_numpy(query: Any, keys: Any) -> tuple[np.ndarray, np.ndarray]:
    import mlx.core as mx

    mx.eval(query, keys)
    q = np.asarray(query.astype(mx.float32))
    k = np.asarray(keys.astype(mx.float32))
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError(
            "Metal Granite Q/K capture expected single-sequence [1, heads, tokens, dim] "
            f"arrays; got q={q.shape}, k={k.shape}"
        )
    return q, k


class MetalQKCaptureSession:
    """Collect the prompt's final query and full keys for one generated prompt.

    MLX arrays are materialized only while a capture session is active. The
    captured payload uses MIA's existing ``qk_cache`` and ``config`` contract.
    """

    def __init__(
        self,
        layer_to_heads: dict[int, list],
        model_config: Any,
        expected_prompts: int = 1,
    ):
        self.layer_to_heads = {int(layer): heads for layer, heads in layer_to_heads.items()}
        self.model_config = model_config
        self.expected_prompts = int(expected_prompts)
        if self.expected_prompts < 1:
            raise ValueError("expected_prompts must be positive")
        self._captured: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def __enter__(self) -> "MetalQKCaptureSession":
        global _ACTIVE_SESSION
        _SESSION_LOCK.acquire()
        if _ACTIVE_SESSION is not None:
            _SESSION_LOCK.release()
            raise RuntimeError("A Metal Q/K capture session is already active")
        _ACTIVE_SESSION = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _ACTIVE_SESSION
        _ACTIVE_SESSION = None
        _SESSION_LOCK.release()

    def capture(self, layer_num: int, query: Any, keys: Any) -> None:
        if (layer_num not in self.layer_to_heads
                or len(self._captured.get(layer_num, ())) >= self.expected_prompts):
            return
        q, k = _mlx_arrays_to_numpy(query, keys)
        q_row = q[0, :, -1, :].reshape(-1).copy()
        k_rows = k[0].transpose(1, 0, 2).reshape(k.shape[2], -1).copy()
        self._captured.setdefault(layer_num, []).append(
            (torch.from_numpy(q_row), torch.from_numpy(k_rows))
        )

    def probes(self) -> list[dict]:
        missing = sorted(
            layer_num for layer_num in self.layer_to_heads
            if len(self._captured.get(layer_num, ())) != self.expected_prompts
        )
        if missing:
            raise RuntimeError(
                "Metal Granite attention did not capture configured layers "
                f"{missing} for {self.expected_prompts} prompt(s); verify the model uses "
                "Granite SDPA and that the "
                "vLLM-Metal worker hook was installed."
            )
        heads = int(_config_value(self.model_config, "num_attention_heads"))
        kv_heads = int(_config_value(
            self.model_config, "num_key_value_heads", heads
        ))
        hidden_size = int(_config_value(self.model_config, "hidden_size"))
        head_dim = int(_config_value(self.model_config, "head_dim", hidden_size // heads))
        multiplier = float(_config_value(
            self.model_config, "attention_multiplier", 1 / np.sqrt(head_dim)
        ))
        config = {
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "attention_multiplier": multiplier,
        }
        payloads = []
        for prompt_index in range(self.expected_prompts):
            qk_cache = {}
            for layer_num, captured in self._captured.items():
                query, keys = captured[prompt_index]
                qk_cache[layer_num] = {
                    "q": [query],
                    "k_all": [keys],
                    "layer_num": layer_num,
                    "heads": list(self.layer_to_heads[layer_num]),
                    "hookq_mode": "last_token",
                }
            payloads.append({"config": config, "qk_cache": qk_cache})
        return payloads


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    return default if value is None else value


def write_qk_artifact(probes: dict, hook_dir: str, run_id: str) -> str:
    """Write a Metal capture using the existing per-rank ``qk.pt`` layout."""
    from mia.core.runtime.tp_shard import rank_dir_name
    from mia.artifacts import write_refused_qk
    from mia.workers._common import clear_rank_artifact

    run_dir = os.path.join(hook_dir, str(run_id), rank_dir_name(0))
    os.makedirs(run_dir, exist_ok=True)
    clear_rank_artifact(run_dir, "qk")
    write_refused_qk(run_dir, {})
    path = os.path.join(run_dir, "qk.pt")
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as stream:
        torch.save(probes, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
    return run_dir


def install_granite_capture(model: Any) -> None:
    """Install output-preserving wrappers on Granite attention and SDPA."""
    global _PATCHED, _ORIGINAL_CALL, _ORIGINAL_SDPA
    if _PATCHED:
        for layer_num, attention in enumerate(_attention_layers(model)):
            _LAYER_BY_ID[id(attention)] = layer_num
        return

    from mlx_lm.models import granite

    _ORIGINAL_CALL = granite.Attention.__call__
    _ORIGINAL_SDPA = granite.scaled_dot_product_attention

    def attention_call(instance, *args, **kwargs):
        previous = getattr(_CURRENT_LAYER, "layer_num", None)
        _CURRENT_LAYER.layer_num = _LAYER_BY_ID.get(id(instance))
        try:
            return _ORIGINAL_CALL(instance, *args, **kwargs)
        finally:
            _CURRENT_LAYER.layer_num = previous

    def scaled_dot_product_attention(query, keys, *args, **kwargs):
        session = _ACTIVE_SESSION
        layer_num = getattr(_CURRENT_LAYER, "layer_num", None)
        if session is not None and layer_num is not None:
            session.capture(layer_num, query, keys)
        return _ORIGINAL_SDPA(query, keys, *args, **kwargs)

    granite.Attention.__call__ = attention_call
    granite.scaled_dot_product_attention = scaled_dot_product_attention
    for layer_num, attention in enumerate(_attention_layers(model)):
        _LAYER_BY_ID[id(attention)] = layer_num
    _PATCHED = True


def _attention_layers(model: Any):
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
    if layers is None:
        raise TypeError("Cannot locate Granite transformer layers on the loaded MLX model")
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise TypeError(
                f"Expected Granite self_attn module, found layer {type(layer).__name__}"
            )
        yield attention


def install_metal_worker_hook() -> bool:
    """Attach MIA's existing install_hooks RPC to vLLM-Metal's worker class."""
    global _INSTALLED_WORKER
    if not metal_plugin_enabled():
        return False
    from vllm_metal.v1.worker import MetalWorker

    if _INSTALLED_WORKER is MetalWorker:
        return True
    original = getattr(MetalWorker, "install_hooks", None)

    def install_hooks(worker):
        runner = getattr(worker, "model_runner", None)
        model = getattr(runner, "model", None)
        if model is None:
            raise RuntimeError("vLLM-Metal model is not loaded; cannot install MIA hooks")
        install_granite_capture(model)
        if original is not None:
            return original(worker)
        return None

    MetalWorker.install_hooks = install_hooks
    _INSTALLED_WORKER = MetalWorker
    return True
