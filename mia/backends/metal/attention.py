"""Q/K capture at vLLM-Metal's MLX SDPA preparation seam."""
from __future__ import annotations

import contextvars
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch


_ACTIVE: contextvars.ContextVar["MetalQKCapture | None"] = contextvars.ContextVar(
    "mia_metal_qk_capture", default=None
)
_PATCH_LOCK = threading.RLock()
_PATCHED_MODULE = None
_ORIGINAL_PREPARE = None


def _attention_layers(model: Any) -> list[tuple[int, Any]]:
    from vllm_metal.attention.patching import find_attn_attr, find_layers

    found = []
    for layer_idx, layer in enumerate(find_layers(model)):
        attr = find_attn_attr(layer)
        if attr is None:
            continue
        attention = getattr(layer, attr)
        attention = getattr(attention, "_inner", attention)
        found.append((layer_idx, attention))
    if not found:
        raise ValueError(f"No vLLM-Metal SDPA attention layers found in {type(model).__name__}")
    return found


def _install_prepare_hook() -> None:
    global _PATCHED_MODULE, _ORIGINAL_PREPARE
    module = __import__(
        "vllm_metal.attention.impls.sdpa", fromlist=["prepare_sdpa_qkv"]
    )
    if _PATCHED_MODULE is not None:
        raise RuntimeError("MIA Metal Q/K hook is already installed")
    original = module.prepare_sdpa_qkv

    def capture_prepare(inner, *args, **kwargs):
        result = original(inner, *args, **kwargs)
        session = _ACTIVE.get()
        if session is not None:
            session._record(inner, result[0], result[1])
        return result

    module.prepare_sdpa_qkv = capture_prepare
    _PATCHED_MODULE = module
    _ORIGINAL_PREPARE = original


def _export_mlx(value: Any) -> np.ndarray:
    import mlx.core as mx

    # MLX bfloat16 does not implement the NumPy buffer protocol.
    return np.asarray(value.astype(mx.float32))


class MetalQKCapture:
    """Collect post-RoPE Q/K arrays without changing model outputs.

    Supply the loaded MLX model, then enter this context around one
    ``vllm.LLM.generate`` call. ``probes`` returns the same qk_cache/config
    shape consumed by MIA's existing attention-tracker analyzer.
    """

    def __init__(self, model: Any, *, hookq_mode: str = "last_token") -> None:
        if os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION", "1") != "1":
            raise RuntimeError(
                "The Metal Q/K adapter currently observes vLLM-Metal's paged SDPA path, "
                "but VLLM_METAL_USE_PAGED_ATTENTION is disabled. The non-paged MLX "
                "attention path needs a separate capture seam."
            )
        if hookq_mode != "last_token":
            raise ValueError(
                "The Metal Q/K adapter currently supports hookq_mode='last_token' only; "
                "all_tokens capture needs full cached-key reconstruction."
            )
        self.model = model
        self.hookq_mode = hookq_mode
        self._layers = _attention_layers(model)
        self._layer_by_object = {id(attn): layer for layer, attn in self._layers}
        self._captured: dict[int, list[dict[str, torch.Tensor]]] = {}
        self._context_token = None

    def __enter__(self) -> "MetalQKCapture":
        _PATCH_LOCK.acquire()
        try:
            if _ACTIVE.get() is not None:
                raise RuntimeError("Nested MIA Metal Q/K capture contexts are not supported")
            _install_prepare_hook()
            self._context_token = _ACTIVE.set(self)
            return self
        except Exception:
            _PATCH_LOCK.release()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        global _PATCHED_MODULE, _ORIGINAL_PREPARE
        try:
            if self._context_token is not None:
                _ACTIVE.reset(self._context_token)
                self._context_token = None
            if _PATCHED_MODULE is not None:
                _PATCHED_MODULE.prepare_sdpa_qkv = _ORIGINAL_PREPARE
                _PATCHED_MODULE = None
                _ORIGINAL_PREPARE = None
        finally:
            _PATCH_LOCK.release()
        return False

    def _record(self, attention: Any, queries: Any, keys: Any) -> None:
        layer = self._layer_by_object.get(id(attention))
        if layer is None:
            return
        import mlx.core as mx

        mx.eval(queries, keys)
        q_array = _export_mlx(queries)
        k_array = _export_mlx(keys)
        if q_array.ndim != 4 or k_array.ndim != 4:
            raise ValueError(
                f"Expected Metal SDPA Q/K in [batch, heads, tokens, dim] layout; "
                f"got Q={q_array.shape}, K={k_array.shape}"
            )
        if q_array.shape[0] != k_array.shape[0]:
            raise ValueError("Metal SDPA Q/K batch dimensions do not match")
        if q_array.shape[2] == 1:
            return

        rows = self._captured.setdefault(layer, [])
        for batch_idx in range(q_array.shape[0]):
            q = q_array[batch_idx].transpose(1, 0, 2).reshape(q_array.shape[2], -1)
            k = k_array[batch_idx].transpose(1, 0, 2).reshape(k_array.shape[2], -1)
            q = q[-1:]
            tensor_q = torch.from_numpy(np.ascontiguousarray(q)).clone()
            tensor_k = torch.from_numpy(np.ascontiguousarray(k)).clone()
            if rows:
                previous = rows[batch_idx] if batch_idx < len(rows) else None
                if previous is not None and previous["k_all"].shape[0] >= tensor_k.shape[0]:
                    continue
                if previous is not None:
                    rows[batch_idx] = {"q": tensor_q, "k_all": tensor_k}
                else:
                    rows.append({"q": tensor_q, "k_all": tensor_k})
            else:
                rows.append({"q": tensor_q, "k_all": tensor_k})

    def probes(self) -> dict[str, Any]:
        """Return CPU probes compatible with ``AttntrackerAnalyzer``."""
        if not self._captured:
            raise RuntimeError("No Q/K values were captured by this Metal Q/K session")
        config = _model_attention_config(self.model, self._layers)
        qk_cache = {}
        for layer_idx, rows in sorted(self._captured.items()):
            if not rows:
                continue
            qk_cache[f"model.layers.{layer_idx}.self_attn.attn"] = {
                "q": [row["q"] for row in rows],
                "k_all": [row["k_all"] for row in rows],
                "layer_num": layer_idx,
                "hookq_mode": self.hookq_mode,
            }
        return {"config": config, "qk_cache": qk_cache}


def _model_attention_config(model: Any, layers: list[tuple[int, Any]]) -> dict[str, int | float]:
    args = getattr(model, "args", None)
    if args is None:
        args = getattr(getattr(model, "model", None), "args", None)
    attention = layers[0][1]

    def required(name: str, *alternatives: str) -> int:
        for source in (args, attention):
            for key in (name, *alternatives):
                value = getattr(source, key, None) if source is not None else None
                if value is not None:
                    return int(value)
        raise ValueError(f"Cannot resolve {name} from the loaded Metal model")

    heads = required("num_attention_heads", "n_heads")
    kv_heads = required("num_key_value_heads", "n_kv_heads")
    try:
        head_dim = required("head_dim")
    except ValueError:
        projection = getattr(attention, "q_proj", None)
        weight = getattr(projection, "weight", None)
        if weight is None:
            raise
        head_dim = int(weight.shape[0]) // heads
    scale = getattr(attention, "scale", None)
    if scale is None:
        scale = head_dim**-0.5
    return {
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "attention_multiplier": float(scale),
    }


@contextmanager
def capture_qk(model: Any, *, hookq_mode: str = "last_token") -> Iterator[MetalQKCapture]:
    """Context-manager form of :class:`MetalQKCapture`."""
    with MetalQKCapture(model, hookq_mode=hookq_mode) as capture:
        yield capture
