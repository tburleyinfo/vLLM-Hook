"""Qwen2 hooks for vLLM-Metal's MLX-backed model execution.

Generation remains owned by vLLM-Metal. This module observes its MLX
attention call and applies configured steering at the Qwen2 residual stream.
"""
from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


_PATCH_LOCK = threading.RLock()
_ACTIVE: "_CaptureSession | None" = None


def _mlx_numpy(value: Any, mx: Any) -> tuple[np.ndarray, torch.dtype]:
    """Export MLX float arrays while preserving the MIA tensor dtype contract."""
    dtype_map = {
        mx.float16: (mx.float16, torch.float16),
        mx.float32: (mx.float32, torch.float32),
        mx.bfloat16: (mx.float32, torch.bfloat16),
    }
    export_dtype = dtype_map.get(value.dtype)
    if export_dtype is None:
        raise TypeError(f"Unsupported MLX Q/K dtype: {value.dtype}")
    mlx_dtype, torch_dtype = export_dtype
    exported = value.astype(mlx_dtype) if value.dtype == mx.bfloat16 else value
    mx.eval(exported)
    return np.asarray(exported), torch_dtype


def _qk_tensors(
    q: np.ndarray,
    k: np.ndarray,
    q_dtype: torch.dtype | None = None,
    k_dtype: torch.dtype | None = None,
    hookq_mode: str = "last_token",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert post-RoPE Q/K arrays to the eager worker's flattened layout."""
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"Expected Q/K arrays [batch, heads, seq, dim], got {q.shape}, {k.shape}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"Q/K batch dimensions differ: {q.shape[0]} != {k.shape[0]}")
    if hookq_mode == "last_token":
        q_rows = np.ascontiguousarray(q[:, :, -1, :].reshape(q.shape[0], -1))
    elif hookq_mode == "all_tokens":
        q_rows = np.ascontiguousarray(
            q.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)
        )
    else:
        raise ValueError(f"hookq_mode={hookq_mode!r} must be 'last_token' or 'all_tokens'")
    k_rows = np.ascontiguousarray(k.transpose(0, 2, 1, 3).reshape(k.shape[0], k.shape[2], -1))
    q_tensor, k_tensor = torch.from_numpy(q_rows.copy()), torch.from_numpy(k_rows.copy())
    if q_dtype is not None:
        q_tensor = q_tensor.to(q_dtype)
    if k_dtype is not None:
        k_tensor = k_tensor.to(k_dtype)
    return q_tensor, k_tensor


def _attention_scores(
    q: np.ndarray,
    k: np.ndarray,
    heads: list[int],
    scale: float,
) -> torch.Tensor:
    """Compute selected-head last-token attention probabilities."""
    query = q[0, :, -1, :].astype(np.float32, copy=False)
    keys = k[0].transpose(1, 0, 2).astype(np.float32, copy=False)
    if keys.shape[0] < query.shape[0]:
        if query.shape[0] % keys.shape[0]:
            raise ValueError("Qwen2 query-head count must be divisible by key/value-head count")
        keys = np.repeat(keys, query.shape[0] // keys.shape[0], axis=0)
    logits = np.einsum("hd,hkd->hk", query, keys) * float(scale)
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    selected = np.ascontiguousarray(probabilities[np.asarray(heads, dtype=np.int64), None, :])
    return torch.from_numpy(selected.copy())


@dataclass
class _CaptureSession:
    prompts: list[Any]
    prompt_lens: list[int]
    output_layers: set[int] | None
    hookq_mode: str
    configs: list[dict[str, Any]]
    capture_qk: bool
    capture_mode: str
    layer_to_heads: dict[int, list[int]]
    qk_cache: list[dict[str, dict[str, Any]]] = field(default_factory=list)
    config: dict[str, Any] | None = None
    layer_ids: dict[int, int] = field(default_factory=dict)
    norm_layers: dict[int, int] = field(default_factory=dict)
    next_layer_id: int = 0
    next_prompt: int = 0
    current_prompt: int | None = None
    current_layer: int | None = None
    current_prefill_complete: bool = False
    last_request: int | None = None
    requested_steer: bool = False
    steering_cache: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.qk_cache = [{} for _ in self.prompts]

    def begin_layer(
        self, block: Any, input_value: Any
    ) -> tuple[int, int | None, int | None]:
        layer_id = self.layer_ids.get(id(block))
        if layer_id is None:
            layer_id = self.next_layer_id
            self.layer_ids[id(block)] = layer_id
            self.norm_layers[id(block.post_attention_layernorm)] = layer_id
            self.next_layer_id += 1

        if layer_id == 0:
            if self.current_prefill_complete:
                self.next_prompt += 1
                self.current_prefill_complete = False
            self.current_prompt = (
                self.next_prompt if self.next_prompt < len(self.prompts) else None
            )
            if self.current_prompt is not None:
                self.last_request = self.current_prompt
            elif self.prompts:
                self.last_request = len(self.prompts) - 1
            if not self.capture_qk and self.current_prompt is not None:
                if int(input_value.shape[-2]) >= self.prompt_lens[self.current_prompt]:
                    self.current_prefill_complete = True
        previous_layer = self.current_layer
        previous_request = self.last_request
        self.current_layer = layer_id
        return layer_id, previous_layer, previous_request

    def capture_attention(self, q: Any, k: Any, mx: Any) -> None:
        if not self.capture_qk:
            return
        layer_id = self.current_layer
        request_idx = self.current_prompt
        if layer_id is None or request_idx is None:
            return

        if len(q.shape) != 4 or len(k.shape) != 4:
            raise ValueError(f"Qwen2 attention returned unexpected Q/K shapes: {q.shape}, {k.shape}")
        if int(q.shape[0]) != 1:
            raise RuntimeError(
                "vLLM-Metal Qwen2 capture currently requires one request per "
                f"forward pass; observed batch size {q.shape[0]}"
            )
        prompt_len = self.prompt_lens[request_idx]
        if layer_id == 0 and not self.current_prefill_complete:
            if int(q.shape[2]) < prompt_len:
                return
            self.current_prefill_complete = True
        if not self.current_prefill_complete:
            return
        if self.output_layers is not None and layer_id not in self.output_layers:
            return

        q_np, q_dtype = _mlx_numpy(q, mx)
        k_np, k_dtype = _mlx_numpy(k, mx)
        if self.config is None:
            heads, head_dim = int(q_np.shape[1]), int(q_np.shape[3])
            self.config = {
                "num_attention_heads": heads,
                "num_key_value_heads": int(k_np.shape[1]),
                "head_dim": head_dim,
                "hidden_size": heads * head_dim,
                "attention_multiplier": float(1.0 / np.sqrt(head_dim)),
            }
        name = f"model.layers.{layer_id}.self_attn"
        entry = self.qk_cache[request_idx].setdefault(
            name,
            (
                {
                    "scores": [],
                    "heads": self.layer_to_heads.get(layer_id, [0]),
                    "layer_num": layer_id,
                    "hookq_mode": self.hookq_mode,
                    "capture": "score",
                }
                if self.capture_mode == "score"
                else {
                    "q": [],
                    "k_all": [],
                    "layer_num": layer_id,
                    "hookq_mode": self.hookq_mode,
                }
            ),
        )
        if self.capture_mode == "score":
            entry["scores"].append(
                _attention_scores(
                    q_np,
                    k_np,
                    entry["heads"],
                    float(self.config["attention_multiplier"]),
                )
            )
        else:
            q_tensor, k_tensor = _qk_tensors(
                q_np, k_np, q_dtype, k_dtype, self.hookq_mode
            )
            entry["q"].append(q_tensor[0])
            entry["k_all"].append(k_tensor[0])

    def steer_input(self, norm: Any, value: Any, mx: Any) -> Any:
        layer_id = self.norm_layers.get(id(norm))
        request_idx = self.last_request
        if layer_id is None or request_idx is None:
            return value
        cfg = self.configs[request_idx]
        steer = cfg.get("steer")
        if not isinstance(steer, dict) or not steer:
            return value
        if not _targets_layer(steer.get("optimal_layer", -1), layer_id):
            return value

        phase = steer.get("phase", "both")
        if phase not in ("prefill", "decode", "both"):
            raise ValueError(f"steering.phase={phase!r} is invalid")
        is_prefill = int(value.shape[-2]) > 1
        if phase == "prefill" and not is_prefill or phase == "decode" and is_prefill:
            return value
        positions = steer.get(
            "positions",
            "all_tokens" if steer.get("apply_at_all_positions", True) else "last_token",
        )
        if positions not in ("all_tokens", "last_token"):
            raise ValueError(f"steering.positions={positions!r} is invalid")

        path = os.path.expanduser(str(steer.get("vector_path", "")))
        data = self.steering_cache.get(path)
        if data is None:
            data = _load_steering_data(steer)
            self.steering_cache[path] = data
        vector = mx.array(data["dir"], dtype=value.dtype)
        if vector.size != value.shape[-1]:
            raise ValueError(
                f"Steering vector width {vector.size} does not match model width {value.shape[-1]}"
            )
        selected = value[..., -1:, :] if positions == "last_token" else value
        if steer.get("method", "adjust_rs") == "add_vector":
            changed = selected + float(steer.get("coefficient", 0)) * vector
        elif steer.get("method", "adjust_rs") == "adjust_rs":
            avg_proj = mx.array(data["avg_proj"], dtype=value.dtype)
            projection = (selected * vector).sum(axis=-1, keepdims=True)
            changed = selected + (avg_proj - projection) * vector
        else:
            raise ValueError(f"Unknown steering method: {steer.get('method')!r}")
        self.requested_steer = True
        if positions == "last_token":
            return mx.concatenate((value[..., :-1, :], changed), axis=-2)
        return changed

    def probes(self) -> dict[str, Any]:
        if self.config is None:
            raise RuntimeError("vLLM-Metal Qwen2 instrumentation captured no Q/K values")
        layer_sets = [set(per_prompt) for per_prompt in self.qk_cache]
        expected_layers = set.union(*layer_sets) if layer_sets else set()
        if not expected_layers:
            raise RuntimeError("vLLM-Metal Qwen2 instrumentation captured no Q/K values")
        for prompt_idx, layers in enumerate(layer_sets):
            missing = expected_layers - layers
            if missing:
                raise RuntimeError(
                    f"vLLM-Metal Qwen2 capture missed layers {sorted(missing)} "
                    f"for prompt {prompt_idx}"
                )
        merged: dict[str, dict[str, Any]] = {}
        for per_prompt in self.qk_cache:
            for name, entry in per_prompt.items():
                target = merged.setdefault(
                    name,
                    {
                        "q": [],
                        "k_all": [],
                        "layer_num": entry["layer_num"],
                        "hookq_mode": self.hookq_mode,
                    },
                )
                if self.capture_mode == "score":
                    target["scores"] = []
                    target["heads"] = entry["heads"]
                    target["capture"] = "score"
                    target["scores"].extend(entry["scores"])
                else:
                    target["q"].extend(entry["q"])
                    target["k_all"].extend(entry["k_all"])
        if not merged:
            raise RuntimeError("vLLM-Metal Qwen2 instrumentation captured no Q/K values")
        return {"config": dict(self.config), "qk_cache": merged}


def _targets_layer(target: Any, layer_id: int) -> bool:
    if target == "all":
        return True
    if isinstance(target, (list, tuple)):
        return any(str(value) == str(layer_id) for value in target)
    try:
        return int(target) == layer_id
    except (TypeError, ValueError):
        return False


def _load_steering_data(config: dict[str, Any]) -> dict[str, np.ndarray]:
    path = os.path.expanduser(str(config.get("vector_path", "")))
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Steering vector not found at: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if "dir" not in loaded:
        raise KeyError(f"Steering vector {path!r} has no 'dir' entry")
    vector = loaded["dir"]
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().float().cpu().numpy()
    result = {"dir": np.asarray(vector, dtype=np.float32)}
    if config.get("method", "adjust_rs") == "adjust_rs":
        avg_proj = loaded.get("avg_proj")
        if avg_proj is None:
            raise KeyError(f"Steering vector {path!r} has no 'avg_proj' entry")
        if isinstance(avg_proj, torch.Tensor):
            avg_proj = avg_proj.detach().float().cpu().numpy()
        result["avg_proj"] = np.asarray(avg_proj, dtype=np.float32)
    return result


@contextlib.contextmanager
def instrument_qwen2(
    *,
    prompts: list[Any],
    prompt_lens: list[int],
    output_layers: set[int] | None,
    hookq_mode: str,
    configs: list[dict[str, Any]],
    capture_qk: bool = True,
    capture_mode: str = "qk",
    layer_to_heads: dict[int, list[int]] | None = None,
):
    """Temporarily instrument the actual vLLM-Metal Qwen2 forward path."""
    global _ACTIVE
    if len(prompts) != len(prompt_lens) or len(prompts) != len(configs):
        raise ValueError("Metal instrumentation requires one token length and config per prompt")
    if capture_mode not in ("qk", "score"):
        raise ValueError(f"qk_capture={capture_mode!r} must be 'qk' or 'score'")

    with _PATCH_LOCK:
        try:
            import mlx.core as mx
            import mlx_lm.models.qwen2 as qwen2
        except ImportError as exc:
            raise RuntimeError(
                "Metal instrumentation requires the installed mlx-lm Qwen2 implementation"
            ) from exc

        session = _CaptureSession(
            prompts=prompts,
            prompt_lens=prompt_lens,
            output_layers=output_layers,
            hookq_mode=hookq_mode,
            configs=configs,
            capture_qk=capture_qk,
            capture_mode=capture_mode,
            layer_to_heads=layer_to_heads or {},
        )
        original_sdpa = qwen2.scaled_dot_product_attention
        original_block_call = qwen2.TransformerBlock.__call__
        patched_norm_classes: dict[type, Any] = {}
        _ACTIVE = session

        def capture_sdpa(q: Any, k: Any, *args: Any, **kwargs: Any) -> Any:
            if _ACTIVE is not None:
                _ACTIVE.capture_attention(q, k, mx)
            return original_sdpa(q, k, *args, **kwargs)

        def block_call(block: Any, *args: Any, **kwargs: Any) -> Any:
            active = _ACTIVE
            if active is None:
                return original_block_call(block, *args, **kwargs)
            input_value = args[0] if args else kwargs.get("x")
            if input_value is None:
                raise TypeError("Qwen2 TransformerBlock requires its hidden-state input")
            layer_id, previous_layer, previous_request = active.begin_layer(
                block, input_value
            )
            norm_cls = type(block.post_attention_layernorm)
            if norm_cls not in patched_norm_classes:
                original_norm_call = norm_cls.__call__
                patched_norm_classes[norm_cls] = original_norm_call

                def norm_call(norm: Any, value: Any, *norm_args: Any, **norm_kwargs: Any) -> Any:
                    session_now = _ACTIVE
                    if session_now is not None:
                        value = session_now.steer_input(norm, value, mx)
                    return original_norm_call(norm, value, *norm_args, **norm_kwargs)

                norm_cls.__call__ = norm_call
            try:
                return original_block_call(block, *args, **kwargs)
            finally:
                active.current_layer = previous_layer
                if previous_layer is not None:
                    active.last_request = previous_request

        qwen2.scaled_dot_product_attention = capture_sdpa
        qwen2.TransformerBlock.__call__ = block_call
        try:
            yield session
        finally:
            qwen2.scaled_dot_product_attention = original_sdpa
            qwen2.TransformerBlock.__call__ = original_block_call
            for norm_cls, original_call in patched_norm_classes.items():
                norm_cls.__call__ = original_call
            _ACTIVE = None
