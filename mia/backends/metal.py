"""MIA instrumentation for the vLLM-Metal MLX model runner."""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import numpy as np
import torch

from mia.errors import MiaConfigurationError


_ACTIVE_QK_CAPTURE: ContextVar[dict | None] = ContextVar(
    "mia_metal_qk_capture", default=None
)
_ACTIVE_STEER: ContextVar[dict | None] = ContextVar("mia_metal_steer", default=None)
_INSTALL_ATTR = "_mia_metal_qk_capture_installed"
_STEER_INSTALL_ATTR = "_mia_metal_steer_installed"


def _is_metal_runner(runner: Any) -> bool:
    return type(runner).__module__.startswith("vllm_metal.")


def _model_layers(model: Any) -> list:
    root = getattr(model, "language_model", model)
    container = getattr(root, "model", root)
    layers = getattr(container, "layers", None)
    if layers is None:
        layers = getattr(root, "layers", None)
    if layers is None:
        raise MiaConfigurationError(
            f"Cannot locate decoder layers in Metal model {type(model).__name__}."
        )
    return list(layers)


def _selected(spec: Any, layer_num: int) -> bool:
    if spec is True:
        return True
    if isinstance(spec, dict):
        return any(int(layer) == layer_num for layer in spec)
    if isinstance(spec, (list, tuple)):
        return any(int(layer) == layer_num for layer in spec)
    return False


def _capture_active(state: dict, layer_num: int) -> bool:
    spec = state["extra"].get("output_qk")
    if spec is None or not _selected(spec, layer_num):
        return False
    if state["extra"].get("hooks_on", "prefill") != "prefill":
        return False
    return True


@contextmanager
def _request_capture(worker: Any, req_id: str, sampling_params: Any) -> Iterator[None]:
    extra = getattr(sampling_params, "extra_args", None) or {}
    if extra.get("save_to_disk"):
        raise MiaConfigurationError(
            "Metal QK durable capture is not implemented; use the in-memory "
            "probe path (save_to_disk=False)."
        )
    if extra.get("hooks_on", "prefill") != "prefill":
        raise MiaConfigurationError(
            "Metal QK capture currently supports hooks_on='prefill' only."
        )
    state = {
        "worker": worker,
        "req_id": str(req_id),
        "extra": extra,
        "mode": extra.get("hookq_mode", getattr(worker, "hookq_mode", "all_tokens")),
    }
    token = _ACTIVE_QK_CAPTURE.set(state)
    try:
        yield
    finally:
        _ACTIVE_QK_CAPTURE.reset(token)


def _as_cpu_tensor(array: Any, mx: Any) -> torch.Tensor:
    values = np.asarray(array.astype(mx.float32))
    return torch.from_numpy(np.array(values, copy=True))


def _qk_capture(
    state: dict,
    layer_num: int,
    name: str,
    inner: Any,
    x: Any,
    cache: Any,
    mx: Any,
) -> None:
    if not _capture_active(state, layer_num):
        return
    required = ("q_proj", "k_proj", "rope", "n_heads", "n_kv_heads")
    missing = [attr for attr in required if not hasattr(inner, attr)]
    if missing:
        raise MiaConfigurationError(
            f"Metal QK capture for {name} requires Qwen-style attributes; missing {missing}."
        )

    batch, seq_len, _ = x.shape
    q_heads = int(inner.n_heads)
    kv_heads = int(inner.n_kv_heads)
    q_proj = inner.q_proj(x).reshape(batch, seq_len, q_heads, -1).transpose(0, 2, 1, 3)
    k_proj = inner.k_proj(x).reshape(batch, seq_len, kv_heads, -1).transpose(0, 2, 1, 3)
    offset = getattr(cache, "offset", 0) if cache is not None else 0
    q_rope = inner.rope(q_proj, offset=offset)
    k_rope = inner.rope(k_proj, offset=offset)
    mx.eval(q_rope, k_rope)

    q_rows = q_rope[0, :, -1, :].reshape(-1)
    k_rows = k_rope[0].transpose(1, 0, 2).reshape(seq_len, -1)
    entry = state["worker"]._captured_states.setdefault(state["req_id"], {})
    layer_entry = entry.setdefault(
        name,
        {"q": [], "k_all": [], "layer_num": layer_num, "hookq_mode": state["mode"]},
    )
    if state["mode"] == "all_tokens":
        q_rows = q_rope[0].transpose(1, 0, 2).reshape(seq_len, -1)
    layer_entry["q"].append(_as_cpu_tensor(q_rows, mx))
    layer_entry["k_all"].append(_as_cpu_tensor(k_rows, mx))


def _wrap_attention(layer: Any, layer_num: int, name: str, mx: Any, nn: Any) -> bool:
    inner = getattr(layer, "self_attn", None)
    if inner is None:
        return False
    if getattr(inner, "_mia_metal_qk_wrapped", False):
        return True

    class MetalQKAttention(nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            object.__setattr__(self, "_mia_inner", wrapped)

        def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
            active = _ACTIVE_QK_CAPTURE.get()
            if active is not None and _capture_active(active, layer_num):
                cache = kwargs.get("cache", args[1] if len(args) > 1 else None)
                _qk_capture(active, layer_num, name, self._mia_inner, x, cache, mx)
            return self._mia_inner(x, *args, **kwargs)

    wrapped = MetalQKAttention(inner)
    object.__setattr__(wrapped, "_mia_metal_qk_wrapped", True)
    setattr(layer, "self_attn", wrapped)
    return True


def install_metal_qk_capture(worker: Any) -> bool:
    """Install request-scoped Q/K capture on a non-paged vLLM-Metal runner."""
    runner = getattr(worker, "model_runner", None)
    if runner is None or not _is_metal_runner(runner):
        return False
    if getattr(runner, _INSTALL_ATTR, False):
        return True

    if getattr(runner, "paged_attention_runtime", None) is not None:
        raise MiaConfigurationError(
            "The MIA Metal QK bridge currently requires the non-paged vLLM-Metal "
            "runner (VLLM_METAL_USE_PAGED_ATTENTION=0)."
        )

    model = getattr(runner, "model", None)
    if model is None:
        raise MiaConfigurationError("The vLLM-Metal runner has no loaded model.")
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise MiaConfigurationError(
            "The vLLM-Metal QK bridge requires mlx and mlx.nn."
        ) from exc

    args = getattr(model, "args", None)
    if args is None:
        args = getattr(getattr(model, "model", None), "args", None)
    if args is None:
        raise MiaConfigurationError("Cannot read model dimensions for Metal QK capture.")
    num_heads = int(getattr(args, "num_attention_heads"))
    num_kv_heads = int(getattr(args, "num_key_value_heads", num_heads))
    hidden_size = int(getattr(args, "hidden_size"))
    head_dim = int(getattr(args, "head_dim", hidden_size // num_heads))
    worker._conf = {
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "attention_multiplier": float(
            getattr(args, "attention_multiplier", head_dim**-0.5)
        ),
    }
    worker.hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
    worker._captured_states = getattr(worker, "_captured_states", None) or {}
    worker._disk_states = getattr(worker, "_disk_states", None) or {}

    installed = 0
    names = {}
    for layer_num, layer in enumerate(_model_layers(model)):
        name = f"model.layers.{layer_num}.self_attn"
        if _wrap_attention(layer, layer_num, name, mx, nn):
            installed += 1
            names[layer_num] = name
    if not installed:
        raise MiaConfigurationError("No compatible Metal self-attention layers were found.")
    worker._qk_module_names = names

    original_prefill = runner._prefill_single

    def prefill_single(req_id: str, token_ids: list[int], sampling_params: Any, *args: Any,
                       **kwargs: Any) -> Any:
        with _request_capture(worker, req_id, sampling_params):
            return original_prefill(req_id, token_ids, sampling_params, *args, **kwargs)

    runner._prefill_single = prefill_single
    setattr(runner, _INSTALL_ATTR, True)
    return True


def _request_steer_config(sampling_params: Any, env_path: str | None) -> dict | None:
    from mia.workers.steer_worker import _resolve_steer_config

    extra = getattr(sampling_params, "extra_args", None) or {}
    return _resolve_steer_config(extra.get("steer"), env_path)


def _steer_vector(worker: Any, config: dict, mx: Any) -> tuple[Any, Any]:
    path = config["vector_path"]
    cached = worker._metal_steer_vectors.get(path)
    if cached is None:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        direction = torch.as_tensor(raw["dir"]).detach().float().cpu().numpy()
        avg_proj = raw.get("avg_proj", 0.0)
        if isinstance(avg_proj, torch.Tensor):
            avg_proj = avg_proj.detach().float().cpu().numpy()
        cached = (mx.array(direction), mx.array(avg_proj))
        worker._metal_steer_vectors[path] = cached
    return cached


def _steer_layer_output(worker: Any, layer_num: int, output: Any, mx: Any) -> Any:
    state = _ACTIVE_STEER.get()
    if state is None:
        return output
    if isinstance(output, tuple):
        if len(output) < 2:
            return output
        prefix, activations = output[0], output[1]
    else:
        prefix, activations = None, output
    if not hasattr(activations, "shape") or len(activations.shape) != 3:
        raise MiaConfigurationError(
            "Metal activation steering expects decoder-layer output shaped [batch, seq, hidden]."
        )

    if state["sequential"]:
        if layer_num == 0:
            state["cursor"] += 1
        cursor = state["cursor"]
        configs = state["configs"][cursor : cursor + 1]
    else:
        configs = state["configs"]

    batch_size, seq_len, _ = activations.shape
    if len(configs) != batch_size:
        raise MiaConfigurationError(
            f"Metal steering mapped {len(configs)} requests to an activation batch of "
            f"{batch_size}."
        )
    mask = np.zeros((batch_size, seq_len), dtype=bool)
    updates = []
    for row, config in enumerate(configs):
        if not config or not _steer_layer_selected(config.get("optimal_layer", -1), layer_num):
            updates.append(None)
            continue
        phase, positions = _steer_modes(config)
        if (phase == "prefill" and not state["prefill"]) or (
            phase == "decode" and state["prefill"]
        ):
            updates.append(None)
            continue
        lo = seq_len - 1 if positions == "last_token" else 0
        mask[row, lo:seq_len] = True
        updates.append((config, _steer_vector(worker, config, mx)))

    if not mask.any():
        return output
    result = activations
    for row, update in enumerate(updates):
        if update is None:
            continue
        config, (direction, avg_proj) = update
        direction = direction.astype(activations.dtype)
        row_values = result[row]
        if config.get("method", "adjust_rs") == "add_vector":
            changed = row_values + float(config.get("coefficient", 0)) * direction
        elif config.get("method", "adjust_rs") == "adjust_rs":
            current = mx.sum(row_values * direction, axis=-1, keepdims=True)
            changed = row_values + (avg_proj - current) * direction
        else:
            raise MiaConfigurationError(
                f"Unknown steering method: {config.get('method')!r}."
            )
        row_mask = mx.array(mask[row])[:, None]
        row_values = mx.where(row_mask, changed, row_values)
        result = mx.concatenate(
            (result[:row], row_values[None], result[row + 1 :]), axis=0
        )
    if prefix is not None:
        return (prefix, result, *output[2:])
    return result


def _steer_layer_selected(optimal_layer: Any, layer_num: int) -> bool:
    if optimal_layer == "all":
        return True
    values = optimal_layer if isinstance(optimal_layer, (list, tuple)) else [optimal_layer]
    for value in values:
        try:
            if int(value) == layer_num:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _steer_modes(config: dict) -> tuple[str, str]:
    from mia.workers.steer_worker import resolve_steer_modes

    try:
        return resolve_steer_modes(config)
    except ValueError as exc:
        raise MiaConfigurationError(str(exc)) from exc


def _wrap_steered_layer(
    layer: Any, layer_num: int, worker: Any, mx: Any, nn: Any
) -> Any:
    class MetalSteeredLayer(nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            object.__setattr__(self, "_mia_inner", wrapped)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            result = self._mia_inner(*args, **kwargs)
            return _steer_layer_output(worker, layer_num, result, mx)

    wrapped = MetalSteeredLayer(layer)
    object.__setattr__(wrapped, "_mia_metal_steer_wrapped", True)
    return wrapped


def install_metal_steering(worker: Any) -> bool:
    """Install activation steering on the non-paged vLLM-Metal MLX runner."""
    runner = getattr(worker, "model_runner", None)
    if runner is None or not _is_metal_runner(runner):
        return False
    if getattr(runner, _STEER_INSTALL_ATTR, False):
        return True
    if getattr(runner, "paged_attention_runtime", None) is not None:
        raise MiaConfigurationError(
            "The MIA Metal steering bridge currently requires the non-paged "
            "vLLM-Metal runner (VLLM_METAL_USE_PAGED_ATTENTION=0)."
        )

    model = getattr(runner, "model", None)
    if model is None:
        raise MiaConfigurationError("The vLLM-Metal runner has no loaded model.")
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise MiaConfigurationError(
            "The vLLM-Metal steering bridge requires mlx and mlx.nn."
        ) from exc

    worker._metal_steer_vectors = {}
    env_path = os.environ.get("MIA_STEER_CONFIG")
    layers = _model_layers(model)
    installed = 0
    for layer_num, layer in enumerate(layers):
        if getattr(layer, "_mia_metal_steer_wrapped", False):
            continue
        # Use the model's normal module assignment so MLX tracks the replacement.
        container = getattr(getattr(model, "language_model", model), "model", None)
        if container is None:
            container = getattr(model, "language_model", model)
        container.layers[layer_num] = _wrap_steered_layer(layer, layer_num, worker, mx, nn)
        installed += 1
    if not installed:
        raise MiaConfigurationError("No decoder layers were found for Metal steering.")

    original_prefill = runner._prefill_single

    def prefill_single(req_id: str, token_ids: list[int], sampling_params: Any, *args: Any,
                       **kwargs: Any) -> Any:
        config = _request_steer_config(sampling_params, env_path)
        state = {"configs": [config], "prefill": True, "sequential": False}
        token = _ACTIVE_STEER.set(state)
        try:
            return original_prefill(req_id, token_ids, sampling_params, *args, **kwargs)
        finally:
            _ACTIVE_STEER.reset(token)

    runner._prefill_single = prefill_single

    original_batched_decode = runner._batched_decode

    def batched_decode(decode_reqs: list, *args: Any, **kwargs: Any) -> Any:
        configs = [
            _request_steer_config(state.sampling_params, env_path)
            for _, state in decode_reqs
        ]
        token = _ACTIVE_STEER.set(
            {"configs": configs, "prefill": False, "sequential": False}
        )
        try:
            return original_batched_decode(decode_reqs, *args, **kwargs)
        finally:
            _ACTIVE_STEER.reset(token)

    runner._batched_decode = batched_decode

    original_sequential_decode = runner._sequential_decode

    def sequential_decode(decode_reqs: list, *args: Any, **kwargs: Any) -> Any:
        configs = [
            _request_steer_config(state.sampling_params, env_path)
            for _, state in decode_reqs
        ]
        token = _ACTIVE_STEER.set(
            {
                "configs": configs,
                "prefill": False,
                "sequential": True,
                "cursor": -1,
            }
        )
        try:
            return original_sequential_decode(decode_reqs, *args, **kwargs)
        finally:
            _ACTIVE_STEER.reset(token)

    runner._sequential_decode = sequential_decode
    setattr(runner, _STEER_INSTALL_ATTR, True)
    return True
