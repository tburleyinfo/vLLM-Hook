"""MIA capture and steering adapters for vLLM-Metal's MLX model runner."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


def is_metal_runner(runner: Any) -> bool:
    """Whether a runner is vLLM-Metal's MLX runner."""
    return type(runner).__module__.startswith("vllm_metal.")


def _model_layers(model: Any) -> list:
    root = getattr(model, "language_model", model)
    container = getattr(root, "model", root)
    layers = getattr(container, "layers", None)
    if layers is None:
        raise RuntimeError(
            f"vLLM-Metal model {type(model).__name__} exposes no transformer layers"
        )
    return layers


def _model_args(runner: Any) -> dict:
    args = getattr(runner, "model_args", None)
    if not isinstance(args, dict) or not args:
        raise RuntimeError("vLLM-Metal runner has no resolved model_args")
    heads = args.get("num_attention_heads", args.get("n_heads"))
    kv_heads = args.get("num_key_value_heads", args.get("n_kv_heads", heads))
    hidden = args.get("hidden_size")
    head_dim = args.get("head_dim")
    if head_dim is None and hidden and heads:
        head_dim = int(hidden) // int(heads)
    if not heads or not kv_heads or not head_dim:
        raise RuntimeError(
            "Cannot resolve Q/K capture dimensions from vLLM-Metal model_args"
        )
    return {
        "num_attention_heads": int(heads),
        "num_key_value_heads": int(kv_heads),
        "hidden_size": int(hidden or int(heads) * int(head_dim)),
        "head_dim": int(head_dim),
        "attention_multiplier": float(
            args.get("attention_multiplier", 1.0 / np.sqrt(int(head_dim)))
        ),
    }


def _requested_layers(spec: Any) -> set[int] | None:
    if spec is True:
        return None
    if isinstance(spec, dict):
        return {int(layer) for layer in spec}
    if isinstance(spec, (list, tuple)):
        return {int(layer) for layer in spec}
    return set()


def _to_cpu_tensor(value: Any, mx) -> torch.Tensor:
    mx.eval(value)
    compatible = np.asarray(value.astype(mx.float32))
    return torch.from_numpy(np.array(compatible, dtype=np.float16, copy=True))


def record_qk_rows(
    worker: Any,
    request_id: str,
    layer_num: int,
    module_name: str,
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    mode: str,
    extra: dict,
) -> None:
    """Append one MLX forward's Q/K rows in the existing eager worker format."""
    bucket = worker._disk_states if extra.get("save_to_disk") else worker._captured_states
    states = bucket.setdefault(str(request_id), {})
    entry = states.get(module_name)
    if entry is None:
        from mia.workers.qk_capture_worker import new_qk_entry

        entry = new_qk_entry(layer_num, mode)
        states[module_name] = entry
    entry["q"].append(q_rows[-1] if mode == "last_token" else q_rows)
    from mia.workers.qk_capture_worker import append_k_prefix

    append_k_prefix(entry, k_rows)


def _attention_name(layer: Any, layer_num: int) -> tuple[str, Any]:
    for attr in ("self_attn", "attention"):
        attn = getattr(layer, attr, None)
        if attn is not None:
            return attr, attn
    raise RuntimeError(
        f"vLLM-Metal transformer layer {layer_num} has no supported attention attribute"
    )


def _qk_capture(
    worker: Any,
    layer_num: int,
    inner: Any,
    x: Any,
    cache: Any,
    cache_offset: Any,
    mx,
) -> None:
    active = getattr(worker, "_mia_metal_active", None)
    if active is None or active["kind"] != "qk":
        return
    contexts = active.get("contexts") or []
    if not contexts:
        mapped = getattr(worker, "_mia_metal_cache_context", {}).get(id(cache))
        contexts = [mapped] if mapped is not None else []
    if not contexts:
        return
    args = worker._conf
    batch, seq_len, _ = x.shape
    if batch != len(contexts):
        raise RuntimeError(
            f"Metal Q/K capture has {batch} model rows for {len(contexts)} requests"
        )
    capture_rows = []
    for batch_idx, context in enumerate(contexts):
        if context is None:
            continue
        requested_layers = context["layers"]
        if requested_layers is not None and layer_num not in requested_layers:
            continue
        if context["phase"] == "prefill" and context["hooks_on"] == "decode":
            continue
        if context["phase"] == "decode" and context["hooks_on"] == "prefill":
            continue
        capture_rows.append((batch_idx, context))
    if not capture_rows:
        return

    q = inner.q_proj(x).reshape(
        batch, seq_len, args["num_attention_heads"], args["head_dim"]
    ).transpose(0, 2, 1, 3)
    k_current = inner.k_proj(x).reshape(
        batch, seq_len, args["num_key_value_heads"], args["head_dim"]
    ).transpose(0, 2, 1, 3)
    if cache is not None:
        q = inner.rope(q, offset=cache_offset)
        k_current = inner.rope(k_current, offset=cache_offset)
    else:
        q = inner.rope(q)
        k_current = inner.rope(k_current)

    cached_keys = getattr(cache, "keys", None) if cache is not None else None
    for batch_idx, context in capture_rows:
        if cached_keys is not None:
            total_len = int(context["seq_len"])
            k = cached_keys[batch_idx : batch_idx + 1, ..., :total_len, :]
        elif context["cache_offset"] == 0:
            k = k_current[batch_idx : batch_idx + 1]
        else:
            raise RuntimeError(
                "Metal Q/K capture cannot reconstruct a pre-existing prefix from this KV cache"
            )
        q_rows = q[batch_idx].transpose(1, 0, 2).reshape(seq_len, -1)
        k_rows = k[0].transpose(1, 0, 2).reshape(
            -1, args["num_key_value_heads"] * args["head_dim"]
        )
        record_qk_rows(
            worker,
            context["request_id"],
            layer_num,
            f"model.layers.{layer_num}.self_attn.attn",
            _to_cpu_tensor(q_rows, mx),
            _to_cpu_tensor(k_rows, mx),
            context["mode"],
            context["extra"],
        )


def _wrap_qk_attention(inner: Any, worker: Any, layer_num: int, mx, nn):
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "_worker", worker)
            object.__setattr__(self, "_layer_num", layer_num)
            object.__setattr__(self, "_mx", mx)
            object.__setattr__(self, "_mia_metal_qk_wrapper", True)

        def __call__(self, x, mask=None, cache=None, **kwargs):
            cache_offset = getattr(cache, "offset", 0) if cache is not None else 0
            result = self._inner(x, mask=mask, cache=cache, **kwargs)
            _qk_capture(
                self._worker,
                self._layer_num,
                self._inner,
                x,
                cache,
                cache_offset,
                self._mx,
            )
            return result

    return Wrapper()


def _request_extra(params: Any) -> dict:
    return dict(getattr(params, "extra_args", None) or {})


@contextmanager
def _active_request(worker: Any, active: dict | None):
    previous = getattr(worker, "_mia_metal_active", None)
    worker._mia_metal_active = active
    try:
        yield
    finally:
        worker._mia_metal_active = previous


def _wrap_runner_prefill(worker: Any, runner: Any) -> None:
    if getattr(runner, "_mia_metal_prefill_wrapped", False):
        return
    original = runner._prefill_single

    def wrapped(req_id, token_ids, sampling_params, *args, **kwargs):
        extra = _request_extra(sampling_params)
        requested = extra.get("output_qk")
        active = None
        if requested is not None:
            active = {
                "kind": "qk",
                "contexts": [
                    {
                        "request_id": str(req_id),
                        "layers": _requested_layers(requested),
                        "mode": extra.get("hookq_mode", "all_tokens"),
                        "hooks_on": extra.get("hooks_on", "prefill"),
                        "phase": "prefill",
                        "cache_offset": 0,
                        "seq_len": len(token_ids),
                        "extra": extra,
                    }
                ],
            }
        with _active_request(worker, active):
            return original(req_id, token_ids, sampling_params, *args, **kwargs)

    runner._prefill_single = wrapped
    original_batch = runner._batched_decode

    def batched(decode_reqs, *args, **kwargs):
        contexts = []
        for req_id, state in decode_reqs:
            extra = _request_extra(state.sampling_params)
            requested = extra.get("output_qk")
            contexts.append(
                None
                if requested is None
                else {
                    "request_id": str(req_id),
                    "layers": _requested_layers(requested),
                    "mode": extra.get("hookq_mode", "all_tokens"),
                    "hooks_on": extra.get("hooks_on", "prefill"),
                    "phase": "decode",
                    "cache_offset": len(state.token_ids) - 1,
                    "seq_len": len(state.token_ids),
                    "extra": extra,
                }
            )
        active = {"kind": "qk", "contexts": contexts}
        with _active_request(worker, active):
            return original_batch(decode_reqs, *args, **kwargs)

    runner._batched_decode = batched

    original_sequential = runner._sequential_decode

    def sequential(decode_reqs, *args, **kwargs):
        cache_context = {}
        for req_id, state in decode_reqs:
            extra = _request_extra(state.sampling_params)
            requested = extra.get("output_qk")
            if requested is None:
                continue
            context = {
                "request_id": str(req_id),
                "layers": _requested_layers(requested),
                "mode": extra.get("hookq_mode", "all_tokens"),
                "hooks_on": extra.get("hooks_on", "prefill"),
                "phase": "decode",
                "cache_offset": len(state.token_ids) - 1,
                "seq_len": len(state.token_ids),
                "extra": extra,
            }
            for layer_cache in state.cache:
                cache_context[id(layer_cache)] = context
        previous = getattr(worker, "_mia_metal_cache_context", {})
        worker._mia_metal_cache_context = cache_context
        with _active_request(worker, {"kind": "qk", "contexts": []}):
            try:
                return original_sequential(decode_reqs, *args, **kwargs)
            finally:
                worker._mia_metal_cache_context = previous

    runner._sequential_decode = sequential
    runner._mia_metal_prefill_wrapped = True


def install_metal_qk_capture(worker: Any) -> None:
    """Install request-scoped Q/K capture on a vLLM-Metal model."""
    import mlx.core as mx
    import mlx.nn as nn

    runner = worker.model_runner
    if getattr(runner, "_paged_attention_runtime", None) is not None:
        raise RuntimeError(
            "MIA's vLLM-Metal Q/K adapter requires the non-paged MLX path "
            "(VLLM_METAL_USE_PAGED_ATTENTION=0)"
        )
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("Cannot install Metal Q/K capture before the model is loaded")
    worker._conf = _model_args(runner)
    worker._captured_states = {}
    worker._disk_states = {}
    worker._artifact_tag = None
    worker._artifact_gran = None
    worker._artifact_gsize = None
    worker.hookq_mode = "all_tokens"
    layers = _model_layers(model)
    for layer_num, layer in enumerate(layers):
        attr, attention = _attention_name(layer, layer_num)
        if not hasattr(attention, "q_proj") or not hasattr(attention, "k_proj") \
                or not hasattr(attention, "rope"):
            raise RuntimeError(
                f"Metal Q/K capture does not recognize layer {layer_num} "
                f"({type(attention).__name__})"
            )
        if not getattr(attention, "_mia_metal_qk_wrapper", False):
            wrapper = _wrap_qk_attention(attention, worker, layer_num, mx, nn)
            setattr(layer, attr, wrapper)
    _wrap_runner_prefill(worker, runner)
    worker._mia_metal_qk = True


def _steer_config_for(context: dict, layer_num: int) -> dict | None:
    from mia.workers.steer_worker import _steer_targets_layer

    config = context.get("steer")
    if not isinstance(config, dict):
        return None
    if not _steer_targets_layer(config.get("optimal_layer", -1), layer_num):
        return None
    return config


def _steering_data(worker: Any, config: dict, mx) -> tuple[Any, Any]:
    from mia.workers.steer_worker import _load_steering_vector

    path = config.get("vector_path")
    if not path:
        raise ValueError("Metal steering requires steering.vector_path")
    cache = worker._mia_metal_steer_vectors
    data = cache.get(path)
    if data is None:
        raw = _load_steering_vector(path)
        direction = raw["dir"]
        if torch.is_tensor(direction):
            direction = direction.detach().cpu().numpy()
        direction = mx.array(np.asarray(direction, dtype=np.float32))
        average = raw.get("avg_proj")
        if torch.is_tensor(average):
            average = average.detach().cpu().numpy()
        average = None if average is None else mx.array(np.asarray(average, dtype=np.float32))
        data = (direction, average)
        cache[path] = data
    return data


def _apply_steering(worker: Any, layer_num: int, output: Any, cache: Any, mx) -> Any:
    active = getattr(worker, "_mia_metal_active", None)
    if active is None or active["kind"] != "steer":
        cache_context = getattr(worker, "_mia_metal_cache_context", {})
        context = cache_context.get(id(cache))
        contexts = [context] if context is not None else []
    else:
        contexts = active["contexts"]
    if not contexts:
        return output
    from mia.workers.steer_worker import resolve_steer_modes, steer_span

    if len(output.shape) != 3 or output.shape[0] != len(contexts):
        raise RuntimeError(
            f"Metal steering layer {layer_num} output shape {tuple(output.shape)} "
            f"does not match {len(contexts)} active request(s)"
        )
    result = output
    for batch_idx, context in enumerate(contexts):
        config = _steer_config_for(context, layer_num)
        if config is None:
            continue
        try:
            phase, positions = resolve_steer_modes(config)
        except ValueError as exc:
            warned = getattr(worker, "_bad_cfg_warned", 0)
            if warned < 4:
                worker._bad_cfg_warned = warned + 1
                print(
                    f"[steer] ignoring request with invalid steer config: {exc}",
                    flush=True,
                )
            continue
        is_prefill = context["is_prefill"]
        start, end = 0, int(output.shape[1])
        span = steer_span(phase, positions, is_prefill, True, start, end)
        if span is None:
            continue
        direction, average = _steering_data(worker, config, mx)
        row = result[batch_idx : batch_idx + 1]
        lo, hi = span
        selected = row[:, lo:hi, :]
        method = config.get("method", "adjust_rs")
        if method == "add_vector":
            coefficient = float(config.get("coefficient", 0))
            steered = selected + coefficient * direction
        elif method == "adjust_rs":
            if average is None:
                raise ValueError(
                    "Metal steering method 'adjust_rs' requires avg_proj in the vector file"
                )
            coefficient = average - mx.sum(selected * direction, axis=-1, keepdims=True)
            steered = selected + coefficient * direction
        else:
            raise ValueError(f"Unknown steering method: {method}")
        steered_row = mx.concatenate(
            (row[:, :lo, :], steered, row[:, hi:, :]), axis=1
        )
        result = mx.concatenate(
            (result[:batch_idx], steered_row, result[batch_idx + 1 :]), axis=0
        )
    return result


def _wrap_steering_layer(inner: Any, worker: Any, layer_num: int, mx, nn):
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "_worker", worker)
            object.__setattr__(self, "_layer_num", layer_num)
            object.__setattr__(self, "_mx", mx)
            object.__setattr__(self, "_mia_metal_steer_wrapper", True)

        def __call__(self, x, mask=None, cache=None, **kwargs):
            result = self._inner(x, mask=mask, cache=cache, **kwargs)
            return _apply_steering(
                self._worker, self._layer_num, result, cache, self._mx
            )

    return Wrapper()


def _sampling_context(
    worker: Any, req_id: str, params: Any, is_prefill: bool
) -> dict:
    from mia.workers.steer_worker import _resolve_steer_config

    extra = _request_extra(params)
    return {
        "request_id": str(req_id),
        "steer": _resolve_steer_config(
            extra.get("steer"), getattr(worker, "_env_config_path", None)
        ),
        "is_prefill": bool(is_prefill),
    }


def _wrap_runner_steering(worker: Any, runner: Any) -> None:
    if getattr(runner, "_mia_metal_steering_wrapped", False):
        return
    original_prefill = runner._prefill_single

    def prefill(req_id, token_ids, sampling_params, *args, **kwargs):
        context = _sampling_context(worker, req_id, sampling_params, True)
        with _active_request(worker, {"kind": "steer", "contexts": [context]}):
            return original_prefill(req_id, token_ids, sampling_params, *args, **kwargs)

    runner._prefill_single = prefill

    original_batch = runner._batched_decode

    def batched(decode_reqs, *args, **kwargs):
        contexts = [
            _sampling_context(worker, req_id, state.sampling_params, False)
            for req_id, state in decode_reqs
        ]
        with _active_request(worker, {"kind": "steer", "contexts": contexts}):
            return original_batch(decode_reqs, *args, **kwargs)

    runner._batched_decode = batched

    original_sequential = runner._sequential_decode

    def sequential(decode_reqs, *args, **kwargs):
        cache_context = {}
        for req_id, state in decode_reqs:
            context = _sampling_context(worker, req_id, state.sampling_params, False)
            for layer_cache in state.cache:
                cache_context[id(layer_cache)] = context
        previous = getattr(worker, "_mia_metal_cache_context", {})
        worker._mia_metal_cache_context = cache_context
        try:
            return original_sequential(decode_reqs, *args, **kwargs)
        finally:
            worker._mia_metal_cache_context = previous

    runner._sequential_decode = sequential
    runner._mia_metal_steering_wrapped = True


def install_metal_steering(worker: Any) -> None:
    """Install activation steering on vLLM-Metal's MLX transformer layers."""
    import mlx.core as mx
    import mlx.nn as nn

    runner = worker.model_runner
    if getattr(runner, "_paged_attention_runtime", None) is not None:
        raise RuntimeError(
            "MIA's vLLM-Metal steering adapter requires the non-paged MLX path "
            "(VLLM_METAL_USE_PAGED_ATTENTION=0)"
        )
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("Cannot install Metal steering before the model is loaded")
    worker._mia_metal_steer_vectors = {}
    worker._env_config_path = os.environ.get("MIA_STEER_CONFIG")
    layers = _model_layers(model)
    for layer_num, layer in enumerate(layers):
        if not getattr(layer, "_mia_metal_steer_wrapper", False):
            wrapper = _wrap_steering_layer(layer, worker, layer_num, mx, nn)
            layers[layer_num] = wrapper
    _wrap_runner_steering(worker, runner)
    worker._mia_metal_steer = True
