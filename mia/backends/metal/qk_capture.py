"""Q/K capture adapter for vLLM-Metal's MLX Granite inference path."""
from __future__ import annotations

import contextlib
import contextvars
import functools
from typing import Any

import numpy as np
import torch

from mia.errors import MiaConfigurationError
from mia.workers._common import quant_clone, resolve_capture_quant
from mia.workers.qk_capture_worker import append_k_prefix, new_qk_entry

_ACTIVE_REQUESTS: contextvars.ContextVar[tuple["_CaptureRequest", ...]] = (
    contextvars.ContextVar("mia_metal_qk_requests", default=())
)
_ACTIVE_ATTENTION: contextvars.ContextVar["_AttentionFrame | None"] = (
    contextvars.ContextVar("mia_metal_qk_attention", default=None)
)
_PATCHED_CLASSES: dict[type, Any] = {}


def _mlx_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    import mlx.core as mx

    mx.eval(value)
    if value.dtype == mx.bfloat16:
        return np.asarray(value.astype(mx.float32))
    return np.asarray(value)


def format_qk_capture(
    queries: np.ndarray,
    keys: np.ndarray,
    batch_index: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert rotated ``[B,H,S,D]`` arrays to the existing probe row contract."""
    if queries.ndim != 4 or keys.ndim != 4:
        raise ValueError(
            "Metal Q/K capture expects 4-D [batch, heads, tokens, head_dim] "
            f"arrays; received q={queries.shape}, k={keys.shape}"
        )
    if batch_index < 0 or batch_index >= queries.shape[0] or batch_index >= keys.shape[0]:
        raise IndexError(f"capture batch index {batch_index} is out of range")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(f"Q/K head dimensions differ: {queries.shape[-1]} and {keys.shape[-1]}")

    q = queries[batch_index]
    if mode == "last_token":
        q = q[:, -1, :]
        q = q.reshape(-1)
    elif mode == "all_tokens":
        q = q.transpose(1, 0, 2).reshape(q.shape[1], -1)
    else:
        raise MiaConfigurationError(f"Unknown hookq_mode for Metal capture: {mode!r}")

    k = keys[batch_index].transpose(1, 0, 2).reshape(keys.shape[2], -1)
    return (
        torch.from_numpy(np.array(q, copy=True)).contiguous(),
        torch.from_numpy(np.array(k, copy=True)).contiguous(),
    )


class _CaptureRequest:
    def __init__(self, worker, req_id: str, extra: dict, batch_index: int, phase: str):
        self.worker = worker
        self.req_id = str(req_id)
        self.extra = extra
        self.batch_index = batch_index
        self.phase = phase

    def enabled_for(self, layer_num: int) -> bool:
        if self.extra.get("output_qk") is None:
            return False
        hooks_on = self.extra.get("hooks_on", "prefill")
        if hooks_on != "both":
            if hooks_on == "prefill" and self.phase != "prefill":
                return False
            if hooks_on == "decode" and self.phase == "prefill":
                return False
        output_spec = self.extra.get("output_qk")
        if isinstance(output_spec, dict):
            return any(int(layer) == layer_num for layer in output_spec)
        if isinstance(output_spec, (list, tuple)):
            return layer_num in {int(layer) for layer in output_spec}
        return True

    def record(self, layer_num: int, module_name: str, queries, keys) -> None:
        q_np, k_np = _mlx_numpy(queries), _mlx_numpy(keys)
        q, k = format_qk_capture(
            q_np, k_np, self.batch_index, self.extra.get("hookq_mode", "all_tokens")
        )
        q, q_scale, q_meta = self._quantize(q)
        k, k_scale, k_meta = self._quantize(k)
        bucket_name = "_disk_states" if self.extra.get("save_to_disk") else "_captured_states"
        bucket = getattr(self.worker, bucket_name)
        layer_states = bucket.setdefault(self.req_id, {})
        entry = layer_states.get(module_name)
        if entry is None:
            entry = new_qk_entry(
                layer_num,
                self.extra.get("hookq_mode", "all_tokens"),
                q_qmeta=q_meta,
                k_qmeta=k_meta,
            )
            layer_states[module_name] = entry
        entry["q"].append(q)
        append_k_prefix(entry, k)
        if q_meta is not None:
            entry["_q_scale"].append(q_scale)
            entry["_k_all_scale"].append(k_scale)

    def _quantize(self, value: torch.Tensor):
        return quant_clone(
            value,
            getattr(self.worker, "_artifact_tag", None),
            getattr(self.worker, "_artifact_gran", None),
            getattr(self.worker, "_artifact_gsize", 128),
        )


class _RopeCaptureProxy:
    def __init__(self, rope, layer_num: int):
        self._rope = rope
        self._layer_num = layer_num

    def __call__(self, *args, **kwargs):
        output = self._rope(*args, **kwargs)
        frame = _ACTIVE_ATTENTION.get()
        if frame is not None and frame.layer_num == self._layer_num:
            frame.rope_outputs.append(output)
        return output


class _AttentionFrame:
    def __init__(self, layer_num: int, module_name: str, requests: tuple[_CaptureRequest, ...]):
        self.layer_num = layer_num
        self.module_name = module_name
        self.requests = tuple(req for req in requests if req.enabled_for(layer_num))
        self.rope_outputs: list[Any] = []
        self.cached_keys = None

    def capture(self) -> None:
        if not self.requests:
            return
        if len(self.rope_outputs) < 2:
            raise MiaConfigurationError(
                f"Metal attention layer {self.layer_num} did not expose post-RoPE Q and K"
            )
        if self.cached_keys is None and any(
            request.phase == "decode" for request in self.requests
        ):
            raise MiaConfigurationError(
                f"Metal KV cache did not expose the full key prefix for decode "
                f"capture at layer {self.layer_num}"
            )
        queries, current_keys = self.rope_outputs[:2]
        keys = self.cached_keys if self.cached_keys is not None else current_keys
        for request in self.requests:
            request.record(self.layer_num, self.module_name, queries, keys)


def _patch_attention_class(attention_class: type) -> None:
    if attention_class in _PATCHED_CLASSES:
        return
    original = attention_class.__call__

    @functools.wraps(original)
    def wrapped(attention, *args, **kwargs):
        layer_num = getattr(attention, "_mia_metal_qk_layer", None)
        requests = _ACTIVE_REQUESTS.get()
        if layer_num is None or not requests:
            return original(attention, *args, **kwargs)

        module_name = getattr(attention, "_mia_metal_qk_name")
        frame = _AttentionFrame(layer_num, module_name, requests)
        if not frame.requests:
            return original(attention, *args, **kwargs)

        frame_token = _ACTIVE_ATTENTION.set(frame)
        cache = kwargs.get("cache", args[2] if len(args) > 2 else None)
        original_update = (
            getattr(cache, "update_and_fetch", None) if cache is not None else None
        )
        if callable(original_update):
            def update_and_fetch(*update_args, **update_kwargs):
                result = original_update(*update_args, **update_kwargs)
                if isinstance(result, tuple) and result:
                    frame.cached_keys = result[0]
                return result

            try:
                cache.update_and_fetch = update_and_fetch
            except (AttributeError, TypeError) as exc:
                _ACTIVE_ATTENTION.reset(frame_token)
                raise MiaConfigurationError(
                    f"Cannot observe the Metal KV cache for attention layer {layer_num}"
                ) from exc

        try:
            result = original(attention, *args, **kwargs)
            frame.capture()
            return result
        finally:
            if callable(original_update):
                cache.update_and_fetch = original_update
            _ACTIVE_ATTENTION.reset(frame_token)

    _PATCHED_CLASSES[attention_class] = original
    attention_class.__call__ = wrapped


def _install_attention_wrappers(model) -> int:
    from vllm_metal.attention.patching import find_attn_attr, find_layers

    patched = 0
    for layer_num, layer in enumerate(find_layers(model)):
        attr = find_attn_attr(layer)
        if attr is None:
            continue
        attention = getattr(layer, attr)
        inner = getattr(attention, "_inner", attention)
        if not all(hasattr(inner, name) for name in ("q_proj", "k_proj", "rope")):
            raise MiaConfigurationError(
                f"Metal Q/K capture cannot instrument layer {layer_num} "
                f"({type(inner).__name__})"
            )
        if not isinstance(getattr(inner, "rope"), _RopeCaptureProxy):
            object.__setattr__(inner, "rope", _RopeCaptureProxy(inner.rope, layer_num))
        object.__setattr__(inner, "_mia_metal_qk_layer", layer_num)
        object.__setattr__(
            inner, "_mia_metal_qk_name", f"model.layers.{layer_num}.self_attn.attn"
        )
        _patch_attention_class(type(inner))
        patched += 1
    if patched == 0:
        raise MiaConfigurationError("No compatible Metal SDPA attention layers were found")
    return patched


@contextlib.contextmanager
def _capture_scope(worker, requests: list[tuple[str, Any]], phase: str):
    contexts = []
    for batch_index, (req_id, params) in enumerate(requests):
        extra = getattr(params, "extra_args", None) or {}
        if extra.get("output_qk") is not None:
            contexts.append(_CaptureRequest(worker, req_id, extra, batch_index, phase))
    token = _ACTIVE_REQUESTS.set(tuple(contexts))
    try:
        yield
    finally:
        _ACTIVE_REQUESTS.reset(token)


def install_metal_qk_capture(worker) -> None:
    """Attach existing Q/K request capture to the vLLM-Metal legacy runner."""
    runner = worker.model_runner
    if not type(runner).__module__.startswith("vllm_metal."):
        raise MiaConfigurationError("Metal Q/K adapter received a non-Metal model runner")
    model = getattr(runner, "_forward_model", None)
    if model is None:
        model = getattr(runner, "model", None)
    if model is None:
        raise MiaConfigurationError("vLLM-Metal model is unavailable for Q/K capture install")

    text_config = getattr(runner.model_config, "hf_config", None)
    text_config = getattr(text_config, "text_config", text_config)
    model_args = getattr(runner, "model_args", {})
    num_heads = int(
        model_args.get("num_attention_heads") or text_config.num_attention_heads
    )
    num_kv_heads = int(
        model_args.get("num_key_value_heads")
        or getattr(text_config, "num_key_value_heads", num_heads)
    )
    hidden_size = int(model_args.get("hidden_size") or text_config.hidden_size)
    head_dim = int(model_args.get("head_dim") or hidden_size // num_heads)
    multiplier = float(
        model_args.get("attention_multiplier") or (1 / (head_dim**0.5))
    )
    worker._conf = {
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "attention_multiplier": multiplier,
    }
    worker._captured_states = {}
    worker._disk_states = {}
    worker.hookq_mode = "all_tokens"
    worker._artifact_tag, worker._artifact_gran, worker._artifact_gsize = (
        resolve_capture_quant("qk")
    )
    attention_layers = _install_attention_wrappers(model)

    original_prefill = runner._prefill_single

    @functools.wraps(original_prefill)
    def prefill(req_id, token_ids, sampling_params, *args, **kwargs):
        with _capture_scope(worker, [(req_id, sampling_params)], "prefill"):
            return original_prefill(req_id, token_ids, sampling_params, *args, **kwargs)

    runner._prefill_single = prefill

    original_sequential = runner._sequential_decode

    @functools.wraps(original_sequential)
    def sequential(decode_reqs, *args, **kwargs):
        token_ids = []
        logprobs = []
        for req_id, state in decode_reqs:
            with _capture_scope(
                worker, [(req_id, state.sampling_params)], "decode"
            ):
                result = original_sequential([(req_id, state)], *args, **kwargs)
            token_ids.extend(result.token_ids)
            logprobs.append(result.logprobs)
        from vllm_metal.v1.sampling_batch import SamplingBatch, _SamplingResult

        return _SamplingResult(
            token_ids,
            SamplingBatch.merge_logprobs_rows(logprobs),
        )

    runner._sequential_decode = sequential

    original_batched = runner._batched_decode

    @functools.wraps(original_batched)
    def batched(decode_reqs, *args, **kwargs):
        requests = [(req_id, state.sampling_params) for req_id, state in decode_reqs]
        with _capture_scope(worker, requests, "decode"):
            return original_batched(decode_reqs, *args, **kwargs)

    runner._batched_decode = batched

    print(f"[mia/metal] installed Q/K capture on {attention_layers} "
          "attention layers", flush=True)
