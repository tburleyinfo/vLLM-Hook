"""Q/K capture adapter for vLLM-Metal's MLX attention modules."""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mia.errors import MiaConfigurationError

try:
    import mlx.nn as _mlx_nn
except ImportError:
    _MLXModule = object
else:
    _MLXModule = _mlx_nn.Module


@dataclass
class _Request:
    request_id: str
    extra: dict
    phase: str = "prefill"


@dataclass
class _AttentionCall:
    layer_num: int
    query_heads: int
    kv_heads: int
    query: Any = None
    key: Any = None
    rope_calls: int = 0


def _mlx_rows(value: Any) -> torch.Tensor:
    """Copy MLX or NumPy data to CPU float32 without converting bfloat16 via NumPy."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))

    import mlx.core as mx

    mx.eval(value)
    return torch.from_numpy(
        np.array(np.asarray(value.astype(mx.float32)), dtype=np.float32, copy=True)
    )


def _flat_heads(value: Any) -> torch.Tensor:
    rows = _mlx_rows(value)
    if rows.ndim != 4 or rows.shape[0] != 1:
        raise MiaConfigurationError(
            "Metal Q/K capture expects a single-request MLX attention tensor "
            "with shape [1, heads, tokens, head_dim]. Use max_num_seqs=1."
        )
    return rows[0].permute(1, 0, 2).reshape(rows.shape[2], -1).contiguous()


def _layer_is_selected(output_spec: Any, layer_num: int) -> bool:
    if isinstance(output_spec, dict):
        return any(int(layer) == layer_num for layer in output_spec)
    if isinstance(output_spec, (list, tuple, set)):
        return any(int(layer) == layer_num for layer in output_spec)
    return bool(output_spec)


def _find_layers(model: Any) -> list:
    root = getattr(model, "language_model", model)
    container = getattr(root, "model", root)
    layers = getattr(container, "layers", None)
    if layers is None:
        layers = getattr(root, "layers", None)
    if layers is None:
        raise MiaConfigurationError(
            f"Cannot locate MLX transformer layers in {type(model).__name__}."
        )
    return list(layers)


def _cached_keys(cache: Any, current_keys: Any) -> Any:
    keys = getattr(cache, "keys", None)
    if keys is None:
        return current_keys
    if getattr(keys, "ndim", None) != 4:
        raise MiaConfigurationError(
            f"Unsupported MLX KV-cache key shape: {getattr(keys, 'shape', None)!r}."
        )
    offset = int(getattr(cache, "offset", keys.shape[-2]))
    if offset > keys.shape[-2]:
        raise MiaConfigurationError(
            "Metal Q/K capture cannot reconstruct keys beyond a rotating KV-cache "
            "window. Increase max_model_len only within the model's full-key window."
        )
    return keys[..., :offset, :]


class _RopeCapture(_MLXModule):
    def __init__(self, inner: Any, backend: "MetalQKCapture", layer_num: int):
        if _MLXModule is not object:
            super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_layer_num", layer_num)

    def __call__(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        output = self._inner(value, *args, **kwargs)
        call = self._backend._attention_call.get()
        if call is not None and call.layer_num == self._layer_num:
            is_query = call.rope_calls == 0
            call.rope_calls += 1
            if is_query:
                call.query = output
            else:
                call.key = output
        return output


class _AttentionCapture(_MLXModule):
    def __init__(
        self,
        inner: Any,
        backend: "MetalQKCapture",
        layer_num: int,
        rope_attr: str,
    ):
        if _MLXModule is not object:
            super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_layer_num", layer_num)
        object.__setattr__(self, "_rope_attr", rope_attr)
        rope = getattr(inner, rope_attr)
        if not isinstance(rope, _RopeCapture):
            setattr(inner, rope_attr, _RopeCapture(rope, backend, layer_num))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        cache = kwargs.get("cache")
        if cache is None and len(args) >= 3:
            cache = args[2]
        request = self._backend._request.get()
        if request is None:
            cached_request = self._backend._cache_requests.get(id(cache))
            if cached_request is not None:
                request = _Request(
                    cached_request.request_id, cached_request.extra, phase="decode"
                )
        if request is None:
            return self._inner(*args, **kwargs)

        output_spec = request.extra.get("output_qk")
        hooks_on = request.extra.get("hooks_on", "prefill")
        if (
            output_spec is None
            or not _layer_is_selected(output_spec, self._layer_num)
            or hooks_on not in ("both", request.phase)
        ):
            return self._inner(*args, **kwargs)

        query_heads = int(getattr(self._inner, "n_heads"))
        kv_heads = int(getattr(self._inner, "n_kv_heads", query_heads))
        call = _AttentionCall(self._layer_num, query_heads, kv_heads)
        token = self._backend._attention_call.set(call)
        try:
            result = self._inner(*args, **kwargs)
        finally:
            self._backend._attention_call.reset(token)

        if call.query is None or call.key is None:
            raise MiaConfigurationError(
                f"Metal Q/K capture did not observe the expected two RoPE calls at "
                f"layer {self._layer_num} ({type(self._inner).__name__})."
            )
        query = _flat_heads(call.query)
        if request.extra.get("hookq_mode", "all_tokens") == "last_token":
            query = query[-1]
        key = _flat_heads(_cached_keys(cache, call.key))
        self._backend._record(request, self._layer_num, query, key)
        return result


class MetalQKCapture:
    """Installs request-scoped Q/K taps while leaving MLX model computation intact."""

    def __init__(self, worker: Any):
        self.worker = worker
        self.runner = worker.model_runner
        self._request: contextvars.ContextVar[_Request | None] = contextvars.ContextVar(
            f"mia_metal_qk_request_{id(self)}", default=None
        )
        self._attention_call: contextvars.ContextVar[_AttentionCall | None] = (
            contextvars.ContextVar(f"mia_metal_qk_call_{id(self)}", default=None)
        )
        self._cache_requests: dict[int, _Request] = {}
        self._install_runner_wrappers()

    @staticmethod
    def _request_extra(sampling_params: Any) -> dict:
        extra = getattr(sampling_params, "extra_args", None)
        return dict(extra) if isinstance(extra, dict) else {}

    def _install_runner_wrappers(self) -> None:
        original_prefill = self.runner._prefill_single

        def prefill_single(req_id, token_ids, sampling_params, *args, **kwargs):
            request = _Request(str(req_id), self._request_extra(sampling_params))
            token = self._request.set(request)
            try:
                result = original_prefill(
                    req_id, token_ids, sampling_params, *args, **kwargs
                )
            finally:
                self._request.reset(token)
            for layer_cache in result[1]:
                self._cache_requests[id(layer_cache)] = _Request(
                    request.request_id, request.extra, phase="decode"
                )
            return result

        self.runner._prefill_single = prefill_single

    def install(self) -> None:
        if getattr(self.runner, "_paged_attention_runtime", None) is not None:
            raise MiaConfigurationError(
                "The vLLM-Metal Q/K adapter currently requires the MLX cache path. "
                "Set VLLM_METAL_USE_PAGED_ATTENTION=0."
            )

        args = getattr(self.runner, "model_args", {})
        query_heads = int(args["num_attention_heads"])
        kv_heads = int(args.get("num_key_value_heads", query_heads))
        hidden_size = int(args["hidden_size"])
        head_dim = int(args.get("head_dim") or hidden_size // query_heads)
        self.worker._conf = {
            "num_attention_heads": query_heads,
            "num_key_value_heads": kv_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "attention_multiplier": float(args.get("attention_multiplier", head_dim**-0.5)),
        }
        self.worker._qk_shard = None
        self.worker._should_capture = True
        self.worker.hookq_mode = "all_tokens"
        self.worker._captured_states = {}
        self.worker._disk_states = {}
        self.worker._qk_refused = {}
        self.worker._artifact_tag = None
        self.worker._artifact_gran = None
        self.worker._artifact_gsize = None

        try:
            layers = _find_layers(self.runner.model)
        except MiaConfigurationError as exc:
            raise MiaConfigurationError(
                f"Cannot install Metal Q/K capture for {type(self.runner.model).__name__}: {exc}"
            ) from exc

        patched = 0
        for layer_num, layer in enumerate(layers):
            attn_attr = next(
                (name for name in ("self_attn", "attention") if hasattr(layer, name)),
                None,
            )
            if attn_attr is None:
                continue
            inner = getattr(layer, attn_attr)
            rope_attr = next(
                (name for name in ("rope", "rotary_emb") if hasattr(inner, name)),
                None,
            )
            if rope_attr is None:
                continue
            if not hasattr(inner, "n_heads"):
                raise MiaConfigurationError(
                    f"Metal Q/K capture requires n_heads on layer {layer_num} "
                    f"({type(inner).__name__})."
                )
            if isinstance(inner, _AttentionCapture):
                continue
            setattr(
                layer,
                attn_attr,
                _AttentionCapture(inner, self, layer_num, rope_attr),
            )
            patched += 1
        if not patched:
            raise MiaConfigurationError(
                "Metal Q/K capture found no RoPE-based attention layers in the loaded model."
            )

    def _record(
        self, request: _Request, layer_num: int, query: torch.Tensor, key: torch.Tensor
    ) -> None:
        extra = request.extra
        mode = extra.get("hookq_mode", "all_tokens")
        if mode not in ("all_tokens", "last_token"):
            raise MiaConfigurationError(f"Unknown hookq_mode={mode!r}.")
        destination = self.worker._disk_states if extra.get("save_to_disk") else (
            self.worker._captured_states
        )
        layer_states = destination.setdefault(request.request_id, {})
        entry = layer_states.setdefault(
            f"layers.{layer_num}.self_attn",
            {
                "q": [],
                "k_all": [],
                "k_prefix_ends": [],
                "layer_num": layer_num,
                "hookq_mode": mode,
            },
        )
        entry["q"].append(query)
        from mia.workers.qk_capture_worker import append_k_prefix

        append_k_prefix(entry, key)

    def clear(self, request_id: str) -> None:
        for cache_id, request in list(self._cache_requests.items()):
            if request.request_id == str(request_id):
                del self._cache_requests[cache_id]


def install_metal_qk_capture(worker: Any) -> MetalQKCapture:
    """Install Q/K capture on a Metal worker using the existing worker API."""
    backend = MetalQKCapture(worker)
    backend.install()
    return backend
