"""Q/K capture at vLLM-Metal's MLX SDPA boundary.

The bridge observes the Q/K arrays prepared for Metal attention and reads the
post-forward keys from that layer's paged MLX cache. Generation remains owned
by vLLM-Metal; this module does not run an alternate model or attention path.
"""
from __future__ import annotations

import contextvars
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


_ACTIVE_SESSION: contextvars.ContextVar["MetalQKCaptureSession | None"] = (
    contextvars.ContextVar("mia_metal_qk_capture_session", default=None)
)
_ACTIVE_CALL: contextvars.ContextVar["_CaptureCall | None"] = contextvars.ContextVar(
    "mia_metal_qk_capture_call", default=None
)
_INSTALLED = False


@dataclass
class _CaptureCall:
    layer_num: int
    inner: Any
    queries: Any = None


@dataclass
class MetalQKCaptureSession:
    """Collect target-engine Q/K values for the active ``MiaLLM.generate`` call."""

    layer_to_heads: dict[int, list[int]]
    prompt_count: int
    captures: dict[int, dict[int, dict[str, Any]]] = field(default_factory=dict)
    config: dict[str, Any] | None = None
    _token: contextvars.Token | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "MetalQKCaptureSession":
        if _ACTIVE_SESSION.get() is not None:
            raise RuntimeError("Nested Metal Q/K capture sessions are not supported.")
        self._token = _ACTIVE_SESSION.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            _ACTIVE_SESSION.reset(self._token)
            self._token = None
        return False

    def capture_prepared_q(self, layer_num: int, queries: Any) -> None:
        call = _ACTIVE_CALL.get()
        if call is not None and call.layer_num == layer_num:
            call.queries = queries

    def capture_layer(self, layer_num: int, kv_cache: Any, context: Any) -> None:
        if layer_num not in self.layer_to_heads:
            return
        call = _ACTIVE_CALL.get()
        if call is None or call.layer_num != layer_num or call.queries is None:
            return

        import mlx.core as mx

        queries = call.queries
        cu_seqlens = _host_ints(context.cu_seqlens)
        context_lens = _host_ints(context.context_lens)
        block_tables = _host_rows(context.block_tables)
        if len(cu_seqlens) != len(context_lens) + 1:
            raise RuntimeError(
                "vLLM-Metal Q/K capture received inconsistent packed sequence metadata."
            )
        if len(block_tables) != len(context_lens):
            raise RuntimeError(
                "vLLM-Metal Q/K capture received inconsistent KV block-table metadata."
            )

        num_heads = int(queries.shape[1])
        head_dim = int(queries.shape[-1])
        cache = kv_cache.key_caches[layer_num]
        cache_block_size = int(kv_cache.block_size)
        layer_records = self.captures.setdefault(layer_num, {})

        for request_idx, seq_len in enumerate(context_lens):
            q_start, q_end = cu_seqlens[request_idx : request_idx + 2]
            q_len = q_end - q_start
            # A one-token decode step is not the prompt's final-token query.
            # The runnable example disables chunked prefill so its prompt is
            # captured in one call.
            if q_len <= 1:
                continue
            if seq_len < q_len:
                raise RuntimeError(
                    f"Q/K capture got sequence length {seq_len} shorter than query length {q_len}."
                )
            prompt_idx = len(layer_records)
            if prompt_idx >= self.prompt_count:
                raise RuntimeError(
                    "vLLM-Metal captured more prompt prefills than this capture session owns."
                )

            block_count = math.ceil(seq_len / cache_block_size)
            block_ids = block_tables[request_idx][:block_count]
            if len(block_ids) != block_count:
                raise RuntimeError(
                    f"Q/K capture has {len(block_ids)} KV blocks for a {seq_len}-token sequence."
                )
            if any(block_id < 0 or block_id >= int(cache.shape[0]) for block_id in block_ids):
                raise RuntimeError("vLLM-Metal Q/K capture found an out-of-range KV block id.")

            block_index = mx.array(block_ids, dtype=mx.int32)
            key_rows = cache[block_index].reshape(
                -1, int(cache.shape[-2]), int(cache.shape[-1])
            )[:seq_len]
            query_rows = queries[0, :, q_end - 1, :]
            mx.eval(query_rows, key_rows)
            q_tensor = _mlx_torch_tensor(query_rows)
            k_tensor = _mlx_torch_tensor(key_rows)

            heads_kv = int(k_tensor.shape[1])
            scale = getattr(call.inner, "scale", None)
            attention_multiplier = (
                float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
            )
            self.config = {
                "num_attention_heads": num_heads,
                "num_key_value_heads": heads_kv,
                "head_dim": head_dim,
                "hidden_size": num_heads * head_dim,
                "attention_multiplier": attention_multiplier,
            }
            layer_records[prompt_idx] = {
                "q": q_tensor.reshape(-1),
                "k_all": k_tensor.reshape(seq_len, -1),
                "layer_num": layer_num,
                "hookq_mode": "last_token",
            }

    def to_probes(self) -> dict[str, Any]:
        if self.config is None or not self.captures:
            raise RuntimeError(
                "vLLM-Metal returned no prompt Q/K captures. Ensure capture_qk is enabled, "
                "the selected model uses Metal SDPA, and prompt prefill is not chunked."
            )

        qk_cache: dict[str, dict[str, Any]] = {}
        for layer_num, records in sorted(self.captures.items()):
            if len(records) != self.prompt_count:
                continue
            ordered = [records[i] for i in range(self.prompt_count)]
            qk_cache[f"model.layers.{layer_num}.self_attn"] = {
                "q": [record["q"] for record in ordered],
                "k_all": [record["k_all"] for record in ordered],
                "layer_num": layer_num,
                "hookq_mode": "last_token",
            }
        expected_layers = set(self.layer_to_heads)
        present_layers = {int(entry["layer_num"]) for entry in qk_cache.values()}
        missing_layers = sorted(expected_layers - present_layers)
        if missing_layers:
            raise RuntimeError(
                f"vLLM-Metal did not capture configured Attention Tracker layers {missing_layers}."
            )
        return {"config": dict(self.config), "qk_cache": qk_cache}


def _host_ints(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def _host_rows(value: Any) -> list[list[int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[int(item) for item in row] for row in value]


def _mlx_torch_tensor(value: Any) -> torch.Tensor:
    import mlx.core as mx

    array = np.asarray(value.astype(mx.float32))
    if value.dtype == mx.float16:
        dtype = torch.float16
    elif value.dtype == mx.bfloat16:
        dtype = torch.bfloat16
    elif value.dtype == mx.float32:
        dtype = torch.float32
    else:
        raise TypeError(f"Unsupported vLLM-Metal Q/K dtype: {value.dtype}")
    return torch.from_numpy(array.copy()).to(dtype=dtype)


def install_qk_capture() -> None:
    """Patch the Metal SDPA helper boundary once, without changing its outputs."""
    global _INSTALLED
    if _INSTALLED:
        return

    import vllm_metal.attention.impls.sdpa as sdpa
    import vllm_metal.attention.impls.sdpa_wrapper as sdpa_wrapper

    original_prepare = sdpa.prepare_sdpa_qkv
    original_forward = sdpa_wrapper.sdpa_forward

    def prepare_with_capture(inner, x, context, *args, **kwargs):
        prepared = original_prepare(inner, x, context, *args, **kwargs)
        session = _ACTIVE_SESSION.get()
        call = _ACTIVE_CALL.get()
        if session is not None and call is not None:
            session.capture_prepared_q(call.layer_num, prepared[0])
        return prepared

    def forward_with_capture(inner, x, context, kv_cache, layer_idx, *args, **kwargs):
        session = _ACTIVE_SESSION.get()
        if session is None:
            return original_forward(
                inner, x, context, kv_cache, layer_idx, *args, **kwargs
            )

        call = _CaptureCall(int(layer_idx), inner)
        call_token = _ACTIVE_CALL.set(call)
        try:
            result = original_forward(
                inner, x, context, kv_cache, layer_idx, *args, **kwargs
            )
            if call.queries is not None:
                session.capture_layer(int(layer_idx), kv_cache, context)
            return result
        finally:
            _ACTIVE_CALL.reset(call_token)

    prepare_with_capture._mia_metal_original = original_prepare
    forward_with_capture._mia_metal_original = original_forward
    sdpa.prepare_sdpa_qkv = prepare_with_capture
    sdpa_wrapper.sdpa_forward = forward_with_capture
    _INSTALLED = True


def metal_plugin_requested() -> bool:
    """Whether the process has selected vLLM-Metal via its plugin setting."""
    selected = os.environ.get("VLLM_PLUGINS", "")
    return "metal" in {part.strip() for part in selected.split(",") if part.strip()}


def write_qk_run(hook_dir: str, run_id: str, probes: dict[str, Any]) -> None:
    """Write the regular single-rank Q/K artifact consumed by MIA's disk analyzer."""
    import os
    import tempfile

    run_dir = os.path.join(hook_dir, str(run_id), "tp_rank_0")
    os.makedirs(run_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="qk.", suffix=".tmp", dir=run_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(
                {
                    "qk_cache": probes["qk_cache"],
                    "config": probes["config"],
                    "meta": {"num_shareds": 1, "tp_rank": 0},
                },
                stream,
            )
        os.replace(temporary, os.path.join(run_dir, "qk.pt"))
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
