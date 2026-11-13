"""Q/K capture for vLLM-Metal's native MLX Llama execution path."""
from __future__ import annotations

import contextvars
import functools
import os
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

_ACTIVE_CAPTURE: contextvars.ContextVar["MetalQKCapture | None"] = (
    contextvars.ContextVar("mia_metal_qk_capture", default=None)
)
_ACTIVE_LAYER: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mia_metal_qk_layer", default=None
)
_LAYER_BY_INSTANCE: dict[int, int] = {}
_INSTALLED = False


def metal_plugin_requested() -> bool:
    """Whether vLLM was explicitly configured to load its Metal plugin."""
    return "metal" in {
        item.strip() for item in os.environ.get("VLLM_PLUGINS", "").split(",")
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).numpy()

    import mlx.core as mx

    value = value.astype(mx.float32)
    mx.eval(value)
    return np.asarray(value)


class MetalQKCapture:
    """One generation call's captured Llama Q/K values, keyed by request/layer."""

    def __init__(self, config: dict, selected_layers: set[int] | None = None):
        self.config = dict(config)
        self.selected_layers = selected_layers
        self.request_ids: tuple[str, ...] = ()
        self.sequential = False
        self._sequential_index = -1
        self._sequential_request: str | None = None
        self._captured: dict[str, dict[int, tuple[Any, Any]]] = {}

    @contextmanager
    def requests(self, request_ids: list[str], *, sequential: bool = False):
        previous = self.request_ids, self.sequential, self._sequential_index
        self.request_ids = tuple(str(req_id) for req_id in request_ids)
        self.sequential = sequential
        self._sequential_index = -1
        self._sequential_request = None
        try:
            yield
        finally:
            self.request_ids, self.sequential, self._sequential_index = previous
            self._sequential_request = None

    def record(self, layer_num: int | None, queries: Any, keys: Any) -> None:
        if layer_num is None:
            return
        if not self.request_ids:
            return

        if self.sequential:
            if layer_num == 0:
                self._sequential_index += 1
                if self._sequential_index >= len(self.request_ids):
                    return
                self._sequential_request = self.request_ids[self._sequential_index]
            req_ids = (self._sequential_request,)
        else:
            req_ids = self.request_ids

        if not req_ids or req_ids[0] is None:
            return
        if self.selected_layers is not None and layer_num not in self.selected_layers:
            return
        if len(queries.shape) != 4 or len(keys.shape) != 4:
            raise ValueError(
                "Metal Q/K capture expected [batch, heads, sequence, head_dim] "
                f"arrays; received Q={queries.shape}, K={keys.shape}"
            )
        batch_size = int(queries.shape[0])
        for batch_idx, req_id in enumerate(req_ids):
            if batch_idx >= batch_size:
                break
            query = queries[batch_idx, :, -1, :]
            key = keys[batch_idx]
            self._captured.setdefault(str(req_id), {})[layer_num] = (query, key)

    def probes(self, request_id: str) -> dict | None:
        layers = self._captured.get(str(request_id))
        if not layers:
            return None

        qk_cache = {}
        for layer_num, (query, key) in sorted(layers.items()):
            query_np = np.ascontiguousarray(_to_numpy(query).reshape(-1))
            key_np = _to_numpy(key)
            if key_np.ndim != 3:
                raise ValueError(
                    f"Metal Llama key cache at layer {layer_num} must have "
                    f"[heads, sequence, head_dim] shape, received {key_np.shape}"
                )
            key_np = np.ascontiguousarray(
                key_np.transpose(1, 0, 2).reshape(key_np.shape[1], -1)
            )
            layer_name = f"model.layers.{layer_num}.self_attn"
            qk_cache[layer_name] = {
                "q": [torch.from_numpy(query_np)],
                "k_all": [torch.from_numpy(key_np)],
                "layer_num": layer_num,
                "hookq_mode": "last_token",
            }
        return {"config": self.config, "qk_cache": qk_cache}


def _record_attention(queries: Any, keys: Any) -> None:
    capture = _ACTIVE_CAPTURE.get()
    if capture is not None:
        capture.record(_ACTIVE_LAYER.get(), queries, keys)


def _bind_model(runner: Any, attention_class: type) -> None:
    from vllm_metal.attention.patching import find_attn_attr, find_layers

    model = runner.model
    for layer_num, layer in enumerate(find_layers(model)):
        attr = find_attn_attr(layer)
        if attr is None:
            continue
        attention = getattr(layer, attr)
        while hasattr(attention, "_inner"):
            attention = attention._inner
        if isinstance(attention, attention_class):
            _LAYER_BY_INSTANCE[id(attention)] = layer_num


def _wrap_runner_method(runner_class: type, name: str, request_ids, sequential=False):
    original = getattr(runner_class, name, None)
    if original is None or getattr(original, "_mia_metal_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        capture = _ACTIVE_CAPTURE.get()
        if capture is None:
            return original(self, *args, **kwargs)
        ids = request_ids(args, kwargs)
        with capture.requests(ids, sequential=sequential):
            return original(self, *args, **kwargs)

    wrapped._mia_metal_wrapped = True
    setattr(runner_class, name, wrapped)


def install() -> None:
    """Patch MLX Llama SDPA and vLLM-Metal request boundaries once."""
    global _INSTALLED
    if _INSTALLED:
        return

    from mlx_lm.models import llama
    from vllm_metal.v1.model_runner import MetalModelRunner

    original_call = llama.Attention.__call__
    if not getattr(original_call, "_mia_metal_wrapped", False):
        @functools.wraps(original_call)
        def attention_call(self, *args, **kwargs):
            token = _ACTIVE_LAYER.set(_LAYER_BY_INSTANCE.get(id(self)))
            try:
                return original_call(self, *args, **kwargs)
            finally:
                _ACTIVE_LAYER.reset(token)

        attention_call._mia_metal_wrapped = True
        llama.Attention.__call__ = attention_call

    original_sdpa = llama.scaled_dot_product_attention
    if not getattr(original_sdpa, "_mia_metal_wrapped", False):
        @functools.wraps(original_sdpa)
        def scaled_dot_product_attention(queries, keys, values, *args, **kwargs):
            _record_attention(queries, keys)
            return original_sdpa(queries, keys, values, *args, **kwargs)

        scaled_dot_product_attention._mia_metal_wrapped = True
        llama.scaled_dot_product_attention = scaled_dot_product_attention

    original_load_model = MetalModelRunner.load_model
    if not getattr(original_load_model, "_mia_metal_wrapped", False):
        @functools.wraps(original_load_model)
        def load_model(runner, *args, **kwargs):
            result = original_load_model(runner, *args, **kwargs)
            _bind_model(runner, llama.Attention)
            return result

        load_model._mia_metal_wrapped = True
        MetalModelRunner.load_model = load_model

    _wrap_runner_method(
        MetalModelRunner,
        "_prefill_single",
        lambda args, kwargs: [kwargs.get("req_id", args[0] if args else "")],
    )
    _wrap_runner_method(
        MetalModelRunner,
        "_batched_decode",
        lambda args, kwargs: [
            req_id for req_id, _ in kwargs.get("decode_reqs", args[0] if args else [])
        ],
    )
    _wrap_runner_method(
        MetalModelRunner,
        "_sequential_decode",
        lambda args, kwargs: [
            req_id for req_id, _ in kwargs.get("decode_reqs", args[0] if args else [])
        ],
        sequential=True,
    )
    _wrap_runner_method(
        MetalModelRunner,
        "_start_paged_forward",
        lambda args, kwargs: (
            [req_id for req_id, _ in kwargs.get("decode_reqs", args[2] if len(args) > 2 else [])]
            + [
                request.req_id
                for request in kwargs.get("prefill_reqs", args[1] if len(args) > 1 else [])
            ]
        ),
    )
    _INSTALLED = True


@contextmanager
def capture_generation(config: dict, selected_layers: set[int] | None = None):
    """Activate one capture while the existing vLLM generation call runs."""
    capture = MetalQKCapture(config, selected_layers)
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _ACTIVE_CAPTURE.reset(token)


def attach_probes(
    outputs: list,
    capture: MetalQKCapture,
    *,
    output_layers: list[set[int] | None] | None = None,
    save_to_disk: bool = False,
    hook_dir: str | None = None,
    run_id: str | None = None,
) -> None:
    """Attach the standard MIA probe shape or persist it as a Q/K run."""
    payloads = []
    for index, output in enumerate(outputs):
        request_id = str(getattr(output, "request_id", index))
        probes = capture.probes(request_id)
        if probes is None:
            raise RuntimeError(
                f"vLLM-Metal completed request {request_id!r} without captured Llama Q/K "
                "tensors; verify the model uses mlx_lm's Llama attention and paged attention "
                "is disabled."
            )
        requested_layers = (
            output_layers[index]
            if output_layers is not None and index < len(output_layers)
            else None
        )
        if requested_layers is not None:
            probes["qk_cache"] = {
                name: entry
                for name, entry in probes["qk_cache"].items()
                if entry["layer_num"] in requested_layers
            }
        payloads.append(probes)
        if not save_to_disk:
            output.probes = probes

    if save_to_disk:
        if not hook_dir or not run_id:
            raise ValueError("save_to_disk requires hook_dir and run_id")
        from mia.workers._common import save_pt_atomic

        os.makedirs(os.path.join(hook_dir, run_id), exist_ok=True)
        if len(payloads) == 1:
            merged = payloads[0]
        else:
            from mia.llm import _merge_probes

            merged = _merge_probes(payloads)
        save_pt_atomic(merged, os.path.join(hook_dir, run_id, "qk.pt"))
    elif len(payloads) > 1:
        from mia.llm import _merge_probes

        outputs[0].probes = _merge_probes(payloads)
