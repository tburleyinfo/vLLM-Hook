"""Q/K capture for vLLM-Metal's MLX-backed Qwen2 execution path."""
from __future__ import annotations

import contextvars
import pickle
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard as zstd

from mia.errors import MiaConfigurationError

_ACTIVE_CAPTURE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "mia_metal_qk_capture", default=None
)
_ACTIVE_LAYER: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "mia_metal_qk_layer", default=None
)
_LOCK = threading.RLock()
_INSTALLED_WORKER = False
_COMPRESSOR = zstd.ZstdCompressor(level=1)


def _capture_spec(extra: dict, layer: int) -> bool:
    requested = extra.get("output_qk")
    if requested is None:
        return False
    if extra.get("qk_capture", "qk") != "qk":
        raise MiaConfigurationError(
            "Metal Q/K capture currently emits raw Q/K probes; "
            "qk_capture='score' is not implemented."
        )
    if isinstance(requested, dict):
        return layer in {int(key) for key in requested}
    if isinstance(requested, (list, tuple)):
        return layer in {int(value) for value in requested}
    return bool(requested)


def _array_to_tensor(value: Any) -> torch.Tensor:
    """Convert an evaluated MLX array through float32 for bfloat16 compatibility."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.asarray(value, dtype=np.float32).copy()).contiguous()

    import mlx.core as mx

    return torch.from_numpy(
        np.asarray(value.astype(mx.float32), dtype=np.float32).copy()
    ).contiguous()


def _observe_qk(queries: Any, keys: Any) -> None:
    active = _ACTIVE_CAPTURE.get()
    layer = _ACTIVE_LAYER.get()
    if active is None or layer is None:
        return

    request_contexts = active["requests"]
    q_shape = tuple(queries.shape)
    k_shape = tuple(keys.shape)
    if len(q_shape) != 4 or len(k_shape) != 4:
        raise MiaConfigurationError(
            "Metal Q/K capture expected Q and K arrays shaped [batch, heads, seq, dim], "
            f"got {q_shape} and {k_shape}."
        )
    if q_shape[0] != len(request_contexts) or k_shape[0] != len(request_contexts):
        raise MiaConfigurationError(
            "Metal Q/K capture could not align attention batch rows to requests: "
            f"batch={q_shape[0]}, requests={len(request_contexts)}."
        )

    for row, request in enumerate(request_contexts):
        extra = request["extra"]
        if not _capture_spec(extra, layer):
            continue
        hooks_on = extra.get("hooks_on", "prefill")
        if hooks_on not in ("both", active["phase"]):
            continue

        mode = extra.get("hookq_mode", "all_tokens")
        q_value = queries[row]
        if mode == "last_token":
            q_value = q_value[:, -1, :]
        elif mode != "all_tokens":
            raise MiaConfigurationError(
                f"Unsupported hookq_mode={mode!r} for Metal Q/K capture."
            )
        k_value = keys[row].transpose(1, 0, 2)

        active["pending"].append(
            (request["req_id"], layer, mode, q_value, k_value)
        )


def _flush_pending(runner: Any, pending: list[tuple]) -> None:
    if not pending:
        return

    import mlx.core as mx

    mx.eval(*(tensor for _rid, _layer, _mode, q, k in pending for tensor in (q, k)))
    output = getattr(runner, "_mia_metal_qk_states", None)
    if output is None:
        output = {"memory": {}, "disk": {}, "config": runner._mia_metal_qk_config}
        runner._mia_metal_qk_states = output

    with _LOCK:
        for req_id, layer, mode, q_value, k_value in pending:
            bucket_name = "disk" if _request_saves_to_disk(runner, req_id) else "memory"
            bucket = output[bucket_name].setdefault(req_id, {})
            module_name = f"model.layers.{layer}.self_attn.attn"
            entry = bucket.setdefault(
                module_name,
                {"q": [], "k_all": [], "layer_num": layer, "hookq_mode": mode},
            )
            entry["q"].append(_array_to_tensor(q_value).reshape(-1))
            entry["k_all"].append(_array_to_tensor(k_value).reshape(k_value.shape[0], -1))


def _request_saves_to_disk(runner: Any, req_id: str) -> bool:
    flags = getattr(runner, "_mia_metal_disk_requests", {})
    return bool(flags.get(req_id, False))


@contextmanager
def _capture_requests(runner: Any, requests: list[tuple[str, Any]], phase: str):
    contexts = []
    disk_requests = getattr(runner, "_mia_metal_disk_requests", None)
    if disk_requests is None:
        disk_requests = {}
        runner._mia_metal_disk_requests = disk_requests

    for req_id, sampling_params in requests:
        extra = getattr(sampling_params, "extra_args", None) or {}
        if extra.get("output_qk") is not None and extra.get("qk_capture", "qk") != "qk":
            raise MiaConfigurationError(
                "Metal Q/K capture currently emits raw Q/K probes; "
                "qk_capture='score' is not implemented."
            )
        contexts.append({"req_id": str(req_id), "extra": extra})
        if extra.get("output_qk") is not None:
            disk_requests[str(req_id)] = bool(extra.get("save_to_disk"))

    state = {"requests": contexts, "phase": phase, "pending": []}
    token = _ACTIVE_CAPTURE.set(state)
    try:
        yield
    except BaseException:
        raise
    else:
        _flush_pending(runner, state["pending"])
    finally:
        _ACTIVE_CAPTURE.reset(token)


def _patch_attention(model: Any) -> tuple[int, dict]:
    from vllm_metal.attention.patching import find_attn_attr, find_layers

    layers = find_layers(model)
    if not layers:
        raise MiaConfigurationError("Metal Q/K capture could not find model attention layers.")

    first_attn = None
    for layer in layers:
        attr = find_attn_attr(layer)
        if attr is not None:
            first_attn = getattr(layer, attr)
            break
    if first_attn is None:
        raise MiaConfigurationError("Metal Q/K capture found no supported attention modules.")

    cls = type(first_attn)
    module = __import__(cls.__module__, fromlist=["scaled_dot_product_attention"])
    if not hasattr(module, "scaled_dot_product_attention"):
        raise MiaConfigurationError(
            "Metal Q/K capture currently requires the mlx_lm Qwen2 attention implementation; "
            f"{cls.__module__} does not expose scaled_dot_product_attention."
        )
    if not (
        hasattr(first_attn, "q_proj")
        and hasattr(first_attn, "k_proj")
        and hasattr(first_attn, "rope")
        and hasattr(first_attn, "n_heads")
        and hasattr(first_attn, "n_kv_heads")
    ):
        raise MiaConfigurationError(
            "Metal Q/K capture requires the split-projection Qwen2 attention contract."
        )

    for layer_num, layer in enumerate(layers):
        attr = find_attn_attr(layer)
        if attr is not None:
            object.__setattr__(getattr(layer, attr), "_mia_metal_layer_num", layer_num)

    call_marker = "_mia_original_call"
    if not hasattr(cls, call_marker):
        original_call = cls.__call__

        def wrapped_call(self, *args, **kwargs):
            layer_token = _ACTIVE_LAYER.set(
                getattr(self, "_mia_metal_layer_num", None)
            )
            try:
                return original_call(self, *args, **kwargs)
            finally:
                _ACTIVE_LAYER.reset(layer_token)

        setattr(cls, call_marker, original_call)
        cls.__call__ = wrapped_call

    sdpa_marker = "_mia_original_scaled_dot_product_attention"
    if not hasattr(module, sdpa_marker):
        original_sdpa = module.scaled_dot_product_attention

        def observing_sdpa(queries, keys, *args, **kwargs):
            _observe_qk(queries, keys)
            return original_sdpa(queries, keys, *args, **kwargs)

        setattr(module, sdpa_marker, original_sdpa)
        module.scaled_dot_product_attention = observing_sdpa

    config = getattr(model, "args", None)
    if config is None:
        config = getattr(model, "config", None)
    if config is None:
        raise MiaConfigurationError(
            "Metal Q/K capture could not resolve Qwen2 model dimensions."
        )
    n_heads = int(getattr(config, "num_attention_heads"))
    n_kv_heads = int(getattr(config, "num_key_value_heads", n_heads))
    hidden = int(getattr(config, "hidden_size"))
    head_dim = int(getattr(config, "head_dim", hidden // n_heads))
    scale = float(getattr(first_attn, "scale", head_dim**-0.5))
    return len(layers), {
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "hidden_size": hidden,
        "head_dim": head_dim,
        "attention_multiplier": scale,
    }


def install_runner(runner: Any) -> None:
    """Install request-scoped Q/K observation on a MetalModelRunner."""
    if getattr(runner, "_mia_metal_qk_installed", False):
        return

    model = getattr(runner, "_forward_model", None)
    if model is None:
        raise MiaConfigurationError("Metal Q/K capture requires a loaded model.")
    num_layers, config = _patch_attention(model)
    runner._mia_metal_qk_config = config
    runner._mia_metal_qk_states = {"memory": {}, "disk": {}, "config": config}
    runner._mia_metal_num_layers = num_layers

    original_prefill = runner._prefill_single

    def prefill(req_id, token_ids, sampling_params, *args, **kwargs):
        with _capture_requests(runner, [(req_id, sampling_params)], "prefill"):
            return original_prefill(req_id, token_ids, sampling_params, *args, **kwargs)

    runner._prefill_single = prefill

    for method_name in ("_batched_decode", "_sequential_decode"):
        original = getattr(runner, method_name, None)
        if original is None:
            continue

        def wrap_decode(bound_method):
            def decode(decode_reqs, *args, **kwargs):
                requests = [
                    (req_id, state.sampling_params)
                    for req_id, state in decode_reqs
                ]
                with _capture_requests(runner, requests, "decode"):
                    return bound_method(decode_reqs, *args, **kwargs)

            return decode

        setattr(runner, method_name, wrap_decode(original))

    runner._mia_metal_qk_installed = True


def get_captured_states(runner: Any, external_req_id: str) -> bytes | None:
    states = getattr(runner, "_mia_metal_qk_states", None)
    if states is None:
        return None
    matches = [
        req_id for req_id in states["memory"]
        if req_id == external_req_id or req_id.startswith(f"{external_req_id}-")
    ]
    if not matches:
        return None
    qk_cache: dict = {}
    with _LOCK:
        for req_id in matches:
            entries = states["memory"].pop(req_id)
            getattr(runner, "_mia_metal_disk_requests", {}).pop(req_id, None)
            for name, entry in entries.items():
                target = qk_cache.setdefault(
                    name,
                    {
                        "q": [],
                        "k_all": [],
                        "layer_num": entry["layer_num"],
                        "hookq_mode": entry["hookq_mode"],
                    },
                )
                target["q"].extend(entry["q"])
                target["k_all"].extend(entry["k_all"])
    payload = {"config": states["config"], "qk_cache": qk_cache}
    return _COMPRESSOR.compress(pickle.dumps(payload))


def clear_captured_states(runner: Any, external_req_id: str) -> None:
    states = getattr(runner, "_mia_metal_qk_states", None)
    if states is None:
        return
    with _LOCK:
        for bucket_name in ("memory", "disk"):
            bucket = states[bucket_name]
            for req_id in list(bucket):
                if req_id == external_req_id or req_id.startswith(f"{external_req_id}-"):
                    del bucket[req_id]
        flags = getattr(runner, "_mia_metal_disk_requests", {})
        for req_id in list(flags):
            if req_id == external_req_id or req_id.startswith(f"{external_req_id}-"):
                del flags[req_id]


def flush_disk(runner: Any, external_req_ids: list[str], run_id: str, hook_dir: str):
    states = getattr(runner, "_mia_metal_qk_states", None)
    if states is None:
        return False

    selected: dict = {}
    with _LOCK:
        for external_req_id in external_req_ids:
            for req_id in list(states["disk"]):
                if req_id == external_req_id or req_id.startswith(f"{external_req_id}-"):
                    for name, entry in states["disk"].pop(req_id).items():
                        target = selected.setdefault(
                            name,
                            {
                                "q": [],
                                "k_all": [],
                                "layer_num": entry["layer_num"],
                                "hookq_mode": entry["hookq_mode"],
                            },
                        )
                        target["q"].extend(entry["q"])
                        target["k_all"].extend(entry["k_all"])
                    getattr(runner, "_mia_metal_disk_requests", {}).pop(req_id, None)
            flags = getattr(runner, "_mia_metal_disk_requests", {})
            for req_id in list(flags):
                if req_id == external_req_id or req_id.startswith(f"{external_req_id}-"):
                    del flags[req_id]
    if not selected:
        return False

    from mia.core.runtime.tp_shard import rank_dir_name
    from mia.workers._common import save_pt_atomic

    run_dir = Path(hook_dir) / str(run_id) / rank_dir_name(0)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_pt_atomic(
        {"config": states["config"], "qk_cache": selected},
        str(run_dir / "qk.pt"),
    )
    return str(run_dir)


def install_worker_api() -> None:
    """Expose the existing worker RPC methods on vLLM-Metal's worker."""
    global _INSTALLED_WORKER
    if _INSTALLED_WORKER:
        return

    from vllm_metal.v1.worker import MetalWorker

    def install_hooks(worker):
        extension = getattr(
            getattr(worker, "parallel_config", None),
            "worker_extension_cls",
            None,
        )
        if extension is not None and "QKCaptureWorker" not in str(extension):
            raise MiaConfigurationError(
                "The vLLM-Metal MIA adapter currently implements capture_qk only; "
                f"worker extension {extension!r} is not implemented."
            )
        install_runner(worker.model_runner)

    def get_states(worker, external_req_id):
        return get_captured_states(worker.model_runner, external_req_id)

    def clear_states(worker, external_req_id):
        clear_captured_states(worker.model_runner, external_req_id)

    def write_disk(worker, external_req_ids, run_id, hook_dir):
        return flush_disk(worker.model_runner, external_req_ids, run_id, hook_dir)

    MetalWorker.install_hooks = install_hooks
    MetalWorker.get_captured_states = get_states
    MetalWorker.clear_captured_states = clear_states
    MetalWorker.flush_disk = write_disk
    _INSTALLED_WORKER = True


def metal_plugin_selected() -> bool:
    """Whether vLLM-Metal is explicitly selected in the plugin environment."""
    return "metal" in {
        item.strip()
        for item in os.environ.get("VLLM_PLUGINS", "").split(",")
        if item.strip()
    }
