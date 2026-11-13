"""Request-scoped Q/K capture for Granite models executed by vLLM-Metal."""
from __future__ import annotations

import os
import pickle
import threading
from typing import Any

import torch
import zstandard as zstd

_COMPRESSOR = zstd.ZstdCompressor(level=1)


def _mlx_to_torch(value: Any) -> torch.Tensor:
    """Materialize an MLX array through a NumPy-compatible dtype."""
    import mlx.core as mx
    import numpy as np

    return torch.from_numpy(np.asarray(value.astype(mx.float32)).copy())


class _CallableProxy:
    """Delegate a callable module while observing its result."""

    def __init__(self, original: Any, callback):
        self.original = original
        self.callback = callback

    def __call__(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        return self.callback(result)


class MetalQKCapture:
    """Patch Granite's attention RoPE callables and gather request Q/K histories.

    The wrapper observes the post-RoPE query and key produced by the actual
    mlx_lm model; generation remains owned by vLLM-Metal.
    """

    def __init__(self, runner, worker):
        self.runner = runner
        self.worker = worker
        self.lock = threading.RLock()
        self.active_extra: list[dict] = []
        self.active_req_ids: list[str] = []
        self.current_req_ids: list[str] = []
        self.active_is_prefill = False
        self.decode_cursor = 0
        self.rope_calls: dict[int, int] = {}
        self.states: dict[str, dict[str, dict]] = {}
        self.original_rope: list[tuple[Any, Any]] = []
        self.model = getattr(runner, "model", None)
        if self.model is None:
            raise RuntimeError("vLLM-Metal has no loaded model for Q/K capture")
        if getattr(runner, "_paged_attention_runtime", None) is not None:
            raise RuntimeError(
                "Metal Q/K capture currently requires VLLM_METAL_USE_PAGED_ATTENTION=0"
            )
        self.layers = getattr(self.model, "layers", None)
        if not isinstance(self.layers, (list, tuple)) or not self.layers:
            raise RuntimeError(
                "Metal Q/K capture supports the mlx_lm Granite model layout "
                "with model.layers"
            )
        args = getattr(self.model, "args", None)
        if args is None and hasattr(self.model, "model"):
            args = getattr(self.model.model, "args", None)
        if args is None:
            raise RuntimeError("Metal Q/K capture cannot read Granite model arguments")
        if os.environ.get("MIA_QK_SCORE") == "1":
            raise RuntimeError(
                "Metal Q/K capture does not implement MIA_QK_SCORE; unset it to "
                "capture raw Q/K."
            )
        self.config = {
            "num_attention_heads": int(args.num_attention_heads),
            "num_key_value_heads": int(args.num_key_value_heads),
            "hidden_size": int(args.hidden_size),
            "head_dim": int(args.hidden_size) // int(args.num_attention_heads),
            "attention_multiplier": float(args.attention_multiplier),
        }
        self.module_names: dict[int, str] = {}
        self._install_attention_wrappers()
        self._patch_runner()

    @staticmethod
    def _active(extra: dict) -> bool:
        return extra.get("output_qk") is not None

    def _install_attention_wrappers(self) -> None:
        for layer_num, layer in enumerate(self.layers):
            attention = getattr(layer, "self_attn", None)
            rope = getattr(attention, "rope", None) if attention is not None else None
            if rope is None:
                raise RuntimeError(
                    f"Metal Q/K capture cannot find Granite self_attn.rope at layer {layer_num}"
                )
            module_name = f"model.layers.{layer_num}.self_attn"
            self.module_names[layer_num] = module_name
            self.rope_calls[layer_num] = 0

            def observe(result, ln=layer_num):
                return self._observe_rope(ln, result)

            self.original_rope.append((attention, rope))
            attention.rope = _CallableProxy(rope, observe)

    def _patch_runner(self) -> None:
        original_prefill = self.runner._prefill_single

        def prefill(req_id, token_ids, sampling_params, *args, **kwargs):
            with self._request_scope([req_id], [sampling_params], is_prefill=True):
                return original_prefill(req_id, token_ids, sampling_params, *args, **kwargs)

        self.runner._prefill_single = prefill

        original_sequential_decode = self.runner._sequential_decode

        def sequential_decode(decode_reqs, *args, **kwargs):
            req_ids = [req_id for req_id, _ in decode_reqs]
            params = [state.sampling_params for _, state in decode_reqs]
            with self._request_scope(req_ids, params, is_prefill=False):
                return original_sequential_decode(decode_reqs, *args, **kwargs)

        self.runner._sequential_decode = sequential_decode

        original_batched_decode = self.runner._batched_decode

        def batched_decode(decode_reqs, *args, **kwargs):
            req_ids = [req_id for req_id, _ in decode_reqs]
            params = [state.sampling_params for _, state in decode_reqs]
            with self._request_scope(req_ids, params, is_prefill=False):
                return original_batched_decode(decode_reqs, *args, **kwargs)

        self.runner._batched_decode = batched_decode

    class _RequestScope:
        def __init__(self, owner, req_ids, sampling_params, is_prefill):
            self.owner = owner
            self.req_ids = req_ids
            self.extras = [
                dict(getattr(params, "extra_args", None) or {}) for params in sampling_params
            ]
            for extra in self.extras:
                if extra.get("output_qk") is None:
                    continue
                if extra.get("save_to_disk"):
                    raise RuntimeError(
                        "Metal Q/K capture does not implement save_to_disk delivery"
                    )
                if extra.get("qk_capture", "qk") != "qk":
                    raise RuntimeError(
                        "Metal Q/K capture currently implements raw Q/K only"
                    )
            self.is_prefill = is_prefill

        def __enter__(self):
            owner = self.owner
            owner.active_req_ids = self.req_ids
            owner.active_extra = self.extras
            owner.current_req_ids = []
            owner.decode_cursor = 0
            owner.active_is_prefill = self.is_prefill
            owner.rope_calls = {layer: 0 for layer in owner.rope_calls}

        def __exit__(self, exc_type, exc, tb):
            owner = self.owner
            owner.active_req_ids = []
            owner.active_extra = []
            owner.current_req_ids = []
            owner.decode_cursor = 0
            owner.active_is_prefill = False
            owner.rope_calls = {layer: 0 for layer in owner.rope_calls}
            return False

    def _request_scope(self, req_ids, sampling_params, is_prefill):
        return self._RequestScope(self, req_ids, sampling_params, is_prefill)

    def _route_rows(self, batch_size: int) -> list[str]:
        if not self.active_req_ids:
            return []
        if len(self.active_req_ids) == batch_size:
            self.current_req_ids = list(self.active_req_ids)
            return self.current_req_ids
        if self.decode_cursor >= len(self.active_req_ids):
            return []
        self.current_req_ids = [self.active_req_ids[self.decode_cursor]]
        self.decode_cursor += 1
        return self.current_req_ids

    def _observe_rope(self, layer_num: int, result):
        call = self.rope_calls[layer_num]
        if layer_num == 0 and call >= 2:
            self.rope_calls = {layer: 0 for layer in self.rope_calls}
            self.current_req_ids = []
            call = 0
        self.rope_calls[layer_num] = call + 1
        # Granite invokes one shared RoPE module first for Q and then for K.
        if call > 1 or not self.active_req_ids:
            return result
        is_query = call == 0
        if result.ndim != 4:
            raise RuntimeError(
                f"Granite layer {layer_num} produced a non-4D RoPE tensor: {result.shape}"
            )
        rows = result.transpose(0, 2, 1, 3).reshape(
            result.shape[0], result.shape[2], -1
        )
        req_ids = self.current_req_ids
        if not req_ids:
            req_ids = self._route_rows(int(result.shape[0]))
        if not req_ids:
            return result
        if len(req_ids) != int(rows.shape[0]):
            raise RuntimeError(
                f"Q/K capture request rows mismatch at layer {layer_num}: "
                f"{len(req_ids)} requests for {rows.shape[0]} rows"
            )
        for row_idx, req_id in enumerate(req_ids):
            extra = self.active_extra[self.active_req_ids.index(req_id)]
            spec = extra.get("output_qk")
            if not self._active(extra):
                continue
            if isinstance(spec, dict) and layer_num not in {int(k) for k in spec}:
                continue
            if isinstance(spec, (list, tuple)) and layer_num not in {int(k) for k in spec}:
                continue
            mode = extra.get("hookq_mode", "all_tokens")
            is_prefill = self.active_is_prefill
            hooks_on = extra.get("hooks_on", "prefill")
            state = self.states.setdefault(req_id, {}).setdefault(
                self.module_names[layer_num],
                {
                    "q": [],
                    "k_all": [],
                    "key_history": None,
                    "capture_this_pass": False,
                    "layer_num": layer_num,
                    "hookq_mode": mode,
                },
            )
            tensor = _mlx_to_torch(rows[row_idx])
            if is_query:
                state["capture_this_pass"] = (
                    hooks_on == "both"
                    or (hooks_on == "prefill" and is_prefill)
                    or (hooks_on == "decode" and not is_prefill)
                )
                if state["capture_this_pass"]:
                    state["q"].append(
                        tensor[-1:].clone() if mode == "last_token" else tensor.clone()
                    )
            else:
                key = tensor.clone()
                if state["key_history"] is not None:
                    key = torch.cat((state["key_history"], key), dim=0)
                state["key_history"] = key
                if state["capture_this_pass"]:
                    state["k_all"].append(key)
                state["capture_this_pass"] = False
        return result

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        from mia.workers._common import iter_matching_req_ids

        with self.lock:
            matched = list(iter_matching_req_ids(self.states, external_req_id))
            if not matched:
                return None
            cache: dict[str, dict] = {}
            for req_id in matched:
                for module_name, state in self.states.pop(req_id).items():
                    if not state["q"] or not state["k_all"]:
                        continue
                    prior = cache.get(module_name)
                    if prior is None:
                        cache[module_name] = {
                            "q": torch.cat(state["q"], dim=0),
                            "k_all": state["k_all"],
                            "layer_num": state["layer_num"],
                            "hookq_mode": state["hookq_mode"],
                        }
                    else:
                        prior["q"] = torch.cat((prior["q"], torch.cat(state["q"], dim=0)))
                        prior["k_all"].extend(state["k_all"])
            if not cache:
                return None
            payload = {"qk_cache": cache, "config": self.config}
            return _COMPRESSOR.compress(pickle.dumps(payload))

    def clear_captured_states(self, external_req_id: str) -> None:
        from mia.workers._common import iter_matching_req_ids

        with self.lock:
            for req_id in iter_matching_req_ids(self.states, external_req_id):
                self.states.pop(req_id, None)

    def close(self) -> None:
        for attention, rope in self.original_rope:
            attention.rope = rope
        self.original_rope.clear()
