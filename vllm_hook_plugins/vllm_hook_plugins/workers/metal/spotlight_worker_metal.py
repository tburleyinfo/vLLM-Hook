"""Metal Spotlight worker mixin for vLLM-Hook.

This mirrors the CUDA Spotlight worker's per-request ``extra_args`` contract
while using MLX module wrappers because vLLM-Metal modules do not expose
PyTorch forward hooks.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

from vllm_hook_plugins.utils.spotlight.utils import compute_spotlight_bias

ATTN_PATTERNS = (
    re.compile(r"^model\.layers\.(\d+)\.self_attn$"),
    re.compile(r"^model\.model\.layers\.(\d+)\.self_attn$"),
)

logger = logging.getLogger(__name__)


class MLXSpotlightWrapper:
    """Wrap an MLX attention module and optionally replace its output."""

    def __init__(self, module: Any, name: str, hook_fn: Any):
        self.module = module
        self.name = name
        self.hook_fn = hook_fn

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self.module(*args, **kwargs)
        replacement = self.hook_fn(args, kwargs, output, self.name)
        return output if replacement is None else replacement


class SpotlightWorkerMetal:
    """Mixin injected into vLLM-Metal workers for Spotlight steering."""

    _spotlight_hooks_installed: bool = False

    def _ensure_extension_state(self) -> None:
        if getattr(self, "_metal_spotlight_extension_ready", False):
            return
        self._debug_hook = os.environ.get(
            "HOOK_DEBUG", os.environ.get("VLLM_HOOK_DEBUG", "")
        ) == "1"
        self._metal_spotlight_extension_ready = True

    def _stage(self, message: str) -> None:
        self._ensure_extension_state()
        if not getattr(self, "_debug_hook", False):
            return
        pid = os.getpid()
        rank = getattr(self, "rank", "?")
        local_rank = getattr(self, "local_rank", "?")
        print(
            f"[metal-spotlight-worker pid={pid} rank={rank} "
            f"local_rank={local_rank}] {message}",
            flush=True,
        )

    def __init__(self, *args: Any, **kwargs: Any):
        self._ensure_extension_state()

    def install_hooks(self) -> None:
        self._ensure_extension_state()
        if getattr(self, "_spotlight_hooks_installed", False):
            return
        self._spotlight_hooks_installed = True
        try:
            self._install_hooks()
            print("Hooks installed successfully", flush=True)
        except Exception as exc:
            print(f"Hook installation failed: {exc}", flush=True)

    @staticmethod
    def _match_attn(name: str) -> int | None:
        for pattern in ATTN_PATTERNS:
            match = pattern.match(name)
            if match:
                return int(match.group(1))
        return None

    def _request_spotlight_params(self, batch_size: int) -> list[dict[str, Any] | None]:
        try:
            req_ids = list(self.model_runner.input_batch.req_ids)
        except Exception:
            return [None] * batch_size

        entries: list[dict[str, Any] | None] = []
        for index in range(batch_size):
            try:
                req_state = self.model_runner.requests.get(req_ids[index])
                sampling_params = getattr(req_state, "sampling_params", None)
                extra = getattr(sampling_params, "extra_args", None)
            except Exception:
                extra = None
            spotlight_cfg = (extra or {}).get("spotlight")
            if not spotlight_cfg or not spotlight_cfg.get("span_ranges"):
                entries.append(None)
                continue
            entries.append(
                {
                    "span_ranges": spotlight_cfg["span_ranges"],
                    "alpha": float(spotlight_cfg.get("alpha", 0.2)),
                }
            )
        return entries

    @staticmethod
    def _inner_module(module: Any) -> Any:
        return getattr(module, "_inner", module)

    @staticmethod
    def _has_attention_api(module: Any) -> bool:
        return all(
            hasattr(module, attr)
            for attr in ("q_proj", "k_proj", "rope", "o_proj", "n_heads")
        )

    @staticmethod
    def _repeat_kv(value: Any, repeats: int) -> Any:
        if repeats == 1:
            return value
        import mlx.core as mx

        return mx.repeat(value, repeats, axis=1)

    @staticmethod
    def _causal_mask(q_len: int, k_len: int) -> Any:
        import mlx.core as mx

        row_idx = mx.arange(q_len)[:, None]
        col_idx = mx.arange(k_len)[None, :]
        return mx.where(col_idx > row_idx, mx.array(-float("inf")), mx.array(0.0))

    def _compute_spotlight_output(
        self,
        attn_module: Any,
        raw_x: mx.array,
        input_args: tuple[Any, ...],
        input_kwargs: dict[str, Any],
        entries: list[dict[str, Any] | None],
    ) -> Any | None:
        import mlx.core as mx
        from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch, torch_to_mlx

        batch, seq_len, _ = raw_x.shape
        if seq_len <= 1 or not any(entries):
            return None

        if not self._has_attention_api(attn_module):
            return None

        n_heads = int(getattr(attn_module, "n_heads"))
        n_kv_heads = int(getattr(attn_module, "n_kv_heads", n_heads))
        head_dim = int(
            getattr(
                attn_module,
                "head_dim",
                attn_module.k_proj.weight.shape[0] // n_kv_heads,
            )
        )
        scale = float(getattr(attn_module, "scale", 1.0 / math.sqrt(head_dim)))

        queries = attn_module.q_proj(raw_x).reshape(batch, seq_len, n_heads, -1)
        keys = attn_module.k_proj(raw_x).reshape(batch, seq_len, n_kv_heads, -1)
        if hasattr(attn_module, "v_proj"):
            values = attn_module.v_proj(raw_x).reshape(
                batch, seq_len, n_kv_heads, -1
            )
        else:
            values = keys

        if hasattr(attn_module, "q_norm"):
            queries = attn_module.q_norm(queries)
        if hasattr(attn_module, "k_norm"):
            keys = attn_module.k_norm(keys)
        if hasattr(attn_module, "v_norm"):
            values = attn_module.v_norm(values)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        cache = input_kwargs.get("cache") if input_kwargs else None
        if cache is None and len(input_args) > 2:
            cache = input_args[2]
        if hasattr(attn_module, "rope"):
            if cache is not None:
                queries = attn_module.rope(queries, offset=cache.offset)
                keys = attn_module.rope(keys, offset=cache.offset)
            else:
                queries = attn_module.rope(queries)
                keys = attn_module.rope(keys)
        else:
            return None

        keys = self._repeat_kv(keys, n_heads // n_kv_heads)
        values = self._repeat_kv(values, n_heads // n_kv_heads)

        logits = mx.matmul(queries.astype(mx.float32), keys.transpose(0, 1, 3, 2).astype(mx.float32))
        logits = logits * scale
        logits = logits + self._causal_mask(seq_len, keys.shape[2]).reshape(
            1, 1, seq_len, keys.shape[2]
        )

        logits_torch = mlx_to_torch(logits, device="cpu")
        spans = [
            entry["span_ranges"] if entry is not None else []
            for entry in entries[:batch]
        ]
        alpha = next(
            (entry["alpha"] for entry in entries[:batch] if entry is not None),
            0.2,
        )
        weights_torch = compute_spotlight_bias(
            logits_torch,
            spans,
            target_proportion=alpha,
        )
        weights = torch_to_mlx(weights_torch).astype(values.dtype)
        attn_out = mx.matmul(weights, values)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
            batch, seq_len, n_heads * head_dim
        )
        return attn_module.o_proj(attn_out)

    def _spotlight_hook(
        self,
        input_args: tuple[Any, ...],
        input_kwargs: dict[str, Any],
        output: Any,
        module_name: str,
    ) -> Any | None:
        try:
            raw_x = input_args[0]
            batch_size = raw_x.shape[0]
            entries = self._request_spotlight_params(batch_size)
            if not any(entries):
                return None
            attn_module = self._wrapped[module_name]
            steered = self._compute_spotlight_output(
                attn_module,
                raw_x,
                input_args,
                input_kwargs,
                entries,
            )
            if steered is None:
                return None
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered
        except Exception as exc:
            logger.error("Metal Spotlight hook error: %s", exc, exc_info=True)
            return None

    def _install_hooks(self) -> None:
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        named_modules = dict(model.named_modules())
        self._hooks = []
        self._wrapped = {}
        matched = []
        seen_targets = set()

        for name, module in named_modules.items():
            layer_idx = self._match_attn(name)
            if layer_idx is None:
                continue

            parent_name, target_name = name.rsplit(".", 1)
            parent = named_modules.get(parent_name)
            if parent is None:
                continue
            target_key = (id(parent), target_name)
            if target_key in seen_targets:
                continue

            inner = self._inner_module(module)
            if not self._has_attention_api(inner):
                continue

            wrapped = MLXSpotlightWrapper(
                module=module,
                name=name,
                hook_fn=self._spotlight_hook,
            )
            self._wrapped[name] = inner
            setattr(parent, target_name, wrapped)
            seen_targets.add(target_key)
            self._hooks.append(
                {
                    "parent": parent,
                    "target_name": target_name,
                    "original_module": module,
                }
            )
            matched.append(name)

        self._stage(f"installed Spotlight hooks on {matched[:3]}")
        if not matched:
            print("No Metal attention modules matched for Spotlight hooks", flush=True)

    def _uninstall_hooks(self) -> None:
        hooks = getattr(self, "_hooks", None)
        if not hooks:
            self._spotlight_hooks_installed = False
            return
        for entry in reversed(hooks):
            try:
                setattr(entry["parent"], entry["target_name"], entry["original_module"])
            except Exception as exc:
                print(
                    f"Error restoring Metal Spotlight hook {entry['target_name']}: {exc}",
                    flush=True,
                )
        hooks.clear()
        self._spotlight_hooks_installed = False
