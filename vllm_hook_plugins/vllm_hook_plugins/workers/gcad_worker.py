"""GCAD Worker Mixin — gated cropped attention-delta steering.

This worker implements the inference-time side of Prompt-Activation Duality's
GCAD idea: apply a saved per-layer attention-delta at the attention-output
boundary, with token-level gating based on query compatibility with saved
system-prompt keys.

Per-request activation is controlled via SamplingParams.extra_args["gcad"].
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import torch
from vllm.forward_context import get_forward_context

from vllm_hook_plugins.utils.gcad import (
    compute_gcad_gate,
    load_gcad_vectors,
    normalize_gcad_layers,
)
from vllm_hook_plugins.workers._common import (
    get_query_metadata,
    iter_matched_modules,
    match_attn,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GCADWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls."""

    if TYPE_CHECKING:
        model_runner: Any

    _gcad_hooks_installed: bool = False

    def install_hooks(self):
        """Install GCAD hooks on attention modules. Idempotent."""
        if self._gcad_hooks_installed:
            return
        self._gcad_hooks_installed = True

        model = getattr(self.model_runner, "model", None)
        if model is None:
            logger.warning("No model found; skipping GCAD hook installation")
            return

        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        num_h = int(getattr(text_cfg, "num_attention_heads", 32))
        hidden = int(getattr(text_cfg, "hidden_size", 4096))
        head_dim = hidden // num_h

        self._gcad_conf = dict(
            num_attention_heads=num_h,
            hidden_size=hidden,
            head_dim=head_dim,
        )
        self._gcad_vector_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

        def _load_layers(vector_path: str) -> Dict[int, Dict[str, Any]]:
            layers = self._gcad_vector_cache.get(vector_path)
            if layers is None:
                layers = normalize_gcad_layers(load_gcad_vectors(vector_path))
                self._gcad_vector_cache[vector_path] = layers
            return layers

        def _resolve_batch(req_ids, bs):
            entries = []
            for i in range(bs):
                req_id = req_ids[i]
                req_state = self.model_runner.requests.get(req_id)
                if req_state is None or req_state.sampling_params is None:
                    entries.append(None)
                    continue
                extra = req_state.sampling_params.extra_args
                cfg = (extra or {}).get("gcad")
                if not isinstance(cfg, dict):
                    entries.append(None)
                    continue
                if not cfg.get("vector_path"):
                    entries.append(None)
                    continue
                entries.append(cfg)
            return entries

        def _should_apply(cfg: Dict[str, Any], req_state) -> bool:
            hooks_on = cfg.get("hooks_on", "decode")
            is_prefill = len(req_state.output_token_ids) == 0
            if hooks_on == "prefill" and not is_prefill:
                return False
            if hooks_on == "decode" and is_prefill:
                return False
            return hooks_on in {"prefill", "decode", "both"}

        def gcad_attention_hook(module, input_tuple, output, *, layer_num: int):
            if torch.cuda.is_current_stream_capturing():
                return None

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)
            if metadata is None:
                return None

            query_start_loc, _seq_lens = get_query_metadata(metadata)
            if query_start_loc is None:
                return None

            try:
                req_ids = self.model_runner.input_batch.req_ids
            except Exception:
                return None

            bs = len(query_start_loc) - 1
            entries = _resolve_batch(req_ids, bs)
            if not entries or not any(entries):
                return None

            try:
                Q = input_tuple[0]
                steered = output.clone()

                for batch_idx, cfg in enumerate(entries):
                    if cfg is None:
                        continue

                    req_state = self.model_runner.requests.get(req_ids[batch_idx])
                    if req_state is None or not _should_apply(cfg, req_state):
                        continue

                    layers = _load_layers(cfg["vector_path"])
                    layer_entry = layers.get(layer_num)
                    if layer_entry is None:
                        continue

                    q_start = int(query_start_loc[batch_idx])
                    q_end = int(query_start_loc[batch_idx + 1])
                    qlen = q_end - q_start
                    if qlen <= 0:
                        continue

                    q_slice = Q[q_start:q_end]
                    orig_dtype = steered.dtype
                    query = q_slice.reshape(qlen, num_h, head_dim).transpose(0, 1).float()

                    mean_sys_key = layer_entry["mean_sys_key"].to(
                        query.device, dtype=query.dtype
                    )
                    d_bar = layer_entry.get("d_bar", 0.0)
                    if isinstance(d_bar, torch.Tensor):
                        d_bar = d_bar.to(query.device, dtype=query.dtype)
                    c_base = float(cfg.get("c_base", layer_entry.get("c_base", 1.0)))
                    sharpness = float(cfg.get("sharpness", layer_entry.get("sharpness", 1.0)))
                    gate = compute_gcad_gate(
                        query,
                        mean_sys_key,
                        d_bar=d_bar,
                        c_base=c_base,
                        sharpness=sharpness,
                    ).to(steered.device, dtype=orig_dtype)

                    delta = layer_entry["delta"].to(steered.device, dtype=orig_dtype)
                    if delta.numel() != steered.shape[-1]:
                        raise ValueError(
                            f"GCAD delta for layer {layer_num} has hidden size "
                            f"{delta.numel()}, expected {steered.shape[-1]}"
                        )
                    steered[q_start:q_end] = (
                        steered[q_start:q_end]
                        + gate.view(-1, 1) * delta.view(1, -1)
                    )

                return steered

            except Exception as e:
                logger.error(f"GCAD hook error: {e}", exc_info=True)
                return None

        self._gcad_hooks = []
        matched = []
        for name, module, layer_num in iter_matched_modules(model, match_attn):
            hook = module.register_forward_hook(
                lambda m, i, o, ln=layer_num: gcad_attention_hook(
                    m, i, o, layer_num=ln
                )
            )
            self._gcad_hooks.append(hook)
            matched.append(name)

        logger.info(f"Installed {len(self._gcad_hooks)} GCAD hooks on: {matched[:3]}...")
