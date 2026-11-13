"""MIA's worker bridge for vLLM-Metal's MLX model runner."""
from __future__ import annotations

import pickle

import torch
import zstandard as zstd

from vllm_metal.v1.worker import MetalWorker

from mia.backends.vllm_metal.qk_capture import (
    install_prompt_qk_capture,
    install_request_capture_context,
    stack_qk_capture,
)
from mia.workers._common import clear_states_for_req, iter_matching_req_ids

_COMPRESSOR = zstd.ZstdCompressor(level=1)


class MiaMetalWorker(MetalWorker):
    """MetalWorker extension implementing MIA's existing Q/K probe RPCs."""

    def install_hooks(self) -> None:
        if getattr(self, "_mia_metal_hooks_installed", False):
            return
        extension = getattr(self.parallel_config, "worker_extension_cls", "") or ""
        additional_config = getattr(self.vllm_config, "additional_config", None) or {}
        worker_kind = additional_config.get("mia_worker")
        if "QKCaptureWorker" not in str(extension) and worker_kind != "qk":
            raise RuntimeError(
                "The vLLM-Metal MIA worker currently integrates capture_qk; "
                f"got worker extension {extension!r}"
            )
        if self.model_runner.paged_attention_runtime is not None:
            raise RuntimeError(
                "The vLLM-Metal MIA Q/K path currently requires "
                "VLLM_METAL_USE_PAGED_ATTENTION=0"
            )
        install_prompt_qk_capture(self)
        install_request_capture_context(self)
        self._mia_metal_states = {}
        self._mia_metal_disk_states = {}
        self._mia_metal_active = None
        self._mia_metal_hooks_installed = True

    def _record_metal_qk(self, active, layer_num: int, q: torch.Tensor,
                         k: torch.Tensor) -> None:
        if layer_num not in active["layers"]:
            return
        if active["mode"] == "last_token":
            q = q[-1]
        if k.ndim != 2 or q.ndim not in (1, 2):
            raise RuntimeError(
                f"Unexpected Metal Q/K capture shapes at layer {layer_num}: "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}"
            )
        target = (self._mia_metal_disk_states if active["extra"].get("save_to_disk")
                  else self._mia_metal_states)
        request = target.setdefault(active["req_id"], {})
        name = self._mia_metal_module_names[layer_num]
        layer = request.setdefault(name, {
            "q": [],
            "k_all": [],
            "layer_num": layer_num,
            "hookq_mode": active["mode"],
        })
        layer["q"].append(q.detach().cpu().contiguous())
        layer["k_all"].append(k.detach().cpu().contiguous())

    def _pack_metal_request(self, request_id: str, bucket: dict) -> dict | None:
        matches = list(iter_matching_req_ids(bucket, request_id))
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple Metal Q/K captures match request {request_id!r}: {matches}"
            )
        if not matches:
            return None
        request = bucket.pop(matches[0])
        packed = {}
        for name, layer in request.items():
            q, k = stack_qk_capture(
                layer["q"], layer["k_all"], layer["hookq_mode"]
            )
            packed[name] = {
                "q": q,
                "k_all": k,
                "layer_num": layer["layer_num"],
                "hookq_mode": layer["hookq_mode"],
            }
        return {
            "qk_cache": packed,
            "config": self._mia_metal_qk_config,
        }

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        self._ensure_mia_metal_state()
        payload = self._pack_metal_request(external_req_id, self._mia_metal_states)
        if payload is None:
            return None
        if not payload["qk_cache"]:
            error = {
                "mia_error": (
                    f"Metal Q/K capture produced no requested layers for "
                    f"request {external_req_id!r}"
                )
            }
            return _COMPRESSOR.compress(pickle.dumps(error))
        return _COMPRESSOR.compress(pickle.dumps(payload))

    def clear_captured_states(self, external_req_id: str) -> None:
        self._ensure_mia_metal_state()
        clear_states_for_req(self._mia_metal_states, external_req_id)
        clear_states_for_req(self._mia_metal_disk_states, external_req_id)

    def flush_disk(self, external_req_ids: list, run_id: str, hook_dir: str) -> bool:
        self._ensure_mia_metal_state()
        items = []
        for request_id in external_req_ids:
            payload = self._pack_metal_request(request_id, self._mia_metal_disk_states)
            items.append((str(request_id), payload))
        from mia.core.delivery.run_artifact import append

        return bool(append(hook_dir, run_id, kind="qk", items=items))

    def _ensure_mia_metal_state(self) -> None:
        if not hasattr(self, "_mia_metal_states"):
            self._mia_metal_states = {}
            self._mia_metal_disk_states = {}

    def load_model(self) -> None:
        super().load_model()
        self._mia_metal_hooks_installed = False

    def shutdown(self) -> None:
        try:
            self._ensure_mia_metal_state()
            self._mia_metal_states.clear()
            self._mia_metal_disk_states.clear()
        finally:
            super_shutdown = getattr(super(), "shutdown", None)
            if callable(super_shutdown):
                super_shutdown()
