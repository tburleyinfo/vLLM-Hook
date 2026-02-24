import os
import math
import inspect
import json
import torch
from typing import Dict, List
from vllm_metal.v1.worker import MetalWorker
from vllm.forward_context import get_forward_context
import re

ATTN_PATTERNS = [
    # GPT-2: transformer.h.<i>.attn
    re.compile(r"^transformer\.h\.(\d+)\.attn.attn$"),
    # OPT: model.decoder.layers.<i>.self_attn
    re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn.attn$"),
    # Qwen/LLaMA: model.layers.<i>.self_attn
    re.compile(r"^model\.layers\.(\d+)\.self_attn.attn$"),
]


def match_attn(name: str):
    for pat in ATTN_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


class ProbeHookQKWorkerMetal(MetalWorker):
    @staticmethod
    def _canonical_attn_layer_name(layer_idx: int) -> str:
        return f"model.layers.{layer_idx}.self_attn.attn"

    @staticmethod
    def _model_accepts_qk_callback(model) -> bool:
        call_fn = getattr(model, "__call__", None)
        if call_fn is None:
            return False
        try:
            sig = inspect.signature(call_fn)
        except (TypeError, ValueError):
            return False
        params = sig.parameters.values()
        has_named = any(p.name == "qk_capture_callback" for p in params)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        return has_named or has_kwargs

    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)

        try:
            # Original:
            # self._install_hooks()
            # print("Metal hooks installed successfully")
            install_result = self._install_hooks()
            mode = install_result.get("mode", "none")
            if mode == "native":
                count = int(install_result.get("count", 0))
                layers = install_result.get("layers", [])
                print(f"Installed {count} hooks on layers: {layers}")
                print("Metal hooks installed successfully (native capture)")
            elif mode == "pytorch":
                count = int(install_result.get("count", 0))
                layers = install_result.get("layers", [])
                print(f"Installed {count} hooks on layers: {layers}")
                print("Metal hooks installed successfully (pytorch hooks)")
            else:
                raise RuntimeError(
                    "No active metal hook capture path was installed."
                )
        except Exception as e:
            print(f"Metal hook installation failed: {e}")
            # Do not continue model execution if hook installation failed.
            raise

        return r

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        self.hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        self.hook_dir = os.environ.get("VLLM_HOOK_DIR")
        self.run_id_file = os.environ.get("VLLM_RUN_ID")
        self.hookq_mode = os.environ.get("VLLM_HOOKQ_MODE", "all_tokens")
        # Avoid forcing torch distributed initialization on local Apple Silicon runs.
        tp_rank = int(os.environ.get("VLLM_TP_RANK", "0"))

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return {"mode": "none"}

        self.layer_to_heads = self._parse_layer_heads()
        self.important_layers = set(self.layer_to_heads.keys())
        if not self.important_layers:
            raise RuntimeError(
                "No hook layers resolved. Set VLLM_HOOK_LAYER_HEADS or provide "
                "VLLM_HOOK_CONFIG with params.important_heads."
            )

        self._run_cache = {}

        # vllm-metal exposes normalized model config through model_runner.model_args.
        # Refresh that dict first so we do not rely on model.config (often missing on MLX models).
        if hasattr(self.model_runner, "_extract_model_args"):
            self.model_runner._extract_model_args()
        model_args = getattr(self.model_runner, "model_args", {}) or {}

        num_h = int(model_args.get("num_attention_heads") or model_args.get("n_heads") or 32)
        num_kv = int(
            model_args.get("num_key_value_heads") or model_args.get("n_kv_heads") or num_h
        )
        hidden = int(model_args.get("hidden_size") or model_args.get("dim") or 4096)
        head_dim = int(model_args.get("head_dim") or (hidden // num_h))
        attn_mult = float(
            model_args.get("attention_multiplier") or (1 / math.sqrt(head_dim))
        )
        self._conf = dict(
            num_attention_heads=num_h,
            num_key_value_heads=num_kv,
            hidden_size=hidden,
            head_dim=head_dim,
            attention_multiplier=attn_mult,
        )

        def _to_torch_cpu(t):
            if hasattr(t, "to_torch"):
                t = t.to_torch()
            return t.detach().cpu()

        def qkv_hook(input, module_name):
            if not os.path.exists(self.hook_flag):
                return None
            elif os.path.exists(self.run_id_file):
                run_id = (open(self.run_id_file).read().strip().split('\n'))[-1]
            else:
                raise Exception("run_id not found.")

            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)
            if metadata is None:
                return

            seq_lens = getattr(metadata, "seq_lens", None)
            if seq_lens is None and module_name in metadata:
                seq_lens = metadata[module_name].seq_lens

            last_indices = torch.cumsum(seq_lens, dim=0)
            bs = len(last_indices)
            last_indices = torch.cat([torch.tensor([0]).to(last_indices.device), last_indices])

            cache = self._run_cache.get(run_id)
            if cache is None:
                cache = {"config": self._conf, "qk_cache": {}}
                self._run_cache[run_id] = cache
            if module_name not in cache["qk_cache"]:
                q_tokens = []
                k_all_tokens = []
            else:
                q_tokens = cache["qk_cache"][module_name]["q"]
                k_all_tokens = cache["qk_cache"][module_name]["k_all"]

            layer_num = match_attn(module_name)
            if self.hookq_mode == "all_tokens":
                q_tokens.extend([_to_torch_cpu(input[0][last_indices[i]:last_indices[i + 1], :]) for i in range(bs)])
            elif self.hookq_mode == "last_token":
                q_tokens.extend(list(_to_torch_cpu(input[0][last_indices[1:] - 1, :])))
            else:
                raise NotImplementedError
            k_all_tokens.extend([_to_torch_cpu(input[1][last_indices[i]:last_indices[i + 1], :]) for i in range(bs)])

            cache["qk_cache"][module_name] = {
                "q": q_tokens,
                "k_all": k_all_tokens,
                "layer_num": layer_num,
            }

            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            cache_path = os.path.join(run_dir, "qk.pt")
            torch.save(cache, cache_path)

        self._hooks = []
        matched = []
        # Prefer model-runner native capture when callback support exists.
        callback_supported = getattr(
            self.model_runner, "_supports_qk_capture_callback", None
        )
        if callback_supported is None:
            callback_supported = self._model_accepts_qk_callback(model)
        if callback_supported is True:
            layer_list = sorted(
                [self._canonical_attn_layer_name(i) for i in self.important_layers]
            )
            print(
                "Using model-runner native Q/K capture path "
                f"for layers: {layer_list}"
            )
            return {"mode": "native", "count": len(layer_list), "layers": layer_list}

        if not hasattr(model, "named_modules"):
            print(
                "Metal model does not expose named_modules(); "
                "cannot install PyTorch-style forward hooks."
            )
            raise RuntimeError(
                "No hook installation path available: "
                "native callback unsupported and model has no named_modules()."
            )

        for name, module in model.named_modules():
            layer_num = match_attn(name)
            if layer_num is None:
                continue
            if layer_num not in self.important_layers:
                continue
            if not hasattr(module, "register_forward_hook"):
                continue
            hook = module.register_forward_hook(lambda m, i, o, n=name: qkv_hook(i, n))
            self._hooks.append(hook)
            matched.append(name)

        print(f"Installed {len(self._hooks)} metal hooks on layers: {matched}")
        if len(self._hooks) == 0:
            raise RuntimeError(
                "PyTorch-style hook path installed zero hooks; "
                "check ATTN_PATTERNS and VLLM_HOOK_LAYER_HEADS."
            )
        return {"mode": "pytorch", "count": len(self._hooks), "layers": matched}

    def _parse_layer_heads(self) -> Dict[int, List[int]]:
        # Original:
        # layer_heads = os.environ.get("VLLM_HOOK_LAYER_HEADS", "")
        layer_heads = os.environ.get("VLLM_HOOK_LAYER_HEADS", "")
        result = {}

        if layer_heads.strip():
            for part in layer_heads.split(";"):
                part = part.strip()
                if not part:
                    continue
                layer_str, heads_str = part.split(":")
                layer_idx = int(layer_str)
                head_indices = sorted([int(h) for h in heads_str.split(",") if h])
                result[layer_idx] = head_indices
            return result

        # Fallback: parse attention-tracker config directly.
        config_path = os.environ.get("VLLM_HOOK_CONFIG", "").strip()
        if config_path:
            if not os.path.exists(config_path):
                raise RuntimeError(
                    f"VLLM_HOOK_CONFIG is set but file does not exist: {config_path}"
                )
            with open(config_path, "r") as f:
                config_data = json.load(f)
            important_heads = (
                config_data.get("params", {}).get("important_heads", [])
            )
            for pair in important_heads:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                layer_idx = int(pair[0])
                head_idx = int(pair[1])
                if layer_idx not in result:
                    result[layer_idx] = []
                result[layer_idx].append(head_idx)

            for layer_idx in list(result.keys()):
                result[layer_idx] = sorted(set(result[layer_idx]))

        return result

    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)
