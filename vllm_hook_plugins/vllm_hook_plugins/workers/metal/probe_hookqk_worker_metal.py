import math
import os
import re
import json
from typing import Dict, List

import torch
from vllm.forward_context import get_forward_context
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1.worker import MetalWorker

ATTN_PATTERNS = [
    re.compile(r"^transformer\.h\.(\d+)\.attn.attn$"),
    re.compile(r"^model\.decoder\.layers\.(\d+)\.self_attn.attn$"),
    re.compile(r"^model\.layers\.(\d+)\.self_attn.attn$"),
]


def match_attn(name: str):
    for pat in ATTN_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


class ProbeHookQKWorkerMetal(MetalWorker):
    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)

        try:
            self._install_hooks()
            print("Hooks installed successfully")
        except Exception as e:
            print(f"Hook installation failed: {e}")
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
        tp_rank = int(os.environ.get("VLLM_TP_RANK", "0"))

        if not all([self.hook_dir, self.hook_flag, self.run_id_file]):
            print("Missing hook environment variables")
            return

        self.layer_to_heads = self._parse_layer_heads()
        self.important_layers = set(self.layer_to_heads.keys())
        self._run_cache = {}

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

        def to_torch_cpu(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu()
            return mlx_to_torch(tensor, device=torch.device("cpu")).detach().cpu()

        def flatten_qkv(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)
            return tensor.transpose(0, 2, 1, 3).reshape(tensor.shape[0], tensor.shape[2], -1)

        def split_requests(tensor, seq_lens):
            if tensor.shape[0] == len(seq_lens):
                return [tensor[i, :seq_len] for i, seq_len in enumerate(seq_lens)]
            if tensor.shape[0] == 1:
                slices = []
                start = 0
                for seq_len in seq_lens:
                    end = start + seq_len
                    slices.append(tensor[0, start:end])
                    start = end
                return slices
            raise RuntimeError(
                f"Cannot split hook tensor with shape {tuple(tensor.shape)} "
                f"using seq_lens={seq_lens}"
            )

        def resolve_seq_lens():
            ctx = get_forward_context()
            metadata = getattr(ctx, "attn_metadata", None)
            seq_lens = getattr(metadata, "seq_lens", None)
            if seq_lens is None:
                return None
            if isinstance(seq_lens, torch.Tensor):
                return [int(x) for x in seq_lens.detach().cpu().tolist()]
            return [int(x) for x in seq_lens]

        def qkv_hook(input, module_name):
            if not os.path.exists(self.hook_flag):
                return None
            if not os.path.exists(self.run_id_file):
                raise RuntimeError("run_id not found.")

            seq_lens = resolve_seq_lens()
            if not seq_lens:
                return None

            run_id = open(self.run_id_file).read().strip().split("\n")[-1]
            cache = self._run_cache.get(run_id)
            if cache is None:
                cache = {"config": self._conf, "qk_cache": {}}
                self._run_cache[run_id] = cache

            q_tokens = cache["qk_cache"].get(module_name, {}).get("q", [])
            k_all_tokens = cache["qk_cache"].get(module_name, {}).get("k_all", [])

            q_flat = flatten_qkv(input[0])
            k_flat = flatten_qkv(input[1])
            q_requests = split_requests(q_flat, seq_lens)
            k_requests = split_requests(k_flat, seq_lens)

            layer_num = match_attn(module_name)
            if self.hookq_mode == "all_tokens":
                q_tokens.extend([to_torch_cpu(q_req) for q_req in q_requests])
            elif self.hookq_mode == "last_token":
                q_tokens.extend([to_torch_cpu(q_req[-1]) for q_req in q_requests])
            else:
                raise NotImplementedError
            k_all_tokens.extend([to_torch_cpu(k_req) for k_req in k_requests])

            cache["qk_cache"][module_name] = {
                "q": q_tokens,
                "k_all": k_all_tokens,
                "layer_num": layer_num,
            }

            run_dir = os.path.join(self.hook_dir, run_id, f"tp_rank_{tp_rank}")
            os.makedirs(run_dir, exist_ok=True)
            torch.save(cache, os.path.join(run_dir, "qk.pt"))

        self._hooks = []
        matched = []
        for name, module in model.named_modules():
            layer_num = match_attn(name)
            if layer_num is None:
                continue
            if layer_num not in self.important_layers:
                continue
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: qkv_hook(i, n)
            )
            self._hooks.append(hook)
            matched.append(name)

        if not self._hooks:
            raise RuntimeError(
                "Installed zero hooks. Ensure mlx-lm exposes model.layers.<i>.self_attn.attn."
            )
        print(f"Installed {len(self._hooks)} hooks on layers: {matched}")

    def _parse_layer_heads(self) -> Dict[int, List[int]]:
        layer_heads = os.environ.get("VLLM_HOOK_LAYER_HEADS", "")
        result = {}

        for part in layer_heads.split(";"):
            part = part.strip()
            if not part:
                continue

            layer_str, heads_str = part.split(":")
            layer_idx = int(layer_str)
            head_indices = sorted([int(h) for h in heads_str.split(",") if h])
            result[layer_idx] = head_indices

        if result:
            return result

        config_path = os.environ.get("VLLM_HOOK_CONFIG", "").strip()
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            for layer_idx, head_idx in config_data.get("params", {}).get("important_heads", []):
                result.setdefault(int(layer_idx), []).append(int(head_idx))
            for layer_idx, heads in list(result.items()):
                result[layer_idx] = sorted(set(heads))

        return result

    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)
