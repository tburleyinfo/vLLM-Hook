from vllm_hook_plugins.registry import PluginRegistry
from vllm_hook_plugins.hook_llm import HookLLM
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker
from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer
import importlib.util
import platform


def _use_metal_workers() -> bool:
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        return False
    return importlib.util.find_spec("vllm_metal") is not None


def register_plugins():
    probe_worker_cls = ProbeHookQKWorker
    steer_worker_cls = SteerHookActWorker
    if _use_metal_workers():
        from vllm_hook_plugins.workers.metal import (
            ProbeHookQKWorkerMetal,
            SteerHookActWorkerMetal,
        )
        probe_worker_cls = ProbeHookQKWorkerMetal
        steer_worker_cls = SteerHookActWorkerMetal

    # Register workers
    PluginRegistry.register_worker("probe_hook_qk", probe_worker_cls)
    PluginRegistry.register_worker("steer_hook_act", steer_worker_cls)
    
    # Register analyzers
    PluginRegistry.register_analyzer("attn_tracker", AttntrackerAnalyzer)
    PluginRegistry.register_analyzer("core_reranker", CorerAnalyzer)

__all__ = [
    "PluginRegistry",
    "HookLLM",
    "ProbeHookQKWorker", 
    "SteerHookActWorker",
    "AttntrackerAnalyzer",
    "CorerAnalyzer",
    "register_plugins"
]
