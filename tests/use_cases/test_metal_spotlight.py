from vllm_hook_plugins.metal._hook_plugin import _resolve_metal_worker_class
from vllm_hook_plugins.workers.metal import SpotlightWorkerMetal


def test_metal_spotlight_worker_imports():
    assert SpotlightWorkerMetal.__name__ == "SpotlightWorkerMetal"


def test_metal_plugin_resolves_spotlight_worker_type():
    path = _resolve_metal_worker_class("spotlight")

    assert path.endswith("spotlight_worker_metal.SpotlightWorkerMetal")
