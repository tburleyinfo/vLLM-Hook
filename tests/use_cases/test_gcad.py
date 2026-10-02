import torch
import pytest
import importlib.util
from pathlib import Path

_UTILS_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_hook_plugins"
    / "vllm_hook_plugins"
    / "utils"
    / "gcad"
    / "utils.py"
)
_SPEC = importlib.util.spec_from_file_location("gcad_utils_under_test", _UTILS_PATH)
gcad_utils = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gcad_utils)

compute_gcad_gate = gcad_utils.compute_gcad_gate
normalize_gcad_layers = gcad_utils.normalize_gcad_layers


def test_gcad_worker_is_registered():
    pytest.importorskip("vllm")
    from vllm_hook_plugins import PluginRegistry, register_plugins

    register_plugins()
    worker = PluginRegistry.get_worker("probe_gcad")
    assert worker.path == "vllm_hook_plugins.workers.gcad_worker.GCADWorker"


def test_normalize_gcad_layers_accepts_string_keys():
    raw = {
        "layers": {
            "3": {
                "delta": torch.ones(8),
                "mean_sys_key": torch.ones(2, 4),
                "d_bar": torch.tensor(0.0),
            }
        }
    }
    layers = normalize_gcad_layers(raw)
    assert list(layers) == [3]
    assert layers[3]["delta"].shape == (8,)


def test_compute_gcad_gate_returns_one_value_per_query():
    query = torch.ones(2, 3, 4)
    mean_sys_key = torch.ones(2, 4)
    gate = compute_gcad_gate(
        query,
        mean_sys_key,
        d_bar=0.0,
        c_base=2.0,
        sharpness=1.0,
    )
    assert gate.shape == (3,)
    assert torch.all(gate > 2.0)
