import json
import os
import sys
from types import ModuleType

import numpy as np
import torch

from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from mia.backends.metal.attntracker import (
    MetalQKCaptureSession,
    write_qk_artifact,
)


def test_metal_qk_capture_uses_mia_probe_contract(monkeypatch):
    mlx_core = ModuleType("mlx.core")
    mlx_core.float32 = np.float32
    mlx_core.eval = lambda *arrays: None
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    mlx.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    config = {
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "hidden_size": 4,
        "head_dim": 2,
        "attention_multiplier": 2**-0.5,
    }
    session = MetalQKCaptureSession({0: [1]}, config)
    query = np.arange(8, dtype=np.float16).reshape(1, 2, 2, 2)
    keys = np.arange(8, dtype=np.float16).reshape(1, 1, 4, 2)

    with session:
        session.capture(0, query, keys)
    [probes] = session.probes()

    assert probes["config"] == config
    entry = probes["qk_cache"][0]
    assert entry["layer_num"] == 0
    assert entry["heads"] == [1]
    assert entry["hookq_mode"] == "last_token"
    assert torch.equal(entry["q"][0], torch.tensor([2.0, 3.0, 6.0, 7.0]))
    assert entry["k_all"][0].shape == (4, 2)
    assert torch.equal(entry["k_all"][0], torch.from_numpy(keys[0, 0].astype(np.float32)))
    scores = AttntrackerAnalyzer("", {0: [1]}).analyze(
        analyzer_spec={
            "input_range": [((0, 2), (2, 4))],
            "attn_func": "sum_normalize",
        },
        probes=probes,
    )["score"]
    assert len(scores) == 1
    assert np.isfinite(scores[0])


def test_metal_qk_capture_fails_if_a_configured_layer_was_not_seen():
    session = MetalQKCaptureSession({4: [2]}, {})
    try:
        session.probes()
    except RuntimeError as exc:
        assert "configured layers [4]" in str(exc)
    else:
        raise AssertionError("missing layer capture must not produce a success-shaped payload")


def test_metal_qk_artifact_uses_existing_ranked_layout(tmp_path):
    probes = {"config": {"head_dim": 2}, "qk_cache": {0: {"layer_num": 0}}}

    run_dir = write_qk_artifact(probes, str(tmp_path), "run-1")

    assert run_dir.endswith("run-1/tp_rank_0")
    saved = torch.load(f"{run_dir}/qk.pt", map_location="cpu", weights_only=False)
    assert saved == probes

    refused = f"{run_dir}/qk_refused.json"
    with open(refused, "w", encoding="utf-8") as stream:
        json.dump({"old-request": "old refusal"}, stream)
    replacement = {"config": {"head_dim": 4}, "qk_cache": {1: {"layer_num": 1}}}
    write_qk_artifact(replacement, str(tmp_path), "run-1")
    saved = torch.load(f"{run_dir}/qk.pt", map_location="cpu", weights_only=False)
    assert saved == replacement
    assert not os.path.exists(refused)
