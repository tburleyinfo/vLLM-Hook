import pytest
import sys
from pathlib import Path


def test_local_mlx_lm_injected_error_surfaces_via_vllm_metal():
    """Metal-only smoke test for local mlx-lm wiring via vllm-metal."""
    local_vllm_metal = Path("/Users/timothyburley/opensource/vllm-metal")
    sys.path.insert(0, str(local_vllm_metal))
    pytest.importorskip("vllm_metal")
    from vllm_metal.v1 import model_runner as mr

    class DummyModel:
        layers = []

    with pytest.raises(RuntimeError, match="LOCAL-MLX-LM-TEST"):
        mr.make_prompt_cache(DummyModel())
