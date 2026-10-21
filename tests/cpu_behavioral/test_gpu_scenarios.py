"""Four real CUDA scenarios, using the same observations as numeric parity."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cross-platform-install'))
from run_parity import SCENARIOS, CASES, environment, prepare_manifest, run_logged

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module')
def gpu_run(tmp_path_factory):
    import torch
    from vllm.platforms import current_platform
    if not torch.cuda.is_available() or not current_platform.is_cuda():
        pytest.fail('These tests require CUDA vLLM and a visible GPU; no fallback or skip.')
    folder = tmp_path_factory.mktemp('gpu-scenarios')
    manifest = prepare_manifest(sys.executable, folder, 1800)
    return folder, manifest


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_gpu_scenario(scenario, gpu_run):
    folder, manifest = gpu_run
    output = folder / f'cuda-{scenario}.json'
    run_logged([sys.executable, '-u', str(CASES / 'parity_case.py'),
                '--scenario', scenario, '--backend', 'cuda',
                '--manifest', str(manifest), '--output', str(output)],
               environment('cuda', sys.executable), folder / f'cuda-{scenario}.log', 1800)
    result = json.loads(output.read_text())
    assert result['backend'] == 'cuda'
    assert result['values']
