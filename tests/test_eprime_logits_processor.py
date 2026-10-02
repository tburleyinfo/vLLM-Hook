import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_hook_plugins"
    / "vllm_hook_plugins"
    / "logits_processors"
    / "eprime.py"
)
spec = importlib.util.spec_from_file_location("eprime_logits_processor", MODULE_PATH)
eprime_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = eprime_module
spec.loader.exec_module(eprime_module)
EPrimeLogitsProcessor = eprime_module.EPrimeLogitsProcessor


class FakeTokenizer:
    vocab = {
        0: " harmless",
        1: " is",
        2: " was",
        3: " useful",
        4: "n't",
        5: " it's",
    }

    def decode(self, token_ids, **_kwargs):
        return "".join(self.vocab[int(i)] for i in token_ids)


def test_blocks_tokens_that_create_banned_forms():
    processor = EPrimeLogitsProcessor(FakeTokenizer())
    scores = torch.zeros(6)

    filtered = processor([0], scores)

    assert torch.isneginf(filtered[1])
    assert torch.isneginf(filtered[2])
    assert torch.isneginf(filtered[5])
    assert filtered[0].item() == 0.0
    assert filtered[3].item() == 0.0


def test_blocks_split_completion_across_token_boundary():
    processor = EPrimeLogitsProcessor(FakeTokenizer(), banned_forms=("is",))
    scores = torch.zeros(6)

    filtered = processor([0], scores)

    assert torch.isneginf(filtered[1])
    assert filtered[3].item() == 0.0
