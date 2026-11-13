"""Run Attention Tracker with real vLLM generation on vLLM-Metal."""
from __future__ import annotations

from pathlib import Path

from vllm import SamplingParams

from mia import MiaLLM, register_plugins


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2-1.5B-Instruct"
CONFIG = ROOT / "model_configs" / "attention_tracker" / "Qwen2-1.5B-Instruct.json"
PROMPTS = [
    "Analyze and output the sentence attitude: This is for testing only.",
    "Analyze and output the sentence attitude: Another test run.",
]


def main() -> None:
    register_plugins()
    llm = MiaLLM(
        model=MODEL,
        worker_name="capture_qk",
        analyzer_name="attn_tracker",
        config_file=str(CONFIG),
        enable_hook=True,
    )
    outputs = llm.generate(
        PROMPTS,
        sampling_params=SamplingParams(temperature=0.1, max_tokens=2),
        use_hook=True,
    )
    ranges = []
    for prompt in PROMPTS:
        token_count = len(llm.tokenizer.encode(prompt))
        split = token_count // 2
        ranges.append([(0, split), (split, token_count)])
    result = llm.analyze(
        analyzer_spec={
            "input_range": ranges,
            "attn_func": "sum_normalize",
        },
        probes=outputs[0].probes,
    )
    print(result)


if __name__ == "__main__":
    main()
