"""Attention Tracker on vLLM-Metal using the repository's existing MiaLLM API."""
import multiprocessing as mp
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "0")
os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "auto")
os.environ.setdefault("VLLM_PLUGINS", "metal")

from vllm import SamplingParams

from mia import MiaLLM, register_plugins
from mia.core._plugin import register as register_mia_plugin


MODEL = "RedHatAI/granite-3.1-2b-instruct-quantized.w4a16"
CONFIG = (
    "model_configs/attention_tracker/"
    "granite-3.1-2b-instruct-quantized.w4a16.json"
)


def main():
    register_plugins()
    register_mia_plugin()
    llm = MiaLLM(
        model=MODEL,
        worker_name="capture_qk",
        analyzer_name="attn_tracker",
        config_file=CONFIG,
        hook_dir=os.path.expanduser("~/.cache/mia"),
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=2048,
    )
    instruction = "Analyze and output the sentence attitude:"
    data = "The weather is nice today, with a soft breeze and sunshine."
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": "Data: " + data},
    ]
    text = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    instruction_len = len(llm.tokenizer.encode(instruction))
    data_len = len(llm.tokenizer.encode(data))
    input_range = ((3, 3 + instruction_len), (-5 - data_len, -5))

    try:
        output = llm.generate(
            text,
            SamplingParams(temperature=0.1, max_tokens=32),
            use_hook=True,
        )
        stats = llm.analyze(
            probes=output[0].probes,
            analyzer_spec={
                "input_range": input_range,
                "attn_func": "sum_normalize",
            },
        )
        print(output[0].outputs[0].text)
        print(f"Attention Tracker score: {stats['score'][0]:.3f}")
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
