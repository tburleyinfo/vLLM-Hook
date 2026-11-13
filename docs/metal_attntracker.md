# Attention Tracker on Apple Silicon

The `metal` optional dependency profile installs vLLM-Metal alongside the
existing CPU/CUDA profiles. Run generation with vLLM's normal `LLM` API and
select the Metal platform using the vLLM-Metal plugin:

```sh
uv sync --extra metal --group test
export VLLM_PLUGINS=metal
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_USE_PAGED_ATTENTION=0
export VLLM_METAL_MEMORY_FRACTION=auto
```

`notebooks/demo_attntracker_colab.ipynb` uses `MiaLLM(worker_name="capture_qk",
analyzer_name="attn_tracker")` with the config
`model_configs/attention_tracker/Meta-Llama-3-8B-Instruct.json`. That exact
public-API path is **not yet connected to vLLM-Metal**: MIA's current worker
expects a vLLM V2 PyTorch runner, while vLLM-Metal loads an MLX model, and the
requested `VLLM_METAL_USE_PAGED_ATTENTION=0` path does not call Metal's paged
SDPA wrapper. Do not interpret the adapter below as completing that path.

The current Q/K adapter is limited to vLLM-Metal's paged SDPA wrapper and
requires the generation forward to run in the current Python execution
context. To exercise that primitive, paged attention must be enabled for the
run (`VLLM_METAL_USE_PAGED_ATTENTION=1`); this differs from the requested
target environment above:

```python
import json
import math
from pathlib import Path

from vllm import LLM, SamplingParams

from mia.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from mia.backends.metal import MetalQKCapture, capture_vllm_metal_attention

model = "meta-llama/Meta-Llama-3-8B-Instruct"
config_path = Path("model_configs/attention_tracker/Meta-Llama-3-8B-Instruct.json")
attention_tracker_config = json.loads(config_path.read_text())
heads = attention_tracker_config["params"]["important_heads"]
layer_to_heads = {}
for layer, head in heads:
    layer_to_heads.setdefault(layer, []).append(head)

llm = LLM(
    model=model,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    enable_prefix_caching=False,
)
tokenizer = llm.get_tokenizer()
hf_config = llm.llm_engine.model_config.hf_text_config
head_dim = int(hf_config.hidden_size) // int(hf_config.num_attention_heads)
capture = MetalQKCapture(
    {
        "num_attention_heads": int(hf_config.num_attention_heads),
        "num_key_value_heads": int(hf_config.num_key_value_heads),
        "head_dim": head_dim,
        "attention_multiplier": float(
            getattr(hf_config, "attention_multiplier", 1 / math.sqrt(head_dim))
        ),
    },
    hookq_mode=attention_tracker_config["hookq"]["hookq_mode"],
)
instruction = "Analyze and output the sentence attitude:"
data = "The weather is nice today, with a soft breeze and sunshine."
prompt = f"{instruction}\nData: {data}"
token_end = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
input_range = (
    (0, token_end(instruction)),
    (token_end(f"{instruction}\nData: "), token_end(prompt)),
)

with capture_vllm_metal_attention(capture):
    output = llm.generate(
        [prompt],
        SamplingParams(max_tokens=1, temperature=0.1),
    )
probes = capture.probes()
scores = AttntrackerAnalyzer("", layer_to_heads).analyze(
    analyzer_spec={"input_range": input_range, "attn_func": "sum_normalize"},
    probes=probes,
)
print(output[0].outputs[0].text)
print(scores["score"])
```

This exercises real vLLM generation through vLLM-Metal and does not change
model outputs or run a second inference stack. Chat-template token range
recipes and the two-case notebook are not yet wired to this primitive. It
requires same-context instrumentation and paged SDPA, so it is not compatible
with the target environment above. See `TRANSLATION.md` for the precise gaps.
