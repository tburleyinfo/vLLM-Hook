# Attention Tracker on vLLM-Metal

The existing notebook's `MiaLLM` calls and analyzer can be retained. The model
configuration is
`model_configs/attention_tracker/Qwen2-1.5B-Instruct.json`; the example
arguments remain `worker_name="capture_qk"`, `analyzer_name="attn_tracker"`,
`config_file=<that JSON>`, and `max_model_len=2048`.

On Apple Silicon, provision the repository's `metal` optional dependency
profile, then set the target runtime configuration before Python imports vLLM:

```sh
MIA_DEMO_MODEL=Qwen/Qwen2-1.5B-Instruct \
MIA_CONFIG_FILE=model_configs/attention_tracker/Qwen2-1.5B-Instruct.json \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_METAL_USE_PAGED_ATTENTION=0 \
VLLM_METAL_MEMORY_FRACTION=auto \
VLLM_PLUGINS=metal \
python examples/demo_attntracker.py
```

**Important:** this command documents the intended invocation, not a working
end-to-end path yet. The repository's current `capture_qk` worker requires a
PyTorch V2 model runner and CUDA attention-module hooks. The target Qwen2 model
uses MLX arrays and the inspected vLLM-Metal runner does not expose that worker
extension surface. `mia.backends.metal_qk` implements conversion of post-RoPE
Q/K arrays to the existing analyzer probe contract, but installing that adapter
in the Metal runner and associating its captures with request outputs remains
unfinished. The command must not be interpreted as runtime validation.
