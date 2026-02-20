# Attention Tracker Metal Implementation Gap Analysis

## Error Context

When running `demo_attntracker.py` on Apple Silicon with vLLM-Metal backend:

```
Processed prompts: 100%|███████████████████████████| 1/1 [00:04<00:00,  4.76s/it]
hook llm generation runtime: 5.979s
[rank0]: Traceback (most recent call last):
[rank0]:   File ".../demo_attntracker.py", line 98, in <module>
[rank0]:     stats = llm.analyze(analyzer_spec={'input_range': input_range, 'attn_func':"sum_normalize"})
[rank0]:   File ".../hook_llm.py", line 207, in analyze
[rank0]:     return self.analyzer.analyze(analyzer_spec)
[rank0]:   File ".../attention_tracker_analyzer.py", line 23, in analyze
[rank0]:     attention_weights = self.compute_attention_from_qk(run_id_file)
[rank0]:   File ".../attention_tracker_analyzer.py", line 34, in compute_attention_from_qk
[rank0]:     cache = load_and_merge_qk_cache(self.hook_dir, run_id)
[rank0]:   File ".../run_utils.py", line 39, in load_and_merge_qk_cache
[rank0]:     raise FileNotFoundError(
[rank0]: FileNotFoundError: No Q/K cache artifacts found for run_id=39852755-3156-49ac-8314-38acccae4e4c
                           under /Users/timothy.burley.ibm/.cache/vllm_hook/_v1_qk_peeks
```

## Root Cause: PyTorch Hook System vs MLX Model Architecture

### The Problem

The [`probe_hookqk_worker_metal.py`](../vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py) attempts to use PyTorch's hook mechanism on MLX models:

```python
# Lines 143-151 in probe_hookqk_worker_metal.py
for name, module in model.named_modules():  # ❌ FAILS: MLX models don't have this
    layer_num = match_attn(name)
    if layer_num is None:
        continue
    if layer_num not in self.important_layers:
        continue
    hook = module.register_forward_hook(lambda m, i, o, n=name: qkv_hook(i, n))  # ❌ FAILS
    self._hooks.append(hook)
    matched.append(name)
```

**Why This Fails:**

- MLX models loaded via `mlx_lm.load()` don't have `named_modules()` (PyTorch-specific)
- MLX models don't support `register_forward_hook()` (PyTorch-specific)
- MLX uses a different computation graph and module system

## Code Translation Map: vLLM → vLLM-Metal

### 1. Worker Base Class

| vLLM (GPU/CUDA) | vLLM-Metal (Apple Silicon) | Location |
|-----------------|----------------------------|----------|
| `from vllm.v1.worker.gpu_worker import Worker as V1Worker` | `from vllm_metal.v1.worker import MetalWorker` | Worker imports |
| `class ProbeHookQKWorker(V1Worker):` | `class ProbeHookQKWorkerMetal(MetalWorker):` | Class definition |

**Files:**

- vLLM: [`probe_hookqk_worker.py:5,27`](../vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py)
- vLLM-Metal: [`probe_hookqk_worker_metal.py:5,27`](../vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py)

### 2. Model Access

| vLLM (GPU/CUDA) | vLLM-Metal (Apple Silicon) | Notes |
|-----------------|----------------------------|-------|
| `model = self.model_runner.model` | `model = self.model_runner.model` | ✅ Same accessor |
| `cfg = model.config` | `model_args = self.model_runner.model_args` | ⚠️ **Different structure** |
| `model.named_modules()` | **NO EQUIVALENT** | ❌ **Missing in MLX** |
| `module.register_forward_hook()` | **NO EQUIVALENT** | ❌ **Missing in MLX** |

**Files:**

- vLLM: [`probe_hookqk_worker.py:40,61`](../vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py)
- vLLM-Metal: [`probe_hookqk_worker_metal.py:40,61-75`](../vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py)

### 3. Model Configuration Access

| vLLM (GPU/CUDA) | vLLM-Metal (Apple Silicon) | Implementation |
|-----------------|----------------------------|----------------|
| `cfg = model.config` | `model_args = self.model_runner.model_args` | Dictionary vs Object |
| `cfg.num_attention_heads` | `model_args.get("num_attention_heads", 32)` | Attribute vs Dict key |
| `cfg.hidden_size` | `model_args.get("hidden_size", 4096)` | Attribute vs Dict key |

**Example Translation:**

```python
# vLLM (GPU/CUDA) - probe_hookqk_worker.py:61-66
cfg = model.config
num_h = int(getattr(cfg, "num_attention_heads"))
num_kv = int(getattr(cfg, "num_key_value_heads", num_h))
hidden = int(getattr(cfg, "hidden_size"))
head_dim = hidden // num_h
attn_mult = float(getattr(cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))

# vLLM-Metal (Apple Silicon) - probe_hookqk_worker_metal.py:70-75
model_args = getattr(self.model_runner, "model_args", {})
num_h = int(model_args.get("num_attention_heads", 32))
num_kv = int(model_args.get("num_key_value_heads", num_h))
hidden = int(model_args.get("hidden_size", 4096))
head_dim = model_args.get("head_dim", hidden // num_h)
attn_mult = float(model_args.get("attention_multiplier", 1 / math.sqrt(head_dim)))
```

### 4. Model Runner Structure

| Component | vLLM Location | vLLM-Metal Location |
|-----------|---------------|---------------------|
| Worker class | `vllm.v1.worker.gpu_worker` | `vllm_metal.v1.worker` |
| Model runner | `vllm.v1.model_runner` | `vllm_metal.v1.model_runner` |
| Model loading | PyTorch `torch.load()` | MLX `mlx_lm.load()` |
| Model type | PyTorch `nn.Module` | MLX model (no direct equivalent) |
| Config storage | `model.config` (object) | `model_runner.model_args` (dict) |

**Files:**

- vLLM-Metal Worker: [`vllm-metal/vllm_metal/v1/worker.py`](../../vllm-metal/vllm_metal/v1/worker.py)
- vLLM-Metal Model Runner: [`vllm-metal/vllm_metal/v1/model_runner.py`](../../vllm-metal/vllm_metal/v1/model_runner.py)

## The Missing Piece: Hook Installation

### What vLLM Does (GPU/CUDA)

```python
# probe_hookqk_worker.py:143-151
for name, module in model.named_modules():
    layer_num = match_attn(name)
    if layer_num is None:
        continue
    if layer_num not in self.important_layers:
        continue
    hook = module.register_forward_hook(lambda m, i, o, n=name: qkv_hook(i, n))
    self._hooks.append(hook)
    matched.append(name)
```

**How it works:**

1. Iterates through PyTorch model's module hierarchy
2. Finds attention layers by name pattern matching
3. Registers forward hooks on those modules
4. Hooks capture Q/K tensors during forward pass
5. Saves tensors to disk as PyTorch `.pt` files

### What vLLM-Metal Needs (Apple Silicon)

**Current Status:** ❌ **NOT IMPLEMENTED**

**Why:** MLX models don't expose:

- Module hierarchy via `named_modules()`
- Hook registration via `register_forward_hook()`
- Individual layer access for interception

### Required Implementation Strategy

To make attention tracking work on Metal, we need to:

#### Option A: Wrap MLX Model Forward Pass

```python
# Pseudocode - NOT YET IMPLEMENTED
class MLXModelWrapper:
    def __init__(self, mlx_model, hook_layers):
        self.mlx_model = mlx_model
        self.hook_layers = hook_layers
        self.cache = {}

    def __call__(self, *args, **kwargs):
        # Intercept model forward pass
        # Extract Q/K from attention layers
        # Save to cache
        output = self.mlx_model(*args, **kwargs)
        return output
```

#### Option B: Modify Model Runner

Modify [`vllm_metal.v1.model_runner.MetalModelRunner`](../../vllm-metal/vllm_metal/v1/model_runner.py) to:

1. **Capture attention outputs** during `execute_model()`
2. **Extract Q/K tensors** from MLX computation graph
3. **Convert MLX arrays to PyTorch** for saving
4. **Save artifacts** in same format as GPU worker

```python
# Pseudocode - NOT YET IMPLEMENTED
class MetalModelRunner:
    def execute_model(self, ...):
        # Existing model execution
        output = self.model(...)

        # NEW: If hooks enabled, capture Q/K
        if self.hooks_enabled:
            qk_data = self._extract_qk_from_output(output)
            self._save_qk_cache(qk_data)

        return output
```

#### Option C: MLX Model Instrumentation

Directly modify MLX model layers to capture Q/K:

```python
# Pseudocode - NOT YET IMPLEMENTED
def instrument_mlx_attention_layers(model, layer_indices):
    """Add Q/K capture to MLX attention layers."""
    for layer_idx in layer_indices:
        original_attn = model.layers[layer_idx].self_attn
        model.layers[layer_idx].self_attn = QKCapturingAttention(original_attn)
```

## Translation Requirements for Full Implementation

### 1. Tensor Conversion

| Operation | vLLM (PyTorch) | vLLM-Metal (MLX) |
|-----------|----------------|------------------|
| Tensor type | `torch.Tensor` | `mx.array` |
| Move to CPU | `tensor.cpu()` | `mx.eval(tensor)` then convert |
| Save to disk | `torch.save()` | Convert to PyTorch first |
| Conversion | N/A | `mlx_to_torch()` from `vllm_metal.pytorch_backend.tensor_bridge` |

**Example:**

```python
# vLLM (GPU/CUDA)
q_cpu = q_tensor.cpu()
torch.save(cache, cache_path)

# vLLM-Metal (Apple Silicon) - NEEDS IMPLEMENTATION
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
q_mlx = mx.array(...)  # MLX tensor
q_torch = mlx_to_torch(q_mlx)  # Convert to PyTorch
q_cpu = q_torch.cpu()
torch.save(cache, cache_path)
```

**Reference:** [`vllm-metal/vllm_metal/pytorch_backend/tensor_bridge.py`](../../vllm-metal/vllm_metal/pytorch_backend/tensor_bridge.py)

### 2. Forward Context

| Component | vLLM (GPU/CUDA) | vLLM-Metal (Apple Silicon) | Status |
|-----------|----------------|----------------------------|--------|
| Context retrieval | `get_forward_context()` | `get_forward_context()` | ✅ Same API |
| Metadata access | `ctx.attn_metadata` | `ctx.attn_metadata` | ⚠️ Needs verification |
| Sequence lengths | `metadata.seq_lens` | `metadata.seq_lens` | ⚠️ Needs verification |

**File:** Both use `vllm.forward_context.get_forward_context`

### 3. Artifact Schema (Must Match)

Both backends MUST produce identical artifact format:

```python
{
    "config": {
        "num_attention_heads": int,
        "num_key_value_heads": int,
        "hidden_size": int,
        "head_dim": int,
        "attention_multiplier": float
    },
    "qk_cache": {
        "<layer_name>": {
            "q": List[torch.Tensor],      # Query tensors (PyTorch format)
            "k_all": List[torch.Tensor],  # Key tensors (PyTorch format)
            "layer_num": int              # Layer index
        }
    }
}
```

**Critical:** Even though Metal uses MLX internally, artifacts must be saved as PyTorch tensors for analyzer compatibility.

## Implementation Checklist

### Phase 1: Understand MLX Model Structure

- [ ] Document how MLX models expose layers
- [ ] Identify where attention computation happens in MLX
- [ ] Determine if MLX provides any hook/callback mechanism
- [ ] Map MLX layer names to vLLM attention patterns

### Phase 2: Design Hook Mechanism

- [ ] Choose implementation strategy (A, B, or C above)
- [ ] Design Q/K tensor capture approach
- [ ] Plan MLX-to-PyTorch conversion pipeline
- [ ] Ensure artifact schema compatibility

### Phase 3: Implement Metal Hooks

- [ ] Modify `probe_hookqk_worker_metal.py` to use new approach
- [ ] Implement Q/K extraction from MLX model
- [ ] Add MLX-to-PyTorch tensor conversion
- [ ] Preserve artifact format for analyzer compatibility

### Phase 4: Test and Validate

- [ ] Verify hooks capture Q/K tensors correctly
- [ ] Confirm artifacts match GPU worker format
- [ ] Test analyzer works with Metal-generated artifacts
- [ ] Validate end-to-end attention tracking on Apple Silicon

## Key Differences Summary

| Aspect | vLLM (GPU/CUDA) | vLLM-Metal (Apple Silicon) |
|--------|----------------|----------------------------|
| **Model Type** | PyTorch `nn.Module` | MLX model (custom) |
| **Config Access** | `model.config` (object) | `model_runner.model_args` (dict) |
| **Module Iteration** | `model.named_modules()` | ❌ Not available |
| **Hook Registration** | `register_forward_hook()` | ❌ Not available |
| **Tensor Type** | `torch.Tensor` | `mx.array` |
| **Tensor Conversion** | N/A | `mlx_to_torch()` required |
| **Hook Mechanism** | PyTorch hooks | ❌ **NEEDS IMPLEMENTATION** |

## Next Steps

1. **Research MLX model internals** to understand layer access
2. **Prototype Q/K capture** using one of the three strategies
3. **Implement tensor conversion** from MLX to PyTorch
4. **Test artifact compatibility** with existing analyzers
5. **Document Metal-specific limitations** and workarounds

## References

- vLLM Worker: [`vllm_hook_plugins/workers/probe_hookqk_worker.py`](../vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py)
- Metal Worker: [`vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py`](../vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py)
- Metal Model Runner: [`vllm-metal/vllm_metal/v1/model_runner.py`](../../vllm-metal/vllm_metal/v1/model_runner.py)
- Tensor Bridge: [`vllm-metal/vllm_metal/pytorch_backend/tensor_bridge.py`](../../vllm-metal/vllm_metal/pytorch_backend/tensor_bridge.py)
- Migration Guide: [`vllm_to_vllm_metal_migration_guide.md`](vllm_to_vllm_metal_migration_guide.md)
