# Apple Silicon / Metal Limitations

## Current Status

As of the current version, **attention tracker hooks are NOT yet implemented for Apple Silicon/Metal/MLX**.

## The Problem

When running `demo_attntracker.py` on Apple Silicon, you'll encounter:

```
FileNotFoundError: No Q/K cache artifacts found for run_id=... under ~/.cache/vllm_hook/_v1_qk_peeks
```

This occurs because:

1. The Metal worker uses **MLX models** (not PyTorch models)
2. MLX models don't support PyTorch's hook system (`named_modules()`, `register_forward_hook()`)
3. The current hook implementation in `probe_hookqk_worker_metal.py` tries to use PyTorch APIs on MLX models
4. The hooks fail to install, so no Q/K cache files are created
5. The analyzer then fails when trying to load non-existent cache files

## Code Evidence

From `hook_llm_mlx.py` lines 60-65:

```python
is_mlx_host = _is_apple_silicon()
if is_mlx_host and enable_hook:
    raise NotImplementedError(
        "Custom vLLM worker hooks are not MLX-compatible yet. "
        "Use enable_hook=False on Apple Silicon for now."
    )
```

## Why Hooks Don't Work

The hook installation code in `probe_hookqk_worker_metal.py` (lines 143-151) attempts:

```python
for name, module in model.named_modules():  # ❌ MLX models don't have named_modules()
    layer_num = match_attn(name)
    if layer_num is None:
        continue
    if layer_num not in self.important_layers:
        continue
    hook = module.register_forward_hook(...)  # ❌ MLX modules don't support PyTorch hooks
    self._hooks.append(hook)
```

MLX models have a completely different architecture and don't expose the same module hierarchy as PyTorch.

## Workarounds

### Option 1: Use GPU/CUDA Machine (Recommended)

Run attention tracker demos on a machine with NVIDIA GPU support where PyTorch hooks work natively.

### Option 2: Wait for MLX Implementation

The MLX hook implementation would require:

1. **MLX-native hook system** - Intercept MLX model forward passes
2. **Capture Q/K tensors** during attention computation in MLX
3. **Convert MLX arrays to PyTorch** for saving/analysis
4. **Modify model runner** to expose attention internals

This is a significant development effort.

### Option 3: Use Alternative Analysis Methods

Consider using:

- Model interpretability tools that work with MLX
- Post-hoc analysis of model outputs
- Gradient-based attribution methods

## What Works on Apple Silicon

The following vLLM-Hook features **DO work** on Apple Silicon:

- ✅ Basic text generation with `HookLLM` (with `enable_hook=False`)
- ✅ Model loading and inference via MLX
- ✅ Sampling and generation parameters
- ✅ Tokenization

The following features **DO NOT work** on Apple Silicon:

- ❌ Attention tracking (`demo_attntracker.py`)
- ❌ Q/K cache extraction
- ❌ Activation steering (likely has similar issues)
- ❌ Any feature requiring custom worker hooks

## Technical Details

### Why the Config Fix Didn't Work

The fix applied to `probe_hookqk_worker_metal.py` (accessing `model_args` instead of `model.config`) solved the **configuration access issue** but didn't solve the **fundamental hook installation problem**.

The hooks still can't be installed because:

- MLX models don't have a `named_modules()` method
- MLX modules don't support `register_forward_hook()`
- The entire hook mechanism is PyTorch-specific

### What Would Be Needed

To implement MLX hooks properly:

```python
# Pseudocode for MLX hook implementation
class MLXAttentionHook:
    def __init__(self, model, layer_indices):
        self.model = model
        self.layer_indices = layer_indices
        self.cache = {}

    def __call__(self, *args, **kwargs):
        # Intercept MLX model forward pass
        # Extract Q, K tensors from attention layers
        # Convert mx.array to torch.Tensor
        # Save to cache
        pass
```

This would require deep integration with MLX's computation graph and model architecture.

## Conclusion

**Attention tracking is not currently supported on Apple Silicon.** Use a CUDA-enabled machine for these features, or wait for MLX hook support to be implemented in a future version.
