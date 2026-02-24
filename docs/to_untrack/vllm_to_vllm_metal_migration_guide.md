# vLLM to vLLM-Metal Migration Guide: File-Level Implementation Strategy

## Executive Summary

This document provides a concrete, file-by-file approach to migrate vLLM-Hook from using the standard vLLM library to vLLM-Metal for Apple Silicon support. The strategy focuses on **replacing vLLM imports with vLLM-Metal equivalents** while maintaining the existing hook architecture and artifact contract.

## Prerequisite: Install vLLM-Metal First

Install vLLM-Metal before implementing backend negotiation and Metal workers.
Run the install command from the `vllm-metal` repository directory (or pass that path explicitly), not from `vLLM-Hook`.
Use Python `>=3.12,<3.14` for `vllm-metal`; Python 3.11 is not supported.

```bash
# Option A: from vllm-metal directory
cd /Users/timothyburley/opensource/vllm-metal
# If vLLM is available on your platform:
pip install -e '.[vllm]'

# On Apple Silicon/macOS, use the installer flow if pip vLLM install fails:
./install.sh

# Option B: from vLLM-Hook directory (explicit path)
cd /Users/timothyburley/opensource/vLLM-Hook
pip install -e ../vllm-metal
pip install -r requirement.txt
pip install -e vllm_hook_plugins
```

If your environment is offline or build-isolated, ensure `maturin` is available locally before running the command.

## Apple Silicon Notebook Environment Flow (Moved from README)

Use this when `demo_attntracker.ipynb` must run from the same environment that installs `vllm-metal`.

From `vLLM-Hook/`:

```bash
./scripts/setup_attntracker_metal_env.sh
```

What it does:
- runs `vllm-metal/install.sh` from the `vllm-metal` repo directory
- creates/uses `vllm-metal/.venv-vllm-metal`
- installs `vLLM-Hook` requirements and `vllm_hook_plugins` into that env
- installs `ipykernel`/`jupyterlab` and registers kernel `vllm-metal-attntracker`

Launch notebook:

```bash
./scripts/run_attntracker_notebook.sh
```

In Jupyter:

```text
Kernel → Change Kernel → vllm-metal-attntracker
```

If `vllm-metal` is not in `../vllm-metal`, set:

```bash
VLLM_METAL_REPO=/absolute/path/to/vllm-metal ./scripts/setup_attntracker_metal_env.sh
```

## Key Insight from Code Analysis

The TODO comment in [`hook_llm.py:48`](my_fork_of_vllm_hook/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/hook_llm.py:48) states:

```python
#TODO: Backend Negotiation needs to happen here.
```

This is the **critical integration point** where platform detection and backend selection must occur.

## Current Architecture Analysis

### Files That Import vLLM Directly

1. **[`hook_llm.py:8`](my_fork_of_vllm_hook/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/hook_llm.py:8)**

   ```python
   from vllm import LLM, SamplingParams
   ```

2. **[`probe_hookqk_worker.py:5`](my_fork_of_vllm_hook/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py:5)**

   ```python
   from vllm.v1.worker.gpu_worker import Worker as V1Worker
   from vllm.forward_context import get_forward_context
   from vllm.distributed import parallel_state as ps
   ```

3. **[`steer_activation_worker.py:5`](my_fork_of_vllm_hook/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py:5)**

   ```python
   from vllm.v1.worker.gpu_worker import Worker as V1Worker
   ```

### vLLM-Metal Equivalent Modules

Based on [`vllm-metal`](my_fork_of_vllm_hook/vllm-metal) structure:

- **Worker**: [`vllm_metal.v1.worker.MetalWorker`](my_fork_of_vllm_hook/vllm-metal/vllm_metal/v1/worker.py:75) replaces `vllm.v1.worker.gpu_worker.Worker`
- **Platform**: [`vllm_metal.platform.MetalPlatform`](my_fork_of_vllm_hook/vllm-metal/vllm_metal/platform.py:22) for platform detection
- **Config**: [`vllm_metal.config.MetalConfig`](my_fork_of_vllm_hook/vllm-metal/vllm_metal/__init__.py:48) for Metal-specific configuration

## Migration Strategy: Three-Phase Approach

### Phase 1: Backend Negotiation in `hook_llm.py`

**Objective**: Implement platform detection and dynamic import selection at the TODO location.

**File**: [`hook_llm.py`](my_fork_of_vllm_hook/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/hook_llm.py)

**Changes Required**:

1. **Add platform detection logic** (before line 48):

   ```python
   import platform
   import sys

   def _detect_backend():
       """Detect whether to use vLLM or vLLM-Metal."""
       if sys.platform == "darwin" and platform.machine() == "arm64":
           try:
               import vllm_metal
               if vllm_metal.MetalPlatform.is_available():
                   return "metal"
           except ImportError:
               pass
       return "vllm"
   ```

2. **Replace static import at line 8** with conditional import:

   ```python
   # Replace:
   # from vllm import LLM, SamplingParams

   # With:
   BACKEND = _detect_backend()

   if BACKEND == "metal":
       # vLLM-Metal uses the same LLM interface but through vllm_metal
       from vllm import LLM, SamplingParams
       # Metal platform will be auto-registered via plugin system
   else:
       from vllm import LLM, SamplingParams
   ```

3. **Store backend info** in `__init__` (after line 28):

   ```python
   self.backend = BACKEND
   os.environ["VLLM_HOOK_BACKEND"] = BACKEND
   ```

4. **Modify worker selection logic** (at line 48-56):

   ```python
   #TODO: Backend Negotiation needs to happen here.
   worker = None
   if worker_name:
       import vllm.plugins
       from vllm_hook_plugins import PluginRegistry
       vllm.plugins.load_general_plugins()

       # Get the appropriate worker class based on backend
       if self.backend == "metal":
           worker = PluginRegistry.get_worker(f"{worker_name}_metal").path
       else:
           worker = PluginRegistry.get_worker(worker_name).path
   ```

### Phase 2: Create Metal-Specific Workers

**Objective**: Create Metal variants of existing workers that inherit from `MetalWorker` instead of `V1Worker`.

#### 2.1 Create `workers/metal/` Directory Structure

```
vllm_hook_plugins/vllm_hook_plugins/workers/
├── __init__.py
├── probe_hookqk_worker.py          # Existing CUDA/GPU worker
├── steer_activation_worker.py      # Existing CUDA/GPU worker
└── metal/
    ├── __init__.py
    ├── probe_hookqk_worker_metal.py
    └── steer_activation_worker_metal.py
```

#### 2.2 Create `probe_hookqk_worker_metal.py`

**File**: `vllm_hook_plugins/vllm_hook_plugins/workers/metal/probe_hookqk_worker_metal.py`

**Key Changes from Original**:

1. **Import MetalWorker instead of V1Worker**:

   ```python
   # Replace:
   # from vllm.v1.worker.gpu_worker import Worker as V1Worker

   # With:
   from vllm_metal.v1.worker import MetalWorker
   ```

2. **Inherit from MetalWorker**:

   ```python
   class ProbeHookQKWorkerMetal(MetalWorker):
   ```

3. **Adapt tensor operations for MLX compatibility**:
   - Keep PyTorch for artifact saving (line 133)
   - Ensure `.cpu()` calls work with Metal tensors
   - Add MLX-to-PyTorch conversion if needed

4. **Handle Metal-specific context**:
   - Verify `get_forward_context()` works with Metal backend
   - May need Metal-specific context retrieval

**Full Implementation Pattern**:

```python
import os
import math
import torch
from typing import Dict, List
from vllm_metal.v1.worker import MetalWorker
from vllm.forward_context import get_forward_context
import re
from vllm.distributed import parallel_state as ps

# ... [Keep ATTN_PATTERNS and match_attn function unchanged] ...

class ProbeHookQKWorkerMetal(MetalWorker):

    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)

        try:
            self._install_hooks()
            print("Metal hooks installed successfully")
        except Exception as e:
            print(f"Metal hook installation failed: {e}")

        return r

    # ... [Rest of implementation identical to probe_hookqk_worker.py] ...
    # Key: Ensure tensor operations are compatible with Metal/MLX tensors
```

#### 2.3 Create `steer_activation_worker_metal.py`

**File**: `vllm_hook_plugins/vllm_hook_plugins/workers/metal/steer_activation_worker_metal.py`

**Key Changes**:

1. **Import MetalWorker**:

   ```python
   from vllm_metal.v1.worker import MetalWorker
   ```

2. **Inherit from MetalWorker**:

   ```python
   class SteerHookActWorkerMetal(MetalWorker):
   ```

3. **Ensure steering vector operations work with Metal tensors**:
   - Line 54: `.to(residuals.device, dtype=residuals.dtype)` should work
   - Verify tensor arithmetic is compatible

### Phase 3: Update Plugin Registry

**Objective**: Register Metal workers alongside existing workers.

**File**: [`__init__.py`](my_fork_of_vllm_hook/vLLM-Hook/vllm_hook_plugins/vllm_hook_plugins/__init__.py)

**Changes Required**:

1. **Import Metal workers** (after line 6):

   ```python
   # Existing imports
   from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
   from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker

   # Add Metal worker imports
   try:
       from vllm_hook_plugins.workers.metal.probe_hookqk_worker_metal import ProbeHookQKWorkerMetal
       from vllm_hook_plugins.workers.metal.steer_activation_worker_metal import SteerHookActWorkerMetal
       METAL_AVAILABLE = True
   except ImportError:
       METAL_AVAILABLE = False
   ```

2. **Register Metal workers** (in `register_plugins()` function, after line 16):

   ```python
   def register_plugins():
       # Register CUDA/GPU workers
       PluginRegistry.register_worker("probe_hook_qk", ProbeHookQKWorker)
       PluginRegistry.register_worker("steer_hook_act", SteerHookActWorker)

       # Register Metal workers if available
       if METAL_AVAILABLE:
           PluginRegistry.register_worker("probe_hook_qk_metal", ProbeHookQKWorkerMetal)
           PluginRegistry.register_worker("steer_hook_act_metal", SteerHookActWorkerMetal)

       # Register analyzers (unchanged)
       PluginRegistry.register_analyzer("attn_tracker", AttntrackerAnalyzer)
       PluginRegistry.register_analyzer("core_reranker", CorerAnalyzer)
   ```

3. **Update `__all__`** (after line 25):

   ```python
   __all__ = [
       "PluginRegistry",
       "HookLLM",
       "ProbeHookQKWorker",
       "SteerHookActWorker",
       "AttntrackerAnalyzer",
       "CorerAnalyzer",
       "register_plugins"
   ]

   if METAL_AVAILABLE:
       __all__.extend([
           "ProbeHookQKWorkerMetal",
           "SteerHookActWorkerMetal"
       ])
   ```

## Implementation Checklist

### Critical Files to Modify

- [ ] **`hook_llm.py`**
  - [ ] Add `_detect_backend()` function
  - [ ] Implement conditional imports based on backend
  - [ ] Store backend in environment variable
  - [ ] Modify worker selection logic (line 48-56)

- [ ] **`workers/metal/probe_hookqk_worker_metal.py`** (NEW)
  - [ ] Create file
  - [ ] Import `MetalWorker` from `vllm_metal.v1.worker`
  - [ ] Copy logic from `probe_hookqk_worker.py`
  - [ ] Adapt for Metal tensor compatibility
  - [ ] Test hook installation and artifact generation

- [ ] **`workers/metal/steer_activation_worker_metal.py`** (NEW)
  - [ ] Create file
  - [ ] Import `MetalWorker` from `vllm_metal.v1.worker`
  - [ ] Copy logic from `steer_activation_worker.py`
  - [ ] Ensure steering operations work with Metal tensors

- [ ] **`workers/metal/__init__.py`** (NEW)
  - [ ] Create file
  - [ ] Export Metal worker classes

- [ ] **`__init__.py`**
  - [ ] Add conditional Metal worker imports
  - [ ] Register Metal workers in `register_plugins()`
  - [ ] Update `__all__` exports

### Testing Strategy

1. **Platform Detection Test**:

   ```python
   # Test that backend detection works correctly
   from vllm_hook_plugins.hook_llm import _detect_backend
   assert _detect_backend() in ["vllm", "metal"]
   ```

2. **Worker Registration Test**:

   ```python
   # Test that Metal workers are registered on Apple Silicon
   from vllm_hook_plugins import PluginRegistry
   if platform.machine() == "arm64":
       assert PluginRegistry.get_worker("probe_hook_qk_metal") is not None
   ```

3. **End-to-End Test**:

   ```python
   # Test that HookLLM works with Metal backend
   from vllm_hook_plugins import HookLLM

   llm = HookLLM(
       model="gpt2",
       worker_name="probe_hook_qk",
       enable_hook=True
   )

   # Should automatically use Metal worker on Apple Silicon
   output = llm.generate(["Hello world"], use_hook=True)
   ```

## Artifact Contract Preservation

**Critical**: Both CUDA and Metal workers MUST produce identical artifact schemas.

### Artifact Schema (from `probe_hookqk_worker.py:106-133`)

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
        "<module_name>": {
            "q": List[torch.Tensor],      # Query tensors
            "k_all": List[torch.Tensor],  # Key tensors
            "layer_num": int              # Layer index
        }
    }
}
```

**Validation**: Metal workers must save artifacts in this exact format to ensure analyzers work without modification.

## Environment Variables

The following environment variables control backend behavior:

- `VLLM_HOOK_BACKEND`: Set to "vllm" or "metal" (auto-detected)
- `VLLM_WORKER_MULTIPROC_METHOD`: Set to "spawn" on macOS (handled by vllm-metal)
- `VLLM_HOOK_DIR`: Hook artifact directory (unchanged)
- `VLLM_HOOK_FLAG`: Hook activation flag (unchanged)
- `VLLM_RUN_ID`: Run identifier (unchanged)
- `VLLM_HOOK_LAYER_HEADS`: Layer-head mapping (unchanged)
- `VLLM_HOOKQ_MODE`: Hook mode (unchanged)
- `VLLM_ACTSTEER_CONFIG`: Steering config path (unchanged)

## Compatibility Matrix

| Platform | Backend | Worker Base Class | Status |
|----------|---------|-------------------|--------|
| Linux + CUDA | vLLM | `vllm.v1.worker.gpu_worker.Worker` | ✅ Existing |
| macOS + Apple Silicon | vLLM-Metal | `vllm_metal.v1.worker.MetalWorker` | 🔄 To Implement |
| macOS + Apple Silicon (fallback) | vLLM | `vllm.v1.worker.gpu_worker.Worker` | ⚠️ May not work |

## Risk Mitigation

### Risk 1: Metal Tensor Incompatibility

**Mitigation**:

- Keep artifact saving in PyTorch format (line 133 in workers)
- Add explicit tensor conversion if needed:

  ```python
  if hasattr(tensor, 'to_torch'):  # MLX tensor
      tensor = tensor.to_torch()
  ```

### Risk 2: Forward Context Differences

**Mitigation**:

- Test `get_forward_context()` behavior with Metal backend
- Add Metal-specific context handling if needed
- Document any behavioral differences

### Risk 3: Hook Registration Failures

**Mitigation**:

- Add comprehensive error handling in `_install_hooks()`
- Log detailed error messages for debugging
- Provide fallback to non-hook mode if hooks fail

## Next Steps

1. **Implement Phase 1**: Backend negotiation in `hook_llm.py`
2. **Implement Phase 2**: Create Metal workers
3. **Implement Phase 3**: Update plugin registry
4. **Test on Apple Silicon**: Validate end-to-end functionality
5. **Document**: Update README with Metal support information

## References

- Original Architecture Doc: [`architecture_vllm_centered_backends_with_vllm_metal.md`](my_fork_of_vllm_hook/vLLM-Hook/docs/architecture_vllm_centered_backends_with_vllm_metal.md)
- GPU to Metal Notes: [`gpu_to_metal.txt`](my_fork_of_vllm_hook/vLLM-Hook/docs/gpu_to_metal.txt)
- vLLM-Metal Source: [`vllm-metal/`](my_fork_of_vllm_hook/vllm-metal)
