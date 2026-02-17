# vLLM-Hook Architecture: vLLM-Centered Backends with Official vLLM-Metal

## Goal

Keep one vLLM-Hook architecture that works across CUDA/Linux and Apple Silicon/macOS, with:

1. vLLM as the inference runtime boundary.
2. Stable hook artifacts and analyzer semantics.
3. First-class compatibility with official `vllm-metal` on Apple Silicon.

## Context

`vllm-metal` is now the official Apple Silicon path for vLLM. It integrates through platform and general plugin entry points and provides:

- `MetalPlatform` (`vllm_metal.platform.MetalPlatform`)
- `MetalWorker` (`vllm_metal.v1.worker.MetalWorker`)
- `MetalModelRunner` (`vllm_metal.v1.model_runner.MetalModelRunner`)

Relevant behavior for vLLM-Hook design:

1. Worker process defaults to `spawn` on macOS for fork safety.
2. Worker class may be auto-set to `vllm_metal.v1.worker.MetalWorker`.
3. Single-device, `uni` execution assumptions for Apple Silicon.
4. Some features are explicitly unsupported (for example MLA/sparse attention), so capability reporting must be explicit.

## Non-Goals

- Creating a separate non-vLLM architecture only for Apple Silicon.
- Diverging analyzer formulas between CUDA and Metal.
- Introducing silent behavior that hides unsupported hooks.

## Design Principles

1. One public API (`HookLLM`, `analyze`, plugin registry).
2. Capture contract is backend-neutral and versioned.
3. Apple-specific runtime constraints are handled in backend adapters, not notebooks.
4. Capability is explicit per run (`capture_supported`, `analysis_supported`, reason).
5. Prefer composition over replacement when integrating `vllm-metal`.

## Approach A (Preferred): Composite Worker for Metal + Hook Capture

### Idea

Provide a Metal-aware hook worker that composes official `MetalWorker` behavior and injects hook capture points without replacing the Metal stack.

### How

1. Add `MetalHookCaptureBackend` in vLLM-Hook that writes the same artifact schema used today.
2. Add worker integration that starts from `vllm_metal.v1.worker.MetalWorker` lifecycle, then attaches capture logic where Q/K tensors are available.
3. Keep analyzer pipeline unchanged (Torch/MLX backends read the same artifacts).

### Pros

- Preserves vLLM-Hook’s core value: capture from live vLLM execution.
- Uses official Apple runtime path, reducing drift from upstream vLLM changes.
- Keeps notebook/API behavior uniform across platforms.

### Risks

- Tight coupling to `vllm-metal` internal call paths where Q/K is available.
- Needs extra compatibility testing across `vllm` and `vllm-metal` versions.

## Recommended Direction

Adopt **Approach A** as the single architecture path:

1. Add Metal composite capture through official `vllm-metal` lifecycle integration.
2. Keep artifact contract identical across CUDA and Metal.
3. Keep analyzer behavior identical by reading the same artifacts through the same analyzer pipeline.

## Contract and Capability Additions

Use stable run metadata additions:

```python
{
  "schema_version": "1.1",
  "runtime": {
    "vllm_platform": "cuda|metal",
    "vllm_metal_enabled": true,
    "vllm_metal_use_mlx": true,
    "vllm_worker_multiproc_method": "spawn|fork"
  },
  "capability": {
    "capture_supported": true,
    "analysis_supported": true,
    "capture_backend": "vllm_probe_hookqk|metal_hook_capture",
    "analysis_backend": "torch|mlx"
  }
}
```

Rules:

1. Unknown major schema version must fail analyzer execution.
2. Capture support and backend choice must be visible to callers.
3. Analyzer formulas must remain identical across Torch and MLX.

## Rollout Plan

### Phase 1: Capability and Contract Hardening

1. Add runtime/capability metadata fields.
2. Centralize artifact read/write behind `ArtifactStore`.
3. Ensure Metal capability reporting is explicit and validated in tests.

### Phase 2: Metal Composite Capture Prototype

1. Implement `MetalHookCaptureBackend`.
2. Integrate with official `MetalWorker` lifecycle.
3. Validate emitted artifacts against current probe-worker schema.

### Phase 3: Parity and Reliability

1. Cross-platform parity tests for analyzer outputs (Torch vs MLX).
2. End-to-end tests on Apple Silicon:
   - `HookLLM.generate(..., use_hook=True)` capture enabled path.
3. Version matrix testing across supported `vllm` and `vllm-metal` combinations.

### Phase 4: Default Enablement

1. Enable Metal capture by default for known-good version combinations.
2. Keep strict capability checks for supported version combinations.
3. Document operational knobs (`VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_METAL_*`).

## Practical Guidance for This Repository

1. Keep `hook_llm.py` as the primary stable interface across platforms.
2. Treat `hook_llm_mlx.py` as a temporary compatibility shim, not the target architecture.
3. Move platform decisions into capability negotiation and backend selection layers.
4. Refactor analyzers to backend adapters, not platform forks.
5. Treat `vllm-metal` classes (`MetalPlatform`, `MetalWorker`, `MetalModelRunner`) as integration anchors rather than re-implementing their responsibilities.

## Explicit Translation to vLLM-Hook Code

This section maps the design directly onto the current "meat" of vLLM-Hook: `workers/` and `analyzers/`.

### Workers: Runtime Adapters + Capture/Steer Backends

Current files:

- `vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py`
- `vllm_hook_plugins/vllm_hook_plugins/workers/steer_activation_worker.py`

Required translation:

1. Keep worker classes as thin runtime adapters (vLLM lifecycle integration only).
2. Move hook logic into backend objects that workers call.
3. Keep artifact contract identical across CUDA and Metal capture paths.

Concretely:

1. `ProbeHookQKWorker`:
   - Today: does model integration + Q/K capture + file write.
   - Target: delegates capture to `VllmQKCaptureBackend` (CUDA path) or `MetalHookCaptureBackend` (Apple path).
2. `SteerHookActWorker`:
   - Today: does model integration + steering math.
   - Target: delegates steering operations to a `SteeringBackend` so worker remains a wrapper.
3. Add Metal runtime adapter:
   - Introduce a Metal hook adapter that composes official `vllm_metal` runtime (`MetalWorker`/`MetalModelRunner`) for capture, instead of relying on `hook_llm_mlx.py`.

### Where Approach A Is Analogous Today

The three Approach A statements map directly to existing vLLM-Hook behavior:

1. "Add `MetalHookCaptureBackend` ... writes same artifact schema":
   - Analog today: `ProbeHookQKWorker` builds and writes capture payload in `qkv_hook` (`vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py:76` and `vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py:133`).
   - Analog today: analyzers assume `{"config": ..., "qk_cache": ...}` contract loaded through merge utility (`vllm_hook_plugins/vllm_hook_plugins/run_utils.py:30`).
2. "Add worker integration ... from `MetalWorker` lifecycle":
   - Analog today: worker lifecycle integration happens in `load_model -> _install_hooks` on `ProbeHookQKWorker` (`vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py:30` and `vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py:41`), with module-level forward hooks registered at `vllm_hook_plugins/vllm_hook_plugins/workers/probe_hookqk_worker.py:144`.
   - Metal equivalent should preserve this lifecycle pattern, but rooted in official Metal worker lifecycle.
3. "Keep analyzer pipeline unchanged":
   - Analog today: `AttntrackerAnalyzer` loads merged artifacts and computes scores without any runtime-specific branch (`vllm_hook_plugins/vllm_hook_plugins/analyzers/attention_tracker_analyzer.py:33`, `vllm_hook_plugins/vllm_hook_plugins/analyzers/attention_tracker_analyzer.py:77`).
   - Analog today: merge/load boundary already centralizes artifact consumption (`vllm_hook_plugins/vllm_hook_plugins/run_utils.py:30`).

### Analyzers: Facade + Compute Backends

Current files:

- `vllm_hook_plugins/vllm_hook_plugins/analyzers/attention_tracker_analyzer.py`
- `vllm_hook_plugins/vllm_hook_plugins/analyzers/core_reranker_analyzer.py`

Required translation:

1. Keep analyzer class names and registry keys stable (`attn_tracker`, `core_reranker`).
2. Split each analyzer into:
   - facade (input validation, artifact load, result shape),
   - compute backend (`torch` and optional `mlx`) with identical formulas.
3. Ensure backend choice is capability/config driven, not notebook-specific branching.

Concretely:

1. `AttntrackerAnalyzer` becomes facade + `AttentionTrackerTorchBackend` + `AttentionTrackerMLXBackend`.
2. `CorerAnalyzer` becomes facade + `CoreRerankerTorchBackend` + `CoreRerankerMLXBackend`.
3. Formulas and output semantics must be parity-tested across compute backends.

### Artifact Store and Merge Boundary

Current file:

- `vllm_hook_plugins/vllm_hook_plugins/run_utils.py`

Required translation:

1. Evolve `run_utils.py` into a formal `ArtifactStore` boundary.
2. Preserve deterministic TP merge behavior.
3. Add schema/provenance/runtime/capability envelope required by this design.

Concretely:

1. `load_and_merge_qk_cache(...)` remains the single merge path.
2. Writes from all capture backends must include schema version + capability metadata.
3. Analyzers must read through the same store path regardless of runtime platform.

### Public API and Registry Stability

Current files:

- `vllm_hook_plugins/vllm_hook_plugins/hook_llm.py`
- `vllm_hook_plugins/vllm_hook_plugins/registry.py`
- `vllm_hook_plugins/vllm_hook_plugins/__init__.py`

Required translation:

1. Keep `HookLLM.generate(...)` and `HookLLM.analyze(...)` stable.
2. Replace worker-name string branching with capability negotiation and backend selection.
3. Keep plugin registration names stable to avoid notebook/API breakage.

Concretely:

1. `hook_llm.py` selects runtime adapter + capture/analyzer backends by detected capability.
2. `registry.py` and `__init__.py` continue exposing same user-facing worker/analyzer identifiers.
3. Capability and backend selection are explicit and testable, not implicit.
