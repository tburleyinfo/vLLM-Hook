# vLLM-Hook Architecture: vLLM-Centered Backends

## Goal

Keep `vllm` as the canonical runtime for model-internal instrumentation, while allowing platform-specific compute backends (Torch, MLX, others) for post-capture analysis.

This avoids splitting the project into parallel systems and preserves the core value of vLLM-Hook: introspection and control of internals from the vLLM execution path.

## Non-Goals

- Replacing `vllm` with MLX as the primary runtime for hook capture.
- Defining independent analyzer semantics per platform.
- Creating analyzer-only features that cannot run against vLLM-produced artifacts.

## Design Principles

1. Capture truth comes from `vllm` workers.
2. Artifacts are backend-neutral and versioned.
3. Analyzer semantics are identical across compute backends.
4. Notebook and API surface remain stable (`HookLLM`, `analyze`, plugin registry).
5. Unsupported capabilities are explicit, not silent fallbacks.

## System Overview

Pipeline:

1. `CaptureBackend` instruments vLLM internals at inference time.
2. `ArtifactStore` persists run-scoped hook artifacts in a stable schema.
3. `AnalyzerBackend` loads artifacts and computes metrics using a selected compute library.
4. `ResultEmitter` returns scores and provenance to callers.

```
Prompt -> HookLLM (vllm runtime)
       -> CaptureBackend(vllm worker hooks)
       -> ArtifactStore(run_id scoped qk/act artifacts)
       -> AnalyzerBackend(torch|mlx|numpy)
       -> scores + provenance
```

## Core Interfaces

### 1) CaptureBackend (online, vLLM-owned)

Responsibilities:

- Install hooks in worker internals.
- Capture tensors and metadata per run.
- Emit artifacts only through `ArtifactStore`.

Suggested interface:

```python
class CaptureBackend(Protocol):
    name: str
    capture_kind: str  # "qk", "activation", ...

    def supports(self, model_config: dict) -> bool: ...
    def begin_run(self, run_id: str, config: dict) -> None: ...
    def record(self, module_name: str, payload: dict) -> None: ...
    def end_run(self, run_id: str) -> None: ...
```

Initial implementation:

- `VllmQKCaptureBackend` implemented by `probe_hookqk_worker.py`.

### 2) ArtifactStore (contract boundary)

Responsibilities:

- Persist and load artifacts by `(run_id, artifact_type)`.
- Attach schema version and provenance.
- Merge partitioned artifacts (TP ranks) deterministically.

Suggested interface:

```python
class ArtifactStore(Protocol):
    def write(self, run_id: str, artifact_type: str, payload: dict) -> None: ...
    def read(self, run_id: str, artifact_type: str) -> dict: ...
    def latest_run_id(self) -> str: ...
    def list_runs(self, limit: int = 100) -> list[str]: ...
```

Default implementation:

- Filesystem store under `hook_dir`, with current `RUN_ID.txt` + `run_id/**/qk.pt`.

### 3) AnalyzerBackend (offline or inline compute)

Responsibilities:

- Compute metrics from `ArtifactStore` outputs only.
- Keep metric semantics identical regardless of tensor library.
- Return numeric results and diagnostic metadata.

Suggested interface:

```python
class AnalyzerBackend(Protocol):
    name: str  # "torch", "mlx", ...

    def supports(self, artifact_type: str) -> bool: ...
    def compute(self, artifact: dict, spec: dict) -> dict: ...
```

Implementations:

- `AttentionTrackerTorchBackend` (current behavior).
- `AttentionTrackerMLXBackend` (same formulas, different tensor ops).

### 4) Analyzer Facade (plugin-visible)

`AttntrackerAnalyzer` becomes a facade:

- Load artifacts via `ArtifactStore`.
- Dispatch to backend selected by environment/config.
- Normalize return payload shape.

This keeps `PluginRegistry` and notebook calls unchanged.

## Artifact Contract

Use explicit schema and provenance:

```python
{
  "schema_version": "1.0",
  "artifact_type": "qk_cache",
  "capture_backend": "vllm_probe_hookqk",
  "provenance": {
    "vllm_version": "...",
    "vllm_hook_version": "...",
    "model_name": "...",
    "model_revision": "...",
    "tokenizer_revision": "...",
    "hookq_mode": "last_token|all_tokens",
    "tensor_parallel_size": 1,
    "tp_rank": 0,
    "timestamp_utc": "..."
  },
  "config": {
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "attention_multiplier": 0.088...
  },
  "qk_cache": {
    "<module_name>": {
      "layer_num": 12,
      "q": [...],
      "k_all": [...]
    }
  }
}
```

Rules:

1. Schema version bump on any incompatible change.
2. Analyzer must reject unknown major versions.
3. Merge ordering must be deterministic (`tp_rank` sorted).
4. Missing provenance fields should emit warnings in results.

## Capability Model

Expose capability flags to callers:

```python
{
  "capture_supported": true,
  "analysis_supported": true,
  "capture_backend": "vllm_probe_hookqk",
  "analysis_backend": "mlx",
  "degraded_mode": false,
  "reason": null
}
```

Examples:

- Apple Silicon with current code: `capture_supported=false` for custom worker hooks.
- Offline analyzer on existing artifacts: `capture_supported=false`, `analysis_supported=true`, `degraded_mode=true`.

## Parity and Validation

For each analyzer metric (for example, attention tracker score):

1. Capture one canonical run artifact with vLLM.
2. Run Torch backend and MLX backend on identical artifact.
3. Compare with tolerance (`rtol=1e-4`, `atol=1e-5` or tuned by metric).
4. Fail CI if drift exceeds threshold.

Add tests:

- Artifact schema validation tests.
- Backend parity tests.
- End-to-end test from `HookLLM.generate(..., use_hook=True)` to `llm.analyze(...)`.

## Rollout Plan

Phase 1: Contract hardening

- Add `schema_version` and provenance fields.
- Wrap load/merge in `ArtifactStore`.

Phase 2: Analyzer backend abstraction

- Refactor `attention_tracker_analyzer.py` into facade + torch backend.
- Add MLX backend implementing identical formulas.

Phase 3: Notebook/API unification

- Add backend selector for analyzer compute only.
- Keep `HookLLM` invocation and plugin names stable.

Phase 4: Optional native MLX capture (future)

- Only if hook points can replicate vLLM semantics.
- Must emit the same artifact contract and pass parity checks.

## Practical Guidance for Current Repository

Short-term path that stays aligned with project goal:

1. Keep run capture in `probe_hookqk_worker.py` under `HookLLM` + `vllm`.
2. Refactor `AttntrackerAnalyzer` into:
   - artifact load logic (shared),
   - compute backend adapter (`torch` now, `mlx` optional).
3. Do not fork notebook logic by runtime; only vary analyzer backend.
4. Report capability state in notebook output instead of silently skipping scores.

This preserves one architecture and one truth source while enabling platform flexibility.
