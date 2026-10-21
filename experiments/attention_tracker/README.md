# MLR-15 / MLR-40 Attention Tracker experiment

This is a separate repository translation workflow. MLR-2's Colab-only workflow
is unchanged. MLR-15 defines the executable port, MLR-40 defines prompt conditions,
MLR-50/53 provide the CPU environment/fixture, and MLR-7 defines behavioral review.

Primary guide: [MLR-15](https://app.notion.com/p/3de5023b4c788136a26bf06a475c042c).
Experiment method: [MLR-40](https://app.notion.com/p/3ea5023b4c7881618f7ec7625b52cd2e).
Acceptance review: [MLR-7](https://app.notion.com/p/3de5023b4c7881d6bd0ecef6ae18d82f).

## First comparison

1. `source_only`: frozen MIA source and the shared acceptance instructions.
2. `source_validated_dag`: exactly the same inputs, plus the source Mermaid DAG.

`manifest.json` freezes the source commit, translator model/effort, inference model
and revision, prompts, head selection, dtype, sampling and capture settings. Both
conditions start from identical disposable repositories. Source snapshots exclude
all uncommitted changes in the main worktree. Each run records source hashes,
target backend hashes, exact prompts, graph hash, CLI version, patches, logs and
independent fixture results. The external evaluator is copied before generation.

The current source DAG is a **draft awaiting human validation**. Pilot mode may
exercise it, but cannot establish the human-validated-DAG claim. Before a controlled
run, set `graph_review.status` to `validated`, name the reviewer, and set
`graph_review.graph_sha256` to the reviewed graph's SHA-256. Review must precede
`prepare`. Do not relabel a completed draft-graph pilot after the fact.

Target-context graphs, deterministic `codeanalyzer-python` extraction, and
LLM-generated graphs are later, separate conditions. They are not mixed into this
first comparison. Numeric tolerances are explicitly **provisional**; MLR-7 does
not currently provide values. Head 0/layer 0 is a diagnostic parity configuration,
not evidence of injection detector quality.

## Run locally through act

Use a native Apple Silicon host, with `act`, Python 3.12+, Git, the authenticated
GitHub Copilot CLI 1.0.97, and the existing CPU and Metal environments. No dependencies are
installed automatically. Model weights must already be cached. The shared model
is the cached Qwen2.5-3B model, with a small explicit 64 MiB cache; CPU and Metal
capture must both be verified independently. MLR-48's server success is not a
capture/parity baseline. The old OPT fixture remains a separate source smoke test.

The installed experiment CLI can be reproduced without a global installation:

```bash
npm install --prefix /private/tmp/mlr15-40-copilot-cli --save-exact @github/copilot@1.0.97
```

```bash
export ATTN_CPU_PYTHON=$HOME/path/to/cpu-env/bin/python
export ATTN_METAL_PYTHON=$HOME/.venv-vllm-metal/bin/python
export ATTN_METAL_SITE=$HOME/.venv-vllm-metal/lib/python3.12/site-packages
export ATTN_COPILOT_BIN=/private/tmp/mlr15-40-copilot-cli/node_modules/.bin/copilot

bash experiments/attention_tracker/act.sh prepare experiment-runs/attention-tracker/my-run
bash experiments/attention_tracker/act.sh baseline experiment-runs/attention-tracker/my-run
bash experiments/attention_tracker/act.sh metal-preflight experiment-runs/attention-tracker/my-run
bash experiments/attention_tracker/act.sh translate experiment-runs/attention-tracker/my-run
bash experiments/attention_tracker/act.sh evaluate experiment-runs/attention-tracker/my-run
bash experiments/attention_tracker/act.sh summary experiment-runs/attention-tracker/my-run
```

For a draft-graph pilot, set `ATTN_PILOT=true` for the translate command. The runner
labels it as pilot evidence and allows generation while prerequisites are still
being established. Controlled generation requires the shared-model CPU fixture,
target evidence, and the graph review gate. `prepare` refuses to overwrite runs.
Translation uses the account's existing GitHub Copilot login and consumes Copilot
usage. On a runner, provide `COPILOT_GITHUB_TOKEN` as an Actions secret. Local
translation uses the existing Copilot state directory without copying credentials.
CLI auto-updates, custom instructions, MCP tools and shell execution are disabled
for translation. Copilot can read and edit the isolated candidate; the independent
fixtures execute it later. The current account exposes only Auto routing. Both
conditions therefore use Auto with provider-default reasoning, explicitly as a pilot.
The runner preserves resolved model IDs and rejects Auto in controlled mode.
A controlled comparison needs an account that permits a fixed coding model.

To reuse a prior passing CPU run, set `ATTN_BASELINE_RUN` for the baseline stage.
The runner requires identical source archive/revision, evaluator hash and fixture
settings, copies the raw evidence, and records its origin and result hash.

`act` runs the workflow on the native host via `self-hosted=-self-hosted`. It skips
remote checkout and artifact upload; run files remain in `experiment-runs/`.
Workflow completion means the requested stage ran, **not** that parity passed.
Inspect `results.json` / `results.md`: failure and blocked states are preserved.
On GitHub, use a provisioned native ARM64 macOS self-hosted runner with the three
environment paths set as repository variables and the same pinned dependencies.
The workflow checks out the harness and uploads evidence; it never pushes code.

## Executable acceptance boundary

To monitor an existing run from a second terminal (Ctrl+C stops only the monitor):

```bash
python3 experiments/attention_tracker/status.py experiment-runs/attention-tracker/my-run --watch
```

New stage invocations also print a progress message every 20 seconds. A log
update shows activity, not successful capture or parity.

The CPU fixture performs real `MiaLLM.generate()` with `capture_qk` and then calls
`attn_tracker`. The Metal candidate must expose `MetalAttentionTracker` as specified
in `fixture.py`, perform real `vllm.LLM.generate()`, capture compatible Q/K probes,
and run the same scoring contract. The independent fixture checks request count,
token IDs, tokenizer identity, ranges, layer/head dimensions, dtypes and finite
values. CPU/Metal comparison rejects mismatched settings or capture contracts
even when scores happen to be close. No mock fallback is allowed.

The initial fixture covers one-token generation, selected layer/head, two prompts,
no prefix caching, and no concurrency. Decode histories, prefix-cache reuse,
concurrent request ordering and further heads/models need subsequent fixtures.
Observed parity never establishes full code equivalence. Multiple repeated trials
and human correction counts are needed before claiming a graph improves translation.

```bash
python3 -m unittest discover -s experiments/attention_tracker -p 'test_*.py' -v
```

## Platform discovery protocol (version 2)

New runs separate source compatibility from target capability. A V2 runner rejection
means the unchanged source API cannot run on this backend. It does not establish that
Metal cannot observe attention modules.

The new sequence is:

```bash
export ATTN_RUN=experiment-runs/attention-tracker/my-platform-pilot
export ATTN_PILOT=true
# Keep the CPU, Metal, target-site, model-cache and Copilot settings above.
bash experiments/attention_tracker/act.sh prepare "$ATTN_RUN"
bash experiments/attention_tracker/act.sh baseline "$ATTN_RUN"
bash experiments/attention_tracker/act.sh discover "$ATTN_RUN"
bash experiments/attention_tracker/act.sh probe "$ATTN_RUN"
bash experiments/attention_tracker/act.sh live-probe "$ATTN_RUN"
bash experiments/attention_tracker/act.sh freeze "$ATTN_RUN"
bash experiments/attention_tracker/act.sh translate "$ATTN_RUN"
bash experiments/attention_tracker/act.sh evaluate "$ATTN_RUN"
python3 experiments/attention_tracker/status.py "$ATTN_RUN" --watch
```

`discover` copies a fixed, bounded selection of installed vLLM-Metal, MLX and
MLX-LM source. It records hashes, package metadata, missing files and source-line
leads for worker access, module observation, post-position Q/K, request delivery
and tensor conversion. This is deterministic source search, not an LLM capability
judgment. The completed vLLM-Hook Apple Silicon port is excluded.

`probe` uses the Metal interpreter to test wrapping/restoring a toy MLX module,
unchanged output, before/after callbacks and MLX-to-Torch conversion. These are
small executable witnesses. They do not verify access to the live vLLM worker,
request association, paged attention capture or numerical parity. Those remain
translation and evaluation obligations. Failed witnesses block freezing.

`freeze` requires a passing CPU baseline and witnesses. It copies one identical
platform reference directory into both candidates and records its file hashes.
Translation and evaluation reject missing or changed context. Discovery cannot
be replaced in an existing run. Use a new run ID for another target installation.

For compatibility, directory names remain `source_only` and
`source_validated_dag`. In version 2 their meanings are **source + shared platform
evidence** and **source + shared platform evidence + source DAG**. This is a new
comparison; do not pool it with the original source-only pilot. Auto routing and
the unreviewed graph still make this pilot evidence. A target-context DAG remains
a separate future condition.

`source-compatibility` is the clearer alias for the optional `metal-preflight`
stage. Its result is diagnostic and does not gate discovery or translation.
Evidence remains local under the run directory when using act. Real GitHub runs
upload the discovery and shared evidence with the other artifacts.

## Live worker source mapping (version 3)

New runs now require `live-probe` **after `probe` and before `freeze`**:

```bash
bash experiments/attention_tracker/act.sh live-probe "$ATTN_RUN"
bash experiments/attention_tracker/act.sh freeze "$ATTN_RUN"
```

This stage loads the pinned model inside a real Metal worker, locates the selected
attention layer using the backend's layer lookup, and records the actual runner,
model, layer and attention classes. It follows `_inner` wrappers and records their
source paths, hashes and class line numbers. It copies only those identified files
from installed `vllm_metal`, `mlx_lm` or `mlx` packages into `live-platform/source`.
No completed upstream Attention Tracker code enters the context.

The supplied generic worker extension wraps the attention call, records input
shapes and keyword names, returns the original output and restores the original
module. Wrapped and restored runs must generate identical deterministic output
tokens. This demonstrates live worker access, generic observation delivery and
restoration. It does not establish reconstructed Q/K correctness, request-level
capture ordering, numerical equality of every internal tensor or score parity.

The manifest explicitly sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` and
`VLLM_METAL_USE_PAGED_ATTENTION=0`, following the reference branch's configuration.
The non-paged path also requires `VLLM_METAL_MEMORY_FRACTION=auto`.
`VLLM_PLUGINS=metal` prevents the checkout’s old hook plugin from being loaded
automatically. These settings apply to the live probe and candidate evaluation. This experiment
is about that non-paged path; it does not establish paged-attention support.

Both conditions receive the same live source map and generic extension. Instructions
now explain that module traversal locates attention but does not expose the internal
Q/K values. The translator must reconstruct the required values using the actual
attention input, projections, positional transforms and cache semantics. The completed
upstream reconstruction stays held out. Failed live observation blocks evidence
freezing. Existing version 1 and 2 runs retain their original protocol.

The installed Torch 2.11 allocator can raise an MPS assertion during vLLM's generic
cleanup after Metal worker shutdown. The live probe records this exact exception
and attempts MPS-specific cache cleanup plus MLX cleanup. Other cleanup exceptions
still fail the probe. Inspect `cleanup_warning` and `cleanup_fallback` in the result;
this local workaround does not establish that the backend defect is fixed.

## Shared repair infrastructure (version 4)

The fixes demonstrated in manual repair run `live-worker-v3-04-debug-05` now live
in ordinary project files:

- `live_precision.py`: inspect the actual worker model and align floating MLX
  parameters to float32 before generation. Records before/after dtype counts.
- `request_identity.py`: record explicit external/internal request fields while
  output requests are registered; restore the callback in finally. Missing or
  ambiguous mappings fail explicitly. No suffix parsing.
- `capture_metadata.py`: provide the MIA capture config schema. Head selection
  stays in tracker settings; hook mode stays in per-layer records. Nonstandard
  attention scaling is refused rather than silently changed.

Discovery includes the helpers in both frozen contexts. The live probe executes
precision alignment and verifies request identity mapping on real generation.
Version 4 freezing requires those checks to pass. The status watcher displays
actual precision and identity verification beneath Live worker observation.
The complete Q/K reconstruction remains the translator's task.

The stage order remains prepare, baseline, discover, probe, live-probe, freeze,
translate, evaluate. Start a new run; prior frozen runs retain their original
protocol. These helpers change the supplied infrastructure, so a fresh translation
pair is necessary to measure performance under version 4. Manual repair success
is not autonomous Copilot success or a graph-effect claim.

## Library-retrieval comparison (protocol 5)

The separate `manifest.retrieval.json` pins merged fork revision `8c0f2f4d` and
compares ordinary repository search with that same search plus deterministic
**target-library** relationships. It does not append the earlier human source DAG.
Older manifests and saved runs retain their original conditions.

`repository_graph.py` parses the bounded target corpus with Python's AST and records
symbols, explicit imports, lexical calls, and class membership with source locations
and hashes. It is not a full CPG: dynamic dispatch, aliases through objects,
re-exports, and dataflow are not inferred. Unresolved calls remain unresolved.
A retrieved edge is not evidence that a hook or capture works.

The frozen query `find layers` seeds a bounded two-hop search. A real-library check
retrieves discovery helpers, their callers, and attention-wrapper constructors.
The corpus includes the target library's own attention-patching code and MLX
module-replacement API. The completed vLLM-Hook Apple Silicon capture port remains
held out. Both conditions receive identical target source, live platform evidence,
precision/request/metadata helpers, source task, model settings, tools and timeout.
Only the graph condition gets `target_relationships/`, containing the retrieved
relationships and source locations. The full graph stays in the run evidence.
The shared worker observation helper is still supplied infrastructure in both
conditions: this measures incremental graph value, not discovery from no knowledge.

```bash
export ATTN_MANIFEST=experiments/attention_tracker/manifest.retrieval.json
export ATTN_RUN=experiment-runs/attention-tracker/2026-10-21-library-retrieval-v5-01
# This run was prepared already; do not prepare it again.
# For a NEW run, use a new ATTN_RUN and run these stages:
bash experiments/attention_tracker/act.sh prepare "$ATTN_RUN"
bash experiments/attention_tracker/act.sh baseline "$ATTN_RUN"
bash experiments/attention_tracker/act.sh discover "$ATTN_RUN"
bash experiments/attention_tracker/act.sh retrieve "$ATTN_RUN"
bash experiments/attention_tracker/act.sh probe "$ATTN_RUN"
bash experiments/attention_tracker/act.sh live-probe "$ATTN_RUN"
bash experiments/attention_tracker/act.sh freeze "$ATTN_RUN"
```

Set the CPU/Metal/site variables described above before running stages. Check the
saved results and context hashes before translating. Translation consumes Copilot
usage. The default manifest still uses Auto, so runs must be labeled pilot;
protocol 5 does not require review of the unused human source DAG. A controlled
run needs a separately pinned fixed model. Fixed timeout/tools do not impose an
equal token or tool-call budget; this is not a token-matched efficiency study.

No new translation outcome or graph-effect claim follows from preparation or
retrieval alone. Independent Metal capture/analyzer checks remain the acceptance
step. This first experiment covers Attention Tracker, not all platform scenarios.

The old MLR-50 CPU environment references a deleted temporary vLLM checkout.
Use the persistent MLR55 CPU environment shown above; do not rebuild dependencies
merely to repair that stale path. The first failed attempt is retained in the
new run’s attempt-history directory.

## Pinned local Copilot executable

The pilot uses CLI 1.0.97. A global install can resolve to another version, which
the version gate correctly rejects before sending a translation request.
The persistent copy used for the protocol-5 pilot is installed with:

```bash
npm install --prefix experiments/attention_tracker/.tools/copilot-1.0.97 --save-exact @github/copilot@1.0.97
```

The local tools folder is ignored by Git. The runner, monitor, and act launcher
prefer that pinned executable automatically; ATTN_COPILOT_BIN still overrides it.
They use the existing signed-in Copilot state. No credential copying is required.
The package install and version gate do not themselves consume a translation request.


## Finish one candidate and open a draft PR

In GitHub Actions, choose **MLR15 MLR40 Attention Tracker Experiment → Run workflow**:

1. Set `stage` to `finish` and `run_dir` to the existing prepared, frozen run.
2. Choose `condition`: `no-graph` or `graph`.
3. Leave `run_tests` checked for Metal parity tests. Uncheck it to skip those tests.
4. Leave `create_draft_pr` checked. Set `pr_base` to `main`.

`finish` reuses an existing successful translation, or generates it if absent.
It requires the earlier preparation and frozen evidence. It does not provision a
Metal environment. Tests run on the configured native Apple Silicon runner.

The draft contains only the selected candidate's saved patch: adapter, optional
helper, notebook, and README. It excludes logs, frozen evidence, and the other
candidate. Changed saved patches or candidate contents are rejected. Conflicts
with the destination branch stop publication instead of merging unrelated work.

The draft states **pass**, **fail**, **blocked**, or **skipped** for validation.
Failed validation leaves the workflow red but still permits a draft for review.
Skipped tests do not establish parity. Notebook cells are not tested end to end.
Choose `both` only for research runs with `create_draft_pr` unchecked.

Enable **Settings → Actions → General → Allow GitHub Actions to create and
approve pull requests** in the repository. Organization policy can restrict this.
[GitHub settings documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository).
The workflow and its scripts must be published to GitHub before remote use.

Local use (never opens a PR through `act`):

```bash
ATTN_CONDITION=no-graph ATTN_RUN_TESTS=false \
  bash experiments/attention_tracker/act.sh finish experiment-runs/attention-tracker/YOUR-RUN
```

Remove `ATTN_RUN_TESTS=false` to run Metal tests. Existing interpreter settings
still apply. The normal command-line default remains `--condition both`.
