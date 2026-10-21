# Attention Tracker translation workflow

Translate one Attention Tracker candidate to Metal, optionally validate it, and
open a draft PR containing only that candidate's changes.

## Setup

The translation job needs a self-hosted runner labeled `macOS` and `ARM64`,
GitHub Copilot CLI **1.0.97**, and existing CPU and Metal Python environments.
Model weights must already be cached; inference runs offline.

Set these repository variables:

| Variable | Value |
| --- | --- |
| `ATTN_CPU_PYTHON` | Full path to the CPU environment's Python |
| `ATTN_METAL_PYTHON` | Full path to the Metal environment's Python |
| `ATTN_METAL_SITE` | Metal environment's `site-packages` directory |
| `ATTN_MODEL_CACHE` | Hugging Face model cache directory |
| `ATTN_COPILOT_BIN` | Full path to Copilot CLI 1.0.97 |

Set the `COPILOT_GITHUB_TOKEN` repository secret for translation authentication.
Translation sends the prepared source and context to Copilot and consumes usage.
Enable **Settings → Actions → General → Allow GitHub Actions to create and
approve pull requests**. Organization policy can restrict this setting.

## Prepare a run

After merging the workflow, open **Actions → MLR15 MLR40 Attention Tracker
Experiment → Run workflow**.

Use `manifest: experiments/attention_tracker/manifest.retrieval.json` for the
current graph/no-graph comparison. Set `pilot: true`: this manifest uses Auto
model routing, so results are pilot evidence. The older `manifest.json` is retained
for the legacy source-graph protocol and its tests.

Choose a new `run_dir`. Run these stages in order, keeping the same directory and
using the same runner host:

```text
prepare → baseline → discover → retrieve → probe → live-probe → freeze
```

Each stage is a separate manual workflow run. The workflow preserves saved files
between stages. For multiple matching runners, use a persistent directory and
runner routing that keeps all stages on the same host.

## Finish and open a draft

Run the workflow again with:

| Input | Value |
| --- | --- |
| `stage` | `finish` |
| `run_dir` | The prepared run's directory |
| `condition` | `graph` or `no-graph` |
| `run_tests` | Checked to run Metal parity; unchecked to skip it |
| `create_draft_pr` | Checked |
| `pr_base` | `main` |
| `pilot` | `true` when using the supplied Auto manifests |

`finish` generates the selected translation if absent, or reuses its successful
saved patch. A failed translation must be investigated rather than overwritten.
Changed patches or candidate contents are rejected.

The draft contains only the selected adapter, optional helper, notebook and README
changes. It excludes the other candidate, logs and frozen context. Conflicts with
the destination branch stop publication. Existing open drafts are reused.

The draft records validation as **pass**, **fail**, **blocked**, or **skipped**.
Failed validation keeps the workflow red but still permits a draft for review.
Skipping tests does not establish parity. Notebook cells are not executed end to
end; a passing fixture establishes only agreement within its tested scope.

For paired research runs, select `condition: both` and uncheck `create_draft_pr`.
No graph advantage is established by a single paired pilot.

## Checks

Pull requests run the protocol tests on Ubuntu without model or GPU dependencies:

```bash
python3 -m unittest discover -s experiments/attention_tracker -p 'test_*.py'
```

Real CPU/Metal validation requires the environments and prepared evidence above.
The runner writes `results.md`, `results.json` and per-stage logs in `run_dir`.
