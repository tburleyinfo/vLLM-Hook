# Repository translation workflow

This workflow builds a translation candidate, optionally validates it, and opens
a draft PR containing only the selected candidate's changes. The first supported
case is Attention Tracker on Metal; graph/no-graph selects how that translation
receives library context. This does not yet translate every repository feature.

## Published layout

- `.github/workflows/repository-translation.yml`: GitHub Actions entry point.
- `.github/scripts/translation/`: scripts used by the workflow.
- `.github/translation/attention_tracker/`: case configuration, validation fixture
  and source context.
- `tests/workflows/translation/`: tests of the published workflow scripts.

PR checks import the scripts from these published paths. They do not depend on
`experiments/`, local launchers, or saved research runs. Real Metal validation
executes the selected translated candidate before its patch is published.

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

After merging the workflow, open **Actions → Repository Translation → Run workflow**.

Use `manifest: .github/translation/attention_tracker/manifest.discovery.json` for the
current graph/no-graph comparison. Set `pilot: true`: this manifest uses Auto
model routing, so results are pilot evidence. The older manifests are retained for legacy protocol tests.

Choose a new `run_dir`. Run these stages in order, keeping the same directory and
using the same runner host:

```text
prepare → baseline → discover → retrieve → freeze
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

The draft contains only the selected candidate's implementation changes.
Generated discovery graphs and plans stay in run artifacts. It excludes the other candidate, logs and frozen context. Conflicts with
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
python3 -m unittest discover -s tests/workflows/translation -p 'test_*.py'
```

Real CPU/Metal validation requires the environments and prepared evidence above.
The runner writes `results.md`, `results.json` and per-stage logs in `run_dir`.


## Shared inputs for this comparison

Both translators receive the source repository, raw Python source from the same
declared target libraries, the same required adapter interface and settings, and
an example-independent hook correspondence. The correspondence maps forward hooks
to callable-module replacement through an owner attribute, with delegation and
restoration requirements. It supplies no application-specific implementation.

No completed wrapper, runtime probe, precision helper, request-ID helper,
metadata helper, live observation result or hand-picked hook-file list is given
to either translator. Independent validation stays outside both candidate trees.
The graph condition additionally receives unseeded source-repository and target-library
graphs. Both conditions may generate example-specific intermediate graphs and plans
under `translation_discovery/`; those are generated outputs, not front-end recipes.
The supplied graphs are immutable and retain source locations.
These syntax facts do not establish that a wrapper preserves runtime behavior.
The previous assisted pilot is not comparable to this protocol.


## Graph design and input boundary

The shared task declares platforms and the validation interface, not implementation
steps. The default manifest rejects an `implementation_steps` field. That structural
check cannot judge whether arbitrary natural-language text is an example recipe;
review custom inputs against this policy. Validation interfaces describe required
behavior and data boundaries, not how to implement them. The independent evaluator
remains outside the candidate tree.

Graph construction reads only the pinned source snapshot and declared target
library source. It selects no example-specific seeds and reads no prior candidate,
evaluation failure or completed translation. The graph condition gets both graphs;
the no-graph condition gets the same underlying source without the supplied graphs.
Both have the same tools and may derive their own intermediate representations.
Generated analysis is hashed in `discovery-artifacts.json` and excluded from the
saved publication patch. It is not frozen as supplied graph input.

This design adapts repository dependency navigation from
[RepoGraph](https://arxiv.org/abs/2410.14684) and source call-graph context from
[AlphaTrans](https://arxiv.org/abs/2410.24117). These papers support the design
choice; their results do not establish a benefit for Apple Silicon translation.
This implementation is not a replication of either system. It does not implement
GraphCoder's control/data-dependence graph or full interprocedural dataflow.
Graph/no-graph runs remain pilot evidence until repeated under controlled settings.

Validation for the current protocol materializes the pinned source plus the saved
implementation patch in a separate run directory. Supplied reference/graph files
and generated discovery artifacts are absent. A candidate that depends on those
files at runtime must fail validation instead of publishing a broken patch.
The publication step still checks that the patch applies to the requested base;
changes to that base since the pinned source can require revalidation.
