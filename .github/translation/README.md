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

## Generate a review draft in one action run

After this workflow is merged into the default branch, open **Actions → Repository
Translation → Run workflow**. For the current artifact and draft review, use:

| Input | Value |
| --- | --- |
| `stage` | `run` (default; prepare through generation and packaging in one job) |
| `condition` | `no-graph` or `graph` |
| `run_tests` | Unchecked to generate artifacts without CPU/Metal validation |
| `create_draft_pr` | Checked |
| `pr_base` | `main` |
| `run_dir` | Leave blank for a unique directory per action run/attempt |
| `manifest` | `.github/translation/attention_tracker/manifest.discovery.json` |
| `pilot` | `true` for the default Auto-routed model |

The graph mode supplies source and target dependency graphs. No-graph gets the
same repository, raw library sources, task and settings without those graphs.
Both may derive their own graphs and plans. Real `vllm.LLM.generate()` through
vLLM-Metal is explicitly required; standalone MLX generation is not a substitute.

A generated candidate can open a draft before the port works. With tests disabled,
CPU and Metal inference are skipped and the draft says **skipped**. With tests enabled,
failed or blocked validation still permits a labeled draft and keeps the action red.
A failed translator or empty/changed patch cannot be published as a completed candidate;
its available evidence is still uploaded. Candidates are never manually repaired.

The draft changes only the selected candidate's implementation and notebooks.
Its body links to the Actions evidence download: original patch, prompts, reports,
generated discovery artifacts, supplied context and notebooks. The download expires
after 14 days. Generated plans and graph context remain artifacts, not runtime
files in the candidate PR. The original patch is the complete record of changed
files, including generated files with names other than the example adapter name.
Publication conflicts with `pr_base` stop the draft rather than overwrite work.
An existing draft for the same base and patch is reused.

`condition: both` is for paired research only; uncheck `create_draft_pr`.
One paired pilot does not establish a graph advantage. Notebook cells are not
executed end to end by this workflow.

## Runner and dependencies

Real generation runs on a self-hosted runner labeled `macOS`, `ARM64`, with
Python **3.12 or later**, Git and GitHub Copilot CLI **1.0.97** already available.
Generation-only drafts need target-library Python sources but do not import the
inference environments or load model weights. CPU and Metal environments plus
cached weights are required only when validation is enabled. Inference is offline.

The preparation, graph, publication and offline checks use Python's standard
library. The action installs only the W&B SDK from `requirements-tracking.txt`
in a separate temporary tools environment. It never installs or upgrades packages
inside the user's CPU or Metal environments. The Ubuntu publishing job uses Git
and the preinstalled GitHub CLI. No application runtime dependency is added.

Set these repository variables:

| Variable | Required for | Value |
| --- | --- | --- |
| `ATTN_METAL_SITE` | Generation | Directory containing declared `vllm_metal`, `mlx_lm`, `mlx` Python sources |
| `ATTN_COPILOT_BIN` | Generation | Full path to Copilot CLI 1.0.97 |
| `TRANSLATION_WANDB_ENTITY` | Generation | Your W&B account/team; current experiments use `tm8ctgzqj8-georgia-institute-of-technology` |
| `TRANSLATION_WANDB_PROJECT` | Generation | `vllm-hook-translation-trials` |
| `ATTN_CPU_PYTHON` | Validation | Full path to the CPU environment's Python |
| `ATTN_METAL_PYTHON` | Validation | Full path to the Metal environment's Python |
| `ATTN_MODEL_CACHE` | Validation | Hugging Face model cache directory |

Set the `COPILOT_GITHUB_TOKEN` and `WANDB_API_KEY` repository secrets.
Each real translating condition gets a dedicated W&B record **before** model
execution; a pair shares a group. Inputs are recorded as settings and hashes,
with model, timings and outcomes updated afterward. Code, translation prompts,
raw logs and machine metadata are not uploaded to W&B. Full evidence stays in
Actions artifacts. Missing tracking configuration stops generation before model
usage. Offline plumbing fixtures use neither service and are not research trials.

Enable **Settings → Actions → General → Allow GitHub Actions to create and approve
pull requests**. Only the Ubuntu publication job has `contents: write` and
`pull-requests: write`; generation has read permission. Model tools cannot publish.

## Review and test before merge

Pushing this PR runs the standard-library tests on GitHub-hosted Ubuntu and then
exercises both graph modes through preparation, frozen context, a synthetic
translator, clean patch materialization and draft bundle creation. The action
uploads `translation-delivery-smoke-<run>-<attempt>` containing both fixture bundles
and their reports. These are visibly labeled **offline plumbing fixtures**;
no model, inference hardware, W&B record or generated candidate PR is used.

```bash
python3 -m unittest discover -s tests/workflows/translation -p 'test_*.py'
python3 tests/workflows/translation/smoke_delivery.py --output /tmp/translation-smoke-unique
```

Publication tests exercise isolated Git index/commit construction and mock only
the network push and GitHub CLI calls. Real draft creation using `GITHUB_TOKEN`
and actual model generation remain a manual action test after merge and runner
setup. GitHub requires a manually dispatched workflow to exist on the default
branch: [GitHub documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

## Resume a prepared run

The separate stages remain available for investigation:

```text
prepare → baseline → discover → retrieve → freeze → finish
```

Use the same explicit persistent `run_dir` and runner host throughout. `finish`
generates only if no saved successful translation exists, then validates unless
`run_tests` is unchecked and prepares a single selected draft bundle. A changed
candidate or saved patch is rejected. The staged `freeze` keeps its original
passing-baseline requirement; the one-run generation-only path is pilot-only.

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
