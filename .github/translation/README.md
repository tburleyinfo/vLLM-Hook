# Repository translation workflow

The shared action translates a repository between declared platforms. It generates
an original implementation patch, saves discovery artifacts and can open a draft
pull request. It contains no required example adapter, model, input prompts,
scoring rule, hand-drawn example graph or example-specific dependency map.

Ordinary examples already in the source repository remain available for code
inspection. The action does not select an example or tell the translator to build
an example-specific interface. Translation should preserve the repository's own APIs.

## Inputs and output

After the workflow is on the default branch, choose **Actions → Repository
Translation → Run workflow**:

| Input | Default | Meaning |
| --- | --- | --- |
| `stage` | `run` | Prepare, discover, freeze, generate and package in one job |
| `condition` | `no-graph` | `graph` adds extracted source/target graphs; `both` runs a pair |
| `manifest` | `.github/translation/manifest.json` | Repository revision, platforms, behavior requirements, target packages and translator settings |
| `validator_profile` | Empty | Optional validator invoked after generation; never a translation input |
| `create_draft_pr` | Checked | Publish the selected condition's original patch as a draft |
| `pr_base` | `main` | Draft destination |
| `run_dir` | Empty | Unique temporary directory for `run`; explicit persistent directory for separate stages |
| `pilot` | Checked | Required for the default Auto-routed model |

Use `condition: both` with `create_draft_pr` unchecked for paired research.
The default manifest describes vLLM CPU/CUDA → vLLM-Metal on Apple Silicon, but
contains no Attention Tracker contract or inference settings. Change platform
names, repository requirements and target packages to use another Python backend.
The action currently uses a native macOS/ARM64 runner for its default target;
change the runner labels for other target environments.

Without a validator, generation does not run inference or require a CPU baseline.
The draft records validation as **skipped**. A selected validator may pass, fail
or block. A failed/blocked report keeps the action red while still allowing a
labeled draft. Failed generation, an empty patch or a changed original candidate
cannot be published as a completed translation. No manual candidate repair occurs.

The draft contains the selected original implementation patch and links to the
Actions evidence artifact. Reports, model usage, translation prompts, discovery
artifacts, declared library sources, graphs and notebooks are uploaded separately.
The saved patch records all changed implementation files regardless of their names.
Evidence expires after 14 days. Workflow infrastructure and discovery scaffolding
cannot enter the generated candidate PR. Publication uses a separate Git index,
checks patch applicability against the selected base and reuses matching drafts.

## Graph and no-graph boundary

Both modes receive the same frozen repository, declared target-library Python
sources, optional general platform notes and repository behavior requirements.
Only graph mode receives machine-extracted source and target graphs. Graphs use
all Python files in those corpora, without example-specific queries or seed lists.
Both modes may derive their own graphs and plans under `translation_discovery/`.
Those generated outputs are hashed separately and excluded from the delivery patch.

The graph records definitions, lexical calls, class membership and source-linked
syntax facts. It does not prove runtime behavior or provide full data/control-flow
analysis. The design draws on [RepoGraph](https://arxiv.org/abs/2410.14684) and
[AlphaTrans](https://arxiv.org/abs/2410.24117); it is not a replication or a proven
Metal translation improvement. One paired pilot does not establish a graph advantage.

## Optional validation interface

Validation is separate from the translation manifest. The shared runner does not
load a profile or its settings until generation has finished. Selecting a validator
does not add test inputs, example interfaces, model choices or scoring rules to the
translation prompt or supplied graph context.

Keep validator profiles under `.github/translation-validators/`, or outside the
source checkout entirely. Other locations inside the repository and generated-run
folders are rejected, so a custom profile cannot leak through source inspection.
A profile JSON contains two paths relative to its own directory:

```json
{"script": "validate.py", "config": "settings.json"}
```

The shared runner copies that directory outside the source/candidate trees, records
its file hashes, then invokes it with the following arguments:

```text
python validate.py --source <frozen-source> --candidate <clean-delivered-candidate> --config <settings.json> --output <validation-result.json>
```

The validator must write JSON with `status: pass`, `fail` or `blocked`, with optional
error/details. A nonzero exit cannot be accepted as success. Missing/broken validators
are recorded as blocked. The candidate is materialized from the frozen source and
saved implementation patch; supplied graphs, reference libraries and generated plans
are absent at runtime. The original generated candidate remains unchanged.
Existing real validation results are preserved; use a new run to change a validator
or investigate repairs. A previously skipped run may select a validator afterward.

The optional [Attention Tracker validator](../translation-validators/attention_tracker/README.md)
is one consumer of this interface. Its model, prompts, head settings and score
tolerances live in that separate directory. It calls the repository's native MIA API
on source and target, rather than requiring a newly generated example adapter.
Passing this validator would support only its limited test scope.

Workflow tooling and optional validators are excluded from frozen source snapshots,
even when the source revision contains them. Excluded paths are
`.github/scripts/translation/`, `.github/translation/`, `.github/translation-validators/`,
`.github/workflows/repository-translation.yml` and `tests/workflows/translation/`.
Thus optional test code cannot leak in through repository inspection or source graphs.
Only the explicitly selected validator profile is copied for post-generation testing.

## Dependencies and tracking

Generation uses Python 3.12+, Git, GitHub Copilot CLI **1.0.97**, and declared target
library **source files**. The graph extractor supports Python source. The shared
runner, graph extraction, artifact packaging and publication use the standard
library. The only installed tool dependency is the pinned W&B SDK from
`requirements-tracking.txt`, in a temporary environment. Inference dependencies
belong to the optional validator's existing environments; the action does not
install or upgrade application/CPU/Metal packages.

Configure these repository variables:

| Variable | Purpose |
| --- | --- |
| `TRANSLATION_TARGET_SITE` | Source directory containing the declared target packages |
| `TRANSLATION_COPILOT_BIN` | Full path to Copilot CLI 1.0.97 |
| `TRANSLATION_WANDB_ENTITY` | W&B team/account; existing trials use `tm8ctgzqj8-georgia-institute-of-technology` |
| `TRANSLATION_WANDB_PROJECT` | `vllm-hook-translation-trials` |
| `TRANSLATION_COPILOT_HOME` | Optional existing Copilot login directory |

Set repository secrets `COPILOT_GITHUB_TOKEN` and `WANDB_API_KEY`. Every real
translating condition gets a dedicated W&B record before model use. A pair shares
a group. Record source/context/prompt hashes, platform requirements, translator,
timing and generation/validation outcomes. No example fixture settings are required
by tracking. Code, full prompts, raw logs and machine metadata are not uploaded to
W&B; complete evidence stays in Actions artifacts. Missing tracking configuration
stops model usage.

Enable **Settings → Actions → General → Allow GitHub Actions to create and approve
pull requests**. Only the Ubuntu publishing job gets write permissions. Generation
has read permission and model tools cannot publish.

## Local act execution

The local launcher runs this same generic workflow on the native host and disables
remote publication:

```bash
export TRANSLATION_TARGET_SITE=/path/to/target/site-packages
export TRANSLATION_COPILOT_BIN=/path/to/copilot
export TRANSLATION_COPILOT_HOME="$HOME/.copilot"
export TRANSLATION_WANDB_ENTITY=your-team
export TRANSLATION_WANDB_PROJECT=vllm-hook-translation-trials
# Provide WANDB_API_KEY through your existing environment; never commit it.
TRANSLATION_CONDITION=graph .github/scripts/translation/act.sh run
```

`TRANSLATION_MANIFEST` selects another generic manifest. `TRANSLATION_VALIDATOR_PROFILE`
selects an optional post-generation profile. No Attention Tracker script is selected
by the launcher. `act` itself is a local prerequisite; GitHub runners do not need it.

For separate stages use the same explicit persistent run directory and runner host:

```text
prepare → discover → retrieve → freeze → finish
```

`evaluate` requires an explicit validator profile. `finish` can reuse the unchanged
saved candidate and its original validation report. Generic runs use schema version 1
and the `no_graph`/`graph` directories. Historical experiment protocols and results
remain in their original experiment history; they are not loaded by this runner.

## Offline checks before merge

```bash
python3 -m unittest discover -s tests/workflows/translation -p 'test_*.py'
python3 tests/workflows/translation/smoke_delivery.py --output /tmp/translation-smoke-unique
```

Tests use a tiny repository and arbitrary target package unrelated to the example.
They check withheld validator content, identical generation inputs with and without
validation, graph-only differences, failed/blocked validators still producing drafts,
immutable original patches, isolated Git publication and W&B metadata boundaries.
PR CI uploads clearly labeled offline fixture bundles for both modes. These checks
call no model, run no inference and create no research experiments.

Real authenticated draft publication and model generation remain a manual action
test after runner setup and merge. GitHub requires a manually dispatched workflow
to exist on the default branch: [GitHub documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).
