# Notebook Translation Task

Translate or refresh the requested vLLM-Hook notebook while preserving the
source notebook's intent, execution order, and user-facing explanation.

Use the repository context before editing:

- Read the source notebook and any existing target notebook.
- Read package modules, tests, configs, and docs referenced by the target
  notebook before deciding the translation is complete.
- Read the relevant setup documentation in `notebooks/README.md`,
  `notebooks/metal/README.md`, and `tests/parity_tests/README.md`.
- Prefer the existing notebook, example, worker, analyzer, and config patterns.
- Keep edits scoped to the translated notebook and directly required supporting
  package code, tests, documentation, or config references.

Translation expectations:

- Preserve the source notebook's logical sections and demo flow.
- Adapt imports, backend classes, worker names, analyzer names, model defaults,
  config paths, and runtime assumptions to the target platform.
- If the target notebook needs a worker, analyzer, package export, model config,
  parity-test hook, or documentation entry that is missing on the selected
  branch, create or port the smallest required dependency from the closest
  existing implementation in the repository history or available branches.
- Keep notebook cells deterministic where the repository already does so.
- Do not commit, push, open a pull request, or expose secrets.
- If parity cannot be made runnable in the workflow environment, leave a clear
  note in your final response explaining the missing runtime prerequisite.

Before finishing:

- Inspect the resulting diff.
- Make sure the source notebook was not accidentally rewritten unless that was
  explicitly required.
- Prefer small, reviewable changes over broad cleanup.
