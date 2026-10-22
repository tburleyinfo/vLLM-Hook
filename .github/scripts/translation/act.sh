#!/usr/bin/env bash
# Same generic action on the local native host; publication is disabled locally.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$repo_root"
stage="${1:-run}"
run_dir="${2:-experiment-runs/translation/local-$(date +%Y%m%d-%H%M%S)}"
args=(workflow_dispatch -j experiment -W .github/workflows/repository-translation.yml -P ubuntu-latest=-self-hosted
  --input "stage=$stage" --input "run_dir=$run_dir"
  --input "manifest=${TRANSLATION_MANIFEST:-.github/translation/manifest.json}"
  --input "condition=${TRANSLATION_CONDITION:-no-graph}"
  --input "validator_profile=${TRANSLATION_VALIDATOR_PROFILE:-}"
  --input "pilot=${TRANSLATION_PILOT:-true}" --input create_draft_pr=false
  --var "TRANSLATION_LOCAL_REPO=$repo_root")
for name in TRANSLATION_TARGET_SITE TRANSLATION_COPILOT_BIN TRANSLATION_COPILOT_HOME TRANSLATION_SOURCE_PYTHON TRANSLATION_TARGET_PYTHON TRANSLATION_MODEL_CACHE TRANSLATION_WANDB_ENTITY TRANSLATION_WANDB_PROJECT; do
  args+=(--var "$name=${!name:-}")
done
# act reads the value from the existing environment; never print credential values.
if [[ -n "${WANDB_API_KEY:-}" ]]; then args+=(--secret WANDB_API_KEY); fi
if [[ -n "${COPILOT_GITHUB_TOKEN:-}" ]]; then args+=(--secret COPILOT_GITHUB_TOKEN); fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp/repository-translation-act}"
exec act "${args[@]}"
