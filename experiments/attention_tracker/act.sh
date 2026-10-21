#!/usr/bin/env bash
# Local native-host workflow runner; artifacts stay in the repository.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"
default_copilot="$repo_root/experiments/attention_tracker/.tools/copilot-1.0.97/node_modules/.bin/copilot"
if [[ ! -x "$default_copilot" ]]; then default_copilot=copilot; fi
stage="${1:-prepare}"
run_dir="${2:-experiment-runs/attention-tracker/local}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp/mlr15-40-act-cache}"
act workflow_dispatch \
  -W .github/workflows/attention-tracker-experiment.yml \
  -P self-hosted=-self-hosted \
  --input "stage=$stage" --input "run_dir=$run_dir" \
  --input "manifest=${ATTN_MANIFEST:-experiments/attention_tracker/manifest.json}" \
  --input "condition=${ATTN_CONDITION:-both}" \
  --input "run_tests=${ATTN_RUN_TESTS:-true}" \
  --input "create_draft_pr=false" \
  --input "pr_base=main" \
  --input "pilot=${ATTN_PILOT:-false}" \
  --input "baseline_run=${ATTN_BASELINE_RUN:-}" \
  --var "ATTN_CPU_PYTHON=${ATTN_CPU_PYTHON:-}" \
  --var "ATTN_METAL_PYTHON=${ATTN_METAL_PYTHON:-}" \
  --var "ATTN_METAL_SITE=${ATTN_METAL_SITE:-}" \
  --var "ATTN_MODEL_CACHE=${ATTN_MODEL_CACHE:-$HOME/.cache/huggingface/hub}" \
  --var "ATTN_COPILOT_BIN=${ATTN_COPILOT_BIN:-$default_copilot}" \
  --var "ATTN_COPILOT_HOME=${ATTN_COPILOT_HOME:-$HOME/.copilot}" \
  --var "ATTN_LOCAL_REPO=$repo_root"
