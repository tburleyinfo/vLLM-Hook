#!/usr/bin/env bash
# Used identically by hosted setup CI and the actual translation job.
set -euo pipefail
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"
root="$RUNNER_TEMP/translation-hosted"
python3 -m venv "$root/tools"
"$root/tools/bin/python" -m pip install -r .github/translation/requirements-tracking.txt
"$root/tools/bin/python" .github/scripts/translation/bootstrap.py \
  --manifest "${TRANSLATION_MANIFEST:-.github/translation/manifest.json}" --output "$root/bootstrap"
echo "TRANSLATION_PYTHON=$root/tools/bin/python" >> "$GITHUB_ENV"
