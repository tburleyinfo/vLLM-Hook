# Optional Attention Tracker validation

This directory is a post-generation test consumer. It is not a translation profile,
implementation recipe or source of model instructions. The shared workflow never
selects it by default, and excludes it from source snapshots and supplied graphs.

To select it, set the action's `validator_profile` input to:

```text
.github/translation-validators/attention_tracker/profile.json
```

For local act use the same path in `TRANSLATION_VALIDATOR_PROFILE`.

- `profile.json` declares the validator script and its settings file.
- `settings.json` holds the test model revision, prompts, selected heads, sampling
  settings, target environment and provisional numeric tolerances.
- `fixture.py` executes generation, capture and analysis using the repository's
  native `MiaLLM` API, for the source or translated repository.
- `validate.py` runs both executions and compares their reported contracts/scores.

Configure `TRANSLATION_SOURCE_PYTHON` and `TRANSLATION_TARGET_PYTHON` as repository
variables (or local environment variables) pointing to existing source and target
inference environments. `TRANSLATION_MODEL_CACHE` selects cached model weights.
Inference is offline; the action installs no inference packages. Missing environments
produce a blocked report and do not prevent a labeled review draft.

The test expects the translated repository to retain its native API. It does not
require `attention_tracker_metal.py` or a `MetalAttentionTracker` adapter. A translation
that does not support this feature may fail; generation is still independently
reviewable. A pass would establish only this limited fixture's agreement, not full
repository equivalence or a graph advantage. Tolerances remain provisional.

Old handwritten example dependency maps, diagrams, experiment manifests and repair
helpers are absent from the shared action. Preserved historical trials remain in
the original experiment records.
