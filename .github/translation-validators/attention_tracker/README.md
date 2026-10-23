# Optional Attention Tracker validation

This directory is a post-generation test consumer. It is not a translation profile,
implementation recipe or source of model instructions. The action selects it for post-generation runtime validation by default and
excludes it from translation source snapshots and supplied graphs.

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

On GitHub, `hosted.py` provisions a Linux CPU environment from the source's CPU
profile and a separate Apple Silicon Metal environment. Both download the exact
same pinned model before offline fixture execution. `hosted_runtime.py` compares
the two reports after both jobs finish. Missing environments/devices and failing
comparisons prevent draft publication. Turn `runtime_validation` off explicitly
to publish complete deliverables without this check.

The default fixture now uses Qwen2.5-0.5B-Instruct at revision
`7ae557604adf67be50417f59c2c2f167def9a775` to fit a small hosted Mac. The previous
3B settings are preserved in `settings-3b.json`; changing the validation model
means these results are not directly interchangeable with prior 3B trials.
Tolerances and prompts are unchanged. Standard hosted Mac GPU availability is
probed rather than assumed. No paid runner is selected by default.

For local validation, configure `TRANSLATION_SOURCE_PYTHON` and
`TRANSLATION_TARGET_PYTHON` to point to existing inference environments and pre-cache
the model. This local path does not install inference packages.

The test expects the translated repository to retain its native API. It does not
require `attention_tracker_metal.py` or a `MetalAttentionTracker` adapter. A translation
that does not support this feature may fail; generation is still independently
reviewable. A pass would establish only this limited fixture's agreement, not full
repository equivalence or a graph advantage. Tolerances remain provisional.

Old handwritten example dependency maps, diagrams, experiment manifests and repair
helpers are absent from the shared action. Preserved historical trials remain in
the original experiment records.
