# Research Experiment Tracking

MLR-21 adds a small research-only tracking layer for Colab experiments. It is
intended for reproducibility on research branches, not for an upstream-facing
vLLM-Hook contribution.

## Architecture

Experiments should produce structured data, then pass that data to a tracker:

```text
experiment -> ExperimentResult -> WandbTracker -> Weights & Biases
```

The experiment and Spotlight implementation do not need to import `wandb`.
`WandbTracker(enabled=False)` returns the exact payload it would log, which is
useful for tests, dry runs, and notebooks without W&B credentials.

## Colab Authentication

Do not commit API keys. In Colab, authenticate with either of these approaches:

```python
import os
from google.colab import userdata

os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

or:

```python
import wandb

wandb.login()
```

To run without W&B, set:

```python
import os

os.environ["WANDB_MODE"] = "disabled"
```

## Minimal Usage

```python
from research.experiment_tracking import (
    ExperimentResult,
    TurnResult,
    WandbTracker,
    collect_runtime_provenance,
)
from research.experiment_tracking.conditions import alpha_sweep_conditions

condition = alpha_sweep_conditions(
    model="Qwen/Qwen2-1.5B-Instruct",
    turns=10,
    temperature=0.0,
    max_tokens=256,
    history_window_messages=16,
    replicate=1,
    notebook="notebooks/demo_spotlight_e_prime_colab.ipynb",
)[2]

turns = [
    TurnResult(
        condition=condition.condition_id,
        turn=1,
        prompt=user_message,
        response=model_response,
        compliant=violation_count == 0,
        violation_count=violation_count,
        state_of_being_count=state_of_being_count,
        contraction_count=contraction_count,
    )
]

result = ExperimentResult(
    condition=condition,
    turns=turns,
    provenance=collect_runtime_provenance(),
    full_result={"transcript": full_transcript},
)

tracker = WandbTracker(
    project="vllm-hook-eprime",
    group=condition.comparison_group,
    run_name="A2-r1-alpha-0.10",
    tags=["MLR-20", "alpha-sweep"],
)
tracker.log_result(result)
```

## W&B Behavior

The tracker logs one execution/replicate as one W&B run.

Supported config/provenance fields include:

- `condition_id`
- `comparison_group`
- `spotlight`
- `alpha`
- `implementation`
- `model`
- `constraint_formulation`
- `constraint_complexity`
- `intervention_timing`
- `history`
- `turns`
- `replicate`
- `temperature`
- `max_tokens`
- `history_window_messages`
- `seed`
- `git_sha`
- `notebook`
- `torch_version`
- `vllm_version`
- `spotlight_version`

Unknown provenance is recorded as `"unknown"` by
`collect_runtime_provenance()` rather than guessed.

Run metrics:

- `compliance_rate`
- `mean_violations`
- `total_violations`
- `mean_state_of_being_count`
- `total_contractions`
- `first_violation_turn`

Turn results are logged as a W&B Table with:

- `condition`
- `turn`
- `prompt`
- `user_message`
- `model_response`
- `compliant`
- `violation_count`
- `state_of_being_count`
- `contraction_count`

The complete structured payload, including full transcripts when provided, is
also stored as a W&B artifact named by condition and replicate.

## MLR-20 Integration

MLR-20 should consume this branch by merging or cherry-picking only the
`research/experiment_tracking` package and this document into a temporary
research-integration branch with the frozen MLR-19 experiment checkpoint.

Do not merge the MLR-19 notebook history into MLR-21. The E-Prime checker,
prompt list, generation loop, and saved notebook outputs remain
experiment-specific logic. The reusable MLR-21 surface is the structured result
schema, metric aggregation, explicit condition list, and W&B adapter.

