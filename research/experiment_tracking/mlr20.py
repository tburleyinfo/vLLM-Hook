"""MLR-20 orchestration helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from research.experiment_tracking.conditions import MLR20_COMPARISON_GROUP
from research.experiment_tracking.schemas import ExperimentResult
from research.experiment_tracking.wandb_adapter import WandbTracker


def log_mlr20_results_sequentially(
    results: Iterable[ExperimentResult],
    *,
    project: str = "vllm-hook-eprime",
    enabled: bool | None = None,
    wandb_module: Any | None = None,
) -> list[dict[str, Any]]:
    """Log each MLR-20 condition as its own W&B run, in input order."""

    payloads: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for result in results:
        key = (result.condition.condition_id, result.condition.replicate)
        if key in seen:
            raise ValueError(
                "Each Notion condition/replicate must map to one distinct W&B run; "
                f"duplicate result for {key[0]} replicate {key[1]}."
            )
        seen.add(key)

        tracker = WandbTracker(
            project=project,
            group=result.condition.comparison_group or MLR20_COMPARISON_GROUP,
            run_name=_mlr20_run_name(result),
            tags=result.condition.tags,
            enabled=enabled,
            wandb_module=wandb_module,
        )
        payloads.append(tracker.log_result(result))

    return payloads


def _mlr20_run_name(result: ExperimentResult) -> str:
    condition = result.condition
    alpha = "NA" if condition.alpha is None else f"{condition.alpha:.2f}"
    spotlight = "spotlight-on" if condition.spotlight else "spotlight-off"
    return f"{condition.condition_id}-r{condition.replicate}-{spotlight}-alpha-{alpha}"
