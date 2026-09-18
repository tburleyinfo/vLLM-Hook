"""Thin Weights & Biases adapter for structured experiment results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from research.experiment_tracking.schemas import ExperimentResult


class WandbTracker:
    """Log structured experiment results to W&B when enabled."""

    def __init__(
        self,
        project: str = "vllm-hook-eprime",
        group: str | None = None,
        run_name: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        enabled: bool | None = None,
        wandb_module: Any | None = None,
    ) -> None:
        self.project = project
        self.group = group
        self.run_name = run_name
        self.tags = list(tags or [])
        self.enabled = _tracking_enabled(enabled)
        self._wandb = wandb_module

    def log_result(self, result: ExperimentResult) -> dict[str, Any]:
        """Log one execution/replicate and return the payload that was sent."""

        payload = result.to_payload()
        payload["config"].update(dict(result.provenance))

        if not self.enabled:
            return payload

        wandb = self._wandb or _import_wandb()
        run = wandb.init(
            project=self.project,
            group=self.group,
            name=self.run_name or _default_run_name(result),
            tags=self.tags or list(result.condition.tags),
            config=payload["config"],
        )
        try:
            table = _build_table(wandb, payload["turns"])
            wandb.log({**payload["metrics"], "turn_results": table})
            _log_artifact(wandb, run, payload)
        finally:
            run.finish()

        return payload


def _tracking_enabled(enabled: bool | None) -> bool:
    if enabled is not None:
        return enabled
    return os.environ.get("WANDB_MODE", "").lower() not in {"disabled", "off"}


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking requested, but the wandb package is not installed. "
            "Install it in Colab with `%pip install wandb` or set "
            "`WANDB_MODE=disabled` for dry runs."
        ) from exc
    return wandb


def _default_run_name(result: ExperimentResult) -> str:
    config = result.condition
    alpha = "none" if config.alpha is None else f"{config.alpha:.2f}"
    return f"{config.condition_id}-r{config.replicate}-alpha-{alpha}"


def _build_table(wandb, rows: list[dict[str, Any]]):
    columns = [
        "condition",
        "turn",
        "prompt",
        "user_message",
        "model_response",
        "compliant",
        "violation_count",
        "state_of_being_count",
        "contraction_count",
    ]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*(row.get(column) for column in columns))
    return table


def _log_artifact(wandb, run, payload: dict[str, Any]) -> None:
    artifact = wandb.Artifact(
        name=f"{payload['config']['condition_id']}-r{payload['config']['replicate']}-result",
        type="experiment-result",
        metadata={
            "condition_id": payload["config"]["condition_id"],
            "replicate": payload["config"]["replicate"],
        },
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "experiment_result.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        artifact.add_file(str(path), name="experiment_result.json")
        run.log_artifact(artifact)

