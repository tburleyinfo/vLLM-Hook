"""Structured experiment records shared by notebooks and tracking adapters."""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExperimentCondition:
    """One scientific condition from a design matrix."""

    condition_id: str
    comparison_group: str
    spotlight: bool
    alpha: float | None
    implementation: str
    model: str
    constraint_formulation: str
    constraint_complexity: str
    intervention_timing: str
    history: str
    turns: int
    replicate: int
    temperature: float
    max_tokens: int
    history_window_messages: int | None
    seed: int | None = None
    notebook: str | None = None
    tags: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_config(self) -> dict[str, Any]:
        config = {
            "condition_id": self.condition_id,
            "comparison_group": self.comparison_group,
            "spotlight": self.spotlight,
            "alpha": self.alpha,
            "implementation": self.implementation,
            "model": self.model,
            "constraint_formulation": self.constraint_formulation,
            "constraint_complexity": self.constraint_complexity,
            "intervention_timing": self.intervention_timing,
            "history": self.history,
            "turns": self.turns,
            "replicate": self.replicate,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "history_window_messages": self.history_window_messages,
            "seed": self.seed,
            "notebook": self.notebook,
        }
        config.update(dict(self.extra))
        return config


@dataclass(frozen=True)
class TurnResult:
    """One model response within an experiment execution."""

    condition: str
    turn: int
    prompt: str
    response: str
    compliant: bool
    violation_count: int
    state_of_being_count: int
    contraction_count: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {
            "condition": self.condition,
            "turn": self.turn,
            "prompt": self.prompt,
            "user_message": self.prompt,
            "model_response": self.response,
            "compliant": self.compliant,
            "violation_count": self.violation_count,
            "state_of_being_count": self.state_of_being_count,
            "contraction_count": self.contraction_count,
        }
        row.update(dict(self.extra))
        return row


@dataclass(frozen=True)
class RunMetrics:
    compliance_rate: float
    mean_violations: float
    total_violations: int
    mean_state_of_being_count: float
    total_contractions: int
    first_violation_turn: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "compliance_rate": self.compliance_rate,
            "mean_violations": self.mean_violations,
            "total_violations": self.total_violations,
            "mean_state_of_being_count": self.mean_state_of_being_count,
            "total_contractions": self.total_contractions,
            "first_violation_turn": self.first_violation_turn,
        }

    @classmethod
    def from_turns(cls, turns: list[TurnResult]) -> "RunMetrics":
        if not turns:
            return cls(0.0, 0.0, 0, 0.0, 0, None)

        total_violations = sum(turn.violation_count for turn in turns)
        total_state_of_being = sum(turn.state_of_being_count for turn in turns)
        total_contractions = sum(turn.contraction_count for turn in turns)
        compliant_count = sum(1 for turn in turns if turn.compliant)
        first_violation_turn = next(
            (turn.turn for turn in turns if turn.violation_count > 0),
            None,
        )
        count = len(turns)
        return cls(
            compliance_rate=compliant_count / count,
            mean_violations=total_violations / count,
            total_violations=total_violations,
            mean_state_of_being_count=total_state_of_being / count,
            total_contractions=total_contractions,
            first_violation_turn=first_violation_turn,
        )


@dataclass(frozen=True)
class ExperimentResult:
    """One actual execution/replicate of an experiment condition."""

    condition: ExperimentCondition
    turns: list[TurnResult]
    metrics: RunMetrics | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    full_result: Mapping[str, Any] | None = None

    def resolved_metrics(self) -> RunMetrics:
        return self.metrics or RunMetrics.from_turns(self.turns)

    def to_payload(self) -> dict[str, Any]:
        return {
            "config": self.condition.to_config(),
            "metrics": self.resolved_metrics().to_dict(),
            "provenance": dict(self.provenance),
            "turns": [turn.to_row() for turn in self.turns],
            "full_result": self.full_result,
        }


def collect_runtime_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Collect runtime provenance without guessing unavailable values."""

    return {
        "git_sha": _git_sha(repo_root),
        "torch_version": _module_version("torch"),
        "vllm_version": _module_version("vllm"),
        "spotlight_version": _spotlight_version(repo_root),
    }


def _git_sha(repo_root: str | Path | None) -> str:
    cwd = Path(repo_root) if repo_root is not None else None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return UNKNOWN


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception:
        return UNKNOWN
    return str(getattr(module, "__version__", UNKNOWN))


def _spotlight_version(repo_root: str | Path | None) -> str:
    sha = _git_sha(repo_root)
    return sha if sha != UNKNOWN else UNKNOWN

