"""Explicit representative conditions for MLR-20-style alpha sweeps."""

from research.experiment_tracking.schemas import ExperimentCondition


def alpha_sweep_conditions(
    *,
    model: str,
    turns: int,
    temperature: float,
    max_tokens: int,
    history_window_messages: int | None,
    replicate: int = 1,
    notebook: str | None = None,
) -> list[ExperimentCondition]:
    """Return valid baseline plus Spotlight alpha conditions.

    The baseline has no alpha and no Spotlight implementation, avoiding invalid
    factorial combinations where alpha is set while Spotlight is disabled.
    """

    common = {
        "comparison_group": "MLR-20-A-alpha-sweep",
        "model": model,
        "constraint_formulation": "E-Prime",
        "constraint_complexity": "state-of-being-verbs-and-listed-contractions",
        "intervention_timing": "prefill",
        "history": "rolling-long-conversation",
        "turns": turns,
        "replicate": replicate,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "history_window_messages": history_window_messages,
        "notebook": notebook,
        "tags": ("MLR-20", "alpha-sweep"),
    }
    return [
        ExperimentCondition(
            condition_id="A0",
            spotlight=False,
            alpha=None,
            implementation="baseline",
            **common,
        ),
        *[
            ExperimentCondition(
                condition_id=condition_id,
                spotlight=True,
                alpha=alpha,
                implementation="spotlight",
                **common,
            )
            for condition_id, alpha in [
                ("A1", 0.05),
                ("A2", 0.10),
                ("A3", 0.15),
                ("A4", 0.20),
            ]
        ],
    ]

