"""Canonical experiment conditions for MLR-20 alpha sweeps."""

from research.experiment_tracking.schemas import ExperimentCondition


MLR20_NOTION_MATRIX_URL = "https://app.notion.com/p/67e02acb45554df7811f1c8b3203c7c8"
MLR20_NOTION_MATRIX_ID = "67e02acb45554df7811f1c8b3203c7c8"
MLR20_COMPARISON_GROUP = "MLR-20-A-alpha-sweep"
MLR20_MODEL = "Qwen/Qwen2-1.5B-Instruct"
MLR20_TURNS = 10
MLR20_REPLICATE = 1

_MLR20_ALPHA_SWEEP_ROWS = (
    ("A0", False, None, "baseline"),
    ("A1", True, 0.05, "spotlight"),
    ("A2", True, 0.10, "spotlight"),
    ("A3", True, 0.15, "spotlight"),
    ("A4", True, 0.20, "spotlight"),
)


def mlr20_alpha_sweep_conditions(
    *,
    temperature: float,
    max_tokens: int,
    history_window_messages: int | None,
    seed: int | None = None,
    notebook: str | None = None,
) -> list[ExperimentCondition]:
    """Return the Notion Experimental Design Matrix rows for MLR-20.

    One returned condition is one Notion row, one ExperimentCondition, and one
    W&B run. Baseline and Spotlight rows are intentionally not combined.
    """

    common = {
        "comparison_group": MLR20_COMPARISON_GROUP,
        "model": MLR20_MODEL,
        "constraint_formulation": "Full E-Prime",
        "constraint_complexity": "Negative enumeration",
        "intervention_timing": "Prefill intervention",
        "history": "Self-propagating history",
        "turns": MLR20_TURNS,
        "replicate": MLR20_REPLICATE,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "history_window_messages": history_window_messages,
        "seed": seed,
        "notebook": notebook,
        "tags": ("MLR-20", "alpha-sweep", "notion-design-matrix"),
        "extra": {
            "design_matrix_source": "Notion Experimental Design Matrix",
            "design_matrix_url": MLR20_NOTION_MATRIX_URL,
            "design_matrix_id": MLR20_NOTION_MATRIX_ID,
            "aggregation": "Global aggregation",
        },
    }

    return [
        ExperimentCondition(
            condition_id=condition_id,
            spotlight=spotlight,
            alpha=alpha,
            implementation=implementation,
            **common,
        )
        for condition_id, spotlight, alpha, implementation in _MLR20_ALPHA_SWEEP_ROWS
    ]


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

    if (
        model == MLR20_MODEL
        and turns == MLR20_TURNS
        and replicate == MLR20_REPLICATE
    ):
        return mlr20_alpha_sweep_conditions(
            temperature=temperature,
            max_tokens=max_tokens,
            history_window_messages=history_window_messages,
            notebook=notebook,
        )

    common = {
        "comparison_group": MLR20_COMPARISON_GROUP,
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
