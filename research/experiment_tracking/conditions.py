"""Canonical experiment conditions for MLR-20 alpha sweeps."""

from research.experiment_tracking.schemas import ExperimentCondition


MLR20_NOTION_MATRIX_URL = "https://app.notion.com/p/67e02acb45554df7811f1c8b3203c7c8"
MLR20_NOTION_MATRIX_ID = "67e02acb45554df7811f1c8b3203c7c8"
MLR20_COMPARISON_GROUP = "MLR-20-A-alpha-sweep"
MLR20_C_SERIES_COMPARISON_GROUP = "MLR-20-C-aggregation-resolution"
MLR20_MODEL = "Qwen/Qwen2-1.5B-Instruct"
MLR20_TURNS = 10
MLR20_REPLICATE = 1

_MLR20_ALPHA_SWEEP_ROWS = (
    (
        "A0",
        "A0 Baseline control",
        "https://app.notion.com/3df5023b4c7881209056dc93108a4431",
        False,
        None,
        "baseline",
    ),
    (
        "A1",
        "A1 Spotlight α=0.05",
        "https://app.notion.com/3df5023b4c7881d8aac6c230109f13a9",
        True,
        0.05,
        "spotlight",
    ),
    (
        "A2",
        "A2 Spotlight α=0.10",
        "https://app.notion.com/3df5023b4c78812593e6d0747d58fe15",
        True,
        0.10,
        "spotlight",
    ),
    (
        "A3",
        "A3 Spotlight α=0.15",
        "https://app.notion.com/3df5023b4c78819aa4c1d3a04d59f6ac",
        True,
        0.15,
        "spotlight",
    ),
    (
        "A4",
        "A4 Spotlight α=0.20 checkpoint",
        "https://app.notion.com/3df5023b4c788165880ff30bcb397089",
        True,
        0.20,
        "spotlight",
    ),
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

    common_extra = {
        "design_matrix_source": "Notion Experimental Design Matrix",
        "design_matrix_url": MLR20_NOTION_MATRIX_URL,
        "design_matrix_id": MLR20_NOTION_MATRIX_ID,
        "aggregation": "Global aggregation",
    }
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
    }

    return [
        ExperimentCondition(
            condition_id=condition_id,
            spotlight=spotlight,
            alpha=alpha,
            implementation=implementation,
            extra={
                **common_extra,
                "notion_experiment": experiment,
                "notion_row_url": notion_row_url,
                "notion_model": "Qwen2-1.5B-Instruct",
                "notion_comparison_group": "A — Alpha sweep",
                "notion_spotlight": "On" if spotlight else "Off",
                "notion_intervention_timing": "Prefill",
                "notion_history": "Self-propagating",
                "notion_implementation": "Global aggregation",
            },
            **common,
        )
        for (
            condition_id,
            experiment,
            notion_row_url,
            spotlight,
            alpha,
            implementation,
        ) in _MLR20_ALPHA_SWEEP_ROWS
    ]


def mlr20_query_preserving_conditions(
    *,
    temperature: float,
    max_tokens: int,
    history_window_messages: int | None,
    seed: int | None = None,
    notebook: str | None = "notebooks/demo_spotlight_e_prime_query_preserving_colab.ipynb",
) -> list[ExperimentCondition]:
    """Return the focused A2/C0/C2 E-Prime comparison for MLR-30/32.

    C2 is a novel aggregation-granularity ablation. It should not be described
    as C1/C1v2 or as paper/reference-faithful.
    """

    common = {
        "comparison_group": MLR20_C_SERIES_COMPARISON_GROUP,
        "spotlight": True,
        "alpha": 0.10,
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
        "tags": ("MLR-20", "MLR-30", "MLR-32", "C-series", "aggregation-resolution"),
    }

    return [
        ExperimentCondition(
            condition_id="A2",
            implementation="spotlight",
            extra={
                "notion_experiment": "A2 Spotlight alpha=.10",
                "notion_comparison_group": "A — Alpha sweep",
                "notion_implementation": "Global aggregation",
                "aggregation": "Global aggregation",
                "worker_name": "probe_spotlight",
                "role": "canonical reference",
            },
            **common,
        ),
        ExperimentCondition(
            condition_id="C0",
            implementation="global_diagnostic",
            extra={
                "notion_experiment": "C0 Global aggregation + per-query diagnostic alpha 0.10",
                "notion_comparison_group": "C — Aggregation resolution",
                "notion_implementation": "Global aggregation with per-query diagnostic logging",
                "aggregation": "Global aggregation",
                "diagnostic": "Record per-query psi_current(i) while retaining canonical global aggregation",
                "spotlight_implementation_mode": "C0_GLOBAL_DIAGNOSTIC",
                "worker_name": "probe_spotlight",
                "role": "diagnostic reference",
            },
            **common,
        ),
        ExperimentCondition(
            condition_id="C2",
            implementation="query_preserving_spotlight",
            extra={
                "notion_experiment": "C2 Query-preserving Spotlight worker alpha=.10",
                "notion_comparison_group": "C — Aggregation resolution",
                "notion_implementation": "Query-preserving Spotlight worker",
                "aggregation": "Query-indexed control",
                "diagnostic": "Novel aggregation-granularity ablation; preserves Q in psi_current/bias",
                "spotlight_implementation_mode": "C2_QUERY_PRESERVING_WORKER",
                "worker_name": "probe_spotlight_query_preserving",
                "role": "novel MLR-30/32 ablation",
            },
            **common,
        ),
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
