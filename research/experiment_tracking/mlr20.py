"""MLR-20 orchestration helpers."""

from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any

from research.experiment_tracking.conditions import MLR20_COMPARISON_GROUP
from research.experiment_tracking.eprime import (
    extract_current_assistant_response,
    turn_result_from_assistant_score,
)
from research.experiment_tracking.schemas import (
    ExperimentCondition,
    ExperimentResult,
    RunMetrics,
)
from research.experiment_tracking.wandb_adapter import WandbTracker


def evaluate_mlr20_condition(
    *,
    condition: ExperimentCondition,
    llm: Any,
    sampling_params: Any,
    seed_history: list[dict[str, str]],
    user_turns: list[str],
    render_prompt: Any,
    score_fn: Any,
    generate_with_spotlight_fn: Any,
    spotlight_span: str,
    provenance: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Evaluate one MLR-20 condition using a caller-owned vLLM engine.

    The function owns per-condition conversational state only. Engine lifetime,
    CUDA memory checks, W&B login, and notebook display concerns stay outside so
    the same evaluator can be called directly while debugging one condition or
    from a sequential batch runner.
    """

    history = [dict(item) for item in seed_history]
    turns = []
    raw_rows = []

    for turn_index, user_message in enumerate(
        user_turns[: condition.turns],
        start=1,
    ):
        prompt = render_prompt(history, user_message)
        started = time.perf_counter()
        if condition.spotlight:
            if condition.alpha is None:
                raise ValueError(
                    f"Condition {condition.condition_id} enables Spotlight but has no alpha."
                )
            outputs = generate_with_spotlight_fn(
                llm,
                prompts=[prompt],
                emph_strings=[spotlight_span],
                alpha=condition.alpha,
                sampling_params=sampling_params,
            )
        else:
            outputs = llm.generate(
                prompts=[prompt],
                sampling_params=sampling_params,
                use_hook=False,
            )

        latency_s = time.perf_counter() - started
        reply = extract_current_assistant_response(outputs[0].outputs[0].text, prompt)
        extra = {
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "history_messages_in_prompt": _history_messages_in_prompt(
                prompt,
                history,
            ),
            "latency_s": latency_s,
        }
        score = score_fn(reply)
        turn = turn_result_from_assistant_score(
            condition=condition.condition_id,
            turn=turn_index,
            user_message=user_message,
            assistant_response=reply,
            score=score,
            extra={**extra, **_score_match_details(score)},
        )
        turns.append(turn)
        raw_rows.append(
            {
                **turn.to_row(),
                "reply": reply,
                "e_prime_score": 1.0 if turn.compliant else 0.0,
                "e_prime_retained": turn.compliant,
                "e_prime_violation_count": turn.violation_count,
                "state_of_being_count": turn.state_of_being_count,
                "contraction_count": turn.contraction_count,
                **dict(turn.extra),
            }
        )
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": reply},
            ]
        )

    return ExperimentResult(
        condition=condition,
        turns=turns,
        metrics=RunMetrics.from_turns(turns),
        provenance=provenance or {},
        full_result={
            "condition_id": condition.condition_id,
            "rows": raw_rows,
        },
    )


def run_mlr20_batch(
    conditions: Iterable[ExperimentCondition],
    *,
    llm: Any,
    sampling_params: Any,
    seed_history: list[dict[str, str]],
    user_turns: list[str],
    render_prompt: Any,
    score_fn: Any,
    generate_with_spotlight_fn: Any,
    spotlight_span: str,
    provenance: dict[str, Any] | None = None,
) -> list[ExperimentResult]:
    """Run isolated MLR-20 conditions sequentially through one vLLM engine."""

    results = []
    seen = set()
    for condition in conditions:
        key = (condition.condition_id, condition.replicate)
        if key in seen:
            raise ValueError(
                "Each Notion condition/replicate must be evaluated once per batch; "
                f"duplicate condition {condition.condition_id} replicate {condition.replicate}."
            )
        seen.add(key)
        results.append(
            evaluate_mlr20_condition(
                condition=condition,
                llm=llm,
                sampling_params=sampling_params,
                seed_history=seed_history,
                user_turns=user_turns,
                render_prompt=render_prompt,
                score_fn=score_fn,
                generate_with_spotlight_fn=generate_with_spotlight_fn,
                spotlight_span=spotlight_span,
                provenance=provenance,
            )
        )
    return results


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


def _history_messages_in_prompt(prompt: str, history: list[dict[str, str]]) -> int:
    """Best-effort prompt history count for notebook diagnostics."""

    count = sum(
        1
        for item in history
        if f"{item['role'].upper()}: {item['content']}" in prompt
    )
    return count


def _score_match_details(score: Any) -> dict[str, Any]:
    return {
        key: score[key]
        for key in ("state_of_being_matches", "contraction_matches")
        if key in score
    }


def _mlr20_run_name(result: ExperimentResult) -> str:
    condition = result.condition
    alpha = "NA" if condition.alpha is None else f"{condition.alpha:.2f}"
    spotlight = "spotlight-on" if condition.spotlight else "spotlight-off"
    return f"{condition.condition_id}-r{condition.replicate}-{spotlight}-alpha-{alpha}"
