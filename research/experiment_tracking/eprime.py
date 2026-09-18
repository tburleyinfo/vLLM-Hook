"""E-Prime tracking helpers that keep scoring scoped to assistant text."""

from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any, Mapping

from research.experiment_tracking.schemas import TurnResult


_NEXT_TURN_RE = re.compile(r"\n(?:USER|ASSISTANT):")


def extract_current_assistant_response(generated_text: str, prompt: str) -> str:
    """Extract only the newly generated assistant text from raw generation output."""

    text = generated_text or ""
    if prompt and text.startswith(prompt):
        text = text[len(prompt) :]

    text = text.lstrip()
    if text.startswith("ASSISTANT:"):
        text = text[len("ASSISTANT:") :].lstrip()

    next_turn = _NEXT_TURN_RE.search(text)
    if next_turn:
        text = text[: next_turn.start()]

    return text.strip().split("\n\n")[0].strip()


def turn_result_from_eprime_row(row: Mapping[str, Any]) -> TurnResult:
    """Build a tracked turn from an E-Prime result row.

    The row must already contain violation fields computed from the generated
    assistant response. User/prompt text is preserved only as context.
    """

    return TurnResult(
        condition=str(row["condition"]),
        turn=int(row["turn"]),
        prompt=str(row.get("prompt", row["user_message"])),
        user_message=str(row["user_message"]),
        response=str(row["reply"]),
        compliant=bool(row["e_prime_retained"]),
        violation_count=int(row["e_prime_violation_count"]),
        state_of_being_count=int(row["state_of_being_count"]),
        contraction_count=int(row["contraction_count"]),
        extra={
            "prompt_chars": int(row["prompt_chars"]),
            "history_messages_in_prompt": int(row["history_messages_in_prompt"]),
            "latency_s": float(row["latency_s"]),
            "state_of_being_matches": row["state_of_being_matches"],
            "contraction_matches": row["contraction_matches"],
            "score_text_source": "assistant_response",
        },
    )


def turn_result_from_assistant_score(
    *,
    condition: str,
    turn: int,
    user_message: str,
    assistant_response: str,
    score: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> TurnResult:
    """Create a tracked turn from an assistant-only E-Prime score."""

    turn_extra = {"score_text_source": "assistant_response"}
    if extra:
        turn_extra.update(dict(extra))

    return TurnResult(
        condition=condition,
        turn=turn,
        prompt=user_message,
        user_message=user_message,
        response=assistant_response,
        compliant=bool(score["e_prime_retained"]),
        violation_count=int(score["e_prime_violation_count"]),
        state_of_being_count=int(score["state_of_being_count"]),
        contraction_count=int(score["contraction_count"]),
        extra=turn_extra,
    )


def turn_result_from_eprime_texts(
    *,
    condition: str,
    turn: int,
    user_message: str,
    assistant_response: str,
    score_fn: Callable[[str], Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> TurnResult:
    """Score only the assistant response, while preserving user text as context."""

    return turn_result_from_assistant_score(
        condition=condition,
        turn=turn,
        user_message=user_message,
        assistant_response=assistant_response,
        score=score_fn(assistant_response),
        extra=extra,
    )
