"""E-Prime decoding-time constraint helpers.

This module intentionally enforces a surface-form rule, not a full English
grammar. It blocks candidate tokens when appending that token would create a
state-of-being verb or a listed contraction in the assistant output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch


DEFAULT_BANNED_FORMS = (
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
)

DEFAULT_BANNED_CONTRACTIONS = (
    "i'm",
    "you're",
    "we're",
    "they're",
    "he's",
    "she's",
    "it's",
    "that's",
    "there's",
    "what's",
    "who's",
    "where's",
    "when's",
    "why's",
    "how's",
)


def default_eprime_bad_words() -> list[str]:
    """Return E-Prime surface forms for vLLM's native bad_words processor."""
    return list(DEFAULT_BANNED_FORMS) + list(DEFAULT_BANNED_CONTRACTIONS)


def _compile_violation_pattern(
    banned_forms: Iterable[str],
    banned_contractions: Iterable[str],
) -> re.Pattern[str]:
    forms = [re.escape(x.lower()) for x in banned_forms]
    contractions = [re.escape(x.lower()) for x in banned_contractions]
    parts = []
    if forms:
        parts.append(r"\b(?:" + "|".join(forms) + r")\b")
    if contractions:
        parts.append(r"\b(?:" + "|".join(contractions) + r")\b")
    return re.compile("|".join(parts), re.IGNORECASE)


@dataclass
class EPrimeLogitsProcessor:
    """Block next-token candidates that create E-Prime surface violations."""

    tokenizer: Any
    banned_forms: Sequence[str] = DEFAULT_BANNED_FORMS
    banned_contractions: Sequence[str] = DEFAULT_BANNED_CONTRACTIONS
    tail_chars: int = 48
    ban_value: float = -float("inf")
    _candidate_text_cache: dict[int, str] = field(default_factory=dict, init=False)
    _mask_cache: dict[str, torch.Tensor] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._pattern = _compile_violation_pattern(
            self.banned_forms,
            self.banned_contractions,
        )

    def __call__(self, *args: Any) -> torch.Tensor:
        """vLLM-compatible callable.

        vLLM versions differ in the exact logits-processor signature. This
        accepts the common forms and treats the last positional argument as the
        logits tensor.
        """
        if len(args) < 2:
            raise TypeError("EPrimeLogitsProcessor expects token ids and logits")

        scores = args[-1]
        token_ids = args[-2]
        if not isinstance(scores, torch.Tensor):
            raise TypeError("last logits-processor argument must be a torch.Tensor")

        generated_ids = self._flatten_token_ids(token_ids)
        prefix_tail = self._decode(generated_ids)[-self.tail_chars :].lower()
        mask = self._bad_token_mask(prefix_tail, scores.shape[-1], scores.device)
        return scores.masked_fill(mask, self.ban_value)

    def would_violate(self, prefix_text: str, candidate_text: str) -> bool:
        combined = (prefix_text[-self.tail_chars :] + candidate_text).lower()
        return bool(self._pattern.search(combined))

    def _bad_token_mask(
        self,
        prefix_tail: str,
        vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        cached = self._mask_cache.get(prefix_tail)
        if cached is None or cached.numel() != vocab_size:
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            for token_id in range(vocab_size):
                token_text = self._candidate_text(token_id)
                if self.would_violate(prefix_tail, token_text):
                    mask[token_id] = True
            self._mask_cache[prefix_tail] = mask
            cached = mask
        return cached.to(device=device)

    def _candidate_text(self, token_id: int) -> str:
        text = self._candidate_text_cache.get(token_id)
        if text is None:
            text = self._decode([token_id])
            self._candidate_text_cache[token_id] = text
        return text

    def _decode(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(list(token_ids))

    @staticmethod
    def _flatten_token_ids(token_ids: Any) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return [int(x) for x in token_ids]


def build_eprime_logits_processor(tokenizer: Any, **kwargs: Any) -> EPrimeLogitsProcessor:
    return EPrimeLogitsProcessor(tokenizer=tokenizer, **kwargs)
