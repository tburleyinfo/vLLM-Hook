"""Token-span helpers for chat-template-aware Attention Tracker prompts."""
from __future__ import annotations

from typing import Any


def chat_template_token_range(
    tokenizer: Any, messages: list[dict], text: str
) -> tuple[int, int]:
    """Find ``text`` in a rendered chat template and return its token interval."""
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    char_start = rendered.find(text)
    if char_start < 0:
        raise ValueError(f"Text span {text!r} is absent from the rendered chat template")
    char_end = char_start + len(text)

    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise TypeError("Tokenizer must provide offset_mapping for chat-template ranges")
    if len(offsets) and len(offsets[0]) and isinstance(offsets[0][0], (list, tuple)):
        offsets = list(offsets[0])
    overlapping = [
        i
        for i, (start, end) in enumerate(offsets)
        if end > start and start < char_end and end > char_start
    ]
    if not overlapping:
        raise ValueError(f"No tokenizer offsets overlap the requested text span {text!r}")
    return overlapping[0], overlapping[-1] + 1
