#!/usr/bin/env python3
"""Classify notebook translation risk before model generation."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


KNOWN_COMPONENTS = {
    "attention-tracker": {
        "identity_markers": ("attention-tracker", "attention_tracker", "attn_tracker", "attention_tracker_analyzer"),
        "support_markers": ("probe_hook_qk",),
        "worker": "probe_hookqk_worker.py",
        "analyzer": "attention_tracker_analyzer.py",
    },
    "core-reranker": {
        "identity_markers": ("core-reranker", "core_reranker", "core_reranker_analyzer"),
        "support_markers": ("probe_hook_qk",),
        "worker": "probe_hookqk_worker.py",
        "analyzer": "core_reranker_analyzer.py",
    },
    "activation-steering": {
        "identity_markers": ("activation-steering", "activation_steer"),
        "support_markers": ("steer_hook_act", "steer_activation_worker"),
        "worker": "steer_activation_worker.py",
        "analyzer": None,
    },
    "hidden-states": {
        "identity_markers": ("hidden-states", "hidden_states", "hidden_states_analyzer"),
        "support_markers": ("probe_hidden_states",),
        "worker": "probe_hidden_states_worker.py",
        "analyzer": "hidden_states_analyzer.py",
    },
    "science-hallucination": {
        "identity_markers": ("science-hallucination", "science_hallucination", "science_hallucination_analyzer"),
        "support_markers": ("probe_hidden_states",),
        "worker": "probe_hidden_states_worker.py",
        "analyzer": "science_hallucination_analyzer.py",
    },
    "spotlight": {
        "identity_markers": ("spotlight", "generate_with_spotlight"),
        "support_markers": ("probe_spotlight", "spotlight_worker"),
        "worker": "spotlight_worker.py",
        "analyzer": None,
    },
}

BACKEND_SENSITIVE_MARKERS = (
    "torch.float16",
    "worker_extension_cls",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "enforce_eager",
    "Flash Attention",
    "CUDA",
    "gpu_memory_utilization",
    "/dev/shm",
)


@dataclass(frozen=True)
class ComponentEvidence:
    name: str
    worker: str | None
    analyzer: str | None


@dataclass(frozen=True)
class Classification:
    complexity: str
    translation_confidence: str
    source_notebook: str
    target_notebook: str
    source_branch: str
    target_branch: str
    target_platform: str
    components: list[dict[str, object]]
    unmapped_or_review_required: list[str]
    reasons: list[str]
    caveat: str


def notebook_text(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    parts: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            parts.extend(str(line) for line in source)
        else:
            parts.append(str(source))
    return "\n".join(parts)


def detect_components(text: str, source_path: Path) -> list[ComponentEvidence]:
    haystack = f"{source_path.as_posix()}\n{text}".lower()
    found: list[ComponentEvidence] = []

    for name, spec in KNOWN_COMPONENTS.items():
        identity_markers = spec["identity_markers"]
        if any(marker.lower() in haystack for marker in identity_markers):
            found.append(
                ComponentEvidence(
                    name=name,
                    worker=spec["worker"],
                    analyzer=spec["analyzer"],
                )
            )

    return found


def has_backend_sensitive_behavior(text: str) -> bool:
    return any(marker.lower() in text.lower() for marker in BACKEND_SENSITIVE_MARKERS)


def target_exists(path: Path) -> bool:
    return path.exists()


def classify(
    source_path: Path,
    target_path: Path,
    source_branch: str,
    target_branch: str,
) -> Classification:
    text = notebook_text(source_path)
    components = detect_components(text, source_path)
    target_platform = "colab"
    target_present = target_exists(target_path)
    backend_sensitive = has_backend_sensitive_behavior(text)

    reasons: list[str] = []
    review_required: list[str] = []

    if components:
        names = ", ".join(component.name for component in components)
        reasons.append(f"Detected known vLLM-Hook component evidence: {names}.")
    else:
        reasons.append("No known worker or analyzer correspondence was detected in the source notebook.")
        review_required.append("unknown notebook components")

    if target_present:
        reasons.append(f"Target notebook already exists on the checked-out target branch: {target_path}.")
    else:
        reasons.append(f"Target notebook does not exist on the checked-out target branch: {target_path}.")

    complexity = "low" if components else "medium"
    confidence = "high" if components else "medium"
    reasons.append("MLR-2 notebook translation is scoped to the repository's Colab implementation path.")

    if backend_sensitive:
        reasons.append("Source notebook contains backend/runtime-sensitive configuration markers.")

    if not review_required and confidence != "high":
        review_required.append("maintainer review of translated notebook semantics")

    return Classification(
        complexity=complexity,
        translation_confidence=confidence,
        source_notebook=source_path.as_posix(),
        target_notebook=target_path.as_posix(),
        source_branch=source_branch,
        target_branch=target_branch,
        target_platform=target_platform,
        components=[asdict(component) for component in components],
        unmapped_or_review_required=dedupe(review_required),
        reasons=dedupe(reasons),
        caveat=(
            "Complexity and translation confidence describe implementation correspondence and review risk; "
            "they are not proof of semantic equivalence or correctness."
        ),
    )


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = re.sub(r"\s+", " ", value.strip()).lower()
        if key and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def render_markdown(result: Classification) -> str:
    lines = [
        "# Notebook Translation Classification",
        "",
        f"- Complexity: **{result.complexity}**",
        f"- Translation confidence: **{result.translation_confidence}**",
        f"- Source notebook: `{result.source_notebook}` from `{result.source_branch}`",
        f"- Target notebook: `{result.target_notebook}` on `{result.target_branch}`",
        f"- Target platform: `{result.target_platform}`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in result.reasons)
    lines.extend(["", "## Unmapped Or Review-Required Components", ""])
    if result.unmapped_or_review_required:
        lines.extend(f"- {item}" for item in result.unmapped_or_review_required)
    else:
        lines.append("- None detected by deterministic rules.")
    lines.extend(["", "## Caveat", "", result.caveat, ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-notebook", required=True, type=Path)
    parser.add_argument("--target-notebook", required=True, type=Path)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--target-branch", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = classify(
        source_path=args.source_notebook,
        target_path=args.target_notebook,
        source_branch=args.source_branch,
        target_branch=args.target_branch,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(result), encoding="utf-8")

    print(f"Complexity: {result.complexity}")
    print(f"Translation confidence: {result.translation_confidence}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
