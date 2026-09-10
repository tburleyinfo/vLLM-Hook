#!/usr/bin/env python3
"""Review pending CUDA/Metal correspondence entries.

This is a deliberately simple CLI companion to correspondence_scanner.py:

  python3 tools/verify_correspondence.py list
  python3 tools/verify_correspondence.py approve probe-hookqk-worker --note "..."
  python3 tools/verify_correspondence.py reject science-hallucination-analyzer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PENDING_PATH = PROJECT_ROOT / "pending_correspondence.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "correspondence.md"


def load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def markdown_escape(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def render_registry(entries: list[dict[str, Any]], path: Path) -> None:
    registry_entries = [
        entry
        for entry in entries
        if entry.get("status") in {"approved", "strong_candidate"}
        or any(row.get("status") in {"approved", "strong_candidate"} for row in entry.get("comparison_rows", []))
    ]
    lines = [
        "# vLLM-Hook CUDA/Metal Correspondence",
        "",
        "This registry is generated from reviewed correspondence entries.",
        "",
        "| Logical Component | CUDA Path | Metal Path | Status | Confidence |",
        "|---|---|---|---|---|",
    ]
    for entry in sorted(registry_entries, key=lambda item: item["logical_component"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(entry["logical_component"]),
                    f"`{markdown_escape(entry['cuda_path'])}`",
                    f"`{markdown_escape(entry['metal_path'])}`",
                    markdown_escape(entry["status"]),
                    markdown_escape(entry["confidence"]),
                ]
            )
            + " |"
        )
    for entry in sorted(registry_entries, key=lambda item: item["logical_component"]):
        lines.extend(
            [
                "",
                f"## {markdown_escape(entry['logical_component'])}",
                "",
                markdown_escape(entry.get("divergence_note", "")),
                "",
                "| Row Status | Kind | CUDA Ref | CUDA | Metal Ref | Metal | Relation |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for row in entry.get("comparison_rows", []):
            if row.get("status") == "rejected":
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_escape(row.get("status", "pending")),
                        markdown_escape(row.get("kind", "")),
                        markdown_escape(row.get("cuda_ref", "")),
                        markdown_escape(row.get("cuda", "")),
                        markdown_escape(row.get("metal_ref", "")),
                        markdown_escape(row.get("metal", "")),
                        markdown_escape(row.get("relation", "")),
                    ]
                )
                + " |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_entry(entries: list[dict[str, Any]], entry_id: str) -> dict[str, Any]:
    for entry in entries:
        if entry["id"] == entry_id:
            return entry
    raise SystemExit(f"Unknown correspondence id: {entry_id}")


def list_entries(entries: list[dict[str, Any]]) -> int:
    if not entries:
        print("No pending correspondence entries found.")
        return 0
    for entry in entries:
        print(
            f"{entry['id']}: {entry['status']} / {entry['confidence']} / "
            f"{entry['logical_component']}"
        )
        print(f"  CUDA : {entry['cuda_path']}")
        print(f"  Metal: {entry['metal_path']}")
        print(f"  Note : {entry['divergence_note']}")
    return 0


def approve_entry(args: argparse.Namespace, entries: list[dict[str, Any]]) -> int:
    entry = find_entry(entries, args.entry_id)
    entry["status"] = "approved"
    if args.note:
        entry["divergence_note"] = args.note
    if args.confidence:
        entry["confidence"] = args.confidence
    save_entries(args.pending, entries)
    render_registry(entries, args.registry)
    print(f"Approved {args.entry_id} and regenerated {args.registry}")
    return 0


def reject_entry(args: argparse.Namespace, entries: list[dict[str, Any]]) -> int:
    entry = find_entry(entries, args.entry_id)
    entry["status"] = "rejected"
    if args.reason:
        entry["reason"] = args.reason
    save_entries(args.pending, entries)
    render_registry(entries, args.registry)
    print(f"Rejected {args.entry_id} and regenerated {args.registry}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending", type=Path, default=PENDING_PATH)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    approve = subparsers.add_parser("approve")
    approve.add_argument("entry_id")
    approve.add_argument("--note")
    approve.add_argument("--confidence", choices=["high", "medium", "low"])

    reject = subparsers.add_parser("reject")
    reject.add_argument("entry_id")
    reject.add_argument("--reason")

    regen = subparsers.add_parser("regen")

    args = parser.parse_args()
    entries = load_entries(args.pending)
    if args.command == "list":
        return list_entries(entries)
    if args.command == "approve":
        return approve_entry(args, entries)
    if args.command == "reject":
        return reject_entry(args, entries)
    if args.command == "regen":
        render_registry(entries, args.registry)
        print(f"Regenerated {args.registry}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
