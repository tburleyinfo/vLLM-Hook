#!/usr/bin/env python3
"""Local web UI for reviewing CUDA/Metal correspondence candidates."""

from __future__ import annotations

import argparse
import difflib
import json
import mimetypes
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PENDING_PATH = PROJECT_ROOT / "pending_correspondence.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "correspondence.md"
DEFAULT_LLM_URL = os.environ.get("CORRESPONDENCE_LLM_URL", "http://127.0.0.1:8033")
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
ENTRY_LOCK = threading.Lock()
MAX_ANALYSIS_CHUNKS = int(os.environ.get("CORRESPONDENCE_ANALYSIS_CHUNKS", "12"))


def load_entries() -> list[dict[str, Any]]:
    if not PENDING_PATH.exists():
        return []
    return json.loads(PENDING_PATH.read_text(encoding="utf-8"))


def save_entries(entries: list[dict[str, Any]]) -> None:
    PENDING_PATH.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def markdown_escape(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def render_registry(entries: list[dict[str, Any]]) -> None:
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
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_entry(entries: list[dict[str, Any]], entry_id: str) -> dict[str, Any] | None:
    return next((entry for entry in entries if entry.get("id") == entry_id), None)


def read_text(path_text: str) -> str:
    path = (PROJECT_ROOT / path_text).resolve()
    if not is_relative_to(path, PROJECT_ROOT):
        raise ValueError("Path escapes project root")
    return path.read_text(encoding="utf-8")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def diff_for(entry: dict[str, Any]) -> str:
    cuda_lines = read_text(entry["cuda_path"]).splitlines()
    metal_lines = read_text(entry["metal_path"]).splitlines()
    return "\n".join(
        difflib.unified_diff(
            cuda_lines,
            metal_lines,
            fromfile=entry["cuda_path"],
            tofile=entry["metal_path"],
            lineterm="",
            n=4,
        )
    )


def entry_payload(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        **entry,
        "cuda_content": read_text(entry["cuda_path"]),
        "metal_content": read_text(entry["metal_path"]),
        "diff": diff_for(entry),
    }


def fetch_web_context(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Best-effort lightweight web search via DuckDuckGo's HTML endpoint."""

    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "vllm-hook-correspondence-review/0.1"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        text = response.read().decode("utf-8", errors="replace")

    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)".*?>(?P<title>.*?)</a>.*?'
        r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
        flags=re.DOTALL,
    )
    for match in pattern.finditer(text):
        title = strip_html(match.group("title"))
        snippet = strip_html(match.group("snippet"))
        href = strip_html(match.group("url"))
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def strip_html(value: str) -> str:
    value = re.sub(r"<.*?>", "", value)
    return (
        value.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#x27;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .strip()
    )


def web_results_to_context(results: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {item['title']} ({item['url']}): {item['snippet']}" for item in results
    )


def empty_web_context(query: str, exc: Exception) -> dict[str, Any]:
    return {
        "ok": True,
        "query": query,
        "provider": "duckduckgo_html",
        "provider_status": "blocked",
        "results": [],
        "context": "",
        "message": f"Web context provider was unavailable: {exc}",
    }


def candidate_context_query(entry: dict[str, Any]) -> str:
    parts = [
        str(entry.get("logical_component") or ""),
        str(entry.get("cuda_path") or ""),
        str(entry.get("metal_path") or ""),
        "vLLM Hook CUDA Metal parity",
    ]
    return " ".join(part for part in parts if part.strip())


def row_context_query(entry: dict[str, Any], row: dict[str, Any]) -> str:
    parts = [
        str(entry.get("logical_component") or ""),
        str(row.get("kind") or ""),
        str(row.get("cuda_ref") or row.get("cuda") or ""),
        str(row.get("metal_ref") or row.get("metal") or ""),
        "vLLM Hook CUDA Metal parity",
    ]
    return " ".join(part for part in parts if part.strip())


def truncate_text(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def compact_relation_row(row: dict[str, Any]) -> dict[str, str]:
    fields = ("kind", "cuda_ref", "cuda", "metal_ref", "metal", "relation", "status", "confidence")
    return {field: truncate_text(row.get(field, ""), 1200) for field in fields if str(row.get(field, "")).strip()}


def explanation_prompt(entry: dict[str, Any], row: dict[str, Any], external_context: str) -> str:
    context = ""
    if external_context.strip():
        context = f"\nReviewer guidance or optional context:\n{truncate_text(external_context, 3000)}\n"
    current_relation = truncate_text(row.get("relation", ""), 1800)
    return f"""Regenerate only the Relationship note for one row in a CUDA/Metal correspondence candidate.

Return only JSON with this shape:
{{
  "relation": "One concise relationship note for this row.",
  "confidence": "high | medium | low"
}}

Do not add or rewrite comparison_rows. Do not update the candidate summary note.
Follow reviewer edit instructions in the guidance/context field unless they conflict with the selected row.
If the reviewer asks to paraphrase, rewrite the current relationship note with different wording while preserving its meaning.
The returned relation must be a complete replacement for the current relationship note, not a small appended edit.
Do not claim external context is correct unless it is supported by the selected row and candidate data.

Logical component: {entry.get("logical_component", "")}
CUDA path: {entry.get("cuda_path", "")}
Metal path: {entry.get("metal_path", "")}

Selected row:
{json.dumps(compact_relation_row(row), indent=2)}

Current relationship note to replace:
{current_relation}

Current candidate summary note:
{truncate_text(entry.get("divergence_note", ""), 1500)}
{context}
"""


def normalize_explanation(data: dict[str, Any]) -> dict[str, Any]:
    confidence = str(data.get("confidence") or "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = ""
    return {
        "relation": str(data.get("relation") or data.get("divergence_note") or "").strip(),
        "confidence": confidence,
    }


def regenerate_explanation(entry_id: str, row: dict[str, Any], external_context: str) -> dict[str, Any]:
    from correspondence_scanner import LocalLLMClient, parse_json_object

    entries = load_entries()
    entry = find_entry(entries, entry_id)
    if entry is None:
        raise ValueError("entry not found")

    client = LocalLLMClient(DEFAULT_LLM_URL, timeout=300)
    models = client.list_models()
    if not models:
        raise RuntimeError("No local LLM models were returned by /v1/models")
    model = client.model or models[0]
    content = client.chat_json(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You refine reviewer-facing correspondence explanations. "
                    "Return only valid JSON and never regenerate matrix rows."
                ),
            },
            {"role": "user", "content": explanation_prompt(entry, row, external_context)},
        ],
    )
    try:
        data = parse_json_object(content)
    except json.JSONDecodeError:
        data = parse_json_object(client.repair_json(model, content))
    return normalize_explanation(data)


def set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = time.time()


def get_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("kind") or "").strip(),
        str(row.get("cuda_ref") or "").strip(),
        str(row.get("cuda") or "").strip(),
        str(row.get("metal_ref") or "").strip(),
        str(row.get("metal") or "").strip(),
    )


def merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {row_key(row) for row in merged}
    for row in incoming:
        normalized = {**row, "status": row.get("status") or "pending"}
        if not normalized.get("generated_at"):
            normalized["generated_at"] = utc_timestamp()
        key = row_key(normalized)
        if key in seen:
            continue
        merged.append(normalized)
        seen.add(key)
    return merged


def publish_rows(job_id: str, entry_id: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with ENTRY_LOCK:
        entries = load_entries()
        entry = find_entry(entries, entry_id)
        if entry is None:
            raise ValueError("entry not found")
        entry["comparison_rows"] = merge_rows(entry.get("comparison_rows", []), rows)
        save_entries(entries)
        render_registry(entries)
    job = get_job(job_id) or {}
    partial_rows = merge_rows(job.get("partial_rows", []), rows)
    set_job(job_id, partial_rows=partial_rows)


def chunk_seed_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = entry.get("comparison_rows") or []
    if not rows:
        return [{"kind": "file", "cuda": entry.get("cuda_path", ""), "metal": entry.get("metal_path", "")}]
    return rows[:MAX_ANALYSIS_CHUNKS]


def line_range_from_ref(ref: str) -> tuple[int, int] | None:
    match = re.search(r":(\d+)(?:-(\d+))?", ref or "")
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    return min(start, end), max(start, end)


def source_excerpt(path_text: str, ref: str, radius: int = 18) -> str:
    line_range = line_range_from_ref(ref)
    if line_range is None:
        return ""
    try:
        lines = read_text(path_text).splitlines()
    except Exception:
        return ""
    start_line, end_line = line_range
    start = max(1, start_line - radius)
    end = min(len(lines), end_line + radius)
    rendered = []
    for number in range(start, end + 1):
        marker = ">>" if start_line <= number <= end_line else "  "
        rendered.append(f"{marker} {number:>5}  {lines[number - 1]}")
    return "\n".join(rendered)


def chunk_code_context(entry: dict[str, Any], seed_row: dict[str, Any]) -> str:
    blocks = []
    cuda_ref = str(seed_row.get("cuda_ref") or "")
    metal_ref = str(seed_row.get("metal_ref") or "")
    if cuda_ref:
        excerpt = source_excerpt(str(entry.get("cuda_path") or ""), cuda_ref)
        if excerpt:
            blocks.append(f"CUDA excerpt around current cuda_ref {cuda_ref}:\n{excerpt}")
    if metal_ref:
        excerpt = source_excerpt(str(entry.get("metal_path") or ""), metal_ref)
        if excerpt:
            blocks.append(f"Metal excerpt around current metal_ref {metal_ref}:\n{excerpt}")
    return "\n\n".join(blocks)


def chunk_prompt(entry: dict[str, Any], seed_row: dict[str, Any], chunk_index: int, total_chunks: int, web_context: str) -> str:
    context = ""
    if web_context.strip():
        context = f"\nOptional external context:\n{web_context.strip()}\n"
    code_context = chunk_code_context(entry, seed_row)
    if code_context:
        code_context = f"\nCode excerpts for checking refs:\n{code_context}\n"
    return f"""You are refining one chunk of a CUDA/Metal correspondence matrix.

Return only JSON with this shape:
{{
  "comparison_rows": [
    {{
      "kind": "exact | strong | mild | weak | partial | divergent | cuda_only | metal_only | no_match | risk | contract | artifact | lifecycle | dependency",
      "cuda": "CUDA-side code fact, method, block, or behavior. Leave blank if absent.",
      "cuda_ref": "CUDA path with line or range that contains the described CUDA behavior. Leave blank if absent, uncertain, or only blank/comment/decorator lines match.",
      "metal": "Metal-side code fact, method, block, or behavior. Leave blank if absent.",
      "metal_ref": "Metal path with line or range that contains the described Metal behavior. Leave blank if absent, uncertain, or only blank/comment/decorator lines match.",
      "relation": "Explain exact match, loose analogy, divergence, missing counterpart, or inspection risk.",
      "status": "pending",
      "confidence": "high | medium | low"
    }}
  ],
  "similarities": ["optional concise notes"],
  "differences": ["optional concise notes"],
  "inspection_risks": ["optional concise notes"],
  "divergence_note": "optional one-paragraph update",
  "confidence": "high | medium | low"
}}

Chunk {chunk_index + 1} of {total_chunks}.
CUDA path: {entry.get("cuda_path", "")}
Metal path: {entry.get("metal_path", "")}
Existing seed row:
{json.dumps(seed_row, indent=2)}
{code_context}
{context}
Focus only on this seed row. You may split weak or divergent matches into
adjacent one-sided rows. Do not regenerate the whole file matrix.
References must point to the smallest non-empty code range that supports the
row description. Do not cite nearby blank lines, import gaps, comments, or
decorators unless the row is specifically about those lines.
If the provided excerpt shows the current ref is empty or does not support the
description, correct the ref to a better line range visible in the excerpt, or
leave that ref blank when no supporting line range is visible.
"""


def list_extend_unique(existing: list[Any], incoming: list[Any]) -> list[str]:
    values = [str(item) for item in existing if str(item).strip()]
    seen = set(values)
    for item in incoming:
        text = str(item).strip()
        if text and text not in seen:
            values.append(text)
            seen.add(text)
    return values


def analyze_entry_job(job_id: str, entry_id: str, web_context: str) -> None:
    generation_started_at = utc_timestamp()
    set_job(
        job_id,
        status="running",
        message="Preparing candidate row chunks",
        generation_started_at=generation_started_at,
        partial_rows=[],
        completed_chunks=0,
        total_chunks=0,
        failed_chunks=[],
        progress=0,
    )
    try:
        from correspondence_scanner import LocalLLMClient, comparison_rows, parse_json_object

        entries = load_entries()
        entry = find_entry(entries, entry_id)
        if entry is None:
            raise ValueError("entry not found")
        with ENTRY_LOCK:
            entries = load_entries()
            entry = find_entry(entries, entry_id)
            if entry is None:
                raise ValueError("entry not found")
            entry["generation_started_at"] = generation_started_at
            entry["generation_completed_at"] = ""
            save_entries(entries)

        client = LocalLLMClient(DEFAULT_LLM_URL, timeout=300)
        models = client.list_models()
        if not models:
            raise RuntimeError("No local LLM models were returned by /v1/models")
        model = client.model or models[0]

        chunks = chunk_seed_rows(entry)
        total_chunks = len(chunks)
        set_job(job_id, total_chunks=total_chunks, message=f"Running 0 of {total_chunks} row chunks")
        summary_updates: dict[str, list[str]] = {"similarities": [], "differences": [], "inspection_risks": []}
        failed_chunks: list[dict[str, Any]] = []
        last_note = ""
        last_confidence = ""

        for index, seed_row in enumerate(chunks):
            set_job(
                job_id,
                status="running",
                message=f"Generating row chunk {index + 1} of {total_chunks}",
                progress=index / total_chunks if total_chunks else 0,
            )
            try:
                content = client.chat_json(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You generate small chunks of candidate correspondence tables. "
                                "Be concrete and concise. Return only valid JSON."
                            ),
                        },
                        {"role": "user", "content": chunk_prompt(entry, seed_row, index, total_chunks, web_context)},
                    ],
                )
                try:
                    data = parse_json_object(content)
                except json.JSONDecodeError:
                    data = parse_json_object(client.repair_json(model, content))
                rows = comparison_rows(data.get("comparison_rows"))
                publish_rows(job_id, entry_id, rows)
                for key in summary_updates:
                    summary_updates[key] = list_extend_unique(summary_updates[key], data.get(key) or [])
                last_note = str(data.get("divergence_note") or last_note)
                last_confidence = str(data.get("confidence") or last_confidence)
            except Exception as exc:
                failed_chunks.append({"chunk": index + 1, "message": str(exc), "seed_row": seed_row})
            completed = index + 1
            set_job(
                job_id,
                completed_chunks=completed,
                failed_chunks=failed_chunks,
                progress=completed / total_chunks if total_chunks else 1,
            )

        with ENTRY_LOCK:
            entries = load_entries()
            entry = find_entry(entries, entry_id)
            if entry is None:
                raise ValueError("entry not found")
            for key, values in summary_updates.items():
                entry[key] = list_extend_unique(entry.get(key, []), values)
            if last_note:
                entry["divergence_note"] = last_note
            if last_confidence in {"high", "medium", "low"}:
                entry["confidence"] = last_confidence
            entry["generation_started_at"] = generation_started_at
            entry["generation_completed_at"] = utc_timestamp()
            save_entries(entries)
            render_registry(entries)

        status = "partial" if failed_chunks else "done"
        message = "Candidate matrix generated"
        if failed_chunks:
            message = f"Generated with {len(failed_chunks)} failed chunk(s)"
        set_job(job_id, status=status, message=message, progress=1, generation_completed_at=entry["generation_completed_at"])
    except Exception as exc:
        failed_at = utc_timestamp()
        try:
            with ENTRY_LOCK:
                entries = load_entries()
                entry = find_entry(entries, entry_id)
                if entry is not None:
                    entry["generation_started_at"] = generation_started_at
                    entry["generation_completed_at"] = failed_at
                    save_entries(entries)
        finally:
            set_job(job_id, status="error", message=str(exc), generation_completed_at=failed_at)


def app_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>vLLM-Hook Correspondence Review</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --panelSoft: #f8fafc;
      --panelHead: #eef3f7;
      --line: #d7dee7;
      --text: #1d2733;
      --muted: #657386;
      --accent: #0a7891;
      --accentSoft: #e3f5f8;
      --warn: #9a6700;
      --danger: #b42318;
      --success: #137a49;
      --aiBg: #17435b;
      --buttonBg: #ffffff;
      --buttonHover: #8ba1b3;
      --inputBg: #ffffff;
      --refBg: #eaf3f8;
      --refText: #20576a;
      --blankBg: #f6f8fa;
      --blankStripe: #e9eef3;
      --emptyRefText: #8b98a7;
      --relationBg: #fffaf0;
      --alignedBg: #f2fbf7;
      --staggeredBg: #fff9e8;
      --unmatchedBg: #fff4f2;
      --code: #101418;
      --codeText: #eef2f3;
      --codeLine: #8ea0a8;
      --codeHit: #31424d;
      --cudaPane: 1fr;
      --metalPane: 1fr;
    }
    body[data-theme="dark"] {
      --bg: #101315;
      --panel: #181d20;
      --panelSoft: #14191c;
      --panelHead: #232a2e;
      --line: #30383d;
      --text: #ecefeb;
      --muted: #a0aaa8;
      --accent: #62c6ad;
      --accentSoft: #17352f;
      --warn: #f1c66b;
      --danger: #ff8a80;
      --success: #71d6a5;
      --aiBg: #2d4254;
      --buttonBg: #20272b;
      --buttonHover: #566167;
      --inputBg: #111619;
      --refBg: #1d3440;
      --refText: #c4e7f2;
      --blankBg: #181b1d;
      --blankStripe: #252a2e;
      --emptyRefText: #7f898d;
      --relationBg: #272216;
      --alignedBg: #142a25;
      --staggeredBg: #252112;
      --unmatchedBg: #2a1818;
      --code: #080b0d;
      --codeText: #e8edf0;
      --codeLine: #73858c;
      --codeHit: #3e5968;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      background: linear-gradient(180deg, var(--panel) 0%, var(--panelSoft) 100%);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    .subtitle { color: var(--muted); font-size: 13px; margin-top: 2px; }
    button, select, textarea, input {
      font: inherit;
    }
    select, textarea, input {
      background: var(--inputBg);
      color: var(--text);
    }
    button {
      border: 1px solid var(--line);
      background: var(--buttonBg);
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--buttonHover); }
    button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    button.ai {
      background: var(--aiBg);
      color: white;
      border-color: var(--aiBg);
    }
    button.danger {
      color: var(--danger);
      border-color: var(--danger);
    }
    .app {
      display: grid;
      grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
      min-height: calc(100vh - 58px);
      overflow: hidden;
    }
    body.sidebar-collapsed .app,
    .app.sidebar-collapsed {
      grid-template-columns: 0 minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panelSoft);
      overflow: auto;
      max-height: calc(100vh - 58px);
    }
    body.sidebar-collapsed aside,
    .app.sidebar-collapsed aside {
      overflow: hidden;
      border-right: 0;
    }
    body.sidebar-collapsed aside > *,
    .app.sidebar-collapsed aside > * {
      display: none;
    }
    .sidebar-toggle {
      position: fixed;
      top: 94px;
      left: calc(340px - 15px);
      z-index: 20;
      display: grid;
      place-items: center;
      width: 30px;
      height: 44px;
      padding: 0;
      border-radius: 999px;
      background: var(--panel);
      color: var(--muted);
      font-size: 22px;
      font-weight: 800;
      line-height: 1;
      box-shadow: 0 4px 14px color-mix(in srgb, var(--text) 14%, transparent);
    }
    body.sidebar-collapsed .sidebar-toggle {
      left: 8px;
    }
    .sidebar-toggle:hover {
      color: var(--accent);
      border-color: var(--accent);
    }
    .filters {
      display: flex;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .filters input, .filters select {
      min-width: 0;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: var(--inputBg);
      color: var(--text);
    }
    .entry {
      width: 100%;
      display: block;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      padding: 12px;
      background: transparent;
      overflow: hidden;
    }
    .entry.active {
      background: var(--accentSoft);
      border-left: 4px solid var(--accent);
      padding-left: 8px;
    }
    .entry-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-weight: 700;
      margin-bottom: 5px;
      min-width: 0;
    }
    .entry-title span:first-child {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.approved { color: var(--accent); border-color: var(--accent); }
    .pill.strong_candidate { color: var(--warn); border-color: var(--warn); }
    .pill.rejected { color: var(--danger); border-color: var(--danger); }
    .path {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    main {
      min-width: 0;
      padding: 16px;
      overflow: auto;
      max-height: calc(100vh - 58px);
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .metric strong { display: block; font-size: 20px; }
    .metric span { color: var(--muted); font-size: 12px; }
    .workbench-head {
      display: grid;
      gap: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 14px;
    }
    .workbench-title {
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .workbench-title h2 {
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
    }
    .workbench-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .workbench-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .meta {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
    }
    .analysis {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(280px, .9fr);
      gap: 12px;
      margin-bottom: 14px;
    }
    .inspector {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .inspector h3 { margin: 0 0 8px; font-size: 14px; }
    .inspector ul { margin: 8px 0 0; padding-left: 20px; }
    .inspector li { margin-bottom: 6px; line-height: 1.35; }
    .webbox {
      display: grid;
      gap: 8px;
    }
    .webbox input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: var(--inputBg);
      color: var(--text);
    }
    .web-results {
      color: var(--muted);
      font-size: 13px;
      max-height: 170px;
      overflow: auto;
      border-top: 1px solid var(--line);
      padding-top: 8px;
      white-space: pre-wrap;
    }
    .draft-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .meta h2 { margin: 0 0 8px; font-size: 20px; }
    .paths {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }
    .note-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 150px;
      gap: 10px;
      align-items: start;
    }
    textarea {
      width: 100%;
      min-height: 84px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .tab.active {
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 700;
    }
    .summaries {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .summary {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .summary h3 { margin: 0 0 8px; font-size: 14px; }
    dl { margin: 0; }
    dt { font-weight: 700; margin-top: 8px; }
    dd { margin: 2px 0 0; color: var(--muted); overflow-wrap: anywhere; }
    .matrix-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: visible;
      margin-bottom: 14px;
    }
    .matrix-head {
      position: sticky;
      top: 0;
      z-index: 6;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      border-radius: 8px 8px 0 0;
      background: var(--panel);
    }
    .matrix-head h3 { margin: 0; font-size: 14px; }
    .matrix-tools {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .pane-sizer {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .pane-sizer input {
      width: 130px;
      min-height: 0;
      accent-color: var(--accent);
    }
    .matrix-hint {
      padding: 0 12px 10px;
      white-space: normal;
    }
    .matrix-rows {
      display: grid;
      gap: 8px;
      padding: 10px;
      max-height: 68vh;
      overflow: auto;
    }
    .matrix-row {
      display: grid;
      grid-template-columns: 28px minmax(120px, .55fr) minmax(180px, var(--cudaPane)) minmax(180px, var(--metalPane)) minmax(118px, .45fr);
      gap: 8px;
      align-items: stretch;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
    }
    .matrix-row.aligned-row {
      background: var(--alignedBg);
    }
    .matrix-row.staggered-row {
      background: var(--staggeredBg);
    }
    .matrix-row.unmatched-row {
      background: var(--unmatchedBg);
    }
    .matrix-row.generated-row {
      box-shadow: inset 3px 0 0 var(--accent);
      animation: generatedPulse 3s ease-out;
    }
    @keyframes generatedPulse {
      0% { background: var(--accentSoft); }
      100% { background: inherit; }
    }
    .row-cell {
      min-width: 0;
      display: grid;
      align-content: start;
      gap: 6px;
    }
    .row-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .row-kind-wrap {
      display: grid;
      gap: 6px;
    }
    .diff-cell {
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
      background: var(--panel);
    }
    .diff-cell.cuda-side {
      border-color: color-mix(in srgb, var(--danger) 22%, var(--line));
    }
    .diff-cell.metal-side {
      border-color: color-mix(in srgb, var(--success) 24%, var(--line));
    }
    .diff-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-height: 30px;
      padding: 5px 7px;
      border-bottom: 1px solid var(--line);
      background: var(--panelHead);
    }
    .diff-head .row-label {
      letter-spacing: 0;
      text-transform: none;
      font-size: 12px;
    }
    .diff-op {
      display: inline-grid;
      place-items: center;
      width: 20px;
      height: 20px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-weight: 700;
    }
    .cuda-side .diff-op {
      color: var(--danger);
      background: color-mix(in srgb, var(--danger) 12%, transparent);
    }
    .metal-side .diff-op {
      color: var(--success);
      background: color-mix(in srgb, var(--success) 14%, transparent);
    }
    .diff-cell .ref-chip {
      border-width: 0 0 1px;
      border-color: var(--line);
      border-radius: 0;
      min-height: 30px;
    }
    .inline-snippet {
      border-bottom: 1px solid var(--line);
      background: var(--code);
      display: none;
      min-width: 0;
    }
    .inline-snippet.open {
      display: block;
    }
    .snippet-scroll {
      max-height: 260px;
      overflow: auto;
      overscroll-behavior: contain;
    }
    .snippet-code {
      display: grid;
      grid-template-columns: max-content max-content;
      width: max-content;
      min-width: 100%;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      line-height: 1.5;
      white-space: pre;
    }
    .snippet-gutter {
      position: sticky;
      left: 0;
      z-index: 1;
      display: grid;
      grid-template-columns: 2.2em 4.5em;
      gap: 0;
      padding: 8px 8px 8px 10px;
      color: var(--codeLine);
      background: var(--code);
      border-right: 1px solid var(--codeHit);
      text-align: right;
      user-select: none;
    }
    .snippet-marker.hit {
      color: var(--accent);
      font-weight: 700;
    }
    .snippet-line {
      padding: 8px 12px;
      color: var(--codeText);
      min-width: 0;
    }
    .snippet-gutter.snippet-row-hit,
    .snippet-line.snippet-row-hit {
      background: var(--codeHit);
    }
    .inline-snippet pre {
      max-height: 260px;
      font-size: 11px;
      line-height: 1.5;
      padding: 8px 10px;
      overflow: auto;
      white-space: pre;
    }
    .inline-snippet .path {
      padding: 7px 10px 0;
      color: var(--codeLine);
      white-space: normal;
    }
    .ref-warning {
      margin: 8px 10px;
      padding: 7px 9px;
      border: 1px solid var(--warn);
      border-radius: 6px;
      color: var(--warn);
      background: color-mix(in srgb, var(--warn) 12%, transparent);
      font-size: 12px;
      white-space: normal;
    }
    .snippet-description {
      margin: 8px 10px;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      background: var(--panelSoft);
      font-size: 12px;
      line-height: 1.35;
      white-space: normal;
    }
    .snippet-description strong {
      color: var(--muted);
      margin-right: 4px;
    }
    .diff-cell textarea {
      border: 0;
      border-radius: 0;
      min-height: 86px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .cuda-side textarea {
      background: color-mix(in srgb, var(--danger) 8%, var(--inputBg));
    }
    .metal-side textarea {
      background: color-mix(in srgb, var(--success) 9%, var(--inputBg));
    }
    .relationship-detail {
      grid-column: 3 / 5;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: color-mix(in srgb, var(--panel) 80%, transparent);
      overflow: hidden;
    }
    .relationship-detail summary {
      cursor: pointer;
      padding: 7px 9px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .relationship-detail textarea {
      border-width: 1px 0 0;
      border-radius: 0;
      min-height: 68px;
      background: var(--relationBg);
    }
    .relationship-tools {
      display: grid;
      gap: 8px;
      border-top: 1px solid var(--line);
      padding: 8px;
      background: color-mix(in srgb, var(--panel) 70%, transparent);
    }
    .relationship-tools input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
    }
    .matrix-row textarea, .matrix-row input {
      width: 100%;
      min-height: 64px;
      border: 1px solid color-mix(in srgb, var(--line) 70%, transparent);
      border-radius: 6px;
      resize: vertical;
      background: color-mix(in srgb, var(--inputBg) 88%, transparent);
      padding: 7px;
      line-height: 1.35;
    }
    .matrix-row input {
      min-height: 32px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .ref-chip {
      display: block;
      width: 100%;
      min-height: 32px;
      border: 1px solid color-mix(in srgb, var(--refText) 20%, var(--line));
      border-radius: 6px;
      background: var(--refBg);
      color: var(--refText);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.35;
      padding: 6px 8px;
      text-align: left;
      overflow-wrap: anywhere;
    }
    .ref-chip.empty-ref {
      background: var(--blankBg);
      color: var(--emptyRefText);
    }
    .blank-cell,
    .blank-cell textarea {
      background: repeating-linear-gradient(
        -45deg,
        var(--blankBg),
        var(--blankBg) 8px,
        var(--blankStripe) 8px,
        var(--blankStripe) 9px
      );
    }
    .relation-cell textarea { background: var(--relationBg); }
    .row-actions {
      display: grid;
      gap: 6px;
    }
    .matrix-row select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px;
      background: var(--inputBg);
      color: var(--text);
      font-size: 12px;
    }
    .select-cell {
      display: grid;
      place-items: start center;
      padding-top: 24px;
    }
    .select-cell input {
      width: 18px;
      min-height: 18px;
    }
    .bulk-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .job-status {
      color: var(--muted);
      font-size: 13px;
    }
    .job-panel {
      display: grid;
      gap: 8px;
      margin-top: 10px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panelSoft);
    }
    .job-panel.hidden { display: none; }
    .job-meter {
      height: 8px;
      border-radius: 999px;
      background: var(--line);
      overflow: hidden;
    }
    .job-meter-fill {
      height: 100%;
      width: 0%;
      background: var(--accent);
      transition: width 200ms ease;
    }
    .job-detail-line {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .job-chunks {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }
    .chunk-dot {
      width: 18px;
      height: 18px;
      display: inline-grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 10px;
      line-height: 1;
    }
    .chunk-dot.done {
      border-color: var(--accent);
      background: var(--accentSoft);
      color: var(--accent);
    }
    .chunk-dot.running {
      border-color: var(--warn);
      color: var(--warn);
    }
    .chunk-dot.failed {
      border-color: var(--danger);
      color: var(--danger);
    }
    .new-row-note {
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .state-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panelSoft);
      color: var(--muted);
      font-size: 12px;
    }
    .state-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--muted);
    }
    .state-dot.running { background: var(--warn); }
    .state-dot.done { background: var(--success); }
    .state-dot.idle { background: var(--accent); }
    .snippet-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 14px;
      overflow: hidden;
    }
    .snippet-panel h3 {
      margin: 0;
      padding: 9px 11px;
      font-size: 13px;
      border-bottom: 1px solid var(--line);
    }
    .snippet-panel .path {
      padding: 8px 11px 0;
      white-space: normal;
    }
    .semantic-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .semantic-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .semantic-card h3 {
      margin: 0 0 8px;
      font-size: 14px;
    }
    .semantic-item {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      margin-top: 8px;
      font-size: 13px;
      line-height: 1.35;
    }
    .semantic-item .path {
      white-space: normal;
      margin-bottom: 4px;
    }
    .codegrid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .codepanel {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .codepanel h3 {
      margin: 0;
      padding: 9px 11px;
      font-size: 13px;
      border-bottom: 1px solid var(--line);
    }
    pre {
      margin: 0;
      background: var(--code);
      color: var(--codeText);
      padding: 12px;
      overflow: auto;
      max-height: 68vh;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      tab-size: 4;
    }
    .code-lines {
      counter-reset: code-line;
    }
    .code-line {
      display: block;
      min-height: 1.45em;
      white-space: pre;
    }
    .code-line::before {
      counter-increment: code-line;
      content: counter(code-line);
      display: inline-block;
      width: 4em;
      margin-right: 1em;
      color: var(--codeLine);
      text-align: right;
      user-select: none;
    }
    .single pre { max-height: 74vh; }
    .hidden { display: none; }
    .empty {
      color: var(--muted);
      padding: 40px;
      text-align: center;
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { max-height: 260px; border-right: 0; border-bottom: 1px solid var(--line); }
      .sidebar-toggle { top: 86px; left: 12px; }
      body.sidebar-collapsed .sidebar-toggle { left: 12px; }
      .summaries, .codegrid, .note-grid, .analysis, .metrics, .semantic-grid { grid-template-columns: 1fr; }
      main { max-height: none; }
      .workbench-head { position: static; }
      .matrix-row { grid-template-columns: 28px 1fr; }
      .row-cell { grid-column: 2; }
      .relationship-detail { grid-column: 2; }
      .select-cell { grid-row: 1 / span 5; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>vLLM-Hook Correspondence Review</h1>
      <div class="subtitle">Inspect candidate correspondences, divergences, and missing counterparts across CUDA and Metal.</div>
    </div>
    <div>
      <button id="themeToggle">Dark Mode</button>
      <button id="rescan">Rescan</button>
      <button id="regen">Regenerate Registry</button>
    </div>
  </header>
  <div class="app">
    <button id="sideToggle" class="sidebar-toggle" type="button" title="Hide queue" aria-label="Hide queue">&lsaquo;</button>
    <aside>
      <div class="filters">
        <input id="search" placeholder="Search pairs">
        <select id="statusFilter">
          <option value="all">All</option>
          <option value="pending">Pending</option>
          <option value="strong_candidate">Strong Candidate</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
        </select>
      </div>
      <div id="entryList"></div>
    </aside>
    <main id="detail"></main>
  </div>
  <script>
    let entries = [];
    let selectedId = null;
    let activeView = "matrix";
    let activeJobId = null;
    let currentPayload = null;
    let paneSplit = Number(localStorage.getItem("correspondencePaneSplit") || 50);

    const entryList = document.getElementById("entryList");
    const detail = document.getElementById("detail");
    const search = document.getElementById("search");
    const statusFilter = document.getElementById("statusFilter");
    const sideToggle = document.getElementById("sideToggle");
    const themeToggle = document.getElementById("themeToggle");
    const appShell = document.querySelector(".app");

    function applyTheme(theme) {
      document.body.dataset.theme = theme;
      localStorage.setItem("correspondenceTheme", theme);
      themeToggle.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
    }

    function applySidebar(collapsed) {
      document.body.classList.toggle("sidebar-collapsed", collapsed);
      appShell?.classList.toggle("sidebar-collapsed", collapsed);
      localStorage.setItem("correspondenceSidebarCollapsed", collapsed ? "1" : "0");
      sideToggle.innerHTML = collapsed ? "&rsaquo;" : "&lsaquo;";
      sideToggle.title = collapsed ? "Show queue" : "Hide queue";
      sideToggle.setAttribute("aria-label", collapsed ? "Show queue" : "Hide queue");
    }

    function toggleSidebar() {
      applySidebar(!document.body.classList.contains("sidebar-collapsed"));
    }

    function applyPaneSplit(value) {
      paneSplit = Math.max(25, Math.min(75, Number(value) || 50));
      const cudaSize = Math.max(0.5, paneSplit / 50);
      const metalSize = Math.max(0.5, (100 - paneSplit) / 50);
      document.documentElement.style.setProperty("--cudaPane", `${cudaSize}fr`);
      document.documentElement.style.setProperty("--metalPane", `${metalSize}fr`);
      localStorage.setItem("correspondencePaneSplit", String(paneSplit));
      const slider = document.getElementById("paneSplit");
      if (slider) slider.value = String(paneSplit);
    }

    const savedTheme = localStorage.getItem("correspondenceTheme");
    const preferredTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    applyTheme(savedTheme || preferredTheme);
    applySidebar(localStorage.getItem("correspondenceSidebarCollapsed") === "1");
    applyPaneSplit(paneSplit);
    themeToggle.addEventListener("click", () => {
      applyTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
    });
    sideToggle.addEventListener("click", toggleSidebar);

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function conciseList(values) {
      if (!values || values.length === 0) return "None";
      return values.slice(0, 18).join(", ");
    }

    function compactRef(ref) {
      if (!ref) return "";
      const match = String(ref).match(/([^/]+\\.py(?::[0-9]+(?:-[0-9]+)?)?)$/);
      return match ? match[1] : ref;
    }

    function compactPath(path) {
      const parts = String(path || "").split("/");
      return parts.slice(-3).join("/");
    }

    function renderRefInput(className, value) {
      const label = compactRef(value);
      return `
        <input type="hidden" class="${className}" value="${escapeHtml(value || "")}">
        <button type="button" class="ref-chip ${label ? "" : "empty-ref"}" title="${escapeHtml(value || "No reference")}">
          ${escapeHtml(label || "blank")}
        </button>
      `;
    }

    function renderCodeLines(content) {
      return String(content || "").split("\\n").map((line) => (
        `<span class="code-line">${escapeHtml(line) || " "}</span>`
      )).join("");
    }

    function refRange(ref) {
      const match = String(ref || "").match(/:([0-9]+)(?:-([0-9]+))?/);
      if (!match) return { start: 1, end: 1 };
      const start = Number(match[1]);
      const end = Number(match[2] || match[1]);
      return { start: Math.min(start, end), end: Math.max(start, end) };
    }

    function rangeHasCode(content, range) {
      const lines = String(content || "").split("\\n");
      for (let number = range.start; number <= range.end; number += 1) {
        const text = lines[number - 1] || "";
        if (text.trim()) return true;
      }
      return false;
    }

    function renderRefWarning(content, range) {
      return rangeHasCode(content, range)
        ? ""
        : `<div class="ref-warning">Reference points at empty lines. Clear it or choose a better code range.</div>`;
    }

    function renderNumberedSnippet(content, range, radius = 8) {
      const lines = String(content || "").split("\\n");
      const start = Math.max(1, range.start - radius);
      const end = Math.min(lines.length, range.end + radius);
      const rendered = [];
      for (let number = start; number <= end; number += 1) {
        const marker = number >= range.start && number <= range.end ? ">>" : "  ";
        rendered.push(`${marker} ${String(number).padStart(5, " ")}  ${lines[number - 1] || ""}`);
      }
      return escapeHtml(rendered.join("\\n"));
    }

    function renderInlineSnippet(content, range, radius = 8) {
      const lines = String(content || "").split("\\n");
      const start = Math.max(1, range.start - radius);
      const end = Math.min(lines.length, range.end + radius);
      const rows = [];
      for (let number = start; number <= end; number += 1) {
        const isHit = number >= range.start && number <= range.end;
        rows.push(`
          <div class="snippet-gutter ${isHit ? "snippet-row-hit" : ""}">
            <span class="snippet-marker ${isHit ? "hit" : ""}">${isHit ? "&gt;&gt;" : ""}</span>
            <span>${escapeHtml(String(number))}</span>
          </div>
          <div class="snippet-line ${isHit ? "snippet-row-hit" : ""}">${escapeHtml(lines[number - 1] || " ")}</div>
        `);
      }
      return `<div class="snippet-scroll"><div class="snippet-code">${rows.join("")}</div></div>`;
    }

    function platformFromRef(ref) {
      return String(ref || "").includes("/metal/") || String(ref || "").includes("_metal.py") ? "metal" : "cuda";
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }
      return response.json();
    }

    async function loadEntries(keepSelection = true) {
      entries = await api("/api/entries");
      if (!keepSelection || !entries.some((entry) => entry.id === selectedId)) {
        selectedId = entries[0]?.id ?? null;
      }
      renderList();
      await renderDetail();
    }

    function visibleEntries() {
      const q = search.value.trim().toLowerCase();
      const status = statusFilter.value;
      return entries.filter((entry) => {
        const matchesStatus = status === "all" || entry.status === status;
        const blob = [entry.id, entry.logical_component, entry.cuda_path, entry.metal_path, entry.divergence_note].join(" ").toLowerCase();
        return matchesStatus && (!q || blob.includes(q));
      });
    }

    function renderList() {
      const visible = visibleEntries();
      renderMetrics();
      if (!visible.length) {
        entryList.innerHTML = '<div class="empty">No matching pairs.</div>';
        return;
      }
      entryList.innerHTML = visible.map((entry) => `
        <button class="entry ${entry.id === selectedId ? "active" : ""}" data-id="${escapeHtml(entry.id)}">
          <span class="entry-title">
            <span>${escapeHtml(entry.logical_component)}</span>
            <span class="pill ${escapeHtml(entry.status)}">${escapeHtml(entry.status)}</span>
          </span>
          <span class="path" title="${escapeHtml(entry.cuda_path)}">${escapeHtml(compactPath(entry.cuda_path))}</span>
          <span class="path" title="${escapeHtml(entry.metal_path)}">${escapeHtml(compactPath(entry.metal_path))}</span>
        </button>
      `).join("");
      entryList.querySelectorAll(".entry").forEach((button) => {
        button.addEventListener("click", async () => {
          selectedId = button.dataset.id;
          renderList();
          await renderDetail();
        });
      });
    }

    function renderMetrics() {
      const counts = entries.reduce((acc, entry) => {
        acc[entry.status] = (acc[entry.status] || 0) + 1;
        return acc;
      }, {});
      const existing = document.getElementById("metrics");
      if (!existing) return;
      existing.innerHTML = `
        <div class="metric"><strong>${entries.length}</strong><span>Total pairs</span></div>
        <div class="metric"><strong>${counts.pending || 0}</strong><span>Pending review</span></div>
        <div class="metric"><strong>${counts.strong_candidate || 0}</strong><span>Strong candidates</span></div>
        <div class="metric"><strong>${counts.approved || 0}</strong><span>Approved</span></div>
        <div class="metric"><strong>${counts.rejected || 0}</strong><span>Rejected</span></div>
      `;
    }

    function renderSummary(title, summary) {
      return `
        <div class="summary">
          <h3>${escapeHtml(title)}</h3>
          <dl>
            <dt>Lines</dt><dd>${escapeHtml(summary.lines)}</dd>
            <dt>Classes</dt><dd>${escapeHtml(conciseList(summary.classes))}</dd>
            <dt>Functions</dt><dd>${escapeHtml(conciseList(summary.functions))}</dd>
            <dt>Imports</dt><dd>${escapeHtml(conciseList(summary.imports))}</dd>
          </dl>
        </div>
      `;
    }

    async function renderDetail() {
      if (!selectedId) {
        detail.innerHTML = '<div class="empty">Run the scanner to create correspondence candidates.</div>';
        return;
      }
      const payload = await api(`/api/entry?id=${encodeURIComponent(selectedId)}`);
      currentPayload = payload;
      detail.innerHTML = `
        <div class="metrics" id="metrics"></div>
        <section class="workbench-head">
          <div class="workbench-title">
            <h2>${escapeHtml(payload.logical_component)}</h2>
            <div class="workbench-meta">
              <span class="pill ${escapeHtml(payload.status)}">${escapeHtml(payload.status)}</span>
              <span class="pill">${escapeHtml(payload.confidence)} confidence</span>
              ${payload.generation_started_at ? `<span class="pill">generated ${escapeHtml(formatTimestamp(payload.generation_started_at))}</span>` : ""}
            </div>
          </div>
          <div class="paths">
            <div><strong>CUDA</strong> <code>${escapeHtml(payload.cuda_path)}</code></div>
            <div><strong>Metal</strong> <code>${escapeHtml(payload.metal_path)}</code></div>
          </div>
          <div class="state-strip">
            <span class="state-dot ${payload.generation_completed_at ? "done" : (payload.generation_started_at ? "running" : "idle")}"></span>
            <span>${payload.generation_started_at ? (payload.generation_completed_at ? "Generation complete" : "Generation running") : "Ready for review"}</span>
            <span>${escapeHtml((payload.comparison_rows || []).length)} matrix rows</span>
            <span>Refs open code excerpts in place</span>
          </div>
          <div class="workbench-actions">
            <div class="tabs">
              <button class="tab ${activeView === "matrix" ? "active" : ""}" data-view="matrix">Matrix</button>
              <button class="tab ${activeView === "semantic" ? "active" : ""}" data-view="semantic">Semantic</button>
              <button class="tab ${activeView === "side" ? "active" : ""}" data-view="side">Code</button>
            </div>
            <div class="actions">
              <button class="ai" id="analyze">Generate Matrix</button>
              <button id="strong">Strong</button>
              <button class="primary" id="approve">Approve</button>
              <button id="save">Save</button>
              <button id="addRow">Add Row</button>
              <button class="danger" id="reject">Reject</button>
            </div>
          </div>
          <div class="job-panel hidden" id="jobPanel">
            <div class="job-detail-line">
              <span class="job-status" id="jobStatus"></span>
              <span id="jobRowsAdded"></span>
            </div>
            <div class="job-detail-line">
              <span id="jobStartedAt"></span>
              <span id="jobCompletedAt"></span>
            </div>
            <div class="job-meter"><div class="job-meter-fill" id="jobMeterFill"></div></div>
            <div class="job-chunks" id="jobChunks"></div>
          </div>
        </section>
        <section class="meta">
          <div class="note-grid">
            <textarea id="note">${escapeHtml(payload.divergence_note)}</textarea>
            <select id="confidence">
              ${["high", "medium", "low"].map((value) => `<option value="${value}" ${payload.confidence === value ? "selected" : ""}>${value}</option>`).join("")}
            </select>
          </div>
        </section>
        <div id="matrixView" class="${activeView === "matrix" ? "" : "hidden"}">
          <div id="snippetPanel"></div>
          ${renderMatrix(payload.comparison_rows || [])}
        </div>
        <div class="summaries">
          ${renderSummary("CUDA Summary", payload.summary.cuda)}
          ${renderSummary("Metal Summary", payload.summary.metal)}
        </div>
        <div id="sideView" class="${activeView === "side" ? "" : "hidden"}">
          <div class="codegrid">
            <div class="codepanel"><h3>CUDA</h3><pre class="code-lines">${renderCodeLines(payload.cuda_content)}</pre></div>
            <div class="codepanel"><h3>Metal</h3><pre class="code-lines">${renderCodeLines(payload.metal_content)}</pre></div>
          </div>
        </div>
        <div id="semanticView" class="${activeView === "semantic" ? "" : "hidden"}">
          ${renderSemanticDiff(payload.comparison_rows || [])}
        </div>
      `;
      renderMetrics();
      detail.querySelectorAll(".tab").forEach((tab) => {
        tab.addEventListener("click", () => {
          activeView = tab.dataset.view;
          renderDetail();
        });
      });
      document.getElementById("save").addEventListener("click", () => updateEntry("pending"));
      document.getElementById("strong").addEventListener("click", () => updateEntry("strong_candidate"));
      document.getElementById("approve").addEventListener("click", () => updateEntry("approved"));
      document.getElementById("reject").addEventListener("click", () => updateEntry("rejected"));
      document.getElementById("addRow").addEventListener("click", addMatrixRow);
      document.getElementById("analyze").addEventListener("click", analyzeEntry);
      attachMatrixControls();
      attachRowButtons();
      detail.querySelectorAll(".ref-chip").forEach((button) => {
        button.addEventListener("click", () => jumpToRef(button.title, button));
      });
    }

    function renderMatrix(rows) {
      if (!rows.length) {
        rows = [{ kind: "observation", cuda: "", metal: "", relation: "No matrix rows yet.", confidence: "" }];
      }
      return `
        <section class="matrix-panel">
          <div class="matrix-head">
            <h3>Correspondence Matrix</h3>
            <div class="matrix-tools">
              <label class="pane-sizer">
                CUDA
                <input id="paneSplit" type="range" min="25" max="75" value="${paneSplit}" aria-label="Adjust CUDA and Metal pane widths">
                Metal
              </label>
              <div class="bulk-actions">
                <button id="selectAllRows">Select All</button>
                <button id="rowPending">Rows Pending</button>
                <button id="rowStrong">Rows Strong</button>
                <button id="rowApprove">Rows Approve</button>
                <button id="rowReject">Rows Reject</button>
                <button class="danger" id="rowRemove">Remove Selected</button>
              </div>
            </div>
          </div>
          <div class="path matrix-hint">Both sides filled means exact, strong, or mild similarity. One-sided rows capture weak, partial, divergent, or unmatched behavior.</div>
          <div class="matrix-rows" id="matrixBody">
            ${rows.map((row) => renderMatrixRow(row)).join("")}
          </div>
        </section>
      `;
    }

    function renderMatrixRow(row, generated = false) {
      const hasCuda = Boolean((row.cuda || "").trim());
      const hasMetal = Boolean((row.metal || "").trim());
      const rowClass = hasCuda && hasMetal ? "aligned-row" : (hasCuda || hasMetal ? "staggered-row" : "unmatched-row");
      return `
        <article class="matrix-row ${rowClass} ${generated ? "generated-row" : ""}">
          <div class="select-cell"><input type="checkbox" class="row-selected" aria-label="Select matrix row"></div>
          <div class="row-cell row-kind-wrap">
            <span class="row-label">Kind</span>
            <input class="row-kind" value="${escapeHtml(row.kind || "observation")}">
            <span class="row-label">Status</span>
            <select class="row-status">
              ${["pending", "strong_candidate", "approved", "rejected"].map((value) => `<option value="${value}" ${(row.status || "pending") === value ? "selected" : ""}>${value}</option>`).join("")}
            </select>
          </div>
          <div class="row-cell diff-cell cuda-side">
            <div class="diff-head">
              <span class="row-label">CUDA</span>
              <span class="diff-op">-</span>
            </div>
            ${renderRefInput("row-cuda-ref", row.cuda_ref || "")}
            <div class="inline-snippet" data-platform="cuda"></div>
            <textarea class="row-cuda ${row.cuda ? "" : "blank-cell"}">${escapeHtml(row.cuda || "")}</textarea>
          </div>
          <div class="row-cell diff-cell metal-side">
            <div class="diff-head">
              <span class="row-label">Metal</span>
              <span class="diff-op">+</span>
            </div>
            ${renderRefInput("row-metal-ref", row.metal_ref || "")}
            <div class="inline-snippet" data-platform="metal"></div>
            <textarea class="row-metal ${row.metal ? "" : "blank-cell"}">${escapeHtml(row.metal || "")}</textarea>
          </div>
          <div class="row-cell row-actions">
            <span class="row-label">Row</span>
            <input type="hidden" class="row-generated-at" value="${escapeHtml(row.generated_at || "")}">
            <span class="path" title="${escapeHtml(row.generated_at || "Existing row")}">${escapeHtml(row.generated_at ? formatTimestamp(row.generated_at) : "existing")}</span>
            <button type="button" class="remove-row">Remove</button>
          </div>
          <details class="relationship-detail">
            <summary>Relationship note</summary>
            <textarea class="row-relation">${escapeHtml(row.relation || "")}</textarea>
            <div class="relationship-tools">
              <input class="row-context-query" value="${escapeHtml(defaultRowContextQuery(row))}" placeholder="Optional context query">
              <div class="draft-actions">
                <button type="button" class="row-context-fetch">Fetch Context</button>
                <button type="button" class="ai row-relation-regenerate">Regenerate</button>
              </div>
              <textarea class="row-context" placeholder="Reviewer guidance, edit request, pasted notes, or optional web snippets for this relationship note"></textarea>
              <div class="web-results row-context-results">No context loaded.</div>
            </div>
          </details>
        </article>
      `;
    }

    function renderSemanticDiff(rows) {
      const paired = [];
      const cudaOnly = [];
      const metalOnly = [];
      for (const row of rows) {
        const hasCuda = Boolean((row.cuda || "").trim());
        const hasMetal = Boolean((row.metal || "").trim());
        if (hasCuda && hasMetal) paired.push(row);
        else if (hasCuda) cudaOnly.push(row);
        else if (hasMetal) metalOnly.push(row);
      }
      return `
        <div class="semantic-grid">
          ${renderSemanticCard("Aligned Exact / Strong / Mild", paired, "Exact, strong, and mild similarities should appear here.")}
          ${renderSemanticCard("Staggered CUDA", cudaOnly, "CUDA-side behavior that is weakly related, divergent, or unmatched.")}
          ${renderSemanticCard("Staggered Metal", metalOnly, "Metal-side behavior that is weakly related, divergent, or unmatched.")}
        </div>
      `;
    }

    function renderSemanticCard(title, rows, emptyText) {
      const items = rows.length ? rows.map((row) => `
        <div class="semantic-item">
          <div><strong>${escapeHtml(row.kind || "observation")}</strong> <span class="pill ${escapeHtml(row.status || "pending")}">${escapeHtml(row.status || "pending")}</span></div>
          ${row.cuda_ref ? `<div class="path">CUDA ${escapeHtml(compactRef(row.cuda_ref))}</div>` : ""}
          ${row.cuda ? `<div>${escapeHtml(row.cuda)}</div>` : ""}
          ${row.metal_ref ? `<div class="path">Metal ${escapeHtml(compactRef(row.metal_ref))}</div>` : ""}
          ${row.metal ? `<div>${escapeHtml(row.metal)}</div>` : ""}
          <div class="path">${escapeHtml(row.relation || "")}</div>
        </div>
      `).join("") : `<p class="path">${escapeHtml(emptyText)}</p>`;
      return `<section class="semantic-card"><h3>${escapeHtml(title)}</h3>${items}</section>`;
    }

    function collectMatrixRows() {
      return Array.from(document.querySelectorAll(".matrix-row")).map((row) => ({
        kind: row.querySelector(".row-kind").value.trim(),
        cuda_ref: row.querySelector(".row-cuda-ref").value.trim(),
        cuda: row.querySelector(".row-cuda").value.trim(),
        metal_ref: row.querySelector(".row-metal-ref").value.trim(),
        metal: row.querySelector(".row-metal").value.trim(),
        relation: row.querySelector(".row-relation").value.trim(),
        generated_at: row.querySelector(".row-generated-at")?.value.trim() || "",
        status: row.querySelector(".row-status").value,
        confidence: document.getElementById("confidence").value,
      })).filter((row) => row.kind || row.cuda_ref || row.cuda || row.metal_ref || row.metal || row.relation);
    }

    function matrixRowKey(row) {
      return [
        row.kind || "",
        row.cuda_ref || "",
        row.cuda || "",
        row.metal_ref || "",
        row.metal || "",
      ].map((value) => String(value).trim()).join("\\u241f");
    }

    function appendGeneratedRows(rows) {
      if (!Array.isArray(rows) || !rows.length) return;
      const body = document.getElementById("matrixBody");
      if (!body) return;
      const existingRows = collectMatrixRows();
      const hasOnlyPlaceholder = existingRows.length === 1
        && existingRows[0].relation === "No matrix rows yet."
        && !existingRows[0].cuda
        && !existingRows[0].metal;
      if (hasOnlyPlaceholder) {
        body.innerHTML = "";
        existingRows.length = 0;
      }
      const seen = new Set(existingRows.map(matrixRowKey));
      const freshRows = rows.filter((row) => {
        const key = matrixRowKey(row);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      if (!freshRows.length) return;
      body.insertAdjacentHTML("beforeend", freshRows.map((row) => renderMatrixRow(row, true)).join(""));
      attachRowButtons();
      detail.querySelectorAll(".ref-chip").forEach((button) => {
        button.onclick = () => jumpToRef(button.title, button);
      });
      const rowsAdded = document.getElementById("jobRowsAdded");
      if (rowsAdded) {
        rowsAdded.className = "new-row-note";
        rowsAdded.textContent = `${freshRows.length} new row${freshRows.length === 1 ? "" : "s"} added`;
      }
    }

    function formatJobStatus(job) {
      const parts = [job.message || job.status || ""];
      if (job.total_chunks) {
        parts.push(`${job.completed_chunks || 0}/${job.total_chunks} chunks`);
      }
      if (job.failed_chunks?.length) {
        parts.push(`${job.failed_chunks.length} failed`);
      }
      return parts.filter(Boolean).join(" · ");
    }

    function formatTimestamp(value) {
      if (!value) return "";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      return date.toLocaleString();
    }

    function renderJobChunks(job) {
      const total = Number(job.total_chunks || 0);
      if (!total) return "";
      const completed = Number(job.completed_chunks || 0);
      const failed = new Set((job.failed_chunks || []).map((item) => Number(item.chunk)));
      const running = job.status === "running" ? completed + 1 : 0;
      const dots = [];
      for (let index = 1; index <= total; index += 1) {
        let state = "";
        if (failed.has(index)) state = "failed";
        else if (index <= completed) state = "done";
        else if (index === running) state = "running";
        dots.push(`<span class="chunk-dot ${state}" title="Chunk ${index}: ${state || "queued"}">${index}</span>`);
      }
      return dots.join("");
    }

    function updateJobPanel(job) {
      const panel = document.getElementById("jobPanel");
      const status = document.getElementById("jobStatus");
      const meter = document.getElementById("jobMeterFill");
      const chunks = document.getElementById("jobChunks");
      const startedAt = document.getElementById("jobStartedAt");
      const completedAt = document.getElementById("jobCompletedAt");
      if (!panel || !status || !meter || !chunks) return;
      panel.classList.remove("hidden");
      status.textContent = formatJobStatus(job);
      const progress = Math.max(0, Math.min(1, Number(job.progress || 0)));
      meter.style.width = `${Math.round(progress * 100)}%`;
      chunks.innerHTML = renderJobChunks(job);
      if (startedAt) startedAt.textContent = job.generation_started_at ? `Started ${formatTimestamp(job.generation_started_at)}` : "";
      if (completedAt) completedAt.textContent = job.generation_completed_at ? `Finished ${formatTimestamp(job.generation_completed_at)}` : "";
    }

    function selectedRows() {
      return Array.from(document.querySelectorAll(".matrix-row")).filter((row) => row.querySelector(".row-selected").checked);
    }

    function setSelectedRowStatus(status) {
      selectedRows().forEach((row) => {
        row.querySelector(".row-status").value = status;
      });
    }

    function removeSelectedRows() {
      selectedRows().forEach((row) => row.remove());
    }

    function attachMatrixControls() {
      const selectAll = document.getElementById("selectAllRows");
      if (!selectAll) return;
      const paneSlider = document.getElementById("paneSplit");
      if (paneSlider) {
        paneSlider.value = String(paneSplit);
        paneSlider.addEventListener("input", () => applyPaneSplit(paneSlider.value));
      }
      selectAll.addEventListener("click", () => {
        const rows = Array.from(document.querySelectorAll(".row-selected"));
        const shouldCheck = rows.some((input) => !input.checked);
        rows.forEach((input) => { input.checked = shouldCheck; });
      });
      document.getElementById("rowPending").addEventListener("click", () => setSelectedRowStatus("pending"));
      document.getElementById("rowStrong").addEventListener("click", () => setSelectedRowStatus("strong_candidate"));
      document.getElementById("rowApprove").addEventListener("click", () => setSelectedRowStatus("approved"));
      document.getElementById("rowReject").addEventListener("click", () => setSelectedRowStatus("rejected"));
      document.getElementById("rowRemove").addEventListener("click", removeSelectedRows);
    }

    function attachRowButtons() {
      document.querySelectorAll(".remove-row").forEach((button) => {
        button.onclick = () => button.closest(".matrix-row").remove();
      });
      document.querySelectorAll(".row-context-fetch").forEach((button) => {
        button.onclick = () => fetchRowContext(button.closest(".matrix-row"));
      });
      document.querySelectorAll(".row-relation-regenerate").forEach((button) => {
        button.onclick = () => regenerateRelationshipNote(button.closest(".matrix-row"));
      });
    }

    function addMatrixRow() {
      if (activeView !== "matrix") {
        activeView = "matrix";
        renderDetail();
        return;
      }
      const body = document.getElementById("matrixBody");
      body.insertAdjacentHTML("beforeend", renderMatrixRow({ kind: "observation", cuda: "", metal: "", relation: "", status: "pending" }));
      attachRowButtons();
    }

    function jumpToRef(ref, sourceButton = null) {
      if (!ref || ref === "No reference") return;
      const platform = platformFromRef(ref);
      const range = refRange(ref);
      const content = platform === "metal" ? currentPayload?.metal_content : currentPayload?.cuda_content;
      const diffCell = sourceButton?.closest(".diff-cell");
      const rowElement = sourceButton?.closest(".matrix-row");
      const describedBehavior = platform === "metal"
        ? rowElement?.querySelector(".row-metal")?.value
        : rowElement?.querySelector(".row-cuda")?.value;
      const descriptionBlock = describedBehavior
        ? `<div class="snippet-description"><strong>Row description:</strong>${escapeHtml(describedBehavior)}</div>`
        : "";
      const inlinePanel = diffCell?.querySelector(".inline-snippet");
      if (inlinePanel) {
        const isOpen = inlinePanel.classList.contains("open") && inlinePanel.dataset.ref === ref;
        if (isOpen) {
          inlinePanel.classList.remove("open");
          inlinePanel.innerHTML = "";
          inlinePanel.dataset.ref = "";
          return;
        }
        inlinePanel.dataset.ref = ref;
        inlinePanel.classList.add("open");
        inlinePanel.innerHTML = `
          <div class="path">${escapeHtml(ref)}</div>
          ${descriptionBlock}
          ${renderRefWarning(content, range)}
          ${renderInlineSnippet(content, range)}
        `;
        inlinePanel.scrollIntoView({ block: "nearest" });
        return;
      }
      const panel = document.getElementById("snippetPanel");
      if (!panel) return;
      panel.innerHTML = `
        <section class="snippet-panel">
          <h3>${platform === "metal" ? "Metal" : "CUDA"} Code Excerpt</h3>
          <div class="path">${escapeHtml(ref)}</div>
          ${descriptionBlock}
          ${renderRefWarning(content, range)}
          <pre>${renderNumberedSnippet(content, range)}</pre>
        </section>
      `;
      panel.scrollIntoView({ block: "nearest" });
    }

    function defaultRowContextQuery(row) {
      return [
        currentPayload?.logical_component || "",
        row.kind || "",
        row.cuda_ref || row.cuda || "",
        row.metal_ref || row.metal || "",
        "vLLM Hook CUDA Metal parity",
      ].filter(Boolean).join(" ");
    }

    function collectRow(rowElement) {
      return {
        kind: rowElement.querySelector(".row-kind").value.trim(),
        cuda_ref: rowElement.querySelector(".row-cuda-ref").value.trim(),
        cuda: rowElement.querySelector(".row-cuda").value.trim(),
        metal_ref: rowElement.querySelector(".row-metal-ref").value.trim(),
        metal: rowElement.querySelector(".row-metal").value.trim(),
        relation: rowElement.querySelector(".row-relation").value.trim(),
        generated_at: rowElement.querySelector(".row-generated-at")?.value.trim() || "",
        status: rowElement.querySelector(".row-status").value,
        confidence: document.getElementById("confidence").value,
      };
    }

    async function updateEntry(status) {
      const note = document.getElementById("note").value;
      const confidence = document.getElementById("confidence").value;
      await api("/api/entry", {
        method: "POST",
        body: JSON.stringify({
          id: selectedId,
          status,
          divergence_note: note,
          confidence,
          comparison_rows: collectMatrixRows(),
        }),
      });
      await loadEntries(true);
    }

    async function analyzeEntry() {
      const button = document.getElementById("analyze");
      const status = document.getElementById("jobStatus");
      const panel = document.getElementById("jobPanel");
      const rowsAdded = document.getElementById("jobRowsAdded");
      button.disabled = true;
      button.textContent = "Generating...";
      if (panel) panel.classList.remove("hidden");
      if (status) status.textContent = "Queued candidate generation...";
      if (rowsAdded) rowsAdded.textContent = "";
      try {
        const webContext = document.getElementById("webContext")?.value || "";
        const payload = await api("/api/analyze", {
          method: "POST",
          body: JSON.stringify({ id: selectedId, web_context: webContext }),
        });
        activeJobId = payload.job_id;
        pollJob(payload.job_id, selectedId);
      } catch (error) {
        alert(error.message);
        button.disabled = false;
        button.textContent = "Generate Matrix";
        status.textContent = "";
      }
    }

    async function pollJob(jobId, entryId) {
      const status = document.getElementById("jobStatus");
      const button = document.getElementById("analyze");
      try {
        const job = await api(`/api/job?id=${encodeURIComponent(jobId)}`);
        updateJobPanel(job);
        if (selectedId === entryId) {
          appendGeneratedRows(job.partial_rows || []);
        }
        if (job.status === "done" || job.status === "partial") {
          if (selectedId === entryId) {
            const note = document.getElementById("note");
            const confidence = document.getElementById("confidence");
            const currentRows = collectMatrixRows();
            await loadEntries(true);
            if (note && document.getElementById("note")) document.getElementById("note").value = note.value;
            if (confidence && document.getElementById("confidence")) document.getElementById("confidence").value = confidence.value;
            appendGeneratedRows(currentRows);
          }
          if (button) {
            button.disabled = false;
            button.textContent = "Generate Matrix";
          }
          return;
        }
        if (job.status === "error") {
          if (button) {
            button.disabled = false;
            button.textContent = "Generate Matrix";
          }
          if (status) status.textContent = `Generation failed: ${job.message}`;
          return;
        }
        setTimeout(() => pollJob(jobId, entryId), 1500);
      } catch (error) {
        if (button) {
          button.disabled = false;
          button.textContent = "Generate Matrix";
        }
        if (status) status.textContent = error.message;
      }
    }

    async function fetchRowContext(rowElement) {
      const results = rowElement.querySelector(".row-context-results");
      const context = rowElement.querySelector(".row-context");
      const query = rowElement.querySelector(".row-context-query").value;
      results.textContent = "Searching...";
      try {
        const payload = await api("/api/context", {
          method: "POST",
          body: JSON.stringify({ id: selectedId, query, row: collectRow(rowElement) }),
        });
        const text = payload.context || "No results found.";
        results.textContent = payload.message || text;
        context.value = text;
      } catch (error) {
        results.textContent = error.message;
      }
    }

    async function regenerateRelationshipNote(rowElement) {
      const button = rowElement.querySelector(".row-relation-regenerate");
      const results = rowElement.querySelector(".row-context-results");
      const relation = rowElement.querySelector(".row-relation");
      button.disabled = true;
      button.textContent = "Regenerating...";
      results.textContent = "Regenerating this relationship note...";
      try {
        const payload = await api("/api/explanation", {
          method: "POST",
          body: JSON.stringify({
            id: selectedId,
            row: collectRow(rowElement),
            web_context: rowElement.querySelector(".row-context").value,
          }),
        });
        if (payload.explanation.relation) {
          const changed = payload.explanation.relation !== relation.value;
          relation.value = payload.explanation.relation;
          results.textContent = changed
            ? "Relationship note regenerated. Save to persist it."
            : "Regeneration returned the same relationship note.";
        } else {
          results.textContent = "Regeneration returned no relationship note.";
        }
      } catch (error) {
        results.textContent = error.message;
      } finally {
        button.disabled = false;
        button.textContent = "Regenerate";
      }
    }

    search.addEventListener("input", renderList);
    statusFilter.addEventListener("change", renderList);
    document.getElementById("regen").addEventListener("click", async () => {
      await api("/api/registry", { method: "POST", body: "{}" });
      await loadEntries(true);
    });
    document.getElementById("rescan").addEventListener("click", async () => {
      await api("/api/rescan", { method: "POST", body: "{}" });
      await loadEntries(false);
    });

    loadEntries(false).catch((error) => {
      detail.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
    });
  </script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "CorrespondenceReview/0.1"

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}", file=sys.stderr)

    def send_body(self, body: str | bytes, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/plain") -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_body(json.dumps(payload), status, "application/json")

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_body(app_html(), content_type="text/html")
            return
        if parsed.path == "/api/entries":
            self.send_json(load_entries())
            return
        if parsed.path == "/api/entry":
            entry_id = parse_qs(parsed.query).get("id", [""])[0]
            entry = find_entry(load_entries(), entry_id)
            if entry is None:
                self.send_json({"error": "entry not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(entry_payload(entry))
            return
        if parsed.path == "/api/job":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            job = get_job(job_id)
            if job is None:
                self.send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(job)
            return
        path = (PROJECT_ROOT / parsed.path.lstrip("/")).resolve()
        if path.is_file() and is_relative_to(path, PROJECT_ROOT):
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_body(path.read_bytes(), content_type=content_type)
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = app_html().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/entry":
            data = self.read_json()
            entries = load_entries()
            entry = find_entry(entries, data.get("id", ""))
            if entry is None:
                self.send_json({"error": "entry not found"}, HTTPStatus.NOT_FOUND)
                return
            for key in ("status", "divergence_note", "confidence", "comparison_rows"):
                if key in data:
                    entry[key] = data[key]
            save_entries(entries)
            render_registry(entries)
            self.send_json({"ok": True, "entry": entry})
            return
        if parsed.path == "/api/analyze":
            data = self.read_json()
            entries = load_entries()
            entry_id = data.get("id", "")
            entry = find_entry(entries, entry_id)
            if entry is None:
                self.send_json({"error": "entry not found"}, HTTPStatus.NOT_FOUND)
                return
            job_id = uuid.uuid4().hex
            set_job(job_id, status="queued", message="Queued", entry_id=entry_id, created_at=time.time())
            thread = threading.Thread(
                target=analyze_entry_job,
                args=(job_id, str(entry_id), str(data.get("web_context") or "")),
                daemon=True,
            )
            thread.start()
            self.send_json({"ok": True, "job_id": job_id}, HTTPStatus.ACCEPTED)
            return
        if parsed.path == "/api/web":
            data = self.read_json()
            query = str(data.get("query") or "").strip()
            if not query:
                self.send_json({"error": "query is required"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                results = fetch_web_context(query)
            except Exception as exc:
                self.send_json(empty_web_context(query, exc))
                return
            self.send_json({"ok": True, "results": results, "context": web_results_to_context(results)})
            return
        if parsed.path == "/api/context":
            data = self.read_json()
            entries = load_entries()
            entry_id = str(data.get("id") or "")
            entry = find_entry(entries, entry_id)
            if entry is None:
                self.send_json({"error": "entry not found"}, HTTPStatus.NOT_FOUND)
                return
            row = data.get("row") if isinstance(data.get("row"), dict) else {}
            query = str(data.get("query") or "").strip() or (
                row_context_query(entry, row) if row else candidate_context_query(entry)
            )
            try:
                results = fetch_web_context(query)
            except Exception as exc:
                self.send_json(empty_web_context(query, exc))
                return
            self.send_json(
                {
                    "ok": True,
                    "query": query,
                    "provider": "duckduckgo_html",
                    "results": results,
                    "context": web_results_to_context(results),
                }
            )
            return
        if parsed.path == "/api/explanation":
            data = self.read_json()
            entry_id = str(data.get("id") or "")
            if not entry_id:
                self.send_json({"error": "id is required"}, HTTPStatus.BAD_REQUEST)
                return
            row = data.get("row")
            if not isinstance(row, dict):
                self.send_json({"error": "row is required"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                explanation = regenerate_explanation(entry_id, row, str(data.get("web_context") or ""))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:
                self.send_json({"error": f"Explanation regeneration failed: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json({"ok": True, "explanation": explanation})
            return
        if parsed.path == "/api/registry":
            entries = load_entries()
            render_registry(entries)
            self.send_json({"ok": True})
            return
        if parsed.path == "/api/rescan":
            from correspondence_scanner import scan, write_pending

            scanned = scan()
            entries = [asdict(entry) for entry in scanned]
            write_pending(scanned, PENDING_PATH)
            render_registry(load_entries())
            self.send_json({"ok": True, "entries": entries})
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not PENDING_PATH.exists():
        from correspondence_scanner import main as scan_main

        scan_main()

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Serving correspondence review UI at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
