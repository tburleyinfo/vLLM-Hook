#!/usr/bin/env python3
"""Local web UI for reviewing CUDA/Metal correspondence candidates."""

from __future__ import annotations

import argparse
import difflib
import json
import mimetypes
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PENDING_PATH = PROJECT_ROOT / "pending_correspondence.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "correspondence.md"
DEFAULT_LLM_URL = "http://127.0.0.1:8033"
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def load_entries() -> list[dict[str, Any]]:
    if not PENDING_PATH.exists():
        return []
    return json.loads(PENDING_PATH.read_text(encoding="utf-8"))


def save_entries(entries: list[dict[str, Any]]) -> None:
    PENDING_PATH.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


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


def set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = time.time()


def get_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def analyze_entry_job(job_id: str, entry_id: str, web_context: str) -> None:
    set_job(job_id, status="running", message="Generating candidate rows")
    try:
        from correspondence_scanner import LocalLLMClient

        entries = load_entries()
        entry = find_entry(entries, entry_id)
        if entry is None:
            raise ValueError("entry not found")

        client = LocalLLMClient(DEFAULT_LLM_URL, timeout=300)
        draft = client.analyze_pair(
            PROJECT_ROOT / entry["cuda_path"],
            PROJECT_ROOT / entry["metal_path"],
            web_context=web_context,
        )
        entry["divergence_note"] = draft.divergence_note
        entry["confidence"] = draft.confidence
        entry["reason"] = draft.reason
        entry["similarities"] = draft.similarities
        entry["differences"] = draft.differences
        entry["inspection_risks"] = draft.inspection_risks
        if draft.comparison_rows:
            entry["comparison_rows"] = [
                {**row, "status": row.get("status") or "pending"}
                for row in draft.comparison_rows
            ]
        save_entries(entries)
        render_registry(entries)
        set_job(job_id, status="done", message="Candidate matrix generated")
    except Exception as exc:
        set_job(job_id, status="error", message=str(exc))


def app_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>vLLM-Hook Correspondence Review</title>
  <style>
    :root {
      --bg: #f6f5f1;
      --panel: #ffffff;
      --panelSoft: #fbfaf7;
      --panelHead: #f1f0ea;
      --line: #d7d5cc;
      --text: #202124;
      --muted: #63645f;
      --accent: #176b5b;
      --accentSoft: #e5f2ed;
      --warn: #8a5a00;
      --danger: #b3261e;
      --aiBg: #243447;
      --buttonBg: #ffffff;
      --buttonHover: #aaa79b;
      --inputBg: #ffffff;
      --refBg: #eef4f7;
      --refText: #254756;
      --blankBg: #faf9f4;
      --blankStripe: #f0eee6;
      --emptyRefText: #a09c91;
      --relationBg: #fffdf5;
      --alignedBg: #f4fbf8;
      --staggeredBg: #fffdf7;
      --unmatchedBg: #fbf7f7;
      --code: #101418;
      --codeText: #eef2f3;
      --codeLine: #8ea0a8;
      --codeHit: #31424d;
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
    aside {
      border-right: 1px solid var(--line);
      background: var(--panelSoft);
      overflow: auto;
      max-height: calc(100vh - 58px);
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
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
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
      margin-bottom: 10px;
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
      overflow: hidden;
      margin-bottom: 14px;
    }
    .matrix-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .matrix-head h3 { margin: 0; font-size: 14px; }
    .matrix-scroll { overflow: auto; max-height: 68vh; }
    table.matrix {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }
    .matrix th {
      position: sticky;
      top: 0;
      background: var(--panelHead);
      z-index: 1;
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px;
    }
    .matrix td {
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      vertical-align: top;
      padding: 0;
    }
    .matrix-row.aligned-row td {
      background: var(--alignedBg);
    }
    .matrix-row.staggered-row td {
      background: var(--staggeredBg);
    }
    .matrix-row.unmatched-row td {
      background: var(--unmatchedBg);
    }
    .matrix th:nth-child(1), .matrix td:nth-child(1) { width: 110px; }
    .matrix th:nth-child(2), .matrix td:nth-child(2) { width: 120px; }
    .matrix th:nth-child(3), .matrix td:nth-child(3),
    .matrix th:nth-child(5), .matrix td:nth-child(5) { width: 22%; }
    .matrix th:nth-child(4), .matrix td:nth-child(4),
    .matrix th:nth-child(6), .matrix td:nth-child(6) { width: 140px; }
    .matrix th:nth-child(7), .matrix td:nth-child(7) { width: 26%; }
    .matrix th:nth-child(8), .matrix td:nth-child(8) { width: 96px; }
    .matrix textarea, .matrix input {
      width: 100%;
      min-height: 74px;
      border: 0;
      border-radius: 0;
      resize: vertical;
      background: transparent;
      padding: 8px;
      line-height: 1.35;
    }
    .matrix input {
      min-height: 38px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .ref-chip {
      display: block;
      width: 100%;
      min-height: 38px;
      border: 0;
      border-radius: 0;
      background: var(--refBg);
      color: var(--refText);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.35;
      padding: 8px;
      text-align: left;
      overflow-wrap: anywhere;
    }
    .ref-chip.empty-ref {
      background: var(--blankBg);
      color: var(--emptyRefText);
    }
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
      padding: 8px;
    }
    .row-actions select {
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
      padding: 10px 4px;
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
      .summaries, .codegrid, .note-grid, .analysis, .metrics, .semantic-grid { grid-template-columns: 1fr; }
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

    const entryList = document.getElementById("entryList");
    const detail = document.getElementById("detail");
    const search = document.getElementById("search");
    const statusFilter = document.getElementById("statusFilter");
    const themeToggle = document.getElementById("themeToggle");

    function applyTheme(theme) {
      document.body.dataset.theme = theme;
      localStorage.setItem("correspondenceTheme", theme);
      themeToggle.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
    }

    const savedTheme = localStorage.getItem("correspondenceTheme");
    const preferredTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    applyTheme(savedTheme || preferredTheme);
    themeToggle.addEventListener("click", () => {
      applyTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
    });

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

    function renderNumberedSnippet(content, startLine, radius = 8) {
      const lines = String(content || "").split("\\n");
      const start = Math.max(1, startLine - radius);
      const end = Math.min(lines.length, startLine + radius);
      const rendered = [];
      for (let number = start; number <= end; number += 1) {
        const marker = number === startLine ? ">>" : "  ";
        rendered.push(`${marker} ${String(number).padStart(5, " ")}  ${lines[number - 1] || ""}`);
      }
      return escapeHtml(rendered.join("\\n"));
    }

    function lineFromRef(ref) {
      const match = String(ref || "").match(/:([0-9]+)/);
      return match ? Number(match[1]) : 1;
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
      const similarities = payload.similarities || [];
      const differences = payload.differences || [];
      const risks = payload.inspection_risks || [];
      detail.innerHTML = `
        <div class="metrics" id="metrics"></div>
        <div class="toolbar">
          <div class="tabs">
            <button class="tab ${activeView === "matrix" ? "active" : ""}" data-view="matrix">Matrix</button>
            <button class="tab ${activeView === "side" ? "active" : ""}" data-view="side">Full Code</button>
            <button class="tab ${activeView === "semantic" ? "active" : ""}" data-view="semantic">Semantic Split</button>
          </div>
          <span class="pill ${escapeHtml(payload.status)}">${escapeHtml(payload.status)} / ${escapeHtml(payload.confidence)} confidence</span>
        </div>
        <section class="meta">
          <h2>${escapeHtml(payload.logical_component)}</h2>
          <div class="paths">
            <div><strong>CUDA</strong> <code>${escapeHtml(payload.cuda_path)}</code></div>
            <div><strong>Metal</strong> <code>${escapeHtml(payload.metal_path)}</code></div>
          </div>
          <div class="note-grid">
            <textarea id="note">${escapeHtml(payload.divergence_note)}</textarea>
            <select id="confidence">
              ${["high", "medium", "low"].map((value) => `<option value="${value}" ${payload.confidence === value ? "selected" : ""}>${value}</option>`).join("")}
            </select>
          </div>
          <div class="actions">
            <button class="ai" id="analyze">Generate Candidate Matrix</button>
            <button id="strong">Strong Candidate</button>
            <button class="primary" id="approve">Approve</button>
            <button id="save">Save Draft</button>
            <button id="addRow">Add Matrix Row</button>
            <button class="danger" id="reject">Reject</button>
          </div>
          <div class="job-status" id="jobStatus"></div>
        </section>
        <div id="matrixView" class="${activeView === "matrix" ? "" : "hidden"}">
          <div id="snippetPanel"></div>
          ${renderMatrix(payload.comparison_rows || [])}
        </div>
        <div class="analysis">
          <section class="inspector">
            <h3>LLM Inspection Notes</h3>
            ${renderFindingList("Similarities", similarities)}
            ${renderFindingList("Differences", differences)}
            ${renderFindingList("Review risks", risks)}
          </section>
          <section class="inspector">
            <h3>Optional Web Context</h3>
            <div class="webbox">
              <input id="webQuery" value="${escapeHtml(defaultWebQuery(payload))}">
              <button id="webSearch">Fetch Web Context</button>
              <textarea id="webContext" placeholder="External snippets or notes to include in the LLM prompt"></textarea>
              <div class="web-results" id="webResults">No web context loaded.</div>
            </div>
          </section>
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
      document.getElementById("webSearch").addEventListener("click", fetchWebContext);
      attachMatrixControls();
      attachRowButtons();
      detail.querySelectorAll(".ref-chip").forEach((button) => {
        button.addEventListener("click", () => jumpToRef(button.title));
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
            <div class="bulk-actions">
              <button id="selectAllRows">Select All</button>
              <button id="rowPending">Rows Pending</button>
              <button id="rowStrong">Rows Strong</button>
              <button id="rowApprove">Rows Approve</button>
              <button id="rowReject">Rows Reject</button>
              <button class="danger" id="rowRemove">Remove Selected</button>
            </div>
          </div>
          <div class="path" style="padding: 0 12px 10px;">Both cells filled should mean exact, strong, or mild similarity. Weak, partial, or divergent candidates should be staggered into one-sided rows with blank opposite cells.</div>
          <div class="matrix-scroll">
            <table class="matrix">
              <thead>
                <tr>
                  <th>Select</th>
                  <th>Kind</th>
                  <th>CUDA Ref</th>
                  <th>CUDA</th>
                  <th>Metal Ref</th>
                  <th>Metal</th>
                  <th>Relation</th>
                  <th>Row Status</th>
                </tr>
              </thead>
              <tbody id="matrixBody">
                ${rows.map((row) => renderMatrixRow(row)).join("")}
              </tbody>
            </table>
          </div>
        </section>
      `;
    }

    function renderMatrixRow(row) {
      const hasCuda = Boolean((row.cuda || "").trim());
      const hasMetal = Boolean((row.metal || "").trim());
      const rowClass = hasCuda && hasMetal ? "aligned-row" : (hasCuda || hasMetal ? "staggered-row" : "unmatched-row");
      return `
        <tr class="matrix-row ${rowClass}">
          <td class="select-cell"><input type="checkbox" class="row-selected"></td>
          <td><input class="row-kind" value="${escapeHtml(row.kind || "observation")}"></td>
          <td>${renderRefInput("row-cuda-ref", row.cuda_ref || "")}</td>
          <td class="${row.cuda ? "" : "blank-cell"}"><textarea class="row-cuda">${escapeHtml(row.cuda || "")}</textarea></td>
          <td>${renderRefInput("row-metal-ref", row.metal_ref || "")}</td>
          <td class="${row.metal ? "" : "blank-cell"}"><textarea class="row-metal">${escapeHtml(row.metal || "")}</textarea></td>
          <td class="relation-cell"><textarea class="row-relation">${escapeHtml(row.relation || "")}</textarea></td>
          <td>
            <div class="row-actions">
              <select class="row-status">
                ${["pending", "strong_candidate", "approved", "rejected"].map((value) => `<option value="${value}" ${(row.status || "pending") === value ? "selected" : ""}>${value}</option>`).join("")}
              </select>
              <button type="button" class="remove-row">Remove</button>
            </div>
          </td>
        </tr>
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
        status: row.querySelector(".row-status").value,
        confidence: document.getElementById("confidence").value,
      })).filter((row) => row.kind || row.cuda_ref || row.cuda || row.metal_ref || row.metal || row.relation);
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

    function jumpToRef(ref) {
      if (!ref || ref === "No reference") return;
      const platform = platformFromRef(ref);
      const line = lineFromRef(ref);
      const content = platform === "metal" ? currentPayload?.metal_content : currentPayload?.cuda_content;
      const panel = document.getElementById("snippetPanel");
      if (!panel) return;
      panel.innerHTML = `
        <section class="snippet-panel">
          <h3>${platform === "metal" ? "Metal" : "CUDA"} Code Excerpt</h3>
          <div class="path">${escapeHtml(ref)}</div>
          <pre>${renderNumberedSnippet(content, line)}</pre>
        </section>
      `;
      panel.scrollIntoView({ block: "nearest" });
    }

    function renderFindingList(title, values) {
      if (!values.length) {
        return `<h3>${title}</h3><p class="path">No LLM notes yet.</p>`;
      }
      return `<h3>${title}</h3><ul>${values.map((value) => `<li>${escapeHtml(value)}</li>`).join("")}</ul>`;
    }

    function defaultWebQuery(payload) {
      return `${payload.logical_component} vLLM Metal MLX CUDA hooks artifact parity`;
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
      button.disabled = true;
      button.textContent = "Generating...";
      status.textContent = "Queued candidate generation...";
      try {
        const webContext = document.getElementById("webContext").value;
        const payload = await api("/api/analyze", {
          method: "POST",
          body: JSON.stringify({ id: selectedId, web_context: webContext }),
        });
        activeJobId = payload.job_id;
        pollJob(payload.job_id, selectedId);
      } catch (error) {
        alert(error.message);
        button.disabled = false;
        button.textContent = "Generate Candidate Matrix";
        status.textContent = "";
      }
    }

    async function pollJob(jobId, entryId) {
      const status = document.getElementById("jobStatus");
      const button = document.getElementById("analyze");
      try {
        const job = await api(`/api/job?id=${encodeURIComponent(jobId)}`);
        if (status) status.textContent = job.message || job.status;
        if (job.status === "done") {
          if (selectedId === entryId) {
            await loadEntries(true);
          }
          return;
        }
        if (job.status === "error") {
          if (button) {
            button.disabled = false;
            button.textContent = "Generate Candidate Matrix";
          }
          if (status) status.textContent = `Generation failed: ${job.message}`;
          return;
        }
        setTimeout(() => pollJob(jobId, entryId), 1500);
      } catch (error) {
        if (button) {
          button.disabled = false;
          button.textContent = "Generate Candidate Matrix";
        }
        if (status) status.textContent = error.message;
      }
    }

    async function fetchWebContext() {
      const results = document.getElementById("webResults");
      const context = document.getElementById("webContext");
      const query = document.getElementById("webQuery").value;
      results.textContent = "Searching...";
      try {
        const payload = await api("/api/web", {
          method: "POST",
          body: JSON.stringify({ query }),
        });
        const text = payload.context || "No results found.";
        results.textContent = text;
        context.value = text;
      } catch (error) {
        results.textContent = error.message;
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
                self.send_json({"error": f"Web context fetch failed: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json({"ok": True, "results": results, "context": web_results_to_context(results)})
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
