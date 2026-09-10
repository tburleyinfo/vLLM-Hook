#!/usr/bin/env python3
"""Build an inspection queue for CUDA/Metal vLLM-Hook correspondence.

The scanner intentionally keeps the durable output deterministic.  An LLM can
draft notes later, but the file pairing, summaries, diffs, and review HTML do
not depend on any external service.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import html
import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "vllm_hook_plugins" / "vllm_hook_plugins"
PENDING_PATH = PROJECT_ROOT / "pending_correspondence.json"
REVIEW_HTML_PATH = PROJECT_ROOT / "docs" / "correspondence_review.html"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "correspondence.md"


@dataclass
class AnalysisDraft:
    is_match: bool
    logical_name: str
    divergence_note: str
    confidence: str
    reason: str = ""
    similarities: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    inspection_risks: list[str] = field(default_factory=list)
    comparison_rows: list[dict[str, str]] = field(default_factory=list)


@dataclass
class CorrespondenceEntry:
    id: str
    logical_component: str
    cuda_path: str
    metal_path: str
    divergence_note: str
    status: str = "pending"
    confidence: str = "medium"
    reason: str = ""
    similarities: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    inspection_risks: list[str] = field(default_factory=list)
    comparison_rows: list[dict[str, str]] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)


class PlaceholderLLMClient:
    """Local placeholder for future Ollama/LocalAI/API-backed analysis."""

    def analyze_pair(self, cuda_path: Path, metal_path: Path) -> AnalysisDraft:
        stem = strip_backend_suffix(cuda_path.stem)
        return AnalysisDraft(
            is_match=True,
            logical_name=logical_name_from_path(cuda_path, stem),
            divergence_note=(
                "Review required: inspect backend-specific capture, loading, "
                "tensor conversion, and artifact contract differences."
            ),
            confidence="medium",
            similarities=[],
            differences=[],
            inspection_risks=[],
            comparison_rows=[],
        )


class LocalLLMClient:
    """OpenAI-compatible local LLM client."""

    def __init__(self, base_url: str, model: str | None = None, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def list_models(self) -> list[str]:
        request = urllib.request.Request(f"{self.base_url}/v1/models")
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("data") or payload.get("models") or []
        names: list[str] = []
        for model in models:
            if isinstance(model, dict):
                names.append(model.get("id") or model.get("name") or model.get("model") or "")
            elif isinstance(model, str):
                names.append(model)
        return [name for name in names if name]

    def analyze_pair(self, cuda_path: Path, metal_path: Path, web_context: str = "") -> AnalysisDraft:
        model = self.model
        if not model:
            models = self.list_models()
            if not models:
                raise RuntimeError("No local LLM models were returned by /v1/models")
            model = models[0]

        cuda_content = trim_for_prompt(cuda_path.read_text(encoding="utf-8"))
        metal_content = trim_for_prompt(metal_path.read_text(encoding="utf-8"))
        prompt = build_analysis_prompt(cuda_path, metal_path, cuda_content, metal_content, web_context)
        content = self.chat_json(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate candidate correspondence tables for maintainers. "
                        "Do not assume two code elements match just because file names are similar. "
                        "Be concrete, skeptical, and concise. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        try:
            data = parse_json_object(content)
        except json.JSONDecodeError:
            repaired = self.repair_json(model, content)
            data = parse_json_object(repaired)
        return analysis_draft_from_data(data, cuda_path)

    def chat_json(self, model: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                completion = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Local LLM request failed: {exc}") from exc

        return completion["choices"][0]["message"]["content"]

    def repair_json(self, model: str, malformed: str) -> str:
        return self.chat_json(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You repair malformed JSON. Return only valid JSON, no markdown.",
                },
                {
                    "role": "user",
                    "content": (
                        "The following text was intended to be one JSON object but is malformed. "
                        "Fix escaping, commas, quotes, and truncation if possible. Preserve the same schema and content.\n\n"
                        f"{malformed}"
                    ),
                },
            ],
        )


def analysis_draft_from_data(data: dict[str, object], cuda_path: Path) -> AnalysisDraft:
    return AnalysisDraft(
        is_match=bool(data.get("is_match", False)),
        logical_name=str(
            data.get("logical_name")
            or logical_name_from_path(cuda_path, strip_backend_suffix(cuda_path.stem))
        ),
        divergence_note=str(data.get("divergence_note") or data.get("reason") or ""),
        confidence=str(data.get("confidence") or "medium"),
        reason=str(data.get("reason") or ""),
        similarities=string_list(data.get("similarities")),
        differences=string_list(data.get("differences")),
        inspection_risks=string_list(data.get("inspection_risks")),
        comparison_rows=stagger_weak_rows(comparison_rows(data.get("comparison_rows"))),
    )


def trim_for_prompt(content: str, limit: int = 42000) -> str:
    if len(content) <= limit:
        return content
    head = content[: limit // 2]
    tail = content[-limit // 2 :]
    return f"{head}\n\n# ... middle omitted for prompt budget ...\n\n{tail}"


def numbered_source(content: str) -> str:
    return "\n".join(f"{index:5d}: {line}" for index, line in enumerate(content.splitlines(), 1))


def parse_json_object(content: str) -> dict[str, object]:
    content = content.strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM response did not contain a JSON object")
    return payload


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def comparison_rows(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "kind": str(item.get("kind") or "observation"),
                "cuda": str(item.get("cuda") or ""),
                "cuda_ref": str(item.get("cuda_ref") or ""),
                "metal": str(item.get("metal") or ""),
                "metal_ref": str(item.get("metal_ref") or ""),
                "relation": str(item.get("relation") or ""),
                "status": str(item.get("status") or "pending"),
                "confidence": str(item.get("confidence") or ""),
            }
        )
    return rows


def stagger_weak_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep exact/strong/mild matches side by side; split weak rows visually."""

    staggered: list[dict[str, str]] = []
    for row in rows:
        kind = row.get("kind", "").lower().strip()
        has_cuda = bool(row.get("cuda", "").strip())
        has_metal = bool(row.get("metal", "").strip())
        if has_cuda and has_metal and should_stagger_kind(kind):
            relation = row.get("relation", "")
            status = row.get("status", "pending")
            confidence = row.get("confidence", "")
            staggered.append(
                {
                    "kind": kind or "cuda_only",
                    "cuda": row.get("cuda", ""),
                    "cuda_ref": row.get("cuda_ref", ""),
                    "metal": "",
                    "metal_ref": "",
                    "relation": f"Staggered from weak/divergent candidate: {relation}",
                    "status": status,
                    "confidence": confidence,
                }
            )
            staggered.append(
                {
                    "kind": kind or "metal_only",
                    "cuda": "",
                    "cuda_ref": "",
                    "metal": row.get("metal", ""),
                    "metal_ref": row.get("metal_ref", ""),
                    "relation": f"Related to previous row, but not strong enough to align side by side: {relation}",
                    "status": status,
                    "confidence": confidence,
                }
            )
        else:
            staggered.append(row)
    return staggered


def should_stagger_kind(kind: str) -> bool:
    """Return True when side-by-side would overstate the correspondence."""

    weak_tokens = ("weak", "diverg", "partial", "no_match", "risk", "loose")
    strong_tokens = ("exact", "strong", "mild", "similar", "same", "contract")
    if any(token in kind for token in strong_tokens):
        return False
    return any(token in kind for token in weak_tokens)


def build_analysis_prompt(
    cuda_path: Path,
    metal_path: Path,
    cuda_content: str,
    metal_content: str,
    web_context: str = "",
) -> str:
    candidate_name = logical_name_from_path(cuda_path, strip_backend_suffix(cuda_path.stem))
    context_block = ""
    if web_context.strip():
        context_block = f"\nOptional external context:\n{web_context.strip()}\n"
    return f"""You are comparing vLLM-Hook CUDA and vLLM-Metal code.

Generate a candidate correspondence matrix. Maybe these files match globally,
maybe they only partially overlap, and maybe individual code elements have no
counterpart. Do not force one-to-one correspondence. Focus on:
- shared contract or user-facing behavior
- capture/injection mechanism
- artifact schema and normalization boundary
- tensor dtype/shape/order/layer/token alignment
- concurrency, lifecycle, cache, and memory behavior

CUDA path: {repo_relative(cuda_path)}
Metal path: {repo_relative(metal_path)}
Candidate logical name from repository heuristics: {candidate_name}
{context_block}
CUDA code:
```python
{numbered_source(cuda_content)}
```

Metal code:
```python
{numbered_source(metal_content)}
```

Return raw JSON only with this shape:
{{
  "is_match": true,
  "logical_name": "{candidate_name}",
  "comparison_rows": [
    {{
      "kind": "exact | strong | mild | weak | partial | divergent | cuda_only | metal_only | no_match | risk | contract | artifact | lifecycle | dependency",
      "cuda": "CUDA-side code fact, method, block, or behavior. Leave blank if absent.",
      "cuda_ref": "CUDA path with line or range, e.g. vllm_hook_plugins/.../file.py:12-24. Leave blank if absent or uncertain.",
      "metal": "Metal-side code fact, method, block, or behavior. Leave blank if absent.",
      "metal_ref": "Metal path with line or range. Leave blank if absent or uncertain.",
      "relation": "Explain exact match, loose analogy, divergence, missing counterpart, or inspection risk.",
      "status": "pending",
      "confidence": "high"
    }}
  ],
  "similarities": ["..."],
  "differences": ["..."],
  "inspection_risks": ["..."],
  "divergence_note": "One maintainer-friendly paragraph.",
  "confidence": "high"
}}
Use confidence high, medium, or low.
Create an exhaustive but useful candidate table. Similarity is represented by
both platform cells being populated in the same row for exact, strong, or mild
similarities. For weak similarities, partial similarities, loose analogies, and
divergences, DO NOT put both platforms in the same row. Instead create adjacent
staggered rows: one CUDA-only row and one Metal-only row, each with the opposite
cell blank, and explain the loose relationship in the relation field. Class
names, method names, shared contracts, and matching artifact concepts may be
aligned when they are at least mildly similar, even if platform suffixes differ.
Leave the opposite platform cell blank when no counterpart exists. Prefer many
precise rows over a broad summary.
Use the numbered source blocks for line references. If a row points to an
entire function or class, use its line range. If a row is conceptual and you are
not sure of the location, leave the reference blank instead of guessing.
"""


def strip_backend_suffix(stem: str) -> str:
    for suffix in ("_metal", "_cuda"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def logical_name_from_path(path: Path, stem: str) -> str:
    known_parts = {
        "hookqk": "HookQK",
        "llm": "LLM",
        "qk": "QK",
        "qkv": "QKV",
        "mlx": "MLX",
        "vllm": "vLLM",
    }
    name = "".join(known_parts.get(part, part.capitalize()) for part in stem.split("_") if part)
    relative = path.relative_to(PACKAGE_ROOT)
    if relative.parts[0] == "workers":
        return f"Worker: {name}"
    if relative.parts[0] == "analyzers":
        return f"Analyzer: {name}"
    if relative.parts[0] == "metal" or stem in {"hook_llm", "run_utils", "_hook_plugin"}:
        return f"Runtime: {name}"
    return f"Component: {name}"


def python_summary(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    summary: dict[str, object] = {
        "lines": len(text.splitlines()),
        "classes": [],
        "functions": [],
        "imports": [],
        "refs": {},
    }
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return summary

    classes: list[str] = []
    functions: list[str] = []
    imports: list[str] = []
    refs: dict[str, dict[str, str]] = {"classes": {}, "functions": {}, "imports": {}}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
            refs["classes"][node.name] = node_ref(path, node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
            refs["functions"].setdefault(node.name, node_ref(path, node))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                refs["imports"].setdefault(alias.name, node_ref(path, node))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)
            refs["imports"].setdefault(module, node_ref(path, node))

    summary["classes"] = sorted(set(classes))
    summary["functions"] = sorted(set(functions))
    summary["imports"] = sorted(set(imports))
    summary["refs"] = refs
    return summary


def node_ref(path: Path, node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", start)
    if not start:
        return repo_relative(path)
    if end and end != start:
        return f"{repo_relative(path)}:{start}-{end}"
    return f"{repo_relative(path)}:{start}"


def candidate_cuda_files() -> Iterable[Path]:
    roots = [
        PACKAGE_ROOT / "workers",
        PACKAGE_ROOT / "analyzers",
        PACKAGE_ROOT,
    ]
    root_level = {"hook_llm.py", "run_utils.py", "_hook_plugin.py"}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.py")):
            if path.name == "__init__.py" or path.parent.name == "metal":
                continue
            if root == PACKAGE_ROOT and path.name not in root_level:
                continue
            yield path


def metal_match_for(cuda_path: Path) -> Path | None:
    relative = cuda_path.relative_to(PACKAGE_ROOT)
    stem = strip_backend_suffix(cuda_path.stem)

    candidates: list[Path] = []
    if relative.parts[0] in {"workers", "analyzers"}:
        candidates.append(cuda_path.parent / "metal" / f"{stem}_metal.py")
    candidates.append(PACKAGE_ROOT / "metal" / f"{stem}_metal.py")
    candidates.append(PACKAGE_ROOT / "metal" / cuda_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_entry(cuda_path: Path, metal_path: Path, llm: PlaceholderLLMClient) -> CorrespondenceEntry:
    draft = llm.analyze_pair(cuda_path, metal_path)
    cuda_summary = python_summary(cuda_path)
    metal_summary = python_summary(metal_path)
    entry_id = strip_backend_suffix(cuda_path.stem).strip("_").replace("_", "-")
    return CorrespondenceEntry(
        id=entry_id,
        logical_component=draft.logical_name,
        cuda_path=repo_relative(cuda_path),
        metal_path=repo_relative(metal_path),
        divergence_note=draft.divergence_note,
        confidence=draft.confidence,
        reason=draft.reason,
        similarities=draft.similarities,
        differences=draft.differences,
        inspection_risks=draft.inspection_risks,
        comparison_rows=draft.comparison_rows or heuristic_comparison_rows(cuda_summary, metal_summary),
        summary={
            "cuda": cuda_summary,
            "metal": metal_summary,
        },
    )


def heuristic_comparison_rows(
    cuda_summary: dict[str, object], metal_summary: dict[str, object]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "kind": "size",
            "cuda": f"{cuda_summary.get('lines', 0)} lines",
            "cuda_ref": "",
            "metal": f"{metal_summary.get('lines', 0)} lines",
            "metal_ref": "",
            "relation": "Implementation size comparison; larger Metal files often indicate wrappers, reconstruction, memory guards, or adapter logic.",
            "status": "pending",
            "confidence": "high",
        }
    ]
    symbol_fields = [
        ("class", "classes", "Class"),
        ("function", "functions", "Function"),
    ]
    for kind, key, label in symbol_fields:
        cuda_items = [str(item) for item in cuda_summary.get(key, [])]
        metal_items = [str(item) for item in metal_summary.get(key, [])]
        cuda_refs = dict(cuda_summary.get("refs", {}).get(key, {}))  # type: ignore[union-attr]
        metal_refs = dict(metal_summary.get("refs", {}).get(key, {}))  # type: ignore[union-attr]
        grouped: dict[str, dict[str, str]] = {}
        for item in cuda_items:
            grouped.setdefault(normalize_symbol(item), {})["cuda"] = item
        for item in metal_items:
            grouped.setdefault(normalize_symbol(item), {})["metal"] = item
        for normalized, sides in sorted(grouped.items()):
            cuda_item = sides.get("cuda", "")
            metal_item = sides.get("metal", "")
            rows.append(
                {
                    "kind": symbol_kind(kind, cuda_item, metal_item),
                    "cuda": f"{label}: {cuda_item}" if cuda_item else "",
                    "cuda_ref": cuda_refs.get(cuda_item, ""),
                    "metal": f"{label}: {metal_item}" if metal_item else "",
                    "metal_ref": metal_refs.get(metal_item, ""),
                    "relation": relation_for_symbol(cuda_item, metal_item, normalized),
                    "status": "pending",
                    "confidence": "medium",
                }
            )

    cuda_imports = set(str(item) for item in cuda_summary.get("imports", []))
    metal_imports = set(str(item) for item in metal_summary.get("imports", []))
    cuda_import_refs = dict(cuda_summary.get("refs", {}).get("imports", {}))  # type: ignore[union-attr]
    metal_import_refs = dict(metal_summary.get("refs", {}).get("imports", {}))  # type: ignore[union-attr]
    for item in sorted(cuda_imports | metal_imports):
        rows.append(
            {
                "kind": "dependency",
                "cuda": f"Import: {item}" if item in cuda_imports else "",
                "cuda_ref": cuda_import_refs.get(item, ""),
                "metal": f"Import: {item}" if item in metal_imports else "",
                "metal_ref": metal_import_refs.get(item, ""),
                "relation": relation_for_presence(item in cuda_imports, item in metal_imports),
                "status": "pending",
                "confidence": "medium",
            }
        )
    return rows


def normalize_symbol(name: str) -> str:
    normalized = name.lower()
    for token in ("metal", "cuda", "mlx", "torch"):
        normalized = normalized.replace(token, "")
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized


def relation_for_symbol(cuda_item: str, metal_item: str, normalized: str) -> str:
    if cuda_item and metal_item:
        if cuda_item == metal_item:
            return "Same named element appears on both platforms; inspect whether behavior and contract match."
        return "Likely platform-named counterparts; inspect whether the Metal implementation preserves the CUDA contract."
    return relation_for_presence(bool(cuda_item), bool(metal_item))


def symbol_kind(base_kind: str, cuda_item: str, metal_item: str) -> str:
    if cuda_item and metal_item:
        if cuda_item == metal_item:
            return "exact"
        return "mild"
    return base_kind


def relation_for_presence(has_cuda: bool, has_metal: bool) -> str:
    if has_cuda and has_metal:
        return "Same named element appears on both platforms; inspect whether behavior and contract match."
    if has_cuda:
        return "CUDA-only element; inspect whether Metal intentionally omits, replaces, or delegates this behavior."
    return "Metal-only element; inspect whether this is wrapper, adapter, reconstruction, conversion, or platform-guard logic."


def scan() -> list[CorrespondenceEntry]:
    llm = PlaceholderLLMClient()
    entries: list[CorrespondenceEntry] = []
    for cuda_path in candidate_cuda_files():
        metal_path = metal_match_for(cuda_path)
        if metal_path is None:
            continue
        entries.append(build_entry(cuda_path, metal_path, llm))
    return entries


def write_pending(entries: list[CorrespondenceEntry], path: Path, refresh_matrix: bool = False) -> None:
    existing_by_id: dict[str, dict[str, object]] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
        if isinstance(existing, list):
            existing_by_id = {
                str(item.get("id")): item for item in existing if isinstance(item, dict) and item.get("id")
            }

    serialized: list[dict[str, object]] = []
    for entry in entries:
        data = asdict(entry)
        old = existing_by_id.get(entry.id)
        if old:
            for key in (
                "status",
                "divergence_note",
                "confidence",
                "reason",
                "similarities",
                "differences",
                "inspection_risks",
                "comparison_rows",
            ):
                if key == "comparison_rows" and refresh_matrix:
                    continue
                if old.get(key):
                    data[key] = old[key]
        serialized.append(data)

    path.write_text(
        json.dumps(serialized, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_registry_template(path: Path) -> None:
    if path.exists() and path.read_text(encoding="utf-8").strip():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# vLLM-Hook CUDA/Metal Correspondence\n\n"
        "This registry is generated from reviewed correspondence entries.\n\n"
        "| Logical Component | CUDA Path | Metal Path | Divergence Note | Status | Confidence |\n"
        "|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )


def render_summary(summary: dict[str, object]) -> str:
    classes = ", ".join(summary.get("classes", [])) or "None"
    functions = ", ".join(summary.get("functions", [])) or "None"
    imports = ", ".join(summary.get("imports", [])[:12]) or "None"
    return (
        f"<dl>"
        f"<dt>Lines</dt><dd>{summary.get('lines', 0)}</dd>"
        f"<dt>Classes</dt><dd>{html.escape(classes)}</dd>"
        f"<dt>Functions</dt><dd>{html.escape(functions)}</dd>"
        f"<dt>Imports</dt><dd>{html.escape(imports)}</dd>"
        f"</dl>"
    )


def render_review_html(entries: list[CorrespondenceEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    for entry in entries:
        cuda_path = PROJECT_ROOT / entry.cuda_path
        metal_path = PROJECT_ROOT / entry.metal_path
        cuda_lines = cuda_path.read_text(encoding="utf-8").splitlines()
        metal_lines = metal_path.read_text(encoding="utf-8").splitlines()
        diff = "\n".join(
            difflib.unified_diff(
                cuda_lines,
                metal_lines,
                fromfile=entry.cuda_path,
                tofile=entry.metal_path,
                lineterm="",
                n=3,
            )
        )
        sections.append(
            "<section>"
            f"<h2>{html.escape(entry.logical_component)}</h2>"
            f"<p><strong>CUDA:</strong> <code>{html.escape(entry.cuda_path)}</code><br>"
            f"<strong>Metal:</strong> <code>{html.escape(entry.metal_path)}</code></p>"
            f"<p><strong>Draft note:</strong> {html.escape(entry.divergence_note)}</p>"
            "<div class=\"summaries\">"
            f"<div><h3>CUDA Summary</h3>{render_summary(entry.summary['cuda'])}</div>"
            f"<div><h3>Metal Summary</h3>{render_summary(entry.summary['metal'])}</div>"
            "</div>"
            "<h3>Unified Diff</h3>"
            f"<pre>{html.escape(diff)}</pre>"
            "<div class=\"codegrid\">"
            f"<div><h3>CUDA</h3><pre>{html.escape('\\n'.join(cuda_lines))}</pre></div>"
            f"<div><h3>Metal</h3><pre>{html.escape('\\n'.join(metal_lines))}</pre></div>"
            "</div>"
            "</section>"
        )

    document = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>vLLM-Hook Correspondence Review</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;color:#202124;background:#f7f7f5}"
        "header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d8d8d2;padding:16px 24px;z-index:1}"
        "main{padding:20px 24px 40px}"
        "section{background:#fff;border:1px solid #d8d8d2;border-radius:8px;margin:0 0 20px;padding:18px}"
        "h1,h2,h3{margin:0 0 10px} p{line-height:1.4}"
        "code{background:#f0f0ec;padding:2px 4px;border-radius:4px}"
        "pre{background:#111;color:#f3f3ed;padding:14px;border-radius:6px;overflow:auto;font-size:12px;line-height:1.45}"
        ".summaries,.codegrid{display:grid;grid-template-columns:1fr 1fr;gap:16px}"
        "dt{font-weight:700;margin-top:8px} dd{margin:2px 0 0}"
        "@media (max-width:900px){.summaries,.codegrid{grid-template-columns:1fr}}"
        "</style></head><body>"
        "<header><h1>vLLM-Hook CUDA/Metal Correspondence Review</h1>"
        f"<p>{len(entries)} candidate pairs generated from repository heuristics.</p></header>"
        f"<main>{''.join(sections)}</main></body></html>"
    )
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending", type=Path, default=PENDING_PATH)
    parser.add_argument("--html", type=Path, default=REVIEW_HTML_PATH)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--refresh-matrix", action="store_true")
    args = parser.parse_args()

    entries = scan()
    args.pending.parent.mkdir(parents=True, exist_ok=True)
    write_pending(entries, args.pending, refresh_matrix=args.refresh_matrix)
    ensure_registry_template(REGISTRY_PATH)
    if not args.no_html:
        render_review_html(entries, args.html)

    print(f"Wrote {len(entries)} candidate pairs to {args.pending}")
    if not args.no_html:
        print(f"Wrote review page to {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
