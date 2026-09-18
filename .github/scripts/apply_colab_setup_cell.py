#!/usr/bin/env python3
"""Insert the canonical Colab setup cell into a generated notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import nbformat
except ModuleNotFoundError:
    nbformat = None


SETUP_MARKER = "VLLM HOOK SETUP AND DEPENDENCY MANAGER"
IMPORT_MARKERS = (
    "import torch",
    "from torch",
    "import vllm",
    "from vllm",
    "import vllm_hook_plugins",
    "from vllm_hook_plugins",
)


def canonical_template(template_path: Path) -> str:
    return template_path.read_text(encoding="utf-8").rstrip() + "\n"


def cell_source(cell) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def first_runtime_import_index(cells: list) -> int:
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)
        if any(marker in source for marker in IMPORT_MARKERS):
            return index
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            return index
    return 0


def read_notebook(notebook_path: Path):
    if nbformat is None:
        with notebook_path.open(encoding="utf-8") as handle:
            notebook = json.load(handle)
        cells = list(notebook.get("cells", []))
    else:
        notebook = nbformat.read(notebook_path, as_version=4)
        cells = list(notebook.cells)
    return notebook, cells


def write_notebook(notebook_path: Path, notebook, cells: list) -> None:
    if nbformat is None:
        notebook["cells"] = cells
        notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        notebook.cells = cells
        nbformat.write(notebook, notebook_path)


def setup_cell_indexes(cells: list) -> list[int]:
    return [
        index
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code" and SETUP_MARKER in cell_source(cell)
    ]


def verify_canonical_setup_cell(notebook_path: Path, template_path: Path) -> None:
    template = canonical_template(template_path)
    _, cells = read_notebook(notebook_path)
    marker_indexes = setup_cell_indexes(cells)
    if len(marker_indexes) != 1:
        raise RuntimeError(
            f"Expected exactly one canonical setup marker in {notebook_path}, found {len(marker_indexes)}."
        )

    setup_index = marker_indexes[0]
    actual = cell_source(cells[setup_index])
    if actual != template:
        raise RuntimeError(f"Setup cell in {notebook_path} does not exactly match {template_path}.")

    import_index = first_runtime_import_index(
        [cell for index, cell in enumerate(cells) if index != setup_index]
    )
    if import_index < setup_index:
        raise RuntimeError(
            f"Canonical setup cell in {notebook_path} occurs after a dependent runtime import."
        )


def apply_setup_cell(notebook_path: Path, template_path: Path) -> None:
    notebook, cells = read_notebook(notebook_path)
    template = canonical_template(template_path)
    if nbformat is None:
        setup_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": template,
        }
    else:
        setup_cell = nbformat.v4.new_code_cell(template)

    marker_indexes = setup_cell_indexes(cells)

    if marker_indexes:
        insertion_index = marker_indexes[0]
        cells = [cell for index, cell in enumerate(cells) if index not in marker_indexes]
    else:
        insertion_index = first_runtime_import_index(cells)

    import_index = first_runtime_import_index(cells)
    insertion_index = min(insertion_index, import_index)
    cells.insert(insertion_index, setup_cell)
    write_notebook(notebook_path, notebook, cells)
    verify_canonical_setup_cell(notebook_path, template_path)

    action = "Replaced" if marker_indexes else "Inserted"
    print(f"{action} canonical Colab setup cell in {notebook_path} at cell index {insertion_index}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        verify_canonical_setup_cell(args.notebook, args.template)
        print(f"Verified canonical Colab setup cell in {args.notebook}.")
    else:
        apply_setup_cell(args.notebook, args.template)


if __name__ == "__main__":
    main()
