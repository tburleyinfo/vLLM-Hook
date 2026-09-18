#!/usr/bin/env python3
"""Regression tests for deterministic Colab setup-cell injection."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import apply_colab_setup_cell


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "notebooks" / "snippets" / "colab_setup.py"
MARKER = "VLLM HOOK SETUP AND DEPENDENCY MANAGER"
STALE_SETUP = f"""# ==============================================================================
# {MARKER}
# ==============================================================================
print("old setup")
"""


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


class ApplyColabSetupCellTest(unittest.TestCase):
    def test_replaces_stale_setup_and_artifact_copy_matches_post_processed_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            notebook_path = tmp / "target_colab.ipynb"
            artifact_path = tmp / "translated-notebook.ipynb"
            notebook = {
                "cells": [
                    markdown_cell("# Demo"),
                    code_cell(STALE_SETUP),
                    code_cell("import torch\nfrom vllm import SamplingParams\nfrom vllm_hook_plugins import HookLLM\n"),
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

            apply_colab_setup_cell.apply_setup_cell(notebook_path, TEMPLATE)
            shutil.copyfile(notebook_path, artifact_path)
            apply_colab_setup_cell.verify_canonical_setup_cell(artifact_path, TEMPLATE)

            result = json.loads(artifact_path.read_text(encoding="utf-8"))
            canonical = TEMPLATE.read_text(encoding="utf-8").rstrip() + "\n"
            setup_cells = [
                cell
                for cell in result["cells"]
                if cell["cell_type"] == "code" and MARKER in apply_colab_setup_cell.cell_source(cell)
            ]
            self.assertEqual(len(setup_cells), 1)
            setup_source = apply_colab_setup_cell.cell_source(setup_cells[0])
            self.assertEqual(setup_source, canonical)
            self.assertIn("VLLM_HOOK_REPO_URL", setup_source)
            self.assertIn("VLLM_HOOK_REPO_BRANCH", setup_source)
            self.assertNotIn('print("old setup")', artifact_path.read_text(encoding="utf-8"))

            setup_index = result["cells"].index(setup_cells[0])
            dependent_import_index = next(
                index
                for index, cell in enumerate(result["cells"])
                if index != setup_index
                and cell["cell_type"] == "code"
                and "import torch" in apply_colab_setup_cell.cell_source(cell)
            )
            self.assertLess(setup_index, dependent_import_index)
            self.assertEqual(artifact_path.read_text(encoding="utf-8"), notebook_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
