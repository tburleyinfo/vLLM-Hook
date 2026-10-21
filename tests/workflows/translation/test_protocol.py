"""Tests for the experiment's evidence gates, independent of model generation."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".github/scripts/translation"))

spec = importlib.util.spec_from_file_location("experiment_runner", Path(__file__).resolve().parents[3] / ".github/scripts/translation/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProtocolTests(unittest.TestCase):
    def result(self):
        return {"status": "pass", "settings": {"model": "same-model"}, "token_ids": [[1, 2]],
                "tokenizer_sha256": "same", "input_ranges": [[[0, 1], [1, 2]]],
                "captures": [{"layer": 0, "q_dtype": "torch.float32"}], "capture_config": {"head_dim": 8},
                "output_count": 1, "generate_executed": True, "analyze_executed": True, "scores": [0.5]}

    def test_close_scores_do_not_override_different_models(self):
        cpu, metal = self.result(), self.result()
        metal["settings"]["model"] = "different-model"
        self.assertEqual(runner.compare(cpu, metal, {"atol": 1e-4, "rtol": 1e-3})["status"], "fail")

    def test_missing_execution_is_not_parity(self):
        cpu, metal = self.result(), self.result()
        metal["generate_executed"] = False
        self.assertEqual(runner.compare(cpu, metal, {"atol": 1e-4, "rtol": 1e-3})["status"], "fail")

    def test_nonfinite_scores_fail(self):
        cpu, metal = self.result(), self.result()
        metal["scores"] = [float("nan")]
        self.assertEqual(runner.compare(cpu, metal, {"atol": 1e-4, "rtol": 1e-3})["status"], "fail")

    def test_documented_tolerance_applied(self):
        cpu, metal = self.result(), self.result()
        metal["scores"] = [0.50001]
        self.assertEqual(runner.compare(cpu, metal, {"atol": 1e-4, "rtol": 1e-3})["status"], "pass")
        metal["scores"] = [0.51]
        self.assertEqual(runner.compare(cpu, metal, {"atol": 1e-4, "rtol": 1e-3})["status"], "fail")

    def test_prompt_conditions_differ_only_by_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            runner.prepare(run, runner.PROFILE / "manifest.json")
            baseline = (run / "source_only/prompt.md").read_text()
            graph_prompt = (run / "source_validated_dag/prompt.md").read_text()
            self.assertTrue(graph_prompt.startswith(baseline))
            self.assertNotIn("flowchart TD", baseline)
            self.assertIn("flowchart TD", graph_prompt)
            # Local uncommitted edits must not enter source/candidate snapshots.
            self.assertFalse((run / "source/.github/scripts/translation/run.py").exists())
            runner.write_json(run / "cpu-result.json", {"status": "pass"})
            protocol = __import__("json").loads((run / "protocol.json").read_text())
            protocol["protocol_version"] = 1  # Legacy evidence gates remain supported.
            protocol["target_backend_recorded"] = True
            runner.write_json(run / "protocol.json", protocol)
            manifest = __import__("json").loads((run / "manifest.json").read_text())
            self.assertIn("fixed Copilot model", runner.translate(run, "source_only")["error"])
            manifest["translator"]["model"] = "example-fixed-model"
            runner.write_json(run / "manifest.json", manifest)
            blocked = runner.translate(run, "source_validated_dag")
            self.assertEqual(blocked["status"], "blocked")
            self.assertIn("Human graph review", blocked["error"])
            (run / "source_only/prompt.md").write_text("tampered")
            self.assertIn("prompt hash", runner.translate(run, "source_only")["error"])

    def test_shared_evidence_gates_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, site = Path(tmp) / "run", Path(tmp) / "site"
            runner.prepare(run, runner.PROFILE / "manifest.json")
            selected = site / "mlx_lm/models/qwen2.py"
            selected.parent.mkdir(parents=True)
            selected.write_text("q_proj = None\nrope = None\n")
            held_out = site / "vllm_hook_plugins/completed_port.py"
            held_out.parent.mkdir(parents=True)
            held_out.write_text("secret completed implementation")
            manifest = __import__("json").loads((run / "manifest.json").read_text())
            manifest["protocol_version"] = 2
            runner.write_json(run / "manifest.json", manifest)
            runner.platform_evidence.discover(run, site)
            self.assertFalse((run / "platform-discovery/source/vllm_hook_plugins").exists())
            self.assertIn("Shared evidence", runner.translate(run, "source_only", pilot=True)["error"])
            runner.write_json(run / "cpu-result.json", {"status": "pass"})
            runner.write_json(run / "platform-probe-result.json", {"status": "fail"})
            with self.assertRaisesRegex(ValueError, "witnesses must pass"):
                runner.platform_evidence.freeze(run)
            runner.write_json(run / "platform-probe-result.json", {"status": "pass", "limits": "toy only"})
            runner.platform_evidence.freeze(run)
            for condition in runner.CONDITIONS:
                runner.platform_evidence.verify_context(run, condition)
            a = runner.platform_evidence.inventory(run / "source_only/candidate/platform_reference")
            b = runner.platform_evidence.inventory(run / "source_validated_dag/candidate/platform_reference")
            self.assertEqual(a, b)
            (run / "source_only/candidate/platform_reference/README.md").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Candidate shared evidence changed"):
                runner.platform_evidence.verify_context(run, "source_only")
            (run / "platform-discovery/source/mlx_lm/models/qwen2.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Discovery evidence changed"):
                runner.platform_evidence.verify_discovery(run)

    def test_version_three_requires_live_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, site = Path(tmp) / "run", Path(tmp) / "site"
            runner.prepare(run, runner.PROFILE / "manifest.json")
            manifest = __import__("json").loads((run / "manifest.json").read_text())
            manifest["protocol_version"] = 3
            runner.write_json(run / "manifest.json", manifest)
            path = site / "mlx_lm/models/qwen2.py"
            path.parent.mkdir(parents=True)
            path.write_text("class Attention: pass\n")
            runner.platform_evidence.discover(run, site)
            runner.write_json(run / "cpu-result.json", {"status": "pass"})
            runner.write_json(run / "platform-probe-result.json", {"status": "pass", "limits": "toy"})
            with self.assertRaises(FileNotFoundError):
                runner.platform_evidence.freeze(run)
            self.assertFalse((run / "shared-context").exists())
            item = {"source_file": str(path), "source_sha256": runner.platform_evidence.sha(path)}
            worker = {"runner": item, "model": item, "layer": item, "attention_chain": [item]}
            runner.write_json(run / "live-worker-result.json", {"status": "pass", "workers": [worker]})
            runner.platform_evidence.collect_live(run)
            manifest["protocol_version"] = 4
            runner.write_json(run / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "precision and request identity"):
                runner.platform_evidence.freeze(run)
            manifest["protocol_version"] = 3
            runner.write_json(run / "manifest.json", manifest)
            runner.platform_evidence.freeze(run)
            for condition in runner.CONDITIONS:
                runner.platform_evidence.verify_context(run, condition)
                self.assertTrue((run / condition / "candidate/platform_reference/live-platform/source/mlx_lm/models/qwen2.py").exists())
            runner.write_json(run / "live-worker-result.json", {"status": "fail"})
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                runner.platform_evidence.verify_live(run)

    def test_live_source_cannot_include_completed_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, site = Path(tmp) / "run", Path(tmp) / "site"
            run.mkdir(); site.mkdir()
            runner.write_json(run / "platform-discovery/index.json", {"target_site": str(site)})
            heldout = site / "vllm_hook_plugins/completed.py"
            heldout.parent.mkdir()
            heldout.write_text("completed capture")
            item = {"source_file": str(heldout), "source_sha256": runner.platform_evidence.sha(heldout)}
            worker = {"runner": item, "model": item, "layer": item, "attention_chain": [item]}
            runner.write_json(run / "live-worker-result.json", {"status": "pass", "workers": [worker]})
            with self.assertRaisesRegex(ValueError, "outside permitted"):
                runner.platform_evidence.collect_live(run)
            self.assertFalse((run / "live-platform").exists())

    def test_cpu_reuse_rejects_different_fixture_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            current, previous = Path(tmp) / "current", Path(tmp) / "previous"
            current.mkdir(); previous.mkdir()
            protocol = {"source_revision": "revision", "source_archive_sha256": "archive", "evaluator_sha256": "fixture"}
            for directory in (current, previous):
                runner.write_json(directory / "protocol.json", protocol)
            runner.write_json(current / "manifest.json", {"fixture": {"model": "same"}})
            runner.write_json(previous / "cpu-result.json", {"status": "pass", "settings": {"model": "different"}})
            with self.assertRaisesRegex(ValueError, "settings differ"):
                runner.reuse_baseline(current, previous)


if __name__ == "__main__":
    unittest.main()
