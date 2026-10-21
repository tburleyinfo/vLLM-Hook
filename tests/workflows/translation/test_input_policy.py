"""Front-end parity and graph-only input policy, independent of example details."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '.github/scripts/translation'))
import run as runner
import repository_graph


class InputPolicyTests(unittest.TestCase):
    def manifest(self):
        return json.loads((runner.PROFILE / 'manifest.discovery.json').read_text())

    def test_shared_inputs_and_generated_analysis_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(self.manifest()))
            run = root / 'run'
            runner.prepare(run, manifest)
            target = root / 'site/mlx_lm/example.py'
            target.parent.mkdir(parents=True)
            target.write_text('def outer(value):\n    return inner(value)\ndef inner(value):\n    return value\n')
            runner.platform_evidence.discover(run, root / 'site')
            repository_graph.write_retrieval(run)
            runner.write_json(run / 'cpu-result.json', {'status': 'pass'})
            runner.platform_evidence.freeze(run)
            ordinary = run / 'source_only/candidate'
            assisted = run / 'source_validated_dag/candidate'
            self.assertEqual(runner.platform_evidence.inventory(ordinary / 'platform_reference'), runner.platform_evidence.inventory(assisted / 'platform_reference'))
            self.assertFalse((ordinary / 'target_relationships').exists())
            self.assertTrue((assisted / 'target_relationships/source/graph.json').exists())
            source_graph = json.loads((assisted / 'target_relationships/source/graph.json').read_text())
            self.assertTrue(source_graph['nodes'])
            self.assertEqual(source_graph['scope'], 'source')
            source_files = {p.relative_to(run / 'source').as_posix() for p in (run / 'source').rglob('*.py')}
            self.assertEqual(set(source_graph['files']), source_files)
            for condition in runner.CONDITIONS:
                runner.platform_evidence.verify_context(run, condition)
                prompt = (run / condition / 'prompt.md').read_text()
                self.assertIn('Both conditions may generate intermediate graphs', prompt)
                self.assertNotIn('post-positional-transform', prompt)
                self.assertNotIn('Reconstruct required Q/K', prompt)
                candidate = run / condition / 'candidate'
                derived = candidate / 'translation_discovery'
                derived.mkdir()
                (derived / 'my-example-graph.json').write_text('{"generated": true}')
                (candidate / 'generated_port.py').write_text('VALUE = 1\n')
                subprocess.run(['git', 'add', '-N', '.'], cwd=candidate, check=True)
                patch = runner.candidate_patch(candidate).decode()
                self.assertIn('generated_port.py', patch)
                self.assertNotIn('my-example-graph.json', patch)
                runner.platform_evidence.verify_context(run, condition)
            policy = json.loads((run / 'protocol.json').read_text())['input_policy']
            self.assertFalse(policy['handwritten_example_recipe_supplied'])
            self.assertTrue(policy['generated_example_specific_graphs_and_plans_allowed'])
            source = run / 'source/mia/artifacts.py'
            source.write_text(source.read_text() + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'Source graph corpus changed'):
                repository_graph.verify(run)

    def test_front_end_has_no_implementation_steps_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest()
            manifest['translation_task']['implementation_steps'] = ['Handwritten example recipe']
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'no implementation recipe'):
                runner.prepare(root / 'run', path)

    def test_project_graph_is_opt_in_and_does_not_read_other_trees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'app.py').write_text('def start(value):\n    return finish(value)\ndef finish(value):\n    return value\n')
            with self.assertRaisesRegex(ValueError, 'target-library'):
                repository_graph.build(root)
            graph = repository_graph.build(root, scope='source')
            self.assertTrue(any(e['to'] == 'app.finish' for e in repository_graph.decode_edges(graph)))

    def test_evaluation_materializes_only_the_saved_delivery_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            source = run / 'source'
            source.mkdir()
            (source / 'app.py').write_text('VALUE = 1\n')
            directory = run / 'source_only'
            candidate = directory / 'candidate'
            import shutil
            shutil.copytree(source, candidate)
            subprocess.run(['git', 'init', '-q'], cwd=candidate, check=True)
            subprocess.run(['git', 'add', '.'], cwd=candidate, check=True)
            subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.com', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'baseline'], cwd=candidate, check=True)
            # Supplied reference content is part of the candidate baseline only.
            for name in ['platform_reference', 'target_relationships']:
                (candidate / name).mkdir()
                (candidate / name / 'input.json').write_text('{}')
            subprocess.run(['git', 'add', '.'], cwd=candidate, check=True)
            subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.com', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'references'], cwd=candidate, check=True)
            (candidate / 'app.py').write_text('VALUE = 2\n')
            (candidate / 'translation_discovery').mkdir()
            (candidate / 'translation_discovery/plan.md').write_text('Generated plan')
            subprocess.run(['git', 'add', '-N', '.'], cwd=candidate, check=True)
            patch = runner.candidate_patch(candidate)
            (directory / 'translation.patch').write_bytes(patch)
            runner.write_json(directory / 'translation-result.json', {'patch_sha256': runner.digest(patch)})
            delivered = runner.materialize_evaluation_candidate(run, 'source_only')
            self.assertEqual((delivered / 'app.py').read_text(), 'VALUE = 2\n')
            for name in ['platform_reference', 'target_relationships', 'translation_discovery']:
                self.assertFalse((delivered / name).exists())
            self.assertTrue((candidate / 'translation_discovery/plan.md').exists())
            (candidate / 'app.py').write_text('VALUE = 3\n')
            with self.assertRaisesRegex(ValueError, 'unchanged saved'):
                runner.materialize_evaluation_candidate(run, 'source_only')
