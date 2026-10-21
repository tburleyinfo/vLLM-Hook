"""Ensure unassisted inputs contain libraries, not a supplied implementation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '.github/scripts/translation'))
import run as runner


class UnassistedTests(unittest.TestCase):
    def test_only_raw_libraries_and_unseeded_graph_are_supplied(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / 'site'
            source = site / 'mlx_lm/models/example.py'
            source.parent.mkdir(parents=True)
            source.write_text('def alpha(x):\n    return beta(x)\ndef beta(x):\n    return x\n')
            manifest = json.loads((runner.PROFILE / 'manifest.unassisted.json').read_text())
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            run = root / 'run'
            runner.prepare(run, path)
            runner.platform_evidence.discover(run, site)
            graph = runner.repository_graph.write_retrieval(run)
            self.assertEqual(graph['query'], 'all library symbols; unseeded')
            runner.write_json(run / 'cpu-result.json', {'status': 'pass'})
            runner.platform_evidence.freeze(run)
            for condition in runner.CONDITIONS:
                runner.platform_evidence.verify_context(run, condition)
                context = run / condition / 'candidate/platform_reference'
                self.assertEqual(set(runner.platform_evidence.inventory(context)), {'README.md', 'HOOK_CORRESPONDENCE.md', 'source/mlx_lm/models/example.py'})
                prompt = (run / condition / 'prompt.md').read_text()
                for supplied_recipe in ('post-positional-transform', 'find_layers', 'live_precision.py', 'request_identity.py', 'capture_metadata.py', 'Reconstruct required Q/K'):
                    self.assertNotIn(supplied_recipe, prompt)
            ordinary = run / 'source_only/candidate'
            assisted = run / 'source_validated_dag/candidate'
            self.assertFalse((ordinary / 'target_relationships').exists())
            self.assertTrue((assisted / 'target_relationships/files/mlx_lm/models/example.py.json').exists())
            self.assertFalse((run / 'shared-context/live-worker-result.json').exists())

    def test_wrapper_primitive_is_extracted_without_example_names(self):
        import repository_graph as graph
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = root / 'mlx/generic.py'
            p.parent.mkdir()
            p.write_text('class Proxy:\n    def __init__(self, original):\n        self.inner = original\n    def __call__(self, *args, **kwargs):\n        return self.inner(*args, **kwargs)\ndef install(owner, name):\n    original = getattr(owner, name)\n    replacement = Proxy(original)\n    setattr(owner, name, replacement)\n')
            facts = graph.build(root)['structural_facts']
            replacement = next(f for f in facts if f['kind'] == 'attribute_replacement')
            self.assertEqual(replacement['receiver'], 'owner')
            self.assertEqual({b['name'] for b in replacement['lexical_bindings']}, {'replacement', 'original'})
            self.assertTrue(any(f['kind'] == 'attribute_write' and f['attribute'] == 'inner' for f in facts))
            self.assertTrue(any(f['kind'] == 'member_invocation' and f['method'] == 'inner' for f in facts))
            self.assertFalse(any('attention' in json.dumps(f).lower() for f in facts))
