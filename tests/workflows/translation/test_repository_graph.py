"""Static retrieval and paired-context integrity checks; no model execution."""
import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".github/scripts/translation"))

import repository_graph as graph


class GraphTests(unittest.TestCase):
    def test_alias_import_finds_discovery_caller_and_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / 'vllm_metal'
            package.mkdir()
            (package / 'common.py').write_text('def find_layers(model):\n    """Find transformer layers."""\n    return model.layers\n')
            (package / 'patch.py').write_text('from .common import find_layers as locate\nclass AttentionWrapper:\n    def __call__(self, x): return x\ndef install(model):\n    for layer in locate(model):\n        layer.attn = AttentionWrapper()\n')
            index = graph.build(root)
            retrieval = graph.retrieve(index, 'find layers', seed_limit=1)
            ids = {node['id'] for node in retrieval['nodes']}
            self.assertTrue({'vllm_metal.common.find_layers', 'vllm_metal.patch.install', 'vllm_metal.patch.AttentionWrapper'} <= ids)
            self.assertIn({'from': 'vllm_metal.patch.install', 'to': 'vllm_metal.common.find_layers', 'kind': 'calls', 'line': 5}, graph.decode_edges(index))
            self.assertEqual(index, graph.build(root))

    def test_dynamic_calls_are_not_fabricated_as_resolved_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'mlx').mkdir()
            (root / 'mlx/base.py').write_text('def call(obj):\n    return obj.unknown()\n')
            index = graph.build(root)
            self.assertFalse(graph.decode_edges(index))
            self.assertEqual(index['unresolved_calls'][0]['expression'], 'obj.unknown')

    def test_held_out_port_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'mia').mkdir()
            (root / 'mia/port.py').write_text('def done(): pass\n')
            with self.assertRaisesRegex(ValueError, 'target-library'):
                graph.build(root)

    def test_retrieval_rejects_corpus_tampering_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            path = run / 'platform-discovery/source/mlx/base.py'
            path.parent.mkdir(parents=True)
            path.write_text('def find_layers(model): return model.layers\n')
            (run / 'manifest.json').write_text(json.dumps({'retrieval': {'query': 'find layers'}}))
            graph.write_retrieval(run)
            graph.verify(run)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                graph.write_retrieval(run)
            path.write_text('def changed(): pass\n')
            with self.assertRaisesRegex(ValueError, 'corpus changed'):
                graph.verify(run)

class PairedContextTests(unittest.TestCase):
    def test_graph_only_condition_and_context_tampering(self):
        import run as runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = json.loads((runner.PROFILE / 'manifest.json').read_text())
            manifest.update(protocol_version=5, retrieval={'query': 'attention'})
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            trial = root / 'trial'
            runner.prepare(trial, manifest_path)
            site = root / 'site'
            source = site / 'mlx_lm/models/qwen2.py'
            source.parent.mkdir(parents=True)
            source.write_text('class Attention: pass\n')
            runner.platform_evidence.discover(trial, site)
            graph.write_retrieval(trial)
            runner.write_json(trial / 'cpu-result.json', {'status': 'pass'})
            runner.write_json(trial / 'platform-probe-result.json', {'status': 'pass', 'limits': 'unit fixture'})
            record = {'source_file': str(source), 'source_sha256': graph.sha(source)}
            worker = {'runner': record, 'model': record, 'layer': record, 'attention_chain': [record]}
            runner.write_json(trial / 'live-worker-result.json', {'status': 'pass', 'workers': [worker], 'precision_verified': True, 'request_identity_verified': True})
            runner.platform_evidence.collect_live(trial)
            runner.platform_evidence.freeze(trial)
            ordinary = trial / 'source_only/candidate'
            assisted = trial / 'source_validated_dag/candidate'
            self.assertEqual(runner.platform_evidence.inventory(ordinary / 'platform_reference'), runner.platform_evidence.inventory(assisted / 'platform_reference'))
            self.assertFalse((ordinary / 'target_relationships').exists())
            self.assertTrue((assisted / 'target_relationships/relationships.md').exists())
            a = (trial / 'source_only/prompt.md').read_text()
            b = (trial / 'source_validated_dag/prompt.md').read_text()
            self.assertTrue(b.startswith(a))
            self.assertNotIn('Source implementation dependency graph:', b)
            for condition in runner.CONDITIONS:
                runner.platform_evidence.verify_context(trial, condition)
            (assisted / 'target_relationships/relationships.md').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'graph context changed'):
                runner.platform_evidence.verify_context(trial, 'source_validated_dag')


if __name__ == '__main__':
    unittest.main()
