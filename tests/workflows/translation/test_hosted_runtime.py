"""CPU/Metal reports gate publication, including failed and missing evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '.github/scripts/translation'))
import hosted_runtime
import publish_candidate

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / '.github/translation-validators/attention_tracker/profile.json'


class HostedRuntimeTests(unittest.TestCase):
    def fixture(self, root, target_status='pass'):
        bundle = root / 'bundle'
        (bundle / 'review-files').mkdir(parents=True)
        metadata = {'patch_sha256': 'candidate', 'source_revision': 'source',
                    'completion': {'status': 'pass'}, 'validation': 'skipped', 'runtime_validation_required': True}
        (bundle / 'metadata.json').write_text(json.dumps(metadata))
        (bundle / 'body.md').write_text('Validation: **skipped**')
        (bundle / 'review-files/README.md').write_text('Validation: **skipped**')
        (bundle / 'review-files/experiment.json').write_text('{}')
        report = {'validator_sha256': hosted_runtime.selected_profile(PROFILE)['sha256'], 'patch_sha256': 'candidate', 'source_revision': 'source', 'status': 'pass',
                  'settings': {}, 'token_ids': [[1]], 'tokenizer_sha256': 'same',
                  'input_ranges': [], 'captures': [], 'capture_config': {}, 'output_count': 1, 'scores': [0.5]}
        cpu, target = root / 'cpu.json', root / 'target.json'
        report['backend'] = 'cpu'
        cpu.write_text(json.dumps(report))
        report['backend'] = 'target'
        report['status'] = target_status
        target.write_text(json.dumps(report))
        return bundle, cpu, target

    def test_only_matching_passed_runtime_evidence_authorizes_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, cpu, target = self.fixture(root)
            result = hosted_runtime.finalize(bundle, PROFILE, cpu, target, root / 'comparison.json')
            self.assertEqual(result['status'], 'pass')
            self.assertEqual(json.loads((bundle / 'metadata.json').read_text())['validation'], 'pass')
            self.assertTrue((bundle / 'review-files/runtime-validation.json').exists())
            self.assertIn('**pass**', (bundle / 'body.md').read_text())
            report = json.loads(target.read_text()); report['patch_sha256'] = 'other'
            target.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'different candidate'):
                hosted_runtime.finalize(bundle, PROFILE, cpu, target, root / 'comparison.json')

    def test_blocked_or_missing_runtime_never_authorizes_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, cpu, target = self.fixture(root, 'blocked')
            for missing in (False, True):
                if missing: target.unlink()
                result = hosted_runtime.finalize(bundle, PROFILE, cpu, target, root / 'comparison.json')
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(json.loads((bundle / 'metadata.json').read_text())['validation'], 'skipped')
                self.assertFalse((bundle / 'review-files/runtime-validation.json').exists())
            (bundle / 'candidate.patch').write_text('unused')
            with self.assertRaisesRegex(ValueError, 'must pass'):
                publish_candidate.publish(bundle, 'example/project', 'main', root)


if __name__ == '__main__':
    unittest.main()
