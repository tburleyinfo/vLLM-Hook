"""The one-run action must package drafts independently of inference success."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '.github/scripts/translation'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as runner
from smoke_delivery import generate_fixture


class PipelineTests(unittest.TestCase):
    def test_failed_baseline_and_missing_metal_still_package_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / 'site/mlx_lm'
            site.mkdir(parents=True)
            (site / 'fixture.py').write_text('def identity(value):\n    return value\n')
            run, bundle = root / 'run', root / 'bundle'
            argv = ['run.py', '--stage', 'run', '--condition', 'graph', '--run-dir', str(run),
                    '--target-site', str(root / 'site'), '--pilot', '--metal-python', str(root / 'missing-python'),
                    '--prepare-pr', str(bundle)]
            def failed_baseline(run, python, timeout):
                runner.write_json(run / 'cpu-result.json', {'status': 'fail', 'error': 'Fixture failed'})
            with patch('sys.argv', argv), patch.object(runner, 'baseline', side_effect=failed_baseline), \
                    patch.object(runner, 'translate', side_effect=generate_fixture), \
                    patch.object(runner, 'evaluate', side_effect=AssertionError('No Metal interpreter available')):
                with self.assertRaises(SystemExit) as caught:
                    runner.main()
            self.assertEqual(caught.exception.code, 1)
            metadata = json.loads((bundle / 'metadata.json').read_text())
            self.assertEqual(metadata['condition'], 'graph')
            self.assertEqual(metadata['validation'], 'blocked')
            self.assertIn('**blocked**', (bundle / 'body.md').read_text())
            self.assertTrue((run / 'source_validated_dag/translation.patch').exists())
            self.assertFalse((run / 'source_only/translation-result.json').exists())

    def test_controlled_run_cannot_skip_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'run'
            argv = ['run.py', '--stage', 'run', '--condition', 'no-graph', '--run-dir', str(run), '--skip-tests']
            with patch('sys.argv', argv), self.assertRaises(SystemExit):
                runner.main()
            self.assertFalse(run.exists())

    def test_tracking_registration_precedes_model_and_blocks_on_failure(self):
        with patch.object(runner.tracking, 'register', side_effect=ValueError('W&B unavailable')), \
                patch.object(runner, 'generate_translation') as generate:
            with self.assertRaisesRegex(ValueError, 'W&B unavailable'):
                runner.translate(Path('unused'), 'source_only', True)
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
