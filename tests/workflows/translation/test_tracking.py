"""Check required registration and metadata boundaries without contacting W&B."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '.github/scripts/translation'))
import tracking


class TrackingTests(unittest.TestCase):
    def test_missing_tracking_credentials_fail_before_sdk_or_model(self):
        with patch.dict(os.environ, {'TRANSLATION_REQUIRE_WANDB': 'true'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'required before translation'):
                tracking.register(Path('unused'), 'no_graph')

    def test_registration_records_hashes_without_prompts_or_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / 'no_graph').mkdir()
            manifest = {'source_revision': 'a'*40, 'translator': {'model': 'auto'},
                        'translation_task': {'source_backend': 'A', 'target_backend': 'B', 'requirements': []}, 'target_packages': ['runtime']}
            protocol = {'schema_version': 1, 'prompt_sha256': {'no_graph': 'prompt-hash'},
                        'shared_context_files': {'module.py': 'hash'},
                        'condition_context_files': {'no_graph': {}}}
            (run / 'manifest.json').write_text(json.dumps(manifest))
            (run / 'protocol.json').write_text(json.dumps(protocol))
            sdk = MagicMock()
            record = sdk.init.return_value
            record.id, record.url, record.summary = 'fixture-id', 'https://wandb.ai/team/project/runs/fixture-id', {}
            env = {'TRANSLATION_REQUIRE_WANDB': 'true', 'WANDB_ENTITY': 'team', 'WANDB_PROJECT': 'project', 'WANDB_API_KEY': 'fixture-key'}
            with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {'wandb': sdk}):
                tracking.register(run, 'no_graph')
            config = sdk.init.call_args.kwargs['config']
            self.assertNotIn('fixture', config)
            self.assertNotIn('tolerance', config)
            self.assertNotIn('fixture-key', json.dumps(config))
            self.assertEqual(config['prompt_sha256'], 'prompt-hash')
            self.assertFalse(config['manual_candidate_repairs'])
            settings = sdk.Settings.call_args.kwargs
            self.assertTrue(settings['disable_git'])
            self.assertTrue(settings['disable_code'])
            self.assertTrue(settings['x_disable_meta'])
            record.finish.assert_called_once()
            self.assertEqual(json.loads((run / 'no_graph/tracking.json').read_text())['path'], 'team/project/fixture-id')


if __name__ == '__main__':
    unittest.main()
