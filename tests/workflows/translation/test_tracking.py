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
                tracking.register(Path('unused'), 'source_only')

    def test_registration_records_hashes_without_prompts_or_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / 'source_only').mkdir()
            manifest = {'source_revision': 'a'*40, 'translator': {'model': 'auto'},
                        'fixture': {'dtype': 'float32', 'prompts': ['private prompt']}, 'tolerance': {'atol': 0.0001}}
            protocol = {'protocol_version': 7, 'prompt_sha256': {'source_only': 'prompt-hash'},
                        'evaluator_sha256': 'evaluator-hash', 'shared_context_files': {'module.py': 'hash'},
                        'condition_context_files': {'source_only': {}}}
            (run / 'manifest.json').write_text(json.dumps(manifest))
            (run / 'protocol.json').write_text(json.dumps(protocol))
            sdk = MagicMock()
            record = sdk.init.return_value
            record.id, record.url, record.summary = 'fixture-id', 'https://wandb.ai/team/project/runs/fixture-id', {}
            env = {'TRANSLATION_REQUIRE_WANDB': 'true', 'WANDB_ENTITY': 'team', 'WANDB_PROJECT': 'project', 'WANDB_API_KEY': 'fixture-key'}
            with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {'wandb': sdk}):
                tracking.register(run, 'source_only')
            config = sdk.init.call_args.kwargs['config']
            self.assertNotIn('private prompt', json.dumps(config))
            self.assertNotIn('fixture-key', json.dumps(config))
            self.assertEqual(config['prompt_sha256'], 'prompt-hash')
            self.assertFalse(config['manual_candidate_repairs'])
            settings = sdk.Settings.call_args.kwargs
            self.assertTrue(settings['disable_git'])
            self.assertTrue(settings['disable_code'])
            self.assertTrue(settings['x_disable_meta'])
            record.finish.assert_called_once()
            self.assertEqual(json.loads((run / 'source_only/tracking.json').read_text())['path'], 'team/project/fixture-id')


if __name__ == '__main__':
    unittest.main()
