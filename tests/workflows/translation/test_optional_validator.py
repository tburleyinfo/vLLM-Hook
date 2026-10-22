"""Optional example validation can run without becoming a generation dependency."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = ROOT / '.github/translation-validators/attention_tracker/validate.py'
spec = importlib.util.spec_from_file_location('optional_validator', VALIDATOR)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class OptionalValidatorTests(unittest.TestCase):
    def report(self, score=0.5):
        return {'status':'pass','settings':{},'token_ids':[[1]],'tokenizer_sha256':'same',
                'input_ranges':[[[0,1]]],'captures':[],'capture_config':{},'output_count':1,'scores':[score]}

    def test_optional_numeric_and_contract_checks_remain_separate(self):
        self.assertEqual(validator.compare(self.report(), self.report(0.50001), {'atol':1e-4,'rtol':1e-3})['status'], 'pass')
        self.assertEqual(validator.compare(self.report(), self.report(0.8), {'atol':1e-4,'rtol':1e-3})['status'], 'fail')
        target = self.report();target['capture_config']={'different':True}
        self.assertEqual(validator.compare(self.report(), target, {'atol':1e-4,'rtol':1e-3})['status'], 'fail')

    def test_missing_optional_environments_blocks_without_importing_inference_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'result.json'
            env = dict(os.environ)
            env.pop('TRANSLATION_SOURCE_PYTHON', None);env.pop('TRANSLATION_TARGET_PYTHON', None)
            result = subprocess.run([sys.executable, str(VALIDATOR), '--source', temporary, '--candidate', temporary,
                '--config', str(VALIDATOR.with_name('settings.json')), '--output', str(output)], env=env, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(output.read_text())['status'], 'blocked')

    def test_shared_manifest_has_no_optional_example_reference(self):
        manifest = json.loads((ROOT / '.github/translation/manifest.json').read_text())
        self.assertNotIn('validator_profile', manifest)
        self.assertNotIn('fixture', manifest)
        self.assertNotIn('attention_tracker', json.dumps(manifest))
        self.assertNotIn('Attention Tracker', (ROOT / '.github/scripts/translation/publish_candidate.py').read_text())


if __name__ == '__main__':
    unittest.main()
