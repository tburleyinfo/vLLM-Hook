import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import status


class VerboseStatusTests(unittest.TestCase):
    def test_waiting_explains_missing_cli_and_does_not_start_anything(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(status.shutil, 'which', return_value=None):
            run = Path(tmp)
            (run/'manifest.json').write_text(json.dumps({'translator': {'model': 'auto'}}))
            result = status.overview(run)
            self.assertIn('WAITING TO START', result)
            self.assertIn('Copilot CLI not found', result)
            self.assertIn('--pilot', result)
            self.assertEqual(len(list(run.iterdir())), 1)

    def test_old_log_is_not_reported_as_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); log = run/'cpu.log'; log.write_text('loading model')
            os.utime(log, (1, 1))
            result = status.describe(run, 'cpu-result.json', 'cpu.log', True)
            self.assertIn('process state unknown', result)
            self.assertNotIn('running;', result)
            self.assertIn('13 bytes', result)

    def test_verbose_result_shows_duration_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run/'cpu-result.json').write_text(json.dumps({'status': 'pass', 'seconds': 12.5}))
            result = status.describe(run, 'cpu-result.json', 'cpu.log', True)
            self.assertIn('seconds: 12.5', result)
            self.assertIn('cpu-result.json', result)

    def test_partial_result_does_not_crash_watcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); (run/'cpu-result.json').write_text('{')
            self.assertIn('being written', status.describe(run, 'cpu-result.json', 'cpu.log'))
