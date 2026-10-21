"""Delivery isolation checks; no network, model, or GPU is needed."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".github/scripts/translation"))

import publish_candidate as delivery


class DeliveryTests(unittest.TestCase):
    def test_bundle_contains_one_original_patch(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for condition, content in [('source_only', b'ordinary'), ('source_validated_dag', b'graph')]:
                directory = root / condition
                directory.mkdir()
                (directory / 'translation.patch').write_bytes(content)
                (directory / 'translation-result.json').write_text(json.dumps({'status': 'pass', 'patch_sha256': hashlib.sha256(content).hexdigest()}))
            (root / 'protocol.json').write_text('{}')
            (root / 'manifest.json').write_text(json.dumps({'source_revision': 'a' * 40}))
            bundle = root / 'bundle'
            delivery.prepare(root, 'source_only', 'skipped', bundle)
            self.assertEqual((bundle / 'candidate.patch').read_bytes(), b'ordinary')
            self.assertEqual(set(p.name for p in bundle.iterdir()), {'candidate.patch', 'metadata.json', 'body.md'})
            self.assertIn('**skipped**', (bundle / 'body.md').read_text())
            (root / 'source_only/translation.patch').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                delivery.prepare(root, 'source_only', 'pass', bundle)

    def test_publish_uses_separate_index_and_one_patch(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            repo = root / 'repo'
            repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=repo).decode().strip()
            git('init', '-q')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.com')
            (repo / 'application.py').write_text('original\n')
            git('add', '.')
            git('commit', '-qm', 'base')
            base = git('rev-parse', 'HEAD')
            git('remote', 'add', 'origin', 'https://github.com/example/project.git')
            (repo / 'application.py').write_text('candidate\n')
            saved = subprocess.check_output(['git', 'diff', '--binary'], cwd=repo)
            # Unrelated staged and unstaged work must survive publication.
            (repo / 'application.py').write_text('user local work\n')
            (repo / 'other.txt').write_text('unrelated\n')
            git('add', 'other.txt')
            before = git('status', '--porcelain')
            index = (repo / '.git/index').read_bytes()
            bundle = root / 'bundle'
            bundle.mkdir()
            (bundle / 'candidate.patch').write_bytes(saved)
            (bundle / 'metadata.json').write_text(json.dumps({'condition': 'graph', 'patch_sha256': hashlib.sha256(saved).hexdigest()}))
            (bundle / 'body.md').write_text('Validation: skipped')
            actual = delivery.command
            pushed = []
            def fake(args, cwd, env=None):
                if args[:2] == ['git', 'fetch']:
                    (repo / '.git/FETCH_HEAD').write_text(base + '\n')
                    return ''
                if args[:2] == ['git', 'ls-remote']:
                    return ''
                if args[:2] == ['git', 'push']:
                    pushed.append(args[-1].split(':')[0])
                    return ''
                if args[0] == 'gh':
                    return '[]' if args[2] == 'list' else 'https://github.com/example/project/pull/1'
                return actual(args, cwd, env)
            with patch.object(delivery, 'command', side_effect=fake):
                delivery.publish(bundle, 'example/project', 'main', repo)
            self.assertEqual(git('status', '--porcelain'), before)
            self.assertEqual((repo / '.git/index').read_bytes(), index)
            self.assertEqual(git('diff-tree', '--no-commit-id', '--name-only', '-r', pushed[0]), 'application.py')
            self.assertEqual(git('show', pushed[0] + ':application.py'), 'candidate')
            self.assertEqual(git('rev-parse', pushed[0] + '^'), base)
            # A forbidden file is rejected before any push.
            (repo / '.github/workflows').mkdir(parents=True)
            (repo / '.github/workflows/candidate.yml').write_text('bad\n')
            git('add', '-N', '.github/workflows/candidate.yml')
            bad = subprocess.check_output(['git', 'diff', '--binary', '--', '.github/workflows/candidate.yml'], cwd=repo)
            (bundle / 'candidate.patch').write_bytes(bad)
            (bundle / 'metadata.json').write_text(json.dumps({'condition': 'graph', 'patch_sha256': hashlib.sha256(bad).hexdigest()}))
            pushed.clear()
            with patch.object(delivery, 'command', side_effect=fake), self.assertRaises(ValueError):
                delivery.publish(bundle, 'example/project', 'main', repo)
            self.assertFalse(pushed)


class FinishTests(unittest.TestCase):
    def test_optional_tests_and_single_condition(self):
        import run as runner
        for skip in (True, False):
            with self.subTest(skip=skip), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                directory = run / 'source_only'
                directory.mkdir()
                (directory / 'translation-result.json').write_text(json.dumps({'status': 'pass', 'patch_sha256': 'saved'}))
                (directory / 'translation.patch').write_bytes(b'original patch')
                args = ['run.py', '--stage', 'finish', '--condition', 'no-graph', '--run-dir', str(run),
                        '--metal-python', __import__('sys').executable, '--prepare-pr', str(run / 'bundle')]
                if skip:
                    args.append('--skip-tests')
                summary = {'conditions': {'source_only': {'comparison': {'status': 'fail'}}}}
                with patch('sys.argv', args), patch.object(runner.platform_evidence, 'verify_context'), \
                     patch.object(runner, 'git', side_effect=[b'original patch', b'']), \
                     patch.object(runner, 'evaluate') as evaluate, \
                     patch.object(runner, 'summarize', return_value=summary), \
                     patch.object(runner.publish_candidate, 'prepare') as prepare:
                    if skip:
                        runner.main()
                        evaluate.assert_not_called()
                    else:
                        with self.assertRaises(SystemExit):
                            runner.main()
                        evaluate.assert_called_once()
                    self.assertEqual(prepare.call_args.args[1], 'source_only')
                    self.assertEqual(prepare.call_args.args[2], 'skipped' if skip else 'fail')


    def test_evaluate_does_not_report_success_for_failed_parity(self):
        import run as runner
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            args = ['run.py', '--stage', 'evaluate', '--condition', 'no-graph',
                    '--run-dir', temporary, '--metal-python', sys.executable]
            summary = {'conditions': {'source_only': {'comparison': {'status': 'fail'}}}}
            with patch('sys.argv', args), patch.object(runner, 'evaluate'), patch.object(runner, 'summarize', return_value=summary):
                with self.assertRaises(SystemExit) as caught:
                    runner.main()
                self.assertEqual(caught.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
