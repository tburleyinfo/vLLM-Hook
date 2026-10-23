"""Generic generation boundaries with a small repository unrelated to MIA or Attention Tracker."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '.github/scripts/translation'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as runner
import repository_graph
from smoke_delivery import generate_fixture


class GenericWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=self.repo).decode().strip()
        git('init', '-q')
        git('config', 'user.name', 'Test')
        git('config', 'user.email', 'test@example.com')
        (self.repo / 'application.py').write_text('def start(value):\n    return finish(value)\ndef finish(value):\n    return value\n')
        # Even when committed alongside source, hidden validators and workflow tools must be withheld.
        for name in ('.github/translation-validators/secret/fixture.py', '.github/scripts/translation/internal.py',
                     '.github/translation/private.json', 'tests/workflows/translation/internal.py'):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('PRIVATE_VALIDATOR_SENTINEL = True\n')
        git('add', '.')
        git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'Generic source')
        self.manifest = json.loads((runner.PROFILE / 'manifest.json').read_text())
        self.manifest.update(source_revision=git('rev-parse', 'HEAD'), target_packages=['runtime_library'], platform_notes=[])
        self.manifest['translation_task'] = {'source_backend': 'Platform A', 'target_backend': 'Platform B', 'requirements': ['Preserve the repository API.']}
        self.manifest_path = self.root / 'manifest.json'
        runner.write_json(self.manifest_path, self.manifest)
        self.site = self.root / 'site'
        target = self.site / 'runtime_library/modules.py'
        target.parent.mkdir(parents=True)
        target.write_text('def invoke(value):\n    return transform(value)\ndef transform(value):\n    return value\n')
        self.repo_patch = patch.object(runner, 'REPO', self.repo)
        self.repo_patch.start()
        self.addCleanup(self.repo_patch.stop)

    def prepared(self, name='trial'):
        run = self.root / name
        runner.prepare(run, self.manifest_path)
        runner.platform_evidence.discover(run, self.site)
        repository_graph.write_retrieval(run)
        runner.platform_evidence.freeze(run)
        return run

    def invoke(self, name, validator=None, expect_failure=False, generator=generate_fixture):
        run, bundle = self.root / name, self.root / (name + '-bundle')
        args = ['run.py', '--stage', 'run', '--run-dir', str(run), '--manifest', str(self.manifest_path),
                '--target-site', str(self.site), '--condition', 'graph', '--pilot', '--prepare-pr', str(bundle)]
        if validator:
            args += ['--validator-profile', str(validator)]
        with patch('sys.argv', args), patch.object(runner, 'translate', side_effect=generator):
            if expect_failure:
                with self.assertRaises(SystemExit) as caught:
                    runner.main()
                self.assertEqual(caught.exception.code, 1)
            else:
                runner.main()
        return run, bundle

    def validator(self, status='fail', exit_code=1):
        directory = self.root / 'validator'
        directory.mkdir()
        runner.write_json(directory / 'profile.json', {'script': 'check.py', 'config': 'settings.json'})
        runner.write_json(directory / 'settings.json', {'secret': 'EVALUATOR_ONLY_MARKER'})
        (directory / 'check.py').write_text(
            "import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser()\n"
            "for k in ('source','candidate','config','output'):p.add_argument('--'+k,required=True)\n"
            "a=p.parse_args()\nassert (Path(a.candidate)/'smoke_translation_fixture.py').exists()\n"
            "assert not (Path(a.candidate)/'platform_reference').exists()\n"
            "assert not (Path(a.candidate)/'translation_discovery').exists()\n"
            f"Path(a.output).write_text(json.dumps({{'status':{status!r},'reason':'selected validator'}}))\nraise SystemExit({exit_code})\n")
        return directory / 'profile.json'

    def test_generic_source_and_graph_only_difference(self):
        run = self.prepared()
        ordinary = run / 'no_graph/candidate'
        graph = run / 'graph/candidate'
        self.assertEqual(runner.platform_evidence.inventory(ordinary / 'platform_reference'), runner.platform_evidence.inventory(graph / 'platform_reference'))
        self.assertFalse((ordinary / 'target_relationships').exists())
        self.assertTrue((graph / 'target_relationships/source/graph.json').exists())
        a, b = [(run / condition / 'prompt.md').read_text() for condition in runner.CONDITIONS]
        self.assertTrue(b.startswith(a))
        for text in (a,b):
            for forbidden in ('attention_tracker_metal', 'MetalAttentionTracker', 'Qwen', 'important_heads', 'fixture settings', 'PRIVATE_VALIDATOR_SENTINEL'):
                self.assertNotIn(forbidden, text)
        for path in (run / 'source').rglob('*'):
            if path.is_file():self.assertNotIn('PRIVATE_VALIDATOR_SENTINEL', path.read_text())
        self.assertEqual(set(json.loads((run / 'target-retrieval/source/graph.json').read_text())['files']), {'application.py'})
        self.assertFalse((run / 'fixture.py').exists())
        self.assertFalse((run / 'source_dag.mmd').exists())
        self.assertFalse((run / 'dependencies.md').exists())

    def test_example_configuration_is_rejected(self):
        for extra in ('fixture', 'tolerance', 'validation_profile'):
            with self.subTest(extra=extra):
                manifest = dict(self.manifest, **{extra: {}})
                runner.write_json(self.manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, 'example settings'):
                    runner.prepare(self.root / extra, self.manifest_path)
        self.manifest['translation_task']['validation_interface'] = {'class': 'ExampleAdapter'}
        runner.write_json(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, 'repository behavior requirements'):
            runner.prepare(self.root / 'interface', self.manifest_path)

    def test_default_generation_produces_draft_without_validator(self):
        with patch.object(runner, 'evaluate', side_effect=AssertionError('No validator selected')):
            run, bundle = self.invoke('default')
        self.assertEqual(json.loads((bundle / 'metadata.json').read_text())['validation'], 'skipped')
        delivered = runner.materialize_evaluation_candidate(run, 'graph')
        self.assertTrue((delivered / 'smoke_translation_fixture.py').exists())
        self.assertFalse((delivered / 'translation_discovery').exists())
        self.assertNotIn(b'translation_discovery', (bundle / 'candidate.patch').read_bytes())

    def test_incomplete_implementation_keeps_evidence_without_draft(self):
        def incomplete(run, condition, pilot):
            report = generate_fixture(run, condition, pilot)
            runner.write_json(run / condition / 'candidate/translation_discovery/completion.json',
                              {'status': 'incomplete', 'remaining_work': ['Wire backend into public API']})
            return report
        run, bundle = self.invoke('incomplete', expect_failure=True, generator=incomplete)
        self.assertFalse(bundle.exists())
        self.assertTrue((run / 'graph/translation.patch').is_file())
        self.assertEqual(json.loads((run / 'graph/completion-result.json').read_text())['status'], 'incomplete')

    def test_failed_optional_validator_preserves_inputs_but_prevents_draft(self):
        without, _ = self.invoke('without')
        profile = self.validator()
        with_validator, bundle = self.invoke('with', profile, expect_failure=True)
        for condition in runner.CONDITIONS:
            self.assertEqual((without / condition / 'prompt.md').read_bytes(), (with_validator / condition / 'prompt.md').read_bytes())
            self.assertNotIn('EVALUATOR_ONLY_MARKER', (with_validator / condition / 'prompt.md').read_text())
            self.assertFalse((with_validator / condition / 'candidate/validation-profile').exists())
        self.assertFalse(bundle.exists())
        evidence = json.loads((with_validator / 'graph/validation-evidence.json').read_text())
        self.assertFalse(evidence['supplied_to_translator'])
        self.assertTrue((with_validator / 'graph/translation.patch').is_file())

    def test_validator_success_with_nonzero_exit_is_failure(self):
        run, bundle = self.invoke('bad-exit', self.validator('pass', 2), expect_failure=True)
        self.assertFalse(bundle.exists())
        self.assertIn('exited unsuccessfully', (run / 'graph/validation-result.json').read_text())

    def test_missing_validator_is_blocked_without_draft(self):
        run, bundle = self.invoke('missing', self.root / 'unavailable-validator/profile.json', expect_failure=True)
        self.assertFalse(bundle.exists())

    def test_original_validation_is_not_overwritten(self):
        run, _ = self.invoke('immutable-validation', self.validator(), expect_failure=True)
        output = run / 'graph/validation-result.json'
        before = output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'already recorded'):
            runner.evaluate(run, 'graph', self.root / 'validator/profile.json', sys.executable, 30)
        self.assertEqual(before, output.read_bytes())

    def test_changed_original_candidate_cannot_be_delivered(self):
        run, _ = self.invoke('tamper')
        (run / 'graph/candidate/smoke_translation_fixture.py').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'unchanged saved'):
            runner.materialize_evaluation_candidate(run, 'graph')

    def test_changed_source_or_graph_context_is_rejected(self):
        run = self.prepared()
        path = run / 'graph/candidate/target_relationships/relationships.md'
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'graph context changed'):
            runner.platform_evidence.verify_context(run, 'graph')

    def test_arbitrary_target_package_and_lexical_edges(self):
        graph = repository_graph.build(self.site)
        self.assertIn({'from': 'runtime_library.modules.invoke', 'to': 'runtime_library.modules.transform', 'kind': 'calls', 'line': 2}, repository_graph.decode_edges(graph))

    def test_actual_generation_wrapper_saves_original_patch(self):
        run = self.prepared()
        def fake_command(argv, candidate, log, **kwargs):
            self.assertIn('--deny-tool=shell', argv)
            self.assertIn('--disable-builtin-mcps', argv)
            (candidate / 'generated.py').write_text('VALUE = 2\n')
            log.write_text(json.dumps({'type':'assistant.message','data':{'content':'Generated generic repository port','model':'fixture-model'}})+'\n')
            return {'status':'pass','exit_code':0,'seconds':1.0}
        original = subprocess.check_output
        def output(argv, **kwargs):
            if argv[0] == 'copilot':return 'GitHub Copilot CLI 1.0.97'
            return original(argv, **kwargs)
        with patch.dict('os.environ', {'TRANSLATION_COPILOT_BIN':'copilot','TRANSLATION_REQUIRE_WANDB':'false'}), \
                patch.object(runner, 'command', side_effect=fake_command), \
                patch.object(subprocess, 'check_output', side_effect=output):
            report = runner.translate(run, 'no_graph', True)
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(report['resolved_models'], ['fixture-model'])
        self.assertIn(b'generated.py', (run / 'no_graph/translation.patch').read_bytes())

    def test_custom_validator_cannot_leak_through_source_location(self):
        profile = self.repo / "tests/example_validator/profile.json"
        profile.parent.mkdir(parents=True)
        runner.write_json(profile, {"script": "check.py", "config": "settings.json"})
        with self.assertRaisesRegex(ValueError, "excluded from translation inputs"):
            runner.check_validator_location(profile, self.root / "run")
        with self.assertRaisesRegex(ValueError, "outside the generated run"):
            runner.check_validator_location(self.root / "run/profile.json", self.root / "run")

    def test_malformed_optional_validator_prevents_draft(self):
        profile = self.validator()
        runner.write_json(profile, {"script": ["bad"], "config": "settings.json"})
        run, bundle = self.invoke("malformed", profile, expect_failure=True)
        self.assertFalse(bundle.exists())

    def test_tracking_failure_prevents_model_usage(self):
        with patch.object(runner.tracking, 'register', side_effect=ValueError('Tracking unavailable')), \
                patch.object(runner, 'generate_translation') as generate:
            with self.assertRaises(ValueError):runner.translate(self.root, 'no_graph', True)
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
