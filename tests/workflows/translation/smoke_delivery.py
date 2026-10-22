"""Exercise generation-to-draft packaging with a labeled fake translator, offline."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '.github/scripts/translation'))
import run as runner


def generate_fixture(run, condition, pilot):
    """This is plumbing test data, never a translation experiment or a model result."""
    candidate = run / condition / 'candidate'
    (candidate / 'smoke_translation_fixture.py').write_text('# Delivery plumbing fixture; not a real translation.\nVALUE = 1\n')
    notes = candidate / 'translation_discovery'
    notes.mkdir()
    (notes / 'plan.md').write_text('Synthetic delivery fixture; no model was called.\n')
    notebook = candidate / 'notebooks/smoke_delivery.ipynb'
    notebook.parent.mkdir(exist_ok=True)
    notebook.write_text(json.dumps({'cells': [], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}))
    runner.git('add', '-N', '.', cwd=candidate)
    data = runner.candidate_patch(candidate)
    (run / condition / 'translation.patch').write_bytes(data)
    return {'status': 'pass', 'patch_sha256': runner.digest(data), 'fixture_only': True}


def smoke(output):
    output.mkdir(parents=True, exist_ok=False)
    site = output / 'site/mlx_lm'
    site.mkdir(parents=True)
    (site / 'fixture.py').write_text('def caller(value):\n    return callee(value)\ndef callee(value):\n    return value\n')
    for label, condition in [('no-graph', 'source_only'), ('graph', 'source_validated_dag')]:
        root = output / label
        run, bundle = root / 'run', root / 'bundle'
        argv = ['run.py', '--stage', 'run', '--condition', label, '--run-dir', str(run),
                '--target-site', str(output / 'site'), '--pilot', '--skip-tests', '--prepare-pr', str(bundle)]
        with patch('sys.argv', argv), patch.object(runner, 'translate', side_effect=generate_fixture), \
                patch.object(runner, 'baseline', side_effect=AssertionError('Generation-only must not run CPU inference')), \
                patch.object(runner, 'evaluate', side_effect=AssertionError('Generation-only must not run Metal inference')):
            runner.main()
        metadata = json.loads((bundle / 'metadata.json').read_text())
        assert metadata['condition'] == label and metadata['validation'] == 'skipped'
        data = (bundle / 'candidate.patch').read_bytes()
        assert b'smoke_translation_fixture.py' in data and b'smoke_delivery.ipynb' in data
        assert b'translation_discovery' not in data
        delivered = runner.materialize_evaluation_candidate(run, condition)
        assert (delivered / 'smoke_translation_fixture.py').exists()
        assert not (delivered / 'platform_reference').exists()
        assert not (delivered / 'target_relationships').exists()
        assert not (delivered / 'translation_discovery').exists()
        # Make downloaded fixture evidence impossible to mistake for a real port.
        body = bundle / 'body.md'
        body.write_text('**OFFLINE PLUMBING FIXTURE — no model or inference was run.**\n\n' + body.read_text())
        runner.write_json(root / 'report.json', {'condition': label, 'fixture_only': True,
            'model_called': False, 'inference_executed': False, 'validation': 'skipped',
            'selected_patch_only': True, 'clean_patch_materialized': True,
            'patch_sha256': metadata['patch_sha256']})
    print('Both graph modes produced isolated, reviewable fixture bundles.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    smoke(parser.parse_args().output.resolve())
