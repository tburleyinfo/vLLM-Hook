"""Run an independent profile on each backend, then gate publication on comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import io


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def selected_profile(profile):
    profile = profile.resolve()
    root = Path(__file__).resolve().parents[2] / 'translation-validators'
    if not profile.is_relative_to(root) or not profile.is_file():
        raise ValueError('Hosted profiles must be in .github/translation-validators')
    definition = json.loads(profile.read_text())
    paths = {}
    for name in ('script', 'config'):
        path = (profile.parent / definition[name]).resolve()
        if not path.is_relative_to(profile.parent) or not path.is_file():
            raise ValueError('Profile files must stay within the validator directory')
        paths[name] = path
    evidence = {}
    for path in sorted(profile.parent.rglob('*')):
        if path.is_symlink():
            raise ValueError('Validator files must not be symlinks')
        if path.is_file() and '__pycache__' not in path.parts:
            evidence[path.relative_to(profile.parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    paths['sha256'] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return paths


def execute(bundle, profile, backend, output):
    selected = selected_profile(profile)
    metadata = json.loads((bundle / 'metadata.json').read_text())
    patch = (bundle / 'candidate.patch').read_bytes()
    if hashlib.sha256(patch).hexdigest() != metadata['patch_sha256']:
        raise ValueError('Changed candidate patch')
    output.mkdir(parents=True, exist_ok=True)
    report = output / 'result.json'
    try:
        tree = output / 'repository'
        tree.mkdir()
        # Source is pinned independently of the publishing branch.
        archive = subprocess.check_output(['git', 'archive', metadata['source_revision']])
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(tree, filter='data')
        if backend == 'target':
            subprocess.run(['git', 'apply', str((bundle / 'candidate.patch').resolve())], cwd=tree, check=True)
        script = profile.resolve().with_name('hosted.py')
        if not script.is_file():
            raise ValueError('Selected profile has no hosted.py backend runner')
        with (output / 'execution.log').open('w') as log:
            process = subprocess.run([sys.executable, str(script), '--backend', backend,
                '--repository', str(tree.resolve()), '--config', str(selected['config']),
                '--output', str(report.resolve())], stdout=log, stderr=subprocess.STDOUT, timeout=6600)
        result = json.loads(report.read_text()) if report.exists() else {'status': 'blocked', 'error': 'Backend produced no report'}
        if process.returncode and result.get('status') == 'pass':
            result = {'status': 'fail', 'error': 'Backend exited unsuccessfully'}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result = {'status': 'blocked', 'error': str(error)}
    result.update(backend=backend, patch_sha256=metadata['patch_sha256'], source_revision=metadata['source_revision'], validator_sha256=selected['sha256'])
    write(report, result)
    return result


def finalize(bundle, profile, cpu, target, output):
    selected = selected_profile(profile)
    metadata = json.loads((bundle / 'metadata.json').read_text())
    if not cpu.is_file() or not target.is_file():
        result = {'status': 'blocked', 'error': 'Missing backend runtime evidence'}
        write(output, result)
        return result
    source, translated = json.loads(cpu.read_text()), json.loads(target.read_text())
    for backend, report in (('cpu', source), ('target', translated)):
        if report.get('backend') != backend:
            raise ValueError('Runtime evidence has the wrong backend')
        if (report.get('patch_sha256') != metadata['patch_sha256'] or report.get('source_revision') != metadata['source_revision']
                or report.get('validator_sha256') != selected['sha256']):
            raise ValueError('Runtime evidence belongs to a different candidate')
    # The compare implementation is trusted profile code, never generated code.
    import importlib.util
    script = selected['script']
    spec = importlib.util.spec_from_file_location('selected_runtime_validator', script)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    settings = json.loads(selected['config'].read_text())
    result = validator.compare(source, translated, settings['tolerance'])
    result.update(patch_sha256=metadata['patch_sha256'], validator_sha256=selected['sha256'], executions={'cpu': source, 'target': translated})
    write(output, result)
    if result['status'] != 'pass':
        return result
    metadata['validation'] = 'pass'
    metadata['runtime_validation_required'] = True
    write(bundle / 'metadata.json', metadata)
    write(bundle / 'review-files/runtime-validation.json', result)
    experiment = json.loads((bundle / 'review-files/experiment.json').read_text())
    experiment['validation'] = result
    write(bundle / 'review-files/experiment.json', experiment)
    body = bundle / 'body.md'
    body.write_text(body.read_text().replace('Validation: **skipped**', 'Validation: **pass**'))
    summary = bundle / 'review-files/README.md'
    summary.write_text(summary.read_text().replace('Validation: **skipped**', 'Validation: **pass**') +
                       '\nCPU/Metal comparison passed within the selected profile; see runtime-validation.json.\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu', 'target'])
    parser.add_argument('--cpu', type=Path)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.bundle, args.profile, args.backend, args.output) if args.backend else finalize(
            args.bundle, args.profile, args.cpu, args.target, args.output)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result = {'status': 'blocked', 'error': str(error)}
        write(args.output / 'result.json' if args.backend else args.output, result)
    raise SystemExit(0 if result['status'] == 'pass' else 1)
