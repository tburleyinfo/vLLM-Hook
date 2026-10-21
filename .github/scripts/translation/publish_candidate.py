"""Publish exactly one saved translation patch, without touching the checkout index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def command(args, cwd, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env).decode().strip()


def prepare(run, condition, validation, output):
    directory = run / condition
    report = json.loads((directory / 'translation-result.json').read_text())
    patch = (directory / 'translation.patch').read_bytes()
    sha = hashlib.sha256(patch).hexdigest()
    if report.get('status') != 'pass' or not patch or sha != report.get('patch_sha256'):
        raise ValueError('Successful translation and unchanged saved patch required')
    protocol = json.loads((run / 'protocol.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    (output / 'candidate.patch').write_bytes(patch)
    label = {'source_only': 'no-graph', 'source_validated_dag': 'graph'}[condition]
    metadata = {'condition': label, 'patch_sha256': sha,
                'source_revision': manifest['source_revision'], 'validation': validation,
                'pilot': protocol.get('pilot', True)}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'body.md').write_text(
        f'Translate Attention Tracker to Metal using **{label}**.\n\n'
        f'Validation: **{validation}**. Skipped validation does not establish parity.\n\n'
        'Only this candidate\'s saved changes are included. Experiment logs and the other candidate are excluded.\n\n'
        'Notebook cells have not been executed end to end. A passing fixture supports only its tested scope; '
        'this pilot does not establish a graph advantage.\n\n'
        f'Source: `{manifest["source_revision"]}`\n\nPatch SHA-256: `{sha}`\n')


def publish(bundle, repo, base, checkout):
    metadata = json.loads((bundle / 'metadata.json').read_text())
    patch = bundle / 'candidate.patch'
    sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    if sha != metadata['patch_sha256'] or metadata['condition'] not in ('graph', 'no-graph'):
        raise ValueError('Invalid publication bundle')
    origin = command(['git', 'remote', 'get-url', 'origin'], checkout)
    if origin.removesuffix('.git').rstrip('/') not in (f'https://github.com/{repo}', f'git@github.com:{repo}'):
        raise ValueError('Destination must match origin')
    command(['git', 'check-ref-format', 'refs/heads/' + base], checkout)
    command(['git', 'fetch', '--no-tags', 'origin', base], checkout)
    parent = command(['git', 'rev-parse', 'FETCH_HEAD'], checkout)
    with tempfile.TemporaryDirectory() as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'),
                   GIT_AUTHOR_NAME='github-actions[bot]', GIT_AUTHOR_EMAIL='41898282+github-actions[bot]@users.noreply.github.com',
                   GIT_COMMITTER_NAME='github-actions[bot]', GIT_COMMITTER_EMAIL='41898282+github-actions[bot]@users.noreply.github.com')
        command(['git', 'read-tree', parent], checkout, env)
        command(['git', 'apply', '--cached', '--check', str(patch.resolve())], checkout, env)
        command(['git', 'apply', '--cached', str(patch.resolve())], checkout, env)
        paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z', parent], cwd=checkout, env=env).decode().split('\0')
        protected = {'.git', '.github', 'platform_reference', 'target_relationships', 'translation_discovery'}
        if not any(paths) or any(Path(path).parts[0] in protected for path in paths if path):
            raise ValueError('Patch changes workflow infrastructure, supplied evidence or internal discovery artifacts')
        tree = command(['git', 'write-tree'], checkout, env)
        title = f'Attention Tracker Metal translation ({metadata["condition"]})'
        branch = f'translation/{metadata["condition"]}-{parent[:10]}-{sha[:12]}'
        existing = command(['gh', 'pr', 'list', '--repo', repo, '--head', branch, '--state', 'open', '--json', 'url'], checkout)
        if json.loads(existing):
            print(json.loads(existing)[0]['url'])
            return
        remote = command(['git', 'ls-remote', '--heads', 'origin', branch], checkout)
        if remote:
            command(['git', 'fetch', '--no-tags', 'origin', branch], checkout)
            commit = command(['git', 'rev-parse', 'FETCH_HEAD'], checkout)
            if (command(['git', 'rev-parse', commit + '^{tree}'], checkout) != tree or
                    command(['git', 'rev-parse', commit + '^'], checkout) != parent):
                raise ValueError('Existing publication branch differs; refusing to overwrite it')
        else:
            commit = command(['git', 'commit-tree', tree, '-p', parent, '-m', title], checkout, env)
        command(['git', 'push', 'origin', f'{commit}:refs/heads/{branch}'], checkout)
        print(command(['gh', 'pr', 'create', '--repo', repo, '--base', base, '--head', branch,
                       '--draft', '--title', title, '--body-file', str((bundle / 'body.md').resolve())], checkout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--base', default='main')
    args = parser.parse_args()
    publish(args.bundle, args.repo, args.base, Path.cwd())
