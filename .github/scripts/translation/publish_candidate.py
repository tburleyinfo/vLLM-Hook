"""Publish exactly one saved translation patch, without touching the checkout index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import hosted_runtime


def command(args, cwd, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env).decode().strip()


def attach_validation(review, profile, metadata, patch):
    selected = hosted_runtime.selected_profile(profile)
    profile = profile.resolve()
    kit = review / 'validation'
    shutil.copytree(profile.parent, kit / 'profile',
                    ignore=shutil.ignore_patterns('.*', '__pycache__'))
    attached_profile = kit / 'profile' / profile.name
    if hosted_runtime.selected_profile(attached_profile, root=kit / 'profile')['sha256'] != selected['sha256']:
        raise ValueError('Attached validator differs from the selected profile')
    shutil.copyfile(Path(hosted_runtime.__file__), kit / 'hosted_runtime.py')
    (kit / 'candidate.patch').write_bytes(patch)
    (kit / 'metadata.json').write_text(json.dumps({
        **metadata, 'validator_sha256': selected['sha256']}, indent=2) + '\n')
    path = f'translation-experiments/{metadata["condition"]}-{metadata["patch_sha256"]}/validation'
    (kit / 'README.md').write_text(
        '# Run validation manually\n\n'
        'Use Python 3.12 and run from the root of a checkout of this PR. The clone must contain '
        f'the pinned source commit `{metadata["source_revision"]}`. The runner verifies the attached '
        'patch and profile and reconstructs the candidate from that commit. Use a new output '
        'directory for each execution.\n\n'
        '```bash\n'
        f'KIT="{path}"\n'
        f'PROFILE="$KIT/profile/{profile.name}"\n'
        'RESULTS="$(mktemp -d)"\n'
        '```\n\n'
        'On Linux, run the CPU reference. On Apple Silicon macOS, run the Metal candidate. '
        'Set the variables above on each machine.\n\n'
        '```bash\n'
        'python3.12 "$KIT/hosted_runtime.py" --bundle "$KIT" --profile "$PROFILE" --backend cpu --output "$RESULTS/cpu"\n'
        'python3.12 "$KIT/hosted_runtime.py" --bundle "$KIT" --profile "$PROFILE" --backend target --output "$RESULTS/target"\n'
        '```\n\n'
        'Copy the CPU and Metal result folders to the same machine and compare them. '
        'If you already have a CPU report for the exact same patch, source and profile, you can reuse it.\n\n'
        '```bash\n'
        'python3.12 "$KIT/hosted_runtime.py" --bundle "$KIT" --profile "$PROFILE" '
        '--cpu "$RESULTS/cpu/result.json" --target "$RESULTS/target/result.json" '
        '--output "$RESULTS/comparison.json"\n'
        '```\n\n'
        'Keep both backend reports and `comparison.json` with the experiment record.\n')
    return path


def prepare(run, condition, validation, output, *, for_runtime=False, validator_profile=None):
    directory = run / condition
    report = json.loads((directory / 'translation-result.json').read_text())
    patch = (directory / 'translation.patch').read_bytes()
    sha = hashlib.sha256(patch).hexdigest()
    if validation not in ('pass', 'skipped') and not (for_runtime and validation == 'pending'):
        raise ValueError('Failed or blocked validation cannot be published')
    completion = json.loads((directory / 'completion-result.json').read_text())
    if completion.get('status') not in ('pass', 'incomplete'):
        raise ValueError('A deliverable status is required')
    if report.get('status') != 'pass' or not patch or sha != report.get('patch_sha256'):
        raise ValueError('Successful translation and unchanged saved patch required')
    protocol = json.loads((run / 'protocol.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    (output / 'candidate.patch').write_bytes(patch)
    label = {'no_graph': 'no-graph', 'graph': 'graph'}[condition]
    metadata = {'condition': label, 'patch_sha256': sha,
                'source_revision': manifest['source_revision'], 'validation': validation,
                'pilot': report.get('experimental_status') != 'fixed-model candidate generation', 'completion': completion,
                'experiment_id': protocol.get('experiment_id', ''), 'use_case': protocol.get('use_case', ''),
                'runtime_validation_required': for_runtime}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    receipt = directory / 'tracking.json'
    if for_runtime and receipt.is_file():
        (output / 'tracking.json').write_bytes(receipt.read_bytes())
    review = output / 'review-files'
    review.mkdir(exist_ok=True)
    validation_path = attach_validation(review, validator_profile, metadata, patch) if validator_profile else None
    status = f'Deliverables: **{completion["status"]}**. Validation: **{validation}**.'
    remaining = completion.get('remaining_work') or completion.get('errors', [])
    handoff = ('\n\nRemaining work:\n' + '\n'.join(f'- {item}' for item in remaining)) if remaining else ''
    summary = {
        'condition': label, 'source_revision': manifest['source_revision'],
        'experiment_id': metadata['experiment_id'], 'use_case': metadata['use_case'],
        'patch_sha256': sha, 'generation': report, 'completion': completion,
        'validation': json.loads((directory / 'validation-result.json').read_text())
            if (directory / 'validation-result.json').exists() else {'status': validation},
        'runtime_correctness': 'Not established by generation; see validation scope.',
        'graph_advantage': 'Not established',
    }
    (review / 'experiment.json').write_text(json.dumps(summary, indent=2) + '\n')
    (review / 'README.md').write_text(
        f'# Translation experiment ({label})\n\n'
        f'Generation: **{report["status"]}**. {status}{handoff}\n\n'
        'Generation success does not establish runtime correctness. '
        'Skipped validation is not a passing functional test.\n\n'
        'Implementation and usage notes are in the PR files. '
        'experiment.json records the original patch hash and results. '
        'discovery/ contains translator-generated research documents, when present.\n')
    for name in ('manifest.json', 'protocol.json'):
        (review / name).write_bytes((run / name).read_bytes())
    if (directory / 'prompt.md').exists():
        (review / 'prompt.md').write_bytes((directory / 'prompt.md').read_bytes())
    discovery = directory / 'candidate/translation_discovery'
    if discovery.exists():
        for path in discovery.rglob('*'):
            if path.is_symlink():
                raise ValueError('Discovery documents must not be symlinks')
            if path.is_file():
                target = review / 'discovery' / path.relative_to(discovery)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    (output / 'body.md').write_text(
        f'Repository translation using **{label}**. {status}{handoff}\n\n'
        'Implementation, usage notes and experiment evidence are included in the changed files.\n'
        + (f'\nManual CPU/Metal test instructions: `{validation_path}/README.md`.\n' if validation_path else ''))



def publish(bundle, repo, base, checkout, artifact_url=None):
    metadata = json.loads((bundle / 'metadata.json').read_text())
    patch = bundle / 'candidate.patch'
    sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    if metadata.get('runtime_validation_required') and metadata.get('validation') != 'pass':
        raise ValueError('CPU/Metal runtime validation must pass before publication')
    if metadata.get('completion', {}).get('status') not in ('pass', 'incomplete') or metadata.get('validation') not in ('pass', 'skipped'):
        raise ValueError('Draft publication requires a deliverable status and no failed validation')
    if not patch or sha != metadata['patch_sha256'] or metadata['condition'] not in ('graph', 'no-graph'):
        raise ValueError('Invalid publication bundle')
    body = (bundle / 'body.md').read_text()
    if artifact_url:
        body += f'\n[Raw execution evidence]({artifact_url}) (optional).\n'
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
        protected = {'.git', '.github', 'platform_reference', 'target_relationships',
                     'translation_discovery', 'translation-experiments'}
        if not any(paths) or any(Path(path).parts[0] in protected for path in paths if path):
            raise ValueError('Patch changes workflow infrastructure, supplied evidence or internal discovery artifacts')
        deleted = command(['git', 'diff', '--cached', '--no-renames', '--diff-filter=D', '--name-only', parent], checkout, env)
        if deleted:
            raise ValueError('Translation must preserve existing files: ' + deleted)
        review = bundle / 'review-files'
        if review.exists():
            prefix = f'translation-experiments/{metadata["condition"]}-{sha}'
            for path in sorted(review.rglob('*')):
                if path.is_symlink():
                    raise ValueError('Review files must not be symlinks')
                if path.is_file():
                    destination = prefix + '/' + path.relative_to(review).as_posix()
                    if command(['git', 'ls-files', '--', destination], checkout, env):
                        raise ValueError('Experiment evidence would overwrite an existing file')
                    blob = command(['git', 'hash-object', '-w', str(path.resolve())], checkout)
                    command(['git', 'update-index', '--add', '--cacheinfo', '100644', blob, destination], checkout, env)
        tree = command(['git', 'write-tree'], checkout, env)
        title = f'Repository translation ({metadata["condition"]})'
        branch = f'translation/result-{metadata["condition"]}-{hashlib.sha256(base.encode()).hexdigest()[:10]}'
        body_file = Path(temporary) / 'body.md'
        body_file.write_text(body)
        existing = command(['gh', 'pr', 'list', '--repo', repo, '--head', branch, '--state', 'open', '--json', 'url,isDraft'], checkout)
        matches = json.loads(existing)
        if matches and not matches[0].get('isDraft'):
            raise ValueError('Existing translation PR is ready for review; refusing automatic replacement')
        remote = command(['git', 'ls-remote', '--heads', 'origin', branch], checkout)
        previous = remote.split()[0] if remote else ''
        if previous:
            command(['git', 'fetch', '--no-tags', 'origin', branch], checkout)
            if command(['git', 'show', '-s', '--format=%ae', previous], checkout) != env['GIT_AUTHOR_EMAIL']:
                raise ValueError('Refusing to replace a publication branch modified by another author')
            if matches and command(['git', 'rev-parse', previous + '^{tree}'], checkout) == tree:
                command(['gh', 'pr', 'edit', matches[0]['url'], '--repo', repo, '--body-file', str(body_file)], checkout)
                print(matches[0]['url'])
                return
        commit = command(['git', 'commit-tree', tree, '-p', parent, '-m', title], checkout, env)
        command(['git', 'push', f'--force-with-lease=refs/heads/{branch}:{previous}',
                 'origin', f'{commit}:refs/heads/{branch}'], checkout)
        if matches:
            url = matches[0]['url']
            command(['gh', 'pr', 'edit', url, '--repo', repo, '--body-file', str(body_file)], checkout)
            print(url)
            return
        print(command(['gh', 'pr', 'create', '--repo', repo, '--base', base, '--head', branch,
                       '--draft', '--title', title, '--body-file', str(body_file)], checkout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--base', default='main')
    parser.add_argument('--artifact-url', help='Evidence artifact URL to link from the draft')
    args = parser.parse_args()
    publish(args.bundle, args.repo, args.base, Path.cwd(), args.artifact_url)
