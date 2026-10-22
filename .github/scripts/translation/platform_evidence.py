"""Freeze declared library source and optional platform notes; no example-specific logic."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import repository_graph


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def inventory(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Reference and source inputs must not contain symlinks')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha(path)
    return files


def discover(run, site):
    destination = run / 'platform-discovery'
    if destination.exists():
        raise ValueError('Discovery already exists; prepare a new run')
    manifest = json.loads((run / 'manifest.json').read_text())
    packages = manifest['target_packages']
    site = Path(site).resolve()
    if not packages or not site.is_dir():
        raise ValueError('Declare target packages and provide their source directory')
    for package in packages:
        if not re.fullmatch(r'[A-Za-z_]\w*', package):
            raise ValueError('Target packages must be top-level Python package names')
        root = site / package
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f'Declared target package is missing or symlinked: {package}')
        for source in sorted(root.rglob('*.py')):
            if not source.resolve().is_relative_to(site) or source.is_symlink() or source.stat().st_size > 1_000_000:
                raise ValueError('Target source is outside the declared corpus or exceeds bounds')
            target = destination / 'source' / source.relative_to(site)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    files = inventory(destination / 'source')
    if not files:
        raise ValueError('No Python source found in declared target packages')
    save(destination / 'index.json', {'files': files, 'packages': packages})
    if (site / 'source-provenance.json').is_file():
        shutil.copyfile(site / 'source-provenance.json', run / 'target-source-provenance.json')


def verify_discovery(run):
    expected = json.loads((run / 'platform-discovery/index.json').read_text())['files']
    if inventory(run / 'platform-discovery/source') != expected:
        raise ValueError('Target source changed after discovery')


def freeze(run):
    verify_discovery(run)
    repository_graph.verify(run)
    context = run / 'shared-context'
    if context.exists():
        raise ValueError('Shared context already frozen')
    shutil.copytree(run / 'platform-discovery/source', context / 'source')
    if (run / 'platform-notes').exists():
        shutil.copytree(run / 'platform-notes', context / 'notes')
    (context / 'README.md').write_text('Declared target-library Python sources and platform notes.\n'
        'Discover repository integration details yourself. No validation profile or example recipe is supplied.\n')
    protocol = json.loads((run / 'protocol.json').read_text())
    protocol['shared_context_files'] = inventory(context)
    protocol['graph_files'] = inventory(run / 'target-retrieval')
    protocol['condition_context_files'] = {}
    for condition in ('no_graph', 'graph'):
        directory = run / condition
        candidate = directory / 'candidate'
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=candidate).strip():
            raise ValueError('Cannot freeze context into a changed candidate')
        shutil.copytree(context, candidate / 'platform_reference')
        prompt = (directory / 'prompt.md').read_text()
        prompt += '\nTarget libraries: platform_reference/source/. Platform notes, if supplied: platform_reference/notes/.\n'
        names = ['platform_reference']
        if condition == 'graph':
            shutil.copytree(run / 'target-retrieval', candidate / 'target_relationships')
            names.append('target_relationships')
            prompt += '\nSupplied graphs: target_relationships/relationships.md and target_relationships/source/relationships.md.\n'
        protocol['condition_context_files'][condition] = inventory(candidate / 'target_relationships')
        subprocess.run(['git', 'add', *names], cwd=candidate, check=True)
        subprocess.run(['git', '-c', 'user.name=Translation', '-c', 'user.email=translation@localhost',
                        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Frozen library context'], cwd=candidate, check=True)
        (directory / 'prompt.md').write_text(prompt)
        protocol['prompt_sha256'][condition] = hashlib.sha256(prompt.encode()).hexdigest()
    save(run / 'protocol.json', protocol)


def verify_context(run, condition):
    protocol = json.loads((run / 'protocol.json').read_text())
    if inventory(run / 'source') != protocol['source_files']:
        raise ValueError('Frozen source repository changed')
    verify_discovery(run)
    repository_graph.verify(run)
    expected = protocol['shared_context_files']
    if inventory(run / 'shared-context') != expected or inventory(run / condition / 'candidate/platform_reference') != expected:
        raise ValueError('Shared reference context changed')
    if inventory(run / 'target-retrieval') != protocol['graph_files']:
        raise ValueError('Supplied graphs changed')
    if inventory(run / condition / 'candidate/target_relationships') != protocol['condition_context_files'][condition]:
        raise ValueError('Candidate graph context changed')
