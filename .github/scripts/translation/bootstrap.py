"""Prepare pinned target Python sources and Copilot on a fresh hosted runner, without inference installs."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import urllib.request
import zipfile


MAX_ARCHIVE_BYTES = 32 * 1024 * 1024


def fetch(url):
    if not url.startswith('https://'):
        raise ValueError('Source archives require HTTPS')
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError('Source archive exceeds download limit')
    return data


def unpack_python(data, record, destination):
    prefix = PurePosixPath(record['root'])
    if prefix.is_absolute() or '..' in prefix.parts:
        raise ValueError('Invalid archive package root')
    files = {}
    def save(name, read, linked=False):
        path = PurePosixPath(name)
        if not path.is_relative_to(prefix) or not name.endswith('.py'):
            return
        relative = path.relative_to(prefix)
        if linked or path.is_absolute() or '..' in path.parts:
            raise ValueError('Archive source contains an unsafe path or link')
        target = destination / record['package'] / str(relative)
        content = read()
        if len(content) > 1_000_000 or target.exists():
            raise ValueError('Duplicate or oversized Python source file')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[str(relative)] = hashlib.sha256(content).hexdigest()
    raw = io.BytesIO(data)
    if zipfile.is_zipfile(raw):
        with zipfile.ZipFile(raw) as archive:
            for item in archive.infolist():
                if not item.is_dir():
                    save(item.filename, lambda item=item: archive.read(item), (item.external_attr >> 16) & 0o170000 == 0o120000)
    else:
        raw.seek(0)
        with tarfile.open(fileobj=raw, mode='r:*') as archive:
            for item in archive:
                if item.isfile() or item.issym() or item.islnk():
                    save(item.name, lambda item=item: archive.extractfile(item).read(), item.issym() or item.islnk())
    if not files:
        raise ValueError('No Python files found under declared package root')
    return files


def prepare_sources(manifest, output):
    records = manifest.get('target_sources', [])
    packages = manifest['target_packages']
    if len(records) != len(packages) or {r['package'] for r in records} != set(packages):
        raise ValueError('Hosted setup requires one pinned target_sources archive per declared package')
    output.mkdir(parents=True, exist_ok=False)
    sources = output / 'sources'
    receipts = []
    for record in records:
        if not re.fullmatch(r'[A-Za-z_]\w*', record['package']) or not re.fullmatch('[a-f0-9]{64}', record['sha256']):
            raise ValueError('Invalid package name or SHA-256')
        data = fetch(record['url'])
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Source archive checksum mismatch: ' + record['package'])
        files = unpack_python(data, record, sources)
        receipts.append(dict(record, python_files=len(files)))
        print(f'Prepared {record["package"]}: {len(files)} Python files', flush=True)
    (sources / 'source-provenance.json').write_text(json.dumps(receipts, indent=2) + '\n')
    return sources


def install_copilot(manifest, output):
    version = manifest['translator']['cli_version']
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?', version):
        raise ValueError('Copilot CLI must use an exact version')
    prefix = output / 'copilot'
    subprocess.run(['npm', 'install', '--prefix', str(prefix), '--no-audit', '--no-fund', '@github/copilot@' + version], check=True)
    executable = prefix / 'node_modules/.bin/copilot'
    actual = subprocess.check_output([str(executable), '--no-auto-update', '--version'], text=True)
    if version not in actual:
        raise ValueError('Installed Copilot does not match the manifest')
    return executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = args.output.resolve()
    sources = prepare_sources(manifest, output)
    executable = install_copilot(manifest, output)
    home = output / 'copilot-state'
    home.mkdir()
    values = {'TRANSLATION_TARGET_SITE': str(sources), 'TRANSLATION_COPILOT_BIN': str(executable), 'TRANSLATION_COPILOT_HOME': str(home)}
    (output / 'environment.json').write_text(json.dumps(values, indent=2) + '\n')
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'], 'a') as environment:
            for key, value in values.items():
                environment.write(f'{key}={value}\n')
    print('Hosted source context and pinned Copilot CLI are ready; no inference packages were installed.')


if __name__ == '__main__':
    main()
