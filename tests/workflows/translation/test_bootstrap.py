"""Hosted setup downloads verified source files, without importing or installing target runtimes."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '.github/scripts/translation'))
import bootstrap


class BootstrapTests(unittest.TestCase):
    def archive(self, files):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w:gz') as archive:
            for name, value in files.items():
                data = value.encode();info = tarfile.TarInfo(name);info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return raw.getvalue()

    def test_pinned_tar_sources_only_without_runtime_import(self):
        data = self.archive({'release/library/module.py': 'raise RuntimeError("must not import")',
                             'release/library/native.so': 'binary', 'release/setup.py': 'build code'})
        record = {'package':'runtime','url':'https://example.invalid/source.tar.gz','root':'release/library',
                  'sha256':hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary, patch.object(bootstrap, 'fetch', return_value=data):
            site = bootstrap.prepare_sources({'target_packages':['runtime'],'target_sources':[record]}, Path(temporary)/'bootstrap')
            self.assertEqual([p.name for p in (site/'runtime').iterdir()], ['module.py'])
            self.assertEqual(json.loads((site/'source-provenance.json').read_text())[0]['sha256'], record['sha256'])

    def test_cross_platform_wheel_extracts_python_without_loading_native_code(self):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw,'w') as archive:
            archive.writestr('library/nn.py','VALUE = 1\n')
            archive.writestr('library/core.cpython-312-darwin.so','not executable')
            archive.writestr('library/core.pyi','stub')
        with tempfile.TemporaryDirectory() as temporary:
            files = bootstrap.unpack_python(raw.getvalue(), {'root':'library','package':'library'}, Path(temporary))
            self.assertEqual(list(files), ['nn.py'])
            self.assertEqual(list((Path(temporary)/'library').iterdir()), [Path(temporary)/'library/nn.py'])

    def test_checksum_mismatch_stops_before_unpack(self):
        record={'package':'runtime','url':'https://example.invalid/source','root':'library','sha256':'0'*64}
        with tempfile.TemporaryDirectory() as temporary, patch.object(bootstrap,'fetch',return_value=b'changed'):
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                bootstrap.prepare_sources({'target_packages':['runtime'],'target_sources':[record]},Path(temporary)/'bootstrap')
            self.assertFalse((Path(temporary)/'bootstrap/sources').exists())

    def test_path_escape_is_rejected(self):
        data=self.archive({'library/../../outside.py':'bad'})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'unsafe path'):
                bootstrap.unpack_python(data,{'root':'library','package':'runtime'},Path(temporary))

    def test_every_declared_package_needs_one_locked_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'one pinned'):
                bootstrap.prepare_sources({'target_packages':['runtime'],'target_sources':[]},Path(temporary)/'bootstrap')

    def test_installs_exact_copilot_version_and_checks_it(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(bootstrap.subprocess,'run') as install, \
                patch.object(bootstrap.subprocess,'check_output',return_value='GitHub Copilot CLI 1.0.97'):
            executable=bootstrap.install_copilot({'translator':{'cli_version':'1.0.97'}},Path(temporary))
            self.assertIn('@github/copilot@1.0.97',install.call_args.args[0])
            self.assertEqual(executable,Path(temporary)/'copilot/node_modules/.bin/copilot')


if __name__=='__main__':unittest.main()
