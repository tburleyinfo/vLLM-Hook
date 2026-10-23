"""Prevent early publication when implementation or notebook deliverables are missing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '.github/scripts/translation'))
import completion


def notebook(code):
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [
        {'cell_type': 'code', 'source': [code], 'outputs': [], 'execution_count': None, 'metadata': {}}]}


class CompletionTests(unittest.TestCase):
    def test_every_source_notebook_needs_a_new_translated_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, candidate = [Path(temporary) / name for name in ('source', 'candidate')]
            (source / 'notebooks').mkdir(parents=True)
            (source / 'notebooks/demo.ipynb').write_text(json.dumps(notebook('original()')))
            (candidate / 'translation_discovery').mkdir(parents=True)
            (candidate / 'translation_discovery/completion.json').write_text(json.dumps({'status': 'complete', 'remaining_work': []}))
            (candidate / 'TRANSLATION.md').write_text('Usage and implemented behavior')
            self.assertEqual(completion.check(source, candidate)['status'], 'incomplete')
            translated = candidate / 'translated_notebooks/notebooks/demo.ipynb'
            translated.parent.mkdir(parents=True)
            translated.write_text(json.dumps(notebook('original()')))
            self.assertIn('copied unchanged', completion.check(source, candidate)['errors'][0])
            translated.write_text(json.dumps(notebook('translated()')))
            self.assertEqual(completion.check(source, candidate)['status'], 'pass')
            executed = notebook('translated()'); executed['cells'][0]['execution_count'] = 1
            translated.write_text(json.dumps(executed))
            self.assertEqual(completion.check(source, candidate)['status'], 'incomplete')

    def test_remaining_implementation_work_blocks_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'translation_discovery').mkdir()
            path = root / 'translation_discovery/completion.json'
            path.write_text(json.dumps({'status': 'complete', 'remaining_work': ['wire capture']}))
            (root / 'TRANSLATION.md').write_text('Incomplete')
            self.assertEqual(completion.check(root, root)['status'], 'incomplete')
            path.write_text('invalid JSON')
            self.assertEqual(completion.check(root, root)['status'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
