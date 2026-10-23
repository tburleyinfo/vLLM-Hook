"""Check declared completion and notebook deliverables without claiming runtime parity."""
import json
from pathlib import Path


def notebook_outputs(source):
    return {p.relative_to(source).as_posix(): 'translated_notebooks/' + p.relative_to(source).as_posix()
            for p in sorted(source.rglob('*.ipynb')) if '.git' not in p.relative_to(source).parts}


def check(source, candidate):
    errors = []
    declaration = candidate / 'translation_discovery/completion.json'
    try:
        declared = json.loads(declaration.read_text())
        if not isinstance(declared, dict) or declared.get('status') != 'complete' or declared.get('remaining_work') != []:
            errors.append('Translator did not declare completed implementation with no remaining work')
    except (OSError, ValueError):
        errors.append('Missing or invalid translation_discovery/completion.json')
    if not (candidate / 'TRANSLATION.md').is_file():
        errors.append('Missing TRANSLATION.md')
    notebooks = notebook_outputs(source)
    for original, output in notebooks.items():
        path = candidate / output
        try:
            notebook = json.loads(path.read_text())
            cells = notebook.get('cells', [])
            code = [''.join(c.get('source', [])) for c in cells if c.get('cell_type') == 'code']
            if notebook.get('nbformat') != 4 or not any(s.strip() for s in code):
                raise ValueError('Notebook must have executable code cells')
            if any(c.get('outputs') or c.get('execution_count') is not None for c in cells if c.get('cell_type') == 'code'):
                raise ValueError('Notebook must have cleared outputs and execution counts')
            original_nb = json.loads((source / original).read_text())
            original_code = [''.join(c.get('source', [])) for c in original_nb.get('cells', []) if c.get('cell_type') == 'code']
            if code == original_code:
                raise ValueError('Notebook code was copied unchanged')
        except (OSError, ValueError, AttributeError, TypeError) as error:
            errors.append(f'{output}: {error}')
    return {'status': 'pass' if not errors else 'incomplete', 'errors': errors,
            'notebooks': notebooks, 'scope': 'Translator completion declaration and notebook structure; runtime validation is separate'}
