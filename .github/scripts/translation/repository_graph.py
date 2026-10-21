"""Bounded, deterministic Python symbol/call retrieval; not a semantic CPG."""
import ast
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    root = Path(root)
    nodes, edges, files, pending = {}, [], {}, []
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root).as_posix()
        if relative.split('/')[0] not in ('mlx', 'mlx_lm', 'vllm_metal'):
            raise ValueError('Graph corpus must contain target-library code only')
        source = path.read_text()
        tree = ast.parse(source, filename=relative)
        module = relative[:-3].replace('/', '.')
        files[relative] = sha(path)
        imports = {}
        for item in tree.body:
            if isinstance(item, ast.ImportFrom):
                prefix = module.split('.')[:-1]
                base = '.'.join(prefix[:len(prefix) - item.level + 1]) if item.level else ''
                base = '.'.join(filter(None, (base, item.module)))
                for alias in item.names:
                    imports[alias.asname or alias.name] = base + '.' + alias.name
            elif isinstance(item, ast.Import):
                for alias in item.names:
                    imports[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]

        def visit(items, parent=''):
            for item in items:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                name = '.'.join(filter(None, (parent, item.name)))
                key = module + '.' + name
                nodes[key] = {'id': key, 'file': relative, 'line': item.lineno,
                              'end_line': item.end_lineno, 'kind': type(item).__name__,
                              'name': item.name, 'doc': ast.get_docstring(item) or ''}
                if parent:
                    edges.append({'from': module + '.' + parent, 'to': key, 'kind': 'defines'})
                # Calls in nested definitions belong to those definitions, not their parent.
                def owned(expr):
                    for child in ast.iter_child_nodes(expr):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                            continue
                        if isinstance(child, ast.Call):
                            pending.append((key, module, parent, imports, child.func, child.lineno))
                        yield from owned(child)
                list(owned(item))
                visit(item.body, name)
        visit(tree.body)
    unresolved = []
    for owner, module, parent, imports, expr, line in pending:
        parts = []
        current = expr
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
        else:
            unresolved.append({'from': owner, 'line': line, 'expression': ast.unparse(expr)})
            continue
        first, *tail = parts
        targets = []
        if first in imports:
            targets.append('.'.join([imports[first], *tail]))
        if first in ('self', 'cls') and parent:
            targets.append('.'.join([module, parent.split('.')[0], *tail]))
        else:
            scope = owner.split('.')[len(module.split('.')):-1]
            for width in range(len(scope), -1, -1):
                targets.append('.'.join([module, *scope[:width], *parts]))
        target = next((candidate for candidate in targets if candidate in nodes), None)
        if target:
            edges.append({'from': owner, 'to': target, 'kind': 'calls', 'line': line})
        else:
            unresolved.append({'from': owner, 'line': line, 'expression': ast.unparse(expr)})
    # Explicit scope/precision: aliases through objects, dynamic dispatch and runtime
    # replacement are not inferred. A call edge is a syntactic lead, not capability proof.
    return {'schema': 1, 'method': 'Python AST definitions, explicit imports and lexical calls',
            'files': files, 'nodes': list(nodes.values()),
            'edges': sorted({json.dumps(e, sort_keys=True) for e in edges}),
            'unresolved_calls': unresolved,
            'limits': 'Not a full CPG. No dataflow, runtime dispatch, re-export or alias inference. No claim of hook correctness.'}


def decode_edges(graph):
    return [json.loads(e) if isinstance(e, str) else e for e in graph['edges']]


def retrieve(graph, query, seed_limit=1, depth=2, node_limit=30):
    words = set(re.findall(r'[a-z]+', query.lower()))
    stop = {'the', 'a', 'to', 'and', 'how', 'in', 'of', 'use'}
    words -= stop
    if not words:
        raise ValueError('Provide a nonempty retrieval query')
    nodes = {n['id']: n for n in graph['nodes']}
    def score(node):
        # Symbol names are deliberately weighted above generic prose.
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', node['name']).lower()
        return sum(4 for word in words if word in text) + sum(1 for word in words if word in node['doc'].lower())
    ranked = sorted(nodes, key=lambda key: (-score(nodes[key]), key))
    seeds = [key for key in ranked if score(nodes[key]) > 0][:seed_limit]
    edges = decode_edges(graph)
    selected, frontier = set(seeds), set(seeds)
    for _ in range(depth):
        candidates = set()
        for edge in edges:
            if edge['from'] in frontier:
                candidates.add(edge['to'])
            if edge['to'] in frontier:
                candidates.add(edge['from'])
        frontier = set(sorted(candidates - selected, key=lambda key: (-score(nodes[key]), key))[:max(0, node_limit - len(selected))])
        selected.update(frontier)
    return {'query': query, 'seeds': seeds, 'depth': depth, 'node_limit': node_limit,
            'nodes': [nodes[key] for key in sorted(selected)],
            'edges': [e for e in edges if e['from'] in selected and e['to'] in selected],
            'limits': graph['limits']}


def write_retrieval(run):
    run = Path(run)
    destination = run / 'target-retrieval'
    if destination.exists():
        raise ValueError('Retrieval already exists; prepare a new run')
    manifest = json.loads((run / 'manifest.json').read_text())
    query = manifest['retrieval']['query']
    root = run / 'platform-discovery/source'
    graph = build(root)
    result = retrieve(graph, query, **{key: manifest["retrieval"][key] for key in ("seed_limit", "depth", "node_limit") if key in manifest["retrieval"]})
    if not result['seeds']:
        raise ValueError('No target-library symbols match the query')
    destination.mkdir()
    (destination / 'graph.json').write_text(json.dumps(graph, indent=2) + '\n')
    (destination / 'retrieval.json').write_text(json.dumps(result, indent=2) + '\n')
    # Edges/locations only: both conditions already receive the identical full corpus.
    lines = ['# Target-library search relationships', '', 'Query: ' + query,
             '', graph['limits'], '', '```mermaid', 'flowchart TD']
    ids = {n['id']: 'n' + str(i) for i, n in enumerate(result['nodes'])}
    for node in result['nodes']:
        label = (node['id'] + ' ' + node['file'] + ':' + str(node['line'])).replace('"', "'")
        lines.append(f'  {ids[node["id"]]}["{label}"]')
    for edge in result['edges']:
        lines.append(f'  {ids[edge["from"]]} -->|{edge["kind"]}| {ids[edge["to"]]}')
    lines += ['```', '', '## Source locations', '']
    lines += [f'- {n["id"]}: platform_reference/source/{n["file"]}:{n["line"]}-{n["end_line"]}' for n in result['nodes']]
    (destination / 'relationships.md').write_text('\n'.join(lines) + '\n')
    result['corpus_sha256'] = graph['files']
    return result


def verify(run):
    run = Path(run)
    graph = json.loads((run / 'target-retrieval/graph.json').read_text())
    actual = {p.relative_to(run / 'platform-discovery/source').as_posix(): sha(p)
              for p in sorted((run / 'platform-discovery/source').rglob('*.py'))}
    if actual != graph['files']:
        raise ValueError('Graph corpus changed after retrieval')
    return graph
