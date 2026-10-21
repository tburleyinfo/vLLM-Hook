"""Bounded, deterministic Python symbol/call retrieval; not a semantic CPG."""
import ast
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, scope="target"):
    if scope not in ("target", "source"):
        raise ValueError("Unknown graph scope")
    root = Path(root)
    nodes, edges, files, pending = {}, [], {}, []
    structural_facts = []
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root).as_posix()
        if any(part in ('.git', '__pycache__', '.venv') for part in path.relative_to(root).parts):
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Graph source escapes selected corpus')
        if scope == 'target' and relative.split('/')[0] not in ('mlx', 'mlx_lm', 'vllm_metal'):
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
        structural_facts.extend(structural_primitives(tree, module, relative))
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
            owner_scope = owner.split('.')[len(module.split('.')):-1]
            for width in range(len(owner_scope), -1, -1):
                targets.append('.'.join([module, *owner_scope[:width], *parts]))
        target = next((candidate for candidate in targets if candidate in nodes), None)
        if target:
            edges.append({'from': owner, 'to': target, 'kind': 'calls', 'line': line})
        else:
            unresolved.append({'from': owner, 'line': line, 'expression': ast.unparse(expr)})
    # Explicit scope/precision: aliases through objects, dynamic dispatch and runtime
    # replacement are not inferred. A call edge is a syntactic lead, not capability proof.
    return {'schema': 1, 'method': 'Python AST definitions, explicit imports and lexical calls',
            'scope': scope, 'files': files, 'nodes': list(nodes.values()),
            'edges': sorted({json.dumps(e, sort_keys=True) for e in edges}),
            'unresolved_calls': unresolved, 'structural_facts': structural_facts,
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
    if manifest.get('protocol_version', 1) >= 6:
        return write_unseeded(run)
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
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest.get('protocol_version', 1) >= 7:
        source = json.loads((run / 'target-retrieval/source/graph.json').read_text())
        actual_source = {p.relative_to(run / 'source').as_posix(): sha(p)
                         for p in sorted((run / 'source').rglob('*.py'))
                         if not any(part in ('.git', '__pycache__', '.venv') for part in p.relative_to(run / 'source').parts)}
        if actual_source != source['files']:
            raise ValueError('Source graph corpus changed after extraction')
    return graph


def write_graph_bundle(root, destination, scope):
    """Serialize all declared source without application-specific seed selection."""
    graph = build(root, scope=scope)
    destination.mkdir(parents=True)
    (destination / 'graph.json').write_text(json.dumps(graph) + '\n')
    lines = [f'# Static {scope} graph', '', graph['limits'], '',
             'All parsed symbols are indexed. No feature-specific query or seeds.',
             'Read per-file JSON to navigate definitions, lexical calls, structural facts and unresolved calls.',
             'Locations refer to the original source: candidate root for source, platform_reference/source for target.',
             'These are syntax facts, not a translation plan or proof of runtime equivalence.', '']
    edges = decode_edges(graph)
    for name in graph['files']:
        nodes = [n for n in graph['nodes'] if n['file'] == name]
        ids = {n['id'] for n in nodes}
        record = {'nodes': nodes, 'edges': [e for e in edges if e['from'] in ids or e['to'] in ids],
                  'unresolved_calls': [e for e in graph['unresolved_calls'] if e['from'] in ids],
                  'structural_facts': [e for e in graph['structural_facts'] if e['file'] == name]}
        path = destination / 'files' / (name + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + '\n')
        lines.append(f'- files/{name}.json ({len(nodes)} symbols)')
    (destination / 'relationships.md').write_text('\n'.join(lines) + '\n')
    return graph


def write_unseeded(run):
    destination = run / 'target-retrieval'
    graph = write_graph_bundle(run / 'platform-discovery/source', destination, 'target')
    manifest = json.loads((run / 'manifest.json').read_text())
    source = None
    if manifest.get('protocol_version', 1) >= 7:
        source = write_graph_bundle(run / 'source', destination / 'source', 'source')
        readme = destination / 'relationships.md'
        readme.write_text(readme.read_text() + '\nSource repository graph: source/relationships.md and source/files/.\n'
            'Derive any example-specific dependency slices or implementation plans yourself.\n')
    result = {'query': 'all library symbols; unseeded' if source is None else 'all source and target symbols; unseeded',
              'nodes': graph['nodes'] + (source['nodes'] if source else []),
              'edges': decode_edges(graph) + (decode_edges(source) if source else []),
              'limits': graph['limits'], 'corpus_sha256': graph['files']}
    (destination / 'retrieval.json').write_text(json.dumps({'query': result['query'], 'symbols': len(result['nodes']), 'relationships': len(result['edges'])}) + '\n')
    return result


def structural_primitives(tree, module, file):
    """Example-agnostic syntax facts for replacement and callable composition.

    Retaining and invoking a member is a possible wrapper mechanism, not proof
    that a class preserves behavior. Local bindings are lexical leads only.
    """
    facts = []
    def owned(root):
        for child in ast.iter_child_nodes(root):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            yield child
            yield from owned(child)
    def expression(node):
        return ast.unparse(node)
    def walk(items, parent=''):
        for item in items:
            if not isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = '.'.join(filter(None, (parent, item.name)))
            scope = module + '.' + name
            if isinstance(item, ast.ClassDef):
                methods = [m for m in item.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if any(m.name == '__call__' for m in methods):
                    facts.append({'kind': 'callable_class', 'scope': scope, 'file': file, 'line': item.lineno})
                walk(item.body, name)
                continue
            statements = sorted(owned(item), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0)))
            bindings = {}
            for node in statements:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
                value = getattr(node, 'value', None)
                for target in targets:
                    if isinstance(target, ast.Name) and value is not None:
                        bindings[target.id] = {'name': target.id, 'value': expression(value), 'line': node.lineno}
                    elif isinstance(target, ast.Attribute) and value is not None:
                        facts.append({'kind': 'attribute_write', 'scope': scope, 'file': file, 'line': node.lineno,
                            'receiver': expression(target.value), 'attribute': target.attr, 'value': expression(value)})
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id == 'setattr' and len(node.args) == 3:
                    referenced = {n.id for n in ast.walk(node.args[2]) if isinstance(n, ast.Name)}
                    chain = []
                    for _ in range(8):
                        fresh = sorted(referenced & bindings.keys() - {b['name'] for b in chain})
                        if not fresh:
                            break
                        for key in fresh:
                            record = bindings[key]
                            chain.append(record)
                            referenced.update(n.id for n in ast.walk(ast.parse(record['value'], mode='eval')) if isinstance(n, ast.Name))
                    facts.append({'kind': 'attribute_replacement', 'scope': scope, 'file': file, 'line': node.lineno,
                        'receiver': expression(node.args[0]), 'attribute_expression': expression(node.args[1]),
                        'replacement_expression': expression(node.args[2]), 'lexical_bindings': chain,
                        'resolution': 'syntax only; aliases, branches and runtime types unresolved'})
                elif isinstance(node.func, ast.Attribute):
                    receiver = expression(node.func.value)
                    if receiver.startswith('self.'):
                        facts.append({'kind': 'member_invocation', 'scope': scope, 'file': file, 'line': node.lineno,
                                      'receiver': receiver, 'method': node.func.attr})
                    elif isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                        facts.append({'kind': 'member_invocation', 'scope': scope, 'file': file, 'line': node.lineno,
                                      'receiver': 'self', 'method': node.func.attr})
            walk(item.body, name)
    walk(tree.body)
    return facts
