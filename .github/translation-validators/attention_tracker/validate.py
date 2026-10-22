"""Optional Attention Tracker comparison using the source API on both repositories."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def compare(source, target, tolerance):
    if source.get('status') != 'pass' or target.get('status') != 'pass':
        return {'status': 'blocked', 'error': 'Both source and translated API executions must pass'}
    for key in ('settings', 'token_ids', 'tokenizer_sha256', 'input_ranges', 'captures', 'capture_config', 'output_count'):
        if source.get(key) != target.get(key):
            return {'status': 'fail', 'error': 'Source/target contract mismatch: ' + key}
    a, b = source.get('scores', []), target.get('scores', [])
    if not a or len(a) != len(b) or not all(math.isfinite(v) for v in a + b):
        return {'status': 'fail', 'error': 'Invalid score data'}
    passed = all(math.isclose(x, y, abs_tol=tolerance['atol'], rel_tol=tolerance['rtol']) for x, y in zip(a, b))
    return {'status': 'pass' if passed else 'fail', 'score_absolute_deltas': [abs(x-y) for x, y in zip(a,b)],
            'claim': 'Selected Attention Tracker fixture only; not whole-repository equivalence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'candidate', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    reports = {}
    for backend, tree, variable in [('cpu', args.source, 'TRANSLATION_SOURCE_PYTHON'), ('target', args.candidate, 'TRANSLATION_TARGET_PYTHON')]:
        python = os.environ.get(variable)
        if not python or not Path(python).is_file():
            reports[backend] = {'status': 'blocked', 'error': variable + ' is unavailable'}
            continue
        output = args.output.parent / (backend + '-result.json')
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', MIA_WRITER_PROCESS='0', VLLM_HOST_IP='127.0.0.1', GLOO_SOCKET_IFNAME='lo0')
        if backend == 'cpu':
            env.update(KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS='1', VLLM_CPU_OMP_THREADS_BIND='nobind',
                       TRITON_DEFAULT_BACKEND='cpu', TRITON_DISABLE_OPENMP='1')
        else:
            env.update(settings.get('target_environment', {}))
        with (args.output.parent / (backend + '-fixture.log')).open('w') as log:
            try:
                process = subprocess.run([python, str(Path(__file__).with_name('fixture.py')), '--backend', backend,
                    '--candidate', str(tree), '--manifest', str(args.config), '--output', str(output)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                report = json.loads(output.read_text()) if output.exists() else {'status': 'blocked', 'error': 'No fixture report'}
                if process.returncode and report.get('status') == 'pass':
                    report = {'status': 'fail', 'error': 'Fixture did not exit cleanly'}
            except subprocess.TimeoutExpired:
                report = {'status': 'blocked', 'error': 'Fixture timed out'}
        reports[backend] = report
    result = compare(reports['cpu'], reports['target'], settings['tolerance'])
    result['executions'] = reports
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
