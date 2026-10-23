"""Provision isolated CPU or Metal validation; never supplied to the translator."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['cpu', 'target'], required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'status': 'blocked', 'stage': 'setup'}
    try:
        environment = args.output.parent / 'runtime'
        python = environment / 'bin/python'
        env = dict(os.environ)
        if args.backend == 'cpu':
            subprocess.run([sys.executable, str(args.repository / 'cross-platform-install/prepare_parity_cpu.py'),
                            '--environment', str(environment)], cwd=args.repository, check=True, env=env)
            env.update(KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS='1', VLLM_CPU_OMP_THREADS_BIND='nobind',
                       TRITON_DEFAULT_BACKEND='cpu', TRITON_DISABLE_OPENMP='1', GLOO_SOCKET_IFNAME='lo',
                       VLLM_PLUGINS='mia_registry,mia')
        else:
            if platform.system() != 'Darwin' or platform.machine() != 'arm64':
                raise RuntimeError('Metal requires an Apple Silicon macOS runner')
            subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
            dependencies = [line.strip() for line in (args.repository / 'requirement.txt').read_text().splitlines()
                            if line.strip() and not line.startswith(('#', 'vllm', 'triton'))]
            subprocess.run([str(python), '-m', 'pip', 'install', 'vllm-metal==0.2.0', *dependencies], check=True)
            subprocess.run([str(python), '-c', 'import mlx.core as mx; assert mx.metal.is_available(), "Metal device unavailable"; mx.eval(mx.ones((1,)))'], check=True)
            subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', '-e', str(args.repository)], check=True)
            env.update(json.loads(args.config.read_text())['target_environment'], GLOO_SOCKET_IFNAME='lo0')
        settings = json.loads(args.config.read_text())['fixture']
        # Fetch exactly the same pinned model on both machines, before offline execution.
        subprocess.run([str(python), '-c', 'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=["*.bin", "*.h5", "*.msgpack"])',
                        settings['model'], settings['model_revision']], check=True, env=env)
        env.update(HF_HUB_OFFLINE='1', MIA_WRITER_PROCESS='0', VLLM_HOST_IP='127.0.0.1')
        result['stage'] = 'fixture'
        process = subprocess.run([str(python), str(Path(__file__).with_name('fixture.py')),
            '--backend', args.backend, '--candidate', str(args.repository), '--manifest', str(args.config),
            '--output', str(args.output)], check=False, env=env, timeout=600)
        result = json.loads(args.output.read_text()) if args.output.exists() else {'status': 'fail', 'error': 'No fixture report'}
        if process.returncode and result.get('status') == 'pass':
            result.update(status='fail', error='Fixture exited unsuccessfully')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result.update(status='blocked', error=str(error))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
