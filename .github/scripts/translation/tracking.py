"""Optional W&B metadata tracking; candidate source and raw logs stay in Actions artifacts."""
import hashlib
import json
import os


def register(run, condition):
    if os.environ.get('TRANSLATION_REQUIRE_WANDB') != 'true':
        return
    entity, project = os.environ.get('WANDB_ENTITY'), os.environ.get('WANDB_PROJECT')
    if not entity or not project or not os.environ.get('WANDB_API_KEY'):
        raise ValueError('W&B entity, project and API key are required before translation')
    import wandb
    manifest = json.loads((run / 'manifest.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    identity = os.environ.get('GITHUB_RUN_ID', run.name) + '-' + os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    settings = wandb.Settings(disable_git=True, disable_code=True, save_code=False,
        x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
        console='off', silent=True, root_dir=str(run), program='repository-translation',
        program_relpath='repository-translation', host='runner', username='researcher')
    config = {'condition': condition, 'source_revision': manifest['source_revision'],
        'schema_version': protocol['schema_version'], 'translator': manifest['translator'],
        'translation_task': manifest['translation_task'],
        'target_packages': manifest['target_packages'],
        'target_environment': manifest.get('target_environment', {}), 'prompt_sha256': protocol['prompt_sha256'][condition],
        'manifest_sha256': hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest(),
        'manual_candidate_repairs': False,
        'shared_context_sha256': hashlib.sha256(json.dumps(protocol['shared_context_files'], sort_keys=True).encode()).hexdigest(),
        'graph_context_sha256': hashlib.sha256(json.dumps(protocol['condition_context_files'][condition], sort_keys=True).encode()).hexdigest()}
    record = wandb.init(entity=entity, project=project,
        id=hashlib.sha256((identity + condition).encode()).hexdigest()[:16], resume='never',
        name=identity + '-' + condition, group=identity, job_type='repository-translation',
        tags=['pilot' if manifest['translator']['model'] == 'auto' else 'fixed-model', condition],
        config=config, settings=settings)
    record.summary.update({'translation_status': 'not_started', 'manual_candidate_repairs': False})
    receipt = {'path': f'{entity}/{project}/{record.id}', 'url': record.url}
    record.finish()
    (run / condition / 'tracking.json').write_text(json.dumps(receipt, indent=2) + '\n')


def update(run, condition, row):
    receipt = run / condition / 'tracking.json'
    if not receipt.exists():
        return
    import wandb
    remote = wandb.Api(timeout=30).run(json.loads(receipt.read_text())['path'])
    translation = row['translation']
    remote.summary.update({'translation_status': translation.get('status'),
        'translation_seconds': translation.get('seconds'),
        'translator_resolved': translation.get('resolved_models', []),
        'patch_sha256': translation.get('patch_sha256'),
        'validation_status': row['validation'].get('status'),
        'validation_error': row['validation'].get('error', row['validation'].get('reason')),
        'manual_candidate_repairs': False})
    remote.summary.update()
