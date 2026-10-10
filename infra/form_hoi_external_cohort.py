"""One frozen four-DEV dispatcher: parallel CPU localization, serial GPU clips.

This is imported by the existing predictor ENTRY. It reads only the qualified
public-transfer receipt and four RGB/text directories, never private references.
No stage implementation, model inference, or baseline selection is duplicated.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import stat
from types import SimpleNamespace

DATA = Path('/srv/world-reward-data/form_hoi_external_dev_v1')


def discover(predictor, code, dev_revision):
    """Freeze all public artifact pins before the first model call."""
    predictor.require(type(dev_revision) is str and re.fullmatch('[0-9a-f]{40}', dev_revision),
        'Exact public DEV producer revision required')
    cohort = predictor.strict((code/'configs/form_hoi_insight_v1.json').read_bytes())
    sequences = [r['sequence_id'] for r in cohort['cohort'] if r['split'] == 'development']
    predictor.require(len(sequences) == len(set(sequences)) == 4 and all(
        type(s) is str and re.fullmatch('[A-Za-z0-9_-]{1,128}', s) for s in sequences),
        'Exactly four frozen DEV sequences; never reserved or rerolled')
    directory = predictor.canonical(DATA/dev_revision)
    receipt_path = directory/'public-transfer.json'; receipt_pin = predictor.artifact(receipt_path)
    predictor.require(0 < receipt_pin['bytes'] <= 65536, 'Bounded public-only transfer receipt required')
    receipt = predictor.strict(receipt_path.read_bytes())
    predictor.require(receipt.get('status') == 'pass' and receipt.get('dev_revision') == dev_revision and
        receipt.get('sequences') == sequences and receipt.get('private_references_transferred') is False and
        receipt.get('heavy_data_local') is False and type(receipt.get('manifest_identity')) is dict,
        'Qualified actual Azure four-DEV public-only transfer required')
    manifest = receipt['manifest_identity']
    predictor.require(set(manifest) == {'bytes','sha256'} and type(manifest['bytes']) is int and
        0 < manifest['bytes'] <= 65536 and re.fullmatch('[0-9a-f]{64}', str(manifest['sha256'])),
        'Independent source transfer-manifest SHA/bytes required')
    predictor.require({p.name for p in directory.iterdir()} == {*sequences, 'public-transfer.json'},
        'Inference transfer namespace may contain public packages only')
    rows = []
    for sid in sequences:
        path = directory/sid/'inputs/input.json'; pin = predictor.artifact(path)
        predictor.require(0 < pin['bytes'] <= 16384, 'Bounded original public input JSON required')
        package = predictor.load_public(path, pin, code)
        predictor.require(package['sequence_id'] == sid, 'Public package sequence differs from frozen cohort')
        rows.append(dict(sequence_id=sid, input=path, pin=pin))
    predictor.require(predictor.artifact(receipt_path) == receipt_pin,
        'Actual public transfer receipt changed during admission')
    return rows, dict(path=str(receipt_path), pin=receipt_pin,
        manifest_identity=manifest, private_references_transferred=False)


def dispatch(rows, stage, invoke):
    """Localize all four concurrently; never overlap GPU stage chains.

    CPU calls already started finish and are individually accounted on failure.
    Serial GPU dispatch fails at the first technical error, preserving its stage
    directory; it does not repeat the same broken model call on the other clips.
    """
    if stage not in ('localize','all'): raise ValueError('Explicit localize/all cohort mode required')
    if stage == 'all':
        for row in rows: invoke(row)
    else:
        failure = None
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [pool.submit(invoke, row) for row in rows]
            for future in as_completed(tasks):
                try: future.result()
                except Exception as error:
                    if failure is None: failure = error
        if failure is not None: raise failure


def run(predictor, code, revision, *, stage, dev_revision, reuse_localizations_from=None,
        reuse_stages_from=None, raster_gate=None, raster_prefix=None, prepare_prefix=None):
    predictor.require(stage in ('localize','all'), 'Explicit cohort stage required')
    binding = predictor.source(predictor.ROOT, code, revision, predictor.ENTRY, predictor.HELPERS)
    rows, transfer = discover(predictor, code, dev_revision)
    result = predictor.ROOT/'results'/f'form-hoi-external-cohort-{revision}-{stage}.json'
    if result.exists():
        predictor.artifact(result); old = predictor.strict(result.read_bytes())
        predictor.require(old.get('status') == 'complete' and old.get('producer_revision') == revision and
            old.get('stage') == stage and old.get('dev_revision') == dev_revision and
            old.get('public_transfer') == transfer and old.get('source_binding') == binding,
            'Existing cohort result is immutable; failed cohorts require diagnosis, not reroll')
        predictor.require(reuse_localizations_from is None or
            old.get('localization_reuse',{}).get('original_localization_producer')==reuse_localizations_from,
            'Existing cohort consumed a different original localization producer')
        for row in rows:
            base = predictor.ROOT/'results'/('form-hoi-external-predict-'+revision)/row['sequence_id']
            for target in (predictor.STAGES if stage == 'all' else ('localize',)):
                predictor.read_stage(base, target)
        return old
    report = dict(schema='world_reward.form_hoi_external_cohort.v1', status='fail', stage=stage,
        producer_revision=revision, dev_revision=dev_revision, public_transfer=transfer, source_binding=binding,
        ground_truth_used=False, private_truth_read=False, reserved_acquired=0,
        CPU_localizer_concurrency=4 if stage=='localize' else 1, GPU_clip_concurrency=0 if stage=='localize' else 1,
        sequence_order=[r['sequence_id'] for r in rows], sequences=[], full_4D_accuracy_verified=False)
    report['raster_runtime_controls'] = dict(capacity=raster_gate, prefix=raster_prefix,
        original_intermediate_outputs_preserved=True, native_fit_operators_unchanged=True)
    if raster_prefix is not None:
        predictor.require(stage == 'all' and raster_gate is not None,
            'Prefix runtime is only an explicit capacity-qualified inference control')
        from raster_prefix_activation import activation
        activation(raster_prefix['path'], {k: raster_prefix[k] for k in ('bytes', 'sha256')},
            current_image=predictor.config(code)['body_image'])
    outcomes = {}
    predictor.require(not (reuse_stages_from and reuse_localizations_from), 'One explicit technical reuse producer required')
    predictor.require(prepare_prefix is None or stage == 'all' and reuse_stages_from is not None,
        'Initializer prefix requires explicit stopped all-cohort stage resume')
    if reuse_stages_from is not None:
        predictor.require(stage == 'all' and raster_gate is not None,
            'Successful-prefix resume requires explicit same-source raster qualification')
        from form_stage_reuse import reuse
        report['stage_reuse'] = reuse(predictor, code, revision, reuse_stages_from, rows,
            allow_body_depth_camera_change=True, prepare_prefix=prepare_prefix)
        report['localization_reuse'] = report['stage_reuse']['original_localization_proof']
    elif reuse_localizations_from is not None:
        from form_prediction_reuse import reuse
        report['localization_reuse']=reuse(predictor,code,revision,reuse_localizations_from,rows)
    # Check all fresh one-shot pairs before any CPU API call, not halfway
    # through localization after the other workers have consumed their tokens.
    for row in rows:
        sid=row['sequence_id']; base=predictor.ROOT/'results'/('form-hoi-external-predict-'+revision)/sid
        if (base/'localize/report.json').is_file():
            predictor.read_stage(base,'localize');continue
        for path in predictor.credential_paths(revision,sid):
            predictor.canonical(path); info=path.lstat()
            predictor.require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and
                not info.st_mode&0o077 and 0<info.st_size<16384,
                'Every fresh sequence requires its own private one-shot credential pair')
    def invoke(row):
        sid = row['sequence_id']; base = predictor.ROOT/'results'/('form-hoi-external-predict-'+revision)/sid
        args = SimpleNamespace(stage=stage, input=row['input'], input_bytes=row['pin']['bytes'],
            input_sha256=row['pin']['sha256'], out=base,
            raster_gate_report=raster_gate['path'] if raster_gate else None,
            raster_gate_bytes=raster_gate['bytes'] if raster_gate else None,
            raster_gate_sha256=raster_gate['sha256'] if raster_gate else None,
            raster_prefix_report=raster_prefix['path'] if raster_prefix else None,
            raster_prefix_bytes=raster_prefix['bytes'] if raster_prefix else None,
            raster_prefix_sha256=raster_prefix['sha256'] if raster_prefix else None)
        record = dict(sequence_id=sid, input=dict(path=str(row['input']), pin=row['pin']), status='fail')
        outcomes[sid] = record
        try:
            predictor.driver(args)
            record['stages'] = {name: dict(path=str(base/name/'report.json'),
                pin=predictor.artifact(base/name/'report.json'))
                for name in (predictor.STAGES if stage=='all' else ('localize',))}
            record['status'] = 'complete'
        except Exception as error:
            record['error_type'] = type(error).__name__; raise
    try:
        dispatch(rows,stage,invoke)
        repeated, after = discover(predictor,code,dev_revision)
        predictor.require(repeated == rows and after == transfer and
            predictor.source(predictor.ROOT,code,revision,predictor.ENTRY,predictor.HELPERS) == binding,
            'Frozen public inputs/source changed during cohort execution')
        report['status'] = 'complete'
    except Exception as error:
        report['error_type'] = type(error).__name__; raise
    finally:
        report['sequences'] = [outcomes.get(r['sequence_id'],dict(sequence_id=r['sequence_id'],status='not_started')) for r in rows]
        predictor.save_json(result,report)
    print(json.dumps(dict(stage='FORM_cohort_'+stage,status='complete',sequences=4,
        GPU_clip_concurrency=report['GPU_clip_concurrency'])),flush=True)
    return report
