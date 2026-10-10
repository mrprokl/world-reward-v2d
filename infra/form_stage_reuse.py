"""Byte-exact public-only technical resume; resume only sealed completed initializers, never failed prepare/fit outputs.

All four original localizations are retained, including original API/source
ancestry. Only the predeclared successful first-sequence stage prefix is copied;
no quality-dependent sample selection or new model/API calls. A changed raster
helper is explicitly original QA, NOT a claim of new-implementation equivalence.
"""
import ast
from pathlib import Path
import re
import shutil

import form_prediction_reuse as localization

PREFIX = ('track', 'body_depth', 'object')
CONTROL_FILES = {'infra/form_hoi_external_predict.py', 'infra/run_form_hoi_external_predict.sh',
    'infra/form_hoi_external_cohort.py', 'infra/run_form_hoi_external_cohort.sh',
    'infra/form_prediction_reuse.py', 'infra/form_stage_reuse.py'}
FUNCTIONS = ('config', 'load_public', 'fixed_K', 'frames', 'read_stage', 'mask', 'sha',
    'artifact', 'seal', 'save_json', 'save_npz', 'localize', 'track', 'load_body',
    'body_transport', 'body_depth', 'pointmap', 'object_mesh')


def stage_execution(raw, stage):
    tree = ast.parse(raw)
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    branch = next(n for n in ast.walk(worker) if isinstance(n, ast.If)
        and ast.unparse(n.test) == f"stage == '{stage}'")
    return ast.dump(ast.Module(body=branch.body, type_ignores=[]), include_attributes=False)


def terminal(p, revision, rows, binding, stop=None):
    if stop is None:
        path = p.ROOT/'results'/f'form-hoi-external-cohort-{revision}-all.json'
        pin = p.artifact(path); value = p.strict(path.read_bytes())
        p.require(value.get('status') in ('fail', 'complete') and value.get('stage') == 'all'
            and value.get('source_binding') == binding, 'Terminal original cohort receipt required')
    else:
        p.require(set(stop) == {'path', 'pin'}, 'Explicit terminal-stop SHA/byte pin required')
        path = p.canonical(Path(stop['path'])); pin = p.artifact(path)
        p.require(path.is_relative_to(p.ROOT/'results') and not {'eval_private', 'gt'}.intersection(path.parts)
            and pin == stop['pin'] and pin['bytes'] <= 65536, 'Bounded independent terminal-stop pin differs')
        value = p.strict(path.read_bytes())
        p.require(value.get('schema') == 'world_reward.form_prediction_terminal_stop.v1'
            and value.get('owned_driver_inactive') is True
            and value.get('owned_stage_container_absent') is True,
            'Actual stopped owned driver/container attestation required')
    p.require(value.get('producer_revision') == revision and value.get('ground_truth_used') is False
        and value.get('private_truth_read') is False and value.get('sequence_order') == [r['sequence_id'] for r in rows],
        'Original terminal producer/four-sequence/public-only lineage differs')
    return dict(path=str(path), pin=pin)


def unchanged_sources(p, code, oldcode, oldraw, *, allow_body_depth_camera_change):
    current = (code/'infra/form_hoi_external_predict.py').read_bytes()
    for name in FUNCTIONS:
        p.require(localization.function_identity(oldraw, name) == localization.function_identity(current, name),
            'Reused algorithm/conditioning changed: ' + name)
    for stage in PREFIX:
        p.require(stage_execution(oldraw, stage) == stage_execution(current, stage),
            'Original native stage execution context changed: ' + stage)
    old_inventory = localization.helper_inventory(oldraw)
    new_inventory = localization.helper_inventory(current)
    identities = {}; camera = {}
    for name in old_inventory:
        p.require(name in new_inventory, 'Original reused helper omitted from current closure')
        if name in CONTROL_FILES: continue
        old = p.artifact(oldcode/name); new = p.artifact(code/name)
        if name == 'infra/camera_render.py' and old != new:
            p.require(allow_body_depth_camera_change is True, 'Changed Body depth raster dependency requires explicit attestation')
            camera = dict(original=old, downstream=new, original_QA_reused_byte_exact=True,
                original_camera_QA_rerun=False, numeric_equivalence_claimed=False,
                scope='body_depth_original_output_only_new_raster_not_executed_in_reused_stage')
        else:
            p.require(old == new, 'Reused source/model/config helper changed: ' + name)
            identities[name] = old
    return identities, camera


def qualify(p, code, new_revision, old_revision, rows, *, stop=None, allow_body_depth_camera_change=False):
    p.require(re.fullmatch('[0-9a-f]{40}', str(old_revision)) and old_revision != new_revision
        and len(rows) == 4 and len({r['sequence_id'] for r in rows}) == 4,
        'Different original producer and complete frozen four-sequence order required')
    oldcode = p.canonical(p.ROOT/'jobs'/old_revision/p.ENTRY/'code')
    oldraw = (oldcode/'infra/form_hoi_external_predict.py').read_bytes()
    binding = p.source(p.ROOT, oldcode, old_revision, p.ENTRY, localization.helper_inventory(oldraw))
    terminal_receipt = terminal(p, old_revision, rows, binding, stop)
    firstbase = p.ROOT/'results'/('form-hoi-external-predict-'+old_revision)/rows[0]['sequence_id']
    inherited = p.strict((firstbase/'localization_reuse.json').read_bytes())
    loc_revision = inherited.get('original_localization_producer')
    loc_sources, loc_proof = localization.qualified_sources(p, code, new_revision, loc_revision, rows)
    identities, camera = unchanged_sources(p, code, oldcode, oldraw,
        allow_body_depth_camera_change=allow_body_depth_camera_change)
    stages = []; source_bindings = {old_revision: binding}
    for index, row in enumerate(rows):
        package = p.load_public(row['input'], row['pin'], code)
        base = p.ROOT/'results'/('form-hoi-external-predict-'+old_revision)/row['sequence_id']
        # Reused intermediate localize must still be the exact original806 API
        # output; its 125 producer ancestry is carried by the compatible proof.
        lineage = localization.localization_lineage(p, base, old_revision)
        p.require(lineage['producer_revision'] == loc_revision
            and all(p.artifact(base/'localize'/name) == pin for name, pin in loc_sources[index]['artifacts'].items()),
            'Intermediate localization/API response/source ancestry changed')
        for stage in PREFIX:
            path = base/stage/'report.json'
            if index:
                p.require(not path.is_file(), 'Only declared first-sequence technical prefix may be reused')
                continue
            directory, report = p.read_stage(base, stage)
            original = original_stage_lineage(p, base, stage, old_revision)
            producer = original['producer_revision']
            if producer not in source_bindings:
                producer_code = p.canonical(p.ROOT/'jobs'/producer/p.ENTRY/'code')
                producer_raw = (producer_code/'infra/form_hoi_external_predict.py').read_bytes()
                source_bindings[producer] = p.source(p.ROOT, producer_code, producer, p.ENTRY,
                    localization.helper_inventory(producer_raw))
                unchanged_sources(p, code, producer_code, producer_raw,
                    allow_body_depth_camera_change=allow_body_depth_camera_change)
            localization.expected_lineage(report, row, producer, source_bindings[producer], p, package)
            prior_loc = report.get('localization_source', {})
            p.require(report.get('source_rehashed_after') is True and all(prior_loc.get(k) == lineage[k]
                for k in ('producer_revision', 'report_pin', 'boxes_pin')), 'Completed stage localization bytes differ')
            stages.append(dict(stage=stage, original_directory=str(directory), original_stage_producer=producer,
                original_source_binding=source_bindings[producer], original_stage_lineage=original,
                original_localization_source=prior_loc, artifacts={**report['artifacts'], 'report.json': p.artifact(path)}))
    p.require(p.artifact(Path(terminal_receipt['path'])) == terminal_receipt['pin']
        and p.source(p.ROOT, oldcode, old_revision, p.ENTRY, localization.helper_inventory(oldraw)) == binding,
        'Original stopped producer/source changed during qualification')
    return stages, dict(schema='world_reward.form_stage_reuse.v1', new_prediction_producer=new_revision,
        original_stage_producer=old_revision, original_source_binding=binding, terminal_receipt=terminal_receipt,
        sequence_order=[r['sequence_id'] for r in rows], sequence_id=rows[0]['sequence_id'],
        public_input=dict(path=str(rows[0]['input']), pin=rows[0]['pin']), ordered_successful_prefix=list(PREFIX),
        unchanged_source_identities=identities, camera_render_change=camera,
        original_localization_producer=loc_revision, original_localization_proof=loc_proof,
        localization_API_calls=0, model_calls=0, original_reports_byte_preserved=True,
        failed_prepare_or_fit_copied=False, incomplete_cohort_scientific_resampling=False,
        ground_truth_used=False, private_truth_read=False)


def reuse(p, code, new_revision, old_revision, rows, *, stop=None, allow_body_depth_camera_change=False, prepare_prefix=None):
    stages, proof = qualify(p, code, new_revision, old_revision, rows,
        stop=stop, allow_body_depth_camera_change=allow_body_depth_camera_change)
    prefix = qualify_prepare_prefix(p, code, old_revision, rows[0], prepare_prefix) if prepare_prefix is not None else None
    # Admission of ALL four original APIs and every destination/artifact precedes
    # the first copy. No failed-stage logs, native parameters or reference files.
    for row in rows:
        base = p.ROOT/'results'/('form-hoi-external-predict-'+new_revision)/row['sequence_id']
        p.require(not base.exists() or not tuple(p.canonical(base).iterdir()), 'Exclusive empty destination required')
    for item in stages:
        for name, pin in item['artifacts'].items():
            p.require(p.artifact(Path(item['original_directory'])/name) == pin, 'Original prefix artifact changed')
    localization.reuse(p, code, new_revision, proof['original_localization_producer'], rows)
    base = p.ROOT/'results'/('form-hoi-external-predict-'+new_revision)/rows[0]['sequence_id']
    for item in stages:
        target = base/item['stage']; target.mkdir(mode=0o755)
        for name, pin in item['artifacts'].items():
            origin = Path(item['original_directory'])/name; destination = target/name
            destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            with origin.open('rb') as stream:
                copied = p.seal(destination, lambda output: shutil.copyfileobj(stream, output, 1 << 20))
            p.require(copied == pin and p.artifact(origin) == pin, 'Byte-exact original prefix copy changed')
        p.read_stage(base, item['stage']); item['copied_directory'] = str(target)
    proof['stages'] = stages
    p.save_json(base/'stage_reuse.json', proof)
    if prefix is not None:
        proof['prepare_prefix'] = copy_prepare_prefix(p, base, prefix, new_revision)
    return proof


def original_stage_lineage(p, base, stage, new_revision):
    """New downstream workers consume original reports without relabeling them."""
    directory, report = p.read_stage(base, stage)
    result = dict(producer_revision=report['producer_revision'], report_pin=p.artifact(directory/'report.json'),
        original_reports_byte_preserved=True, source_binding=report['source_binding'])
    if report['producer_revision'] == new_revision: return result
    p.require(stage in PREFIX, 'Only the declared successful technical prefix may have old stage ancestry')
    path = base/'stage_reuse.json'; pin = p.artifact(path); proof = p.strict(path.read_bytes())
    p.require(proof.get('schema') == 'world_reward.form_stage_reuse.v1'
        and proof.get('new_prediction_producer') == new_revision
        and proof.get('original_stage_producer') == proof.get('original_source_binding', {}).get('producer_revision')
        and proof.get('sequence_id') == base.name and proof.get('ordered_successful_prefix') == list(PREFIX)
        and proof.get('ground_truth_used') is False and proof.get('private_truth_read') is False
        and proof.get('localization_API_calls') == 0 and proof.get('failed_prepare_or_fit_copied') is False
        and proof.get('original_reports_byte_preserved') is True, 'Original stage resume lineage required')
    rows = proof['stages']; p.require([r['stage'] for r in rows] == list(PREFIX), 'Complete original prefix proof required')
    item = next(r for r in rows if r['stage'] == stage)
    p.require(item['copied_directory'] == str(directory) and item['artifacts'] == {
            **report['artifacts'], 'report.json': result['report_pin']}
        and report['producer_revision'] == item.get('original_stage_producer', proof['original_stage_producer'])
        and report['source_binding'] == item.get('original_source_binding', proof['original_source_binding'])
        and report.get('localization_source') == item['original_localization_source']
        and report.get('input_pin') == proof['public_input']['pin'], 'Original copied stage/source byte pins differ')
    result['reuse_proof'] = dict(path=str(path), pin=pin)
    result['camera_QA_source'] = proof['camera_render_change'] if stage == 'body_depth' else {}
    return result


PREPARE_PREFIX_PRODUCER = '80eef93ad61c6a331e977092f99d50608cd72e9b'
PREPARE_PREFIX_REPORT = dict(bytes=29828, sha256='795cd4261cabd463dbd1fa9471dcbc37a400846c778c00497c34e3310e393633')
PREPARE_PREFIX_FILES = {
    'shared_initializer.pkl': dict(bytes=430872, sha256='1c4c2611ee8869618627eac30327b6b14dce8847d2c3f01053f746316ab35b13'),
    'object_prior.npz': dict(bytes=6585737, sha256='64733a597e20fda9815e0a8b7be46825ca073ef7c792b8e9d0145490c048387e'),
    'object_metric.glb': dict(bytes=13385120, sha256='c5ee3e62fdc369cb5da0cdc8e03f5e478edd5a1e6b33188e68e2c0412c021d9f'),
}


def initialization_source(raw, name):
    """Compare the completed numerical prefix, not new export/import plumbing."""
    tree = ast.parse(raw); node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = node.body
    if name == 'prepare':
        end = next(i for i,n in enumerate(body) if isinstance(n, ast.ImportFrom) and n.module == 'prep.prepare_mhr_wild_export')
        body = body[:end]
    else:
        body = body[:-1]  # New extraction's sole return; original statements stay exact.
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def qualify_prepare_prefix(p, code, revision, row, approval):
    p.require(revision == PREPARE_PREFIX_PRODUCER and set(approval) == {'path', 'bytes', 'sha256'},
        'Explicit independently qualified technical prefix only')
    path = p.canonical(Path(approval['path'])); pin = {k:approval[k] for k in ('bytes', 'sha256')}
    p.require(path.is_relative_to(p.ROOT/'results') and 0 < pin['bytes'] <= 16384 and p.artifact(path) == pin,
        'Bounded sealed prefix approval identity required')
    value = p.strict(path.read_bytes())
    directory = p.ROOT/'results'/('form-hoi-external-predict-'+revision)/row['sequence_id']/'prepare'
    expected = dict(schema='world_reward.form_prepare_prefix_approval.v1', producer_revision=revision,
        sequence_id=row['sequence_id'], failed_report=dict(path=str(directory/'report.json'), pin=PREPARE_PREFIX_REPORT),
        artifacts=PREPARE_PREFIX_FILES)
    p.require(value == expected and p.artifact(directory/'report.json') == PREPARE_PREFIX_REPORT,
        'Original failed receipt and all three independently pinned prefix artifacts required')
    p.require({x.name for x in directory.iterdir()} == {*PREPARE_PREFIX_FILES, 'report.json', 'technical_traceback.txt'},
        'Exact failed initializer inventory, without partial export or reference, required')
    oldcode = p.ROOT/'jobs'/revision/p.ENTRY/'code'; oldraw=(oldcode/'infra/form_hoi_external_predict.py').read_bytes()
    binding=p.source(p.ROOT, oldcode, revision, p.ENTRY, localization.helper_inventory(oldraw))
    failed=p.strict((directory/'report.json').read_bytes()); package=p.load_public(row['input'],row['pin'],code)
    localization.expected_lineage(failed,row,revision,binding,p,package)
    p.require(failed.get('status') == 'fail' and failed.get('stage') == 'prepare'
        and failed.get('error_type') == 'ModuleNotFoundError', 'Exact diagnosed import failure required')
    current=(code/'infra/form_hoi_external_predict.py').read_bytes()
    p.require(initialization_source(oldraw,'prepare') == initialization_source(current,'prepare_initialization')
        and localization.function_identity(oldraw,'pose_initializer') == localization.function_identity(current,'pose_initializer'),
        'Completed native initialization and full pose solver must remain unchanged')
    for name, artifact in PREPARE_PREFIX_FILES.items():
        p.require(p.artifact(directory/name) == artifact, 'Original completed initializer bytes differ')
    for stage in PREFIX:
        actual = original_stage_lineage(p,directory.parent,stage,revision)
        p.require(failed['consumed_intermediate_sources'][stage] == actual, 'Failed initializer upstream ancestry differs')
    p.require(p.artifact(path) == pin and p.artifact(directory/'report.json') == PREPARE_PREFIX_REPORT
        and p.source(p.ROOT,oldcode,revision,p.ENTRY,localization.helper_inventory(oldraw)) == binding,
        'Original failed source/report or prefix approval changed during qualification')
    return dict(approval=dict(path=str(path),pin=pin), original_directory=str(directory),
        original_failed_report=PREPARE_PREFIX_REPORT, original_source_binding=binding,
        original_public_input=row, decoder_identity=failed['decoder_identity'], body_assets=failed['body_assets'],
        inference_source_identity=failed['inference_source_identity'],
        consumed_intermediate_sources=failed['consumed_intermediate_sources'], artifacts=PREPARE_PREFIX_FILES)


def copy_prepare_prefix(p, base, proof, revision):
    out=base/'prepare_prefix'; out.mkdir(mode=0o755)
    for name,pin in proof['artifacts'].items():
        origin=Path(proof['original_directory'])/name
        with origin.open('rb') as stream:
            copied=p.seal(out/name, lambda f:shutil.copyfileobj(stream,f,1<<20))
        p.require(copied == pin and p.artifact(origin) == pin, 'Original initializer byte copy changed')
    record=dict(proof,schema='world_reward.form_prepare_prefix_reuse.v1',new_prediction_producer=revision,
        sequence_id=base.name, final_prediction=False, native_export_complete=False,
        failed_prepare_or_fit_copied=False, ground_truth_used=False, private_truth_read=False)
    # Paths in input records are strings in sealed JSON.
    record['original_public_input']=dict(sequence_id=base.name,input=str(proof['original_public_input']['input']),pin=proof['original_public_input']['pin'])
    pin=p.save_json(out/'report.json',record); out.chmod(0o555)
    return dict(path=str(out/'report.json'),pin=pin)


def consume_prepare_prefix(p, base, out, report, package, revision):
    """Pure array/geometry checks, then copies; no model, reference or fitting."""
    directory=base/'prepare_prefix'
    if not directory.exists(): return None
    import numpy as np
    import pickle
    import trimesh
    p.require({x.name for x in directory.iterdir()} == {*PREPARE_PREFIX_FILES,'report.json'}, 'Exact sealed initializer-only prefix required')
    metadata=p.strict((directory/'report.json').read_bytes()); rp=p.artifact(directory/'report.json')
    p.require(metadata.get('schema') == 'world_reward.form_prepare_prefix_reuse.v1'
        and metadata.get('new_prediction_producer') == revision and metadata.get('sequence_id') == base.name
        and metadata.get('artifacts') == PREPARE_PREFIX_FILES and metadata.get('original_failed_report') == PREPARE_PREFIX_REPORT
        and metadata.get('original_public_input',{}).get('pin') == report['input_pin']
        and all(metadata.get(k) is False for k in ('final_prediction','native_export_complete',
            'failed_prepare_or_fit_copied','ground_truth_used','private_truth_read')), 'Explicit nonprediction prefix ancestry required')
    for stage in PREFIX:
        actual=original_stage_lineage(p,base,stage,revision)
        previous=metadata['consumed_intermediate_sources'][stage]
        p.require(all(actual[k] == previous[k] for k in ('producer_revision','report_pin','source_binding')),
            'Copied initializer must consume identical original masks/body/gauge/object stage bytes')
    for name,pin in PREPARE_PREFIX_FILES.items(): p.require(p.artifact(directory/name) == pin, 'Sealed prefix bytes differ')
    # Only independently pinned locally produced pickle, never dataset pickle/GT.
    with (directory/'shared_initializer.pkl').open('rb') as stream: initializer=pickle.load(stream)
    params={k:initializer[k] for k in p.NATIVE_PARAMETER_DIMS}; p.validate_native_parameters(params,96,require_shared_identity=True)
    p.require(initializer['frames'] == [f'{i:06d}' for i in range(96)] and initializer['body_model'] == 'mhr'
        and initializer['metadata'].get('public_sequence_id') == package['sequence_id']
        and initializer['metadata'].get('shared_identity_geometry_redecoded') is True,
        'Full96 native shared-identity initializer required')
    for key,count in (('mhr_joints',127),('mhr_keypoints',70)):
        a=initializer[key]; p.require(type(a) is np.ndarray and a.dtype == np.float32 and a.shape == (96,count,3)
            and np.isfinite(a).all(), 'Original full96 decoded geometry required')
    with np.load(directory/'object_prior.npz',allow_pickle=False) as z:
        p.require(set(z.files) == {'vertices','faces','rotation','translation','observed','frame_index'}, 'Exact original pose NPZ ABI required')
        v,f,r,t,observed,indices=(z[k].copy() for k in ('vertices','faces','rotation','translation','observed','frame_index'))
    p.require(v.dtype == np.float32 and v.ndim == 2 and v.shape[1] == 3 and np.isfinite(v).all()
        and f.dtype == np.int64 and f.ndim == 2 and f.shape[1] == 3 and len(f)>0 and f.min()>=0 and f.max()<len(v)
        and r.dtype == np.float32 and r.shape == (96,3,3) and np.isfinite(r).all()
        and t.dtype == np.float32 and t.shape == (96,3) and np.isfinite(t).all()
        and observed.dtype == np.bool_ and observed.shape == (96,) and observed[0] and observed[-1]
        and indices.dtype == np.int64 and np.array_equal(indices,np.arange(96))
        and np.allclose(r@r.swapaxes(-1,-2),np.eye(3),atol=1e-5) and np.allclose(np.linalg.det(r),1.,atol=1e-5),
        'Full original geometry, SO3 poses and observation flags required')
    gauge=p.strict((base/'body_depth/gauge.json').read_bytes()); K=np.asarray(gauge['K']); p.require(K.shape==(3,3)
        and np.isfinite(K).all() and gauge['alignment']['shared_scale']>0, 'Same positive shared predicted-human gauge required')
    obj=trimesh.load(base/'object/object.glb',force='mesh',process=False)
    transform=p.strict((base/'object/transform.json').read_bytes()); scale=np.asarray(transform['scale'])
    p.require(scale.shape==(3,) and np.isfinite(scale).all() and (scale>0).all()
        and np.array_equal((np.asarray(obj.vertices)*scale[None]).astype(np.float32),v)
        and np.array_equal(np.asarray(obj.faces,np.int64),f), 'Entire source object and once-baked scale must agree')
    metric=trimesh.load(directory/'object_metric.glb',force='mesh',process=False)
    p.require(np.array_equal(np.asarray(metric.vertices,np.float32),v) and np.array_equal(np.asarray(metric.faces,np.int64),f),
        'Original whole metric GLB and numeric mesh must agree')
    for name,pin in PREPARE_PREFIX_FILES.items():
        with (directory/name).open('rb') as stream: copied=p.seal(out/name,lambda output:shutil.copyfileobj(stream,output,1<<20))
        p.require(copied==pin,'Native initializer-only output copy differs')
    report.update(decoder_identity=metadata['decoder_identity'],body_assets=metadata['body_assets'],
        inference_source_identity=metadata['inference_source_identity'], prepare_initializer_reuse=dict(report_pin=rp,
            original_failed_report=PREPARE_PREFIX_REPORT, final_prediction=False, model_calls=0, pose_fit_replayed=False))
    p.require(p.artifact(directory/'report.json')==rp,'Original prefix metadata changed')
    return initializer,v,f,r,t,observed
