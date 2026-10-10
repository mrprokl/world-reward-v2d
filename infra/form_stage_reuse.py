"""Byte-exact public-only technical resume; never reuse failed prepare/fit.

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
    'infra/form_prediction_reuse.py'}
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
    stages = []
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
            localization.expected_lineage(report, row, old_revision, binding, p, package)
            p.require(report.get('source_rehashed_after') is True
                and report.get('localization_source') == lineage,
                'Completed stage must bind original byte-exact localizer/source')
            stages.append(dict(stage=stage, original_directory=str(directory),
                original_localization_source=lineage, artifacts={**report['artifacts'], 'report.json': p.artifact(path)}))
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


def reuse(p, code, new_revision, old_revision, rows, *, stop=None, allow_body_depth_camera_change=False):
    stages, proof = qualify(p, code, new_revision, old_revision, rows,
        stop=stop, allow_body_depth_camera_change=allow_body_depth_camera_change)
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
        and proof.get('original_stage_producer') == report['producer_revision']
        and proof.get('sequence_id') == base.name and proof.get('ordered_successful_prefix') == list(PREFIX)
        and proof.get('ground_truth_used') is False and proof.get('private_truth_read') is False
        and proof.get('localization_API_calls') == 0 and proof.get('failed_prepare_or_fit_copied') is False
        and proof.get('original_reports_byte_preserved') is True, 'Original stage resume lineage required')
    rows = proof['stages']; p.require([r['stage'] for r in rows] == list(PREFIX), 'Complete original prefix proof required')
    item = next(r for r in rows if r['stage'] == stage)
    p.require(item['copied_directory'] == str(directory) and item['artifacts'] == {
            **report['artifacts'], 'report.json': result['report_pin']}
        and report['source_binding'] == proof['original_source_binding']
        and report.get('localization_source') == item['original_localization_source']
        and report.get('input_pin') == proof['public_input']['pin'], 'Original copied stage/source byte pins differ')
    result['reuse_proof'] = dict(path=str(path), pin=pin)
    result['camera_QA_source'] = proof['camera_render_change'] if stage == 'body_depth' else {}
    return result
