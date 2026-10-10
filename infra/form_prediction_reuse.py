"""Authenticated byte-exact localization reuse after an infrastructure failure.

Only successful public-only localization artifacts are copied. The original
report producer, source closure, model response and scientific conditioning stay
untouched. No failed tracking artifacts, credentials or reference inputs move.
"""
import ast
from pathlib import Path
import re

SOURCE_FILES = ('configs/form_hoi_external_predict_v1.json',
    'configs/form_hoi_external_dev_v1.json', 'configs/form_hoi_insight_v1.json',
    'src/world_reward/gemini_localization.py', 'src/world_reward/vertex_retry.py')


def helper_inventory(raw):
    """Read a literal source inventory without executing historical code."""
    tree=ast.parse(raw)
    assignments=[node for node in tree.body if isinstance(node,ast.Assign) and
        any(isinstance(target,ast.Name) and target.id=='HELPERS' for target in node.targets)]
    if len(assignments)!=1:raise ValueError('Original literal helper inventory required')
    config=[node for node in tree.body if isinstance(node,ast.Assign) and
        any(isinstance(target,ast.Name) and target.id=='CONFIG'for target in node.targets)]
    if len(config)!=1:raise ValueError('Original literal config path required')
    config_path=ast.literal_eval(config[0].value)
    value=assignments[0].value
    if not isinstance(value,ast.Tuple):raise ValueError('Original literal helper tuple required')
    inventory=tuple(config_path if isinstance(node,ast.Name) and node.id=='CONFIG'
        else ast.literal_eval(node)for node in value.elts)
    if (type(inventory) is not tuple or not inventory or len(inventory)!=len(set(inventory)) or
            any(type(p) is not str or p.startswith('/') or '..' in Path(p).parts for p in inventory)):
        raise ValueError('Original complete safe helper inventory required')
    return inventory


def function_identity(raw,name):
    tree=ast.parse(raw)
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(nodes)!=1:raise ValueError('Original localization function required')
    return ast.dump(nodes[0],include_attributes=False)


def expected_lineage(report,row,old_revision,binding,predictor,package):
    predictor.require(report.get('producer_revision')==old_revision and report.get('source_binding')==binding and
        report.get('sequence_id')==row['sequence_id'] and report.get('input_pin')==row['pin'] and
        report.get('video_pin')==package['video_pin'] and report.get('dataset')=='nvidia/form-hoi' and
        report.get('original_frame_indices')==list(range(96)) and
        report.get('full_source_frames')==package['full_source_frames'] and
        report.get('ground_truth_used') is False and report.get('private_truth_read') is False and
        report.get('reference_inputs_mounted') is False and report.get('hand_labeled_test') is False and
        report.get('oracle_modes')==[], 'Original successful localization public/source lineage differs')


def qualified_sources(predictor,code,new_revision,old_revision,rows):
    predictor.require(type(old_revision) is str and re.fullmatch('[0-9a-f]{40}',old_revision) and
        old_revision!=new_revision,'Explicit different original localization producer required')
    oldcode=predictor.canonical(predictor.ROOT/'jobs'/old_revision/predictor.ENTRY/'code')
    raw=(oldcode/'infra/form_hoi_external_predict.py').read_bytes()
    binding=predictor.source(predictor.ROOT,oldcode,old_revision,predictor.ENTRY,helper_inventory(raw))
    current=(code/'infra/form_hoi_external_predict.py').read_bytes()
    predictor.require(function_identity(raw,'localize')==function_identity(current,'localize'),
        'Localization algorithm changed; infrastructure replay must not relabel a different experiment')
    identities={}
    for name in SOURCE_FILES:
        previous=predictor.artifact(oldcode/name); now=predictor.artifact(code/name)
        predictor.require(previous==now,'Original conditioning/config/model or technical-retry helper changed')
        identities[name]=now
    cfg=predictor.config(code)
    cohort_path=predictor.ROOT/'results'/f'form-hoi-external-cohort-{old_revision}-localize.json'
    cohort_pin=predictor.artifact(cohort_path); cohort=predictor.strict(cohort_path.read_bytes())
    predictor.require(cohort.get('status')=='complete' and cohort.get('stage')=='localize' and
        cohort.get('producer_revision')==old_revision and cohort.get('ground_truth_used') is False and
        cohort.get('private_truth_read') is False and cohort.get('source_binding')==binding and
        cohort.get('sequence_order')==[r['sequence_id']for r in rows],
        'All four original localizations must have completed before reuse')
    original_records=cohort.get('sequences',[])
    predictor.require(len(original_records)==4 and
        [r.get('sequence_id')for r in original_records]==[r['sequence_id']for r in rows] and
        all(r.get('status')=='complete'for r in original_records),
        'No partial-cohort localization selection or response shopping')
    sources=[]
    for row,record in zip(rows,original_records):
        sid=row['sequence_id'];package=predictor.load_public(row['input'],row['pin'],code)
        base=predictor.ROOT/'results'/('form-hoi-external-predict-'+old_revision)/sid
        directory,report=predictor.read_stage(base,'localize')
        predictor.require(set(report['artifacts'])=={'boxes.json'} and
            record.get('input')==dict(path=str(row['input']),pin=row['pin']) and
            record.get('stages',{}).get('localize')==dict(path=str(directory/'report.json'),
                pin=predictor.artifact(directory/'report.json')),
            'Only original sealed complete RGB/text localization artifacts may be reused')
        expected_lineage(report,row,old_revision,binding,predictor,package)
        boxes=predictor.strict((directory/'boxes.json').read_bytes())
        predictor.require(boxes==dict(rows=report['calls'],conditioning='public_RGB_object_prompt_action_only') and
            report.get('localizer')==cfg['localizer'] and
            [r.get('frame_index')for r in report['calls']]==cfg['seed_frames'] and
            all(type(r.get('rgb_sha256')) is str and re.fullmatch('[0-9a-f]{64}',r['rgb_sha256'])for r in report['calls']),
            'Exact original seed/RGB/description-action conditioned responses required')
        sources.append(dict(sequence_id=sid,directory=directory,
            artifacts={name:predictor.artifact(directory/name)for name in ('boxes.json','report.json')}))
    predictor.require(predictor.artifact(cohort_path)==cohort_pin and
        predictor.source(predictor.ROOT,oldcode,old_revision,predictor.ENTRY,helper_inventory(raw))==binding,
        'Original immutable cohort/source changed while validating reuse')
    return sources,dict(schema='world_reward.form_localization_reuse.v1',
        new_prediction_producer=new_revision,original_localization_producer=old_revision,
        original_cohort=dict(path=str(cohort_path),pin=cohort_pin),original_source_binding=binding,
        unchanged_localizer_source_identities=identities,localization_API_calls=0,
        original_reports_byte_preserved=True,failed_tracking_artifacts_copied=False,
        ground_truth_used=False,private_truth_read=False)


def reuse(predictor,code,new_revision,old_revision,rows):
    sources,proof=qualified_sources(predictor,code,new_revision,old_revision,rows)
    prepared=[]
    # Validate all destinations and all original artifacts before first copy.
    for src in sources:
        base=predictor.ROOT/'results'/('form-hoi-external-predict-'+new_revision)/src['sequence_id']
        if base.exists():
            predictor.canonical(base)
            predictor.require(not tuple(base.iterdir()),'Localization reuse requires exclusive empty destination')
        prepared.append((src,base))
    records=[]
    for src,base in prepared:
        if not base.exists():base.mkdir(parents=True,mode=0o755)
        target=base/'localize';target.mkdir(mode=0o755)
        for name,pin in src['artifacts'].items():
            path=src['directory']/name
            predictor.require(predictor.artifact(path)==pin,'Original localization artifact changed before copy')
            with path.open('rb')as stream:
                copied=predictor.seal(target/name,lambda out:out.write(stream.read()))
            predictor.require(copied==pin and predictor.artifact(path)==pin,'Byte-exact localization copy failed')
        predictor.read_stage(base,'localize')
        record=dict(sequence_id=src['sequence_id'],original_directory=str(src['directory']),
            copied_directory=str(target),original_artifacts=src['artifacts'],
            original_report_producer=old_revision,original_reports_byte_preserved=True)
        predictor.save_json(base/'localization_reuse.json',dict(proof,sequence=record))
        records.append(record)
    return dict(proof,sequences=records)


def localization_lineage(predictor,base,new_revision):
    """GPU-stage ancestry without mounting the historical producer directory."""
    directory,report=predictor.read_stage(base,'localize')
    result=dict(producer_revision=report['producer_revision'],report_pin=predictor.artifact(directory/'report.json'),
        boxes_pin=predictor.artifact(directory/'boxes.json'),original_reports_byte_preserved=True)
    if report['producer_revision']!=new_revision:
        path=base/'localization_reuse.json';pin=predictor.artifact(path);proof=predictor.strict(path.read_bytes())
        row=proof['sequence']
        predictor.require(proof.get('schema')=='world_reward.form_localization_reuse.v1' and
            proof.get('new_prediction_producer')==new_revision and
            proof.get('original_localization_producer')==report['producer_revision'] and
            proof.get('ground_truth_used') is False and proof.get('private_truth_read') is False and
            proof.get('localization_API_calls')==0 and proof.get('failed_tracking_artifacts_copied') is False and
            proof.get('original_reports_byte_preserved') is True and
            row.get('sequence_id')==base.name and row.get('copied_directory')==str(directory),
            'Original localization reuse ancestry required')
        predictor.require(row['original_artifacts']=={'report.json':result['report_pin'],'boxes.json':result['boxes_pin']} and
            report['source_binding']==proof['original_source_binding'], 'Original copied report/box/source byte pins differ')
        result['reuse_proof']=dict(path=str(path),pin=pin)
    return result
