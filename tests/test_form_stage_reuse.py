"""Tiny byte/source prefix resume tests; no model, RGB, cloud or reference reads."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT/'infra'), str(ROOT/'src')]
import form_hoi_external_predict as p
import form_prediction_reuse as loc
import form_stage_reuse as reuse
from test_form_prediction_reuse import fixture as local_fixture, rewrite, OLD as LOCAL, NEW

PREVIOUS = 'd'*40


def fixture(tmp_path, monkeypatch, *, changed_camera=False):
    code, loccode, rows, locbinding = local_fixture(tmp_path, monkeypatch)
    oldcode = tmp_path/'jobs'/PREVIOUS/p.ENTRY/'code'
    raw = (code/'infra/form_hoi_external_predict.py').read_bytes()
    old_inventory = loc.helper_inventory(raw)
    for base in (code, oldcode):
        for name in old_inventory:
            path = base/name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT/name).read_bytes()); path.chmod(0o444)
    binding = dict(producer_revision=PREVIOUS, closure_sha256='e'*64, helpers={})
    monkeypatch.setattr(p, 'source', lambda root, c, revision, entry, helpers:
        locbinding if revision == LOCAL else binding)
    loc.reuse(p, oldcode, PREVIOUS, LOCAL, rows)
    expected = {}
    for row in rows:
        expected[row['sequence_id']] = {'video_pin': {'bytes': 456, 'sha256': 'f'*64}, 'full_source_frames': 764}
    firstbase = tmp_path/'results'/('form-hoi-external-predict-'+PREVIOUS)/rows[0]['sequence_id']
    lineage = loc.localization_lineage(p, firstbase, PREVIOUS)
    for stage in reuse.PREFIX:
        d = firstbase/stage; d.mkdir()
        payloads = ['payload.bin'] if stage != 'body_depth' else ['body.npz', 'depth/000000.npz', 'gauge.json']
        artifacts = {}
        for name in payloads:
            file = d/name; file.parent.mkdir(parents=True, exist_ok=True)
            artifacts[name] = p.seal(file, lambda stream, data=(stage+name).encode(): stream.write(data))
        report = dict(status='complete', stage=stage, producer_revision=PREVIOUS, source_binding=binding,
            sequence_id=rows[0]['sequence_id'], input_pin=rows[0]['pin'], video_pin={'bytes': 456, 'sha256': 'f'*64},
            dataset='nvidia/form-hoi', original_frame_indices=list(range(96)), full_source_frames=764,
            ground_truth_used=False, private_truth_read=False, reference_inputs_mounted=False,
            hand_labeled_test=False, oracle_modes=[], artifacts=artifacts, source_rehashed_after=True,
            localization_source=lineage)
        p.save_json(d/'report.json', report)
    failed = firstbase/'prepare'; failed.mkdir(); p.save_json(failed/'report.json', dict(status='fail', stage='prepare'))
    (failed/'unsealed-partial').write_bytes(b'failed-never-copied')
    cohort = tmp_path/'results'/f'form-hoi-external-cohort-{PREVIOUS}-all.json'
    p.save_json(cohort, dict(status='fail', stage='all', producer_revision=PREVIOUS,
        source_binding=binding, ground_truth_used=False, private_truth_read=False,
        sequence_order=[r['sequence_id'] for r in rows],
        sequences=[dict(sequence_id=rows[0]['sequence_id'], status='fail')]+
            [dict(sequence_id=r['sequence_id'], status='not_started') for r in rows[1:]]))
    if changed_camera:
        path = code/'infra/camera_render.py'; path.chmod(0o600)
        path.write_bytes(path.read_bytes()+b'\n# independently checked raster-only change\n'); path.chmod(0o444)
    return code, oldcode, rows, firstbase, cohort


def test_byte_exact_three_stages_all_four_API_preserved_no_failed_prepare(tmp_path, monkeypatch):
    code, oldcode, rows, first, _ = fixture(tmp_path, monkeypatch)
    proof = reuse.reuse(p, code, NEW, PREVIOUS, rows)
    assert proof['ordered_successful_prefix'] == ['track', 'body_depth', 'object']
    assert proof['localization_API_calls'] == proof['model_calls'] == 0
    assert proof['original_localization_producer'] == LOCAL
    assert proof['incomplete_cohort_scientific_resampling'] is False
    for i, row in enumerate(rows):
        base = tmp_path/'results'/('form-hoi-external-predict-'+NEW)/row['sequence_id']
        assert {f.name for f in base.iterdir()} == ({'localize', 'localization_reuse.json', 'stage_reuse.json', *reuse.PREFIX}
            if i == 0 else {'localize', 'localization_reuse.json'})
        lineage = loc.localization_lineage(p, base, NEW)
        assert lineage['producer_revision'] == LOCAL
        for name in ('boxes.json', 'report.json'):
            original = tmp_path/'results'/('form-hoi-external-predict-'+LOCAL)/row['sequence_id']/'localize'/name
            assert p.artifact(base/'localize'/name) == p.artifact(original)
    for item in proof['stages']:
        for name, pin in item['artifacts'].items():
            assert p.artifact(Path(item['copied_directory'])/name) == pin
            assert (Path(item['copied_directory'])/name).read_bytes() == (first/item['stage']/name).read_bytes()
        lineage = reuse.original_stage_lineage(p, tmp_path/'results'/('form-hoi-external-predict-'+NEW)/rows[0]['sequence_id'], item['stage'], NEW)
        assert lineage['producer_revision'] == PREVIOUS and lineage['camera_QA_source'] == {}
    assert (first/'prepare/unsealed-partial').read_bytes() == b'failed-never-copied'


def test_changed_camera_only_explicitly_original_QA_not_claimed_new_numeric_equivalence(tmp_path, monkeypatch):
    code, _, rows, _, _ = fixture(tmp_path, monkeypatch, changed_camera=True)
    with pytest.raises(ValueError): reuse.reuse(p, code, NEW, PREVIOUS, rows)
    proof = reuse.reuse(p, code, NEW, PREVIOUS, rows, allow_body_depth_camera_change=True)
    camera = proof['camera_render_change']
    assert camera['original'] != camera['downstream']
    assert camera['original_QA_reused_byte_exact'] and not camera['numeric_equivalence_claimed']
    base = tmp_path/'results'/('form-hoi-external-predict-'+NEW)/rows[0]['sequence_id']
    assert reuse.original_stage_lineage(p, base, 'body_depth', NEW)['camera_QA_source'] == camera
    assert reuse.original_stage_lineage(p, base, 'object', NEW)['camera_QA_source'] == {}


@pytest.mark.parametrize('fault', ['track_algorithm', 'body_algorithm', 'object_algorithm', 'common_algorithm',
    'worker_context', 'config', 'dependency', 'missing_prefix', 'failed_prefix', 'late_sequence', 'GT', 'private', 'payload',
    'API_input', 'API_proof', 'source_report', 'terminal_running', 'terminal_order', 'destination'])
def test_all_inputs_qualified_before_first_copy_no_rerolls_or_result_selection(tmp_path, monkeypatch, fault):
    code, oldcode, rows, first, cohort = fixture(tmp_path, monkeypatch)
    if fault.endswith('algorithm'):
        name = {'track_algorithm': 'track', 'body_algorithm': 'body_depth', 'object_algorithm': 'object_mesh',
            'common_algorithm': 'frames'}[fault]
        path = code/'infra/form_hoi_external_predict.py'; path.chmod(0o600)
        raw = path.read_text(); raw = raw.replace('def '+name+'(', 'def '+name+'_changed(', 1)
        path.write_text(raw); path.chmod(0o444)
    elif fault == 'worker_context':
        path = code/'infra/form_hoi_external_predict.py'; path.chmod(0o600)
        raw = path.read_text().replace('with torch.inference_mode():track(p,c,base,out,report)',
            'with torch.no_grad():track(p,c,base,out,report)')
        path.write_text(raw); path.chmod(0o444)
    elif fault in ('config', 'dependency'):
        path = code/('configs/form_hoi_external_predict_v1.json' if fault == 'config' else 'src/world_reward/pointmap.py')
        path.chmod(0o600); path.write_bytes(path.read_bytes()+b'\n'); path.chmod(0o444)
    elif fault == 'missing_prefix': (first/'object/report.json').unlink()
    elif fault == 'failed_prefix': rewrite(first/'object/report.json', lambda r: r.update(status='fail'))
    elif fault == 'late_sequence':
        late = first.parent/rows[1]['sequence_id']/'track'; late.mkdir(); p.save_json(late/'report.json', dict(status='complete'))
    elif fault in ('GT', 'private', 'source_report'):
        key = {'GT': 'ground_truth_used', 'private': 'reference_inputs_mounted', 'source_report': 'source_binding'}[fault]
        rewrite(first/'body_depth/report.json', lambda r: r.update({key: True}))
    elif fault == 'payload':
        path = first/'track/payload.bin'; path.chmod(0o600); path.write_bytes(b'changed'); path.chmod(0o444)
    elif fault == 'API_input': rows[-1]['pin']['sha256'] = '9'*64
    elif fault == 'API_proof': rewrite(first/'localization_reuse.json', lambda r: r.update(new_prediction_producer=NEW))
    elif fault == 'terminal_running': rewrite(cohort, lambda r: r.update(status='running'))
    elif fault == 'terminal_order': rewrite(cohort, lambda r: r.update(sequence_order=list(reversed(r['sequence_order']))))
    else:
        base = tmp_path/'results'/('form-hoi-external-predict-'+NEW)/rows[-1]['sequence_id']; base.mkdir(parents=True)
        (base/'keep').write_bytes(b'not-overwritten')
    with pytest.raises((ValueError, FileNotFoundError)): reuse.reuse(p, code, NEW, PREVIOUS, rows)
    dest = tmp_path/'results'/('form-hoi-external-predict-'+NEW)
    assert not dest.exists() or {f.name for f in dest.iterdir()} == {rows[-1]['sequence_id']}


def test_explicit_actual_terminal_stop_pin_alternative_without_guessing_job_alive(tmp_path, monkeypatch):
    code, _, rows, _, cohort = fixture(tmp_path, monkeypatch); cohort.unlink()
    path = tmp_path/'results/owned-stop.json'
    pin = p.save_json(path, dict(schema='world_reward.form_prediction_terminal_stop.v1',
        producer_revision=PREVIOUS, ground_truth_used=False, private_truth_read=False,
        sequence_order=[r['sequence_id'] for r in rows], owned_driver_inactive=True,
        owned_stage_container_absent=True))
    for wrong in (dict(pin, sha256='a'*64),):
        with pytest.raises(ValueError): reuse.reuse(p, code, NEW, PREVIOUS, rows, stop=dict(path=str(path), pin=wrong))
    proof = reuse.reuse(p, code, NEW, PREVIOUS, rows, stop=dict(path=str(path), pin=pin))
    assert proof['terminal_receipt']['pin'] == pin


def test_new_stage_lineage_and_tampered_original_proof_failclosed(tmp_path, monkeypatch):
    code, _, rows, _, _ = fixture(tmp_path, monkeypatch)
    reuse.reuse(p, code, NEW, PREVIOUS, rows)
    base = tmp_path/'results'/('form-hoi-external-predict-'+NEW)/rows[0]['sequence_id']
    rewrite(base/'stage_reuse.json', lambda r: r.update(original_stage_producer='f'*40))
    with pytest.raises(ValueError): reuse.original_stage_lineage(p, base, 'track', NEW)
    fresh = base/'prepare'; fresh.mkdir()
    p.save_json(fresh/'report.json', dict(status='complete', stage='prepare', producer_revision=NEW,
        source_binding={}, ground_truth_used=False, private_truth_read=False, artifacts={}))
    assert reuse.original_stage_lineage(p, base, 'prepare', NEW)['producer_revision'] == NEW


def test_actual_completed_initialization_source_is_identical_to_failed80():
    import subprocess
    raw=subprocess.check_output(['rtk','git','show',reuse.PREPARE_PREFIX_PRODUCER+':infra/form_hoi_external_predict.py'],cwd=ROOT)
    current=(ROOT/'infra/form_hoi_external_predict.py').read_bytes()
    assert reuse.initialization_source(raw,'prepare') == reuse.initialization_source(current,'prepare_initialization')
    assert loc.function_identity(raw,'pose_initializer') == loc.function_identity(current,'pose_initializer')
    with pytest.raises(AssertionError):
        assert reuse.initialization_source(raw,'prepare') == reuse.initialization_source(current.replace(b'decode_batch_size=16',b'decode_batch_size=8',1),'prepare_initialization')


def test_nested_successful_stage_ancestry_is_preserved_without_relabeling(tmp_path, monkeypatch):
    code,oldcode,rows,first,_=fixture(tmp_path,monkeypatch)
    middle='e'*40; middlecode=tmp_path/'jobs'/middle/p.ENTRY/'code'
    import shutil
    shutil.copytree(oldcode,middlecode)
    # First create real125-like prefix copying; then qualify/copy it again80-like.
    real_source=p.source
    binding=dict(producer_revision=middle,closure_sha256='9'*64,helpers={})
    monkeypatch.setattr(p,'source',lambda root,c,revision,entry,helpers:
        binding if revision==middle else real_source(root,c,revision,entry,helpers))
    reuse.reuse(p,middlecode,middle,PREVIOUS,rows)
    cohort=tmp_path/'results'/f'form-hoi-external-cohort-{middle}-all.json'
    p.save_json(cohort,dict(status='fail',stage='all',producer_revision=middle,source_binding=binding,
        ground_truth_used=False,private_truth_read=False,sequence_order=[r['sequence_id'] for r in rows]))
    proof=reuse.reuse(p,code,NEW,middle,rows)
    assert proof['original_stage_producer']==middle
    assert all(r['original_stage_producer']==PREVIOUS for r in proof['stages'])
    base=tmp_path/'results'/('form-hoi-external-predict-'+NEW)/rows[0]['sequence_id']
    assert reuse.original_stage_lineage(p,base,'track',NEW)['producer_revision']==PREVIOUS
    assert p.read_stage(base,'track')[1]['producer_revision']==PREVIOUS


def test_prefix_approval_is_independent_exact_and_nonprediction(tmp_path, monkeypatch):
    from types import SimpleNamespace
    rev=reuse.PREPARE_PREFIX_PRODUCER; sid='FORM_DEV_0'; directory=tmp_path/'results'/('form-hoi-external-predict-'+rev)/sid/'prepare'
    directory.mkdir(parents=True)
    path=tmp_path/'results/approval.json'; data=dict(schema='world_reward.form_prepare_prefix_approval.v1',producer_revision=rev,
        sequence_id=sid,failed_report=dict(path=str(directory/'report.json'),pin=reuse.PREPARE_PREFIX_REPORT),artifacts=reuse.PREPARE_PREFIX_FILES)
    approval_pin=p.save_json(path,data)
    fake=SimpleNamespace(ROOT=tmp_path,canonical=p.canonical,require=p.require,strict=p.strict,
        artifact=lambda f: approval_pin if f==path else reuse.PREPARE_PREFIX_REPORT)
    # No first-seen artifact admissibility; even a valid manifest cannot bypass actual source/report checks.
    with pytest.raises(ValueError, match='failed initializer inventory'): reuse.qualify_prepare_prefix(fake,tmp_path/'code',rev,dict(sequence_id=sid),dict(path=path,**approval_pin))
    rewrite(path,lambda r:r['artifacts']['object_prior.npz'].update(sha256='f'*64)); approval_pin=p.artifact(path)
    with pytest.raises(ValueError,match='independently pinned'): reuse.qualify_prepare_prefix(fake,tmp_path/'code',rev,dict(sequence_id=sid),dict(path=path,**approval_pin))


@pytest.mark.parametrize('fault',[None,'not_SO3','scale_mismatch','source_stage_changed'])
def test_completed_prefix_consumption_validates_geometry_without_model_or_pose_replay(tmp_path,monkeypatch,fault):
    import json
    import numpy as np
    import pickle
    from types import SimpleNamespace
    from test_shared_identity import native_parameters
    rev=NEW; base=tmp_path/'results/new/FORM_DEV_0'; d=base/'prepare_prefix'; d.mkdir(parents=True)
    out=base/'prepare'; out.mkdir(); (base/'object').mkdir(); (base/'body_depth').mkdir()
    params=p.share_first_frame_identity(native_parameters(),96)
    initializer=dict(params,frames=[f'{i:06d}' for i in range(96)],body_model='mhr',
        metadata=dict(public_sequence_id=base.name,shared_identity_geometry_redecoded=True),
        mhr_joints=np.zeros((96,127,3),np.float32),mhr_keypoints=np.zeros((96,70,3),np.float32))
    pins={'shared_initializer.pkl':p.seal(d/'shared_initializer.pkl',lambda f:pickle.dump(initializer,f,protocol=4))}
    v=np.asarray([[0,0,0],[.3,0,0],[0,.3,0]],np.float32); f=np.asarray([[0,1,2]],np.int64)
    r=np.broadcast_to(np.eye(3,dtype=np.float32),(96,3,3)).copy()
    if fault=='not_SO3':r[3,0,0]=2
    pins['object_prior.npz']=p.save_npz(d/'object_prior.npz',vertices=v,faces=f,rotation=r,
        translation=np.ones((96,3),np.float32),observed=np.ones(96,np.bool_),frame_index=np.arange(96,dtype=np.int64))
    pins['object_metric.glb']=p.seal(d/'object_metric.glb',lambda stream:stream.write(b'unit-test-mesh-placeholder'))
    (base/'object/object.glb').write_bytes(b'unit-test-object-placeholder')
    mesh=SimpleNamespace(vertices=v,faces=f)
    monkeypatch.setitem(sys.modules,'trimesh',SimpleNamespace(load=lambda *_a,**_k:mesh))
    p.save_json(base/'object/transform.json',dict(scale=[2,2,2] if fault=='scale_mismatch' else [1,1,1]))
    p.save_json(base/'body_depth/gauge.json',dict(K=np.eye(3).tolist(),alignment=dict(shared_scale=1.)))
    lineage=dict(producer_revision=PREVIOUS,report_pin={'bytes':1,'sha256':'a'*64},source_binding={})
    p.save_json(d/'report.json',dict(schema='world_reward.form_prepare_prefix_reuse.v1',new_prediction_producer=rev,
        sequence_id=base.name,artifacts=pins,original_failed_report=reuse.PREPARE_PREFIX_REPORT,
        original_public_input=dict(pin={'bytes':1,'sha256':'f'*64}),consumed_intermediate_sources={stage:lineage for stage in reuse.PREFIX},
        final_prediction=False,native_export_complete=False,failed_prepare_or_fit_copied=False,ground_truth_used=False,private_truth_read=False,
        decoder_identity={},body_assets={},inference_source_identity={}))
    monkeypatch.setattr(reuse,'PREPARE_PREFIX_FILES',pins)
    monkeypatch.setattr(reuse,'original_stage_lineage',lambda *_a:dict(lineage,report_pin={}) if fault=='source_stage_changed' else lineage)
    report=dict(input_pin={'bytes':1,'sha256':'f'*64})
    if fault:
        with pytest.raises(ValueError):reuse.consume_prepare_prefix(p,base,out,report,dict(sequence_id=base.name),rev)
        assert not tuple(out.iterdir())
    else:
        result=reuse.consume_prepare_prefix(p,base,out,report,dict(sequence_id=base.name),rev)
        assert result[1].tobytes()==v.tobytes() and set(x.name for x in out.iterdir())==set(pins)
        assert report['prepare_initializer_reuse']['model_calls']==0 and not report['prepare_initializer_reuse']['pose_fit_replayed']
