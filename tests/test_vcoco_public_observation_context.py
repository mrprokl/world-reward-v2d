"""Authored48 saved endpoint ABI; no actual photos/model/qualification."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import time

import pytest
import vcoco_public_observation_context as p
import test_vcoco_fit_cal_endpoint_observations as authored


def save(path,value):
    raw=value if type(value)is bytes else p.endpoint.original.encode(value)
    if path.exists():path.chmod(0o600)
    path.write_bytes(raw);path.chmod(0o400);return p.rt.identity(path)


def fixture(tmp_path,monkeypatch):
    import numpy as np
    monkeypatch.setattr(np,'savez',np.savez_compressed)  # authored zeros only; all native shapes/bytes preserved
    context,inputs,input_pin=authored.fixture(tmp_path);events=[]
    models=authored.fake_model(monkeypatch,events)
    records=p.endpoint.observe(context,inputs,models,time.monotonic()+30)
    deadline=time.monotonic()+30;code=tmp_path/'caller';code.mkdir();output=tmp_path/'new';output.mkdir()
    current=dict(producer_revision='b'*40,helpers={'current':'only'})
    old=dict(producer_revision='a'*40,helpers={'old':'only'})
    proof=dict(source=old,inputs_identity=input_pin,images=48,profile=p.endpoint.PROFILE,image_id=p.endpoint.original.IMAGE,owl_runtime={'actual':'authored'})
    proof_pin=save(context.output/'proof.json',proof)
    native=dict(schema=p.endpoint.SCHEMA,stage='native_full48_endpoint_observations',status='pass',phase='complete',
        producer_revision='a'*40,image_id=p.endpoint.original.IMAGE,profile=p.endpoint.PROFILE,
        model_loads=2,models_released=True,source_inputs_assets_runtime_rehashed_after=True,images=records,
        runtime_identity=proof['owl_runtime'],proof_identity=proof_pin,
        person_forward_calls=48,image_embed_calls=48,objectness_calls=48,box_calls=48)
    flags=('reference_metadata_read','split_metadata_read','FIT_performed','ground_truth_used','challenge_inputs_used','actor_selection_performed','ownership_verified','quality_verified','adoption')
    native.update({k:False for k in flags});native_pin=save(context.output/'native.json',native)
    report=dict(schema=p.endpoint.SCHEMA,stage='full48_endpoint_observations_host',status='pass',source_binding=old,
        producer_revision='a'*40,image_id=p.endpoint.original.IMAGE,acquired_images=48,native_exit_status=0,
        outputs_sealed=True,owned_cleanup_verified=True,source_inputs_assets_runtime_rehashed_after=True,
        native_report_identity=native_pin,native_images=records,public_inputs_identity=input_pin,**{k:False for k in flags})
    report_pin=save(context.output/'report.json',report);save(context.output/'.container.cid',b'c'*64);context.output.chmod(0o500)
    monkeypatch.setattr(p,'source_identity',lambda code,binding:None)
    c=p.PublicObservationContext(context.inputs,context.output,output)
    args=dict(code=code,current_source=current,endpoint_source=old,report_pin=report_pin,deadline=deadline)
    return c,args,inputs,records,report,native


def test_explicit_paths_injective_canonical_and_no_overrides(tmp_path):
    with pytest.raises(ValueError):p.PublicObservationContext(tmp_path,tmp_path/'banks',tmp_path/'out')
    real=tmp_path/'real';real.mkdir();alias=tmp_path/'alias';alias.symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError):p.PublicObservationContext(alias,tmp_path/'b',tmp_path/'o')


def native_fixture(tmp_path,monkeypatch):
    c,args,inputs,records,report,native=fixture(tmp_path,monkeypatch)
    projection=p.authenticate_endpoint(c,**args)
    banks=tmp_path/'native_banks';banks.mkdir()
    for row in projection['banks']:save(banks/row['file'],(c.endpoint_banks/row['file']).read_bytes())
    banks.chmod(0o500);path=tmp_path/'projection.json';pin=save(path,projection)
    native_context=p.PublicObservationContext(c.inputs,banks,c.output)
    ref=p.public_bank_reference(native_context,projection_path=path,projection_pin=pin,code=args['code'],
        native_source={'helpers':{n:dict(bytes=1,sha256='c'*64)for n in p.HELPERS}},deadline=time.monotonic()+30,native_mounts=False)
    return ref,args,projection


def test_complete48_saved_reference_and_immutable_public_returns(tmp_path,monkeypatch):
    ref,args,projection=native_fixture(tmp_path,monkeypatch)
    assert len(ref.images)==len(ref.banks)==48 and ref.banks==projection['banks']
    before=ref._projection;r=ref.banks;r[0]['person_ids']=['foreign'];assert ref._projection==before
    ref.verify(time.monotonic()+30);arrays=p.load_bank(ref,ref.banks[47],time.monotonic()+30)
    assert len(arrays)==17 and arrays['person_model_logits'].shape==(1,900,256)
    raw=ref._projection.decode()
    assert all(k not in raw for k in ('source_binding','acquisition','qualification','producer_revision','private'))


@pytest.mark.parametrize('fault',['unpin','failed','old_source','native_fail','flag','missing','mode','bank','current'])
def test_any_unsealed_or_incomplete_lineage_fails_before_forward(tmp_path,monkeypatch,fault):
    c,args,_,_,report,native=fixture(tmp_path,monkeypatch)
    if fault=='unpin':args['report_pin']['sha256']='f'*64
    elif fault=='failed':report['status']='fail'
    elif fault=='old_source':args['endpoint_source']={'producer_revision':'f'*40}
    elif fault=='native_fail':native['status']='fail'
    elif fault=='flag':native['reference_metadata_read']=True
    elif fault=='missing':c.endpoint_banks.chmod(0o700);(c.endpoint_banks/'image_000047.npz').unlink();c.endpoint_banks.chmod(0o500)
    elif fault=='mode':c.endpoint_banks.chmod(0o700)
    elif fault=='bank':
        q=c.endpoint_banks/'image_000000.npz';q.chmod(0o600);q.write_bytes(b'bad');q.chmod(0o400)
    else:monkeypatch.setattr(p,'source_identity',lambda *a:(_ for _ in()).throw(ValueError('Current source differs')))
    if fault in('failed','native_fail','flag'):
        if fault!='failed':report['native_report_identity']=save(c.endpoint_banks/'native.json',native)
        args['report_pin']=save(c.endpoint_banks/'report.json',report)
    with pytest.raises(ValueError):p.authenticate_endpoint(c,**args)


def test_constructor_replacement_cannot_forge_reference_tables(tmp_path,monkeypatch):
    ref,args,projection=native_fixture(tmp_path,monkeypatch)
    altered=deepcopy(projection);altered['banks'][0]['person_ids']=['foreign']
    with pytest.raises(ValueError):replace(ref,_projection=p.endpoint.original.encode(altered))
    with pytest.raises(ValueError):replace(ref,_states=b'{}')


def test_posthash_all48_and_inode_states_unchanged(tmp_path,monkeypatch):
    ref,args,projection=native_fixture(tmp_path,monkeypatch)
    q=ref.context.inputs/'image_000047.jpg';q.chmod(0o600);q.write_bytes(b'changed');q.chmod(0o400)
    with pytest.raises(ValueError):ref.verify(time.monotonic()+30)


def test_native_public_projection_rejects_private_extra_and_original_receipt_namespace(tmp_path,monkeypatch):
    ref,args,projection=native_fixture(tmp_path,monkeypatch)
    extra=deepcopy(projection);extra['banks'][0]['private_proof']={'split':'FIT'}
    q=ref._projection_path;pin=save(q,extra)
    with pytest.raises(ValueError):p.public_bank_reference(ref.context,projection_path=q,projection_pin=pin,
        code=args['code'],native_source={'helpers':{n:dict(bytes=1,sha256='c'*64)for n in p.HELPERS}},deadline=time.monotonic()+30)


def test_native_never_reads_original_host_reports_and_accepts_local_bind_parent_modes(tmp_path,monkeypatch):
    ref,args,projection=native_fixture(tmp_path,monkeypatch)
    original=tmp_path/'out';original.chmod(0o700);original.rename(tmp_path/'host_only_not_mounted')
    ref.context.inputs.chmod(0o755);ref.context.endpoint_banks.chmod(0o755)
    current=p.rt.strict(ref._native_source)
    native=p.public_bank_reference(ref.context,projection_path=ref._projection_path,
        projection_pin=p.rt.strict(ref._projection_identity),code=args['code'],native_source=current,
        deadline=time.monotonic()+30,native_mounts=True)
    native.verify(time.monotonic()+30)
    assert not hasattr(native,'_deadline') and all('acquire'not in n for n in current['helpers'])


def test_native_original_52_file_namespace_and_host_source_proof_rejected(tmp_path,monkeypatch):
    ref,args,projection=native_fixture(tmp_path,monkeypatch)
    context=p.PublicObservationContext(ref.context.inputs,tmp_path/'out',ref.context.output)
    with pytest.raises(ValueError,match='only48'):
        p.public_bank_reference(context,projection_path=ref._projection_path,
            projection_pin=p.rt.strict(ref._projection_identity),code=args['code'],
            native_source=p.rt.strict(ref._native_source),deadline=time.monotonic()+30)
    with pytest.raises(ValueError,match='host/private'):
        replace(ref,_native_source=p.endpoint.original.encode(args['current_source']))
