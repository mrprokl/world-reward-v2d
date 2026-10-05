"""Synthetic arrays and bounded host mocks only; no real labels/media/model."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('hocap_retention_under_test',REPO/'infra/hocap_point_retention_eval.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def arrays(t=5):
    # Three original seeds: two object queries, one HAND query, a Q0 seed and
    # a final-anchor query. Different SEG values do not encode class index+1.
    queries=np.array([[0,.5,.5],[0,.5,1.5],[0,1.5,.5],[t-1,.5,.5]],np.float64)
    return dict(frame_index=np.arange(t,dtype=np.int64),point_indices=np.arange(4,dtype=np.int64),
        query_points=queries,query_birth_frame_indices=queries[:,0].astype(np.int64),
        query_owner_indices=np.array([0,0,0,2],np.int64),query_owner_ids=np.array(['first']*3+['last'],dtype='U128'),
        seed_ids=np.array(['first','empty','last'],dtype='U128'),seed_birth_frame_indices=np.array([0,0,t-1],np.int64),
        tracks=np.tile(np.array([.5,.5],np.float32),(4,t,1)),visible=np.ones((4,t),bool),image_size=np.array([2,2],np.int64))


def labels(t=5):
    return [(np.array([[4,8],[13,0]],np.uint8),{4:0,8:1,13:64})for _ in range(t)]


def evaluate(a,frames=None):
    frames=labels(len(a['frame_index']))if frames is None else frames
    return m.evaluate_clip(np,a,lambda t,shape:frames[t],deadline=time.monotonic()+10)


def test_frozen_protocol_primary_catalog_no_threshold_or_rematching():
    b=(REPO/m.PROTOCOL).read_bytes();assert dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())==m.PROTOCOL_PIN
    assert len(m.CATALOG)==66 and m.CATALOG[-2:]==('RIGHT_HAND','LEFT_HAND')
    p=json.loads(b);assert p['private_evaluation']['archive']['sha256']=='8dc68344d05f75f45bc1e68f95b8093353785f8ec123344135c50e36ed17425c'
    assert m.BUDGET==300 and m.GRACE==60 and m.IMAGE.endswith('10d1e4')


def test_mapping_sorted_positive_native_values_not_class_plus_one():
    seg=np.array([[7,0],[99,50]],np.uint16);inds=np.array([3,0,65],np.int32);names=np.array([m.CATALOG[i]for i in inds])
    result=m.label_arrays(np,seg,inds,names,(2,2));assert result=={7:3,50:0,99:65}
    assert seg.dtype==np.uint16 and inds.dtype==np.int32
    for value in (seg,inds,names):assert value.flags.writeable


@pytest.mark.parametrize('fault',['boolseg','floatseg','negative','shape','boolclass','objectnames','duplicate','wrongname','cardinality'])
def test_reference_unknown_or_ambiguous_is_inconclusive_not_background(fault):
    seg=np.array([[2,0],[8,0]],np.int16);inds=np.array([0,1],np.int64);names=np.array(m.CATALOG[:2])
    if fault=='boolseg':seg=seg.astype(bool)
    elif fault=='floatseg':seg=seg.astype(np.float32)
    elif fault=='negative':seg[0,0]=-1
    elif fault=='shape':seg=seg[:1]
    elif fault=='boolclass':inds=inds.astype(bool)
    elif fault=='objectnames':names=names.astype(object)
    elif fault=='duplicate':inds[:]=0;names[:]=m.CATALOG[0]
    elif fault=='wrongname':names[0]=m.CATALOG[1]
    else:inds=inds[:1];names=names[:1]
    with pytest.raises(m.ReferenceError):m.label_arrays(np,seg,inds,names,(2,2))


def test_strict_future_all_missing_occluded_wrong_background_hand_and_absent_denominators():
    a=arrays();a['tracks'][0,1]=[1.5,.5]  # wrong object
    a['tracks'][0,2]=[.5,1.5]  # HAND
    a['tracks'][0,3]=[1.5,1.5] # background
    a['tracks'][0,4]=[-1,.5]   # outgrid
    a['visible'][1,2]=False
    frame=labels();frame[3]=(np.array([[0,8],[13,0]],np.uint8),{8:1,13:64}) # class0 absent
    b=copy.deepcopy(a);result=evaluate(a,frame)
    assert result['possible']==8 and result['A']['possible']==result['B']['possible']==8
    assert result['all_query_future_possible']==12 and result['initial_HAND_queries']==1
    assert result['B']['wrong_object']==3 and result['B']['HAND']==1 and result['B']['background']==2
    assert result['B']['hidden']==1 and result['B']['outgrid']==1 and result['B']['reference_absent']==1
    assert result['seed_diagnostics'][1]['status']=='NO_QUERIES' and result['seed_diagnostics'][2]['status']=='NO_FUTURE'
    assert result['per_query'][-1]['future_possible']==0 and result['per_query'][-1]['A']['possible']==0
    for name in a:np.testing.assert_array_equal(a[name],b[name])


def test_all_hidden_or_object_class_absent_never_shrinks_denominator():
    a=arrays();a['visible'][:]=False;r=evaluate(a)
    assert r['B']['correct']==0 and r['B']['hidden']==8 and r['B']['correct_fraction']==0
    assert r['A']['correct']==8 and r['possible']==8
    frames=labels();frames[1:]=[(np.zeros((2,2),np.uint8),{})]*4;r=evaluate(arrays(),frames)
    assert r['possible']==8 and r['B']['reference_absent']==8 and r['B']['background']==8


def test_qualify_all_frames_even_after_bad_reference_do_not_score_any_subset():
    a=arrays();seen=[]
    def read(frame,shape):
        seen.append(frame)
        if frame==2:raise m.ReferenceError('manufactured conflict')
        return labels()[frame]
    r=m.evaluate_clip(np,a,read,deadline=time.monotonic()+10)
    assert seen==list(range(5)) and not r['reference_schema_qualified'] and r['reference_error_count']==1
    assert 'A'not in r and m.decision([r,r])=='INCONCLUSIVE'


def test_q0_and_last_anchor_are_explicit_no_observations_not_perfect_score():
    a=arrays();a['query_points']=a['query_points'][3:];a['point_indices']=np.arange(1,dtype=np.int64)
    for n in ('query_birth_frame_indices','query_owner_indices','query_owner_ids','tracks','visible'):a[n]=a[n][3:]
    r=evaluate(a);assert r['possible']==0 and r['A']['correct_fraction']is None and m.decision([r,r])=='INCONCLUSIVE'
    a['query_points']=a['query_points'][:0];a['point_indices']=a['point_indices'][:0]
    for n in ('query_birth_frame_indices','query_owner_indices','query_owner_ids','tracks','visible'):a[n]=a[n][:0]
    r=evaluate(a);assert all(x['status']=='NO_QUERIES'for x in r['seed_diagnostics']) and r['possible']==0


def test_decision_each_clip_correct_strict_wrong_nonstrict_no_pool_rescue():
    c=dict(reference_schema_qualified=True,possible=10,A=dict(correct=5,wrong_object=2),B=dict(correct=6,wrong_object=2))
    assert m.decision([c,c])=='QUALIFIED_SEMANTIC_MASK_POINT_RETENTION'
    second=copy.deepcopy(c);second['B']['correct']=5
    assert m.decision([c,second])=='REJECT'
    second['B']['correct']=9;second['B']['wrong_object']=3
    assert m.decision([c,second])=='REJECT'


def test_queries_permutation_preserves_micro_and_full_row_identity():
    a=arrays();r=evaluate(a);perm=np.array([2,0,3,1]);b=copy.deepcopy(a)
    for k in ('query_points','query_birth_frame_indices','query_owner_indices','query_owner_ids','tracks','visible'):b[k]=b[k][perm]
    b['point_indices']=np.arange(4,dtype=np.int64);other=evaluate(b)
    assert other['A']==r['A'] and other['B']==r['B'] and other['possible']==r['possible']


def test_only_three_npz_fields_opened_private_other_arrays_unreadable(tmp_path):
    # Forbidden object array cannot load under allow_pickle=False; success proves
    # it is not accessed despite remaining present in the original archive.
    raw=io.BytesIO();np.savez(raw,seg_mask=np.array([[42,0]],np.uint16),obj_class_inds=np.array([0],np.int32),
        obj_class_names=np.array([m.CATALOG[0]],dtype='U8'),obj_poses=np.array([object()],dtype=object))
    path=tmp_path/'tiny.zip'
    with zipfile.ZipFile(path,'w')as z:z.writestr('label.npz',raw.getvalue())
    with zipfile.ZipFile(path)as z:
        seg,mapping=m.read_label(np,z,'label.npz',(1,2))
    assert mapping=={42:0} and seg.dtype==np.uint16
    tree=ast.parse((REPO/'infra/hocap_point_retention_eval.py').read_text());read=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='read_label')
    assert not any(isinstance(n,ast.Constant)and n.value=='cam_K'for n in ast.walk(read))


def test_first_reference_read_is_after_both_full_prediction_array_validations(monkeypatch):
    calls=[];proof=dict(native={'predictions':[dict(clip='first',file='first.npz'),dict(clip='second',file='second.npz')]},proof={'bank':{'banks':[]}},folder=Path('/unused'),
        protocol={'private_evaluation':{'archive':{'path':'/PRIVATE_SHOULD_NOT_OPEN','bytes':1,'sha256':'a'*64}}})
    def parse(np,path,row,banks):
        calls.append(row['clip'])
        if row['clip']=='second':raise ValueError('second seal/array invalid')
        return {}
    monkeypatch.setattr(m,'prediction_arrays',parse);monkeypatch.setattr(np,'__version__','1.26.3')
    rt=type('RT',(),{'canonical':lambda *a:pytest.fail('private touched before both arrays')})()
    with pytest.raises(ValueError,match='second seal'):m.evaluate(rt,None,proof,time.monotonic()+10)
    assert calls==['first','second']


def test_cpu_wrapper_and_runtime_no_gpu_models_or_parent_mounts():
    subprocess.run(['bash','-n',str(REPO/'infra/run_hocap_point_retention_eval.sh')],check=True)
    s=(REPO/'infra/hocap_point_retention_eval.py').read_text()
    assert '--gpus'not in s and 'import torch'not in s and 'import hocap_boots_track'not in s
    assert "'--cap-drop','ALL','--cap-add','DAC_READ_SEARCH'"in s and "'--read-only'"in s
    assert 'cam_K'not in s and 'obj_poses'not in s and "'--memory','8g'"in s
    assert '--dispatch' in (REPO/'infra/run_hocap_point_retention_eval.sh').read_text()


def seal(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True));path.chmod(0o444)
    raw=path.read_bytes();return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def public_proof_fixture(tmp_path,monkeypatch):
    """Real RT ledger + small complete producer receipts, not source self-auth."""
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(m,'ROOT',root)
    code=root/'current';code.mkdir();rev='a'*40;old=root/'jobs'/rev/'run_hocap_boots_track/code';old.mkdir(parents=True)
    required=('infra/hocap_boots_track.py','infra/run_hocap_boots_track.sh',m.PROTOCOL,
        'infra/robotap_boots_infer.py','infra/tracker_noise_experiment.py','configs/hocap_boots_protocol_v1.json','configs/robotap_boots_inference_pins.json')
    for n in required:
        dest=old/n;dest.parent.mkdir(parents=True,exist_ok=True)
        if (REPO/n).exists():dest.write_bytes((REPO/n).read_bytes())
    for n in ('revision','source-sha256'):(old.parent/n).write_text((rev if n=='revision'else'b'*64)+'\n');(old.parent/n).chmod(0o444)
    for p in old.rglob('*'):p.chmod(0o555 if p.is_dir()else 0o444)
    old.chmod(0o555)
    helper=code/'infra/mediapipe_cpu_runtime_verify.py';helper.parent.mkdir();helper.write_bytes((REPO/'infra/mediapipe_cpu_runtime_verify.py').read_bytes());helper.chmod(0o444)
    rt=m.runtime(code);sb=rt.source(root,old,rev,'run_hocap_boots_track',required)
    p=json.loads((REPO/m.PROTOCOL).read_text());tinyclips=[dict(clip=c['clip'],frames=5,anchors=[0,1,2,3,4])for c in p['cohort']['clips']];p['cohort']['clips']=tinyclips
    manifest=dict(subject='subject_5',clips=[dict(clip=c['clip'],camera=p['cohort']['camera'],num_frames=5)for c in tinyclips],
        images=[dict(clip=c['clip'],frame_position=i,source_frame_id=i,camera=p['cohort']['camera'])for c in tinyclips for i in range(5)])
    mp=root/'public_manifest.json';mpin=seal(mp,manifest);p['input']['manifest']={**mpin,'path':str(mp)}
    protocol_path=code/m.PROTOCOL;ppin=seal(protocol_path,p);monkeypatch.setattr(m,'PROTOCOL_PIN',ppin)
    oldp=old/'configs/hocap_boots_protocol_v1.json';oldp.chmod(0o644);v=json.loads(oldp.read_text());v['retention_protocol']=ppin;v['frames']=[5,5];op=seal(oldp,v)
    # The modified manufactured original ledger is recomputed independently,
    # so an unknown old-source mutation remains detected by real authentication.
    sb=rt.source(root,old,rev,'run_hocap_boots_track',required)
    rp_rel='results/cpu_runtime.json';rr=dict(status='pass',phase='complete',stage='bootstapir_runtime_verify',child_image_id=m.IMAGE,
        source_rehashed_after=True,gpu_execution=False,checkpoint_read=False,rgb_or_labels_read=False,native_sources={'source.py':{'bytes':1,'sha256':'f'*64}},
        cpu_import={'numpy':'1.26.3','cuda_initialized':False})
    rpin=seal(root/rp_rel,rr);rpc={'runtime':{**rpin,'report_path':rp_rel}}
    seal(code/'configs/robotap_boots_inference_pins.json',rpc)
    (old/'configs/robotap_boots_inference_pins.json').chmod(0o644);seal(old/'configs/robotap_boots_inference_pins.json',rpc)
    sb=rt.source(root,old,rev,'run_hocap_boots_track',required)
    bank=root/m.BANK;rows=[]
    for c in tinyclips:
        for i in range(5):
            path=bank/'bank'/f"{c['clip']}_anchor_{i:06d}.npz";path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'TINY_PUBLIC_OPAQUE');path.chmod(0o444)
            rows.append(dict(clip=c['clip'],file=str(path.relative_to(bank)),**rt.identity(path)))
    safe=dict(source_binding={'original_bank_source':'digest'},public_manifest_identity=mpin)
    sp=seal(bank/'sanitized_input_proof.json',safe)
    br=dict(status='pass',phase='complete',stage='hocap_native_all_mask_bank',all_native_returned_masks=True,private_labels_read=False,
        ground_truth_used=False,amg_calls=10,anchors_retained=10,original_source_public_models_rehashed_after=True,banks=rows,source_binding_sha256=m.digest(safe['source_binding']))
    bp=seal(bank/'report.json',br);bhp=seal(bank/'host.json',dict(status='pass',stage='hocap_amg_host',native_report_identity=bp,owned_cleanup_verified=True,source_public_rehashed_after=True))
    proof=dict(source_binding=sb,retention_protocol_identity=ppin,public_manifest_identity=mpin,protocol_identity=op,
        bank=dict(report=bp,host=bhp,sanitized=sp,source_binding=safe['source_binding'],banks=rows),runtime={'runtime':rpin,'native_sources':rr['native_sources']})
    folder=root/m.PREDICTIONS;safe_pin=seal(folder/'proof.json',proof);preds=[]
    for c in tinyclips:
        name=c['clip']+'_tracks.npz';path=folder/name;path.write_bytes(b'TINY_SEALED_PUBLIC_PREDICTION');path.chmod(0o444)
        preds.append(dict(clip=c['clip'],frames=5,file=name,queries=1,**rt.identity(path)))
    nr=dict(schema='world_reward.hocap_boots_tracks.v1',stage='hocap_native_all_seed_full_t_tracks',status='pass',phase='complete',producer_revision=rev,
        image_id=m.IMAGE,model_loads=1,rgb_decodes=1493,native_calls_attempted=2,native_calls_returned=2,native_calls_completed=2,
        all_original_jpegs_decoded_before_model=True,all_source_public_bank_assets_rehashed_after=True,private_labels_read=False,opaque_metadata_read=False,
        calibration_read=False,ground_truth_used=False,challenge_inputs_used=False,oracle_modes=[],quality_verified=False,adoption=False,
        elapsed_seconds=1,proof_identity=safe_pin,source_binding_sha256=m.digest(sb),public_manifest_identity=mpin,native_parameters=v['native'],bank_report_identity=bp,predictions=preds)
    npin=seal(folder/'report.json',nr)
    hp=seal(folder/'host.json',dict(stage='hocap_boots_host',status='pass',producer_revision=rev,image_id=m.IMAGE,native_exit_status=0,
        native_report_identity=npin,owned_cleanup_verified=True,source_public_bank_assets_rehashed_after=True,private_labels_read=False,quality_verified=False,adoption=False))
    (folder/'.container.cid').write_bytes(b'e'*64);(folder/'.container.cid').chmod(0o444);folder.chmod(0o555)
    return rt,code,npin,hp,folder,old


@pytest.mark.parametrize('fault',[None,'second_prediction','host_failure','source_mutation','image_wrong'])
def test_real_full_source_two_prediction_proof_before_private_path(tmp_path,monkeypatch,fault):
    rt,code,npin,hpin,folder,old=public_proof_fixture(tmp_path,monkeypatch)
    if fault=='second_prediction':p=folder/'20231027_113202_tracks.npz';p.chmod(0o644);p.write_bytes(b'CHANGED');p.chmod(0o444)
    elif fault=='host_failure':p=folder/'host.json';p.chmod(0o644);v=json.loads(p.read_text());v['status']='fail';hpin=seal(p,v)
    elif fault=='source_mutation':p=old/'infra/robotap_boots_infer.py';p.chmod(0o644);p.write_bytes(p.read_bytes()+b'\n');p.chmod(0o444)
    elif fault=='image_wrong':p=folder/'report.json';p.chmod(0o644);v=json.loads(p.read_text());v['image_id']='sha256:'+'f'*64;npin=seal(p,v)
    private_calls=[];real=rt.identity
    def identity(path,*a,**kw):
        if 'quarantine'in Path(path).parts:private_calls.append(path);pytest.fail('Private file before both authentic predictions')
        return real(path,*a,**kw)
    monkeypatch.setattr(rt,'identity',identity)
    if fault:
        with pytest.raises(ValueError):m.authenticate(rt,code,npin,hpin)
    else:
        result=m.authenticate(rt,code,npin,hpin);assert len(result['native']['predictions'])==2 and result['original']==old
    assert not private_calls


def test_public_array_reader_exact_dtype_seed_zeros_fulltimeline(tmp_path):
    sys.path.insert(0,str(REPO/'infra'))
    # Reuse only a manufactured bank builder, never real prediction/model code.
    test_spec=importlib.util.spec_from_file_location('tiny_boots_fixtures',REPO/'tests/test_hocap_boots_track.py')
    fixture=importlib.util.module_from_spec(test_spec);test_spec.loader.exec_module(fixture)
    p,rows=fixture.manufacture_banks(tmp_path);prepared=fixture.m.prepare_queries(rows,p,time.monotonic()+10)[0]
    t=prepared['frames'];h,w=prepared['height'],prepared['width'];prepared['video']=np.zeros((t,h,w,3),np.uint8)
    report=fixture.base_report();a=fixture.fake_prediction(prepared,report);del a['static_tracks'],a['static_visible'];a.update(prepared['metadata'])
    path=tmp_path/'prediction.npz';np.savez(path,**a)
    row=dict(queries=prepared['query_count'],seeds=len(a['seed_ids']),frames=t,zero_query_seeds=5)
    banks=[(Path(p['bank'])/r['file'],r)for r in rows if r['clip']==prepared['clip']]
    got=m.prediction_arrays(np,path,row,banks)
    assert len(got)==19 and all(not v.flags.writeable for v in got.values()) and len(got['point_indices'])>32
    a['tracks']=a['tracks'].astype(np.float64);np.savez(path,**a)
    with pytest.raises(ValueError):m.prediction_arrays(np,path,row,banks)


def test_global_all_reference_frames_qualified_before_any_metric(monkeypatch):
    calls=[];a=arrays();npver=np.__version__;monkeypatch.setattr(np,'__version__','1.26.3')
    proof=dict(native={'predictions':[dict(clip='first',file='first.npz',frames=5),dict(clip='second',file='second.npz',frames=5)]},proof={'bank':{'banks':[]}},
        folder=Path('/public'),protocol={'cohort':{'subject':'subject_5','camera':'cam'},'private_evaluation':{'archive':{'path':'/private/labels.zip','bytes':1,'sha256':'a'*64}}},files={})
    monkeypatch.setattr(m,'prediction_arrays',lambda *args:a)
    class File:
        def __init__(self,mode):self.st_mode=mode
    original_stat=Path.stat;monkeypatch.setattr(Path,'stat',lambda p:File(0o700 if p.name=='private'else 0o400))
    rt=type('RT',(),{'canonical':lambda _,p:p,'identity':lambda *args:{'bytes':1,'sha256':'a'*64}})()
    class Z:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def infolist(self):
            from types import SimpleNamespace
            return [SimpleNamespace(filename=f'subject_5/{c}/cam/label_{i:06d}.npz')for c in ('first','second')for i in range(5)]
    monkeypatch.setattr(m.zipfile,'ZipFile',lambda *args:Z())
    def read(np,z,name,shape):calls.append(('reference',name));return labels()[int(name[-10:-4])]
    monkeypatch.setattr(m,'read_label',read)
    original=m.evaluate_clip
    def score(*args,**kw):
        assert len([x for x in calls if x[0]=='reference'])>=10;calls.append(('metric',''));return original(*args,**kw)
    monkeypatch.setattr(m,'evaluate_clip',score)
    r=m.evaluate(rt,None,proof,time.monotonic()+10)
    assert r['decision']=='REJECT' and calls[10][0]=='metric'


def test_late_receipt_publication_never_leaves_pass(tmp_path,monkeypatch):
    monkeypatch.setattr(m.time,'monotonic',lambda:100)
    path=tmp_path/'receipt.json'
    with pytest.raises(TimeoutError):m.publish(None,path,{'status':'pass','phase':'complete'},99)
    assert json.loads(path.read_text())['status']=='fail' and path.stat().st_mode&0o777==0o444


@pytest.mark.parametrize('native_fail',[False,True])
def test_mock_real_host_shared_deadline_narrow_private_mount_owned_cleanup(tmp_path,monkeypatch,native_fail):
    from types import SimpleNamespace
    rt,code,npin,hpin,folder,old=public_proof_fixture(tmp_path,monkeypatch)
    proof=m.authenticate(rt,code,npin,hpin);root=m.ROOT;rev='c'*40
    private=tmp_path/'quarantine/labels.zip';private.parent.mkdir();private.parent.chmod(0o700);private.write_bytes(b'TINY_OPAQUE_PRIVATE');private.chmod(0o400)
    private_pin=rt.identity(private);proof['protocol']['private_evaluation']['archive']={**private_pin,'path':str(private)}
    monkeypatch.setattr(m,'runtime',lambda code:rt);monkeypatch.setattr(m,'source',lambda *args:{'frozen_source':True})
    monkeypatch.setattr(m,'authenticate',lambda *args:copy.deepcopy(proof));monkeypatch.setattr(m.sys,'platform','linux')
    monkeypatch.setattr(m.os,'geteuid',lambda:0);monkeypatch.setattr(m.os,'uname',lambda:SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',rev);monkeypatch.setenv('WR_ROOT',str(root))
    calls=[];cid='e'*64
    def run(argv,**kw):
        calls.append(argv)
        if argv[:3]==['docker','image','inspect']:return SimpleNamespace(returncode=0,stdout=(m.IMAGE+'\n').encode(),stderr=b'')
        if argv[:2]==['docker','inspect']:return SimpleNamespace(returncode=1,stdout=b'\n',stderr=('error: no such object: '+cid+'\n').encode())
        assert argv[:2]==['docker','run'] and '--gpus'not in argv
        assert f'type=bind,src={private},dst={private},readonly'in argv
        assert not any('src='+str(private.parent)+','in v for v in argv)
        assert any(v.startswith('WR_RETENTION_DEADLINE=')for v in argv)
        assert argv.count('--native')==1 and '--dispatch'not in argv
        out=root/'results'/('hocap-point-retention-'+rev);(out/'container.cid').write_bytes(cid.encode())
        if native_fail:return SimpleNamespace(returncode=1,stdout=b'',stderr=b'EXACT MANUFACTURED FAILURE')
        nr=dict(stage='hocap_point_retention_native',status='pass',phase='complete',producer_revision=rev,source_binding={'frozen_source':True},
            prediction_report_identity=npin,prediction_host_identity=hpin,source_prediction_reference_rehashed_after=True,image_id=m.IMAGE,
            gpu_used=False,model_executed=False,quality_3D_verified=False,adoption=False,decision='INCONCLUSIVE',clips=[{},{}],
            private_keys_accessed=sorted(m.LABEL_KEYS),private_archive_identity=private_pin)
        seal(out/'native.json',nr);return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
    monkeypatch.setattr(m.subprocess,'run',run)
    argv=['--dispatch','--prediction-report-bytes',str(npin['bytes']),'--prediction-report-sha256',npin['sha256'],
          '--prediction-host-bytes',str(hpin['bytes']),'--prediction-host-sha256',hpin['sha256']]
    try:
        if native_fail:
            with pytest.raises(ValueError,match='CPU evaluation failed'):m.main(argv)
        else:assert m.main(argv)['status']=='pass'
    finally:m.signal.alarm(0)
    out=root/'results'/('hocap-point-retention-'+rev);host=json.loads((out/'report.json').read_text())
    assert host['native_exit_status']==int(native_fail) and host['owned_cleanup_verified']
    assert host['status']==('fail'if native_fail else'pass') and out.stat().st_mode&0o777==0o555
    assert host['native_stderr_tail']==('EXACT MANUFACTURED FAILURE'if native_fail else None)
    assert not any(c[:2]==['docker','rm']for c in calls)
