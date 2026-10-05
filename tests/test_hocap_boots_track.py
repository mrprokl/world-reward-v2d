"""Manufactured arrays/host spies only: no real images, model, GPU or HO-Cap labels."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'infra'))
spec=importlib.util.spec_from_file_location('hocap_boots_track_under_test',ROOT/'infra/hocap_boots_track.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def seal_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)); path.chmod(0o444)
    return m.rt.identity(path,2<<20)


def policy(tmp_path):
    p=json.loads((ROOT/m.PROTOCOL).read_text())
    p.update(bank=str(tmp_path/'bank'),output=str(tmp_path/'output'),frames=[5,6])
    p['inputs']=str(tmp_path/'inputs')
    return p


def manufacture_banks(tmp_path, *, empty=False, size=(4,4), frames=None):
    p=policy(tmp_path); rows=[]; h,w=size
    if frames is not None: p['frames']=frames
    folder=Path(p['bank'])/'bank'; folder.mkdir(parents=True)
    for clip,t in zip(p['clips'],p['frames']):
        for k in range(5):
            anchor=k*(t-1)//4; frame=dict(clip=clip,frames=t,frame_index=anchor,height=h,width=w)
            masks=[np.zeros((h,w),bool)] if empty else [np.ones((h,w),bool)]*3+[np.zeros((h,w),bool)]
            records=[dict(segmentation=mask,bbox=[0,0,w,h],crop_box=[0,0,w,h],point_coords=[[.5,.5]],
                area=int(mask.sum()),predicted_iou=.9,stability_score=.97) for mask in masks]
            a,audit=m.bank.bank_arrays(records,frame,{**p,'packed_mask_bitorder':'little'})
            name=f"bank/{clip}_anchor_{anchor:06d}.npz"; path=Path(p['bank'])/name
            np.savez(path,**a); path.chmod(0o444)
            rows.append(dict(clip=clip,frames=t,anchor_frame_index=anchor,width=w,height=h,file=name,
                decoded_rgb_sha256='a'*64,**m.rt.identity(path),**audit,elapsed_seconds=.01))
    folder.chmod(0o555)
    return p,rows


def rewrite_npz(path, mutate):
    with np.load(path,allow_pickle=False) as z: a={n:z[n] for n in z.files}
    mutate(a); path.chmod(0o644); np.savez(path,**a); path.chmod(0o444)


def fake_prediction(data, report):
    report['native_calls_attempted']+=1; report['native_calls_returned']+=1
    t,h,w=data['video'].shape[:3]; q=data['query_points']; n=len(q)
    tracks256=np.broadcast_to(np.array([10,20],np.float32),(n,t,2)).copy()
    tracks=(tracks256*np.array([w,h],np.float32))/np.array([256,256],np.float32)
    return dict(tracks=tracks,tracks_256=tracks256,occlusion=np.zeros((n,t),np.float32),
        expected_dist=np.zeros((n,t),np.float32),visible=np.arange(t)[None,:]>q[:,0,None],
        query_points=q.copy(),point_indices=data['point_indices'].copy(),frame_index=np.arange(t,dtype=np.int64),
        static_tracks=np.broadcast_to(q[:,[2,1]][:,None],(n,t,2)).copy(),static_visible=np.ones((n,t),bool))


def base_report():
    return dict(native_calls_attempted=0,native_calls_returned=0,native_calls_completed=0)


def test_protocol_full_bytes_and_predeclared_settings():
    raw=(ROOT/m.PROTOCOL).read_bytes()
    assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==m.PROTOCOL_PIN
    p=json.loads(raw); science=(ROOT/m.SCIENCE).read_bytes()
    assert p['retention_protocol']==dict(bytes=len(science),sha256=hashlib.sha256(science).hexdigest())
    assert p['native']['query_chunk_size']==32 and p['maximum_queries_per_clip']==4096
    assert p['budget_seconds']==1800 and p['cleanup_grace_seconds']==60
    assert p['static_diagnostic_arrays_published'] is False


def test_all_five_banks_mixed_births_above32_empty_seeds_preserved(tmp_path):
    p,rows=manufacture_banks(tmp_path)
    prepared=m.prepare_queries(rows,p,time.monotonic()+20)
    for c in prepared:
        assert c['query_count']==240 > 32
        assert np.array_equal(c['point_indices'],np.arange(240))
        a=c['metadata']; assert len(a['seed_ids'])==20
        assert np.sum(np.diff(a['seed_query_offsets'])==0)==5
        assert np.array_equal(a['seed_anchor_offsets'],[0,4,8,12,16,20])
        assert np.array_equal(a['query_owner_ids'],a['seed_ids'][a['query_owner_indices']])
        assert np.array_equal(c['query_points'][:,0],a['query_birth_frame_indices'])
        assert a['seed_mask_sha256'][0]==a['seed_mask_sha256'][1]  # duplicates not grouped


@pytest.mark.parametrize('kind',['extra','offset','owner','id','birth','query','nan','timeline','crop','score','area','dtype'])
def test_bank_rejects_malformed_evidence_without_fitting(tmp_path,kind):
    p,rows=manufacture_banks(tmp_path); row=rows[0]; path=Path(p['bank'])/row['file']
    def change(a):
        if kind=='extra': a['private_segmentation']=np.zeros(1)
        elif kind=='offset': a['query_offsets'][1]=-1
        elif kind=='owner': a['query_owner_indices'][0]=1
        elif kind=='id': a['mask_ids'][0]='invented'
        elif kind=='birth': a['query_birth_frame_indices'][0]=1
        elif kind=='query': a['query_points'][0,1]+=.25
        elif kind=='nan': a['native_bbox_xywh'][0,0]=np.nan
        elif kind=='timeline': a['frame_index'][1]=0
        elif kind=='crop': a['native_crop_box_xywh'][0,0]=1
        elif kind=='score': a['predicted_iou'][0]=1.1
        elif kind=='area': a['mask_area'][0]-=1
        elif kind=='dtype': a['query_points']=a['query_points'].astype(np.float32)
    rewrite_npz(path,change)
    with pytest.raises(ValueError): m.bank_arrays(path,row,p)


def test_nonzero_packbits_padding_rejected(tmp_path):
    p,rows=manufacture_banks(tmp_path,size=(3,3)); path=Path(p['bank'])/rows[0]['file']
    rewrite_npz(path,lambda a:a['packed_masks'].__setitem__((0,-1),255))
    with pytest.raises(ValueError,match='padding'): m.bank_arrays(path,rows[0],p)


@pytest.mark.parametrize('cap',[0,100])
def test_whole_bank_limit_never_truncates(tmp_path,cap):
    p,rows=manufacture_banks(tmp_path); p['maximum_queries_per_clip']=cap
    with pytest.raises(ValueError,match='no truncation'): m.prepare_queries(rows,p,time.monotonic()+20)


def test_no_observations_fail_before_predictor(tmp_path):
    p,rows=manufacture_banks(tmp_path,empty=True)
    with pytest.raises(ValueError,match='NO_OBSERVATIONS'): m.prepare_queries(rows,p,time.monotonic()+20)


def test_prediction_capacity_gate_before_model(tmp_path):
    p,rows=manufacture_banks(tmp_path); p['maximum_prediction_bytes']=1
    with pytest.raises(ValueError,match='before model'): m.prepare_queries(rows,p,time.monotonic()+20)


def test_two_calls_full_original_timeline_only_static_fields_omitted(tmp_path):
    p,rows=manufacture_banks(tmp_path); prepared=m.prepare_queries(rows,p,time.monotonic()+20)
    for c in prepared: c['video']=np.zeros((c['frames'],c['height'],c['width'],3),np.uint8)
    captured={}; r=base_report()
    def save(clip,arrays):
        captured[clip]={n:v.copy() for n,v in arrays.items()}; return dict(file=clip+'_tracks.npz',bytes=1,sha256='a'*64)
    m.track_all(prepared,fake_prediction,save,r,p,time.monotonic()+20)
    assert r['native_calls_completed']==r['native_calls_attempted']==r['native_calls_returned']==2
    for arrays,c in zip(captured.values(),prepared):
        assert not {'static_tracks','static_visible'}&set(arrays)
        assert set(arrays)=={'tracks','tracks_256','occlusion','expected_dist','visible','query_points','point_indices','frame_index',*c['metadata']}
        assert arrays['tracks'].shape==(240,c['frames'],2)
        assert len(arrays['seed_ids'])==20 and not arrays['visible'][:,0].any()
        assert 'video' not in c


def test_partial_native_error_is_not_empty_or_retry(tmp_path):
    p,rows=manufacture_banks(tmp_path); prepared=m.prepare_queries(rows,p,time.monotonic()+20)
    for c in prepared: c['video']=np.zeros((c['frames'],4,4,3),np.uint8)
    r=base_report(); calls=[]
    def prediction(data,report):
        calls.append(data['video'].shape[0])
        if len(calls)==2:
            report['native_calls_attempted']+=1; raise RuntimeError('manufactured native failure')
        return fake_prediction(data,report)
    with pytest.raises(RuntimeError): m.track_all(prepared,prediction,lambda c,a:{'file':c},r,p,time.monotonic()+20)
    assert len(calls)==2 and len(r['predictions'])==r['native_calls_completed']==1
    assert r['native_calls_attempted']==2 and r['native_calls_returned']==1


def test_full_bank_before_all_predictions_and_budget_fail(tmp_path):
    p,rows=manufacture_banks(tmp_path)
    with pytest.raises(TimeoutError): m.prepare_queries(rows,p,time.monotonic()-1)
    prepared=m.prepare_queries(rows,p,time.monotonic()+20)
    with pytest.raises(ValueError,match='decoded'): m.track_all(prepared,None,None,base_report(),p,time.monotonic()+20)


def public_bank_receipt(tmp_path):
    p,rows=manufacture_banks(tmp_path,frames=[702,791]); rev='a'*40
    safe=dict(schema='world_reward.hocap_amg_public_input_proof.v1',input_report_identity=dict(bytes=1,sha256='b'*64),
        extraction_producer_revision='c'*40,extraction_source_sha256='d'*64,public_manifest_identity=p['public_manifest'],
        frames=1493,clips=[dict(clip=c,camera=p['camera'],num_frames=t) for c,t in zip(p['clips'],p['frames'])],
        private_labels_read=False,calibration_read=False,opaque_metadata_exported=False,challenge_inputs_used=False,
        source_binding={'helpers':{},'closure_sha256':'e'*64,'markers':{}},frontend_proof_sha256='f'*64,
        protocol_identity=m.bank.PROTOCOL_PIN)
    safe_pin=seal_json(Path(p['bank'])/'sanitized_input_proof.json',safe)
    r=dict(schema='world_reward.hocap_amg_bank.v1',stage='hocap_native_all_mask_bank',status='pass',phase='complete',
        producer_revision=rev,image_id='sha256:'+'1'*64,private_labels_read=False,calibration_read=False,ground_truth_used=False,
        opaque_metadata_read=False,challenge_inputs_used=False,oracle_modes=[],network='none',quality_verified=False,
        adoption=False,boots_executed=False,association_executed=False,all_native_returned_masks=True,model_loads=1,
        rgb_decodes=1493,amg_calls=10,amg_attempts=10,anchors_retained=10,all_original_jpegs_decoded_before_model=True,
        original_source_public_models_rehashed_after=True,public_manifest_identity=p['public_manifest'],
        sanitized_input_proof_identity=safe_pin,source_binding_sha256=m.bank.json_digest(safe['source_binding']),
        frontend_proof_sha256=safe['frontend_proof_sha256'],image_size=[4,4],banks=rows,bank_bytes=sum(r['bytes'] for r in rows))
    pin=seal_json(Path(p['bank'])/'report.json',r)
    host=dict(stage='hocap_amg_host',status='pass',producer_revision=rev,image_id=r['image_id'],native_exit_status=0,
        owned_cleanup_verified=True,source_public_rehashed_after=True,native_report_identity=pin,opaque_input_receipt_mounted=False,
        private_labels_read=False,quality_verified=False,sanitized_input_proof_identity=safe_pin)
    p['bank_host_report']=seal_json(Path(p['bank'])/'host.json',host)
    return p,pin,r


def test_complete_public_bank_and_unknown_inventory(tmp_path):
    p,pin,r=public_bank_receipt(tmp_path)
    proof=m.authenticate_bank(p,pin,historical=False)
    assert proof['report']==pin and len(proof['banks'])==10
    seal_json(Path(p['bank'])/'unknown.json',{'not_allowed':True})
    with pytest.raises(ValueError,match='unknown'): m.authenticate_bank(p,pin,historical=False)


def test_public_bank_npz_mutation_rejected_before_load(tmp_path):
    p,pin,r=public_bank_receipt(tmp_path); path=Path(p['bank'])/r['banks'][0]['file']
    path.chmod(0o644); path.write_bytes(b'changed'); path.chmod(0o444)
    with pytest.raises(ValueError,match='bytes differ'): m.authenticate_bank(p,pin,historical=False)


@pytest.mark.parametrize('field',['elapsed_seconds','owned_cleanup_verified','image_id'])
def test_independent_host_bytes_reject_otherwise_valid_mutation(tmp_path,field):
    p,pin,r=public_bank_receipt(tmp_path); path=Path(p['bank'])/'host.json'
    value=json.loads(path.read_text())
    value[field]=.01 if field=='elapsed_seconds' else False if field=='owned_cleanup_verified' else 'sha256:'+'9'*64
    path.chmod(0o644); path.write_text(json.dumps(value)); path.chmod(0o444)
    with pytest.raises(ValueError,match='identity'): m.authenticate_bank(p,pin,historical=False)


def test_decode_every_original_frame_no_color_or_size_conversion(tmp_path,monkeypatch):
    from PIL import Image
    p,rows=manufacture_banks(tmp_path); prepared=m.prepare_queries(rows,p,time.monotonic()+20); calls=[]
    class ManufacturedImage:
        format='JPEG'; mode='RGB'; size=(4,4)
        def __enter__(self): return self
        def __exit__(self,*_): pass
        def load(self): pass
        def __array__(self,dtype=None,copy=None): return np.zeros((4,4,3),np.uint8)
    def image(path): calls.append(path); return ManufacturedImage()
    monkeypatch.setattr(Image,'open',image)
    clips=[dict(clip=c['clip'],num_frames=c['frames'],records=[{'path':str(t)} for t in range(c['frames'])]) for c in prepared]
    report={'rgb_decodes':0}; m.decode_all(clips,prepared,time.monotonic()+20,report)
    assert report['rgb_decodes']==len(calls)==11
    assert all(c['video'].dtype==np.uint8 for c in prepared)
    ManufacturedImage.mode='L'
    with pytest.raises(ValueError,match='no conversions'): m.decode_all(clips,prepared,time.monotonic()+20,{'rgb_decodes':0})


def test_public_receipt_rejects_tamper_before_arrays(tmp_path):
    p,pin,r=public_bank_receipt(tmp_path)
    path=Path(p['bank'])/'report.json'; path.chmod(0o644); path.write_text('{}'); path.chmod(0o444)
    with pytest.raises(ValueError,match='identity'): m.authenticate_bank(p,pin,historical=False)


@pytest.mark.parametrize('flag',['private_labels_read','adoption','boots_executed','quality_verified'])
def test_forged_unsafe_bank_facts_rejected(tmp_path,flag):
    p,pin,r=public_bank_receipt(tmp_path); r[flag]=True
    path=Path(p['bank'])/'report.json'; path.chmod(0o644); path.unlink(); pin=seal_json(path,r)
    with pytest.raises(ValueError,match='facts'): m.authenticate_bank(p,pin,historical=False)


def test_receipt_write_exclusive_and_late_pass_demoted(tmp_path):
    path=tmp_path/'report.json'
    with pytest.raises(TimeoutError): m.write_json(path,dict(status='pass',phase='complete'),deadline=time.monotonic()-1)
    assert json.loads(path.read_text())['status']=='fail' and path.stat().st_mode&0o777==0o444
    with pytest.raises(FileExistsError): m.write_json(path,dict(status='pass'))


@pytest.mark.parametrize('case',['absent','live','foreign','daemon','rmfail','stillpresent','timeout'])
def test_cleanup_exact_cid_and_independent_absence(tmp_path,monkeypatch,case):
    cid='a'*64; revision='b'*40; name='owned'; image='sha256:'+'c'*64
    path=tmp_path/'.container.cid'; path.write_text(cid+'\n'); calls=[]
    original=m.rt.canonical
    class StatProxy:
        def __init__(self,path): self.path=path
        def lstat(self):
            import types
            s=self.path.lstat()
            return types.SimpleNamespace(st_mode=s.st_mode,st_nlink=s.st_nlink,st_uid=0,st_size=s.st_size)
    monkeypatch.setattr(m.rt,'canonical',lambda p:StatProxy(path) if Path(p)==path else original(p))
    def run(argv,**kwargs):
        calls.append(argv)
        if case=='timeout': raise subprocess.TimeoutExpired(argv,5)
        if argv[1]=='inspect':
            if case=='absent': return subprocess.CompletedProcess(argv,1,b'',f'\nError: No such object: {cid}\n'.encode())
            if case=='daemon': return subprocess.CompletedProcess(argv,1,b'',b'Cannot connect to daemon\n')
            actual=image+'|/'+name+'|'+m.ENTRY+'|'+revision
            if case=='foreign': actual+='bad'
            return subprocess.CompletedProcess(argv,0,(actual+'\n').encode(),b'')
        if argv[1]=='rm': return subprocess.CompletedProcess(argv,1 if case=='rmfail' else 0,(cid+'\n').encode(),b'')
        return subprocess.CompletedProcess(argv,0,(cid+'\n').encode() if case=='stillpresent' else b'',b'')
    monkeypatch.setattr(m.subprocess,'run',run)
    if case in ('live','absent'):
        m.cleanup(path,name,revision,image); assert path.stat().st_mode&0o777==0o444
    else:
        with pytest.raises((ValueError,subprocess.TimeoutExpired)): m.cleanup(path,name,revision,image)
    if case in ('foreign','daemon','absent','timeout'): assert not any(a[1]=='rm' for a in calls)


def test_source_and_shell_scopes_no_legacy_query_cap_or_private_mount():
    text=(ROOT/'infra/hocap_boots_track.py').read_text(); shell=(ROOT/'infra/run_hocap_boots_track.sh').read_text()
    assert 'boots.validate_video(' not in text and 'boots.native_prediction(' in text and 'boots.validate_prediction(' in text
    assert 'noise.runtime_evidence(code)' in text and 'bank.public_inputs(' in text
    assert "'--memory','64g','--cpus','4'" in text and "'--network','none'" in text
    assert '1860s' in shell and "--bank-report" not in shell  # args forwarded, no invented actual pin
    assert 'run_hocap_boots_track/code' in shell and 'world-reward-ncc-h100-02' in shell
    assert "'--mount',f'type=bind,src={out},dst={out}'" in text


def test_runtime_import_is_model_free():
    assert 'torch' not in m.__dict__ and 'numpy' not in m.__dict__
    assert m.ENTRY=='run_hocap_boots_track' and set(m.HELPERS)>={m.PROTOCOL,m.SCIENCE}
