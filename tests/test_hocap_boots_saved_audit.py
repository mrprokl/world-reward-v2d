"""Synthetic arrays/host-control spies, never actual inference or HO-Cap labels."""
import ast
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

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'infra'))
spec=importlib.util.spec_from_file_location('hocap_saved_under_test',REPO/'infra/hocap_boots_saved_audit.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def artifact(path,raw,mode=0o444):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(mode)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def manufactured_predictions(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(m,'ROOT',root)
    p=json.loads((REPO/m.track.PROTOCOL).read_text());p.update(bank=str(root/'bank'),frames=[5,6])
    rows=[];pins={};old={'predictions':[]}
    for clip,t in zip(p['clips'],p['frames']):
        for k in range(5):
            frame=k*(t-1)//4;mask=np.ones((4,4),bool)
            records=[dict(segmentation=mask,bbox=[0,0,4,4],crop_box=[0,0,4,4],point_coords=[[.5,.5]],
                area=16,predicted_iou=.9,stability_score=.97)]*3+[dict(segmentation=np.zeros((4,4),bool),bbox=[0,0,4,4],
                crop_box=[0,0,4,4],point_coords=[[.5,.5]],area=0,predicted_iou=.9,stability_score=.97)]
            a,audit=m.track.bank.bank_arrays(records,dict(clip=clip,frames=t,frame_index=frame,width=4,height=4),
                {**p,'packed_mask_bitorder':'little'})
            name=f'bank/{clip}_anchor_{frame:06d}.npz';path=Path(p['bank'])/name;path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,**a);path.chmod(0o444)
            rows.append(dict(clip=clip,frames=t,anchor_frame_index=frame,width=4,height=4,file=name,**audit))
    evidence={'bank':{'banks':rows}}
    prepared=m.track.prepare_queries(rows,p,time.monotonic()+10)
    for c in prepared:
        q,t=c['query_count'],c['frames'];tracks256=np.full((q,t,2),128,np.float32)
        a=dict(tracks_256=tracks256,tracks=tracks256*np.array([4,4],np.float32)/np.array([256,256],np.float32),
            occlusion=np.zeros((q,t),np.float32),expected_dist=np.zeros((q,t),np.float32),visible=np.ones((q,t),bool),
            query_points=c['query_points'],point_indices=c['point_indices'],frame_index=np.arange(t,dtype=np.int64),**c['metadata'])
        name=c['clip']+'_tracks.npz';path=root/m.ORIGINAL/name;path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,**a);path.chmod(0o444);pin=m.rt.identity(path);pins[name]=pin
        old['predictions'].append(dict(file=name))
    monkeypatch.setattr(m,'PINS',pins)
    return root,p,evidence,old


def test_complete_mixed_birth_above32_all19keys_no_static_published(tmp_path,monkeypatch):
    root,p,evidence,old=manufactured_predictions(tmp_path,monkeypatch)
    before={n:(root/m.ORIGINAL/n).read_bytes()for n in m.PINS}
    rows=m.validate_saved_predictions(p,evidence,old,time.monotonic()+10)
    assert len(rows)==2 and all(r['queries']==240>32 and r['seeds']==20 for r in rows)
    for name in m.PINS:
        path=root/m.ORIGINAL/name
        assert path.read_bytes()==before[name]
        with np.load(path,allow_pickle=False)as z:assert len(z.files)==19 and 'static_tracks'not in z.files


@pytest.mark.parametrize('field',['tracks','tracks_256','query_points','visible','seed_ids','seed_birth_frame_indices','seed_query_offsets','extra'])
def test_saved_prediction_mutation_rejected_without_refit(tmp_path,monkeypatch,field):
    root,p,evidence,old=manufactured_predictions(tmp_path,monkeypatch);path=root/m.ORIGINAL/old['predictions'][0]['file']
    with np.load(path,allow_pickle=False)as z:a={n:z[n]for n in z.files}
    if field in ('tracks','tracks_256','query_points'):a[field].flat[0]+=.25
    elif field=='visible':a[field]=a[field].astype(np.float32)
    elif field=='seed_ids':a[field][0]='forged'
    elif field=='seed_birth_frame_indices':a[field][0]=1
    elif field=='seed_query_offsets':a[field][1]+=1
    else:a['private_labels']=np.zeros(1)
    path.chmod(0o644);np.savez(path,**a);path.chmod(0o444)
    with pytest.raises(ValueError):m.validate_saved_predictions(p,evidence,old,time.monotonic()+10)


def test_byte_pins_original_cid_and_protocol_are_exact():
    assert m.REV=='9ab14d6e5dd662d8b50b7fee9b0656bd14afea59' and len(m.PINS)==7
    assert m.PINS['.container.cid']==dict(bytes=64,sha256=hashlib.sha256(m.CID.encode()).hexdigest())
    assert m.PINS['report.json']['bytes']==2006 and m.PINS['host.json']['bytes']==547
    assert m.PINS['proof.json']['bytes']==11004 and m.BANK_PIN['bytes']==7495
    assert m.BUDGET==120 and m.GRACE==60


@pytest.mark.parametrize('stdout,stderr,rc,accepted',[
    (b'\n',b'error: no such object: CID\n',1,True),
    (b'',b'Error: No such container: CID\n',1,True),
    (b'',b'\nError: No such object: CID\n',1,True),
    (b'[]\n',b'error: no such object: CID\n',1,False),
    (b'',b'Cannot connect to Docker daemon\n',1,False),
    (b'',b'error: no such object: OTHER\n',1,False),
    (b'',b'error: no such object: CID\n',124,False),
    (b'',b'error: no such object: CID\nextra',1,False)])
def test_exact_actual_lowercase_absence_not_arbitrary_strip(stdout,stderr,rc,accepted):
    r=subprocess.CompletedProcess([],rc,stdout,stderr);assert m.absent_result(r,'CID')is accepted


def test_independent_cid_and_name_absence_no_removal(monkeypatch):
    calls=[]
    def command(argv,seconds=5):
        calls.append(argv)
        return subprocess.CompletedProcess(argv,1,b'\n',b'error: no such object: CID\n')if argv[1]=='inspect'else subprocess.CompletedProcess(argv,0,b'',b'')
    monkeypatch.setattr(m,'command',command)
    assert m.absence('CID','name')
    assert len(calls)==3 and not any(a[1]=='rm'for a in calls)
    def present(argv,seconds=5):
        return command(argv,seconds)if argv[1]=='inspect'else subprocess.CompletedProcess(argv,0,b'CID\n',b'')
    monkeypatch.setattr(m,'command',present)
    with pytest.raises(ValueError,match='Independent'):m.absence('CID','name')


def test_original_failed_unit_never_reset_and_gpu_idle(monkeypatch):
    calls=[]
    def command(argv,seconds=5):
        calls.append(argv)
        if argv[0]=='systemctl':raw=b'LoadState=loaded\nActiveState=failed\nSubState=failed\nResult=exit-code\nExecMainStatus=1\n'
        else:raw=b''
        return subprocess.CompletedProcess(argv,0,raw,b'')
    monkeypatch.setattr(m,'command',command);monkeypatch.setattr(m,'absence',lambda cid,name:True)
    assert m.host_environment()['original_unit_state']['ActiveState']=='failed'
    assert not any('reset-failed'in a or 'stop'in a for a in calls)
    def active(argv,seconds=5):
        r=command(argv,seconds)
        return subprocess.CompletedProcess(argv,0,r.stdout.replace(b'ActiveState=failed',b'ActiveState=active'),b'')
    monkeypatch.setattr(m,'command',active)
    with pytest.raises(ValueError,match='FAILED'):m.host_environment()


def test_historical_byte_mutation_rejected_before_source_or_arrays(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();folder=root/m.ORIGINAL;folder.mkdir(parents=True)
    pins={n:artifact(folder/n,m.CID.encode()if n=='.container.cid'else b'{}',
        0o644 if n=='.container.cid'else 0o400 if n=='native.log'else 0o444)for n in m.PINS}
    folder.chmod(0o555);monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'PINS',pins)
    monkeypatch.setattr(m.track,'protocol',lambda _: {})
    (folder/'report.json').chmod(0o644);(folder/'report.json').write_bytes(b'changed');(folder/'report.json').chmod(0o444)
    with pytest.raises(ValueError,match='bytes differ'):m.saved_inputs(REPO,time.monotonic()+10)
    assert (folder/'.container.cid').stat().st_mode&0o777==0o644


def test_budget_expiration_and_no_late_pass(tmp_path):
    with pytest.raises(TimeoutError):m.check(time.monotonic()-1)
    path=tmp_path/'native.json'
    with pytest.raises(TimeoutError):m.track.write_json(path,dict(status='pass',phase='complete'),deadline=time.monotonic()-1)
    assert json.loads(path.read_text())['status']=='fail'


def test_hash_only_snapshot_parent_no_unverified_siblings(tmp_path):
    parent=tmp_path/'snapshot';parent.mkdir()
    (parent/'code').mkdir();artifact(parent/'revision',b'a'*40+b'\n');artifact(parent/'source-sha256',b'b'*64+b'\n')
    parent.chmod(0o555);m.snapshot_parent(parent)
    parent.chmod(0o755);artifact(parent/'unverified-asset',b'foreign');parent.chmod(0o555)
    with pytest.raises(ValueError,match='source-only'):m.snapshot_parent(parent)


def test_source_static_no_old_execution_model_private_decoder_or_gpu():
    raw=(REPO/'infra/hocap_boots_saved_audit.py').read_text();tree=ast.parse(raw)
    assert 'track.preflight('not in raw and 'track.source('not in raw
    assert 'track.boots.validate_prediction('in raw and 'track.prepare_queries('in raw
    assert 'rt.source(ROOT,old,REV,track.ENTRY,track.HELPERS)'in raw
    assert 'torch.load'not in raw and 'Image.open'not in raw and 'native_prediction('not in raw
    assert "'--gpus'"not in raw and "'CUDA_VISIBLE_DEVICES=-1'"in raw
    assert "readonly=name!='.container.cid'"in raw and "folder/'.container.cid').chmod"not in raw
    assert not any(isinstance(n,ast.Import)and any(a.name=='torch'for a in n.names)for n in ast.walk(tree))
    subprocess.run(['bash','-n',str(REPO/'infra/run_hocap_boots_saved_audit.sh')],check=True)
    assert '180s'in (REPO/'infra/run_hocap_boots_saved_audit.sh').read_text()
