"""Tiny lifecycle seams for CPU private evaluator; no Docker/data/network."""
import ast
import importlib.util
import hashlib
import json
from pathlib import Path
import stat
import time
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    s = importlib.util.spec_from_file_location('wr_oi_eval_tests', REPO/'infra/openimages_holds_evaluate.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def test_frozen_config_binds_completed_full_cohort_and_both_metadata_manifests(gate):
    raw=(REPO/gate.CONFIG).read_bytes();c=json.loads(raw)
    assert gate.CONFIG_PIN==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert c['predictions']['folder']=='results/hoi-detr-external-cohort128-v3'
    assert all(c['predictions'][k]['bytes']>0 for k in ('host','native','observations_manifest'))
    assert c['acquisition_manifest']['identity']['bytes']==135808
    assert c['original_acquisition_manifest']['identity']['bytes']==127742
    assert all('storage.googleapis.com/openimages/' in r['url'] for r in c['references'])


def value(gate, image='sha256:'+'e'*64, revision='a'*40, name='name', mounts=()):
    return dict(Image=image, Name='/'+name, Config=dict(User='1000:1000', Labels={'world-reward.job': gate.ENTRY, 'world-reward.revision': revision}),
                HostConfig=dict(NetworkMode='none', ReadonlyRootfs=True, Privileged=False, CapDrop=['ALL'], SecurityOpt=['no-new-privileges'],
                                NanoCpus=4_000_000_000, Memory=4*(1<<30), DeviceRequests=[], Devices=[], Binds=[], VolumesFrom=[]),
                Mounts=[dict(Source=a, Destination=b, RW=not ro, Type='bind') for a,b,ro in mounts])


@pytest.mark.parametrize('fault', [None,'image','owner','revision','network','gpu','memory','mount','caps'])
def test_exact_owned_cpu_container_no_gpu_extra_mount(gate, fault):
    mounts = {('/source','/source',True)}; v=value(gate,mounts=mounts)
    if fault=='image': v['Image']='other'
    if fault=='owner': v['Config']['User']='0'
    if fault=='revision': v['Config']['Labels']['world-reward.revision']='b'*40
    if fault=='network': v['HostConfig']['NetworkMode']='host'
    if fault=='gpu': v['HostConfig']['DeviceRequests']=[{'Count':-1}]
    if fault=='memory': v['HostConfig']['Memory']=0
    if fault=='mount': v['Mounts'].append(dict(Source='/GT',Destination='/GT',RW=False,Type='bind'))
    if fault=='caps': v['HostConfig']['CapDrop']=[]
    if fault:
        with pytest.raises(ValueError):gate.validate_container(v,'name','a'*40,'sha256:'+'e'*64,mounts)
    else:gate.validate_container(v,'name','a'*40,'sha256:'+'e'*64,mounts)


def test_late_pass_demoted_on_owned_descriptor(gate,tmp_path):
    p=tmp_path/'report.json';v=dict(status='pass')
    gate.write_receipt(p,v,time.monotonic()-1,time.monotonic()-1)
    assert json.loads(p.read_bytes())['status']=='fail' and v['status']=='fail'
    assert not p.stat().st_mode&0o222


@pytest.mark.parametrize('fault',[None,'foreign','survives','badcid'])
def test_cleanup_exactcid_never_name_fallback(gate,tmp_path,monkeypatch,fault):
    cid='c'*64;p=tmp_path/'container.cid';p.write_text(cid if fault!='badcid' else 'invalid')
    real=Path.lstat
    monkeypatch.setattr(Path,'lstat',lambda self: SimpleNamespace(st_mode=stat.S_IFREG|0o644,st_nlink=1,st_uid=0,st_size=len(p.read_bytes())) if self==p else real(self))
    calls=[];seen=0
    def cmd(args,deadline):
        nonlocal seen
        calls.append(args)
        if args[1]=='ps':
            seen+=1;return cid if seen==1 or fault=='survives' else ''
        if args[1]=='inspect':return json.dumps(value(gate,revision='b'*40 if fault=='foreign' else 'a'*40))
        if args[1]=='rm':return cid
        raise AssertionError(args)
    monkeypatch.setattr(gate,'command',cmd)
    if fault:
        with pytest.raises(ValueError):gate.cleanup_container(p,'name','a'*40,'sha256:'+'e'*64,set(),time.monotonic()+10)
        if fault in ('foreign','badcid'):assert not any(c[1]=='rm' for c in calls)
    else:
        gate.cleanup_container(p,'name','a'*40,'sha256:'+'e'*64,set(),time.monotonic()+10)
        assert [c for c in calls if c[1]=='rm']==[['docker','rm','-f',cid]]


def test_shared_deadline_failure_receipt_before_reference_io(gate):
    tree=ast.parse((REPO/'infra/openimages_holds_evaluate.py').read_bytes())
    funcs={n.name:ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert funcs['run'].index('started = time.monotonic()')<funcs['run'].index('helper(code)')
    assert funcs['host_run'].index('authenticate_predictions(')<funcs['host_run'].index('acquire_references(')
    assert 'finally:' in funcs['host_run'] and "out / 'report.json'" in funcs['host_run']
    assert 'time.monotonic() >= deadline' in funcs['host_run']
    assert 'original.parent' in funcs['host_run'] and 'DeviceRequests' in funcs['validate_container']
    assert (gate.BUDGET_SECONDS,gate.CHILD_SECONDS,gate.CLEANUP_SECONDS)==(600,300,60)


@pytest.mark.parametrize('fault',[None,'omitted','replaced','duplicate','hostfail','nativefail','strict','cleanup','frame','reference','counts'])
def test_all61_qualified_predictions_required_before_references(gate,monkeypatch,fault):
    ids=[format(i,'016x') for i in range(128)]
    pin={'bytes':1,'sha256':'d'*64};source={'helpers':('only-source',)}
    c=dict(predictions=dict(folder='results/fixture',producer_revision='a'*40,host=pin,native=pin,observations_manifest=pin),
           acquisition_manifest=dict(path='/qualified/manifest.json',identity=pin))
    h=dict(status='pass',producer_revision='a'*40,phase='complete',native_report_identity=pin,actual_model_qualified=True,
           source_binding=source,observations_identity=pin)
    h.update({k:True for k in ('source_rehashed_after','inputs_rehashed_after','image_unchanged','owned_containers_removed','owned_overlay_removed')})
    n=dict(status='pass',native_model_loads=1,fixed_cohort_slots=128,reference_geometry_read=False,source_binding=source,
           strict_checkpoint=dict(keys=1796,weights_only=True,strict=True),observations=pin,native_forward_calls=61,acquired_slots=61,missing_slots=67)
    for k in ('ground_truth_used','challenge_inputs_used','adoption','quality_verified'):h[k]=n[k]=False
    rows=[dict(image_id=i,file=i+'.npz',original_frame_index=0,observations=pin) for i in ids[:61]]
    if fault=='omitted':rows.pop();n['native_forward_calls']=n['acquired_slots']=60
    if fault=='replaced':rows[-1].update(image_id=ids[-1],file=ids[-1]+'.npz')
    if fault=='duplicate':rows[-1]=rows[0]
    if fault=='hostfail':h['status']='fail'
    if fault=='nativefail':n['status']='fail'
    if fault=='strict':n['strict_checkpoint']['strict']=False
    if fault=='cleanup':h['owned_containers_removed']=False
    if fault=='frame':rows[0]['original_frame_index']=1
    if fault=='reference':n['reference_geometry_read']=True
    if fault=='counts':n['missing_slots']=55
    values={'report.json':h,'native.json':n,'observations_manifest.json':dict(observations=rows),
            'manifest.json':dict(records=[dict(image_id=i,status='acquired' if k<61 else 'unscorable_header') for k,i in enumerate(ids)])}
    rt=SimpleNamespace(canonical=lambda p:p,pinned=lambda p,*_:values[p.name],identity=lambda *_:pin,source=lambda *_:source)
    monkeypatch.setattr(gate,'source_parent',lambda *_:None)
    monkeypatch.setattr(gate,'authenticate_input_qualification',lambda *_:None)
    if fault:
        with pytest.raises(ValueError):gate.authenticate_predictions(rt,c,time.monotonic()+30)
    else:
        prior,_,_=gate.authenticate_predictions(rt,c,time.monotonic()+30)
        assert len(prior['manifest']['observations'])==61 and len(prior['acquisition']['records'])==128
