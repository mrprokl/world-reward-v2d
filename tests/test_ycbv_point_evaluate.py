"""Manufactured public receipts/arrays and private adapter gates, no real data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'infra'))
spec=importlib.util.spec_from_file_location('wr_ycbv_evaluate_test',REPO/'infra/ycbv_point_evaluate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def write(p,raw):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o400);return gate.files.identity(p)


def record(p,data,revision='b'*40,script='c'*64):
    data=dict(data,producer_revision=revision,script_sha256=script)
    return dict(**write(p,(json.dumps(data)+'\n').encode()),producer_revision=revision,script_sha256=script)


def fixture(tmp_path):
    base=tmp_path/gate.BASE;base.mkdir(parents=True)
    retention=dict(bytes=12,sha256='a'*64)
    acq=record(base/'report.json',dict(stage='external_ycbv_contiguous_rgb_only_acquisition',status='pass',phase='complete',
      dataset_revision='5c2c4aa229800355648cd268040aa814f8dc94f0',license='MIT',device='cpu',gpu_used=False,
      challenge_inputs_used=False,inference_performed=False,source_rehashed_after=True,disposable_archives_removed=True,
      selection_before_private_annotation_values=True,selected_frames=288,all_instances_retained=True,
      private_annotations_exported_as_inference_inputs=False,retention_receipt=retention))
    rows=[];predictions={};initial_masks={};maskrows=[]
    comp=base/'comparison_v1';comp.mkdir();write(comp/'.container.cid',b'e'*64)
    for scene in gate.SCENES:
        arrays=dict(frame_index=np.arange(96,dtype=np.int64),baseline_poses=np.tile(np.eye(4),(96,1,1)),candidate_poses=np.tile(np.eye(4),(96,1,1)))
        p=comp/f'scene_{scene:06d}.npz';np.savez_compressed(p,**arrays);p.chmod(0o400);identity=gate.files.identity(p);predictions[p.name]=identity
        ids={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())for k,v in arrays.items()}
        rows.append(dict(file=p.name,scene_id=scene,frames=96,arrays=ids,**identity))
        m=np.zeros((480,640),np.uint8);m[:3,:4]=255
        p=base/f'automatic_masks_v1/scene_{scene:06d}/masks/1/000000.png';p.parent.mkdir(parents=True);Image.fromarray(m).save(p);p.chmod(0o400)
        mp=gate.files.identity(p);initial_masks[str(scene)]=mp
        for frame in range(96):maskrows.append(dict(scene_id=scene,frame_id=frame,file=f'scene_{scene:06d}/masks/1/{frame:06d}.png',**mp))
    track=record(comp/'report.json',dict(stage='public_ycbv_same_native25_pool_boots_point_comparison',status='pass',phase='complete',
      full_original_frame_coverage=True,same_native_valid_pool=True,native_pool_frozen_before_tracking_and_rankings=True,
      all_inputs_models_sources_after_reverified=True,ground_truth_used=False,private_annotations_read=False,sensor_depth_used=False,
      source_camera_calibration_used=False,human_scale_used=False,hand_labeled_test=False,oracle_initial_queries=False,oracle_modes=[],challenge_inputs_used=False,
      whole_pilot_GPU_elapsed_seconds=800.,outputs=rows))
    masks=record(base/'automatic_masks_v1/report.json',dict(stage='public_ycbv_point_native_object_masks',status='pass',phase='complete',
      private_annotations_read=False,query='object.',frames_completed=288,all_inputs_sources_assets_outputs_rehashed=True,masks=maskrows))
    pins=dict(schema='world-reward-ycbv-point-evaluation-pins-v1',acquisition_report=acq,retention=retention,track_report=track,
      masks_report=masks,predictions=predictions,initial_masks=initial_masks)
    return tmp_path,pins


def test_all_public_predictions_before_any_private_read(tmp_path,monkeypatch):
    root,pins=fixture(tmp_path);original=Path.open
    def safe(path,*args,**kwargs):
        assert 'eval_private'not in path.parts;return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',safe)
    rows,_=gate.public_predictions(root,pins);assert len(rows)==3 and all(x['baseline'].shape==(96,4,4)for x in rows)
    assert all(x['automatic_mask'].sum()==12 for x in rows)


def test_prediction_receipt_tamper_before_private(tmp_path):
    root,pins=fixture(tmp_path);p=root/gate.BASE/'comparison_v1/scene_000050.npz';p.chmod(0o600);p.write_bytes(b'bad');p.chmod(0o400)
    with pytest.raises(ValueError):gate.public_predictions(root,pins)


@pytest.mark.parametrize('fault',['missing','extra','producer','schema','floatbytes','floatframe','overbudget','GT'])
def test_public_or_pin_contract_stops(tmp_path,fault):
    root,pins=fixture(tmp_path)
    if fault=='missing':pins['predictions'].pop('scene_000050.npz')
    elif fault=='extra':write(root/gate.BASE/'comparison_v1/extra.json',b'{}')
    elif fault=='producer':pins['track_report']['producer_revision']='d'*40
    elif fault=='schema':pins['schema']='wrong'
    elif fault=='floatbytes':pins['retention']['bytes']=12.
    else:
        p=root/gate.BASE/'comparison_v1/report.json';data=json.loads(p.read_bytes())
        if fault=='floatframe':data['outputs'][2]['frames']=96.
        elif fault=='overbudget':data['whole_pilot_GPU_elapsed_seconds']=3601.
        else:data['ground_truth_used']=True
        p.chmod(0o600);pins['track_report']=record(p,data)
    with pytest.raises(ValueError):gate.public_predictions(root,pins)


def test_private_retention_exact_all_files_and_no_label_interpretation(tmp_path):
    base=tmp_path/gate.BASE/'eval_private';base.mkdir(parents=True,mode=0o700)
    files=[]
    for name in('hf_readme','publisher_readme','publisher_license','bop_format','bop_params'):
        path=f'source/licenses/{name}.txt';files.append(dict(file=path,**write(base/path,b'license fixture')))
    path='source/licenses/dataset_info.md';files.append(dict(file=path,**write(base/path,b'license fixture')))
    for scene in gate.SCENES:
        prefix=f'source/test/{scene:06d}/'
        for kind in('camera','gt','gt_info'):
            path=prefix+f'scene_{kind}.json';files.append(dict(file=path,**write(base/path,b'not JSON but hash only')))
        for frame in range(96):
            for folder,suffix in(('depth',''),('mask','_000000'),('mask_visib','_000000')):
                path=prefix+f'{folder}/{frame:06d}{suffix}.png';files.append(dict(file=path,**write(base/path,b'not PNG but hash only')))
    data=dict(files=files,all_instances_retained=True)
    rp=write(base/'retention-receipt.json',json.dumps(data).encode())
    result=gate.private_identities(tmp_path,dict(retention=rp));assert len(result)==879
    write(base/'source/unpinned.json',b'{}')
    with pytest.raises(ValueError):gate.private_identities(tmp_path,dict(retention=rp))


def test_wrapper_cpu_narrow_and_shell_syntax():
    shell=REPO/'infra/run_ycbv_point_evaluate.sh';subprocess.run(['bash','-n',str(shell)],check=True)
    text=shell.read_text();assert '--gpus'not in text and '--network none'in text and 'CUDA_VISIBLE_DEVICES='in text
    assert 'dst=$BASE/eval_private'in text or '"$BASE/eval_private"'in text
    assert '--cap-drop ALL'in text and '--read-only'in text and '--user 0:0'in text


@pytest.mark.parametrize('fault',[None,'marker','helper','revision'])
def test_host_producer_full_marker_linkage(tmp_path,fault):
    reports={};pins={}
    for name,key,job,entry in(('acquisition','acquisition_report','run_ycbv_point_acquire','infra/ycbv_point_acquire.py'),
      ('track','track_report','run_ycbv_point_track','infra/ycbv_point_track.py'),
      ('masks','masks_report','run_ycbv_point_masks','infra/ycbv_point_masks.py')):
        revision='b'*40;code=tmp_path/'jobs'/revision/job/'code';hp=write(code/entry,b'# fixed original helper\n')
        write(code.parent/'revision',(revision+'\n').encode());write(code.parent/'source-sha256',('d'*64+'\n').encode())
        markers={n:gate.files.identity(code.parent/n)for n in('revision','source-sha256')}
        pins[key]=dict(bytes=12,sha256='a'*64,producer_revision=revision,script_sha256=hp['sha256'])
        binding=dict(helpers={entry:hp},markers=markers)if name=='masks'else dict(files={entry:hp},markers=markers)
        reports[name]=dict(source_binding=binding)if name=='masks'else dict(source_helpers=binding)
    if fault=='marker':
        p=tmp_path/'jobs'/('b'*40)/'run_ycbv_point_masks/source-sha256';p.chmod(0o600);write(p,('e'*64+'\n').encode())
    elif fault=='helper':
        p=tmp_path/'jobs'/('b'*40)/'run_ycbv_point_track/code/infra/ycbv_point_track.py';p.chmod(0o600);write(p,b'# changed helper\n')
    elif fault=='revision':
        p=tmp_path/'jobs'/('b'*40)/'run_ycbv_point_acquire/revision';p.chmod(0o600);write(p,('e'*40+'\n').encode())
    if fault is None:assert set(gate.producer_sources(tmp_path,pins,reports))=={'acquisition','track','masks'}
    else:
        with pytest.raises(ValueError):gate.producer_sources(tmp_path,pins,reports)


def snapshot_fixture():
    identity=dict(bytes=1,sha256='a'*64)
    return dict(schema='world_reward.ycbv_evaluator_host_snapshot.v1',pins_sha256='b'*64,
        public_sha256='c'*64,producer_sha256='d'*64,
        source=dict(files={name:dict(identity)for name in gate.SOURCE},
            markers={name:dict(identity)for name in('revision','source-sha256')},complete_closure_sha256='e'*64),
        image=dict(Id=gate.IMAGE,Architecture='amd64',Os='linux',RootFS=dict(Type='layers',Layers=['sha256:'+'f'*64])))


def host_seal_fixture(tmp_path,monkeypatch):
    out=tmp_path/gate.BASE/gate.OUTPUT;out.mkdir(parents=True,mode=0o700)
    before=snapshot_fixture();revision='b'*40
    report=dict(stage=gate.STAGE,status='pass',phase='complete',producer_revision=revision,
        script_sha256=before['source']['files'][gate.SOURCE[0]]['sha256'],
        source_helpers={k:before['source'][k]for k in('files','markers')},
        host_wrapper_post_verified=False,requires_independent_host_post_receipt=True,
        public_after_reverified=True,source_after_reverified=True,predictions_frozen_before_private_values=True,
        gpu_used=False,inference_performed=False,private_arrays_exported=False,alignment_performed=False,scale_fitted=False,
        budget_seconds=gate.BUDGET,elapsed_seconds=3.25,scientific_decision='REJECT')
    original=write(out/'report.json',json.dumps(report).encode())
    monkeypatch.setattr(gate,'host_snapshot',lambda *args:copy.deepcopy(before))
    return out,before,revision,report,original


def test_host_seal_operational_not_scientific_decision_and_no_private_reads(tmp_path,monkeypatch):
    out,before,revision,_,original=host_seal_fixture(tmp_path,monkeypatch);old_open=Path.open
    def no_labels(path,*args,**kwargs):
        assert 'eval_private'not in path.parts;return old_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',no_labels)
    assert gate.write_host_seal(tmp_path,tmp_path/'code',revision,before,original,0,True)
    seal=json.loads((out/'host-post.json').read_bytes())
    assert seal['status']=='pass'and seal['container_report']==original
    assert seal['image']==before['image']and seal['original_source']==before['source']
    assert seal['owned_container_cleanup_verified']and seal['source_public_producer_image_post_verified']
    assert seal['private_labels_read']is False and seal['quality_decision_exported']is False
    assert 'scientific_decision'not in seal and 'decision'not in seal
    assert (out/'host-post.json').stat().st_mode&0o777==0o400
    assert gate.files.identity(out/'report.json')==original
    assert json.loads((out/'report.json').read_bytes())['host_wrapper_post_verified']is False


@pytest.mark.parametrize('role',['source','pins_sha256','public_sha256','producer_sha256','image'])
def test_host_seal_post_tamper_fails_closed_without_report_rewrite(tmp_path,monkeypatch,role):
    out,before,revision,_,original=host_seal_fixture(tmp_path,monkeypatch);after=copy.deepcopy(before)
    if role=='source':after[role]['complete_closure_sha256']='f'*64
    elif role=='image':after[role]['RootFS']['Layers'].reverse();after[role]['RootFS']['Layers'].append('sha256:'+'a'*64)
    else:after[role]='f'*64
    monkeypatch.setattr(gate,'host_snapshot',lambda *args:after)
    assert not gate.write_host_seal(tmp_path,tmp_path/'code',revision,before,original,0,True)
    seal=json.loads((out/'host-post.json').read_bytes())
    assert seal['status']=='fail'and 'container_report'not in seal and not seal['quality_decision_exported']
    assert gate.files.identity(out/'report.json')==original


@pytest.mark.parametrize('exit_status,cleanup',[(1,True),(124,True),(0,False),(130,False)])
def test_host_seal_requires_actual_exit_zero_and_owned_cleanup(tmp_path,monkeypatch,exit_status,cleanup):
    out,before,revision,_,original=host_seal_fixture(tmp_path,monkeypatch)
    assert not gate.write_host_seal(tmp_path,tmp_path/'code',revision,before,original,exit_status,cleanup)
    seal=json.loads((out/'host-post.json').read_bytes())
    assert seal['wrapper_exit_status']==exit_status and seal['owned_container_cleanup_verified']==cleanup
    assert seal['status']=='fail'and gate.files.identity(out/'report.json')==original


@pytest.mark.parametrize('field,value',[('status','fail'),('phase','incomplete'),('producer_revision','e'*40),
    ('source_helpers',{}),('host_wrapper_post_verified',True),('requires_independent_host_post_receipt',False),
    ('source_after_reverified',False),('public_after_reverified',False),('private_arrays_exported',True),
    ('budget_seconds',240.),('elapsed_seconds',241.),('elapsed_seconds',float('nan'))])
def test_host_seal_requires_original_complete_unmodified_contract(tmp_path,monkeypatch,field,value):
    out,before,revision,report,_=host_seal_fixture(tmp_path,monkeypatch);report[field]=value
    p=out/'report.json';p.chmod(0o600);original=write(p,json.dumps(report).encode())
    assert not gate.write_host_seal(tmp_path,tmp_path/'code',revision,before,original,0,True)
    assert json.loads((out/'host-post.json').read_bytes())['status']=='fail'
    assert gate.files.identity(p)==original


def test_host_seal_exclusive_no_overwrite(tmp_path,monkeypatch):
    out,before,revision,_,report_identity=host_seal_fixture(tmp_path,monkeypatch)
    assert gate.write_host_seal(tmp_path,tmp_path/'code',revision,before,report_identity,0,True)
    original=gate.files.identity(out/'host-post.json')
    with pytest.raises(FileExistsError):gate.write_host_seal(tmp_path,tmp_path/'code',revision,before,report_identity,0,True)
    assert gate.files.identity(out/'host-post.json')==original


def test_host_seal_report_is_pinned_before_cleanup_not_first_seen_after(tmp_path,monkeypatch):
    out,before,revision,report,original=host_seal_fixture(tmp_path,monkeypatch)
    report['scientific_decision']='a changed but otherwise valid aggregate'
    path=out/'report.json';path.chmod(0o600);current=write(path,json.dumps(report).encode())
    assert current!=original
    assert not gate.write_host_seal(tmp_path,tmp_path/'code',revision,before,original,0,True)
    assert gate.files.identity(path)==current
    assert json.loads((out/'host-post.json').read_bytes())['status']=='fail'


def test_host_seal_no_report_no_quality_and_metadata_failure_is_sanitized(tmp_path,monkeypatch):
    out,before,revision,_,original=host_seal_fixture(tmp_path,monkeypatch)
    (out/'report.json').unlink()
    def failed(*args):raise RuntimeError('SECRET_SHOULD_NOT_ESCAPE')
    monkeypatch.setattr(gate,'host_snapshot',failed)
    assert not gate.write_host_seal(tmp_path,tmp_path/'code',revision,before,original,1,False)
    raw=(out/'host-post.json').read_bytes()
    assert b'SECRET_SHOULD_NOT_ESCAPE'not in raw and json.loads(raw)['error_type']=='RuntimeError'


@pytest.mark.parametrize('fault',[None,'foreign_image','platform','layers','config_env','query_error'])
def test_host_snapshot_hash_only_safe_image_projection(tmp_path,monkeypatch,fault):
    root,pins=fixture(tmp_path);code=tmp_path/'code';write(code/gate.PINS,json.dumps(pins).encode())
    before=snapshot_fixture();full={k:before['source'][k]for k in('files','markers')}
    full['files']=dict(full['files'],**{'infra/extra_code.py':dict(bytes=2,sha256='f'*64)})
    calls=[];real_public=gate.public_predictions
    monkeypatch.setattr(gate,'ROOT',root)
    monkeypatch.setattr(gate,'source_binding',lambda c,r,*,host=False:(calls.append(('source',host))or full))
    def public(r,p,*,decode):
        assert decode is False;calls.append(('public',decode));return real_public(r,p,decode=decode)
    monkeypatch.setattr(gate,'public_predictions',public)
    monkeypatch.setattr(gate,'producer_sources',lambda *args:(calls.append(('producers',))or {'verified':True}))
    image=copy.deepcopy(before['image'])
    if fault=='foreign_image':image['Id']='sha256:'+'0'*64
    elif fault=='platform':image['Architecture']='arm64'
    elif fault=='layers':image['RootFS']['Layers']=['invalid']
    elif fault=='config_env':image['Config']={'Env':['SECRET=not-exportable']}
    def docker(command,**kwargs):
        assert command[:4]==['docker','image','inspect',gate.IMAGE]
        assert 'Config'not in command[-1]and kwargs['timeout']==10 and kwargs['capture_output']
        assert set(kwargs['env'])=={'PATH','HOME','DOCKER_HOST'}
        return subprocess.CompletedProcess(command,1 if fault=='query_error'else 0,json.dumps(image).encode(),b'')
    monkeypatch.setattr(gate.subprocess,'run',docker);old_open=Path.open
    def no_labels(path,*args,**kwargs):
        assert 'eval_private'not in path.parts;return old_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',no_labels)
    if fault is None:
        result=gate.host_snapshot(root,code,'b'*40)
        assert result['image']==image and result['source']['files']==before['source']['files']
        assert re_digest(result['source']['complete_closure_sha256'])
        assert calls==[('source',True),('public',False),('producers',)]
    else:
        with pytest.raises(ValueError):gate.host_snapshot(root,code,'b'*40)


def re_digest(value):return len(value)==64 and all(c in'0123456789abcdef'for c in value)


def test_source_binding_host_complete_closure_container_narrow(tmp_path,monkeypatch):
    revision='b'*40;code=tmp_path/'jobs'/revision/gate.JOB/'code'
    for name in(*gate.SOURCE,'infra/extra_code.py'):write(code/name,b'# immutable fixture\n')
    for p in reversed([code,*code.rglob('*')]):
        if p.is_dir():p.chmod(0o555)
    write(code.parent/'revision',(revision+'\n').encode());write(code.parent/'source-sha256',('d'*64+'\n').encode())
    monkeypatch.setattr(gate,'ROOT',tmp_path);monkeypatch.setattr(gate,'__file__',str(code/gate.SOURCE[0]))
    monkeypatch.setattr(gate.files,'__file__',str(code/gate.SOURCE[2]))
    assert set(gate.source_binding(code,revision,host=True)['files'])==set(gate.SOURCE)|{'infra/extra_code.py'}
    with pytest.raises(ValueError):gate.source_binding(code,revision)


@pytest.mark.parametrize('fault',[None,'child_fail','foreign','bad_cid','missing_cid','ps_failure','multi_cid','inspect_failure','rm_failure','post_tamper'])
def test_wrapper_finish_owned_cleanup_and_seal_order(tmp_path,fault):
    # Execute the genuine trap function against tiny Docker/source spies, never a daemon.
    text=(REPO/'infra/run_ycbv_point_evaluate.sh').read_text()
    finish=text[text.index('finish() {'):text.index('\ntrap finish EXIT;')]
    code=tmp_path/'code';out=tmp_path/'out';out.mkdir(mode=0o700);code.mkdir()
    cid='c'*64;image=gate.IMAGE;revision='b'*40;name='unit-name';cidfile=out/'.container.cid'
    if fault!='missing_cid':cidfile.write_text('bad'if fault=='bad_cid'else cid)
    driver='''import json,sys
def write_host_seal(root,code,revision,before,report_before,status,cleanup):
 out=code.parent/'out'
 calls=json.loads((code.parent/'calls.json').read_text())
 (out/'host-post.json').write_text(json.dumps(dict(status=status,cleanup=cleanup,calls=calls)))
 return status==0 and cleanup
'''
    (code/'infra').mkdir();(code/'infra/tudl_holdout_inputs.py').write_text('from pathlib import Path\nimport hashlib\ndef identity(p):\n raw=p.read_bytes();return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())\n');(out/'report.json').write_text('{}');(code/'infra/ycbv_point_evaluate.py').write_text('from pathlib import Path\nROOT=Path('+repr(str(tmp_path))+')\n'+driver)
    (tmp_path/'calls.json').write_text('[]')
    bindir=tmp_path/'bin';bindir.mkdir();spy=bindir/'docker'
    spy.write_text('''#!/usr/bin/python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['FAKE_HOME']);f=os.environ['FAKE_FAULT'];a=sys.argv[1:]
q=p/'calls.json';calls=json.loads(q.read_text());calls.append(a);q.write_text(json.dumps(calls))
cid='c'*64
if a[0]=='ps':
 if f=='ps_failure':sys.exit(1)
 if not(p/'removed').exists():print(cid+'\\n'+'d'*64 if f=='multi_cid'else cid)
elif a[0]=='inspect':
 print('foreign'if f=='foreign'else os.environ['FAKE_IMAGE']+'|/unit-name|run_ycbv_point_evaluate|'+'b'*40)
 if f=='inspect_failure':sys.exit(1)
elif a[0]=='rm':
 if f=='rm_failure':sys.exit(1)
 (p/'removed').write_text('owned-only')
''');spy.chmod(0o755)
    timeout=bindir/'timeout';timeout.write_text('#!/bin/bash\nshift;exec "$@"\n');timeout.chmod(0o755)
    # The actual env-isolated Python seal call gets only its structural arguments.
    if fault=='post_tamper':
        p=code/'infra/ycbv_point_evaluate.py';p.write_text(p.read_text().replace('return status==0 and cleanup','return False'))
    script=tmp_path/'finish.sh'
    script.write_text('set -euo pipefail\nROOT='+repr(str(tmp_path))+';CODE='+repr(str(code))+';REV='+revision+';OUT='+repr(str(out))+
        ';IMAGE='+image+';NAME='+name+';CIDFILE='+repr(str(cidfile))+";BEFORE='{}'\n"+finish+
        '\ntrap finish EXIT\nexit '+('7'if fault=='child_fail'else'0')+'\n')
    environment={'PATH':str(bindir)+':/usr/bin:/bin','HOME':str(tmp_path),'FAKE_HOME':str(tmp_path),'FAKE_FAULT':fault or'none','FAKE_IMAGE':image}
    result=subprocess.run(['bash',str(script)],env=environment,capture_output=True,text=True)
    calls=json.loads((tmp_path/'calls.json').read_text());removed=any(x[0]=='rm'for x in calls)
    assert removed is(fault not in('foreign','bad_cid','missing_cid','ps_failure','multi_cid','inspect_failure'))
    seal=json.loads((out/'host-post.json').read_text())
    assert seal['calls']==calls  # The seal runs strictly after the final cleanup query.
    assert result.returncode==(0 if fault is None else 7 if fault=='child_fail'else 1)
    if fault in(None,'child_fail','post_tamper'):
        assert seal['cleanup']and calls[-1][0]=='ps'and (tmp_path/'removed').exists()
    else:assert seal['cleanup']is False


def test_wrapper_host_post_seal_after_exact_cleanup():
    text=(REPO/'infra/run_ycbv_point_evaluate.sh').read_text()
    assert text.index('BEFORE="$(host_identity)"')<text.index('docker run --rm')
    finish=text[text.index('finish() {'):text.index('\ntrap finish EXIT;')]
    assert finish.index('docker rm -f "$CID"')<finish.index("driver['write_host_seal']")
    assert finish.index("identity=files.identity(")<finish.index('docker rm -f "$CID"')
    assert 'json.loads(sys.argv[4])'in finish and '"$REPORT_BEFORE"'in finish
    assert '[[ "$IDS" == "$CID" ]]'in finish and '--filter "id=$CID"'in finish
    assert 'run_ycbv_point_evaluate|$REV'in finish and '"$IMAGE|/$NAME|'in finish
    assert 'STATUS=$?'in finish and '"$STATUS" "$CLEANUP_OK"'in finish
