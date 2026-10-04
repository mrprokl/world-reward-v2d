"""Tiny manufactured prediction/label arrays, native/Docker spies only."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'infra'),str(REPO/'src')]
SPEC=importlib.util.spec_from_file_location('hand_eval_test',REPO/'infra/mediapipe_hand_evaluate.py')
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)
import mediapipe_hand_scan as scan
import mediapipe_cpu_runtime_verify as rt
from world_reward.hand_scan import NativeHandResult,scan_hands


def seal(path,raw,mode=0o400):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(mode)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def predictions(folder,index,frames=2):
    xyz=np.zeros((1,21,3),np.float64);xyz[0,:,0]=np.linspace(.2,.3,21);xyz[0,:,1]=np.linspace(.3,.4,21)
    def callback(t,_):
        n=0 if t==0 else 1
        return NativeHandResult(xyz[:n],xyz[:n],tuple(['Left']*n),np.array([.9]*n,np.float64))
    result=scan_hands(((t,np.zeros((480,640,3),np.uint8))for t in range(frames)),callback,total_frames=frames,
        image_size=(480,640),method='synthetic',source_refs={'source':'synthetic'},budget_seconds=10.)
    path=folder/f'sequence_{index:03d}.npz';folder.mkdir(parents=True,exist_ok=True);scan.serialize(result,path,np)
    seal(Path(str(path)+'.json'),json.dumps({'native_categories':[[],[dict(category_name='Left',display_name='',index=0)]]}).encode())
    return path


def source(root,revision,entry,helpers):
    code=root/'jobs'/revision/entry/'code'
    for name in helpers:seal(code/name,(REPO/name).read_bytes()if(REPO/name).exists()else b'future independently frozen fixture pin',0o444)
    seal(code/'src/world_reward/__init__.py',b'',0o444)
    for p in reversed(list(code.rglob('*'))):
        if p.is_dir():p.chmod(0o555)
    code.chmod(0o555);seal(code.parent/'revision',(revision+'\n').encode(),0o444);seal(code.parent/'source-sha256',b'd'*64+b'\n',0o444)
    return code


def test_lossless_reload_all_slots_fullT_support_and_categories(tmp_path):
    path=predictions(tmp_path,4)
    obs=gate.reload_observations(rt,np,path,Path(str(path)+'.json'),2,{'source':'synthetic'})
    assert obs.frame_index.tolist()==[0,1]and tuple(f.count for f in obs.frames)==(0,1)
    assert obs.frames[1].original_xy.values[0,0].tolist()==[128.,144.]
    assert obs.frames[1].native_handedness_labels==('Left',)and obs.frames[1].native_handedness_scores.tolist()==[.9]
    assert not obs.frames[1].original_xy.values.flags.writeable


@pytest.mark.parametrize('fault',['indices','offsets','support','pixel','capacity','dtype','category'])
def test_native_restore_rejects_silent_coordinate_or_support_changes(tmp_path,fault):
    path=predictions(tmp_path,4)
    if fault=='category':
        categories=Path(str(path)+'.json');categories.chmod(0o600);seal(categories,b'{"native_categories":[[],[]]}')
    else:
        with np.load(path,allow_pickle=False)as archive:a={k:archive[k]for k in archive.files}
        if fault=='indices':a['frame_index'][0]=1
        elif fault=='offsets':a['frame_offsets'][-1]=99
        elif fault=='support':a['xy_supported'][0,0]=False
        elif fault=='pixel':a['pixel_xy'][0,0,0]+=.5
        elif fault=='capacity':a['capacity_saturation'][0]=True
        else:a['normalized_xyz']=a['normalized_xyz'].astype(np.float32)
        path.chmod(0o600)
        with path.open('wb')as f:np.savez(f,**a)
        path.chmod(0o400)
    with pytest.raises(ValueError):gate.reload_observations(rt,np,path,Path(str(path)+'.json'),2,{'source':'synthetic'})


@pytest.fixture
def native(tmp_path,monkeypatch):
    root=tmp_path/'root';revision='a'*40;code=source(root,revision,gate.ENTRY,gate.HELPERS)
    folder=root/'predictions';seqs=[];private={};frozen={}
    for index in(4,39,74):
        predictions(folder,index);s=dict(subject='20200820-subject-03',sequence=f'20200820_{index:06d}',camera='836212060125',sequence_lex_index=index,frames=2);seqs.append(s)
        labels=root/'private'/s['sequence'];labels.mkdir(parents=True,mode=0o700);private[str(index)]=str(labels)
        for t in range(2):
            path=labels/f'labels_{t:06d}.npz';seg=np.zeros((480,640),np.uint8);joint=np.full((1,21,2),-1.,np.float32)
            if t:seg[120:210,110:210]=255;joint[0,:,0]=np.linspace(128.,192.,21);joint[0,:,1]=np.linspace(144.,192.,21)
            with path.open('xb')as f:np.savez(f,seg=seg,joint_2d=joint,joint_3d=np.array(['FORBIDDEN'],object),pose_y=np.array(['FORBIDDEN'],object))
            path.chmod(0o400);frozen[path]=rt.identity(path,16<<20)
    seal(folder/'report.json',b'{}');frozen.update({p:rt.identity(p,32<<20)for p in folder.iterdir()})
    manifest=root/'manifest.json';manifest_pin=seal(manifest,json.dumps({'sequences':seqs}).encode());frozen[manifest]=manifest_pin
    out=root/'results'/('mediapipe-hand-evaluate-'+revision)/'diagnostics';out.mkdir(mode=0o700,parents=True)
    proof=dict(source_binding=rt.source(root,code,revision,gate.ENTRY,gate.HELPERS),remaining_seconds=10.,
        predictions=str(folder),private_folders=private,manifest_path=str(manifest),manifest_pin=manifest_pin,
        frozen={str(p):pin for p,pin in frozen.items()},scan_pins={'producer_revision':'b'*40})
    proof_path=root/'proof.json';proof_pin=seal(proof_path,json.dumps(proof).encode())
    venv=root/'venv';venv.mkdir();(venv/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    monkeypatch.setattr(rt,'VENV',venv);monkeypatch.setattr(gate.sys,'prefix',str(venv));monkeypatch.setattr(gate.sys,'base_prefix','different')
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    monkeypatch.setattr(np,'__file__',str(venv/'lib/numpy/__init__.py'));monkeypatch.setattr(gate.sys,'platform','linux')
    monkeypatch.setattr(gate.os,'geteuid',lambda:0);monkeypatch.setenv('WR_CODE_REVISION',revision);monkeypatch.delitem(sys.modules,'mediapipe',raising=False)
    original_iterdir=Path.iterdir;original_read=Path.read_text;original_stat=Path.stat
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')])if p==Path('/sys/class/net')else original_iterdir(p))
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'CapEff:\t0000000000000004\n'if p==Path('/proc/self/status')else original_read(p,*a,**k))
    def owned(p,*a,**kw):
        s=original_stat(p,*a,**kw)
        return types.SimpleNamespace(st_uid=0,st_mode=s.st_mode)if p==out else s
    monkeypatch.setattr(Path,'stat',owned)
    return dict(root=root,code=code,out=out,proof_path=proof_path,proof_pin=proof_pin,frozen=frozen,private=private)


def test_native_reads_only_seg_joint2d_after_all_prediction_restore_and_preserves_originals(native,monkeypatch):
    reads=[];original=np.lib.npyio.NpzFile.__getitem__
    def recorded(self,key):reads.append(key);return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',recorded)
    before={p:(p.read_bytes(),p.stat().st_mode)for p in native['frozen']}
    report=gate.run_native(rt,native['root'],native['code'],native['out'],native['proof_path'],native['proof_pin'])
    assert report['pooled']['diagnostic_status']=='pass'and report['pooled']['counts']['frames']==6
    assert report['pooled']['counts']['scored_joints']==51 and report['verified_joint_indices']==[0,*range(5,21)]
    assert report['private_fields_decoded']==['seg','joint_2d']and not report['adoption']and not report['models_loaded']
    assert 'joint_3d'not in reads and 'pose_y'not in reads
    assert reads.index('joint_2d')>=33  # ALL 3×11 numeric prediction arrays before labels.
    assert all((p.read_bytes(),p.stat().st_mode)==v for p,v in before.items())


def test_all_seven_prediction_freezes_precede_first_private_value(native,monkeypatch):
    path=next(p for p in native['frozen']if p.name.endswith('.npz.json'));path.chmod(0o600);path.write_bytes(b'CHANGED');path.chmod(0o400)
    monkeypatch.setattr(np,'load',lambda *_a,**_k:pytest.fail('No values may be decoded before all freezes'))
    with pytest.raises(ValueError):gate.run_native(rt,native['root'],native['code'],native['out'],native['proof_path'],native['proof_pin'])


def test_private_npz_rejects_wrong_publisher_shape(native):
    path=Path(native['private']['4'])/'labels_000001.npz';path.chmod(0o600)
    with path.open('wb')as f:np.savez(f,seg=np.zeros((480,640),np.uint8),joint_2d=np.zeros((21,2),np.float32))
    path.chmod(0o400);proof=json.loads(native['proof_path'].read_bytes());proof['frozen'][str(path)]=rt.identity(path,16<<20)
    native['proof_path'].chmod(0o600);native['proof_pin']=seal(native['proof_path'],json.dumps(proof).encode())
    with pytest.raises(ValueError):gate.run_native(rt,native['root'],native['code'],native['out'],native['proof_path'],native['proof_pin'])


def test_no_host_scientific_or_model_import_and_source_closure():
    script='import sys;sys.path[:0]=['+repr(str(REPO/'infra'))+','+repr(str(REPO/'src'))+'];import mediapipe_hand_evaluate as e;e.modules(__import__("pathlib").Path('+repr(str(REPO))+'));assert "numpy" not in sys.modules and "mediapipe" not in sys.modules'
    subprocess.run([sys.executable,'-I','-B','-S','-c',script],check=True,capture_output=True)
    from infra import azure_job
    files={str(p.relative_to(REPO)):p.read_bytes()for folder in('infra','src','configs')for p in(REPO/folder).rglob('*')if p.is_file()and '__pycache__'not in p.parts}
    files['pyproject.toml']=(REPO/'pyproject.toml').read_bytes()
    closure=set(azure_job.runtime_bundle_paths(files,'infra/run_mediapipe_hand_evaluate.sh'))
    assert set(gate.HELPERS)-{'configs/mediapipe_cpu_runtime_pins.json',gate.SCAN_PINS,'configs/dexycb_hand_acquire_pins.json'}<=closure
    assert 'src/world_reward/hand_evaluation.py'in closure


def test_shell_deadline_no_gpu_model_or_arbitrary_argument():
    shell=(REPO/'infra/run_mediapipe_hand_evaluate.sh').read_text()
    assert '330s'in shell and '[[ $# == 0 ]]'in shell
    assert not any(word in shell for word in('pip install','--gpus','nvidia-smi','curl '))


@pytest.mark.parametrize('fault',['','extra','mode','pin'])
def test_private_inventory_only_three_selected_camera_dirs_original_modes(tmp_path,monkeypatch,fault):
    original=Path.stat
    def uid(p,*a,**kw):
        s=original(p,*a,**kw)
        return types.SimpleNamespace(**{n:(1000 if n=='st_uid'else getattr(s,n))for n in dir(s)if n.startswith('st_')})
    monkeypatch.setattr(Path,'stat',uid)
    seqs=[];retained={}
    for index in(4,39,74):
        s=dict(subject='20200820-subject-03',sequence=f'20200820_{index:06d}',camera='836212060125',frames=1);seqs.append(s)
        relative=f"{s['subject']}/{s['sequence']}/{s['camera']}/labels_000000.npz"
        path=tmp_path/'validation/dexycb_hand_v1/eval_private'/relative
        retained[relative]=seal(path,b'opaque private bytes');path.parent.chmod(0o700)
    if fault=='extra':seal(path.parent/'extra.npz',b'foreign')
    elif fault=='mode':path.chmod(0o444)
    elif fault=='pin':retained[relative]['sha256']='0'*64
    if fault:
        with pytest.raises(ValueError):gate.private_inventory(rt,tmp_path,{'sequences':seqs},retained)
    else:
        files,folders=gate.private_inventory(rt,tmp_path,{'sequences':seqs},retained)
        assert len(files)==len(folders)==3


@pytest.fixture
def host(tmp_path,monkeypatch):
    root=tmp_path/'root';revision='a'*40;code=source(root,revision,gate.ENTRY,gate.HELPERS)
    manifest={'sequences':[]};folders=[]
    for index in(4,39,74):
        s=dict(subject='20200820-subject-03',sequence=f'20200820_{index:06d}',camera='836212060125',sequence_lex_index=index,frames=2)
        manifest['sequences'].append(s);p=root/'private'/s['sequence'];p.mkdir(parents=True,mode=0o700);folders.append(p)
    manifest_path=root/'public/manifest.json';manifest_pin=seal(manifest_path,json.dumps(manifest).encode())
    original_revision='b'*40;pred=root/'results'/('mediapipe-hand-scan-'+original_revision)/'predictions'
    for index in(4,39,74):predictions(pred,index)
    seal(pred/'report.json',b'{}');host_path=pred.parent/'report.json';host_pin=seal(host_path,b'{}')
    frozen={p:rt.identity(p,32<<20)for p in pred.iterdir()};frozen.update({manifest_path:manifest_pin,host_path:host_pin})
    evidence=dict(evidence={'manifest':manifest,'acquisition':{'manifest':manifest_pin},'image':{'Id':'sha256:'+'c'*64}},
        scan_pins={'producer_revision':original_revision},predictions=pred,frozen=frozen,private_folders=folders,manifest_path=manifest_path)
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    monkeypatch.setattr(gate.sys,'platform','linux');monkeypatch.setattr(gate.os,'geteuid',lambda:0)
    monkeypatch.setattr(gate.os,'uname',lambda:types.SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    monkeypatch.setattr(gate,'modules',lambda _:(scan,rt));monkeypatch.setattr(gate,'authenticate',lambda *_:evidence)
    monkeypatch.setattr(rt,'control',lambda *_a,**_kw:b'');cleanups=[]
    monkeypatch.setattr(rt,'cleanup',lambda *_:cleanups.append(True))
    return root,code,revision,evidence,cleanups


def test_host_mount_boundary_snapshot_markers_no_models_or_rgb_and_private_cap(host,monkeypatch):
    root,code,revision,evidence,cleanups=host;commands=[]
    def fake(command,**kw):
        commands.append(command);out=root/'results'/('mediapipe-hand-evaluate-'+revision)/'diagnostics'
        report=dict(stage='mediapipe_hand_private_evaluation',status='pass',phase='complete',producer_revision=revision,
            source_binding=rt.source(root,code,revision,gate.ENTRY,gate.HELPERS),private_fields_decoded=['seg','joint_2d'],
            private_values_read=True,native_graphs=0,prediction_values_modified=False,models_loaded=False,gpu_used=False,
            accuracy_threshold_calibrated=False,adoption=False,quality_claim=False)
        seal(out/'report.json',json.dumps(report).encode())
        for i in(4,39,74):seal(out/f'sequence_{i:03d}.json',b'{}')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(gate.subprocess,'run',fake)
    result=gate.run(root,code,revision)
    assert result['status']=='pass'and result['owned_cleanup_verified']and result['private_values_read']is True
    assert len(commands)==1 and cleanups==[True]
    command=commands[0];mounts=[command[i+1]for i,v in enumerate(command)if v=='--mount']
    assert f'type=bind,src={code.parent},dst={code.parent},readonly'in mounts
    assert sum(str(folder)in m for folder in evidence['private_folders']for m in mounts)==3
    assert sum(',readonly'in m for m in mounts)==14
    assert command[command.index('--cap-add')+1]=='DAC_READ_SEARCH'and command[command.index('--cap-drop')+1]=='ALL'
    assert not any(word in '|'.join(mounts)for word in('weights','vendor','hand_landmarker.task','inputs','RGB','wheels'))
    assert all(m.endswith(',readonly')or '/diagnostics,'in m for m in mounts)


def test_host_native_timeout_retains_fail_and_cleans_only_owned_log(host,monkeypatch):
    root,code,revision,evidence,cleanups=host
    monkeypatch.setattr(gate.subprocess,'run',lambda *a,**kw:(_ for _ in()).throw(subprocess.TimeoutExpired(a[0],300)))
    with pytest.raises(ValueError):gate.run(root,code,revision)
    control=root/'results'/('mediapipe-hand-evaluate-'+revision);report=json.loads((control/'report.json').read_bytes())
    assert report['status']=='fail'and report['private_values_read']is None and cleanups==[True]
    assert not(control/'.native-output').exists()


def test_host_prehash_deadline_stops_before_container(host,monkeypatch):
    root,code,revision,evidence,_=host;clock=[0.];monkeypatch.setattr(gate.time,'monotonic',lambda:clock[0])
    def slow(*_):clock[0]=301.;return evidence
    monkeypatch.setattr(gate,'authenticate',slow)
    monkeypatch.setattr(gate.subprocess,'run',lambda *_a,**_k:pytest.fail('Deadline failure precedes container'))
    with pytest.raises(ValueError):gate.run(root,code,revision)


def test_private_capability_not_arbitrary_override(native,monkeypatch):
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'CapEff:\t0000000000000002\n'if p==Path('/proc/self/status')else original(p,*a,**k))
    with pytest.raises(ValueError):gate.run_native(rt,native['root'],native['code'],native['out'],native['proof_path'],native['proof_pin'])


def test_private_parser_forbidden_field_fails_before_archive(tmp_path):
    with pytest.raises(ValueError):gate.private_field(rt,np,tmp_path/'never-open.npz','joint_3d')


def test_native_wrong_output_namespace_fails_before_any_values(native,monkeypatch):
    monkeypatch.setattr(np,'load',lambda *_a,**_k:pytest.fail('Path guard must precede values'))
    with pytest.raises(ValueError):gate.run_native(rt,native['root'],native['code'],native['root']/'wrong',native['proof_path'],native['proof_pin'])


def test_snapshot_parent_extra_file_fails_before_container(host,monkeypatch):
    root,code,revision,_,_=host;seal(code.parent/'FOREIGN',b'not allowed')
    monkeypatch.setattr(gate.subprocess,'run',lambda *_a,**_k:pytest.fail('Parent gate precedes container'))
    with pytest.raises(ValueError):gate.run(root,code,revision)


@pytest.mark.parametrize('fault',['','host_fail','missing_output','native_timeline','native_labels','source_change','output_changed'])
def test_genuine_sealed_scan_authentication_all_outputs_before_private_values(tmp_path,monkeypatch,fault):
    root=tmp_path/'root';revision='b'*40;old=source(root,revision,scan.ENTRY,scan.HELPERS)
    binding=rt.source(root,old,revision,scan.ENTRY,scan.HELPERS);control=root/'results'/('mediapipe-hand-scan-'+revision)
    out=control/'predictions';seqs=[];declared=[]
    for i in(4,39,74):
        predictions(out,i);s=dict(subject='20200820-subject-03',sequence=f'20200820_{i:06d}',camera='836212060125',sequence_lex_index=i,frames=2)
        seqs.append(s);declared.append(dict(sequence_lex_index=i,frames=2,file=f'sequence_{i:03d}.npz'))
    native=dict(status='pass',phase='complete',producer_revision=revision,source_binding=binding,native_calls=6,native_graphs=1,
        outputs=declared,**{k:False for k in('gpu_used','private_values_read','accuracy_verified','availability_is_visibility',
                                           'handedness_is_detection_confidence','actor_identity_inferred')})
    if fault=='native_timeline':native['outputs'][0]['frames']=1
    elif fault=='native_labels':native['private_values_read']=True
    seal(out/'report.json',json.dumps(native).encode());output_pins={p.name:rt.identity(p,32<<20)for p in out.iterdir()}
    manifest_path=root/scan.BASE/'inputs/manifest.json';manifest_pin=seal(manifest_path,json.dumps({'sequences':seqs,'images':[{}]*6}).encode())
    acquisition={'report':seal(root/scan.BASE/'report.json',b'{"retained_files":{}}'),'manifest':manifest_pin};qualified={'proof':'synthetic qualification'}
    evidence=dict(acquisition=acquisition,runtime=qualified,manifest={'sequences':seqs,'images':[{}]*6})
    host=dict(stage='mediapipe_hand_scan_host_seal',status='fail'if fault=='host_fail'else'pass',producer_revision=revision,
        source_binding=binding,acquisition_pins=acquisition,runtime_pins=qualified,source_rehashed_after=True,owned_cleanup_verified=True,
        gpu_used=False,private_values_read=False,accuracy_verified=False,elapsed_seconds=1.,outputs=output_pins,native_report=native)
    pins=dict(schema='world_reward.mediapipe_hand_scan_pins.v1',producer_revision=revision,report=seal(control/'report.json',json.dumps(host).encode()))
    code=tmp_path/'code';seal(code/gate.SCAN_PINS,json.dumps(pins).encode())
    if fault=='missing_output':(out/'sequence_004.npz.json').unlink()
    elif fault=='output_changed':
        p=out/'sequence_004.npz';p.chmod(0o600);p.write_bytes(b'CHANGED');p.chmod(0o400)
    elif fault=='source_change':
        p=old/'infra/mediapipe_hand_scan.py';p.chmod(0o644);p.write_bytes(b'CHANGED');p.chmod(0o444)
    monkeypatch.setattr(scan,'authenticate',lambda *_:evidence);private_calls=[]
    monkeypatch.setattr(gate,'private_inventory',lambda *_:(private_calls.append(True)or ({},[])))
    if fault:
        with pytest.raises((ValueError,FileNotFoundError)):gate.authenticate(scan,rt,root,code)
        assert private_calls==[]
    else:
        result=gate.authenticate(scan,rt,root,code)
        assert result['predictions']==out and private_calls==[True]and len(result['frozen'])==10


def mask_predictions(folder,index,frames=2):
    import hand_mask_infer as masks
    from world_reward.hand_mask_proposals import propose_hand_masks
    observations=gate.reload_observations(rt,np,predictions(folder/'scan',index),folder/'scan'/f'sequence_{index:03d}.npz.json',frames,{'source':'synthetic'})
    class Synthetic:
        def set_image(self,_):pass
        def predict(self,**kw):
            n=len(kw['box']);value=np.zeros((n,480,640),bool)
            value[:,120:210,110:210]=True
            if 'point_coords'not in kw:value[:,120:210,110:160]=False
            return(value,np.full(n,.8),None)if n==1 else(value[:,None],np.full((n,1),.8),None)
    rows=[]
    for t,hands in enumerate(observations.frames):
        p=propose_hand_masks(np.zeros((480,640,3),np.uint8),hands,Synthetic(),frame_index=t)
        file=f'sequence_{index:03d}_frame_{t:06d}.npz';pin=masks.save_frame(rt,np,folder/file,p)
        rows.append(dict(file=file,sequence_lex_index=index,frame_index=t,slots=len(p.local_ids),usable_boxes=int(p.box_usable.sum()),
            b_interventions=int((~p.b_reuses_a).sum()),a_supported=int(p.mask_supported_a.sum()),b_supported=int(p.mask_supported_b.sum()),**pin))
    return rows


@pytest.fixture
def native_masks(native,monkeypatch):
    import hand_mask_infer as masks
    root,revision=native['root'],'e'*40;code=source(root,revision,gate.ENTRY,gate.source_helpers('v2'))
    monkeypatch.setenv('WR_CODE_REVISION',revision);monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    folder=root/'masks';folder.mkdir();rows=[]
    for i in(4,39,74):rows.extend(mask_predictions(folder,i))
    seal(folder/'report.json',b'{}');frozen={p:rt.identity(p,16<<20)for p in folder.iterdir()if p.is_file()}
    frozen.update({p:pin for p,pin in native['frozen'].items()if '/private/'in str(p)or p.name=='manifest.json'})
    out=gate.control_path(root,revision,'v2')/'diagnostics';out.mkdir(parents=True,mode=0o700)
    proof=json.loads(native['proof_path'].read_bytes());proof.update(source_binding=rt.source(root,code,revision,gate.ENTRY,gate.source_helpers('v2')),
        mask_cohort='v2',mask_infer_pins={'producer_revision':'c'*40},mask_rows=rows,predictions=str(folder),frozen={str(p):pin for p,pin in frozen.items()})
    native['proof_path'].chmod(0o600);pin=seal(native['proof_path'],json.dumps(proof).encode())
    original=Path.stat
    def owned(p,*a,**kw):
        s=original(p,*a,**kw)
        return types.SimpleNamespace(st_uid=0,st_mode=s.st_mode)if p==out else s
    monkeypatch.setattr(Path,'stat',owned);monkeypatch.setattr(masks,'__file__',str(code/'infra/hand_mask_infer.py'))
    return dict(native,code=code,out=out,proof_pin=pin,frozen=frozen,rows=rows)


def test_mask_native_all_frames_losslessly_restored_before_seg_only_no_joint_or_models(native_masks,monkeypatch):
    import hand_mask_infer as masks
    loads=[];real=masks.load_frame;private_reads=[];real_private=gate.private_field
    monkeypatch.setattr(masks,'load_frame',lambda rt,np,path,t:(loads.append((path.name,t))or real(rt,np,path,t)))
    def private(rt,np,path,name):
        private_reads.append(name);assert len(loads)>=6;return real_private(rt,np,path,name)
    monkeypatch.setattr(gate,'private_field',private)
    before={p:p.read_bytes()for p in native_masks['frozen']}
    r=gate.run_native(rt,native_masks['root'],native_masks['code'],native_masks['out'],native_masks['proof_path'],native_masks['proof_pin'],'v2')
    assert r['private_fields_decoded']==['seg']and private_reads==['seg']*6
    assert r['pooled']['original_frames']==6 and r['pooled']['positive_frames']==3 and r['pooled']['unlabelled_frames']==3
    assert r['pooled']['a']['mean_positive_dice']<r['pooled']['b']['mean_positive_dice']==1.
    assert r['mask_forward_gate']=='pass'and r['pooled']['positive_b_intervention_frames']==3
    assert not r['false_positive_truth_certified']and not r['physical_identity_inferred']and not r['challenge_adoption']
    assert all(p.read_bytes()==raw for p,raw in before.items())and not({'mediapipe','torch'}&sys.modules.keys())
    assert len(loads)==12 and all(len(json.loads((native_masks['out']/f'sequence_{i:03d}.json').read_bytes())['frame_index'])==2 for i in(4,39,74))


@pytest.mark.parametrize('fault',['mask_changed','duplicate_frame','counts','reuse','missing_frame','seg_dtype'])
def test_mask_native_rejects_tamper_before_private_or_typed_seg(native_masks,monkeypatch,fault):
    proof=json.loads(native_masks['proof_path'].read_bytes());row=proof['mask_rows'][1];path=Path(proof['predictions'])/row['file']
    if fault=='duplicate_frame':proof['mask_rows'][1]=proof['mask_rows'][0]
    elif fault=='counts':row['b_interventions']=0
    elif fault=='missing_frame':proof['mask_rows'].pop()
    elif fault=='seg_dtype':
        path=Path(proof['private_folders']['4'])/'labels_000001.npz';path.chmod(0o600)
        with path.open('wb')as f:np.savez(f,seg=np.zeros((480,640),np.int32))
        path.chmod(0o400);proof['frozen'][str(path)]=rt.identity(path,16<<20)
    else:
        with np.load(path)as a:arrays={k:a[k]for k in a.files}
        if fault=='reuse':arrays['b_reuses_a'][:]=True
        else:arrays['masks_b'][0,0,0]=True
        path.chmod(0o600)
        with path.open('wb')as f:np.savez_compressed(f,**arrays)
        path.chmod(0o400)
        if fault=='reuse':proof['frozen'][str(path)]=rt.identity(path,16<<20)
    native_masks['proof_path'].chmod(0o600);pin=seal(native_masks['proof_path'],json.dumps(proof).encode())
    if fault!='seg_dtype':monkeypatch.setattr(gate,'private_field',lambda *_:pytest.fail('All mask schema/counts freeze before private values'))
    with pytest.raises((ValueError,KeyError)):gate.run_native(rt,native_masks['root'],native_masks['code'],native_masks['out'],native_masks['proof_path'],pin,'v2')


def test_mask_pool_weights_every_positive_frame_and_never_drops_inconclusive_or_regressing_clip():
    def result(delta,interventions=1):
        arm=lambda values:types.SimpleNamespace(iou=values,dice=values,mean_positive_iou=0.,mean_positive_dice=0.,**{k:0 for k in('positive_empty_union_frames','total_predicted_pixels',
            'total_object_label_pixels','total_background_label_pixels','positive_predicted_pixels','positive_object_label_pixels','positive_background_label_pixels')})
        n=len(delta);return types.SimpleNamespace(frame_index=np.arange(n),unlabelled=np.zeros(n,bool),positive_frames=n,
            no_proposal_frames=0,positive_no_proposal_frames=0,b_intervention_frames=interventions,positive_b_intervention_frames=interventions,
            paired_dice_delta=tuple(delta),mean_positive_paired_dice_delta=sum(delta)/n if n else None,
            a=arm((0.,)*n),b=arm(tuple(delta)))
    pooled,gate_status=gate.mask_pool([result([.2]),result([.1]*3),result([0.])])
    assert pooled['mean_positive_paired_dice_delta']==pytest.approx(.1)and gate_status=='pass'
    assert gate.mask_pool([result([.9]),result([-.1]),result([.9])])[1]=='fail'
    assert gate.mask_pool([result([.9]),result([.1],0),result([.9])])[1]=='fail'
    assert gate.mask_pool([result([.9]),result([]),result([.9])])[1]=='inconclusive'
    assert gate.mask_pool([result([])]*3)[0]['mean_positive_paired_dice_delta']is None


def test_mask_profile_fixed_cli_and_host_remains_scientific_import_free():
    assert gate.source_helpers()==gate.HELPERS and 'src/world_reward/hand_mask_evaluation.py'in gate.source_helpers('v2')
    assert 'configs/dexycb_hand_acquire_pins.json'not in gate.source_helpers('v2')
    assert gate.TEMPORAL_MASK_PINS in gate.source_helpers('v3')and 'configs/dexycb_hand_acquire_v3_pins.json'in gate.source_helpers('v3')
    assert 'configs/mediapipe_hand_scan_v2_pins.json'not in gate.source_helpers('v3')
    with pytest.raises(ValueError):gate.source_helpers('v4')
    script='import sys;sys.path[:0]=['+repr(str(REPO/'infra'))+','+repr(str(REPO/'src'))+'];import mediapipe_hand_evaluate as e;import hand_mask_infer;e.source_helpers("v2");assert not({"numpy","torch","mediapipe"}&sys.modules.keys())'
    subprocess.run([sys.executable,'-I','-B','-S','-c',script],check=True,capture_output=True)
    shell=(REPO/'infra/run_mediapipe_hand_evaluate.sh').read_text()
    assert '"$1" == --mask-cohort && "$2" == v2'in shell and '"$@"'in shell
    assert '"$1" == --mask-cohort && "$2" == v3'in shell


@pytest.mark.parametrize('fault',['','host_fail','core_omitted','original_source','missing_frame','native_private','call_counts','output_changed','row_count'])
def test_mask_genuine_original_host_fullT_source_and_outputs_before_private(tmp_path,monkeypatch,fault):
    import hand_mask_infer as masks
    root=tmp_path/'root';revision='b'*40;old=source(root,revision,masks.ENTRY,masks.HELPERS)
    binding=rt.source(root,old,revision,masks.ENTRY,masks.HELPERS);control=root/'results'/('hand-mask-infer-'+revision);out=control/'predictions'
    out.mkdir(parents=True);rows=[];seqs=[];images=[]
    for i in(4,39,74):
        seqs.append(dict(subject='20200903-subject-04',sequence=f'20200903_{i:06d}',camera='836212060125',sequence_lex_index=i,frames=2))
        for row in mask_predictions(out,i):
            image=dict(sequence_lex_index=i,frame_position=row['frame_index'],source_frame_id=row['frame_index'],file=f'{i}_{row["frame_index"]}.jpg',sha256='e'*64)
            images.append(image);row.update(rgb_file=image['file'],rgb_sha256=image['sha256']);rows.append(row)
    # Synthetic creation scratch is not part of the immutable producer output.
    import shutil
    shutil.rmtree(out/'scan')
    code=tmp_path/'code';protocol_raw=(REPO/masks.PROTOCOL).read_bytes();protocol_pin=seal(code/masks.PROTOCOL,protocol_raw,0o444)
    manifest_path=root/scan.profile('v2')['base']/'inputs/manifest.json';manifest={'sequences':seqs,'images':images}
    manifest_pin=seal(manifest_path,json.dumps(manifest).encode());acq={'manifest':manifest_pin,'report':seal(root/scan.profile('v2')['base']/'report.json',b'{"retained_files":{}}')}
    evidence=dict(manifest=manifest,acquisition=acq,image={'Id':'sha256:'+'c'*64});scan_pins={'producer_revision':'f'*40}
    native=dict(stage='public_full_t_paired_hand_sam2',status='pass',phase='complete',producer_revision=revision,source_binding=binding,
        protocol_identity=protocol_pin,manifest_identity=manifest_pin,scan_pins=scan_pins,image_id=gate.SAM2_IMAGE,budget_seconds=900,
        source_rehashed_after=True,all_original_frames=True,network='none',device='cuda',sam2_native_postprocessing=True,
        private_values_read=False,quality_verified=False,identity_accepted=False,contacts_inferred=False,geometry_inferred=False,
        encoder_attempts=3,encoder_completed=3,a_attempts=3,a_completed=3,b_attempts=3,b_completed=3,outputs=rows)
    if fault=='core_omitted':binding=copy.deepcopy(binding);binding['helpers'].pop('src/world_reward/hand_mask_proposals.py');native['source_binding']=binding
    elif fault=='native_private':native['private_values_read']=True
    elif fault=='call_counts':native['b_completed']=2
    elif fault=='row_count':rows[1]['b_interventions']=2
    seal(out/'report.json',json.dumps(native).encode());outputs={p.name:rt.identity(p,16<<20)for p in out.iterdir()}
    host=dict(stage='hand_mask_infer_host_seal',status='fail'if fault=='host_fail'else'pass',producer_revision=revision,source_binding=binding,
        protocol_identity=protocol_pin,scan_pins=scan_pins,image_id=gate.SAM2_IMAGE,private_values_read=False,quality_verified=False,
        owned_cleanup_verified=True,source_rehashed_after=True,elapsed_seconds=1.,native_report=native,outputs=outputs)
    pins=dict(schema='world_reward.hand_mask_infer_pins.v1',producer_revision=revision,report=seal(control/'report.json',json.dumps(host).encode()))
    seal(code/gate.MASK_PINS,json.dumps(pins).encode(),0o444)
    if fault=='missing_frame':(out/rows[0]['file']).unlink()
    elif fault=='output_changed':p=out/rows[0]['file'];p.chmod(0o600);p.write_bytes(b'changed');p.chmod(0o400)
    elif fault=='original_source':p=old/'infra/hand_mask_infer.py';p.chmod(0o600);p.write_bytes(b'changed');p.chmod(0o444)
    monkeypatch.setattr(masks,'__file__',str(code/'infra/hand_mask_infer.py'))
    monkeypatch.setattr(masks,'authenticate_scan',lambda *_:(evidence,scan_pins,root/'oldscan',{}))
    private_calls=[];monkeypatch.setattr(gate,'private_inventory',lambda *_:(private_calls.append(True)or ({},[])))
    if fault:
        with pytest.raises((ValueError,FileNotFoundError)):gate.authenticate_masks(scan,rt,root,code)
        assert private_calls==[]
    else:
        result=gate.authenticate_masks(scan,rt,root,code)
        assert private_calls==[True]and len(result['mask_rows'])==6 and result['predictions']==out
        assert len(result['native_frozen'])==11 and result['mask_infer_pins']==pins


def test_mask_host_mounts_only_exact_private_mask_artifacts_and_marker_parent(host,monkeypatch):
    root,_,revision,evidence,cleanups=host;code=source(root,'e'*40,gate.ENTRY,gate.source_helpers('v2'));revision='e'*40
    evidence=dict(evidence,mask_infer_pins={'producer_revision':'c'*40},mask_rows=[])
    monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]));monkeypatch.setattr(gate,'authenticate_masks',lambda *_:evidence)
    command=[]
    def fake(args,**_):
        command.extend(args);out=gate.control_path(root,revision,'v2')/'diagnostics'
        r=dict(stage='mediapipe_hand_mask_private_evaluation',status='pass',phase='complete',producer_revision=revision,
            source_binding=rt.source(root,code,revision,gate.ENTRY,gate.source_helpers('v2')),private_fields_decoded=['seg'],
            private_values_read=True,native_graphs=0,prediction_values_modified=False,models_loaded=False,gpu_used=False,
            accuracy_threshold_calibrated=False,adoption=False,quality_claim=False,cohort='v2',mask_infer_pins=evidence['mask_infer_pins'],
            physical_identity_inferred=False,false_positive_truth_certified=False,challenge_adoption=False)
        seal(out/'report.json',json.dumps(r).encode())
        for i in(4,39,74):seal(out/f'sequence_{i:03d}.json',b'{}')
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(gate.subprocess,'run',fake)
    result=gate.run(root,code,revision,'v2');mounts=[command[i+1]for i,v in enumerate(command)if v=='--mount']
    assert result['status']=='pass'and result['cohort']=='v2'and cleanups==[True]and command[-2:]==['--mask-cohort','v2']
    assert f'type=bind,src={code.parent},dst={code.parent},readonly'in mounts
    assert not any(word in '|'.join(mounts)for word in('weights','vendor','hand_landmarker.task','inputs','RGB','wheels'))
    assert command[command.index('--cap-add')+1]=='DAC_READ_SEARCH'and '--gpus'not in command
