"""Manufactured sealed48 metadata/byte lineage; no NumPy, RGB or real roles."""
import ast
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest
import vcoco_full_interaction_inputs as p


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if type(value) is bytes else p.join.encode(value)
    if path.exists(): path.chmod(0o600)
    path.write_bytes(raw); path.chmod(0o400)
    return pin(raw)


def projected_root(monkeypatch, root):
    original = Path.lstat
    def lstat(path, *args, **kwargs):
        s = original(path, *args, **kwargs)
        if path == root or root in path.parents:
            values = {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_nlink',
                'st_uid', 'st_gid', 'st_mtime_ns', 'st_ctime_ns')}
            values.update(st_uid=0, st_gid=0); return SimpleNamespace(**values)
        return s
    monkeypatch.setattr(Path, 'lstat', lstat)


def source_tree(code, revision, names, executable):
    for n in names: write(code/n, n.encode())
    write(code.parent/'revision', (revision+'\n').encode())
    write(code.parent/'source-sha256', (p.DECLARATION['source_XZ_sha256']+'\n').encode())
    for path in (code, *code.rglob('*')):
        path.chmod(0o555 if path.is_dir() or str(path.relative_to(code)) in executable else 0o444)
    for n in ('revision', 'source-sha256'): (code.parent/n).chmod(0o444)


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'root'; root.mkdir(mode=0o700); projected_root(monkeypatch, root)
    monkeypatch.setattr(p, 'ROOT', root); monkeypatch.setattr(p.join, 'ROOT', root)
    out = root/'results/join'; out.mkdir(parents=True, mode=0o700); monkeypatch.setattr(p.join, 'OUTPUT', out)
    revision, entry = 'b'*40, 'run_new_cost'
    code = root/'jobs'/revision/entry/'code'
    old = root/'jobs'/p.JOIN_REV/p.join.ENTRY/'code'
    names = (*p.join.NATIVE_FILES, 'infra/run_vcoco_full_interaction_join.sh')
    helpers = (*names, p.HELPER, 'infra/run_new_cost.sh')
    monkeypatch.setattr(p.join, 'helpers', lambda: names)
    source_tree(old, p.JOIN_REV, names, p.join.EXECUTABLES)
    source_tree(code, revision, helpers, p.join.EXECUTABLES)
    monkeypatch.setattr(p, '__file__', str(code/p.HELPER))
    monkeypatch.setattr(p.join, '__file__', str(code/p.join.NATIVE_FILES[0]))
    for n in helpers:
        key = Path(n).stem if n.startswith('infra/') else '.'.join(Path(n).with_suffix('').parts[1:])
        if key in sys.modules: monkeypatch.setattr(sys.modules[key], '__file__', str(code/n))
    monkeypatch.setattr(p.join, 'NUMERICAL_PIN', p.rt.identity(code/p.join.NATIVE_FILES[2]))
    binding = p.rt.source(root, old, p.JOIN_REV, p.join.ENTRY, names)
    monkeypatch.setattr(p, 'DECLARATION', dict(p.DECLARATION, files=len(names), entries=binding['entries'], closure_sha256=binding['closure_sha256']))
    original = dict(binding=binding, states=p.join.modes(old, p.join.EXECUTABLES))

    endpoint_out, hoi_out, pose_dest = root/'endpoint', root/'hoi', root/'pose'
    banks, poses, hoi_rows = [], [], []
    for i in range(48):
        base = dict(image_id=f'{i:032x}', original_slot=i, acquired_ordinal=i, original_frame_index=0,
            image_size=[8, 12], file=f'image_{i:06d}.npz')
        rows = []
        for folder, fields in ((endpoint_out,p.join.numerical.ENDPOINT_FIELDS),
                               (pose_dest/'banks',p.join.numerical.POSE_FIELDS),(hoi_out,p.join.numerical.HOI_FIELDS)):
            identity = write(folder/base['file'], bytes([i+1]))
            arrays = {n: dict(shape=[],dtype='<i8',sha256=hashlib.sha256(n.encode()).hexdigest()) for n in fields}
            rows.append(dict(base,identity=identity,arrays=arrays))
        e,v,h = rows
        e.update(person_ids=[],person_retained_rows=0)
        v.update(person_ids=[],persons=0,endpoint_bank_identity=e['identity'])
        h.update(source_person_ids=[],hand_object_pairs=0,endpoint_bank_identity=e['identity'])
        banks.append(e); poses.append(v); hoi_rows.append(h)
    pose_projection = dict(schema='world_reward.public_person_pose_reference.v1',rows=poses)
    projection_path = pose_dest/'public-pose.json'; projection_pin = write(projection_path,pose_projection)
    eproof = dict(source={'old_endpoint':True}, files={str(endpoint_out/r['file']):r['identity'] for r in banks}, states={'endpoint':(1,2)})
    hproof = dict(rows=hoi_rows,files={str(hoi_out/r['file']):r['identity'] for r in hoi_rows},states={'hoi':(3,4)})
    pproof = dict(rows=poses,projection_path=str(projection_path),projection_identity=projection_pin,
        files={str(pose_dest/'banks'/r['file']):r['identity'] for r in poses}, states={'pose':(5,6)},
        pose_receipt_pins=deepcopy(p.POSE_PINS),import_identity=deepcopy(p.POSE_IMPORT_PIN),
        import_source={'producer_revision':p.POSE_REPLICA_REV},original_pose_revision='e'*40,
        sender_source_live_verified=False,sender_runtime_live_verified=False)
    calls=[]
    endpoint = SimpleNamespace(endpoint=SimpleNamespace(OUTPUT=endpoint_out))
    def sender(code_, current, deadline):
        assert code_ == code and current['producer_revision'] == revision
        calls.append('endpoint'); return {'banks':deepcopy(banks)}, {}, {}, deepcopy(eproof)
    endpoint.sender_inputs = sender
    hoi = SimpleNamespace(OUTPUT=hoi_out)
    pose = SimpleNamespace(DEST=pose_dest,POSE_REV='e'*40,PUBLIC_SCHEMA=pose_projection['schema'])
    def receiver(code_, rev, receipt):
        assert code_ == code and rev == p.POSE_REPLICA_REV and receipt == p.POSE_IMPORT_PIN
        calls.append('pose'); return deepcopy(pproof)
    pose.authenticate_receiver = receiver
    monkeypatch.setattr(p.join,'host_modules',lambda:(endpoint,hoi,pose))
    def hoi_inputs(code_, current, projection, deadline):
        assert code_ == code and current['producer_revision'] == revision and projection['banks'] == banks
        calls.append('hoi'); return deepcopy(hproof)
    monkeypatch.setattr(p.join,'hoi_inputs',hoi_inputs)
    current = p._current(code,revision,entry,helpers)
    before, proof = p._inputs(code,current,original,time.monotonic()+30,lambda:None)
    evidence=[]
    for i,r in enumerate(proof['records']):
        metadata = p.join.expected_evidence_metadata(r)
        arrays = {n:dict(shape=shape,dtype=dtype,sha256=hashlib.sha256(n.encode()).hexdigest()) for n,(shape,dtype)in metadata.items()}
        raw = [{n:r[k]['arrays'][n]for n in fields} for k,fields in zip(('endpoint','pose','hoi'),
            (p.join.numerical.ENDPOINT_FIELDS,p.join.numerical.POSE_FIELDS,p.join.numerical.HOI_FIELDS))]
        digest=hashlib.sha256()
        for prefix in ('base__','local__','bridge__'):
            for n in metadata:
                if n.startswith(prefix):digest.update(p.join.encode([n[len(prefix):],arrays[n]]))
        for prefix in ('base','local','bridge'):digest.update(p.join.encode([arrays[prefix+'_features'],arrays[prefix+'_supported']]))
        row=dict(image_id=r['endpoint']['image_id'],original_slot=i,acquired_ordinal=i,persons=0,objects=3600,native_pairs=0,
            person_side_object_rows=0,person_side_pair_rows=0,pair_object_rows=0,supported_counts=[0,0,0],nan_counts=[0,0,0],
            raw_bank_fingerprint=hashlib.sha256(p.join.encode(raw)).hexdigest(),evidence_fingerprint=digest.hexdigest(),
            file=f'image_{i:06d}.npz',identity=write(out/f'image_{i:06d}.npz',b'evidence'),person_ids=[],
            object_id_fingerprint=arrays['object_ids'],arrays=arrays,source_observation_references=[
                [n,'f'*64]for n in('world_reward.person_pose_observations.v1','world_reward.generic_object_observations.v1','world_reward.hoi_detr_observations.v1')])
        evidence.append(row)
    proof_pin = write(out/'proof.json',proof)
    native=dict(schema=p.join.SCHEMA,stage='native_full48_interaction_join',status='pass',phase='complete',
        producer_revision=p.JOIN_REV,image_id=p.join.IMAGE,proof_identity=proof_pin,source_inputs_rehashed_after=True,
        all144_banks_prevalidated=True,input_arrays_per_image=47,output_arrays_per_image=46,models_loaded=0,
        images=evidence,**{n:False for n in p.join.FLAGS})
    native_pin=write(out/'native.json',native);write(out/'.container.cid',('a'*64+'\n').encode())
    host=dict(schema=p.join.SCHEMA,stage='full48_interaction_join_host',status='pass',phase='complete',producer_revision=p.JOIN_REV,
        source_binding=binding,input_proof=p.normalized(before),image_id=p.join.IMAGE,models_loaded=0,images=evidence,
        native_identity=native_pin,native_exit_status=0,pose_receipt_pins=deepcopy(p.POSE_PINS),pose_replica_identity=deepcopy(p.POSE_IMPORT_PIN),
        outputs_sealed=True,owned_cleanup_verified=True,source_inputs_image_rehashed_after=True,output_capacity_gate_passed=True,
        saved_outputs={x.name:p.rt.identity(x)for x in out.iterdir()},**{n:False for n in p.join.FLAGS})
    pins={'report.json':write(out/'report.json',host),'native.json':native_pin,'proof.json':proof_pin}
    out.chmod(0o500);runtime=dict(Id=p.join.IMAGE,Architecture='amd64',Os='linux',RootFS={'Type':'layers','Layers':['sha256:'+'1'*64]*44})
    commands=[]
    def command(argv, deadline):
        commands.append(argv)
        if argv[:3]==['docker','image','inspect']:return p.join.encode(runtime).decode()
        if argv[:2]==['docker','ps']:return ''
        raise AssertionError(argv)
    monkeypatch.setattr(p.join,'command',command);calls.clear()
    return SimpleNamespace(code=code,old=old,revision=revision,entry=entry,helpers=helpers,pins=pins,host=host,native=native,
        proof=proof,before=before,original=original,pproof=pproof,eproof=eproof,hproof=hproof,calls=calls,commands=commands,runtime=runtime,out=out)


def invoke(f, checkpoint=lambda:None):
    return p.authenticate_saved_join(f.code,f.revision,f.entry,f.helpers,f.pins,checkpoint,deadline=time.monotonic()+30)


def replace_receipts(f):
    for name,value in (('report.json',f.host),('native.json',f.native),('proof.json',f.proof)):
        path=f.out/name;path.chmod(0o600);f.pins[name]=write(path,value)
    f.host['native_identity']=f.pins['native.json']
    f.native['proof_identity']=f.pins['proof.json'];f.pins['native.json']=write(f.out/'native.json',f.native)
    f.host['native_identity']=f.pins['native.json']
    for n in ('native.json','proof.json'):f.host['saved_outputs'][n]=f.pins[n]
    f.pins['report.json']=write(f.out/'report.json',f.host)


def test_complete_original_context_roundtrip_public_outputs_and_no_mutation(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);before=deepcopy((f.before,f.proof,f.pins));result=invoke(f)
    assert result['source_inputs_outputs_image_rehashed_after']is True and len(result['rows'])==48
    assert len(result['public_proof']['files'])==144 and len(result['evidence_files'])==48 and len(result['output_files'])==52
    assert result['host_lineage']['original_input_proof']['current_source']==f.original
    assert result['host_lineage']['current_source']!=f.original
    assert result['public_proof']==p.normalized(f.proof) and (f.before,f.proof,f.pins)==before
    assert f.calls==['endpoint','hoi','pose']*2 and len([a for a in f.commands if a[:3]==['docker','image','inspect']])==2
    assert any('name=^/world-reward-vcoco-full-interaction-join-'+p.JOIN_REV[:12]+'$' in a for a in f.commands)
    raw=p.join.encode((result['public_proof'],result['rows'],result['evidence_files'])).decode()
    assert not any('"'+n+'"'in raw for n in('photo_id','official_split','publisher_metadata','native_image_id','input_proof'))
    assert result['sender_source_live_verified']is result['sender_runtime_live_verified']is False
    result['rows'][0]['persons']=99;assert f.native['images'][0]['persons']==0
    result['host_lineage']['join_declaration']['files']=0;assert p.DECLARATION['files']>0


@pytest.mark.parametrize('fault',['hostfail','nativefail','nativeexit','sourcepost','cleanup','seal','capacity','models','rgb',
    'hostsource','newsourceinoldproof','intmapping','missingrow','missingfile','extra','outputmode','outputdirectory',
    'symlink','hardlink','savedhash','rawfingerprint','evidencefingerprint','objectdtype','refs','posepin','importpin','publication'])
def test_saved_failure_schema_alias_and_lineage_rejected(tmp_path,monkeypatch,fault):
    f=fixture(tmp_path,monkeypatch)
    if fault=='hostfail':f.host['status']='fail'
    elif fault=='nativefail':f.native['status']='fail'
    elif fault=='nativeexit':f.host['native_exit_status']=1
    elif fault=='sourcepost':f.host['source_inputs_image_rehashed_after']=False
    elif fault=='cleanup':f.host['owned_cleanup_verified']=False
    elif fault=='seal':f.host['outputs_sealed']=False
    elif fault=='capacity':f.host['output_capacity_gate_passed']=False
    elif fault=='models':f.host['models_loaded']=1
    elif fault=='rgb':f.host['RGB_decoded']=True
    elif fault=='hostsource':f.host['source_binding']['entries']+=1
    elif fault=='newsourceinoldproof':f.host['input_proof']['current_source']=p.normalized(p._current(f.code,f.revision,f.entry,f.helpers))
    elif fault=='intmapping':f.host['input_proof']['endpoint']['states']['0']=[999]
    elif fault=='missingrow':f.native['images'].pop()
    elif fault=='missingfile':f.out.chmod(0o700);f.out.joinpath('image_000047.npz').unlink();f.out.chmod(0o500)
    elif fault=='extra':f.out.chmod(0o700);write(f.out/'foreign.json',b'untouched');f.out.chmod(0o500)
    elif fault=='outputmode':f.out.joinpath('image_000000.npz').chmod(0o600)
    elif fault=='outputdirectory':f.out.chmod(0o700)
    elif fault=='symlink':
        f.out.chmod(0o700);q=f.out/'image_000000.npz';q.rename(f.out/'original');q.symlink_to(f.out/'original');f.out.chmod(0o500)
    elif fault=='hardlink':__import__('os').link(f.out/'image_000000.npz',tmp_path/'alias')
    elif fault=='savedhash':f.host['saved_outputs']['image_000000.npz']['sha256']='e'*64
    elif fault=='rawfingerprint':f.native['images'][0]['raw_bank_fingerprint']='e'*64
    elif fault=='evidencefingerprint':f.native['images'][0]['evidence_fingerprint']='e'*64
    elif fault=='objectdtype':f.native['images'][0]['arrays']['object_ids']['dtype']='<i8'
    elif fault=='refs':f.native['images'][0]['source_observation_references'][0][0]='not_person'
    elif fault=='posepin':f.host['pose_receipt_pins']['proof.json']['bytes']+=1
    elif fault=='importpin':f.host['pose_replica_identity']['bytes']+=1
    else:f.native['publication_failed']=True
    replace_receipts(f)
    with pytest.raises((ValueError,FileNotFoundError)):invoke(f)
    if fault=='extra':assert (f.out/'foreign.json').read_bytes()==b'untouched'


@pytest.mark.parametrize('fault',['closure','entries','xz','oldmode','oldorigin','currenthelper','currentmode','currentorigin'])
def test_complete_old_and_new_source_guards(tmp_path,monkeypatch,fault):
    f=fixture(tmp_path,monkeypatch)
    if fault=='closure':monkeypatch.setitem(p.DECLARATION,'closure_sha256','0'*64)
    elif fault=='entries':monkeypatch.setitem(p.DECLARATION,'entries',999)
    elif fault=='xz':
        q=f.old.parent/'source-sha256';q.chmod(0o600);write(q,('0'*64+'\n').encode());q.chmod(0o444)
    elif fault=='oldmode':f.old.joinpath(p.join.NATIVE_FILES[0]).chmod(0o555)
    elif fault=='oldorigin':
        monkeypatch.setattr(p.join,'source',lambda *a:pytest.fail('Old origin producer must not execute'))
        monkeypatch.setattr(p.join,'authenticate',lambda *a:pytest.fail('Old producer context must not execute'))
        assert invoke(f)['source_inputs_outputs_image_rehashed_after'];return
    elif fault=='currenthelper':
        q=f.code/p.join.NATIVE_FILES[0];q.chmod(0o600);write(q,b'changed');q.chmod(0o444)
    elif fault=='currentmode':f.code.chmod(0o755)
    else:monkeypatch.setattr(p,'__file__',str(tmp_path/'foreign.py'))
    with pytest.raises(ValueError):invoke(f)


@pytest.mark.parametrize('fault',['output','input','source','runtime','CID','name','latecheckpoint'])
def test_post_capture_and_exact_process_absence(tmp_path,monkeypatch,fault):
    f=fixture(tmp_path,monkeypatch);original=p.join.image;count=[0]
    def image(deadline):
        count[0]+=1;result=original(deadline)
        if count[0]==1:
            if fault=='output':q=f.out/'image_000000.npz';q.chmod(0o600);write(q,b'changed')
            elif fault=='input':q=Path(f.proof['records'][0]['paths'][0]);q.chmod(0o600);write(q,b'changed')
            elif fault=='source':q=f.old/p.join.NATIVE_FILES[0];q.chmod(0o600);write(q,b'changed');q.chmod(0o444)
            elif fault=='runtime':f.runtime['RootFS']['Layers'][0]='sha256:'+'2'*64
        return result
    monkeypatch.setattr(p.join,'image',image)
    if fault in('CID','name'):
        command=p.join.command
        def live(argv,deadline):
            if argv[:2]==['docker','ps']and any(x.startswith('id=')if fault=='CID'else x.startswith('name=')for x in argv):return 'a'*64
            return command(argv,deadline)
        monkeypatch.setattr(p.join,'command',live)
    def checkpoint():
        if fault=='latecheckpoint'and count[0]==2:raise TimeoutError('authored late deadline')
    with pytest.raises((ValueError,TimeoutError)):invoke(f,checkpoint)
    assert not any(a[:2]==['docker','rm']for a in f.commands)


@pytest.mark.parametrize('pins',[{}, {'report.json':{}}, {'report.json':{'bytes':True,'sha256':'a'*64},
    'native.json':{'bytes':1,'sha256':'a'*64},'proof.json':{'bytes':1,'sha256':'a'*64}}])
def test_required_independent_controls_fail_before_read(monkeypatch,pins):
    monkeypatch.setattr(p,'_current',lambda *a:pytest.fail('Invalid controls must precede reads'))
    with pytest.raises(ValueError):p.authenticate_saved_join(Path('/unopened'),'b'*40,'new',(),pins,lambda:None,deadline=time.monotonic()+10)


@pytest.mark.parametrize('deadline',[False,None,float('inf'),0.])
def test_invalid_or_expired_explicit_deadline_before_io(monkeypatch,deadline):
    monkeypatch.setattr(p,'_current',lambda *a:pytest.fail('No source read after deadline'))
    with pytest.raises(TimeoutError):p.authenticate_saved_join(Path('/unopened'),'b'*40,'new',(),{},lambda:None,deadline=deadline)


def test_original_full_git_archive_helpers_and_exec_mode_identity():
    import azure_job
    raw=subprocess.check_output(['rtk','proxy','git','archive','--format=tar',p.JOIN_REV])
    with tarfile.open(fileobj=io.BytesIO(raw))as archive:
        files={r.name:(r,archive.extractfile(r).read())for r in archive if r.isfile()}
    selected=azure_job.runtime_bundle_paths({n:v[1]for n,v in files.items()},'infra/'+p.join.ENTRY+'.sh')
    assert len(selected)==p.DECLARATION['files'] and set(p.join.helpers())<=set(selected)
    assert all(Path(n).read_bytes()==files[n][1]for n in p.join.helpers())
    assert p.join.EXECUTABLES==frozenset(n for n in selected if files[n][0].mode&0o111)
    assert p.HELPER not in selected


@pytest.mark.parametrize('people,pairs',[(0,False),(0,True),(1,False),(1,True)])
def test_metadata_summary_digest_matches_unchanged_original_numerics(people,pairs):
    from test_vcoco_interaction_observations import fixture as arrays_fixture
    from test_vcoco_full_interaction_core import at_slot
    args,_=arrays_fixture(people,pairs);args=at_slot(args,47)
    value=p.join.numerical.reconstruct_interaction(*args,population=48)
    row=p.join.summary(value);row['arrays']={n:p.join.numerical._identity(a)for n,a in p.join.evidence_arrays(value).items()}
    record=dict(endpoint=args[0],pose=args[2],hoi=args[4])
    p._summaries([p.normalized(row)],dict(records=[p.normalized(record)]))


def test_host_import_closure_no_numpy_torch_or_original_execution():
    root=Path(p.__file__).resolve().parents[1]
    script=f'''import sys
sys.path[:0]=[{str(root/'infra')!r},{str(root/'src')!r}]
class Deny:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('numpy','torch','onnxruntime'):raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
import vcoco_full_interaction_inputs as p
assert len(p.join.helpers())==96
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
'''
    subprocess.run([sys.executable,'-I','-B','-c',script],check=True,capture_output=True)
    tree=ast.parse(Path(p.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute):
            assert n.func.attr not in('authenticate','run','reproduce','join_all','reconstruct_interaction','load_bank','cleanup','census')
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            targets=n.targets if isinstance(n,ast.Assign)else[n.target]
            assert not any(isinstance(t,ast.Attribute)for t in targets)
