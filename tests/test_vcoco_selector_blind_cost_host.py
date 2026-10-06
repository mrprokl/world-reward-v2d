"""Tiny authored/mocked host contracts; no Azure, arrays, roles or live image."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import vcoco_selector_blind_cost as q


ROOT = Path(__file__).resolve().parents[1]
REV = 'a'*40


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write(path, value, mode=0o400):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if type(value) is bytes else q.join.encode(value)
    if path.exists(): path.chmod(0o600)
    path.write_bytes(raw); path.chmod(mode)
    return pin(raw)


def projected_owner(monkeypatch, root):
    original, original_stat, original_fstat = Path.lstat, Path.stat, os.fstat
    def project(path, s):
        if path == root or root in path.parents:
            fields = {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_nlink',
                'st_uid', 'st_gid', 'st_mtime_ns', 'st_ctime_ns')}
            uid = 1000 if path.name == 'export.json' else 0
            fields.update(st_uid=uid, st_gid=uid)
            return SimpleNamespace(**fields)
        return s
    monkeypatch.setattr(Path, 'lstat', lambda path,*a,**kw:project(path,original(path,*a,**kw)))
    monkeypatch.setattr(Path, 'stat', lambda path,*a,**kw:project(path,original_stat(path,*a,**kw)))
    def descriptor(fd):
        s = original_fstat(fd)
        for path in (root, *root.rglob('*')):
            try: known = original(path)
            except FileNotFoundError: continue
            if (s.st_dev, s.st_ino) == (known.st_dev, known.st_ino): return project(path, s)
        return s
    monkeypatch.setattr(os, 'fstat', descriptor)


def population():
    from vcoco_fit_cal_selection import NAMESPACE
    cohort = dict(records=[]); records = []; files = {}
    for i in range(48):
        native_id = 1000+i
        opaque = hashlib.sha256((NAMESPACE+f'{native_id:012d}').encode()).hexdigest()[:32]
        cohort['records'].append(dict(slot=i, split='FIT' if i<32 else 'CAL', image_id=native_id))
        base = dict(original_slot=i, acquired_ordinal=i, original_frame_index=0, image_id=opaque,
            image_size=[10,20], file=f'image_{i:06d}.npz', identity=dict(bytes=1,sha256='0'*64))
        endpoint = dict(base,person_ids=[],person_retained_rows=0)
        pose = dict(base,persons=0,person_ids=[],endpoint_bank_identity=base['identity'])
        hoi = dict(base,hand_object_pairs=0,source_person_ids=[],endpoint_bank_identity=base['identity'])
        paths = [f'/public/{key}/image_{i:06d}.npz' for key in ('endpoint','pose','hoi')]
        files.update({n:base['identity'] for n in paths})
        records.append(dict(endpoint=endpoint,pose=pose,hoi=hoi,paths=paths))
    proof = dict(schema=q.join.SCHEMA,producer_revision='b'*40,image_id=q.IMAGE,
        source={},records=records,files=files)
    return cohort,proof


def test_membership_fixed32_hash_folds_balanced_injective_opaque_and_unchanged():
    cohort,proof=population();before=deepcopy((cohort,proof))
    result=q.membership(cohort,proof['records'])
    assert len(result)==32 and [r['slot']for r in result]==list(range(32))
    assert len({r['image_id']for r in result})==32
    assert all(set(r)=={'slot','image_id','fold'}for r in result)
    assert [sum(r['fold']==fold for r in result)for fold in range(4)]==[8]*4
    order=sorted(result,key=lambda r:(hashlib.sha256((q.CV_NAMESPACE+r['image_id']).encode()).hexdigest(),r['image_id']))
    assert [r['fold']for r in order]==[i%4 for i in range(32)]
    assert (cohort,proof)==before


@pytest.mark.parametrize('fault',['missing','extra','split','order','duplicate','opaque','slot','nativezero','nativebig','nativebool'])
def test_membership_invalid_mapping_never_reassigns(fault):
    cohort,proof=population()
    if fault=='missing':cohort['records'].pop()
    elif fault=='extra':proof['records'].append(proof['records'][0])
    elif fault=='split':cohort['records'][32]['split']='FIT'
    elif fault=='order':cohort['records'][0],cohort['records'][1]=cohort['records'][1],cohort['records'][0]
    elif fault=='duplicate':cohort['records'][1]['image_id']=cohort['records'][0]['image_id'];proof['records'][1]['endpoint']['image_id']=proof['records'][0]['endpoint']['image_id']
    elif fault=='opaque':proof['records'][0]['endpoint']['image_id']='f'*32
    elif fault=='slot':proof['records'][0]['endpoint']['original_slot']=1
    else:cohort['records'][0]['image_id']={'nativezero':0,'nativebig':10**12,'nativebool':True}[fault]
    with pytest.raises((ValueError,KeyError)):q.membership(cohort,proof['records'])


@pytest.mark.parametrize('where',['cohort','endpoint'])
def test_boolean_slot_is_not_original_integer_zero(where):
    cohort,proof=population()
    if where=='cohort':cohort['records'][0]['slot']=False
    else:proof['records'][0]['endpoint']['original_slot']=False
    with pytest.raises(ValueError):q.membership(cohort,proof['records'])


def payload_fixture(tmp_path,monkeypatch):
    cohort,proof=population();code=tmp_path/'jobs'/REV/q.ENTRY/'code'
    for n in q.NATIVE_FILES:write(code/n,n.encode(),0o444)
    markers={n:write(code.parent/n,(REV+'\n').encode()if n=='revision'else('f'*64+'\n').encode(),0o444)
             for n in ('revision','source-sha256')}
    helperpins={n:q.rt.identity(code/n,2<<20,empty=True)for n in q.NATIVE_FILES}
    monkeypatch.setattr(q.join,'NUMERICAL_PIN',helperpins[q.join.NATIVE_FILES[2]])
    for n in q.NATIVE_FILES:
        key=Path(n).stem if n.startswith('infra/')else'.'.join(Path(n).with_suffix('').parts[1:])
        if key in sys.modules:monkeypatch.setattr(sys.modules[key],'__file__',str(code/n))
    before=dict(saved=dict(host_lineage=dict(current_source=dict(binding=dict(helpers=helperpins,markers=markers))),
        public_proof=proof,rows=[dict(image_id=r['endpoint']['image_id'])for r in proof['records']]),
        selection=q.membership(cohort,proof['records']),cohort_lineage={'private_sentinel':'never_native'},image={'host_sentinel':True})
    payload=q.public_payload(before,REV)
    return code,payload,before


def test_public_payload_only_opaque_complete_metadata_no_hostlineage(tmp_path,monkeypatch):
    code,payload,before=payload_fixture(tmp_path,monkeypatch)
    assert set(payload)=={'schema','producer_revision','image_id','manifest','public_proof','summary_rows','selection','source'}
    assert set(payload['source'])=={'files','markers'} and set(payload['source']['files'])==set(q.NATIVE_FILES)
    assert len(payload['public_proof']['files'])==144 and len(payload['selection'])==32
    raw=q.join.encode(payload).decode()
    assert not any(s in raw for s in ('private_sentinel','host_sentinel','cohort_lineage','native_image_id','official_split','publisher_metadata'))
    assert q.native_source(code,REV,payload)


@pytest.mark.parametrize('fault',['extra_payload','manifest','producer','image','files','hidden_file','pin','marker','origin','empty_missing'])
def test_thin_native_source_schema_bytes_origins_no_hiddenhelpers(tmp_path,monkeypatch,fault):
    code,payload,before=payload_fixture(tmp_path,monkeypatch)
    if fault=='extra_payload':payload['cohort']={'secret':True}
    elif fault=='manifest':payload['manifest']['images']=31
    elif fault=='producer':payload['producer_revision']='b'*40
    elif fault=='image':payload['image_id']='sha256:'+'0'*64
    elif fault=='files':payload['source']['files']['infra/foreign.py']=dict(bytes=1,sha256='0'*64)
    elif fault=='hidden_file':write(code/'infra/foreign.py',b'foreign',0o444)
    elif fault=='pin':payload['source']['files'][q.NATIVE_FILES[0]]['sha256']='0'*64
    elif fault=='marker':write(code.parent/'revision',('b'*40+'\n').encode(),0o444)
    elif fault=='origin':monkeypatch.setattr(q,'__file__',str(tmp_path/'foreign.py'))
    else:(code/q.NATIVE_FILES[-1]).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):q.native_source(code,REV,payload)


@pytest.mark.parametrize('fault',['source_extra','selection_missing','selection_slotbool','selection_foldbool','selection_wrongfold','selection_unknownid','selection_duplicate','selection_extra'])
def test_native_rejects_nested_hidden_source_or_reassigned_cv(tmp_path,monkeypatch,fault):
    code,payload,before=payload_fixture(tmp_path,monkeypatch)
    if fault=='source_extra':payload['source']['host_lineage']={}
    elif fault=='selection_missing':payload['selection'].pop()
    elif fault=='selection_slotbool':payload['selection'][0]['slot']=False
    elif fault=='selection_foldbool':payload['selection'][0]['fold']=bool(payload['selection'][0]['fold'])
    elif fault=='selection_wrongfold':payload['selection'][0]['fold']=(payload['selection'][0]['fold']+1)%4
    elif fault=='selection_unknownid':payload['selection'][0]['image_id']='f'*32
    elif fault=='selection_duplicate':payload['selection'][1]['image_id']=payload['selection'][0]['image_id']
    else:payload['selection'][0]['private_label']=1
    with pytest.raises(ValueError):q.native_source(code,REV,payload)


def graph_fixture(tmp_path,monkeypatch):
    import research_image_identity as identity
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(q,'ROOT',root);projected_owner(monkeypatch,root)
    layers=['sha256:'+f'{i:064x}'for i in range(44)]
    graph=dict(source_OCI_index_id=identity.INDEX_ID,platform_manifest_id='sha256:de690d04d890f7eaf301babee01e6224bf874d41d06615cba9efd20f8596aea2',
        image_id=identity.CONFIG_ID,rootfs_layers=44,rootfs_diff_ids=layers,sealed_image_tar_sha256=identity.EXPORT_SHA,
        image_content_changed=False,image_rebuilt=False)
    export=dict(stage='world_reward_task_only_export',status='pass',producer_revision='671d10cacb146c1ef3c1edd22af155eb994a4122',
        image_id=identity.INDEX_ID,image_tag=identity.TAG,
        artifacts=[dict(file='assets.tar',bytes=1,sha256='0'*64),dict(file='image.tar',bytes=14565534720,sha256=identity.EXPORT_SHA)],
        docker_state_copied=False,user_disk_copied=False,secrets_transferred=False,challenge_inputs_transferred=False)
    folder=root/'transfer/vm02-v1'
    ep=write(folder/'export.json',export,0o600)
    imported=dict(graph,stage='world_reward_research_import',status='pass',export_receipt_sha256=ep['sha256'],
        producer_revision='c'*40,task_assets_extracted_as_UID=1000,private_annotations_mounted_in_GPU_smoke=False,
        CUDA_execution_verified=True,model_inference_verified=False,source_unit_restarted=False,GPU_smoke_sha256='f'*64)
    values={'export.json':export,'import.json':imported,'image-identity-v2.json':graph}
    pins={n:write(folder/n,v,0o600 if n=='export.json'else 0o644)for n,v in values.items()}
    monkeypatch.setattr(q,'IMAGE_PINS',pins)
    live=dict(Id=q.IMAGE,Architecture='amd64',Os='linux',RootFS=dict(Type='layers',Layers=layers.copy()))
    calls=[]
    monkeypatch.setattr(q.join,'image',lambda deadline:(calls.append('live'),deepcopy(live))[1])
    return SimpleNamespace(folder=folder,values=values,pins=pins,live=live,calls=calls)


def test_legacy_graph_writable_exactpins_prepost_no_archive_or_gpuclaim(tmp_path,monkeypatch):
    f=graph_fixture(tmp_path,monkeypatch);before={n:q.join.snapshot(f.folder/n)for n in f.pins}
    result=q.image_graph(time.monotonic()+30)
    assert result['legacy_receipts_writable']is True and result['sealed_archive_rehashed']is False
    assert result['historical_GPU_qualification_inherited']is False and result['pins']==f.pins
    assert len(result['live']['RootFS']['Layers'])==44 and f.calls==['live']
    assert {n:q.join.snapshot(f.folder/n)for n in f.pins}==before


@pytest.mark.parametrize('fault',['pin','mode','symlink','hardlink','index','config','exportsha','rootfscount','rootfsorder','platform','rebuild','importstatus','live','post'])
def test_legacy_graph_alias_graph_mode_drift_rejected(tmp_path,monkeypatch,fault):
    f=graph_fixture(tmp_path,monkeypatch)
    if fault=='pin':f.pins['import.json']['sha256']='0'*64
    elif fault=='mode':(f.folder/'export.json').chmod(0o400)
    elif fault=='symlink':
        p=f.folder/'import.json';p.rename(f.folder/'original');p.symlink_to(f.folder/'original')
    elif fault=='hardlink':os.link(f.folder/'import.json',tmp_path/'alias')
    elif fault=='live':f.live['RootFS']['Layers'].reverse()
    elif fault=='post':
        def mutate(deadline):write(f.folder/'import.json',b'changed',0o644);return deepcopy(f.live)
        monkeypatch.setattr(q.join,'image',mutate)
    else:
        imp=f.values['import.json'];graph=f.values['image-identity-v2.json']
        if fault=='index':imp['source_OCI_index_id']=graph['source_OCI_index_id']='sha256:'+'0'*64
        elif fault=='config':imp['image_id']=graph['image_id']='sha256:'+'0'*64
        elif fault=='exportsha':imp['sealed_image_tar_sha256']=graph['sealed_image_tar_sha256']='0'*64
        elif fault=='rootfscount':imp['rootfs_layers']=graph['rootfs_layers']=43
        elif fault=='rootfsorder':imp['rootfs_diff_ids']=list(reversed(graph['rootfs_diff_ids']))
        elif fault=='platform':imp['platform_manifest_id']=graph['platform_manifest_id']='sha256:'+'0'*64
        elif fault=='rebuild':imp['image_rebuilt']=True
        else:imp['status']='fail'
        for n in ('import.json','image-identity-v2.json'):f.pins[n]=write(f.folder/n,f.values[n],0o644)
    with pytest.raises((ValueError,FileNotFoundError)):q.image_graph(time.monotonic()+30)


def test_host_import_is_stdlib_and_numeric_imports_only_native_local():
    script=f'''import sys
sys.path[:0]=[{str(ROOT/'infra')!r},{str(ROOT/'src')!r}]
class Deny:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('numpy','torch','onnxruntime','scipy'):raise AssertionError(name)
sys.meta_path.insert(0,Deny())
import vcoco_selector_blind_cost
assert not any(n in sys.modules for n in ('numpy','torch','onnxruntime','scipy'))
'''
    subprocess.run([sys.executable,'-I','-B','-c',script],check=True,capture_output=True)
    tree=ast.parse((ROOT/'infra/vcoco_selector_blind_cost.py').read_text())
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names]if isinstance(node,ast.Import)else[node.module]
            assert not any(n.split('.')[0]in('numpy','torch','scipy')for n in names)


def test_wrapper_zeroargs_lease_deadline_and_no_ambient_secret():
    raw=(ROOT/'infra/run_vcoco_selector_blind_cost.sh').read_text()
    assert '[[ $# == 0 ]]'in raw and 'world-reward-ncc-h100-02'in raw
    assert '3615s env -i'in raw and 'umask 077'in raw and 'set +x'in raw
    assert 'flock'in raw and '9' in raw and '.world-reward-h100.lock'in raw
    subprocess.run(['bash','-n',str(ROOT/'infra/run_vcoco_selector_blind_cost.sh')],check=True,capture_output=True)


def test_cleanup_exactcid_only_and_recheck_after_remove(tmp_path,monkeypatch):
    projected_owner(monkeypatch,tmp_path)
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(q,'OUTPUT',out)
    raw=('a'*64+'\n').encode();write(out/'.container.cid',raw)
    name='owned';calls=[];alive=[True]
    def command(argv,deadline):
        calls.append(argv)
        if argv[:2]==['docker','ps']:
            return 'a'*64 if alive[0]else''
        if argv[:2]==['docker','inspect']:return q.IMAGE+'|/'+name+'|'+q.ENTRY+'|'+REV
        if argv[:2]==['docker','rm']:alive[0]=False;return''
        raise AssertionError(argv)
    monkeypatch.setattr(q.join,'command',command)
    q.cleanup(name,REV,time.monotonic()+30)
    assert len([a for a in calls if a[:2]==['docker','rm']])==1
    assert sum(any(x.startswith('id=')for x in a)for a in calls)>=2
    assert stat.S_IMODE((out/'.container.cid').stat().st_mode)==0o400


def test_cleanup_foreign_renamed_container_never_deleted(tmp_path,monkeypatch):
    projected_owner(monkeypatch,tmp_path)
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(q,'OUTPUT',out);write(out/'.container.cid',('a'*64+'\n').encode())
    calls=[]
    def command(argv,deadline):
        calls.append(argv)
        return 'a'*64 if argv[:2]==['docker','ps']else q.IMAGE+'|/foreign|'+q.ENTRY+'|'+REV
    monkeypatch.setattr(q.join,'command',command)
    with pytest.raises(ValueError):q.cleanup('owned',REV,time.monotonic()+30)
    assert not any(a[:2]==['docker','rm']for a in calls)


@pytest.mark.parametrize('fault',['symlink','hardlink','foreign_uid','replaced_cid'])
def test_cleanup_cid_alias_or_owner_drift_never_chmod_or_delete(tmp_path,monkeypatch,fault):
    projected_owner(monkeypatch,tmp_path)
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(q,'OUTPUT',out)
    cid=out/'.container.cid';raw=('a'*64+'\n').encode();write(cid,raw,0o600)
    if fault=='symlink':
        target=tmp_path/'foreign';write(target,raw,0o644);cid.unlink();cid.symlink_to(target)
    elif fault=='hardlink':os.link(cid,tmp_path/'alias')
    elif fault=='foreign_uid':
        previous=Path.lstat
        def foreign(path,*a,**kw):
            s=previous(path,*a,**kw)
            return SimpleNamespace(**{**vars(s),'st_uid':999,'st_gid':999})if path==cid else s
        monkeypatch.setattr(Path,'lstat',foreign)
    calls=[];chmods=[]
    def command(argv,deadline):
        calls.append(argv)
        if fault=='replaced_cid'and len(calls)==1:
            cid.rename(out/'retained-original');write(cid,raw,0o600)
        return''
    monkeypatch.setattr(q.join,'command',command)
    monkeypatch.setattr(q.os,'fchmod',lambda *a:chmods.append(a))
    with pytest.raises(ValueError):q.cleanup('owned',REV,time.monotonic()+30)
    assert not chmods and not any(a[:2]==['docker','rm']for a in calls)
    if fault=='symlink':assert stat.S_IMODE(target.stat().st_mode)==0o644
    else:assert stat.S_IMODE(cid.stat().st_mode)==0o600


@pytest.mark.parametrize('fault',[None,'native','empty_native','post','cleanup','latepublication'])
def test_mocked_host_argv_fullpublic_mounts_finally_and_sealed_status(tmp_path,monkeypatch,fault):
    import sealed_callback_publication as publication
    code,payload,before=payload_fixture(tmp_path,monkeypatch)
    out=tmp_path/'fresh-output';monkeypatch.setattr(q,'OUTPUT',out);projected_owner(monkeypatch,tmp_path)
    for name in list(payload['public_proof']['files']):
        old=payload['public_proof']['files'].pop(name);new=tmp_path/'raw'/Path(name).relative_to('/public')
        payload['public_proof']['files'][str(new)]=write(new,b'raw',0o400)
    monkeypatch.setattr(q,'public_payload',lambda a,r:deepcopy(payload))
    counts={'auth':0,'native':0,'cleanup':0};argvs=[]
    def authenticate(*args):
        counts['auth']+=1
        if fault=='post'and counts['auth']==2:return dict(drift=True)
        return deepcopy(before)
    monkeypatch.setattr(q,'authenticate',authenticate)
    monkeypatch.setattr(q.join,'command',lambda *a:'')
    def cleanup(*args):
        counts['cleanup']+=1
        if fault=='cleanup':raise ValueError('SECRET_SENTINEL')
    monkeypatch.setattr(q,'cleanup',cleanup)
    monkeypatch.setattr(q,'validate_native',lambda *a:None)
    def run(argv,**kwargs):
        argvs.append((argv,kwargs));counts['native']+=1
        write(out/'.container.cid',('a'*64+'\n').encode(),0o400)
        write(out/'native.json',b'' if fault=='empty_native' else dict(status='fail'if fault=='native'else'pass'),0o400)
        return SimpleNamespace(returncode=1 if fault=='native'else 0)
    monkeypatch.setattr(q.subprocess,'run',run)
    original=publication.publish
    def publish(*args,**kwargs):
        if fault=='latepublication':
            previous=kwargs['check'];calls=[0]
            def late(deadline):
                calls[0]+=1
                if calls[0]==1:raise TimeoutError('SECRET_SENTINEL')
                return previous(deadline)
            kwargs['check']=late
        return original(*args,**kwargs)
    monkeypatch.setattr(publication,'publish',publish)
    oldhandlers={s:q.signal.getsignal(s)for s in(q.signal.SIGALRM,q.signal.SIGTERM,q.signal.SIGINT)}
    report=q.host(code,REV)
    assert counts==dict(auth=2,native=1,cleanup=1)
    assert {s:q.signal.getsignal(s)for s in oldhandlers}==oldhandlers
    assert q.signal.getitimer(q.signal.ITIMER_REAL)[0]==0
    assert report['status']==('pass'if fault is None else'fail')
    assert report['decision']==(q.DECISION if fault is None else'CLOSED_BLIND_COST')
    assert report['invalid_empty_output_leaves']==(['native.json']if fault=='empty_native'else[])
    saved=q.rt.strict((out/'report.json').read_bytes());assert q.join.encode(saved)==q.join.encode(report)
    assert 'SECRET_SENTINEL'not in (out/'report.json').read_text()
    assert stat.S_IMODE(out.stat().st_mode)==0o500
    assert {p.name for p in out.iterdir()}=={'report.json','proof.json','native.json','.container.cid'}
    assert all(stat.S_IMODE(p.stat().st_mode)==0o400 for p in out.iterdir())
    argv,kwargs=argvs[0]
    assert argv[:2]==['docker','run']and argv[argv.index('--memory')+1]=='64g'
    assert argv[argv.index('--memory-swap')+1]=='64g'and argv[argv.index('--cpus')+1]=='4'
    assert argv[argv.index('--network')+1]=='none'and '--read-only'in argv
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8'in argv
    mounts=[argv[i+1]for i,v in enumerate(argv)if v=='--mount']
    expected={str(code/n)for n in q.NATIVE_FILES}|{str(code.parent/n)for n in('revision','source-sha256')}|set(payload['public_proof']['files'])
    assert set(mounts)=={f'type=bind,src={p},dst={p},readonly'for p in expected}|{f'type=bind,src={out},dst={out}'}
    assert not any(n in '\n'.join(mounts)for n in('eval_private','cohort.json','.jpg','role_'))
    assert 0<kwargs['timeout']<=q.BUDGET-30 and '--deadline'in argv


def test_main_host_only_zeroargs_and_matching_fd9(tmp_path,monkeypatch,capsys):
    root=tmp_path/'root';code=root/'jobs'/REV/q.ENTRY/'code';code.mkdir(parents=True)
    lock=root/'jobs/.world-reward-h100.lock';lock.write_bytes(b'')
    monkeypatch.setattr(q,'ROOT',root);monkeypatch.setattr(q,'__file__',str(code/'infra/vcoco_selector_blind_cost.py'))
    monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',REV)
    monkeypatch.setenv('DOCKER_HOST','unix://'+str(root/'docker.sock'))
    monkeypatch.setattr(q.os,'geteuid',lambda:0);monkeypatch.setattr(q.os,'uname',lambda:SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    realfstat=q.os.fstat;monkeypatch.setattr(q.os,'fstat',lambda fd:lock.stat()if fd==9 else realfstat(fd))
    calls=[];monkeypatch.setattr(q,'host',lambda *a:(calls.append(a),dict(status='pass',decision=q.DECISION))[1])
    monkeypatch.setattr(sys,'argv',['driver']);q.main();assert len(calls)==1
    assert json.loads(capsys.readouterr().out)['status']=='pass'
    monkeypatch.setattr(q.os,'fstat',lambda fd:SimpleNamespace(st_dev=0,st_ino=0))
    with pytest.raises(ValueError):q.main()
    monkeypatch.setattr(sys,'argv',['driver','--revision',REV])
    with pytest.raises(ValueError):q.main()
    monkeypatch.setattr(sys,'argv',['driver','--unknown'])
    with pytest.raises(SystemExit):q.main()
