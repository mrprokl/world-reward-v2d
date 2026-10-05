"""Tiny authored arrays/mocked lifecycle only; never native models or Docker."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

import ownership_endpoint_bank as p


def seal(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw); path.chmod(0o400)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def config():
    return json.loads((Path(__file__).resolve().parents[1]/p.CONFIG).read_bytes())


def test_frozen_original_full_helper_pins_no_old_edits():
    cfg = config(); root = Path(__file__).resolve().parents[1]
    assert cfg['budget_seconds'] == 600 and cfg['native_cleanup_seconds'] == 20
    assert cfg['maximum_images'] == 96 and cfg['maximum_input_bytes'] == 96*(16 << 20)
    assert set(cfg['reused_helper_pins']) == set(p.original.HELPERS)
    for name, wanted in cfg['reused_helper_pins'].items():
        raw = (root/name).read_bytes()
        assert wanted == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert cfg['original_recipe'] == cfg['reused_helper_pins'][p.original.CONFIG]
    assert cfg['reused_helper_pins']['infra/rgb_endpoint_bank.py'] == dict(bytes=34983,
        sha256='7a4641eb7da72cd4610750b19758ee0b7afe0c7ac2865ce8e1f6ddb6715d3081')


@pytest.mark.parametrize('fault',[None,'budget','boolcount','policy','recipe','helper','origin'])
def test_configuration_exact_new_and_old_contract(tmp_path,monkeypatch,fault):
    code=tmp_path/'code';cfg=config(); source=dict(helpers=deepcopy(cfg['reused_helper_pins']))
    raw=(Path(__file__).resolve().parents[1]/p.original.CONFIG).read_bytes()
    seal(code/p.original.CONFIG,raw)
    original_policy=json.loads(raw)
    monkeypatch.setattr(p.original,'__file__',str(code/'infra/rgb_endpoint_bank.py'))
    monkeypatch.setattr(p.rgb_inputs,'__file__',str(code/'src/world_reward/rgb_bank_inputs.py'))
    if fault=='budget':cfg['budget_seconds']=601
    elif fault=='boolcount':cfg['maximum_images']=True
    elif fault=='policy':cfg['confidence']=.2
    elif fault=='recipe':cfg['original_recipe']=dict(bytes=1,sha256='a'*64)
    elif fault=='helper':source['helpers']['infra/rgb_endpoint_bank.py']=dict(bytes=1,sha256='a'*64)
    elif fault=='origin':monkeypatch.setattr(p.original,'__file__',str(tmp_path/'foreign.py'))
    source['helpers'][p.CONFIG]=seal(code/p.CONFIG,p.encode(cfg))
    if fault:
        with pytest.raises(ValueError):p.configuration(code,source)
    else:
        actual,old=p.configuration(code,source)
        assert actual==cfg and old==original_policy


def fixture_inputs():
    return dict(schema='world_reward.rgb_proposal_inputs.v1', images=[dict(image_id=f'{slot+1:032x}',
        file=f'image_{slot:06d}.jpg', bytes=1, sha256='a'*64, width=9, height=7) for slot in (0,95)])


@pytest.mark.parametrize('persons', [0, 2])
def test_original_callbacks_all_raw3600_and_slot95_no_fallback(tmp_path, monkeypatch, persons):
    import world_reward.owlv2_object_observations as owlobs
    calls = []; rgb = np.arange(7*9*3, dtype=np.uint8).reshape(7,9,3); before = rgb.tobytes()
    def detect(actual):
        calls.append('gdi')
        boxes = np.asarray([[0,0,3,4],[4,1,7,5]][:persons], dtype=np.float32).reshape(-1,4)
        ids = np.array([[101,102]], dtype=np.int64); mask = np.ones((1,2),dtype=np.int64)
        logits = np.full((1,900,256), -np.inf,dtype=np.float32); logits[:,:,:2] = 0
        return boxes, np.full(persons,.8,dtype=np.float32), ['person']*persons, np.full((1,900,4),.5,dtype=np.float32), logits,ids,mask,256
    def owl(model, actual, frame, operations):
        calls.append('owl'); boxes = np.zeros((3600,4),dtype=np.float32)
        return owlobs.Owlv2ObjectObservations(frame,(7,9),(60,60),np.arange(3600,dtype=np.int64),
            boxes,np.arange(3600,dtype=np.float32),boxes.copy())
    monkeypatch.setattr(owlobs, 'infer_owlv2_object_frame', owl); monkeypatch.setattr(p,'OUTPUT',tmp_path)
    operations = SimpleNamespace(tensor_ops=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append('sync'))))
    rows = p.observe(fixture_inputs(),detect,None,operations,time.monotonic()+10,decode=lambda row:rgb)
    assert calls == ['gdi','owl','sync']*2 and rgb.tobytes() == before
    assert [r['original_slot'] for r in rows] == [0,95]
    assert [r['acquired_ordinal'] for r in rows] == [0,1]
    assert [r['bank_index'] for r in rows] == [0,1]
    assert [r['file'] for r in rows] == ['image_000000.npz','image_000095.npz']
    for row in rows:
        assert row['person_retained_rows'] == persons and row['owl_patches'] == 3600
        with np.load(tmp_path/row['file'],allow_pickle=False) as bank:
            assert len(bank.files) == 17 and bank['owl_patch_ids'].tolist() == list(range(3600))
            assert bank['person_retained_boxes'].shape == (persons,4)
            assert bank['person_model_pred_boxes'].shape == (1,900,4)
            assert np.isneginf(bank['person_model_logits'][:,:,2:]).all()
    with pytest.raises(FileExistsError): p.observe(fixture_inputs(),detect,None,operations,time.monotonic()+10,decode=lambda row:rgb)


def dispatch_fixture(tmp_path, monkeypatch, *, native_fail=False):
    root = tmp_path/'root'; code = root/'code'; code.mkdir(parents=True)
    out = root/'results/out'; out.parent.mkdir(); (root/'jobs').mkdir()
    (root/'jobs/.world-reward-h100.lock').write_bytes(b'lock')
    cfg = config(); source = dict(producer_revision='a'*40, helpers={},markers={}); commands=[]
    for name in p.NATIVE_FILES: source['helpers'][name] = seal(code/name,b'# original source')
    for name in ('revision','source-sha256'): source['markers'][name]=seal(code.parent/name,b'marker')
    inputs=fixture_inputs(); data=tmp_path/'data';data.mkdir(); seal(data/'manifest.json',b'public')
    for row in inputs['images']: seal(data/row['file'],b'rgb')
    prior=dict(owl=dict(native_runtime={}),runtime={},grounding={},acquisition={}); recipe={'original':True}
    monkeypatch.setattr(p,'ROOT',root); monkeypatch.setattr(p,'OUTPUT',out); monkeypatch.setattr(p,'DATA',data)
    monkeypatch.setattr(p.rt,'source',lambda *a:source); monkeypatch.setattr(p,'configuration',lambda *a:(cfg,recipe))
    monkeypatch.setattr(p,'public_inputs',lambda *a,**k:inputs)
    monkeypatch.setattr(p.original,'qualifications',lambda *a,**k:prior); monkeypatch.setattr(p.original,'asset_files',lambda *a:{})
    old_pin=p.rt.pinned
    monkeypatch.setattr(p.rt,'pinned',lambda path,pin,maximum=1<<20:{} if Path(path)==code/p.owl.CONFIG else old_pin(path,pin,maximum))
    monkeypatch.setattr(p,'command',lambda args,d:(commands.append(args) or ''))
    monkeypatch.setattr(p,'cleanup',lambda *a:commands.append(['cleanup']))
    def run(cmd,**kwargs):
        commands.append(cmd);(out/'.container.cid').write_text('d'*64)
        proof_pin=p.rt.identity(out/'proof.json',2<<20); rows=[]
        for ordinal,image in enumerate(inputs['images']):
            slot=p.rgb_inputs.original_slot(image)
            rows.append(dict(image_id=image['image_id'],original_frame_index=0,bank_index=ordinal,acquired_ordinal=ordinal,
                original_slot=slot,image_size=[7,9],input_file=image['file'],input_identity={k:image[k] for k in ('bytes','sha256')},
                file=f'image_{slot:06d}.npz',identity=seal(out/f'image_{slot:06d}.npz',b'bank'),owl_patches=3600,
                person_native_queries=900,person_postprocessor_rows=0,person_retained_rows=0,person_ids=[]))
        report=dict(schema=cfg['schema'],stage='native_ownership_endpoint_banks',status='fail' if native_fail else 'pass',
            phase='complete',producer_revision='a'*40,image_id=p.IMAGE,proof_identity=proof_pin,model_loads=2,
            source_inputs_runtime_assets_rehashed_after=True,all_patches_retained=True,images=rows,runtime_identity={},
            person_query='person.',confidence=.3,text_threshold=.25,nms_iou=.7)
        for n in ('person_forward_calls','image_embed_calls','objectness_calls','box_calls'): report[n]=2
        for n in ('dwpose_calls','sam_calls','hoi_calls','tracking_calls'):report[n]=0
        for n in ('ground_truth_used','reference_metadata_read','split_metadata_read','challenge_inputs_used','actor_selection_performed','ownership_verified','quality_verified','adoption'):report[n]=False
        seal(out/'native.json',p.encode(report));return SimpleNamespace(returncode=1 if native_fail else 0)
    monkeypatch.setattr(p.subprocess,'run',run)
    return code,cfg,source,inputs,out,commands


def test_host_slot95_holes_narrow_mounts_and_sealed_report(tmp_path,monkeypatch):
    code,_,_,_,out,commands=dispatch_fixture(tmp_path,monkeypatch)
    result=p.dispatch(code,'a'*40,dict(bytes=1,sha256='e'*64),2)
    assert result['status']=='pass' and result['owned_cleanup_verified'] and result['outputs_sealed']
    assert [r['original_slot'] for r in result['native_images']]==[0,95]
    cmd=next(c for c in commands if c[:2]==['docker','run'])
    mounts=[cmd[i+1] for i,t in enumerate(cmd) if t=='--mount']
    assert not any('eval_private' in m or 'reference' in m or 'rights' in m or f'src={code},' in m for m in mounts)
    assert f'type=bind,src={p.DATA}/image_000095.jpg,dst={p.DATA}/image_000095.jpg,readonly' in mounts
    assert cmd[cmd.index('--memory')+1]=='64g' and cmd[cmd.index('--cpus')+1]=='4'
    assert '-i' in cmd and '-I' in cmd and '-B' in cmd and 'HF_HUB_OFFLINE=1' in cmd
    assert out.stat().st_mode & 0o777==0o500 and all(f.stat().st_mode & 0o777==0o400 for f in out.iterdir())


@pytest.mark.parametrize('fault',['native','source','prior','input','counter','slot','ordinal','late','proof'])
def test_changed_incomplete_or_late_run_cannot_pass(tmp_path,monkeypatch,fault):
    code,_,source,inputs,out,commands=dispatch_fixture(tmp_path,monkeypatch,native_fail=fault=='native')
    original_run=p.subprocess.run;clock=[0.]
    if fault=='late':monkeypatch.setattr(p.time,'monotonic',lambda:clock[0])
    def run(*args,**kwargs):
        result=original_run(*args,**kwargs)
        if fault=='source':monkeypatch.setattr(p.rt,'source',lambda *a:dict(source,mutation=True))
        elif fault=='prior':monkeypatch.setattr(p.original,'qualifications',lambda *a,**k:{'foreign':True})
        elif fault=='input':monkeypatch.setattr(p,'public_inputs',lambda *a,**k:dict(inputs,mutation=True))
        elif fault=='proof':(out/'proof.json').chmod(0o600);seal(out/'proof.json',b'changed')
        elif fault in ('counter','slot','ordinal'):
            path=out/'native.json';value=json.loads(path.read_bytes())
            if fault=='counter':value['box_calls']=1
            elif fault=='slot':value['images'][1]['original_slot']=1
            else:value['images'][1]['acquired_ordinal']=95
            path.chmod(0o600);seal(path,p.encode(value))
        elif fault=='late':clock[0]=601.
        return result
    monkeypatch.setattr(p.subprocess,'run',run)
    result=p.dispatch(code,'a'*40,dict(bytes=1,sha256='e'*64),2)
    assert result['status']=='fail' and result['owned_cleanup_verified'] and ['cleanup'] in commands
    assert json.loads((out/'host.json').read_bytes())['status']=='fail'


def test_prior_failure_precedes_model_container_and_output(tmp_path,monkeypatch):
    code,_,_,_,out,commands=dispatch_fixture(tmp_path,monkeypatch)
    def fail(*a,**k):raise ValueError('secret URL should never enter receipt')
    monkeypatch.setattr(p.original,'qualifications',fail)
    with pytest.raises(ValueError):p.dispatch(code,'a'*40,dict(bytes=1,sha256='e'*64),2)
    assert not out.exists() and not commands


def test_native_wrong_image_before_models_safe_error_class(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out)
    pin=seal(out/'proof.json',p.encode({}));monkeypatch.setenv('WR_IMAGE_ID','foreign')
    result=p.native(tmp_path,'a'*40,config(),pin,time.monotonic()+1)
    assert result['status']=='fail' and result['model_loads']==0 and result['phase']=='authentication'
    assert 'secret' not in p.encode(result).decode() and p.error(Exception('secret'))=='other'


@pytest.mark.parametrize('ending',['','\n'])
def test_cleanup_cannot_remove_foreign_container(tmp_path,monkeypatch,ending):
    path=tmp_path/'cid';path.write_text('a'*64+ending);calls=[]
    def command(args,deadline):
        calls.append(args)
        if args[1]=='inspect':return p.IMAGE+'|/owned|'+p.ENTRY+'|'+'b'*40
        if 'id='+'a'*64 in args:return 'a'*64
        return ''
    monkeypatch.setattr(p,'command',command);p.cleanup(path,'owned','b'*40,time.monotonic()+1)
    assert ['docker','rm','-f','a'*64] in calls
    monkeypatch.setattr(p,'command',lambda a,d:'foreign' if a[1]=='inspect' else 'a'*64)
    with pytest.raises(ValueError):p.cleanup(path,'owned','b'*40,time.monotonic()+1)


def test_partial_observation_keeps_saved_rows_and_completed_counter(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'OUTPUT',tmp_path);rows=[];count=[0]
    def bank(rgb,row,ordinal,*args):
        if ordinal==1:raise RuntimeError('bounded fail, no receipt text')
        return {},dict(image_id=row['image_id'],bank_index=ordinal)
    monkeypatch.setattr(p.original,'bank_arrays',bank)
    operations=SimpleNamespace(tensor_ops=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None)))
    def completed():count[0]+=1
    with pytest.raises(RuntimeError):
        p.observe(fixture_inputs(),None,None,operations,time.monotonic()+1,decode=lambda row:None,
            saver=lambda *a:dict(file='image_000000.npz'),records=rows,completed=completed)
    assert len(rows)==count[0]==1 and rows[0]['original_slot']==0


def test_deadline_crossed_during_final_host_fsync_rewrites_fail(tmp_path,monkeypatch):
    code,_,_,_,out,_=dispatch_fixture(tmp_path,monkeypatch);clock=[0.]
    monkeypatch.setattr(p.time,'monotonic',lambda:clock[0]);native_fsync=p.os.fsync
    def fsync(fd):
        native_fsync(fd)
        if (out/'host.json').exists():clock[0]=601.
    monkeypatch.setattr(p.os,'fsync',fsync)
    result=p.dispatch(code,'a'*40,dict(bytes=1,sha256='e'*64),2)
    assert result['status']=='fail' and result['post_error_type']=='TimeoutError'
    assert json.loads((out/'host.json').read_bytes())['status']=='fail'


def test_source_closure_under_isolated_python_and_no_old_global_mutation(tmp_path):
    root=Path(__file__).resolve().parents[1]
    for name in p.NATIVE_FILES:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes())
    result=subprocess.run([sys.executable,'-I','-B','-c','import runpy;runpy.run_path('+repr(str(tmp_path/p.NATIVE_FILES[0]))+')'],capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr.decode()
    source=(root/p.NATIVE_FILES[0]).read_text();compile(source,p.NATIVE_FILES[0],'exec')
    assert 'original.load_models(policy)' in source and 'original.bank_arrays(' in source and 'saver or original.save_bank' in source
    assert 'original.DATA =' not in source and 'original.OUTPUT =' not in source
    assert 'original.public_inputs(' not in source and 'original.decode(' not in source and 'original.main(' not in source
    assert "DOCKER_HOST='unix://'+str(ROOT/'docker.sock')" in source
    wrapper=root/'infra/run_ownership_endpoint_bank.sh'
    assert subprocess.run(['bash','-n',str(wrapper)],capture_output=True).returncode==0
    assert 'env -i' in wrapper.read_text() and '-I -B' in wrapper.read_text() and '640s' in wrapper.read_text()
