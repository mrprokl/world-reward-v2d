"""Data-free lifecycle fixtures; no actual RoboTAP, network, model or native GPU."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import stat
import subprocess
import sys
import time
import types

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'infra'),str(REPO/'src')]
import tracker_noise_experiment as gate
import robotap_boots_public as public


def raw_pin(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def save(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o444);return raw_pin(raw)
def encode(value):return (json.dumps(value,allow_nan=False,sort_keys=True)+'\n').encode()
def example(seed=1,frames=24,points=4):
    rng=np.random.default_rng(seed)
    return dict(video=rng.integers(0,256,size=(frames,4,6,3),dtype=np.uint8),
                points=rng.uniform(.2,.8,size=(points,frames,2)).astype(np.float32),
                occluded=np.zeros((points,frames),bool))


@pytest.fixture
def lifecycle(tmp_path,monkeypatch):
    root=tmp_path/'runtime';code=root/'code';code.mkdir(parents=True)
    monkeypatch.setattr(gate,'ROOT',root)
    original=root/gate.ORIGINAL;pins=dict(pickles={})
    dictionaries=[{'z_unused':example(44),'a_fit':example(11)},
                  {'z_unused':example(45),'b_test':example(12),'a_test':example(13)}]
    for name,data in zip(gate.SPLITS,dictionaries):pins['pickles'][name]=save(original/name,pickle.dumps(data,protocol=4))
    monkeypatch.setattr(gate,'acquisition_evidence',lambda code,hash_pickles:(pins,{'original_verified':True}))
    out=root/gate.BASE/'public';out.mkdir(parents=True)
    result=gate.publish(code,out,time.monotonic()+30)
    before={'files':{'infra/tracker_noise_experiment.py':raw_pin(b'fixture source')},'markers':{},'producer_revision':'a'*40}
    def seal(stage,result):
        report=dict(stage='tracker_noise_'+stage+'_v1',status='pass',phase='complete',producer_revision='a'*40,
                    source_before=before,source_after=before,all_inputs_rehashed_after=True,
                    script_sha256=before['files']['infra/tracker_noise_experiment.py']['sha256'],
                    image_id=gate.IMAGES[stage],gpu_used=stage=='infer',challenge_inputs_used=False,
                    contact_verified=False,causal_camera_noise_verified=False,full_hoi_verified=False,adoption=False,**result)
        report_pin=save(root/gate.BASE/stage/'report.json',encode(report))
        return dict(producer_revision='a'*40,report=report_pin,files=result['outputs'])
    stagepins=dict(schema='world_reward.tracker_noise_stage_pins.v1',public=seal('public',result))
    save(code/gate.PINS,encode(stagepins))
    return dict(root=root,code=code,pins=pins,result=result,dictionaries=dictionaries,stagepins=stagepins,seal=seal)


def test_public_real_restricted_pickle_selects_all_keys_before_queries(tmp_path,monkeypatch):
    root=tmp_path/'root';monkeypatch.setattr(gate,'ROOT',root);out=root/gate.BASE/'public';out.mkdir(parents=True)
    dictionaries=[{'z_unused':example(),'same_key':example(2)}, {'second':example(4),'same_key':example(3)}]
    pins={'pickles':{}};events=[]
    for name,data in zip(gate.SPLITS,dictionaries):pins['pickles'][name]=save(root/gate.ORIGINAL/name,pickle.dumps(data,protocol=4))
    monkeypatch.setattr(gate,'acquisition_evidence',lambda *a:(pins,{}))
    read,queries=public.read_private,public.initial_queries
    def record_read(path,pin):events.append(('read',path.name));return read(path,pin)
    def record_query(value):
        assert (out/'selection.json').exists();events.append(('query',None));return queries(value)
    monkeypatch.setattr(public,'read_private',record_read);monkeypatch.setattr(public,'initial_queries',record_query)
    result=gate.publish(root,out,time.monotonic()+30)
    assert [e[0] for e in events]==['read','read','query','query','query']
    assert [(r['source_pickle'],r['video_key']) for r in result['videos']]==[
        (gate.SPLITS[0],'same_key'),(gate.SPLITS[1],'same_key'),(gate.SPLITS[1],'second')]
    assert len(result['outputs'])==4 and all(r['frames']==24 for r in result['videos'])
    assert all(stat.S_IMODE(p.stat().st_mode)==0o444 for p in (out/'inputs').iterdir())


def test_full_public_roundtrip_and_no_replacement(lifecycle):
    rows,row,_=gate.public_records(lifecycle['code'])
    assert rows==lifecycle['result']['videos'] and len(row['files'])==4
    value=example(points=33);value['occluded'][1]=True
    _,queries,indices,missing=public.initial_queries(value)
    assert 1 not in indices and 32 not in indices and missing==[1] and len(queries)==31
    value['occluded'][:32]=True
    with pytest.raises(ValueError,match='no fallback'):public.initial_queries(value)


@pytest.mark.parametrize('field,value',[('schema','old'),('future_labels_public',True),('initial_queries_external_oracles',False)])
def test_manifest_resealed_invalid_still_rejected(lifecycle,field,value):
    path=lifecycle['root']/gate.BASE/'public/inputs/manifest.json'
    data=json.loads(path.read_bytes());data[field]=value;path.chmod(0o644);save(path,encode(data))
    with pytest.raises(ValueError):gate.public_records(lifecycle['code'],pinned=False)


def add_predictions(fixture,singular=False):
    root,rows=fixture['root'],fixture['result']['videos'];outputs={}
    for i,row in enumerate(rows):
        with np.load(root/gate.BASE/'public/inputs'/row['file'],allow_pickle=False) as saved:
            queries=saved['query_points'].copy();indices=saved['point_indices'].copy()
        split=0 if i==0 else 1;private=fixture['dictionaries'][split][row['video_key']]
        t=row['frames'];rng=np.random.default_rng(60+i)
        errors=rng.normal(size=(len(indices),t,2))+rng.normal(size=(t,2))[None]
        xy=private['points'][indices].astype(np.float64)*[row['width'],row['height']]+errors
        tracks256=(xy/[row['width'],row['height']]*256).astype(np.float32)
        if singular:tracks256[:]=0
        tracks=tracks256*np.array([row['width'],row['height']],np.float32)/np.float32(256)
        pred=dict(tracks=tracks,tracks_256=tracks256,occlusion=np.full((len(indices),t),-3,np.float32),
                  expected_dist=np.full((len(indices),t),-3,np.float32),visible=np.ones((len(indices),t),bool),
                  query_points=queries,point_indices=indices,frame_index=np.arange(t,dtype=np.int64),
                  static_tracks=np.broadcast_to(queries[:,[2,1]][:,None],(len(indices),t,2)).copy(),
                  static_visible=np.ones((len(indices),t),bool))
        path=root/gate.BASE/'infer/predictions'/row['file'];path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:np.savez(stream,**pred)
        path.chmod(0o444);outputs[path.name]=gate.identity(path)
    result=dict(outputs=outputs,actual_native_inference=True,native_calls_attempted=3,native_calls_returned=3,native_calls_completed=3)
    fixture['stagepins']['infer']=fixture['seal']('infer',result)
    path=fixture['code']/gate.PINS;path.chmod(0o644);save(path,encode(fixture['stagepins']))


def test_actual_pure_fit_freezes_before_heldout_and_all_predictions_before_labels(lifecycle,monkeypatch):
    add_predictions(lifecycle);out=lifecycle['root']/gate.BASE/'evaluate';out.mkdir()
    read=public.read_private;events=[]
    def guarded(path,pin):
        events.append(path.name)
        if path.name.endswith('split4.pkl'):
            assert gate.identity(out/'model.npz') and (out/'model-metadata.json').exists()
        return read(path,pin)
    monkeypatch.setattr(public,'read_private',guarded)
    result=gate.evaluate(lifecycle['code'],out,time.monotonic()+30)
    assert events==['robotap_split3.pkl','robotap_split4.pkl'] and len(result['heldout_results'])==2
    metadata=json.loads((out/'model-metadata.json').read_bytes())
    assert metadata['fit_clip_keys']==[Path(gate.SPLITS[0]).name+':a_fit']
    assert all(r['fit_clip_keys']==metadata['fit_clip_keys'] for r in result['heldout_results'])
    assert all('matched_anchor_shift' in r and r['adoption'] is False for r in result['heldout_results'])
    assert set(result['outputs'])=={'model.npz','model-metadata.json'}
    assert result['decision'] in ('REJECT','QUALIFIED_COMPOSITE_PREDICTIVE_NOISE')


@pytest.mark.parametrize('corruption',['last_prediction','visibility','source_query'])
def test_all_native_artifacts_firewall_before_any_private_read(lifecycle,monkeypatch,corruption):
    add_predictions(lifecycle);root=lifecycle['root'];out=root/gate.BASE/'evaluate';out.mkdir()
    row=lifecycle['result']['videos'][-1];path=root/gate.BASE/'infer/predictions'/row['file']
    if corruption=='last_prediction':path.chmod(0o644);save(path,b'corrupt')
    else:
        with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
        if corruption=='visibility':arrays['visible'][:]=False
        else:arrays['query_points'][0,0]+=1
        path.chmod(0o644)
        with path.open('wb') as stream:np.savez(stream,**arrays)
        path.chmod(0o444)
        stage=lifecycle['stagepins']['infer'];stage['files'][row['file']]=gate.identity(path)
        reportpath=root/gate.BASE/'infer/report.json';report=json.loads(reportpath.read_bytes());report['outputs']=stage['files']
        reportpath.chmod(0o644);stage['report']=save(reportpath,encode(report))
        config=lifecycle['code']/gate.PINS;config.chmod(0o644);save(config,encode(lifecycle['stagepins']))
    def forbidden(*args):raise AssertionError('private read before full prediction firewall')
    monkeypatch.setattr(public,'read_private',forbidden)
    with pytest.raises(ValueError):gate.evaluate(lifecycle['code'],out,time.monotonic()+30)


def test_singular_fit_abstains_without_opening_test_labels(lifecycle,monkeypatch):
    add_predictions(lifecycle);out=lifecycle['root']/gate.BASE/'evaluate';out.mkdir()
    read=public.read_private;events=[]
    def singular(path,pin):
        events.append(path.name);assert not path.name.endswith('split4.pkl');data=read(path,pin)
        # All fit increments constant: no jitter/fallback is allowed.
        for value in data.values():value['occluded'][:]=True;value['occluded'][:,0]=False
        return data
    monkeypatch.setattr(public,'read_private',singular)
    with pytest.raises(ValueError):gate.evaluate(lifecycle['code'],out,time.monotonic()+30)
    assert events==['robotap_split3.pkl'] and not (out/'model.npz').exists()


@pytest.mark.parametrize('gains,decision',[
    ([.2,.1],'QUALIFIED_COMPOSITE_PREDICTIVE_NOISE'),([.2,0.],'REJECT'),([-.1,.5],'REJECT'),
    ([None,.2],'INCONCLUSIVE')])
def test_gate_each_clip_synchronous_only_no_pooled_or_matched_selection(gains,decision):
    rows=[dict(synchronous=dict(supported_pairs=1,complete_samples=2,mean_log_likelihood_gain=g),
               shifted={'mean_log_likelihood_gain':-100.},matched_anchor_shift={'mean_log_likelihood_delta':-100.}) for g in gains]
    assert gate.noise_decision(rows)==decision


def test_empty_support_and_nonfinite_gain_failclosed():
    rows=[dict(synchronous=dict(supported_pairs=0,complete_samples=0,mean_log_likelihood_gain=None))]*2
    assert gate.noise_decision(rows)=='INCONCLUSIVE'
    for g in (float('nan'),float('inf')):
        row=dict(synchronous=dict(supported_pairs=1,complete_samples=2,mean_log_likelihood_gain=g))
        with pytest.raises(ValueError):gate.noise_decision([row,row])


def test_actual_source_binding_modes_markers_and_exact_protocol(tmp_path,monkeypatch):
    root=tmp_path/'runtime';rev='b'*40;code=root/'jobs'/rev/gate.JOB/'code'
    save(code/'infra/tracker_noise_experiment.py',Path(gate.__file__).read_bytes())
    save(code/gate.PROTOCOL,encode(gate.EXPECTED));save(code.parent/'revision',(rev+'\n').encode())
    save(code.parent/'source-sha256',('a'*64+'\n').encode());save(code/'src/world_reward/__init__.py',b'')
    for p in (code,*code.rglob('*')):p.chmod(0o555 if p.is_dir() else 0o444)
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/'infra/tracker_noise_experiment.py'))
    proof=gate.source_binding(code,rev);assert len(proof['files'])==3
    config=code/gate.PROTOCOL;config.chmod(0o644)
    with pytest.raises(ValueError):gate.source_binding(code,rev)
    config.chmod(0o444);marker=code.parent/'revision';marker.chmod(0o644);save(marker,b'c'*40+b'\n')
    with pytest.raises(ValueError):gate.source_binding(code,rev)


def test_closed_cohort_verified_host_only(tmp_path,monkeypatch):
    root=tmp_path/'runtime';code=root/'code';monkeypatch.setattr(gate,'ROOT',root)
    pins=json.loads((REPO/gate.NATIVE_PINS).read_bytes())
    rows=[dict(source_pickle=f'eval_private/pickles/robotap/robotap_split{i}.pkl',video_key=f'closed_{i}') for i in range(3)]
    manifest=dict(videos=rows,selection=[dict(pickle_file=r['source_pickle'],video_key=r['video_key']) for r in rows])
    path=root/gate.native.PUBLIC/'inputs/manifest.json';pins['public']['files']['manifest.json']=save(path,encode(manifest))
    save(code/gate.NATIVE_PINS,encode(pins));assert gate.closed_cohort_evidence(code)['new_source_splits_disjoint'] is True
    manifest['videos'][0]['source_pickle']=gate.SPLITS[0];path.chmod(0o644)
    pins['public']['files']['manifest.json']=save(path,encode(manifest));config=code/gate.NATIVE_PINS;config.chmod(0o644);save(config,encode(pins))
    with pytest.raises(ValueError):gate.closed_cohort_evidence(code)


def test_gpu_mounts_only_public_and_native_no_pickles(lifecycle,monkeypatch,capsys):
    root,code=lifecycle['root'],lifecycle['code'];rev='a'*40
    pins=json.loads((REPO/gate.NATIVE_PINS).read_bytes());raw=b'procedural independently verified CPU receipt'
    pins['runtime'].update(raw_pin(raw));save(code/gate.NATIVE_PINS,encode(pins))
    save(code/'configs/robotap_boots_protocol.json',(REPO/'configs/robotap_boots_protocol.json').read_bytes())
    mirror=root/'results'/('tracker-noise-runtime-proof-'+rev)/'report.json';save(mirror,raw)
    monkeypatch.setattr(gate.os,'geteuid',lambda:0);monkeypatch.setattr(gate.os,'uname',lambda:types.SimpleNamespace(sysname='Linux'))
    monkeypatch.setattr(gate,'source_binding',lambda *a:{})
    monkeypatch.setattr(gate,'stage_evidence',lambda *a:{})
    monkeypatch.setattr(gate,'closed_cohort_evidence',lambda *a:{})
    gate.host_control(code,rev,'infer','mounts');mounts=capsys.readouterr().out.splitlines()
    assert all('/eval_private/' not in p and '.pkl' not in p and '/evaluate/' not in p for p in mounts)
    assert len(mounts)==17 and any(str(mirror)+'\t'+str(root/pins['runtime']['report_path'])==p for p in mounts)
    assert all('readonly' not in p for p in mounts)  # shell adds RO to every exact src/dst.


def test_bash_syntax_budget_gpu_lock_after_acquisition_and_real_source_closure():
    wrapper=REPO/'infra/run_tracker_noise_experiment.sh';subprocess.run(['bash','-n',str(wrapper)],check=True)
    text=wrapper.read_text();assert text.index('flock --nonblock 9;gpu_idle')>text.index('exec 9<"$LOCK"')
    assert '--network none' in text and '--cap-drop ALL' in text and 'TOTAL=1350' in text and 'TOTAL=900' in text
    assert '$(limit 135)' in text and '$(limit "$SECONDS_LIMIT")' in text
    import azure_job
    files={p:Path(REPO/p).read_bytes() for p in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if (REPO/p).is_file()}
    for p in ('infra/tracker_noise_experiment.py','infra/run_tracker_noise_experiment.sh',gate.PROTOCOL,'src/world_reward/tracker_noise_null.py'):files[p]=(REPO/p).read_bytes()
    selected=azure_job.runtime_bundle_paths(files,'infra/run_tracker_noise_experiment.sh')
    assert {'infra/tracker_noise_experiment.py','infra/robotap_boots_public.py','infra/robotap_boots_infer.py',
            'infra/robotap_boots_evaluate.py','src/world_reward/tracker_noise_null.py'}<=set(selected)
    assert 'infra/cari_runner.py' not in selected


def test_no_host_model_or_array_import_required():
    subprocess.run([sys.executable,'-I','-c',
        'import sys;sys.path.insert(0,'+repr(str(REPO/'infra'))+');import tracker_noise_experiment;'
        'assert "numpy" not in sys.modules and "torch" not in sys.modules'],check=True)


def test_predict_invokes_original_native_helper_once_per_full_video(lifecycle,monkeypatch):
    rows=lifecycle['result']['videos'];out=lifecycle['root']/gate.BASE/'infer';out.mkdir()
    events=[]
    torch=types.SimpleNamespace(__version__='2.5.1+cu124',
        cuda=types.SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda x:events.append(('cuda_seed',x)),synchronize=lambda:None),
        backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
                                      cudnn=types.SimpleNamespace(allow_tf32=True,benchmark=True)),
        manual_seed=lambda x:events.append(('seed',x)),
        use_deterministic_algorithms=lambda x,**kw:events.append(('deterministic',x,kw)))
    monkeypatch.setitem(sys.modules,'torch',torch);monkeypatch.setattr(np,'__version__','1.26.3')
    monkeypatch.setattr(gate,'runtime_evidence',lambda code:({},{}))
    utils,model=object(),object()
    monkeypatch.setattr(gate.native,'native_modules',lambda path:(object(),utils))
    def load(t,p,path):assert t is torch and path.name=='bootstapir_checkpoint_v2.pt';return model
    monkeypatch.setattr(gate.native,'load_model',load)
    def forward(t,u,m,data,report):
        assert t is torch and u is utils and m is model
        report['native_calls_attempted']+=1;report['native_calls_returned']+=1
        n,T=len(data['point_indices']),len(data['video']);events.append(('native',T,data['point_indices'].tolist()))
        tracks256=np.zeros((n,T,2),np.float32)
        return dict(tracks=tracks256.copy(),tracks_256=tracks256,occlusion=np.full((n,T),-3,np.float32),
            expected_dist=np.full((n,T),-3,np.float32),visible=np.ones((n,T),bool),
            query_points=data['query_points'],point_indices=data['point_indices'],frame_index=np.arange(T,dtype=np.int64),
            static_tracks=np.broadcast_to(data['query_points'][:,[2,1]][:,None],(n,T,2)).copy(),static_visible=np.ones((n,T),bool))
    monkeypatch.setattr(gate.native,'native_prediction',forward)
    report=dict(native_calls_attempted=0,native_calls_returned=0,native_calls_completed=0)
    result=gate.predict(lifecycle['code'],out,time.monotonic()+30,report)
    assert report==dict(native_calls_attempted=3,native_calls_returned=3,native_calls_completed=3)
    assert [e[1:] for e in events if e[0]=='native']==[(r['frames'],r['point_indices']) for r in rows]
    assert ('deterministic',False,dict(warn_only=False)) in events and len(result['outputs'])==3
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False


@pytest.mark.parametrize('field,value',[('gpu_used',True),('adoption',True),('script_sha256','f'*64)])
def test_forged_completed_public_scope_rejected_even_resealed(lifecycle,field,value):
    path=lifecycle['root']/gate.BASE/'public/report.json';report=json.loads(path.read_bytes());report[field]=value
    path.chmod(0o644);lifecycle['stagepins']['public']['report']=save(path,encode(report))
    config=lifecycle['code']/gate.PINS;config.chmod(0o644);save(config,encode(lifecycle['stagepins']))
    with pytest.raises(ValueError):gate.public_records(lifecycle['code'])


def test_qualified_key_checked_before_query_values_and_preserves_split_identity():
    assert gate.clip_key(gate.SPLITS[0],'same')!=gate.clip_key(gate.SPLITS[1],'same')
    for key in ('x'*257,'bad\x7f',''):
        with pytest.raises(ValueError):gate.clip_key(gate.SPLITS[0],key)


def test_mutated_fit_artifact_blocks_heldout_before_open(lifecycle,monkeypatch):
    add_predictions(lifecycle);out=lifecycle['root']/gate.BASE/'evaluate';out.mkdir()
    read=public.read_private;events=[];original_check=gate.acquisition.check_deadline
    def guarded(path,pin):
        events.append(path.name);assert not path.name.endswith('split4.pkl');return read(path,pin)
    monkeypatch.setattr(public,'read_private',guarded)
    # Mutate at the end of fit processing, before the NEXT split's pre-read gate.
    original_gc=gate.gc.collect
    def mutate():
        original_gc()
        path=out/'model-metadata.json'
        if path.exists():path.chmod(0o644);save(path,b'{"tampered":true}')
    monkeypatch.setattr(gate.gc,'collect',mutate)
    with pytest.raises(ValueError,match='before opening heldout'):gate.evaluate(lifecycle['code'],out,time.monotonic()+30)
    assert events==['robotap_split3.pkl']


@pytest.mark.parametrize('mutate',[False,True])
def test_main_seals_realistic_stage_and_posthash_failure_without_gpu(tmp_path,monkeypatch,mutate):
    root=tmp_path/'runtime';code=root/'code';code.mkdir(parents=True);out=root/gate.BASE/'public';out.mkdir(parents=True,mode=0o700)
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setenv('WR_IMAGE_ID',gate.IMAGES['public']);monkeypatch.setenv('WR_VM02_VERIFIED','1')
    monkeypatch.setattr(gate.os,'geteuid',lambda:1000);monkeypatch.setattr(gate.os,'uname',lambda:types.SimpleNamespace(sysname='Linux'))
    original_stat=Path.stat;original_iterdir=Path.iterdir
    def own_stat(path,*a,**kw):
        s=original_stat(path,*a,**kw)
        return types.SimpleNamespace(st_uid=1000,st_mode=s.st_mode) if path==out else s
    def offline(path):return iter([Path('lo')]) if path==Path('/sys/class/net') else original_iterdir(path)
    monkeypatch.setattr(Path,'stat',own_stat);monkeypatch.setattr(Path,'iterdir',offline)
    source={'files':{'infra/tracker_noise_experiment.py':raw_pin(b'fixture source')},'markers':{},'producer_revision':'a'*40}
    monkeypatch.setattr(gate,'source_binding',lambda *a:source)
    monkeypatch.setattr(gate,'stage_evidence',lambda *a:{'frozen_source':True})
    def publish(code,out,deadline):
        pin=save(out/'inputs/manifest.json',b'fixture')
        if mutate:(out/'inputs/manifest.json').chmod(0o644)
        return dict(outputs={'manifest.json':pin})
    monkeypatch.setattr(gate,'publish',publish)
    result=gate.main(['--stage','public']);report=json.loads((out/'report.json').read_bytes())
    assert result==int(mutate) and report['status']==('fail' if mutate else 'pass')
    assert report['image_id']==gate.IMAGES['public'] and report['gpu_used'] is False and report['native_calls_attempted']==0
    assert report['script_sha256']==source['files']['infra/tracker_noise_experiment.py']['sha256']
    assert report['elapsed_seconds']<=120 and stat.S_IMODE((out/'report.json').stat().st_mode)==0o444
    assert ('posthash_failure_type' in report)==mutate
