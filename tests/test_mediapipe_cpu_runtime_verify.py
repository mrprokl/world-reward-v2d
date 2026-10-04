"""Opaque byte fixtures and native API spies, never real Docker/model execution."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

REPO=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('mediapipe_runtime_test',REPO/'infra/mediapipe_cpu_runtime_verify.py')
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


def seal(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o444)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def host(tmp_path,monkeypatch):
    root=tmp_path/'root';revision='a'*40;code=root/'jobs'/revision/gate.ENTRY/'code'
    for name in gate.HELPERS:
        original=REPO/name;seal(code/name,original.read_bytes()if original.exists()else b'fixture future actual pin only')
    seal(code.parent/'revision',(revision+'\n').encode());seal(code.parent/'source-sha256',b'b'*64+b'\n')
    for p in reversed(list(code.rglob('*'))):
        if p.is_dir():p.chmod(0o555)
    code.chmod(0o555);(root/'results').mkdir()
    wheel=root/'fixtures'/'opaque-1.0-py3-none-any.whl';wheel_pin=seal(wheel,b'opaque wheel; no native code')
    task=root/'fixtures'/'hand_landmarker.task';task_pin=seal(task,b'opaque task; no model nodes')
    frozen={wheel:wheel_pin,task:task_pin};proof=dict(manifest={'packages':[dict(name='opaque',version='1.0')]},wheels={wheel:dict(filename=wheel.name,**wheel_pin)},task=task,
        task_pin=task_pin,frozen=frozen,prior={'proof':'procedural no actual acquisition'},verified_downloads={},dependency_pins={},notices={},dependency_source={})
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    monkeypatch.setattr(gate.sys,'platform','linux');monkeypatch.setattr(gate.os,'geteuid',lambda:0)
    monkeypatch.setattr(gate.os,'uname',lambda:types.SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    monkeypatch.setattr(gate,'prerequisites',lambda *a:proof)
    return dict(root=root,code=code,revision=revision,proof=proof)


def tools(host,monkeypatch,fault=''):
    calls=[];revision=host['revision'];source=gate.source(host['root'],host['code'],revision,gate.ENTRY,gate.HELPERS)
    layers=['sha256:'+'1'*64];base=dict(Id=gate.BASE,Architecture='amd64',Os='linux',RootFS=dict(Type='layers',Layers=layers),Labels={})
    child=dict(Id='sha256:'+'2'*64,Architecture='amd64',Os='linux',RootFS=dict(Type='layers',Layers=layers+['sha256:'+'3'*64]),
        Labels={'world_reward.mediapipe_cpu.revision':revision,'world_reward.mediapipe_cpu.source':source['closure_sha256'],
                'world_reward.mediapipe_cpu.manifest':gate.MANIFEST_PIN['sha256']})
    state={'built':False,'live':False}
    def fake(args,**kwargs):
        calls.append(args)
        if args[:3]==['docker','image','inspect']:
            if args[3]==gate.TARGET and '--format'not in args:
                return subprocess.CompletedProcess(args,0 if fault=='occupied_image'else 1,b'',b'No such image')
            value=base if args[3]==gate.BASE else child
            if fault=='image_layer'and value is child:value=copy.deepcopy(child);value['RootFS']['Layers'][0]='sha256:'+'4'*64
            if fault=='image_label'and value is child:value=copy.deepcopy(child);value['Labels']['world_reward.mediapipe_cpu.source']='0'*64
            raw=' '.join(json.dumps(value[k])for k in('Id','Architecture','Os','RootFS','Labels')).encode()
            return subprocess.CompletedProcess(args,0,raw,b'')
        if args[:2]==['docker','ps']:
            return subprocess.CompletedProcess(args,0,('d'*64+'\n').encode()if state['live']or fault=='occupied_container'else b'',b'')
        if args[:2]==['docker','build']:
            state['built']=True
            assert kwargs['timeout']<=600 and args[args.index('--network')+1]=='none'
            kwargs['stdout'].write(b'ordinary opaque build log\n')
            return subprocess.CompletedProcess(args,9 if fault=='build'else 0,b'',b'')
        if args[:2]==['docker','run']:
            cid=Path(args[args.index('--cidfile')+1]);cid.write_text('d'*64+'\n');state['live']=fault!='auto_removed'
            smoke=dict(status='pass',native_graph_loaded=True,native_graph_closed=True,detect_calls=0,gpu_used=False,dataset_read=False,
                       versions={'opaque':'1.0'},task_identity=host['proof']['task_pin'])
            if fault=='smoke_receipt':smoke['native_graph_loaded']=False
            kwargs['stdout'].write(json.dumps(smoke).encode()+b'\n')
            if fault=='source_post':
                path=host['code']/'infra/Dockerfile.mediapipe_cpu';path.chmod(0o644);path.write_bytes(b'changed');path.chmod(0o444)
            if fault=='timeout':raise subprocess.TimeoutExpired(args,120)
            if fault=='interrupt':gate.signal.getsignal(gate.signal.SIGTERM)(None,None)
            return subprocess.CompletedProcess(args,7 if fault=='smoke'else 0,b'',b'')
        if args[:2]==['docker','inspect']:
            name='world-reward-mediapipe-verify-'+revision[:12]
            raw='foreign'if fault=='foreign'else child['Id']+'|/'+name+'|'+revision
            return subprocess.CompletedProcess(args,0,raw.encode(),b'')
        if args[:3]==['docker','rm','-f']:
            if fault!='cleanup':state['live']=False
            return subprocess.CompletedProcess(args,9 if fault=='cleanup'else 0,b'',b'')
        raise AssertionError('Unexpected control invocation')
    monkeypatch.setattr(gate.subprocess,'run',fake);return calls


def test_new_child_offline_venv_context_cpu_graph_and_original_byte_postseal(host,monkeypatch):
    calls=tools(host,monkeypatch);report=gate.run(host['root'],host['code'],host['revision'])
    assert report['status']=='pass'and report['source_rehashed_after']and report['owned_cleanup_verified']
    assert report['detect_calls']==0 and report['gpu_used']is False and report['quality_verified']is False
    assert report['license_eligibility_verified']is False and report['task_constituent_license_verified']is False
    build=next(c for c in calls if c[:2]==['docker','build']);assert build[build.index('--tag')+1]==gate.TARGET
    assert build[build.index('--cpu-period')+1]=='100000'and build[build.index('--cpu-quota')+1]=='400000'
    command=next(c for c in calls if c[:2]==['docker','run'])
    assert '--gpus'not in command and command[command.index('--network')+1]=='none'and command[command.index('--cap-drop')+1]=='ALL'
    assert command[command.index('--memory')+1]=='8g'and command[command.index('--cpus')+1]=='4'
    assert 'JAX_PLATFORMS=cpu'in command and 'CUDA_VISIBLE_DEVICES=-1'in command and 'OPENBLAS_NUM_THREADS=1'in command
    mounts=[command[i+1]for i,v in enumerate(command)if v=='--mount'];assert len(mounts)==2 and all(v.endswith(',readonly')for v in mounts)
    assert all('data'in v or str(host['code'])in v or str(host['proof']['task'])in v for v in mounts)
    out=host['root']/'results'/('mediapipe-cpu-runtime-verify-'+host['revision'])
    assert not(out/'build-context').exists()and(out/'report.json').stat().st_mode&0o777==0o400
    assert host['proof']['task'].read_bytes()==b'opaque task; no model nodes'


@pytest.mark.parametrize('fault',['occupied_image','occupied_container','build','image_layer','image_label','smoke',
                                 'smoke_receipt','timeout','interrupt','source_post','foreign','cleanup'])
def test_failures_are_sealed_without_retag_retry_or_foreign_cleanup(host,monkeypatch,fault):
    calls=tools(host,monkeypatch,fault)
    with pytest.raises(ValueError):gate.run(host['root'],host['code'],host['revision'])
    out=host['root']/'results'/('mediapipe-cpu-runtime-verify-'+host['revision'])
    receipt=json.loads((out/'report.json').read_bytes());assert receipt['status']=='fail'and not receipt['quality_verified']
    assert len([c for c in calls if c[:2]==['docker','build']])<=1
    assert len([c for c in calls if c[:2]==['docker','run']])<=1
    assert not(out/'build-context').exists()
    if fault=='foreign':assert not any(c[:3]==['docker','rm','-f']for c in calls)


@pytest.mark.parametrize('fault',['source','marker','alias','existing','future_pins'])
def test_missing_proofs_or_occupied_original_namespace_stop_before_build(host,monkeypatch,fault):
    if fault in('source','marker'):
        p=host['code']/'infra/mediapipe_hands_acquire.py'if fault=='source'else host['code'].parent/'revision';p.chmod(0o644)
    elif fault=='alias':
        original=host['code']/'infra/Dockerfile.mediapipe_cpu';original.parent.chmod(0o755);original.unlink();original.symlink_to('missing')
    elif fault=='existing':
        p=host['root']/'results'/('mediapipe-cpu-runtime-verify-'+host['revision']);p.mkdir();(p/'user.txt').write_text('preserve')
    else:
        monkeypatch.setattr(gate,'prerequisites',lambda *a:(_ for _ in()).throw(ValueError('Actual acquisition receipt pins missing')))
    calls=tools(host,monkeypatch)if fault in('existing','future_pins')else[]
    with pytest.raises((ValueError,FileNotFoundError)):gate.run(host['root'],host['code'],host['revision'])
    assert calls==[]
    if fault=='existing':assert(p/'user.txt').read_text()=='preserve'


def test_context_copy_is_independent_readonly_not_hardlinked_and_refuses_corrupt_wheel(host,tmp_path):
    p=tmp_path/'context';p.mkdir();gate.make_context(p,host['code'],host['proof'])
    original=next(iter(host['proof']['wheels']));copied=p/'wheels'/original.name
    assert gate.identity(original)==gate.identity(copied)and original.stat().st_ino!=copied.stat().st_ino
    assert original.stat().st_nlink==copied.stat().st_nlink==1
    other=tmp_path/'failed';other.mkdir();original.chmod(0o644);original.write_bytes(b'bad');original.chmod(0o444)
    with pytest.raises(ValueError):gate.make_context(other,host['code'],host['proof'])


def test_fixed_dockerfile_base_venv_and_no_online_resolution_or_other_packages():
    value=(REPO/'infra/Dockerfile.mediapipe_cpu').read_text()
    assert value.splitlines()[0]=='FROM '+gate.BASE
    assert 'python -m venv /opt/world-reward-mediapipe'in value and '--system-site-packages'not in value
    assert '"--no-index","--no-deps"'in value and '"check"'in value
    assert not any(word in value for word in('apt-get','pip install','--upgrade','curl','ADD http','headless','cuda'))


def test_actual_runtime_closure_has_dockerfile_acquisition_and_no_predictor_dependency(monkeypatch):
    subprocess.run(['rtk','proxy','bash','-n',str(REPO/'infra/run_mediapipe_cpu_runtime_verify.sh')],check=True)
    monkeypatch.syspath_prepend(str(REPO/'infra'));import azure_job
    files={str(p.relative_to(REPO)):p.read_bytes()for folder in('infra','src','configs')for p in(REPO/folder).rglob('*')
           if p.is_file()and '__pycache__'not in p.parts}
    files['pyproject.toml']=(REPO/'pyproject.toml').read_bytes()
    selected=set(azure_job.runtime_bundle_paths(files,'infra/run_mediapipe_cpu_runtime_verify.sh'))
    assert {'infra/Dockerfile.mediapipe_cpu','infra/mediapipe_cpu_runtime_verify.py',
            'infra/mediapipe_cpu_dependencies_acquire.py','infra/mediapipe_hands_acquire.py'}<=selected
    assert 'infra/dexycb_identity_infer.py'not in selected and 'infra/mediapipe_hand_observe.py'not in selected


def test_native_smoke_creates_and_closes_exact_cpu_options_without_detect_or_images(tmp_path,monkeypatch):
    venv=tmp_path/'venv';venv.mkdir();(venv/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    manifest=tmp_path/'manifest.json';manifest_pin=seal(manifest,b'{"packages":[{"name":"mediapipe","version":"0.10.21"}]}')
    task=tmp_path/'hand_landmarker.task';task_pin=seal(task,b'opaque no model bytes')
    monkeypatch.setattr(gate,'VENV',venv);monkeypatch.setattr(gate,'MANIFEST_PIN',manifest_pin)
    monkeypatch.setattr(gate.sys,'platform','linux');monkeypatch.setattr(gate.sys,'prefix',str(venv))
    monkeypatch.setattr(gate.sys,'base_prefix','/different_base');monkeypatch.setattr(gate.os,'geteuid',lambda:0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1');monkeypatch.setenv('JAX_PLATFORMS','cpu')
    original_iterdir=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')])if p==Path('/sys/class/net')else original_iterdir(p))
    from importlib import metadata
    monkeypatch.setattr(metadata,'version',lambda name:'0.10.21')
    monkeypatch.setattr(gate.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a[0],0,b''))
    observed=[]
    class Options:
        def __init__(self,**kw):self.__dict__.update(kw)
    class Base(Options):Delegate=types.SimpleNamespace(CPU='CPU')
    class Graph:
        def __enter__(self):observed.append('opened');return self
        def __exit__(self,*_):observed.append('closed')
        def detect(self,*_):raise AssertionError('Dataset/image inference forbidden in runtime gate')
    class Landmarker:
        @staticmethod
        def create_from_options(options):observed.append(options);return Graph()
    mp=types.ModuleType('mediapipe');mp.__file__=str(venv/'lib/mediapipe/__init__.py')
    np=types.ModuleType('numpy');np.__file__=str(venv/'lib/numpy/__init__.py');np.__version__='1.26.4'
    cv=types.ModuleType('cv2');cv.__file__=str(venv/'lib/cv2/__init__.py');cv.__version__='4.11.0'
    tasks=types.ModuleType('mediapipe.tasks');python=types.ModuleType('mediapipe.tasks.python');python.BaseOptions=Base
    vision=types.ModuleType('mediapipe.tasks.python.vision');vision.HandLandmarker=Landmarker;vision.HandLandmarkerOptions=Options
    vision.RunningMode=types.SimpleNamespace(IMAGE='IMAGE')
    for name,module in [('mediapipe',mp),('numpy',np),('cv2',cv),('mediapipe.tasks',tasks),
                        ('mediapipe.tasks.python',python),('mediapipe.tasks.python.vision',vision)]:
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.delitem(sys.modules,'torch',raising=False)
    result=gate.smoke(manifest,task,str(task_pin['bytes']),task_pin['sha256'])
    options=observed[0];assert options.running_mode=='IMAGE'and options.num_hands==4
    assert options.base_options.delegate=='CPU'and options.base_options.model_asset_path==str(task)
    assert options.min_hand_detection_confidence==options.min_hand_presence_confidence==options.min_tracking_confidence==.5
    assert observed[1:]==['opened','closed']and result['detect_calls']==0
    assert result['gpu_used']is False and result['dataset_read']is False


@pytest.mark.parametrize('fault',['missing','tamper','schema','source_sha'])
def test_independent_acquisition_pin_files_block_without_models(tmp_path,fault):
    code=tmp_path/'code';value=dict(schema='world_reward.mediapipe_cpu_dependencies_acquire_pins.v1',
        producer_revision='a'*40,report=dict(bytes=1,sha256='b'*64),helper=dict(bytes=1,sha256='c'*64))
    if fault=='schema':value['schema']='unknown'
    if fault=='source_sha':value['helper']['sha256']='not independently pinned'
    path=code/gate.DEP_PINS
    if fault!='missing':seal(path,(json.dumps(value)+'\n').encode())
    if fault=='tamper':path.chmod(0o644)
    with pytest.raises((ValueError,FileNotFoundError)):
        gate.acquire_pins(code,gate.DEP_PINS,'world_reward.mediapipe_cpu_dependencies_acquire_pins.v1')


@pytest.mark.parametrize('fault',['','receipt','marker','wheel','task','helper','original_receipt','original_source','download_pin'])
def test_actual_acquisition_functions_supply_runtime_proof_and_reject_mutations(tmp_path,monkeypatch,fault):
    # Reuse the original acquisition's procedural byte cohort, not a fabricated PASS ABI.
    spec=importlib.util.spec_from_file_location('dependency_fixture',REPO/'tests/test_mediapipe_cpu_dependencies_acquire.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    monkeypatch.syspath_prepend(str(REPO/'infra'))
    spec=importlib.util.spec_from_file_location('dependency_runtime_fixture',REPO/'infra/mediapipe_cpu_dependencies_acquire.py')
    dep=importlib.util.module_from_spec(spec);spec.loader.exec_module(dep)
    root,old,revision,rows,closed=fixture.setup_verify(dep,tmp_path,monkeypatch)
    original_raw=closed.read_bytes();result=dep.acquire(root,old,revision,verify_acquired=True)
    config=dict(schema='world_reward.mediapipe_cpu_dependencies_acquire_pins.v1',producer_revision=revision,
                report=fixture.digest((root/dep.RESULT/'report.json').read_bytes()),
                helper=fixture.digest((old/'infra/mediapipe_cpu_dependencies_acquire.py').read_bytes()))
    files={n:(old/n).read_bytes()if(old/n).exists()else(REPO/n).read_bytes()for n in gate.HELPERS if n!=gate.DEP_PINS}
    files[gate.DEP_PINS]=json.dumps(config).encode()
    code=fixture.readonly_snapshot(root,'c'*40,gate.ENTRY,files)
    monkeypatch.setattr(dep,'__file__',str(code/'infra/mediapipe_cpu_dependencies_acquire.py'))
    monkeypatch.setattr(dep.mp,'__file__',str(code/'infra/mediapipe_hands_acquire.py'))
    monkeypatch.setattr(gate,'MANIFEST_PIN',dep.MANIFEST_PIN)
    original=gate.importlib.import_module
    monkeypatch.setattr(gate.importlib,'import_module',lambda name:dep if name=='mediapipe_cpu_dependencies_acquire'else original(name))
    download_config=json.loads((old/dep.DOWNLOAD_PINS).read_bytes())
    paths={'receipt':root/dep.RESULT/'report.json','marker':old.parent/'source-sha256',
           'wheel':root/dep.EVIDENCE/rows[1]['filename'],'task':root/dep.mp.WEIGHTS/'hand_landmarker.task',
           'helper':code/'infra/mediapipe_cpu_dependencies_acquire.py','original_receipt':closed,
           'original_source':root/'jobs'/download_config['producer_revision']/dep.JOB/'code/infra/mediapipe_cpu_dependencies_acquire.py',
           'download_pin':code/dep.DOWNLOAD_PINS}
    if fault:
        path=paths[fault];path.chmod(0o644);path.write_bytes(b'CHANGED');path.chmod(0o444)
        with pytest.raises((ValueError,dep.mp.AcquisitionError)):gate.prerequisites(root,code)
    else:
        proof=gate.prerequisites(root,code)
        assert len(proof['wheels'])==26 and len(result['artifacts'])==51
        assert proof['dependency_source']==result['source_binding']and proof['task_pin']==gate.identity(proof['task'])
        assert proof['verified_downloads']==result['prior_verified_downloads']and closed.read_bytes()==original_raw
        assert json.loads(original_raw)['status']=='fail'and result['network_used']is False and result['new_downloads']==0


def test_context_copy_counts_toward_build_deadline_before_any_build(host,monkeypatch):
    calls=tools(host,monkeypatch);clock=[0.0];original=gate.make_context
    monkeypatch.setattr(gate.time,'monotonic',lambda:clock[0])
    def slow(*args):original(*args);clock[0]=601.0
    monkeypatch.setattr(gate,'make_context',slow)
    with pytest.raises(ValueError):gate.run(host['root'],host['code'],host['revision'])
    assert not any(c[:2]==['docker','build']for c in calls)


def test_context_cleanup_error_keeps_a_sealed_failure(host,monkeypatch):
    tools(host,monkeypatch)
    monkeypatch.setattr(gate.shutil,'rmtree',lambda *_:(_ for _ in()).throw(OSError('Do not expose raw process secrets')))
    with pytest.raises(ValueError):gate.run(host['root'],host['code'],host['revision'])
    out=host['root']/'results'/('mediapipe-cpu-runtime-verify-'+host['revision'])
    receipt=json.loads((out/'report.json').read_bytes())
    assert receipt['status']=='fail'and receipt['context_error_type']=='OSError'and(out/'build-context').exists()
    assert 'secrets'not in json.dumps(receipt)
