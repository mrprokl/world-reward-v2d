"""Tiny independent CPU verification fixtures, never model/data/network work."""
import hashlib
import json
from pathlib import Path
import subprocess
import types

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'infra/run_bootstapir_runtime_verify.sh'
SOURCE=WRAPPER.read_text().split("<<'PYVERIFY'\n",1)[1].rsplit('\nPYVERIFY',1)[0]


@pytest.fixture
def verify():
    module=types.ModuleType('bootstapir_runtime_verify_fixture');exec(compile(SOURCE,str(WRAPPER),'exec'),module.__dict__);return module


def sha(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def save(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o400);return sha(raw)


@pytest.fixture
def runtime(verify,tmp_path,monkeypatch):
    root=tmp_path/'runtime';rev='a'*40;buildrev='b'*40
    code=root/'jobs'/rev/'run_bootstapir_runtime_verify/code'
    code.mkdir(parents=True);(root/'results').mkdir()
    save(code.parent/'revision',(rev+'\n').encode());save(code.parent/'source-sha256',('c'*64+'\n').encode())
    save(code/verify.HELPER,WRAPPER.read_bytes())
    pins=json.loads((ROOT/verify.CONFIG).read_text());pins['original_build_report']['producer_revision']=buildrev;pins['build_output']='results/build-procedural'
    original=root/pins['build_output'];oldcode=root/'jobs'/buildrev/'run_bootstapir_runtime_build/code'
    save(oldcode.parent/'revision',(buildrev+'\n').encode())
    dependencies={'wheels':[{'name':name,'version':version,'filename':name+'.whl','license':'Apache-2.0','publisher_license':{}}for name,version in(('dm-tree','0.1.10'),('einshape','1.0'),('absl-py','2.5.0'),('attrs','26.1.0'),('wrapt','1.17.3'))]}
    evidence={}
    for row in dependencies['wheels']:
        wheel=save(original/'wheels'/row['filename'],(row['name']+' tiny procedural wheel').encode());publisher=save(original/'publisher-licenses'/(row['name']+'.LICENSE'),b'procedural licence')
        row.update(wheel);row['publisher_license'].update(publisher)
        evidence[row['name']]={'wheel':wheel,'publisher_license':publisher,'license':row['license'],'embedded_notices':{'LICENSE':publisher}}
    pins['original_build_helpers']={
        'infra/run_bootstapir_runtime_build.sh':save(oldcode/'infra/run_bootstapir_runtime_build.sh',b'procedural historical build source'),
        'configs/bootstapir_runtime_pins.json':save(oldcode/'configs/bootstapir_runtime_pins.json',json.dumps(dependencies).encode())}
    pins['original_cpu_import_log']=save(original/'cpu-import.log',b'ModuleNotFoundError: einshape.torch\n')
    pins['original_build_log']=save(original/'build.log',b'procedural historical compiler log')
    base={'image_id':pins['base_image_id'],'layers':['sha256:'+str(i).zfill(64)for i in range(44)]}
    child={'image_id':pins['child_image_id'],'layers':base['layers']+['sha256:'+str(i).zfill(64)for i in range(44,47)]}
    pins['ordered_rootfs_sha256']=hashlib.sha256(json.dumps(child['layers'],separators=(',',':')).encode()).hexdigest()
    report={'stage':'bootstapir_runtime_build','status':'fail','phase':'cpu_import','producer_revision':buildrev,'source_rehashed_after':True,'model_or_data_read':False,'gpu_execution':False,'child_image_id':pins['child_image_id'],'ordered_rootfs_sha256':pins['ordered_rootfs_sha256'],'base_layers':44,'child_layers':47,'source_helpers':pins['original_build_helpers'],'dependencies':evidence}
    pins['original_build_report'].update(save(original/'report.json',json.dumps(report).encode()))
    protocol=json.loads((ROOT/'configs/robotap_boots_protocol.json').read_text())
    source=root/protocol['namespace']/'assets/tapnet_source'
    for name,row in protocol['source']['files'].items():row.update(save(source/name,('procedural unused native source '+name).encode()))
    acq={'report':{'producer_revision':'d'*40,'script_sha256':'e'*64}}
    acq['report'].update(save(root/protocol['namespace']/'report.json',json.dumps({'status':'pass',**acq['report']}).encode()))
    for key,value in(('source_protocol',protocol),('acquisition_pins',acq)):
        pins[key].update(save(code/pins[key]['path'],json.dumps(value).encode()))
    save(code/verify.CONFIG,json.dumps(pins).encode());(code/'empty.py').write_bytes(b'')
    for path in(code,*code.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)
    monkeypatch.setattr(verify,'ROOT',root);monkeypatch.setattr(verify,'PIN',sha((code/verify.CONFIG).read_bytes()))
    monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',rev)
    calls=[];fault={}
    def image(value,deadline):
        calls.append(('image',value));result=dict(base if value==pins['base_image_id']else child);result['layers']=result['layers'][:]
        if fault.get('image'):result['layers'][-1]='sha256:'+'f'*64
        return result
    monkeypatch.setattr(verify,'image',image)
    def control(args,deadline):calls.append(('control',args));return b''
    monkeypatch.setattr(verify,'control',control)
    versions={r['name']:r['version']for r in dependencies['wheels']}
    def run(args,**kwargs):
        calls.append(('run',args,kwargs));record={'versions':versions,'python':'3.11','torch':'2.5.1+cu124','numpy':'1.26.3','tree_cpu_verified':True,'source_native_einshape_cpu_verified':True,'native_bilinear_cpu_verified':True,'native_tapir_modules_imported':True,'models_instantiated':False,'cuda_initialized':False}
        if fault.get('native'):record['source_native_einshape_cpu_verified']=False
        kwargs['stdout'].write((json.dumps(record)+'\n').encode())
        save(Path(args[args.index('--cidfile')+1]),('f'*64+'\n').encode())
        if fault.get('after'):
            p=source/'tapnet/torch/utils.py';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o400)
        return subprocess.CompletedProcess(args,int(fault.get('status',0)))
    monkeypatch.setattr(verify.subprocess,'run',run)
    return dict(root=root,code=code,rev=rev,pins=pins,original=original,source=source,calls=calls,fault=fault,out=root/'results'/('bootstapir-runtime-verify-'+rev))


def test_real_pins_preserve_original_failed_build(verify):
    raw=(ROOT/verify.CONFIG).read_bytes();assert verify.PIN==sha(raw)
    pins=json.loads(raw);assert pins['original_build_report']==dict(bytes=4390,sha256='5ceff1a96cc8a705f2e8dfa43f17eb8486c57a821aecc32a3c8147ac74cab9a3',producer_revision='a06b703d017d40f9fedf2b1e9e19f11cfe40aa0f')
    assert pins['child_image_id']=='sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4'
    assert pins['base_layers']==44 and pins['child_layers']==47 and pins['budget_seconds']==120
    assert pins['azure_vm_name']=='world-reward-ncc-h100-02'and pins['azure_resource_group']=='WORLD-REWARD-RESEARCH'


def test_correct_source_native_cpu_api_no_model_or_gpu(verify):
    assert "from tapnet.torch import utils,nets,tapir_model"in verify.PROBE
    assert "utils.einshape('ab->ba',x)"in verify.PROBE and 'utils.bilinear('in verify.PROBE
    assert 'verify();sys.path.insert' in verify.PROBE and 'not torch.cuda.is_initialized()'in verify.PROBE
    assert not any(word in verify.PROBE for word in('einshape.torch','TAPIR(','torch.load','cuda(','jax','pickle'))


def test_complete_independent_verification_preserves_every_original_byte(verify,runtime,capsys):
    before={p:p.read_bytes()for p in runtime['original'].rglob('*')if p.is_file()}
    assert verify.run()==0
    report=json.loads((runtime['out']/'report.json').read_text());summary=json.loads(capsys.readouterr().out)
    assert report['status']=='pass'and report['source_rehashed_after']is True and report['original_build_status']=='fail'and report['original_build_failure_preserved']is True
    assert report['checkpoint_read']is False and report['rgb_or_labels_read']is False and report['gpu_execution']is False
    assert len(report['native_sources'])==7 and report['cpu_import']['native_tapir_modules_imported']is True
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert set(summary)=={'stage','status','phase','report','original_build_status','gpu_execution'}
    args,kwargs=next((row[1],row[2])for row in runtime['calls']if row[0]=='run')
    assert args[:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=5s','100s']
    assert args.count('--mount')==7 and args[args.index('--user')+1]=='0:0'
    assert '--network'in args and args[args.index('--network')+1]=='none'and '--gpus'not in args
    assert '--cap-drop'in args and '--read-only'in args and '--security-opt'in args
    assert all('checkpoint'not in args[i+1]and 'eval_private'not in args[i+1]for i,v in enumerate(args)if v=='--mount')
    assert kwargs['env']==verify.SAFE_ENV and kwargs['stderr']==subprocess.STDOUT
    assert runtime['out'].stat().st_mode&0o777==0o700
    assert all(p.stat().st_mode&0o777==0o400 for p in runtime['out'].iterdir()if p.is_file())


@pytest.mark.parametrize('fault',['report','log','source','wheel','publisher','acquisition'])
def test_independently_pinned_bytes_before_native_import(verify,runtime,fault):
    if fault=='report':path=runtime['original']/'report.json'
    elif fault=='log':path=runtime['original']/'cpu-import.log'
    elif fault=='source':path=runtime['source']/'tapnet/torch/utils.py'
    elif fault=='wheel':path=runtime['original']/'wheels/dm-tree.whl'
    elif fault=='publisher':path=runtime['original']/'publisher-licenses/dm-tree.LICENSE'
    else:path=runtime['root']/'validation/robotap_boots_v1/report.json'
    path.chmod(0o644);path.write_bytes(b'changed');path.chmod(0o400)
    assert verify.run()==1
    assert not any(row[0]=='run'for row in runtime['calls'])


@pytest.mark.parametrize('fault',['native','image','after','status'])
def test_native_or_post_mutation_failure_not_promoted(verify,runtime,fault):
    runtime['fault'][fault]=7 if fault=='status'else True
    assert verify.run()==1
    report=json.loads((runtime['out']/'report.json').read_text());assert report['status']=='fail'
    assert report['original_build_status']=='fail'and json.loads((runtime['original']/'report.json').read_text())['status']=='fail'
    if fault=='after':assert report['source_rehashed_after']is False


@pytest.mark.parametrize('fault',['out','config','writable','marker','namespace'])
def test_preflight_rejects_changes_before_any_docker(verify,runtime,monkeypatch,fault):
    if fault=='out':runtime['out'].mkdir()
    elif fault=='config':monkeypatch.setattr(verify,'PIN',{'bytes':1,'sha256':'0'*64})
    elif fault=='writable':(runtime['code']/verify.HELPER).chmod(0o644)
    elif fault=='marker':
        p=runtime['code'].parent/'revision';p.chmod(0o644);p.write_text('c'*40+'\n')
    else:monkeypatch.setenv('WR_CODE',str(runtime['root']))
    with pytest.raises(ValueError):verify.run()
    assert not runtime['calls']


def test_log_owned_before_subprocess_and_sanitized_failure(verify,runtime,capsys):
    runtime['fault']['status']=7;assert verify.run()==1
    args,kwargs=next((row[1],row[2])for row in runtime['calls']if row[0]=='run')
    assert kwargs['stdout'].closed
    report=json.loads((runtime['out']/'report.json').read_text());assert report['cpu_native_import_log']==sha((runtime['out']/'cpu-native-import.log').read_bytes())
    assert 'traceback'not in capsys.readouterr().out.lower()


def test_static_syntax_and_source_closure_no_old_build_execution(monkeypatch):
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    source=WRAPPER.read_text();assert len(source.splitlines())<=180 and '123s /usr/bin/python3 -I -B'in source
    assert not any(word in source for word in('docker build','docker tag','pip install','urllib','torch.load','pickle','--gpus','flock'))
    monkeypatch.syspath_prepend(str(ROOT/'infra'));import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes()for folder in('infra','src','configs')for p in(ROOT/folder).rglob('*')if p.is_file()and '__pycache__'not in p.parts};files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    selected=azure_job.runtime_bundle_paths(files,'infra/run_bootstapir_runtime_verify.sh')
    assert {'infra/run_bootstapir_runtime_verify.sh','configs/bootstapir_runtime_verify_pins.json','configs/robotap_boots_protocol.json','configs/robotap_boots_acquisition_pins.json'}<=set(selected)
    # The historical builder can be retained as literal provenance, never run.
    assert 'infra/robotap_boots_acquire.py'not in selected
