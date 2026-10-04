"""Data-free wheel/CPU-build contract fixtures; no network, Docker or models."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import types
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'infra/run_bootstapir_runtime_build.sh'
SOURCE = WRAPPER.read_text().split("<<'PYBUILD'\n", 1)[1].rsplit('\nPYBUILD', 1)[0]


@pytest.fixture
def build():
    module = types.ModuleType('bootstapir_runtime_build_fixture')
    exec(compile(SOURCE, str(WRAPPER), 'exec'), module.__dict__)
    return module


@pytest.fixture
def pins():return json.loads((ROOT / 'configs/bootstapir_runtime_pins.json').read_text())


def wheel(name, version, license=b'procedural permission\n', fault=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr(name.replace('-', '_')+'.dist-info/METADATA', f'Name: {name}\nVersion: {version}\n')
        if fault != 'missing':archive.writestr(name+'.dist-info/licenses/LICENSE', license)
        if fault == 'traversal':archive.writestr('../foreign', b'bad')
        if fault == 'duplicate':
            with pytest.warns(UserWarning):archive.writestr(name+'.dist-info/licenses/LICENSE', license)
        if fault == 'symlink':
            item = zipfile.ZipInfo('foreign-link');item.external_attr = 0o120777 << 16;archive.writestr(item, b'foreign')
    return stream.getvalue()


def test_actual_small_frozen_cpu_dependencies(build, pins):
    assert build.validate(pins) == pins
    raw = (ROOT / build.CONFIG).read_bytes()
    assert build.CONFIG_PIN == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert sum(row['bytes'] for row in pins['wheels']) == 493734
    assert pins['platform'] == dict(os='linux', architecture='amd64', python='3.11', torch='2.5.1+cu124', numpy='1.26.3')
    assert pins['azure_vm_name'] == 'world-reward-ncc-h100-02' and pins['azure_resource_group'] == 'WORLD-REWARD-RESEARCH'
    assert all(row['publication_date'] <= '2026-09-30' for row in pins['wheels'])
    assert [row['license'] for row in pins['wheels']] == ['Apache-2.0', 'Apache-2.0', 'Apache-2.0', 'MIT', 'BSD-2-Clause']


@pytest.mark.parametrize('fault', ['base', 'target', 'budget', 'name', 'missing', 'sha', 'url', 'date', 'license'])
def test_invalid_runtime_pin_shape_rejected(build, pins, fault):
    if fault == 'base':pins['base_image_id'] = 'sha256:'+'0'*64
    elif fault == 'target':pins['target_image'] = 'foreign:latest'
    elif fault == 'budget':pins['budget_seconds'] = 601
    elif fault == 'name':pins['wheels'][0]['name'] = 'foreign'
    elif fault == 'missing':pins['wheels'].pop()
    elif fault == 'sha':pins['wheels'][0]['sha256'] = 'bad'
    elif fault == 'url':pins['wheels'][0]['url'] = 'https://example.com/a.whl'
    elif fault == 'date':pins['wheels'][0]['publication_date'] = '2026-10-01'
    else:pins['wheels'][0]['license'] = 'noncommercial'
    with pytest.raises(ValueError):build.validate(pins)


def test_checksum_before_wheel_parsing_and_unsigned_exact_transport(build, monkeypatch):
    raw = b'procedural bytes'; row = dict(url='https://files.pythonhosted.org/packages/a.whl', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    class Response:
        status = 200
        def __enter__(self):return self
        def __exit__(self, *args):pass
        def geturl(self):return row['url']
        def read(self, maximum):assert maximum == len(raw)+1;return raw
    class Opener:
        def open(self, request, timeout):assert request.headers['Accept-encoding'] == 'identity' and 0 < timeout <= 30;return Response()
    monkeypatch.setattr(build.urllib.request, 'build_opener', lambda *args: Opener())
    assert build.fetch(row, build.time.monotonic()+30) == raw
    with pytest.raises(ValueError, match='SHA'):build.fetch({**row, 'sha256':'0'*64}, build.time.monotonic()+30)
    with pytest.raises(ValueError, match='Unsigned'):build.fetch({**row, 'url':row['url']+'?token=never'}, build.time.monotonic()+30)
    with pytest.raises(ValueError, match='budget'):build.fetch(row, build.time.monotonic()-1)


def test_license_lineending_equivalence_records_actual_bytes(build, tmp_path):
    path = tmp_path/'test.whl'; path.write_bytes(wheel('dm-tree', '0.1.10', b'procedural permission\r\n'))
    result = build.wheel_notice(path, dict(name='dm-tree', version='0.1.10'), b'procedural permission\n')
    record = next(iter(result.values()))
    assert record == dict(bytes=23, sha256=hashlib.sha256(b'procedural permission\r\n').hexdigest())


@pytest.mark.parametrize('fault', ['missing', 'different', 'traversal', 'duplicate', 'symlink', 'version'])
def test_missing_or_changed_zip_license_and_structure_fail(build, tmp_path, fault):
    path = tmp_path/'test.whl'; path.write_bytes(wheel('dm-tree', '0.1.10', b'wrong' if fault == 'different' else b'procedural permission\n', fault))
    with pytest.raises(ValueError):build.wheel_notice(path, dict(name='dm-tree', version='other' if fault == 'version' else '0.1.10'), b'procedural permission\n')


def test_recipe_exact_no_deps_no_network_and_cpu_only_probe(build, pins):
    recipe = build.recipe(pins).decode()
    assert recipe.startswith('FROM '+build.BASE+'\n')
    assert '--no-index --no-deps --no-cache-dir' in recipe and '--force-reinstall' in recipe
    assert all(recipe.count('/opt/bootstapir-runtime/wheels/'+row['filename']) == 1 for row in pins['wheels'])
    assert not any(word in recipe for word in ('git+', 'jax', 'tensorflow', 'numpy==', 'torch==', 'curl', 'apt-get'))
    assert 'tree.map_structure' in build.PROBE and "einshape('ab->ba',x)" in build.PROBE
    assert 'not torch.cuda.is_initialized()' in build.PROBE and '.cuda(' not in build.PROBE
    assert 'tapir' not in build.PROBE.lower() and 'load(' not in build.PROBE


def test_bounded_private_log_and_no_environment_disclosure(build, tmp_path, monkeypatch):
    log = tmp_path/'build.log';calls=[]
    def run(args, **kwargs):
        calls.append((args, kwargs));kwargs['stdout'].write(b'private procedural diagnostic\n');return subprocess.CompletedProcess(args, 7)
    monkeypatch.setattr(build.subprocess, 'run', run)
    with pytest.raises(ValueError, match='inspect owned Azure build log') as error:build.command(['docker','build','--network','none'], log, build.time.monotonic()+20)
    assert 'diagnostic' not in str(error.value)
    args, kwargs = calls[0];assert args[:3] == ['/usr/bin/timeout','--signal=TERM','--kill-after=5s']
    assert kwargs['env'] == build.SAFE_ENV and kwargs['stderr'] == subprocess.STDOUT
    assert set(kwargs['env']) == {'PATH','HOME','LANG','DOCKER_HOST','DOCKER_BUILDKIT'}
    assert log.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):build.exclusive(log,b'new')


@pytest.fixture
def runtime(build, pins, tmp_path, monkeypatch):
    root = tmp_path/'runtime';rev = 'a'*40;code = root/'jobs'/rev/'run_bootstapir_runtime_build/code'
    (code/'infra').mkdir(parents=True);(code/'configs').mkdir();(root/'results').mkdir()
    (code.parent/'revision').write_text(rev+'\n');(code.parent/'source-sha256').write_text('b'*64+'\n')
    (code/build.HELPER).write_text(WRAPPER.read_text());responses={};license=b'procedural permission\n'
    for row in pins['wheels']:
        raw=wheel(row['name'],row['version'],license);responses[row['url']]=raw
        row['bytes']=len(raw);row['sha256']=hashlib.sha256(raw).hexdigest()
        primary=row['publisher_license'];primary['bytes']=len(license);primary['sha256']=hashlib.sha256(license).hexdigest();responses[primary['url']]=license
        if 'publisher_version_source' in row:
            raw=b'version 1.0 Apache';source=row['publisher_version_source'];source['bytes']=len(raw);source['sha256']=hashlib.sha256(raw).hexdigest();responses[source['url']]=raw
    (code/build.CONFIG).write_text(json.dumps(pins));(code/'empty.py').write_bytes(b'')
    for path in(code,*code.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)
    monkeypatch.setattr(build,'ROOT',root);monkeypatch.setattr(build,'CONFIG_PIN',build.identity(code/build.CONFIG));monkeypatch.setattr(build,'validate',lambda value:value)
    monkeypatch.setenv('WR_ROOT',str(root));monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',rev)
    monkeypatch.setattr(build.sys,'platform','linux');monkeypatch.setattr(build,'fetch',lambda record,deadline:responses[record['url']])
    base={'image_id':build.BASE,'layers':['sha256:'+str(i).zfill(64)for i in range(44)]}
    child={'image_id':'sha256:'+'f'*64,'layers':base['layers']+['sha256:'+'e'*64]};calls=[];fault={}
    def inspect(image,deadline,absent=False):calls.append(('inspect',image,absent));return None if absent else copy.deepcopy(base if image==build.BASE else child)
    monkeypatch.setattr(build,'inspect',inspect)
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,maximum):return json.dumps(dict(name=pins['azure_vm_name'],resourceGroupName=pins['azure_resource_group'])).encode()
    class Opener:
        def open(self,request,timeout):assert request.full_url.startswith('http://169.254.169.254/metadata/instance/compute');return Response()
    monkeypatch.setattr(build.urllib.request,'build_opener',lambda *args:Opener())
    def run(args,**kwargs):calls.append(('subprocess',args));return subprocess.CompletedProcess(args,0,b'',b'')
    monkeypatch.setattr(build.subprocess,'run',run)
    def command(args,log,deadline):
        calls.append(('command',args))
        if 'run' in args:
            record=dict(versions={r['name']:r['version']for r in pins['wheels']},tree_cpu_verified=True,einshape_torch_cpu_verified=True,cuda_initialized=False)
            build.exclusive(log,(json.dumps(record)+'\n').encode())
            build.exclusive(Path(args[args.index('--cidfile')+1]),('c'*64+'\n').encode())
        else:build.exclusive(log,b'procedural compiler diagnostic\n')
        if fault.get('operation')=='build' and 'build'in args:raise ValueError('procedural private build error')
    monkeypatch.setattr(build,'command',command)
    return dict(root=root,code=code,rev=rev,pins=pins,calls=calls,fault=fault,out=root/'results'/('bootstapir-runtime-build-'+rev))


def test_complete_cpu_runtime_without_checkpoint_or_data(build,runtime,capsys):
    assert build.run()==0
    report=json.loads((runtime['out']/'report.json').read_text());summary=json.loads(capsys.readouterr().out)
    assert report['status']=='pass'and report['install_verified']is True and report['source_rehashed_after']is True
    assert report['model_or_data_read']is False and report['gpu_execution']is False and report['benchmark_ready']is False and report['license_closure_verified']is False
    assert 'dependencies'not in summary and set(summary)=={'stage','status','phase','report','gpu_execution'}
    commands=[v for kind,*tail in runtime['calls']if kind=='command'for v in tail]
    assert commands[0][commands[0].index('--network')+1]=='none'
    assert '--network'in commands[1]and '--gpus'not in commands[1]and '--mount'not in commands[1]
    assert runtime['out'].stat().st_mode&0o777==0o700
    assert all(p.stat().st_mode&0o777==0o400 for p in runtime['out'].rglob('*')if p.is_file())


def test_preserve_failed_build_and_private_logs(build,runtime,capsys):
    runtime['fault']['operation']='build';assert build.run()==1
    report=json.loads((runtime['out']/'report.json').read_text());assert report['status']=='fail'and report['phase']=='offline_build'
    assert 'procedural private build error'not in json.dumps(report)and 'procedural'not in capsys.readouterr().out
    assert (runtime['out']/'Dockerfile').exists()and (runtime['out']/'build.log').exists()


def test_no_retag_or_source_mutation_adopted(build,runtime,monkeypatch):
    old=build.inspect
    def inspect(image,deadline,absent=False):
        if absent:raise ValueError('existing target tag')
        return old(image,deadline,absent)
    monkeypatch.setattr(build,'inspect',inspect)
    assert build.run()==1
    assert not any(row[0]=='command'for row in runtime['calls'])


def test_post_run_changed_helper_is_fail_not_pass(build,runtime,monkeypatch):
    old=build.command
    def command(args,log,deadline):
        old(args,log,deadline)
        if 'run'in args:
            path=runtime['code']/build.HELPER;path.chmod(0o644);path.write_text('changed');path.chmod(0o444)
    monkeypatch.setattr(build,'command',command)
    assert build.run()==1
    report=json.loads((runtime['out']/'report.json').read_text())
    assert report['status']=='fail'and report['source_rehashed_after']is False


@pytest.mark.parametrize('fault',['output','config','source','marker'])
def test_preflight_rejects_overwrite_or_changed_source(build,runtime,fault):
    if fault=='output':runtime['out'].mkdir()
    elif fault=='config':build.CONFIG_PIN={'bytes':1,'sha256':'0'*64}
    elif fault=='source':(runtime['code']/build.HELPER).chmod(0o644)
    else:(runtime['code'].parent/'revision').write_text('c'*40+'\n')
    with pytest.raises(ValueError):build.run()
    assert not runtime['calls']


def test_bash_syntax_budget_and_literal_config_runtime_closure(monkeypatch):
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    source=WRAPPER.read_text();assert len(source.splitlines())<=180
    assert '603s /usr/bin/python3 -I -B' in source and 'exec /usr/bin/env -i' in source
    assert not any(word in source for word in('flock','--gpus','pickle','torch.load','tapnet_source','systemctl stop','docker push','docker tag'))
    monkeypatch.syspath_prepend(str(ROOT/'infra'));import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes()for folder in('infra','src','configs')for p in(ROOT/folder).rglob('*')if p.is_file()and '__pycache__'not in p.parts};files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    selected=azure_job.runtime_bundle_paths(files,'infra/run_bootstapir_runtime_build.sh')
    assert {'infra/run_bootstapir_runtime_build.sh','configs/bootstapir_runtime_pins.json'}<=set(selected)
    assert 'infra/robotap_boots_acquire.py'not in selected
