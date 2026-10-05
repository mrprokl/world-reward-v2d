"""Manufactured archives/wheels/lifecycle only; no installs/Torch/GPU/network."""
import ast
import copy
import email.message
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location('wr_hoi_runtime_test', REPO/'infra/hoi_detr_runtime_verify.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Response(io.BytesIO):
    def __init__(self, raw, url, *, status=200, encoding='identity', length=None):
        super().__init__(raw); self.url = url; self.status = status; self.headers = email.message.Message()
        self.headers['Content-Encoding'] = encoding; self.headers['Content-Length'] = str(len(raw) if length is None else length); self.headers['Content-Type'] = 'application/octet-stream'
    def geturl(self): return self.url
    def read(self, n=-1):
        assert 0 < n <= 1 << 20
        return super().read(n)


class Opener:
    def __init__(self, rows): self.rows = rows; self.calls = []
    def open(self, request, timeout):
        assert dict(request.header_items()) == {'Accept-encoding': 'identity'} and 0 < timeout <= 30
        self.calls.append(request.full_url); return self.rows[request.full_url]


def load_helpers(m, tmp_path):
    code = tmp_path/'code'; code.mkdir()
    for n in m.HELPERS[3:]:
        p = code/n; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((REPO/n).read_bytes()); p.chmod(0o444)
    for p in sorted(code.rglob('*'), reverse=True):
        if p.is_dir(): p.chmod(0o555)
    code.chmod(0o555)
    rt, mp, acq = m.helpers(code)
    def publish(a, b): os.link(a, b); a.unlink()
    mp.publish = publish
    return rt, mp, acq


def blob(raw, mode='100644'):
    return dict(mode=mode, bytes=len(raw), git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())


def tar(prefix, files, links, extra=None):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w:gz') as z:
        t = tarfile.TarInfo(prefix+'/'); t.type = tarfile.DIRTYPE; z.addfile(t)
        for n, (raw, mode) in files.items():
            t = tarfile.TarInfo(prefix+'/'+n); t.size = len(raw); t.mode = mode; z.addfile(t, io.BytesIO(raw))
        for n, target in links.items():
            t = tarfile.TarInfo(prefix+'/'+n); t.type = tarfile.SYMTYPE; t.linkname = target; z.addfile(t)
        if extra:
            t, raw = extra; z.addfile(t, io.BytesIO(raw) if t.isfile() else None)
    return out.getvalue()


def fixture_source(m, tmp_path):
    rt, mp, acq = load_helpers(m, tmp_path); data = tmp_path/'data'; data.mkdir()
    files = {'LICENSE': (b'Apache License\n', 0o644), 'setup.py': (b'# Original tiny setup never executed\n', 0o644),
             'mmcv/ops/a.py': (b'raise RuntimeError("DO NOT IMPORT")\n', 0o644), 'mmcv/ops/csrc/a.cu': (b'// Tiny CUDA not compiled\n', 0o644),
             'mmcv/native_exec.py': (b'# executable source mode\n', 0o755), 'mmcv/empty.py': (b'', 0o644),
             'docs/image.png': (b'OPAQUE NEVER DECODE', 0o644), 'docs/vendor/NOTICE.txt': (b'Notice retained\n', 0o644)}
    links = {'docs/zh_cn/mmcv-logo.png': '../docs/mmcv-logo.png'}
    rows = {n: blob(raw, '100755' if mode & 0o111 else '100644') for n, (raw, mode) in files.items()} | {n: blob(v.encode(), '120000') for n, v in links.items()}
    spec = copy.deepcopy(json.loads((REPO/m.PROTOCOL).read_bytes())['source'])
    spec.update(blob_count=len(rows), expanded_bytes=sum(r['bytes'] for r in rows.values()), git_root_tree_sha1=acq.git_tree(rows))
    spec['required_source_pins'] = {n: dict(bytes=len(files[n][0]), sha256=hashlib.sha256(files[n][0]).hexdigest()) for n in ('LICENSE', 'setup.py')}
    path = data/'source.tar.gz'; path.write_bytes(tar(spec['archive_prefix'], files, links)); path.chmod(0o444)
    return rt, mp, acq, data, files, links, spec, path


def test_full_source_tree_handles_native_executable_modes_and_no_media(gate, tmp_path):
    rt, mp, acq, data, files, links, spec, path = fixture_source(gate, tmp_path)
    records = []; values, proof = gate.unpack(rt, mp, acq, data, path, spec, gate.time.monotonic()+30, [], set(), records)
    assert values is records and proof['all_original_blobs_authenticated'] and proof['native_executable_git_modes_preserved']
    assert proof['symlink_texts_verified'] == 1 and not proof['links_materialized']
    assert not (data/'source/mmcv/docs/image.png').exists() and not (data/'source/mmcv/docs/zh_cn/mmcv-logo.png').exists()
    assert (data/'source/mmcv/docs/vendor/NOTICE.txt').read_bytes() == b'Notice retained\n'
    assert (data/'source/mmcv/mmcv/empty.py').read_bytes() == b''
    r = next(r for r in records if r['file'].endswith('native_exec.py'))
    assert r['original_git_mode'] == '100755' and (data/r['file']).stat().st_mode & 0o777 == 0o444
    assert not list(data.rglob('*.part'))


@pytest.mark.parametrize('fault', ['wrong_blob', 'excluded_blob', 'wrong_mode', 'link', 'missing_link', 'duplicate', 'traversal', 'hardlink', 'foreign_dir', 'special', 'count', 'tree', 'non_utf8', 'nul', 'member_cap'])
def test_source_tree_faults_no_native_execution(gate, tmp_path, fault):
    rt, mp, acq, data, files, links, spec, path = fixture_source(gate, tmp_path); files = dict(files); links = dict(links); extra = None
    if fault == 'wrong_blob': files['mmcv/ops/a.py'] = (b'# changed\n', 0o644)
    elif fault == 'excluded_blob': files['docs/image.png'] = (b'changed opaque', 0o644)
    elif fault == 'wrong_mode': files['mmcv/native_exec.py'] = (files['mmcv/native_exec.py'][0], 0o644)
    elif fault == 'link': links['docs/zh_cn/mmcv-logo.png'] = '/etc/passwd'
    elif fault == 'missing_link': links = {}
    elif fault == 'count': spec['blob_count'] += 1
    elif fault == 'tree': spec['git_root_tree_sha1'] = '0'*40
    elif fault == 'non_utf8': files['mmcv/ops/a.py'] = (b'\xff', 0o644)
    elif fault == 'nul': files['mmcv/ops/a.py'] = (b'\0', 0o644)
    elif fault == 'member_cap': spec['maximum_member_bytes'] = 1
    else:
        n = '../escape' if fault == 'traversal' else spec['archive_prefix']+('/LICENSE' if fault == 'duplicate' else '/extra')
        t = tarfile.TarInfo(n); t.size = 1; raw = b'X'
        if fault == 'hardlink': t.type = tarfile.LNKTYPE; t.size = 0; t.linkname = '/etc/passwd'
        elif fault == 'special': t.type = tarfile.CHRTYPE; t.size = 0
        elif fault == 'foreign_dir': t.type = tarfile.DIRTYPE; t.size = 0
        extra = (t, raw)
    path.chmod(0o644); path.write_bytes(tar(spec['archive_prefix'], files, links, extra)); path.chmod(0o444)
    records = []
    with pytest.raises((ValueError, UnicodeDecodeError)):
        gate.unpack(rt, mp, acq, data, path, spec, gate.time.monotonic()+30, [], set(), records)
    assert not (tmp_path/'escape').exists()
    # Partial retained files stay in caller's inventory for safe failure cleanup.
    assert {data/r['file'] for r in records} == {p for p in (data/'source').rglob('*') if p.is_file()} if (data/'source').exists() else not records


def test_exact_opaque_download_and_failures_no_retry(gate, tmp_path):
    rt, mp, acq = load_helpers(gate, tmp_path); data = tmp_path/'data'; data.mkdir(); b = b'TINY SOURCE ONLY'; url = 'https://files.pythonhosted.org/packages/tiny/source.whl'
    row = dict(file='wheels/tiny.whl', url=url, bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
    opener = Opener({url: Response(b, url)}); value = gate.fetch(mp, acq, data, row, gate.time.monotonic()+30, [], set(), opener)
    assert value['bytes'] == len(b) and value['sha256'] == row['sha256'] and opener.calls == [url]


@pytest.mark.parametrize('fault', ['hash', 'length', 'encoding', 'redirect', 'overflow', 'html', 'pointer', 'status'])
def test_bounded_public_transport_faults(gate, tmp_path, fault):
    rt, mp, acq = load_helpers(gate, tmp_path); data = tmp_path/'data'; data.mkdir(); b = b'TINY PUBLIC'; url = 'https://files.pythonhosted.org/packages/tiny/file.whl'
    row = dict(file='wheels/tiny.whl', url=url, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()); kw = {}; final = url
    if fault == 'hash': b = b'X'+b[1:]
    elif fault == 'length': kw['length'] = len(b)+1
    elif fault == 'encoding': kw['encoding'] = 'gzip'
    elif fault == 'redirect': final = 'https://evil.invalid/?SECRET'
    elif fault == 'overflow': b += b'X'; kw['length'] = row['bytes']
    elif fault == 'html': b = b'<html>'; kw['length'] = row['bytes']
    elif fault == 'pointer': b = b'version https://git-lfs.github.com/spec/v1'; kw['length'] = row['bytes']
    else: kw['status'] = 403
    opener = Opener({url: Response(b, final, **kw)})
    with pytest.raises(ValueError): gate.fetch(mp, acq, data, row, gate.time.monotonic()+30, [], set(), opener)
    assert opener.calls == [url] and not (data/row['file']).exists()


def wheel(path, *, name='tiny', version='1', grant=b'MIT License\n', extra=None):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name+'-1.dist-info/METADATA', 'Name: '+name+'\nVersion: '+version+'\nRequires-Dist: testdependency>=1\n')
        z.writestr(name+'-1.dist-info/LICENSE', grant)
        z.writestr(name+'/source.py', 'raise RuntimeError("DO NOT EXECUTE")')
        if extra: z.writestr(*extra)


def test_original_wheel_metadata_notices_without_install(gate, tmp_path):
    p = tmp_path/'tiny.whl'; wheel(p)
    result = gate.wheel_notice(p, dict(name='tiny', version='1', license='MIT'), b'MIT License\n')
    assert result['license'] == 'MIT' and result['requires_dist'] == ['testdependency>=1'] and result['notices']


@pytest.mark.parametrize('fault', ['name', 'version', 'grant', 'traversal', 'symlink'])
def test_wheel_notice_faults(gate, tmp_path, fault):
    p = tmp_path/'tiny.whl'; extra = None
    if fault == 'traversal': extra = ('../escape', b'X')
    elif fault == 'symlink':
        i = zipfile.ZipInfo('tiny/link'); i.create_system = 3; i.external_attr = (stat.S_IFLNK | 0o777) << 16; extra = (i, b'/etc/passwd')
    wheel(p, name='wrong' if fault == 'name' else 'tiny', version='2' if fault == 'version' else '1', grant=b'bad grant' if fault == 'grant' else b'MIT License\n', extra=extra)
    with pytest.raises(ValueError): gate.wheel_notice(p, dict(name='tiny', version='1', license='MIT'), b'MIT License\n')


def inspect_value(image_id, name, revision, mounts, native):
    return dict(Image=image_id, Name='/'+name, Config=dict(User='1000:1000', Labels={'world-reward.job':'run_hoi_detr_runtime_verify', 'world-reward.revision':revision}),
                HostConfig=dict(NetworkMode='none', ReadonlyRootfs=True, Privileged=False, CapDrop=['ALL'], SecurityOpt=['no-new-privileges'], Binds=None, VolumesFrom=None, Devices=None,
                    DeviceRequests=[dict(Driver='nvidia', Count=-1, Capabilities=[['gpu']])] if native else None),
                Mounts=[dict(Type='bind', Source=str(a), Destination=str(b), RW=not c) for a,b,c in mounts])


def test_container_plan_public_only_and_native_gpu_modes(gate):
    p = json.loads((REPO/gate.PROTOCOL).read_bytes()); code = Path('/srv/code/code'); data = Path(p['data_root']); out = Path('/srv/results'); rev = 'a'*40
    for native in (False, True):
        name, cid, mounts, args = gate.container_plan(code, data, out, p['base_image_id'], rev, native=native, p=p, proof_pin=dict(bytes=12, sha256='b'*64), deadline=10)
        assert len(mounts) == 3 and mounts[0] == (code.parent, code.parent, True) and mounts[1] == (data, data, True) and mounts[2] == (out, out, False)
        assert '--network' in args and 'none' in args and '--read-only' in args and '--cap-drop' in args and 'ALL' in args
        assert ('--gpus' in args) is native and ('--native' in args) is native and ('--compile' in args) is not native
        assert ('128g' in args) is not native and ('16g' in args) is native
        assert not any(x in ' '.join(args).lower() for x in ('checkpoint', 'epoch_5', 'hocap', 'rgb', 'challenge', 'docker.sock'))
        gate.validate_container(inspect_value(p['base_image_id'], name, rev, mounts, native), image_id=p['base_image_id'], name=name, revision=rev, mounts=mounts, native=native)


@pytest.mark.parametrize('fault', ['image', 'owner', 'user', 'network', 'readonly', 'privileged', 'mount', 'foreign_volume', 'gpu', 'devices', 'capability', 'binds'])
def test_sandbox_foreign_mount_or_identity_rejected(gate, fault):
    image_id = 'sha256:'+'a'*64; rev = 'b'*40; mounts = [(Path('/srv/code'), Path('/srv/code'), True)]; name = 'owned'; v = inspect_value(image_id, name, rev, mounts, True)
    if fault == 'image': v['Image'] = 'sha256:'+'c'*64
    elif fault == 'owner': v['Config']['Labels']['world-reward.revision'] = 'c'*40
    elif fault == 'user': v['Config']['User'] = '0:0'
    elif fault == 'network': v['HostConfig']['NetworkMode'] = 'host'
    elif fault == 'readonly': v['HostConfig']['ReadonlyRootfs'] = False
    elif fault == 'privileged': v['HostConfig']['Privileged'] = True
    elif fault == 'mount': v['Mounts'][0]['Source'] = '/private/data'
    elif fault == 'foreign_volume': v['Mounts'].append(dict(Type='volume'))
    elif fault == 'gpu': v['HostConfig']['DeviceRequests'] = None
    elif fault == 'devices': v['HostConfig']['Devices'] = [{'PathOnHost':'/dev/private'}]
    elif fault == 'capability': v['HostConfig']['DeviceRequests'][0]['Capabilities'] = [['gpu','compute']]
    else: v['HostConfig']['Binds'] = ['/private:/private']
    with pytest.raises(ValueError): gate.validate_container(v, image_id=image_id, name=name, revision=rev, mounts=mounts, native=True)


def test_source_only_manifest_full_hash_and_no_foreign_artifacts(gate, tmp_path):
    rt, mp, acq = load_helpers(gate, tmp_path); data = tmp_path/'inputs'; (data/'source/mmcv').mkdir(parents=True)
    b = b'# tiny\n'; p = data/'source/mmcv/a.py'; p.write_bytes(b); p.chmod(0o444)
    m = dict(schema='world_reward.hoi_detr_mmcv_inputs.v1', source=dict(all_original_blobs_authenticated=True), artifacts=[dict(file='source/mmcv/a.py', bytes=len(b), sha256=hashlib.sha256(b).hexdigest())])
    raw = json.dumps(m).encode(); (data/'manifest.json').write_bytes(raw); (data/'manifest.json').chmod(0o444)
    for d in (data/'source/mmcv', data/'source', data): d.chmod(0o555)
    pin = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()); assert gate.data_proof(rt, data, pin) == m
    data.chmod(0o755); (data/'foreign.pth').write_bytes(b'OPAQUE'); (data/'foreign.pth').chmod(0o444); data.chmod(0o555)
    with pytest.raises(ValueError, match='foreign'): gate.data_proof(rt, data, pin)


def test_real_pins_compiler_and_operator_scope(gate):
    raw = (REPO/gate.PROTOCOL).read_bytes(); c = json.loads(raw)
    assert dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == gate.PROTOCOL_PIN
    assert c['source']['git_root_tree_sha1'] == 'e5a9980219f78ae70ec004c18d7be58773925870'
    assert (c['source']['blob_count'], c['source']['expanded_bytes']) == (1026,14700206)
    assert c['source']['symlinks']['docs/zh_cn/mmcv-logo.png']['target'] == '../docs/mmcv-logo.png'
    assert c['build_cpus'] == c['maximum_jobs'] == 32 and c['cpu_memory'] == '128g' and c['gpu_cpus'] == 4
    assert [r['name'] for r in c['wheels']] == ['addict','yapf','tomli']
    assert all(c[k] is False for k in gate.FLAGS) and c['operators']['softNMS']['device'] == 'cpu'
    assert c['operators']['softNMS']['native_cpu_only'] is True
    helper = (REPO/'infra/hoi_detr_acquire.py').read_bytes(); assert dict(bytes=len(helper),sha256=hashlib.sha256(helper).hexdigest()) == gate.ACQUIRE_PIN
    source = (REPO/gate.HELPERS[0]).read_text(); ast.parse(source)
    assert 'extractall' not in source and 'torch.load' not in source and 'import mmdet' not in source and 'import projects' not in source
    assert "out/'compile_proof.json'" in source and "out/'operators_proof.json'" in source and "(out/'proof.json').unlink()" not in source
    assert c['build']['MAX_JOBS'] == '32'
    subprocess.run(['bash','-n',str(REPO/gate.HELPERS[1])],check=True)


@pytest.mark.parametrize('name', ['mmcv/ops/csrc/common/x.cuh','mmcv/ops/csrc/pytorch/x.cpp','mmcv/ops/csrc/common/x.hpp','mmcv/ops/x.py','requirements/runtime.txt','setup.py','MANIFEST.in','docs/vendor/LICENSE'])
def test_all_numerical_source_and_notices_retained(gate, tmp_path, name):
    rt, mp, acq = load_helpers(gate, tmp_path); assert gate.keep_source(acq,name)


@pytest.mark.parametrize('name', ['docs/image.png','tests/test_ops/test_nms.py','demo/a.mp4','mmcv/__pycache__/x.pyc','mmcv/private.json','weights/model.pth'])
def test_media_weights_or_unneeded_tests_never_retained(gate, tmp_path, name):
    rt, mp, acq = load_helpers(gate, tmp_path); assert not gate.keep_source(acq,name)


def test_actual_vm02_image_absence_contract(gate, monkeypatch):
    target = 'world-reward/hoi-detr-mmcv-native-v3'
    def absent(args, **kwargs):
        assert args == ['docker','image','inspect',target,'--format','{{.Id}}']
        return subprocess.CompletedProcess(args,1,b'\n',b'Error response from daemon: No such image: world-reward/hoi-detr-mmcv-native-v3:latest\n')
    monkeypatch.setattr(gate.subprocess,'run',absent)
    gate.absent(target,gate.time.monotonic()+30,image_name=True)


@pytest.mark.parametrize('fault',['daemon','permission','rc','present','wrong_name','wrong_tag'])
def test_absence_never_accepts_ambiguous_daemon_or_foreign_name(gate, monkeypatch, fault):
    stderr = b'Error response from daemon: No such image: world-reward/hoi-detr-mmcv-native-v3:latest\n'; stdout = b'\n'; rc=1
    if fault == 'daemon': stderr=b'Cannot connect to the Docker daemon\n'
    elif fault == 'permission': stderr=b'permission denied\n'
    elif fault == 'rc': rc=2
    elif fault == 'present': rc=0; stdout=b'sha256:'+b'a'*64+b'\n'; stderr=b''
    elif fault == 'wrong_name': stderr=b'Error response from daemon: No such image: foreign:latest\n'
    else: stderr=b'Error response from daemon: No such image: world-reward/hoi-detr-mmcv-native-v3:other\n'
    monkeypatch.setattr(gate.subprocess,'run',lambda args,**kw:subprocess.CompletedProcess(args,rc,stdout,stderr))
    with pytest.raises(ValueError): gate.absent('world-reward/hoi-detr-mmcv-native-v3',gate.time.monotonic()+30,image_name=True)


def test_owned_scratch_cleanup_rejects_replacement_inode(gate,tmp_path):
    class RT:
        @staticmethod
        def canonical(p):
            assert p.resolve()==p and not p.is_symlink();return p
    out=tmp_path/'out';out.mkdir();folder=out/'.scratch';folder.mkdir();t=folder.lstat();inode=(t.st_dev,t.st_ino,t.st_uid)
    original=out/'old';folder.rename(original);folder.mkdir();(folder/'foreign').write_bytes(b'preserve')
    with pytest.raises(ValueError,match='original owned'):gate.remove_owned_folder(RT,folder,inode,out)
    assert (folder/'foreign').read_bytes()==b'preserve' and original.is_dir()


def test_owned_scratch_cleanup_rejects_symlink_tree(gate,tmp_path):
    class RT:
        @staticmethod
        def canonical(p):return p
    out=tmp_path/'out';out.mkdir();folder=out/'.scratch';folder.mkdir();(folder/'link').symlink_to('/etc/passwd');t=folder.lstat()
    with pytest.raises(ValueError,match='symlink'):gate.remove_owned_folder(RT,folder,(t.st_dev,t.st_ino,t.st_uid),out)
    assert (folder/'link').is_symlink()


def test_owned_scratch_cleanup_removes_only_exact_original(gate,tmp_path):
    class RT:
        @staticmethod
        def canonical(p):return p
    out=tmp_path/'out';out.mkdir();folder=out/'.scratch';folder.mkdir();(folder/'tiny').write_bytes(b'owned');t=folder.lstat()
    other=out/'other';other.write_bytes(b'preserve');gate.remove_owned_folder(RT,folder,(t.st_dev,t.st_ino,t.st_uid),out)
    assert not folder.exists() and other.read_bytes()==b'preserve'


def test_production_config_validates_exact_no_cpu_substitution(gate):
    class RT:
        @staticmethod
        def pinned(path,pin,maximum):return json.loads(path.read_bytes())
    p=gate.protocol(RT,REPO)
    assert p['operators']['MSDeformAttn']['heads']==8 and p['operators']['MSDeformAttn']['head_dimension']==32
    assert p['operators']['MSDeformAttn']['levels']==5 and p['compiler']['torch_cuda_arch_list']=='9.0'


def test_containers_use_explicit_nvidia_driver_not_docker_default(gate):
    p=json.loads((REPO/gate.PROTOCOL).read_bytes())
    _,_,_,args=gate.container_plan(Path('/srv/code/code'),Path('/srv/data'),Path('/srv/out'),p['base_image_id'],'a'*40,native=True,p=p,proof_pin=dict(bytes=1,sha256='b'*64),deadline=20)
    assert args[args.index('--gpus')+1]=='driver=nvidia,count=all'


def test_actual_entry_runtime_bundle_has_complete_readonly_helper_closure(gate):
    spec=importlib.util.spec_from_file_location('azure_job_runtime_test',REPO/'infra/azure_job.py');azure=importlib.util.module_from_spec(spec);spec.loader.exec_module(azure)
    files={str(p.relative_to(REPO)):p.read_bytes() for top in('infra','src','configs')for p in(REPO/top).rglob('*')if p.is_file()and'__pycache__'not in p.parts}
    files['pyproject.toml']=(REPO/'pyproject.toml').read_bytes()
    selected=azure.runtime_bundle_paths(files,gate.HELPERS[1])
    assert set(gate.HELPERS)<=set(selected) and len(selected)>len(gate.HELPERS)


@pytest.mark.parametrize('fault',['byte','size','writable','symlink'])
def test_helper_authenticated_before_any_native_import(gate,tmp_path,fault):
    rt,mp,acq=load_helpers(gate,tmp_path);code=tmp_path/'code';p=code/'infra/hoi_detr_acquire.py'
    if fault=='symlink':
        p.parent.chmod(0o755);p.unlink();p.symlink_to(REPO/'infra/hoi_detr_acquire.py')
    else:
        p.chmod(0o644)
        if fault=='byte':b=p.read_bytes();p.write_bytes(b'!'+b[1:]);p.chmod(0o444)
        elif fault=='size':p.write_bytes(b'raise RuntimeError("DO NOT IMPORT")');p.chmod(0o444)
    with pytest.raises(ValueError):gate.helpers(code)


def test_create_registered_before_control_and_no_cleanup_after_final_receipt(gate):
    source=(REPO/gate.HELPERS[0]).read_text()
    assert "containers.append((cidfile, name, base['Id'])); command(args, deadline)" in source
    assert "containers.append((cidfile, name, child['Id'])); command(args, deadline)" in source
    final=source[source.index("        try:\n            if r['status'] == 'pass':"):]
    assert final.index("out.chmod(0o555)") < final.index("acq.write_receipt(out/'report.json'")
    tail=final[final.index("acq.write_receipt(out/'report.json'"):]
    assert '.unlink()' not in tail and '.chmod(' not in tail


def stub_base_versions(gate, monkeypatch, tmp_path, *, changed=None, header=True):
    """Tiny public metadata seam; never import/install the actual native stack."""
    import importlib.metadata as metadata
    import sysconfig
    p = json.loads((REPO/gate.PROTOCOL).read_bytes())
    found = dict(p['base_distributions'])
    if changed:
        found[changed[0]] = changed[1]
    include = tmp_path/'include'; include.mkdir()
    if header:
        (include/'Python.h').write_bytes(b'/* tiny header availability only */')
    torch = SimpleNamespace(__version__=p['platform']['torch'], _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False))
    numpy = SimpleNamespace(__version__=p['platform']['numpy'])
    monkeypatch.setitem(sys.modules, 'torch', torch); monkeypatch.setitem(sys.modules, 'numpy', numpy)
    monkeypatch.setattr(gate.sys, 'version', p['platform']['python']+' manufactured test')
    monkeypatch.setattr(sysconfig, 'get_config_var', lambda name:p['platform']['SOABI'] if name=='SOABI' else None)
    monkeypatch.setattr(sysconfig, 'get_path', lambda name:str(include) if name=='include' else None)
    calls = []
    def version(name):
        calls.append(name); return found[name]
    monkeypatch.setattr(metadata, 'version', version)
    return p, torch, numpy, calls


def test_actual_ef12_metadata_901_without_install_or_downgrade(gate, monkeypatch, tmp_path):
    p, torch, numpy, calls = stub_base_versions(gate, monkeypatch, tmp_path)
    actual_torch, actual_numpy, report = gate.versions(p)
    assert actual_torch is torch and actual_numpy is numpy
    assert report['base_distributions'] == p['base_distributions'] and report['base_distributions']['importlib-metadata']=='9.0.1'
    assert calls == list(p['base_distributions'])


@pytest.mark.parametrize('changed', [('importlib-metadata','8.5.0'), ('importlib-metadata','9.0.2'), ('Pillow','10.3.0')])
def test_actual_base_metadata_mismatch_still_rejected(gate, monkeypatch, tmp_path, changed):
    p, _, _, _ = stub_base_versions(gate, monkeypatch, tmp_path, changed=changed)
    with pytest.raises(ValueError, match='Frozen unchanged base dependencies/headers required'):
        gate.versions(p)


def test_base_header_presence_still_required(gate, monkeypatch, tmp_path):
    p, _, _, _ = stub_base_versions(gate, monkeypatch, tmp_path, header=False)
    with pytest.raises(ValueError, match='Frozen unchanged base dependencies/headers required'):
        gate.versions(p)


def test_technical_v3_namespaces_preserve_original_v1_failure(gate):
    p = json.loads((REPO/gate.PROTOCOL).read_bytes())
    assert p['schema']=='world_reward.hoi_detr_runtime.v3'
    assert gate.DATA==Path('/srv/world-reward-data/hoi_detr_runtime_v3') and p['data_root']==str(gate.DATA)
    assert p['output']=='results/hoi-detr-runtime-v3' and p['target_image']=='world-reward/hoi-detr-mmcv-native-v3'
    assert (p['budget_seconds'],p['cleanup_grace_seconds'],p['outer_seconds'])==(1800,60,1860)
    source = (REPO/gate.HELPERS[0]).read_text()
    assert '/srv/world-reward-data/hoi_detr_runtime_v1' not in source and 'results/hoi-detr-runtime-v1' not in source
    assert gate.ENTRY=='run_hoi_detr_runtime_verify' and gate.PROTOCOL=='configs/hoi_detr_runtime_v1.json'


@pytest.mark.parametrize('message', [
    'Frozen unchanged base dependencies/headers required',
    'Original CP311/Torch/NumPy ABI required',
    'Original full native extension build failed; no retry/patch',
    'Native CUDA MSDeformAttn disagrees with original PyTorch reference',
    'Original CPU soft-NMS decay/removal/indices differ',
    'Independent artifact identity differs',
])
def test_native_failure_reports_only_known_original_requirement(gate, message):
    assert gate.native_failure_requirement(ValueError(message))==message


def test_native_failure_redacts_unknown_type_args_and_string_subclasses(gate):
    class ValueErrorSubclass(ValueError): pass
    class StringSubclass(str): pass
    message='Frozen unchanged base dependencies/headers required'
    for error in (ValueError('secret=DO_NOT_LOG'), RuntimeError(message), ValueError(message, 'secret'), ValueError(),
                  ValueErrorSubclass(message), ValueError(StringSubclass(message)), ValueError({'secret':'DO_NOT_LOG'})):
        assert gate.native_failure_requirement(error)=='redacted_non_allowlisted_error'


def test_native_requirement_allowlist_is_literal_and_covers_native_guards(gate):
    tree=ast.parse((REPO/gate.HELPERS[0]).read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='NATIVE_REQUIREMENTS'for t in n.targets))
    assert isinstance(assignment.value,ast.Call) and isinstance(assignment.value.func,ast.Name) and assignment.value.func.id=='frozenset'
    assert set(ast.literal_eval(assignment.value.args[0]))==gate.NATIVE_REQUIREMENTS
    native_functions={'helpers','source','check','data_proof','native_proof','versions','compile_native','native_ops','operator_cases'}
    for function in (n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in native_functions):
        for call in ast.walk(function):
            if isinstance(call,ast.Call)and isinstance(call.func,ast.Name)and call.func.id=='require':
                assert isinstance(call.args[1],ast.Constant) and call.args[1].value in gate.NATIVE_REQUIREMENTS


@pytest.mark.parametrize('known', [True,False])
def test_native_main_failure_receipt_uses_redacted_requirement_only(gate, monkeypatch, tmp_path, known):
    p=json.loads((REPO/gate.PROTOCOL).read_bytes()); out=tmp_path/p['output'];out.mkdir(parents=True);records=[]
    message='Frozen unchanged base dependencies/headers required' if known else 'secret=DO_NOT_LOG'
    def failing_compile(*args): raise ValueError(message)
    class Acquisition:
        @staticmethod
        def write_receipt(path,value,started,deadline): records.append((path,value))
    monkeypatch.setattr(gate,'ROOT',tmp_path);monkeypatch.setattr(gate,'helpers',lambda code:(None,None,Acquisition))
    monkeypatch.setattr(gate,'protocol',lambda rt,code:p);monkeypatch.setattr(gate,'compile_native',failing_compile)
    monkeypatch.setenv('WR_CODE',str(tmp_path/'code'));monkeypatch.setenv('WR_CODE_REVISION','a'*40);monkeypatch.setenv('WR_RUNTIME_DEADLINE','100')
    monkeypatch.setattr(sys,'argv',['gate','--compile','--proof-bytes','12','--proof-sha256','b'*64])
    with pytest.raises(ValueError): gate.main()
    assert len(records)==1 and records[0][0]==out/'compile.json'
    value=records[0][1]
    assert value['status']=='fail' and value['error_type']=='ValueError' and all(value[k]is False for k in gate.FLAGS)
    assert value['requirement']==(message if known else 'redacted_non_allowlisted_error')
    assert 'DO_NOT_LOG' not in json.dumps(value) and 'traceback' not in value


@pytest.mark.parametrize("build", [True, False])
def test_only_image_packaging_uses_actual_local_builder_no_pull(gate, monkeypatch, tmp_path, build):
    calls=[]
    class Result: returncode=0
    def run(args, **kwargs):
        calls.append((args, kwargs['env']))
        return Result()
    monkeypatch.setattr(gate.subprocess, 'run', run)
    args=['docker', 'build', '--pull=false', '--network', 'none'] if build else ['docker', 'start', '-a', 'owned']
    gate.command(args, gate.time.monotonic()+10, log=tmp_path/'test.log')
    assert calls[0][1] == (dict(gate.SAFE_ENV, DOCKER_BUILDKIT='0') if build else gate.SAFE_ENV)
    source=(REPO/gate.HELPERS[0]).read_text()
    assert "command(['docker', 'build', '--pull=false', '--network', 'none'" in source
    assert 'DOCKER_BUILDKIT' not in gate.SAFE_ENV
