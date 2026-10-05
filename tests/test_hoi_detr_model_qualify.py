"""Tiny manufactured source/proof/ABI controls; no Torch/install/native/network."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location('wr_hoi_model_tests', REPO/'infra/hoi_detr_model_qualify.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def helper(name):
    spec = importlib.util.spec_from_file_location('wr_model_test_'+name, REPO/('infra/'+name+'.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture
def rt(): return helper('mediapipe_cpu_runtime_verify')


@pytest.fixture
def p(gate): return json.loads((REPO/gate.PROTOCOL).read_bytes())


def pin(raw): return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def test_protocol_exact_null_dependency_is_not_qualification(gate, rt, p):
    assert pin((REPO/gate.PROTOCOL).read_bytes()) == gate.PROTOCOL_PIN
    assert p['runtime']['report'] == dict(bytes=13248,sha256='906bf42aaeaf6612586de1b93c4039a5f969233022f346d64ae84a975debb445') and p['budget_seconds'] == 1800
    assert p['gpu_memory'] == '64g' and all(p[k] is False for k in gate.FLAGS)
    assert all(x in gate.HELPERS for x in ('infra/hoi_detr_runtime_verify.py', 'src/world_reward/hoi_detr_observations.py'))
    p=copy.deepcopy(p);p['runtime']['report']=None
    with pytest.raises(ValueError, match='not yet available'):
        gate.authenticate_runtime(SimpleNamespace(pinned=lambda *_: pytest.fail('No unpinned receipt read')), None, p, 1)


def test_protocol_declares_exact_publisher_and_native_sources(p):
    assert p['fairscale']['bytes'] == 266261 and p['fairscale']['publisher_license']['bytes'] == 1739
    assert p['fairscale']['build_cuda_extensions'] == '0'
    assert p['native_config']['original_load_from'].endswith('/epoch_5.pth')
    assert p['native_config']['inference_overrides'] == {'load_from': None, 'model.train_cfg': None}
    assert p['procedural_RGB'] == dict(height=96, width=128, original_frame_index=0, generator='uint8_xy_integer_ramp_v1')


def test_exact_nonnumeric_source_path_removal_preserves_AST(gate, p):
    path = p['compatibility_patch']['remove_exact_sys_path_append']
    raw = ("import sys\n\nsys.path.append('"+path+"')\nimport torch\ndef loss(x):\n return x*2+1\n").encode()
    derived, record = gate.remove_source_path(raw, path)
    assert derived == raw.replace(("sys.path.append('"+path+"')\n").encode(), b'\n')
    assert record['original'] == pin(raw) and record['derived'] == pin(derived) and record['remaining_AST_unchanged']
    assert ast.dump(ast.parse(derived).body[-1]) == ast.dump(ast.parse(raw).body[-1])


@pytest.mark.parametrize('raw', [b'import sys\n', b"sys.path.append('/wrong')\n", b"sys.path.append('/path', other=1)\n", b"sys.path.append('/path'); x=2\n", b"sys.path.append('/path')\nsys.path.append('/path')\n", b"sys.path.append(\n '/path'\n)\n"])
def test_source_patch_abstains_other_shapes(gate, raw):
    with pytest.raises(ValueError): gate.remove_source_path(raw, '/path')


def cfg(p):
    return dict(load_from=p['native_config']['original_load_from'], resume_from=None,
                model=dict(type='CoDETR', train_cfg=[dict(a=1)], backbone=dict(pretrained=None, init_cfg=None, use_act_checkpoint=True), query_head=dict(num_query=1500, num_classes=3), rpn_head=dict(type='RPN'), roi_head=[dict(type='ROI')], bbox_head=[dict(type='ATSS')]), data=dict(test=dict(ann_file='UNUSED_PRIVATE_TEXT')))


def test_native_config_only_explicit_inference_overrides(gate, p):
    c = cfg(p); before = copy.deepcopy(c); result = gate.configuration_policy(c, p['native_config'])
    assert c['load_from'] is None and c['model']['train_cfg'] is None and result['downloads'] is False
    before['load_from'] = None; before['model']['train_cfg'] = None
    assert c == before and 'rpn_head' in c['model'] and 'roi_head' in c['model'] and 'bbox_head' in c['model']


@pytest.mark.parametrize('field', ['pretrained', 'init_cfg', 'resume_from', 'load_from'])
def test_active_nested_weight_loader_is_forbidden(gate, p, field):
    c = cfg(p); c['model']['roi_head'][0][field] = 'http://foreign/model'
    with pytest.raises(ValueError, match='loader forbidden'): gate.configuration_policy(c, p['native_config'])


@pytest.mark.parametrize('change', ['locator', 'checkpointing', 'query', 'classes', 'override'])
def test_native_config_drift_rejected(gate, p, change):
    c = cfg(p)
    if change == 'locator': c['load_from'] = '/other/epoch_5.pth'
    elif change == 'checkpointing': c['model']['backbone']['use_act_checkpoint'] = False
    elif change == 'query': c['model']['query_head']['num_query'] = 32
    elif change == 'classes': c['model']['query_head']['num_classes'] = 1
    else: p['native_config']['inference_overrides']['load_from'] = 'anything'
    with pytest.raises(ValueError): gate.configuration_policy(c, p['native_config'])


class Tensor:
    def __init__(self, shape=(2,), dtype='float32', finite=True): self.shape, self.dtype, self.finite = shape, dtype, finite
    def numel(self): return 2
    def element_size(self): return 4


class Model:
    def __init__(self): self.expected = {'native.weight': Tensor()}; self.calls = []
    def state_dict(self): return self.expected
    def load_state_dict(self, values, strict):
        self.calls.append((values, strict)); return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def fake_torch():
    return SimpleNamespace(Tensor=Tensor, isfinite=lambda x: SimpleNamespace(all=lambda: x.finite))


def test_all_native_state_safe_strict_no_key_remapping(gate):
    model = Model(); state = {'native.weight': Tensor()}
    result = gate.strict_checkpoint(model, {'state_dict': state, 'meta': {}}, fake_torch())
    assert model.calls == [(state, True)] and result['keys'] == 1 and result['strict'] and result['weights_only']


@pytest.mark.parametrize('fault', ['missing', 'extra', 'moduleprefix', 'dtype', 'shape', 'nontensor', 'nan', 'nostate'])
def test_checkpoint_fails_before_partial_load(gate, fault):
    model = Model(); state = {'native.weight': Tensor()}; c = {'state_dict': state}
    if fault == 'missing': state.clear()
    elif fault == 'extra': state['other'] = Tensor()
    elif fault == 'moduleprefix': c['state_dict'] = {'module.native.weight': Tensor()}
    elif fault == 'dtype': state['native.weight'] = Tensor(dtype='float16')
    elif fault == 'shape': state['native.weight'] = Tensor(shape=(3,))
    elif fault == 'nontensor': state['native.weight'] = 1
    elif fault == 'nan': state['native.weight'] = Tensor(finite=False)
    else: c = {}
    with pytest.raises(ValueError): gate.strict_checkpoint(model, c, fake_torch())
    assert not model.calls


def tar_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as z:
        for name, raw, kind in entries:
            t = tarfile.TarInfo(name); t.size = len(raw); t.mode = 0o644
            if kind == 'link': t.type = tarfile.SYMTYPE; t.linkname = '/foreign'; t.size = 0
            if kind == 'fifo': t.type = tarfile.FIFOTYPE; t.size = 0
            z.addfile(t, io.BytesIO(raw) if t.isfile() else None)
    return buffer.getvalue()


def fairscale_fixture(gate, rt, tmp_path):
    target = tmp_path/'source'; target.mkdir(); archive = tmp_path/'original.tar.gz'; license = b'Original BSD grant\n'
    archive.write_bytes(tar_bytes([('fairscale-0.4.13/LICENSE', license, 'file'), ('fairscale-0.4.13/fairscale/__init__.py', b'# original tiny fixture\n', 'file')]))
    c = dict(publisher_license=pin(license), maximum_expanded_bytes=4096, maximum_members=8)
    return target, archive, c


def test_full_small_sdist_licensed_without_install(gate, rt, tmp_path):
    target, archive, c = fairscale_fixture(gate, rt, tmp_path)
    gate.unpack_fairscale(rt, helper('hoi_detr_acquire'), archive, target, c, 1, lambda _: None)
    assert (target/'LICENSE').read_bytes() == b'Original BSD grant\n'
    assert (target/'fairscale/__init__.py').exists()


@pytest.mark.parametrize('fault', ['traversal', 'absolute', 'duplicate', 'link', 'fifo', 'prefix', 'license', 'members', 'expanded'])
def test_sdist_safe_full_shape_and_grant(gate, rt, tmp_path, fault):
    target, archive, c = fairscale_fixture(gate, rt, tmp_path)
    entries = [('fairscale-0.4.13/LICENSE', b'Original BSD grant\n', 'file')]
    if fault in ('traversal', 'absolute', 'prefix'): entries.append(({'traversal': 'fairscale-0.4.13/../bad', 'absolute': '/bad', 'prefix': 'other/file'}[fault], b'a', 'file'))
    elif fault == 'duplicate': entries *= 2
    elif fault in ('link', 'fifo'): entries.append(('fairscale-0.4.13/bad', b'', fault))
    elif fault == 'license': c['publisher_license']['sha256'] = '0'*64
    elif fault == 'members': c['maximum_members'] = 0
    else: c['maximum_expanded_bytes'] = 1
    archive.write_bytes(tar_bytes(entries))
    with pytest.raises(ValueError): gate.unpack_fairscale(rt, helper('hoi_detr_acquire'), archive, target, c, 1, lambda _: None)


def test_overlay_full_inventory_roundtrip_empty_file_and_mutation(gate, rt, tmp_path):
    folder = tmp_path/'overlay'; folder.mkdir(); (folder/'a.py').write_bytes(b'a=1\n'); (folder/'empty.py').write_bytes(b'')
    gate.seal_tree(folder); rows = gate.inventory(rt, folder)
    gate.check_inventory(rt, folder, list(reversed(rows)))
    (folder/'a.py').chmod(0o644); (folder/'a.py').write_bytes(b'a=2\n'); (folder/'a.py').chmod(0o444)
    with pytest.raises(ValueError): gate.check_inventory(rt, folder, rows)


@pytest.mark.parametrize('fault', ['symlink', 'hardlink', 'writable', 'foreign'])
def test_overlay_inventory_rejects_alias_and_extra(gate, rt, tmp_path, fault):
    folder = tmp_path/'overlay'; folder.mkdir(); f = folder/'x'; f.write_bytes(b'x'); gate.seal_tree(folder); rows = gate.inventory(rt, folder)
    folder.chmod(0o755)
    if fault == 'symlink': (folder/'y').symlink_to(f)
    elif fault == 'hardlink': os.link(f, folder/'y')
    elif fault == 'writable': f.chmod(0o644)
    else: (folder/'y').write_bytes(b'y'); (folder/'y').chmod(0o444)
    with pytest.raises(ValueError): gate.check_inventory(rt, folder, rows)


def test_container_scope_overlay_vs_full_model(gate, p, tmp_path):
    code, out = tmp_path/'code', tmp_path/'out'; image = 'sha256:'+'e'*64
    proof = dict(runtime=dict(image=dict(Id=image)), pin=dict(bytes=123, sha256='1'*64))
    for phase in ('overlay', 'model'):
        name, cid, mounts, args = gate.container_plan(code, out, 'a'*40, p, proof, phase, 1800)
        assert name.endswith('a'*12) and cid.name == phase+'.cid'
        assert '--network' in args and args[args.index('--network')+1] == 'none' and '--read-only' in args and '--cap-drop' in args
        assert args[args.index('--memory')+1] == ('64g' if phase == 'model' else '16g')
        assert ('--gpus' in args) == (phase == 'model') and (phase != 'model' or 'driver=nvidia,count=all' in args)
        assert not any('datasets' in str(a) or 'labels' in str(a) or 'RGB' in str(a) for a, _, _ in mounts)
        if phase == 'model':
            assert (gate.DATA/'weights/epoch_5.pth', gate.DATA/'weights/epoch_5.pth', True) in mounts
            assert (out/'.overlay', out/'.overlay', True) in mounts
            assert not any(a == gate.DATA for a, _, _ in mounts)
        else: assert all(a != gate.DATA/'weights/epoch_5.pth' for a, _, _ in mounts)


def value_for(mounts, name, rev, image, phase):
    return dict(Image=image, Name='/'+name, Config=dict(User='1000:1000', Labels={'world-reward.job': 'run_hoi_detr_model_qualify', 'world-reward.revision': rev}),
        HostConfig=dict(NetworkMode='none', ReadonlyRootfs=True, Privileged=False, CapDrop=['ALL'], SecurityOpt=['no-new-privileges'], Binds=None, Devices=[], VolumesFrom=None,
            DeviceRequests=[] if phase == 'overlay' else [dict(Driver='nvidia', Count=-1, Capabilities=[['gpu']])]),
        Mounts=[dict(Type='bind', Source=str(a), Destination=str(b), RW=not ro) for a, b, ro in mounts])


@pytest.mark.parametrize('fault', [None, 'owner', 'image', 'label', 'network', 'writable', 'gpu', 'extra', 'volume'])
def test_inspect_sandbox_is_actual_not_self_declared(gate, p, tmp_path, fault):
    image = 'sha256:'+'e'*64; rev = 'a'*40
    name, _, mounts, _ = gate.container_plan(tmp_path/'code', tmp_path/'out', rev, p, dict(runtime=dict(image=dict(Id=image)), pin=pin(b'x')), 'model', 1800)
    v = value_for(mounts, name, rev, image, 'model')
    if fault == 'owner': v['Config']['User'] = '0'
    elif fault == 'image': v['Image'] = 'sha256:'+'f'*64
    elif fault == 'label': v['Config']['Labels']['world-reward.job'] = 'foreign'
    elif fault == 'network': v['HostConfig']['NetworkMode'] = 'host'
    elif fault == 'writable': v['HostConfig']['ReadonlyRootfs'] = False
    elif fault == 'gpu': v['HostConfig']['DeviceRequests'][0]['Driver'] = ''
    elif fault == 'extra': v['Mounts'].append(dict(Type='bind', Source='/private', Destination='/private', RW=False))
    elif fault == 'volume': v['Mounts'].append(dict(Type='volume'))
    if fault is None: gate.validate_container(v, mounts, name, rev, image, 'model')
    else:
        with pytest.raises(ValueError): gate.validate_container(v, mounts, name, rev, image, 'model')


@pytest.mark.parametrize('fault', [None, 'foreign', 'rmfail', 'still_present', 'badcid', 'missing'])
def test_exact_owned_cleanup_no_daemon_absence_guess(gate, tmp_path, fault, monkeypatch):
    cid = 'c'*64; file = tmp_path/'model.cid'; file.write_text(cid+'\n'); image='sha256:'+'e'*64; calls=[]
    monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    class Runtime:
        def command(self, args, deadline):
            calls.append(args)
            if args[1] == 'inspect': return image+'|/name|'+('foreign' if fault == 'foreign' else gate.ENTRY)+'|rev'
            if args[1] == 'rm':
                if fault == 'rmfail': raise ValueError('daemon failure')
                return cid
            n = sum(x[1] == 'ps' for x in calls)
            return cid if n == 1 or fault == 'still_present' else ''
        def absent(self, name, deadline): calls.append(['absent', name])
    if fault == 'badcid': file.write_text('invalid')
    if fault == 'missing': file.unlink()
    # Production CID belongs to root; local fixtures may be an unprivileged user.
    lstat_real = Path.lstat
    if file.exists():
        size = lstat_real(file).st_size
        monkeypatch.setattr(Path, 'lstat', lambda self: SimpleNamespace(st_mode=stat.S_IFREG|0o644, st_nlink=1, st_uid=0, st_size=size) if self == file else lstat_real(self))
    if fault in ('foreign', 'rmfail', 'still_present', 'badcid'):
        with pytest.raises(ValueError): gate.cleanup(Runtime(), file, 'name', 'rev', image, 1)
        assert fault != 'foreign' or not any(x[1] == 'rm' for x in calls)
    else: gate.cleanup(Runtime(), file, 'name', 'rev', image, 1)


def test_native_entry_has_safe_load_and_original_seam_not_fake_output(gate):
    tree = ast.parse((REPO/'infra/hoi_detr_model_qualify.py').read_bytes())
    gpu = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gpu_model')
    text = ast.unparse(gpu)
    assert 'weights_only=True' in text and "map_location='cpu'" in text
    assert 'infer_hoi_detr_frame(model, rgb' in text and 'NativeHOIOperations' in text
    assert 'build_detector(cfg.model' in text and 'model.to(device=' in text and '.eval()' in text
    assert "import_custom_modules=False" in text and "LoadImageFromWebcam" in text and 'rgb[:, :, ::-1]' in text
    assert 'init_weights' not in text and 'mmdet.apis' not in [n.module for n in ast.walk(gpu) if isinstance(n, ast.ImportFrom)]
    assert not any(isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names) for n in tree.body)


def test_host_registers_cleanup_before_create_and_never_builds_image():
    tree = ast.parse((REPO/'infra/hoi_detr_model_qualify.py').read_bytes()); host = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    text = ast.unparse(host)
    assert text.index('containers.append(') < text.index('runtime.command(args, deadline)')
    assert 'docker build' not in text and 'image rm' not in text
    assert 'os.O_RDONLY | os.O_NOFOLLOW' in text and 'fcntl.LOCK_EX | fcntl.LOCK_NB' in text
    assert text.index('runtime.remove_owned_folder') < text.index("out / 'report.json'")


def test_shell_source_only_linux_exact_namespace():
    s = (REPO/'infra/run_hoi_detr_model_qualify.sh').read_text()
    assert 'run_hoi_detr_model_qualify/code' in s and '1860s' in s and 'env -i' in s
    assert 'docker' not in s.replace('Source closure:', '')
    assert '--kill-after=5s' in s and 'world-reward-ncc-h100-02' in s


@pytest.mark.parametrize('error, expected', [
    (ValueError('Strict all model state keys/shapes/dtypes required'), 'Strict all model state keys/shapes/dtypes required'),
    (ValueError('Inclusive native runtime deadline'), 'Inclusive native runtime deadline'),
    (ValueError('/secret/path?token=NOT_LOGGED'), 'redacted_non_allowlisted_error'),
    (RuntimeError('Offline original FairScale wheel build failed'), 'redacted_non_allowlisted_error'),
    (ValueError('Offline original FairScale wheel build failed', 'extra'), 'redacted_non_allowlisted_error'),
])
def test_failure_reason_fixed_literal_only(gate, error, expected):
    runtime = helper('hoi_detr_runtime_verify')
    assert gate.failure_requirement(error, runtime) == expected


def test_child_failure_captures_known_reason_and_pass_disposes_logs():
    text = (REPO/'infra/hoi_detr_model_qualify.py').read_text()
    assert "requirement=failure_requirement(exc, runtime)" in text
    assert "for name in ('overlay.log', 'model.log')" in text
    host = ast.unparse(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'run'))
    assert host.index("for name in ('overlay.log', 'model.log')") < host.index("acq.write_receipt(out / 'report.json'")


def test_original_failsafe_writer_demotes_after_final_fsync(tmp_path, monkeypatch):
    acq = helper('hoi_detr_acquire'); report = dict(status='pass', phase='complete'); path = tmp_path/'report.json'
    monkeypatch.setattr(acq, 'check', lambda _: (_ for _ in ()).throw(ValueError('Inclusive acquisition deadline')))
    assert acq.write_receipt(path, report, 1, 2)['status'] == 'fail'
    stored = json.loads(path.read_bytes()); assert stored['status'] == 'fail' and stored['phase'] == 'receipt_sealing'
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError): acq.write_receipt(path, dict(status='fail'), 1, 2)


def test_inventory_package_and_distinfo_have_one_complete_canonical_order(gate,rt,tmp_path):
    root=tmp_path/'site';root.mkdir()
    for name in ('fairscale/__init__.py','fairscale-0.4.13.dist-info/INSTALLER','fairscale/z.py','fairscale/nn/a.py'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'Original');path.chmod(0o444)
    gate.seal_tree(root);rows=gate.inventory(rt,root)
    assert [r['file']for r in rows]==sorted(['fairscale/__init__.py','fairscale-0.4.13.dist-info/INSTALLER','fairscale/z.py','fairscale/nn/a.py'])
    gate.check_inventory(rt,root,list(reversed(rows)))
    with pytest.raises(ValueError):gate.check_inventory(rt,root,rows[:-1])
    root.chmod(0o755);p=root/'fairscale/z.py';p.chmod(0o644);p.write_bytes(b'Changed');p.chmod(0o444)
    with pytest.raises(ValueError):gate.check_inventory(rt,root,rows)


def test_missing_terminaltables_has_pinned_pure_overlay_not_base_install(p):
    w=p['import_wheels'][0]
    assert w['name']=='terminaltables' and w['version']=='3.1.10' and w['bytes']==15155
    assert w['publication_date']<'2026-09-30' and w['license']=='MIT'
    assert w['publisher_license']['bytes']==1065
    assert w['sha256']=='e4fdc4179c9e4aab5f674d80f09d76fa436b96fdc698a8505e0a36bf0804a874'


def test_all_native_imports_are_qualified_cpu_before_gpu_lease(gate):
    source=(REPO/gate.HELPERS[0]).read_text()
    cpu=source[source.index('def cpu_overlay('):source.index('def gpu_model(')]
    assert "for name in ('fairscale.nn.checkpoint', 'mmdet.models.builder', 'projects.models', 'mmdet.datasets.pipelines')" in cpu
    assert 'full_import_closure_qualified=True' in cpu
    assert 'model = build_detector' not in cpu and 'torch.load' not in cpu
    assert 'import_custom_modules=False' in cpu


class NativeEMAStub:
    def __init__(self,momentum):self.skip_buffers=False;self.checkpoint=None;self.momentum=momentum
    def before_run(self,runner):
        self.param_ema_buffer={k:'ema_'+k.replace('.','_')for k in list(runner.model.expected)}
        for key,name in self.param_ema_buffer.items():runner.model.expected[name]=Tensor()


def test_complete_original_ema_backup_schema_registered_no_key_drop(gate,p):
    model=Model();original=model.expected['native.weight']
    result=gate.register_original_checkpoint_buffers(model,SimpleNamespace(custom_hooks=[p['checkpoint_buffers']['native_hook']]),NativeEMAStub,p['checkpoint_buffers'])
    assert set(model.state_dict())=={'native.weight','ema_native_weight'}
    assert model.expected['native.weight'] is original and not model.calls
    assert result['original_state_keys']==result['registered_backup_keys']==1 and result['full_state_keys']==2
    assert result['checkpoint_keys_discarded']==0 and not result['ema_swapped']
    state={k:Tensor()for k in model.expected}
    gate.strict_checkpoint(model,{'state_dict':state},fake_torch())
    assert model.calls==[(state,True)]


@pytest.mark.parametrize('fault',['collision','hook','swap','registration','skip','resume'])
def test_native_ema_registration_rejects_ambiguous_or_changed_schema(gate,p,fault):
    model=Model();c=copy.deepcopy(p['checkpoint_buffers']);cfg=SimpleNamespace(custom_hooks=[c['native_hook']])
    hook=NativeEMAStub
    if fault=='collision':model.expected['native_weight']=Tensor()
    elif fault=='hook':cfg.custom_hooks=[dict(type='OtherHook')]
    elif fault=='swap':c['swap_or_update']=True
    elif fault=='registration':c['registration_only']=False
    else:
        class Bad(NativeEMAStub):
            def __init__(self,momentum):
                super().__init__(momentum)
                if fault=='skip':self.skip_buffers=True
                else:self.checkpoint='foreign'
        hook=Bad
    with pytest.raises(ValueError):gate.register_original_checkpoint_buffers(model,cfg,hook,c)
    assert not model.calls
