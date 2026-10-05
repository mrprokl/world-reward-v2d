"""Manufactured native-shaped callbacks only; no Torch, rig, GPU or download."""
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'infra'))
import mhr_direct_bridge as q


def model_fixture():
    names = [f'native_parameter_{i}' for i in range(249)]
    columns = (11, 23, 71, 103)
    for name, col in zip(q.PARAMETERS, columns):
        names[col] = name
    joints = ['root', 'l_elbow', 'l_wrist', 'l_index1', 'l_index2', 'l_index3',
              'r_elbow', 'r_wrist', 'r_index1', 'r_index2', 'r_index3']
    joints += [f'other_{j}' for j in range(116)]
    parents = np.asarray([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9] + [0] * 116, np.int64)
    transform = np.zeros((889, 249), np.float32)
    for col, joint in zip(columns, (1, 6, 4, 9)):
        transform[7 * joint + 5, col] = 1
    idx = np.zeros((18439, 8), np.int64)
    idx[:, 0] = np.arange(18439) % 10 + 1
    weights = np.zeros(idx.shape, np.float32); weights[:, 0] = 1
    arrays = dict(bounds=np.tile(np.array([-1, 1], np.float32), (249, 1)), transform=transform,
                  parents=parents, faces=np.tile(np.array([[0, 1, 2]], np.int64), (36874, 1)),
                  lbs_indices=idx, lbs_weights=weights)
    methods = {}
    returns = dict(forward='Tuple[Tensor, Tensor]', get_parameter_names='List[str]', get_joint_names='List[str]',
                   get_parameter_transform='Tensor', get_parameter_limits='Tensor', get_num_identity_blendshapes='int',
                   get_num_face_expression_blendshapes='int')
    for name, result in returns.items():
        args = [('self', '__torch__.selected.MHRDemo')]
        if name == 'forward':
            args += [('identity_coeffs', 'Tensor'), ('model_parameters', 'Tensor'),
                     ('face_expr_coeffs', 'Tensor'), ('apply_correctives', 'bool')]
        methods[name] = SimpleNamespace(schema=SimpleNamespace(arguments=[SimpleNamespace(name=k, type=v) for k, v in args],
            returns=[SimpleNamespace(type=result)]), code=f'def {name}(self):\n return self.original\n', graph='original graph ' + name)
    model = SimpleNamespace(_c=SimpleNamespace(_get_method=lambda name: methods[name]),
        get_parameter_names=lambda: names, get_joint_names=lambda: joints,
        get_parameter_transform=lambda: arrays['transform'], get_parameter_limits=lambda: arrays['bounds'],
        get_num_identity_blendshapes=lambda: 45, get_num_face_expression_blendshapes=lambda: 72,
        get_lbsw=lambda: (arrays['lbs_indices'], arrays['lbs_weights']),
        character_torch=SimpleNamespace(parameter_transform=SimpleNamespace(parameter_names=names, parameter_transform=transform),
            skeleton=SimpleNamespace(joint_names=joints, joint_parents=parents), mesh=SimpleNamespace(faces=arrays['faces'])))
    return model, arrays, methods


def test_fixed_names_not_guessed_columns_literal_full_bounds_and_no_alias():
    model, arrays, _ = model_fixture()
    meta = q.inspect_metadata(model); batch = q.named_batch(meta)
    assert [r['column'] for r in meta['controls']] == [11, 23, 71, 103]
    assert batch['parameters'].shape == (9, 204)
    for j, record in enumerate(meta['controls']):
        assert np.array_equal(batch['parameters'][1+2*j:3+2*j, record['column']], np.array(q.STEPS, np.float32))
    assert np.count_nonzero(batch['parameters']) == 8
    assert not batch['identity'].any() and not batch['expression'].any()
    for value in (*meta['arrays'].values(), batch['parameters'], batch['identity'], batch['expression']):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    saved = meta['arrays']['bounds'].copy(); arrays['bounds'][0, 0] = -.5
    assert np.array_equal(saved, meta['arrays']['bounds'])


@pytest.mark.parametrize('fault', ['schema', 'methodsource', 'names', 'cycle', 'jointroot', 'wrongrotation',
                                  'oppositeside', 'nosupport', 'nan', 'lbs', 'submodule', 'faces'])
def test_metadata_malformed_or_unknown_mapping_stops(fault):
    model, arrays, methods = model_fixture()
    if fault == 'schema': methods['forward'].schema.arguments[2].name = 'guessed_pose'
    elif fault == 'methodsource': methods['forward'].code = 'x' * 100001
    elif fault == 'names': model.get_parameter_names()[11] = 'unknown_elbow'
    elif fault == 'cycle': arrays['parents'][2] = 3
    elif fault == 'jointroot': model.get_joint_names()[2] = 'unknown_wrist'
    elif fault == 'wrongrotation': arrays['transform'][7+0, 11] = 1
    elif fault == 'oppositeside': arrays['transform'][6*7+5, 11] = 1
    elif fault == 'nosupport': arrays['lbs_indices'][:] = 0
    elif fault == 'nan': arrays['bounds'][40, 0] = np.nan
    elif fault == 'lbs': arrays['lbs_weights'][0, 0] = 0
    elif fault == 'submodule': model.character_torch.parameter_transform.parameter_transform = arrays['transform'] + 1
    else: arrays['faces'][0, 0] = 18439
    with pytest.raises((ValueError, KeyError)):
        q.inspect_metadata(model)


@pytest.mark.parametrize('column,value', [(11, -.001), (147, .001), (204, .001)])
def test_unsupported_fixed_positive_or_native_zero_reference_stops_without_clipping(column, value):
    model, arrays, _ = model_fixture()
    if column == 11: arrays['bounds'][column, 1] = value
    else: arrays['bounds'][column] = [value, 1.]
    with pytest.raises(ValueError, match='literal bounds'):
        q.named_batch(q.inspect_metadata(model))


class ModelCall:
    def __init__(self, fixture, *, fault=None):
        self.__dict__.update(vars(fixture)); self.calls = []; self.fault = fault

    def __call__(self, identity, parameters, expression, correctives):
        self.calls.append((identity.copy(), parameters.copy(), expression.copy(), correctives))
        v = np.zeros((9, 18439, 3), np.float32)
        v[:, :, 0] = np.linspace(-20, 20, 18439)
        s = np.zeros((9, 127, 8), np.float32); s[:, :, 6:8] = 1
        meta = q.inspect_metadata(self)
        for j, record in enumerate(meta['controls']):
            for k, step in enumerate(q.STEPS):
                row = 1 + 2*j + k; s[row, record['global_closure'], 3] = np.sin(step / 2)
                s[row, record['global_closure'], 6] = np.cos(step / 2)
                v[row, :, 2] = step
        if self.fault == 'outside': s[1, 126, 0] = .001
        elif self.fault == 'scale': s[1, 4, 7] = 1.001
        elif self.fault == 'nomotion': s[2] = s[0]
        elif self.fault == 'nan': v[1, 0, 0] = np.nan
        elif self.fault == 'shape': v = v[:8]
        elif self.fault == 'changed': self.character_torch.skeleton.joint_parents[126] = 1
        elif self.fault == 'inputmutation': parameters[0, 0] = 3
        elif self.fault == 'wrongmaterial':
            first = meta['controls'][0]
            idx, w = meta['arrays']['lbs_indices'], meta['arrays']['lbs_weights']
            supported = np.where(np.isin(idx, first['global_closure']), w, 0.).sum(1) > 0
            v[1:3, supported] = v[0, supported]
        return v, s


def torch_fixture():
    calls = []
    return SimpleNamespace(float32=np.float32, no_grad=nullcontext,
        as_tensor=lambda value, **kwargs: value,
        cuda=SimpleNamespace(synchronize=lambda: calls.append('sync'))), calls


def test_one_original_batched_native_call_no_second_reference_or_extra_methods():
    model = ModelCall(model_fixture()[0]); torch, sync = torch_fixture(); checks = []
    record = q.run_control(model, torch, check=lambda: checks.append('check'))
    assert len(model.calls) == 1 and sync == ['sync'] and len(checks) == 4
    identity, parameters, expression, correctives = model.calls[0]
    assert correctives is True and parameters.shape == (9, 204) and identity.shape == (9, 45) and expression.shape == (9, 72)
    assert record['native_calls'] == 1 and record['decoded_frames'] == 9
    assert record['SAM_checkpoint_used'] is False and record['adoption'] is False
    assert len(record['controls']) == 4 and all(r['vertex_motion_max_m'] > 0 for r in record['controls'])
    assert all(record[k] == 0 for k in ('render_calls', 'tracker_calls', 'optimizer_calls'))


@pytest.mark.parametrize('fault', ['outside', 'scale', 'nomotion', 'nan', 'shape', 'changed', 'inputmutation', 'wrongmaterial'])
def test_native_invariant_failure_stops_one_call_no_retry(fault):
    model = ModelCall(model_fixture()[0], fault=fault)
    with pytest.raises(ValueError):
        q.run_control(model, torch_fixture()[0], check=lambda: None)
    assert len(model.calls) == 1


def test_all_boundchecks_before_first_native_call():
    model = ModelCall(model_fixture()[0]); model.get_parameter_limits()[103, 1] = 0
    with pytest.raises(ValueError, match='literal bounds'):
        q.run_control(model, torch_fixture()[0], check=lambda: None)
    assert model.calls == []


def test_source_import_is_stdlib_only_and_no_executable_loader():
    source = (ROOT / 'infra/mhr_direct_bridge.py').read_text(); tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import): assert all(a.name in {'hashlib', 'json'} for a in node.names)
        if isinstance(node, ast.ImportFrom): assert node.module in {'__future__', 'pathlib'}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in
                   {'jit.load', 'load', '_ensure_head'} for n in ast.walk(tree))
    subprocess.run([sys.executable, '-I', '-B', '-S', '-c',
        f"import runpy,sys;runpy.run_path({str(ROOT/'infra/mhr_direct_bridge.py')!r});assert 'numpy' not in sys.modules;assert 'torch' not in sys.modules"],
        check=True, capture_output=True)


def selected_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'root'; code = root / 'current'; code.mkdir(parents=True)
    out = root / 'results/mhr-official-release-license-v2'; out.mkdir(parents=True)
    producer = 'c41620005e8e9167529045d3cae072baa8b79c8c'
    old = root / 'jobs' / producer / 'acquire_weights/code'; old.mkdir(parents=True)
    (old.parent / 'source-sha256').write_text('a' * 64 + '\n')
    model = root / 'weights/mhr/mhr_model.pt'; model.parent.mkdir(parents=True); model.write_bytes(b'fake tiny model')
    model_id = dict(bytes=model.stat().st_size, sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    monkeypatch.setattr(q, 'MODEL_ID', model_id)
    protocol = dict(model=dict(**model_id, existing_path='weights/mhr/mhr_model.pt', accepted_uids=[model.stat().st_uid]))
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): path.chmod(0o644)  # manufactured test files only
        path.write_text(json.dumps(value)); path.chmod(0o444)
        return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    protocol_id = save(code / q.PROTOCOL, protocol); monkeypatch.setattr(q, 'PROTOCOL_ID', protocol_id)
    source = dict(fixture='full original source')
    report = dict(stage='mhr_official_release_license_v2', status='pass', phase='complete', producer_revision=producer,
        existing_model=model_id, member_model=model_id, protocol_identity=protocol_id, selected_members_decoded=2,
        selected_expanded_bytes=696121606, selected_payloads_opened=['assets/LICENSE.txt', 'assets/mhr_model.pt'], inactive_payloads_opened=0,
        source_before=source, elapsed_seconds=5., asset_license_matches_primary_exactly=True, model_byte_identical=True,
        model_copy_written=False, model_member_extracted=False, source_rehashed_after=True, existing_model_rehashed_after=True,
        archive_rehashed_after=True, owned_archive_removed=True, models_loaded=False, packages_installed=False, gpu_used=False,
        dataset_read=False, sam_provenance_relabelled=False, competition_eligibility_verified=False, training_overlap_verified=False, adoption=False)
    files = {'report.json': save(out / 'report.json', report)}
    for name in ('release-metadata.json', 'asset-notice-0.txt', 'primary-LICENSE', 'primary-README.md'):
        files[name] = save(out / name, {'manufactured': name})
    pins = dict(schema='world_reward.mhr_official_release_selected_qualification_pins.v1', producer_revision=producer,
        status='pass', independent_saved_receipts_audit=True, source_archive_sha256='a'*64, files=files)
    pins_id = save(code / q.PINS, pins); monkeypatch.setattr(q, 'PINS_ID', pins_id); out.chmod(0o555)
    def identity(path, *args, **kwargs):
        path = Path(path); return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    rt = SimpleNamespace(canonical=lambda p: Path(p), source=lambda *args: source, identity=identity,
        pinned=lambda path, pin, cap: json.loads(Path(path).read_bytes()) if identity(path) == pin else (_ for _ in ()).throw(ValueError('pin')))
    return root, code, rt, report, pins, source, save


def test_selected_release_authentication_is_hash_only_and_rechecked(tmp_path, monkeypatch):
    root, code, rt, *_ = selected_fixture(tmp_path, monkeypatch)
    proof = q.authenticate_release(root, code, rt); q.recheck_release(proof, root, rt)
    assert proof['producer_revision'] == 'c41620005e8e9167529045d3cae072baa8b79c8c'
    assert len(proof['frozen']) == 8 and proof['model_path'].read_bytes() == b'fake tiny model'
    proof['model_path'].write_bytes(b'changed')
    with pytest.raises(ValueError): q.recheck_release(proof, root, rt)


@pytest.mark.parametrize('fault', ['report', 'model', 'originalsource', 'extra', 'archive', 'eligibility'])
def test_selected_failure_before_model_execution(tmp_path, monkeypatch, fault):
    root, code, rt, report, pins, source, save = selected_fixture(tmp_path, monkeypatch)
    out = root / 'results/mhr-official-release-license-v2'
    if fault in ('report', 'eligibility'):
        if fault == 'report': report['status'] = 'fail'
        else: report['competition_eligibility_verified'] = True
        pins['files']['report.json'] = save(out / 'report.json', report)
        monkeypatch.setattr(q, 'PINS_ID', save(code / q.PINS, pins))
    elif fault == 'model': (root / 'weights/mhr/mhr_model.pt').write_bytes(b'wrong')
    elif fault == 'originalsource': rt.source = lambda *args: {'wrong': 'source'}
    elif fault == 'extra': out.chmod(0o755); (out / 'unexpected').write_text('extra'); out.chmod(0o555)
    else: (root/'jobs'/pins['producer_revision']/'acquire_weights/source-sha256').write_text('b'*64+'\n')
    with pytest.raises(ValueError): q.authenticate_release(root, code, rt)
