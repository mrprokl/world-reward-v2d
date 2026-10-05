"""Tiny public arrays / mocked lifecycle; no model, media download, Docker or GPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

import rgb_endpoint_bank as p


def seal(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw); path.chmod(0o400)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def config():
    return json.loads((Path(__file__).resolve().parents[1]/p.CONFIG).read_bytes())


def inputs_fixture(tmp_path, monkeypatch):
    data = tmp_path/'inputs'; data.mkdir(); monkeypatch.setattr(p, 'DATA', data)
    images = []
    for index in (0, 2):
        name = f'image_{index:06d}.jpg'; pin = seal(data/name, b'opaque original JPEG'+bytes([index]))
        images.append(dict(image_id=f'{index+1:032x}', file=name, width=9, height=7, **pin))
    value = dict(schema='world_reward.rgb_proposal_inputs.v1', images=images)
    pin = seal(data/'manifest.json', p.encode(value))
    return data, value, pin


def test_public_projection_order_holes_allowed_original_files_only(tmp_path, monkeypatch):
    _, value, pin = inputs_fixture(tmp_path, monkeypatch)
    assert p.public_inputs(pin, 2) == value
    assert [x['file'] for x in value['images']] == ['image_000000.jpg', 'image_000002.jpg']


@pytest.mark.parametrize('fault', ['count', 'sha', 'extra', 'schema', 'split', 'name', 'duplicate', 'order', 'area', 'bool'])
def test_bad_public_projection_fails_before_runtime_model(tmp_path, monkeypatch, fault):
    data, value, pin = inputs_fixture(tmp_path, monkeypatch)
    count = 2
    if fault == 'count': count = 1
    elif fault == 'sha': pin = dict(pin, sha256='0'*64)
    elif fault == 'extra': seal(data/'reference.json', b'forbidden')
    else:
        if fault == 'schema': value['schema'] = 'cohort_with_GT'
        elif fault == 'split': value['images'][0]['split'] = 'DEV'
        elif fault == 'name': value['images'][0]['file'] = '../outside.jpg'
        elif fault == 'duplicate': value['images'][1]['image_id'] = value['images'][0]['image_id']
        elif fault == 'order': value['images'].reverse()
        elif fault == 'area': value['images'][0]['width'] = 1 << 25
        elif fault == 'bool': value['images'][0]['height'] = True
        (data/'manifest.json').chmod(0o600); pin = seal(data/'manifest.json', p.encode(value))
    with pytest.raises(ValueError): p.public_inputs(pin, count)


def test_config_actual_qualification_pins_reused_helper_bytes():
    cfg = config(); root = Path(__file__).resolve().parents[1]
    assert cfg['budget_seconds'] == 600 and cfg['maximum_images'] == 64
    assert cfg['model_loads'] == 2 and cfg['person_query'] == 'person.'
    assert (cfg['confidence'], cfg['text_threshold'], cfg['nms_iou']) == (.3, .25, .7)
    assert cfg['owl_qualification']['producer_revision'] == '5b03f2ffc78eeb3a8439770b67cd4923c32c4e58'
    assert cfg['owl_qualification']['native']['bytes'] == 2647 and cfg['owl_qualification']['host']['bytes'] == 14465
    for name, wanted in cfg['helper_pins'].items():
        raw = (root/name).read_bytes()
        assert wanted == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def native_arrays(person_count=0):
    boxes = np.asarray([[0, 0, 3, 4]]*person_count, dtype=np.float32).reshape(-1, 4)
    scores = np.full(person_count, .8, dtype=np.float32)
    ids = np.asarray([[101, 102]], dtype=np.int64); mask = np.ones((1, 2), dtype=np.int64)
    logits = np.full((1, 900, 256), -np.inf, dtype=np.float32); logits[:, :, :2] = 0
    raw = np.full((1, 900, 4), .5, dtype=np.float32)
    return boxes, scores, ['person']*person_count, raw, logits, ids, mask, 256


@pytest.mark.parametrize('person_count', [0, 2])
def test_full_raw_and_empty_person_bank_owl_patch_census_no_selection(tmp_path, monkeypatch, person_count):
    import world_reward.owlv2_object_observations as o
    rgb = np.arange(7*9*3, dtype=np.uint8).reshape(7, 9, 3); before = rgb.tobytes(); calls = []
    def observation(model, actual, frame, operations):
        calls.append(frame)
        boxes = np.zeros((3600, 4), dtype=np.float32)
        return o.Owlv2ObjectObservations(frame, (7, 9), (60, 60), np.arange(3600, dtype=np.int64),
            boxes, np.arange(3600, dtype=np.float32), boxes.copy())
    monkeypatch.setattr(o, 'infer_owlv2_object_frame', observation)
    row = dict(image_id='1'*32, file='image_000002.jpg', width=9, height=7, bytes=2, sha256='a'*64)
    arrays, meta = p.bank_arrays(rgb, row, 1, lambda x: native_arrays(person_count), None, None)
    assert calls == [0] and meta['original_frame_index'] == 0 and meta['bank_index'] == 1
    assert arrays['original_frame_index'].dtype == np.int64 and arrays['original_frame_index'].shape == ()
    assert arrays['image_size'].tolist() == [7, 9] and rgb.tobytes() == before
    assert arrays['person_model_logits'].shape == (1, 900, 256)
    assert np.isneginf(arrays['person_model_logits'][:, :, 2:]).all()
    assert arrays['owl_objectness_logits'].shape == (3600,) and arrays['owl_patch_ids'].tolist() == list(range(3600))
    assert meta['person_retained_rows'] == (1 if person_count else 0)  # exact native duplicate dedup only
    assert arrays['person_retained_boxes'].shape == (meta['person_retained_rows'], 4)
    saved = p.save_bank(tmp_path, 1, arrays)
    assert saved['file'] == 'image_000001.npz'
    with np.load(tmp_path/saved['file'], allow_pickle=False) as actual:
        assert set(actual.files) == set(arrays)
        assert all(actual[n].tobytes() == a.tobytes() for n, a in arrays.items())
    with pytest.raises(FileExistsError): p.save_bank(tmp_path, 1, arrays)


def dispatch_fixture(tmp_path, monkeypatch, *, native_fail=False):
    root = tmp_path/'root'; code = root/'code'; code.mkdir(parents=True)
    out = root/'results/out'; out.parent.mkdir(); (root/'jobs').mkdir()
    (root/'jobs/.world-reward-h100.lock').write_bytes(b'lock')
    cfg = config(); source = dict(producer_revision='a'*40, helpers={}, markers={}); commands = []
    for name in p.NATIVE_FILES: source['helpers'][name] = seal(code/name, b'# original file')
    for name in ('revision', 'source-sha256'): source['markers'][name] = seal(code.parent/name, b'marker')
    inputs = dict(schema='world_reward.rgb_proposal_inputs.v1', images=[dict(image_id='b'*32,
        file='image_000000.jpg', width=7, height=9, bytes=1, sha256='c'*64)])
    data = tmp_path/'data'; data.mkdir(); seal(data/'manifest.json', b'public'); seal(data/'image_000000.jpg', b'rgb')
    prior = dict(owl=dict(native_runtime={}), runtime={}, grounding={}, acquisition={})
    monkeypatch.setattr(p, 'ROOT', root); monkeypatch.setattr(p, 'OUTPUT', out); monkeypatch.setattr(p, 'DATA', data)
    monkeypatch.setattr(p.rt, 'source', lambda *a: source); monkeypatch.setattr(p, 'configuration', lambda *a: cfg)
    monkeypatch.setattr(p, 'public_inputs', lambda *a: inputs); monkeypatch.setattr(p, 'qualifications', lambda *a, **k: prior)
    monkeypatch.setattr(p, 'asset_files', lambda *a: {})
    original_pin = p.rt.pinned
    monkeypatch.setattr(p.rt, 'pinned', lambda path, pin, maximum=1 << 20:
        {} if Path(path) == code/p.owl.CONFIG else original_pin(path, pin, maximum))
    monkeypatch.setattr(p, 'command', lambda args, d: (commands.append(args) or ''))
    monkeypatch.setattr(p, 'cleanup', lambda *a: commands.append(['cleanup']))
    def run(cmd, **kwargs):
        commands.append(cmd); (out/'.container.cid').write_text('d'*64)
        proof_pin = p.rt.identity(out/'proof.json', 2 << 20)
        row = dict(image_id='b'*32, original_frame_index=0, bank_index=0, image_size=[9, 7],
            input_file='image_000000.jpg', input_identity=dict(bytes=1, sha256='c'*64), file='image_000000.npz',
            identity=seal(out/'image_000000.npz', b'bank'), owl_patches=3600, person_native_queries=900,
            person_postprocessor_rows=0, person_retained_rows=0, person_ids=[])
        report = dict(schema=cfg['schema'], stage='native_all_person_owl_banks', status='fail' if native_fail else 'pass',
            phase='complete', producer_revision='a'*40, image_id=p.IMAGE, proof_identity=proof_pin, model_loads=2,
            source_inputs_runtime_assets_rehashed_after=True, all_patches_retained=True, images=[row], runtime_identity={},
            person_query='person.', confidence=.3, text_threshold=.25, nms_iou=.7)
        for n in ('person_forward_calls', 'image_embed_calls', 'objectness_calls', 'box_calls'): report[n] = 1
        for n in ('dwpose_calls', 'sam_calls', 'hoi_calls', 'tracking_calls'): report[n] = 0
        for n in ('ground_truth_used', 'reference_metadata_read', 'split_metadata_read', 'challenge_inputs_used',
                  'actor_selection_performed', 'ownership_verified', 'quality_verified', 'adoption'): report[n] = False
        seal(out/'native.json', p.encode(report)); return SimpleNamespace(returncode=1 if native_fail else 0)
    monkeypatch.setattr(p.subprocess, 'run', run)
    return code, cfg, source, inputs, out, commands


def test_mock_real_host_complete_zero_person_one_image_narrow_mounts(tmp_path, monkeypatch):
    code, _, _, _, out, commands = dispatch_fixture(tmp_path, monkeypatch)
    result = p.dispatch(code, 'a'*40, dict(bytes=1, sha256='e'*64), 1)
    assert result['status'] == 'pass' and result['owned_cleanup_verified'] and result['outputs_sealed']
    cmd = next(c for c in commands if c[:2] == ['docker', 'run'])
    mounts = [cmd[i+1] for i, token in enumerate(cmd) if token == '--mount']
    assert not any('metadata' in n or 'reference_' in n or f'src={code},' in n for n in mounts)
    assert sum(n.endswith('readonly') for n in mounts) == len(p.NATIVE_FILES)+4
    assert '-i' in cmd and '-I' in cmd and '-B' in cmd and 'HF_HUB_OFFLINE=1' in cmd
    assert cmd[cmd.index('--gpus')+1] == 'device=0' and cmd[cmd.index('--memory')+1] == '64g'
    assert out.stat().st_mode & 0o777 == 0o500 and all(f.stat().st_mode & 0o777 == 0o400 for f in out.iterdir())


@pytest.mark.parametrize('fault', ['native', 'source', 'prior', 'input', 'census', 'late', 'proof'])
def test_host_cannot_publish_partial_or_mutated_pass_cleanup_still_attempted(tmp_path, monkeypatch, fault):
    code, _, source, inputs, out, commands = dispatch_fixture(tmp_path, monkeypatch, native_fail=fault == 'native')
    original_run = p.subprocess.run; clock = [0.]
    if fault == 'late': monkeypatch.setattr(p.time, 'monotonic', lambda: clock[0])
    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if fault == 'source': monkeypatch.setattr(p.rt, 'source', lambda *a: dict(source, mutation=True))
        elif fault == 'prior': monkeypatch.setattr(p, 'qualifications', lambda *a, **k: {'foreign': True})
        elif fault == 'input': monkeypatch.setattr(p, 'public_inputs', lambda *a: dict(inputs, mutation=True))
        elif fault == 'proof':
            (out/'proof.json').chmod(0o600); seal(out/'proof.json', b'changed')
        elif fault == 'census':
            f = out/'native.json'; report = json.loads(f.read_bytes()); report['box_calls'] = 0
            f.chmod(0o600); seal(f, p.encode(report))
        elif fault == 'late': clock[0] = 601.
        return result
    monkeypatch.setattr(p.subprocess, 'run', run)
    result = p.dispatch(code, 'a'*40, dict(bytes=1, sha256='e'*64), 1)
    assert result['status'] == 'fail' and result['owned_cleanup_verified'] and ['cleanup'] in commands
    assert json.loads((out/'host.json').read_bytes())['status'] == 'fail'


def test_prior_qualification_failure_precedes_container_and_output(tmp_path, monkeypatch):
    code, _, _, _, out, commands = dispatch_fixture(tmp_path, monkeypatch)
    def fail(*a, **k): raise ValueError('not qualified')
    monkeypatch.setattr(p, 'qualifications', fail)
    with pytest.raises(ValueError): p.dispatch(code, 'a'*40, dict(bytes=1, sha256='e'*64), 1)
    assert not out.exists() and not commands


def test_cleanup_exact_owned_no_foreign_removal(tmp_path, monkeypatch):
    path = tmp_path/'cid'; path.write_text('a'*64); calls = []
    def command(args, deadline):
        calls.append(args)
        if args[1] == 'inspect': return p.IMAGE+'|/owned|'+p.ENTRY+'|'+'b'*40
        if 'id='+'a'*64 in args: return 'a'*64
        return ''
    monkeypatch.setattr(p, 'command', command)
    p.cleanup(path, 'owned', 'b'*40, time.monotonic()+1)
    assert ['docker', 'rm', '-f', 'a'*64] in calls
    monkeypatch.setattr(p, 'command', lambda a, d: 'foreign' if a[1] == 'inspect' else 'a'*64)
    with pytest.raises(ValueError): p.cleanup(path, 'owned', 'b'*40, time.monotonic()+1)


def test_native_fails_before_models_on_wrong_image_and_no_secret_exception(tmp_path, monkeypatch):
    out = tmp_path/'out'; out.mkdir(); monkeypatch.setattr(p, 'OUTPUT', out)
    pin = seal(out/'proof.json', p.encode({})); monkeypatch.setenv('WR_IMAGE_ID', 'foreign')
    result = p.native(tmp_path, 'a'*40, config(), pin, time.monotonic()+1)
    assert result['status'] == 'fail' and result['phase'] == 'authentication' and result['model_loads'] == 0
    assert p.error(Exception('secret signed URL')) == 'other'


def test_source_no_hidden_selector_or_partial_weight_load_and_shell_syntax():
    root = Path(__file__).resolve().parents[1]; source = Path(p.__file__).read_text()
    compile(source, p.__file__, 'exec')
    assert "text='person.'" in source and 'target_sizes=[rgb.shape[:2]]' in source
    assert 'output_loading_info=True' in source and "'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'" in source
    assert 'owl.strict_state(object_model, state, torch)' in source and 'strict=False' not in source
    assert 'torch.autocast(' not in source and 'select_interacting_actor(' not in source and 'argsort(' not in source
    assert "DOCKER_HOST='unix://'+str(ROOT/'docker.sock')" in source
    wrapper = (root/'infra/run_rgb_endpoint_bank.sh').read_text()
    assert '640s' in wrapper and 'env -i' in wrapper and '-I -B' in wrapper
    assert 'world-reward-ncc-h100-02' in wrapper


def qualification_fixture(tmp_path, monkeypatch):
    cfg = config(); root = tmp_path/'root'; old = root/'qualification'; old.mkdir(parents=True)
    source = dict(helpers={p.owl.CONFIG: {}, 'infra/owlv2_native_qualify.py': {}}, producer_revision='b'*40)
    policy = {'schema': 'world_reward.owlv2_native_qualification.v1', 'assets': {}}
    code = tmp_path/'code'; source['helpers'][p.owl.CONFIG] = seal(code/p.owl.CONFIG, p.encode(policy))
    source['helpers']['infra/owlv2_native_qualify.py'] = seal(code/'infra/owlv2_native_qualify.py', b'original')
    for name in p.owl.NATIVE_FILES:
        source['helpers'].setdefault(name, dict(bytes=1, sha256='a'*64))
    prior = {'producer_revision': 'c'*40, 'report_identity': dict(bytes=1, sha256='a'*64)}
    rows = [dict(file=f'procedural_{i:02d}.npz', original_frame_index=i, patches=3600,
        identity=seal(old/f'procedural_{i:02d}.npz', b'original'+bytes([i]))) for i in range(3)]
    safe = dict(source=source, image_id=p.IMAGE, assets={}, native_files={n: source['helpers'][n] for n in p.owl.NATIVE_FILES})
    proof_pin = seal(old/'proof.json', p.encode(safe)); seal(old/'.container.cid', b'd'*64)
    native = dict(schema=policy['schema'], stage='native_owlv2_procedural', status='pass', phase='complete',
        producer_revision='b'*40, image_id=p.IMAGE, model_loads=1, image_embed_calls=3, objectness_calls=3, box_calls=3,
        source_runtime_assets_rehashed_after=True, dataset_read=False, quality_verified=False, ownership_verified=False,
        adopted=False, proof_identity=proof_pin, frames=rows, runtime_identity={'original': True})
    native_pin = seal(old/'native.json', p.encode(native))
    host = dict(schema=policy['schema'], stage='owlv2_procedural_qualification_host', status='pass',
        producer_revision='b'*40, source_binding=source, runtime_binding={'runtime': True},
        native_report_identity=native_pin, native_exit_status=0, image_id=p.IMAGE,
        owned_cleanup_verified=True, outputs_sealed=True, source_runtime_assets_rehashed_after=True,
        acquisition_binding=prior, native_frames=rows)
    host_pin = seal(old/'host.json', p.encode(host)); cfg['owl_qualification'] = dict(producer_revision='b'*40, host=host_pin, native=native_pin)
    monkeypatch.setattr(p.owl, 'OUTPUT', old); monkeypatch.setattr(p.rt, 'source', lambda *a: source)
    monkeypatch.setattr(p.owl, 'runtime_proof', lambda *a, **k: {'runtime': True})
    monkeypatch.setattr(p.gdi, 'runtime_identity', lambda *a, **k: {'original': True})
    monkeypatch.setattr(p.owl, 'acquisition', lambda *a: prior)
    return code, cfg, host, native, old


@pytest.mark.parametrize('fault', [None, 'hostfail', 'nativefail', 'partial', 'source', 'runtime', 'decode', 'oldoutput', 'extra'])
def test_actual_prior_whole_qualification_not_merely_current_runtime(tmp_path, monkeypatch, fault):
    code, cfg, host, native, out = qualification_fixture(tmp_path, monkeypatch)
    if fault == 'oldoutput':
        f = out/'procedural_01.npz'; f.chmod(0o600); seal(f, b'corruption')
    elif fault == 'extra': seal(out/'foreign.json', b'foreign')
    elif fault:
        if fault == 'hostfail': host['status'] = 'fail'
        elif fault == 'nativefail': native['status'] = 'fail'
        elif fault == 'partial': native['box_calls'] = 2
        elif fault == 'source': host['source_binding'] = {}
        elif fault == 'runtime': host['runtime_binding'] = {}
        elif fault == 'decode': native['dataset_read'] = True
        for filename, data, key in (('native.json', native, 'native'), ('host.json', host, 'host')):
            path = out/filename; path.chmod(0o600); new_pin = seal(path, p.encode(data)); cfg['owl_qualification'][key] = new_pin
        host['native_report_identity'] = cfg['owl_qualification']['native']
        (out/'host.json').chmod(0o600); cfg['owl_qualification']['host'] = seal(out/'host.json', p.encode(host))
    if fault:
        with pytest.raises(ValueError): p.qualifications(code, cfg)
    else:
        result = p.qualifications(code, cfg)
        assert result['owl']['producer_revision'] == 'b'*40 and result['owl']['native_runtime'] == {'original': True}


def test_native_source_import_closure_available_under_isolated_python(tmp_path):
    # Native -I starts with only the driver; its explicit src path must precede
    # transitive GDI import of world_reward.prompt_selection.
    root = Path(__file__).resolve().parents[1]
    assert "str(Path(__file__).resolve().parents[1]/'src')" in (root/'infra/rgb_endpoint_bank.py').read_text()
    for name in p.NATIVE_FILES:
        target = tmp_path/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root/name).read_bytes())
    import subprocess, sys
    result = subprocess.run([sys.executable, '-I', '-B', '-c',
        'import runpy;runpy.run_path('+repr(str(tmp_path/'infra/rgb_endpoint_bank.py'))+')'],
        capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()
