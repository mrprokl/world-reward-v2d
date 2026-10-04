"""Tiny authored composition/provenance; no real RGB, private truth or GPU."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'infra'))
spec = importlib.util.spec_from_file_location('wr_ycbv_track_test', ROOT / 'infra/ycbv_point_track.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def independent_pins():
    identity = {'bytes': 12, 'sha256': 'a' * 64}
    producer = {**identity, 'producer_revision': 'b' * 40, 'script_sha256': 'c' * 64}
    return {'schema': 'world-reward-ycbv-point-track-pins-v2', 'input_pins': identity.copy(),
        **{kind: {'report': producer.copy(), 'files': {n: identity.copy() for n in gate.stage_names(kind)}} for kind in gate.STAGES}}


def test_independent_full_stage_pins_exact():
    pins = independent_pins(); assert gate.validate_track_pins(pins) is pins
    assert len(pins['depth_init']['files']) == 3 and len(pins['masks']['files']) == 288 and len(pins['objects']['files']) == 12


@pytest.mark.parametrize('fault', ['missing', 'schema', 'extra', 'manifest', 'producer', 'sha', 'bytes', 'missing_frame', 'extra_frame', 'extra_private'])
def test_bad_prerequisite_pins_fail_before_any_io(monkeypatch, fault):
    pins = independent_pins()
    if fault == 'missing': pins.pop('depth_init')
    elif fault == 'schema': pins['schema'] = 'legacy'
    elif fault == 'extra': pins['reference_mesh'] = {}
    elif fault == 'manifest': pins['input_pins']['bytes'] = True
    elif fault == 'producer': pins['objects']['report']['producer_revision'] = 'main'
    elif fault == 'sha': pins['masks']['report']['sha256'] = 'A' * 64
    elif fault == 'bytes': pins['depth_init']['report']['bytes'] = 0
    elif fault == 'missing_frame': pins['masks']['files'].pop(next(iter(pins['masks']['files'])))
    elif fault == 'extra_frame': pins['masks']['files']['scene_000048/masks/1/000097.png'] = pins['input_pins'].copy()
    else: pins['objects']['files']['scene_000048/private.json'] = pins['input_pins'].copy()
    monkeypatch.setattr(Path, 'open', lambda *a, **k: pytest.fail('Bad pins opened a file'))
    with pytest.raises(ValueError): gate.validate_track_pins(pins)


@pytest.mark.parametrize('fault', ['missing', 'bool', 'negative', 'zero', 'nan', 'overrun', 'wrong_cap'])
def test_actual_component_gpu_durations_no_missing_overrun_or_cap_substitution(fault):
    reports = {k: {'budget_seconds': v[4], v[5]: 10.} for k, v in gate.STAGES.items()}
    kind = 'depth_init'; field = gate.STAGES[kind][5]
    if fault == 'missing': reports[kind].pop(field)
    elif fault == 'bool': reports[kind][field] = True
    elif fault == 'negative': reports[kind][field] = -1
    elif fault == 'zero': reports[kind][field] = 0
    elif fault == 'nan': reports[kind][field] = np.nan
    elif fault == 'overrun': reports[kind][field] = 300.1
    else: reports[kind]['budget_seconds'] = 301
    with pytest.raises(ValueError): gate.remaining_budget(reports)


def test_remaining_uses_original_measured_durations_not_maximum_caps():
    reports = {k: {'budget_seconds': v[4], v[5]: x} for (k, v), x in zip(gate.STAGES.items(), (35.2, 211.7, 547.3))}
    times, remaining = gate.remaining_budget(reports)
    assert remaining == pytest.approx(2805.8) and sum(times.values()) + remaining == 3600


def test_unbound_receipt_not_parsed_and_duplicate_json_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'report.json'; path.write_bytes(b'not JSON'); path.chmod(0o400)
    called = False
    def forbidden(*a, **k):
        nonlocal called; called = True; pytest.fail('Unbound bytes reached JSON')
    monkeypatch.setattr(gate.depth.files, 'strict_json', forbidden)
    with pytest.raises(ValueError): gate.checked_json(path, {'bytes': 9, 'sha256': 'a' * 64})
    assert not called


@pytest.mark.parametrize('fault', ['partial_init', 'missing_masks', 'tampered_mesh', 'unpinned_report'])
def test_all_stage_bytes_checked_before_any_stage_json_or_model(tmp_path, monkeypatch, fault):
    pins = independent_pins(); parsed = []; original_json = gate.depth.files.strict_json
    monkeypatch.setattr(gate.depth.files, 'strict_json', lambda raw: parsed.append(raw) or original_json(raw))
    code = tmp_path / 'code'; code.mkdir(); root = tmp_path / 'root'; root.mkdir()
    pinpath = code / gate.PIN_FILE; pinpath.parent.mkdir(); pinpath.write_text(json.dumps(pins)); pinpath.chmod(0o444)
    inputpath = code / gate.depth.PIN_FILE; inputpath.write_text('{}'); inputpath.chmod(0o444)
    # Fake only identity/public-reader boundaries: no real archive/media/models.
    original_identity = gate.depth.files.identity
    def identity(path):
        if path == pinpath: return original_identity(path)
        if path == inputpath: return pins['input_pins']
        for kind, (folder, *_rest) in gate.STAGES.items():
            base = root / gate.BASE / folder
            if path == base / 'report.json':
                row = {k: pins[kind]['report'][k] for k in ('bytes', 'sha256')}
                if fault == 'unpinned_report' and kind == 'objects': row['sha256'] = '0' * 64
                return row
            for name, wanted in pins[kind]['files'].items():
                if path == base / name:
                    if (fault == 'partial_init' and kind == 'depth_init') or (fault == 'missing_masks' and kind == 'masks'):
                        raise FileNotFoundError('own missing prerequisite')
                    if fault == 'tampered_mesh' and kind == 'objects': return {'bytes': 12, 'sha256': '0' * 64}
                    return wanted
        pytest.fail('Unexpected path outside public/stage identities')
    monkeypatch.setattr(gate.depth.files, 'identity', identity)
    original_checked = gate.checked_json
    monkeypatch.setattr(gate, 'checked_json', lambda path, wanted: {} if path == inputpath else pytest.fail('Stage JSON reached before full identity gate'))
    monkeypatch.setattr(gate.depth, 'public_inputs', lambda *a: ([], {'manifest': pins['input_pins'], 'RGB_identities': {}}))
    with pytest.raises((ValueError, FileNotFoundError)): gate.prerequisites(root, code)
    assert len(parsed) == 1  # Independent pins only, not any stage report.


def geometry():
    vertices = np.array([[-4., -3.13, 2.], [5., -3.13, 2.], [5., 4.07, 2.], [-4., 4.07, 2.]])
    return dict(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]], np.int64),
        R0=np.eye(3), t0=np.zeros(3), K=np.array([[32., 0., 16.], [0., 24., 12.], [0., 0., 1.]]), scale=np.array(1.))


def authored_scene(monkeypatch):
    monkeypatch.setattr(gate, 'FRAMES', 3); monkeypatch.setattr(gate.depth, 'WIDTH', 32); monkeypatch.setattr(gate.depth, 'HEIGHT', 24)
    geo = {k: gate.freeze(v) for k, v in geometry().items()}; masks = [np.ones((24, 32), bool) for _ in range(3)]
    y, x = np.mgrid[:24, :32]; points = np.stack(((x + .5 - 16) / 32 * 2, (y + .5 - 12) / 24 * 2, np.full_like(x, 2.)), -1).astype(np.float64)
    video = [np.full((24, 32, 3), i, dtype=np.uint8) for i in range(3)]
    frames = [(i, video[i], points.copy()) for i in range(3)]
    return geo, {'mask': masks[0]}, frames, masks, np.tile(geo['vertices'], (2048, 1))


def test_full_native_pool_frozen_before_boots_same_rankings_no_query_replacement(monkeypatch):
    from world_reward.rigid_alignment import RigidAlignment
    data = authored_scene(monkeypatch); geo, initial, frames, masks, sample = data; calls = []
    def raster(v, f, K, w, h):
        calls.append('raster'); assert (w, h) == (32, 24); return np.ones((h, w), bool)
    def align(s, p, R, t):
        # Authored accepted fits place every original slot safely in front;
        # this is a tiny dependency-injected backend, never a real ICP claim.
        calls.append('align'); return RigidAlignment(np.eye(3), np.zeros(3), 'own', .01, .01, 40, 1)
    def track(video, queries):
        assert len(calls) == 3 * 25 * 3  # Initial raster, unchanged align, fitted raster per slot.
        calls.append('track'); assert video.shape == (3, 24, 32, 3) and len(queries) == 32
        assert not queries.flags.writeable and np.all(queries[:, 0] == 0)
        tracks = np.broadcast_to(queries[:, [2, 1]] * [256 / 32, 256 / 24], (3, 32, 2)).copy()
        return tracks, np.ones((3, 32), bool), {'own_test': True}
    result, queries, detail = gate.compose_scene(geo, initial, iter(frames), masks,
        sample_surface=sample, mesh_centroid=np.array([0., 0., 2.]), raster=raster, track=track, align=align)
    assert calls[-1] == 'track' and calls.count('track') == 1 and detail == {'own_test': True}
    assert result.pool.rotations.shape == (3, 25, 3, 3) and result.pool.frame_index.tolist() == [0, 1, 2]
    assert result.baseline.rotations.shape == result.candidate.rotations.shape == (3, 3, 3)
    assert len(queries.canonical_points) == 32 and result.pool.valid_candidates.all()
    assert result.pool.greedy_indices.tolist() == [0, 0, 0]


def test_missing_original_frame_fails_before_any_tracking(monkeypatch):
    from world_reward.rigid_alignment import RigidAlignment
    geo, initial, frames, masks, sample = authored_scene(monkeypatch)
    def align(s, p, R, t): return RigidAlignment(R.copy(), t.copy(), 'own', .01, .01, 40, 1)
    def forbidden(*a): pytest.fail('Incomplete original timeline reached Boots')
    with pytest.raises(ValueError, match='coverage'):
        gate.compose_scene(geo, initial, iter(frames[:2]), masks, sample_surface=sample, mesh_centroid=np.array([0., 0., 2.]),
            raster=lambda v, f, K, w, h: np.ones((h, w), bool), track=forbidden, align=align)


def test_no_automatic_queries_fails_without_fallback_or_boots(monkeypatch):
    geo, initial, frames, masks, sample = authored_scene(monkeypatch); masks[0][:] = False
    with pytest.raises(ValueError, match='Fewer than8'):
        gate.compose_scene(geo, initial, iter(frames), masks, sample_surface=sample, mesh_centroid=np.zeros(3),
            raster=lambda *a: pytest.fail('Unsupported queries reached pool'), track=lambda *a: pytest.fail('Unsupported queries reached Boots'))


@pytest.mark.parametrize('fault', ['scaled', 'camera', 'dtype', 'faces', 'reflection', 'extra'])
def test_canonical_geometry_no_second_scale_camera_fit_or_triangle_drop(fault):
    geo = geometry(); geo['faces'] = np.vstack((geo['faces'], geo['faces']))
    if fault == 'scaled': geo['scale'] = np.array(2.)
    elif fault == 'camera': geo['K'][0, 0] = 31
    elif fault == 'dtype': geo['vertices'] = geo['vertices'].astype(np.float32)
    elif fault == 'faces': geo['faces'] = geo['faces'].astype(np.int32)
    elif fault == 'reflection': geo['R0'] = np.diag([-1., 1., 1.])
    else: geo['GT'] = np.array(0.)
    with pytest.raises(ValueError): gate.load_geometry(geo, np.array([[32., 0., 16.], [0., 24., 12.], [0., 0., 1.]]))


def test_original_native_camera_uses_normalized_axes_not_rounded_f800(monkeypatch):
    monkeypatch.setattr(gate.depth, 'WIDTH', 4); monkeypatch.setattr(gate.depth, 'HEIGHT', 3)
    normalized = np.array([[5. / 4, 0., .5], [0., 5. / 3, .5], [0., 0., 1.]], np.float32)
    K = np.diag([4., 3., 1.]) @ normalized
    y, x = np.mgrid[:3, :4]; z = np.full((3, 4), 2., dtype=np.float32)
    points = np.stack(((x + .5 - K[0, 2]) / K[0, 0] * z,
        (y + .5 - K[1, 2]) / K[1, 1] * z, z), -1).astype(np.float32)
    initial = dict(depth=z, points=points, mask=np.ones((3, 4), bool), intrinsics=normalized, frame_index=np.array(0, np.int64), source_frame_id=np.array(1, np.int64))
    before = gate.depth.array_identities(initial); actual = gate.native_initial_camera(initial)
    np.testing.assert_array_equal(actual, K); assert not actual.flags.writeable
    assert actual[1, 1] != 5. and gate.depth.array_identities(initial) == before
    bad = dict(initial, intrinsics=np.diag([5 / 3, 5 / 4, 1.]).astype(np.float32))
    with pytest.raises(ValueError): gate.native_initial_camera(bad)


def test_raw_open_surface_not_repaired_or_required_closed_but_zeroarea_fails():
    geo = geometry(); geo['faces'] = np.vstack((geo['faces'], geo['faces']))
    loaded = gate.load_geometry(geo, geo['K'])
    np.testing.assert_array_equal(loaded['vertices'], geo['vertices']); np.testing.assert_array_equal(loaded['faces'], geo['faces'])
    collapsed = dict(geo, vertices=geo['vertices'].copy()); collapsed['vertices'][1] = collapsed['vertices'][0]
    with pytest.raises(ValueError, match='nondegenerate'): gate.load_geometry(collapsed, geo['K'])
    source = (ROOT / 'infra/ycbv_point_track.py').read_text()
    assert "require(mesh.is_watertight" not in source and 'closedness_required=False' in source


@pytest.mark.parametrize('fault', ['extra', 'missing', 'tampered', 'reorder'])
def test_all_three_saved_output_bytes_inventory_reverified_before_seal(tmp_path, fault):
    rows = []
    (tmp_path / 'report.json').write_bytes(b'tiny receipt'); (tmp_path / '.container.cid').write_bytes(b'a' * 64)
    for scene in gate.SCENES:
        path = tmp_path / f'scene_{scene:06d}.npz'; path.write_bytes(f'own bytes {scene}'.encode()); path.chmod(0o400)
        rows.append(dict(file=path.name, scene_id=scene, frames=gate.FRAMES, **gate.depth.files.identity(path)))
    gate.verify_output_inventory(tmp_path, rows)
    if fault == 'extra': (tmp_path / 'untracked.npz').write_bytes(b'extra')
    elif fault == 'missing': (tmp_path / rows[0]['file']).unlink()
    elif fault == 'tampered':
        path = tmp_path / rows[2]['file']; path.chmod(0o600); path.write_bytes(b'tampered'); path.chmod(0o400)
    else: rows.reverse()
    with pytest.raises(ValueError): gate.verify_output_inventory(tmp_path, rows)


def test_primary_boots_pins_are_exact_committed_evidence_not_caller_claims():
    protocol = json.loads((ROOT / 'configs/robotap_boots_protocol.json').read_text())
    assert gate.BOOTS_SOURCE == {n: {k: v[k] for k in ('bytes', 'sha256')} for n, v in protocol['source']['files'].items()}
    assert gate.BOOTS_CHECKPOINT == {k: protocol['checkpoint'][k] for k in ('bytes', 'sha256')}
    original = json.loads((ROOT / 'configs/robotap_boots_inference_pins.json').read_text())
    assert gate.BOOTS_RUNTIME == {k: original['runtime'][k] for k in gate.BOOTS_RUNTIME}
    assert gate.IMAGE == original['image_id']


def test_wrapper_syntax_narrow_offline_lock_owned_cleanup_image_ancestry():
    wrapper = ROOT / 'infra/run_ycbv_point_track.sh'; text = wrapper.read_text(); subprocess.run(['bash', '-n', str(wrapper)], check=True)
    for literal in ('--network none', '--read-only', '--user 0:0', '--cap-drop ALL', 'no-new-privileges', '--cidfile',
        'exec 9<', 'flock --nonblock 9', '--entrypoint /usr/bin/env', 'len(c)!=47', 'len(p)!=44', 'c[:44]!=p', '3630s'):
        assert literal in text
    assert 'exec 9>' not in text and 'src=$BASE/eval_private' not in text and 'src=$CODE,dst=$CODE' not in text
    assert "'report.json')" in text and 'WR_YCBV_HOST_PROOF_SHA256' in text


def test_host_import_no_numpy_torch_models_or_other_cohort_calls():
    script = "import runpy,sys;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2]);assert not {'numpy','torch','moge','robotap_boots_infer'}&sys.modules.keys()"
    subprocess.run([sys.executable, '-I', '-B', '-c', script, str(ROOT / 'infra'), str(ROOT / 'infra/ycbv_point_track.py')], check=True)
    source = (ROOT / 'infra/ycbv_point_track.py').read_text()
    assert 'net.infer(tensor[None], fov_x=fov)' in source and 'index == 0:' in source
    assert 'force_projection=' not in source and 'apply_mask=' not in source and 'native.main(' not in source and 'native.bindings(' not in source


def test_actual_runtime_bundle_keeps_numeric_helpers_but_container_mounts_no_recipes():
    launch_spec = importlib.util.spec_from_file_location('wr_ycbv_track_bundle', ROOT / 'infra/azure_job.py')
    launcher = importlib.util.module_from_spec(launch_spec); launch_spec.loader.exec_module(launcher)
    source = {str(p.relative_to(ROOT)): p.read_bytes() for folder in ('infra', 'src', 'configs') for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.json')}
    selected = launcher.runtime_bundle_paths(source, 'infra/run_ycbv_point_track.sh')
    assert set(gate.SOURCE_FILES) <= set(selected)
    assert 'infra/object_pose_smoke.py' not in gate.SOURCE_FILES
    assert 'configs/ycbv_point_protocol.json' not in gate.SOURCE_FILES


def test_numerical_candidate_and_tracking_functions_unchanged():
    import ast,subprocess
    old=subprocess.run(['git','show','HEAD:infra/ycbv_point_track.py'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    before=ast.parse(old);after=ast.parse((ROOT/'infra/ycbv_point_track.py').read_text())
    for name in ('compose_scene','native_boots','native_initial_camera','load_geometry'):
        a=next(n for n in before.body if isinstance(n,ast.FunctionDef)and n.name==name)
        b=next(n for n in after.body if isinstance(n,ast.FunctionDef)and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def test_updated_transitive_source_hashes_are_exact():
    for name,sha in gate.FROZEN.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
