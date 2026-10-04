"""Tiny official-packer boundary fixtures: no real data/models/official source."""
from dataclasses import asdict, replace
import csv
import hashlib
import importlib.util
import io
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from world_reward.contracts import Reconstruction
from world_reward.submission import (Track1Episode, Track1Layout, SubmissionRows,
    assemble_submission, decode_submission, read_template, write_submission)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "https://github.com/mrprokl/world-reward-v2d/commit/" + "d" * 40


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src")); monkeypatch.syspath_prepend(str(ROOT / "infra"))
    spec = importlib.util.spec_from_file_location("official_pack_gate_test", ROOT / "infra/official_track1_pack_gate.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def ids(ep=15, frames=(2, 3, 4, 8, 9, 10), budget=4096):
    result = []
    for array in range(8):
        count = {0: 46, 1: 23, 2: 15, 3: 3, 4: 1, 5: 1, 6: budget, 7: budget}[array]
        for frame in frames if array in (0, 3, 4) else (999999,):
            result.extend(f"t1_{ep:06d}_{frame:06d}_{array}_{point:06d}" for point in range(count))
    return result


def episode(n=97):
    rec = Reconstruction(pose=np.arange(n * 136, dtype=np.float32).reshape(n, 136) / 100,
        scales=np.arange(68, dtype=np.float32) / 100, shape=np.arange(45, dtype=np.float32) / 50,
        object_rotation=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        object_translation=np.column_stack((np.zeros((n, 2), np.float32), np.full(n, 2, np.float32))),
        object_scale=np.asarray(1., np.float32))
    vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    return Track1Episode(rec, vertices, faces, np.zeros(72, np.float32),
        dict(input_track="track_1", ground_truth_used=False, hand_labeled_test=False, oracle_modes=[]), n)


def output_rows(gate, ep=None, layout=None):
    ep = episode() if ep is None else ep
    layout = Track1Layout.from_row_ids(ids()) if layout is None else layout
    rows = assemble_submission(layout, {15: ep}, code_commit_url=COMMIT)
    return ep, layout, rows


@pytest.mark.parametrize("n", [96, 97, 501])
def test_exact_controls_full_timeline_constant_identity_and_padding(gate, n):
    ep, layout, rows = output_rows(gate, episode(n))
    report = gate.verify_output(rows, layout, ep, COMMIT)
    assert report["full_source_frames"] == n and report["original_frame_indices"] == [2, 3, 4, 8, 9, 10]
    assert report["oriented_triangles_exact"] and report["schema_roundtrip_verified"]
    assert report["padding_faces"] == report["padding_vertices"] == 4092
    assert report["numerical_geometry_independently_reverified"] is False


def test_exact_oriented_cyclic_reordering_and_vertex_weld_equivalent(gate):
    ep = episode(); perm = np.array([2, 0, 3, 1]); inverse = np.argsort(perm)
    actual = gate.oriented_triangles(ep.object_vertices[perm], np.roll(inverse[ep.object_faces[::-1]], 1, axis=1))
    expected = gate.oriented_triangles(ep.object_vertices, ep.object_faces)
    np.testing.assert_array_equal(actual, expected)
    duplicate_vertices = np.concatenate((ep.object_vertices, ep.object_vertices[:1]))
    duplicate_faces = ep.object_faces.copy(); duplicate_faces[duplicate_faces == 0] = 4
    np.testing.assert_array_equal(gate.oriented_triangles(duplicate_vertices, duplicate_faces), expected)
    padding = np.concatenate((duplicate_faces, [[0, 0, 0], [1, 1, 2]]))
    np.testing.assert_array_equal(gate.oriented_triangles(duplicate_vertices, padding), expected)
    collinear_vertices = np.concatenate((ep.object_vertices, [[.2, 0, 0]]))
    np.testing.assert_array_equal(gate.oriented_triangles(collinear_vertices, np.concatenate((ep.object_faces, [[0, 1, 4]]))), expected)
    tiny = ep.object_vertices.astype(np.float64) * 1e-100
    assert len(gate.oriented_triangles(tiny, ep.object_faces)) == 4


@pytest.mark.parametrize("fault", ["flip", "delete", "duplicate", "shrink", "epsilon"])
def test_no_oriented_surface_change_accepted(gate, fault):
    ep, layout, rows = output_rows(gate)
    values = rows.values.copy(); keys = layout.keys
    face = np.flatnonzero(keys[:, 2] == 7); vertex = np.flatnonzero(keys[:, 2] == 6)
    if fault == "flip": values[face[0], :2] = values[face[0], [1, 0]]
    elif fault == "delete": values[face[0]] = 0
    elif fault == "duplicate": values[face[4]] = values[face[0]]
    elif fault == "shrink": values[vertex] *= .99
    else: values[vertex[1], 0] += 1e-12
    with pytest.raises(ValueError, match="oriented triangle"):
        gate.verify_output(replace(rows, values=values), layout, ep, COMMIT)


@pytest.mark.parametrize("fault", ["pose", "rotation", "translation", "scales", "shape", "scale", "commit", "nonfinite", "badpad"])
def test_output_changes_failclosed(gate, fault):
    ep, layout, rows = output_rows(gate); values = rows.values.copy()
    arrays = {"pose": 0, "scales": 1, "shape": 2, "rotation": 3, "translation": 4, "scale": 5}
    if fault in arrays:
        selected = np.flatnonzero(layout.keys[:, 2] == arrays[fault])[0]; values[selected, 0] += .01
    elif fault == "nonfinite": values[0, 0] = np.nan
    elif fault == "badpad": values[np.flatnonzero(layout.keys[:, 2] == 6)[-1]] = [.2, .3, .4]
    changed = replace(rows, values=values, code_commit_url=rows.code_commit_url if fault != "commit" else rows.code_commit_url[:-1] + "a")
    with pytest.raises(ValueError): gate.verify_output(changed, layout, ep, COMMIT)


def test_original_sample_rowid_selection_order_and_full_envelope(gate):
    full_ids = ids(0) + ids(15)
    rng = np.random.default_rng(21); full_ids = [full_ids[i] for i in rng.permutation(len(full_ids))]
    # Generic pure helper may be tested with a tiny explicit envelope; production
    # calls default exact30/740780/9877 and cannot override those CLI constants.
    full = Track1Layout.from_row_ids(full_ids)
    with pytest.raises(ValueError): gate.selected_layout(full, 15)
    other = Track1Layout.from_row_ids(ids(0) + ids(1))
    selected = gate.selected_layout(other, 1, expected_rows=len(other.row_ids), expected_frames=12, expected_episodes=2)
    assert selected.row_ids == tuple(row for row in other.row_ids if row.startswith("t1_000001_"))
    with pytest.raises(ValueError): gate.selected_layout(other, 29, expected_rows=len(other.row_ids), expected_frames=12, expected_episodes=2)
    assert gate.row_id_digest(selected.row_ids) == hashlib.sha256(("\n".join(selected.row_ids) + "\n").encode()).hexdigest()


def test_actual_arrow_projection_never_reads_original_xyz(monkeypatch, tmp_path):
    pa = pytest.importorskip("pyarrow"); pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "own_template.parquet"
    source_ids = ids(budget=8)
    pq.write_table(pa.table({"row_id": source_ids, "x": [float("nan")] * len(source_ids), "y": ["forbidden"] * len(source_ids)}), path)
    real = pq.read_table; columns = []
    def checked(path, **kwargs):
        columns.append(kwargs); assert kwargs == {"columns": ["row_id"]}; return real(path, **kwargs)
    monkeypatch.setattr(pq, "read_table", checked)
    assert read_template(path).row_ids == tuple(source_ids)
    assert columns == [{"columns": ["row_id"]}]


def pack_fixture(gate, tmp_path, n=501):
    root = tmp_path / "root"; root.mkdir()
    mesh = root / "outputs/episode_000015/cari_shared_export_v1/object_aligned.glb"
    mesh.parent.mkdir(parents=True); mesh.write_bytes(b"opaque own GLB fixture, never decoded"); mesh.chmod(0o444)
    scratch = root / "scratch"; scratch.mkdir()
    ep, layout, _ = output_rows(gate, episode(n)); loaded = SimpleNamespace(episode=ep)
    calls = []
    def packer(args):
        calls.append(args)
        with args.sample.open(newline="") as stream:
            rows = list(csv.reader(stream))
        assert rows[0] == ["row_id"] and tuple(row[0] for row in rows[1:]) == layout.row_ids
        assert args.sample.name == "row_ids_only.csv" and args.commit == COMMIT
        assert {p.name for p in args.episodes.iterdir()} == {"episode_000015.npz", "episode_000015_object.glb"}
        assert (args.episodes / "episode_000015_object.glb").read_bytes() == mesh.read_bytes()
        with np.load(args.episodes / "episode_000015.npz", allow_pickle=False) as data:
            assert set(data.files) == set(gate.NPZ_KEYS) and data["pose"].shape == (n, 136)
            for key in gate.NPZ_KEYS:
                expected = np.asarray(getattr(ep.reconstruction, key))
                assert data[key].dtype == expected.dtype and data[key].tobytes() == expected.tobytes()
        rows = assemble_submission(layout, {15: ep}, code_commit_url=args.commit)
        write_submission(args.out, rows, layout, {15: ep})
    return root, scratch, ep, layout, loaded, calls, packer


def test_one_original_packer_callback_has_exact_filename_and_sixkey_fullN_ABI(gate, tmp_path):
    root, scratch, ep, layout, loaded, calls, packer = pack_fixture(gate, tmp_path)
    report = {}; before = gate.array_fingerprint(ep)
    gate.pack_once(root, scratch, loaded, layout, report, lambda: None, commit=COMMIT, packer=packer)
    assert len(calls) == 1 and report["official_packer_attempts"] == report["official_packer_returns"] == report["official_packer_validated"] == 1
    assert gate.array_fingerprint(ep) == before and report["packing_roundtrip"]["full_source_frames"] == 501
    assert report["scratch_Parquet"]["bytes"] > 0
    assert all(not p.stat().st_mode & 0o222 for p in (calls[0].sample, *(calls[0].episodes.iterdir())))


@pytest.mark.parametrize("fault", ["raises", "trajectory", "input_npz", "source_mesh", "badoutput"])
def test_packer_fail_or_mutation_never_validated(gate, tmp_path, fault):
    root, scratch, ep, layout, loaded, calls, packer = pack_fixture(gate, tmp_path)
    def bad(args):
        if fault == "raises": raise RuntimeError("own fixture failure")
        packer(args)
        if fault == "trajectory": ep.reconstruction.pose[0, 0] += 1
        elif fault == "input_npz":
            path = args.episodes / "episode_000015.npz"; path.chmod(0o644); path.write_bytes(b"changed")
        elif fault == "source_mesh":
            path = root / "outputs/episode_000015/cari_shared_export_v1/object_aligned.glb"; path.chmod(0o644); path.write_bytes(b"changed")
        else: args.out.write_bytes(b"invalid Parquet")
    report = {}
    with pytest.raises((ValueError, RuntimeError, OSError)):
        gate.pack_once(root, scratch, loaded, layout, report, lambda: None, commit=COMMIT, packer=bad)
    assert report.get("official_packer_validated", 0) == 0


def runtime_fixture(gate, tmp_path, monkeypatch):
    """Synthetic actual-observation-shaped JSON only, never production config."""
    helpers = {}
    for name in gate.BUILDER_HELPERS:
        path = tmp_path / "fixture_sources" / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(("own builder fixture " + name).encode()); path.chmod(0o444)
        helpers[name] = gate.identity(path)
    monkeypatch.setattr(gate, "BUILDER_HELPERS", helpers)
    image = "sha256:" + "e" * 64
    probe = dict(versions=gate.PARENT_VERSIONS.copy(), pyarrow="19.0.1", python="3.11.11",
                 CPU_import_verified=True, Torch_loaded=False, Joblib_loaded=False)
    parent = dict(probe, pyarrow=None, CPU_import_verified=False)
    wheel = {key: value for key, value in gate.ARROW_WHEEL.items() if key != "version"}
    receipt = dict(stage="world_reward_official_pack_CPU_image_build", status="pass", phase="complete",
        producer_revision=gate.BUILDER_REVISION, script_sha256=helpers["infra/official_pack_image_build.py"]["sha256"],
        image_id=image, image_tag="world-reward/official-pack-cpu:0.1", base_image_id=gate.BASE_ID,
        source_helpers=helpers, budget_seconds=300, download_budget_seconds=120,
        GPU_used=False, Torch_loaded=False, Joblib_loaded=False, challenge_inputs_used=False,
        data_or_models_read=False, secret_material_used=False, build_network="none", base_pull_performed=False,
        parent_unchanged_verified=True, temporary_wheel_context_removed=True, runtime_dependency_installation=False,
        only_pyarrow_added=True, source_helpers_rehashed=True, wheel_download_verified=True,
        wheel=wheel, parent_versions=parent, CPU_probe=probe)
    raw = json.dumps(receipt).encode(); receipt_id = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    pins = dict(schema="world-reward-official-pack-runtime-pins-v1", image_id=image, base_id=gate.BASE_ID,
        build_receipt=receipt_id | dict(producer_revision=gate.BUILDER_REVISION, script_sha256=receipt["script_sha256"]),
        wheel=gate.ARROW_WHEEL.copy(), source_helpers=helpers, versions=gate.RUNTIME_VERSIONS.copy())
    return pins, receipt, raw


def lifecycle_fixture(gate, tmp_path, monkeypatch):
    for name in ("torch", "joblib"): monkeypatch.delitem(sys.modules, name, raising=False)
    root = tmp_path / "root"; root.mkdir(); code = tmp_path / "code"; code.mkdir()
    runtime_pins, runtime_receipt, runtime_raw = runtime_fixture(gate, tmp_path, monkeypatch)
    for name in gate.BUILDER_HELPERS:
        path = code / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((tmp_path / "fixture_sources" / name).read_bytes()); path.chmod(0o444)
    runtime_path = code / gate.RUNTIME_PINS; runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime_pins)); runtime_path.chmod(0o444)
    receipt_path = root / gate.BUILD_RECEIPT; receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(runtime_raw); receipt_path.chmod(0o444)
    monkeypatch.setattr(gate, "check_runtime_packages", lambda expected: expected.copy())
    for name in ("infra/official_track1_pack_gate.py", "infra/run_official_track1_pack_gate.sh", "infra/cari_shared_episode_loader.py", "src/world_reward/submission.py", "infra/official_pack_geometry.py"):
        p = code / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((ROOT / name).read_bytes()); p.chmod(0o444)
    mesh = root / "outputs/episode_000015/cari_shared_export_v1/object_aligned.glb"; mesh.parent.mkdir(parents=True)
    files = {}
    for name in ("report.json", "trajectory.npz", "native_parameters.npz", "target.npy", "object_aligned.glb"):
        p = mesh.parent / name; p.write_bytes(("opaque fixture " + name).encode()); p.chmod(0o444); files[name] = gate.identity(p)
    spec = dict(episode_index=15, total_frames=501, camera_name="front_stereo_camera_left", height=1152, width=1536)
    pins = dict(schema="world-reward-cari-shared-export-pins-v1", clip_spec=spec, export_files=files,
        export=files["report.json"] | dict(producer_revision="b" * 40, script_sha256="c" * 64))
    pin = code / "configs/cari_clip_000015_shared_export_pins.json"; pin.parent.mkdir(exist_ok=True); pin.write_text(json.dumps(pins)); pin.chmod(0o444)
    sample = root / gate.SAMPLE_RELATIVE; sample.parent.mkdir(parents=True); sample.write_bytes(b"opaque own sample fixture"); sample.chmod(0o444)
    monkeypatch.setattr(gate, "SAMPLE_SHA", gate.identity(sample)["sha256"])
    monkeypatch.setattr(gate, "SAMPLE_BYTES", sample.stat().st_size)
    monkeypatch.setattr(gate, "official_sources", lambda _: {"source": "own already-verified source gate fixture"})
    layout = Track1Layout.from_row_ids(ids())
    monkeypatch.setattr(gate, "selected_layout", lambda original, selected: original)
    ep = episode(501)
    manifest = dict(stage="world_reward_shared_native_episode_consumer", episode_index=15, frames=501,
        original_frame_coverage_verified=True, integrity_and_schema_verified=True, native_direct_export_consumed=True,
        old_LM_conversion_used=False, numerical_geometry_independently_reverified=False, quality_verified=False,
        challenge_performance_verified=False, submission_eligibility_verified=False, submission_eligible=False,
        final_Parquet_produced=False, export_report_sha256=pins["export"]["sha256"], export_files=files)
    def consumer(*args): return SimpleNamespace(episode=ep, manifest=manifest)
    def packer(args):
        rows = assemble_submission(layout, {15: ep}, code_commit_url=args.commit)
        write_submission(args.out, rows, layout, {15: ep})
    out = root / gate.output_relative(15); out.mkdir(parents=True)
    return root, out, code, pin, sample, ep, manifest, consumer, packer, lambda _: layout


def test_frozen_receipt_cleanup_and_engineering_only_claims(gate, tmp_path, monkeypatch):
    root, out, code, _, _, ep, _, consumer, packer, reader = lifecycle_fixture(gate, tmp_path, monkeypatch)
    report = gate.execute(root, out, code, 15, "d" * 40, consumer=consumer, packer=packer, template_reader=reader)
    assert report["status"] == "pass" and report["scratch_removed"] and set(p.name for p in out.iterdir()) == {"report.json"}
    assert report["frames"] == 501 and report["official_packer_validated"] == 1
    assert report["image_id"] != gate.BASE_ID and report["parent_image_id"] == gate.BASE_ID
    assert report["actual_package_versions"] == gate.RUNTIME_VERSIONS and report["runtime_pins_build_receipt_rehashed"]
    assert report["original_sample_prediction_columns_read"] is False
    assert all(report[key] is False for key in ("GPU_used", "numerical_geometry_independently_reverified", "quality_verified", "submission_eligible", "final_Parquet_produced", "complete_challenge_submission_created"))
    assert report["Kaggle_upload_calls"] == report["model_calls"] == report["optimizer_calls"] == report["render_calls"] == 0
    assert json.loads((out / "report.json").read_text()) == report and not (out / "report.json").stat().st_mode & 0o222


@pytest.mark.parametrize("fault", ["packer", "pin", "sample", "extra", "source", "quality", "frames", "export_payload"])
def test_fail_cleanup_no_false_receipt_and_no_overwrite(gate, tmp_path, monkeypatch, fault):
    root, out, code, pin, sample, ep, manifest, consumer, packer, reader = lifecycle_fixture(gate, tmp_path, monkeypatch)
    def bad(args):
        if fault == "packer": raise RuntimeError("own packer fixture failure")
        packer(args)
        if fault in ("pin", "sample", "export_payload"):
            p = pin if fault == "pin" else sample if fault == "sample" else root / "outputs/episode_000015/cari_shared_export_v1/target.npy"
            p.chmod(0o644); p.write_bytes(b"changed"); p.chmod(0o444)
        elif fault == "source":
            p = code / "src/world_reward/submission.py"; p.chmod(0o644); p.write_bytes(b"changed"); p.chmod(0o444)
        elif fault == "extra": (out / "unexpected.npz").write_bytes(b"not permitted")
    if fault == "quality": manifest["quality_verified"] = True
    elif fault == "frames": object.__setattr__(ep, "total_video_frames", 500)
    with pytest.raises((ValueError, RuntimeError)):
        gate.execute(root, out, code, 15, "d" * 40, consumer=consumer, packer=bad, template_reader=reader)
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "fail" and report["quality_verified"] is False and report["final_Parquet_produced"] is False
    assert not list(out.glob(".official-pack-smoke-*")) and not list(out.glob("*.parquet"))
    with pytest.raises(ValueError): gate.execute(root, out, code, 15, "d" * 40)


def test_pinned_official_sixfile_namespace_closure_exact_before_import(gate, tmp_path, monkeypatch):
    kit = tmp_path / "vendor/v2d_submission_kit"; expected = {}
    for name in gate.OFFICIAL_SOURCES:
        p = kit / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(("own file fixture " + name).encode()); p.chmod(0o644)
        expected[name] = gate.identity(p, immutable=False)
    monkeypatch.setattr(gate, "OFFICIAL_SOURCES", expected)
    assert gate.official_sources(tmp_path) == expected
    p = kit / "v2dlb/__init__.py"; p.write_bytes(b"unexpected initializer"); p.chmod(0o444)
    with pytest.raises(ValueError, match="namespace"): gate.official_sources(tmp_path)
    p.unlink(); changed = kit / "tools/pack_reconstruction.py"; changed.chmod(0o644); changed.write_bytes(b"changed")
    with pytest.raises(ValueError): gate.official_sources(tmp_path)


def test_actual_six_source_ids_and_public_commit(gate):
    assert len(gate.OFFICIAL_SOURCES) == 6 and gate.SAMPLE_BYTES == 3495851
    assert gate.OFFICIAL_SOURCES["tools/pack_reconstruction.py"] == dict(bytes=11149, sha256="d197a2110c5f52dd08ac65c27aa3d519799a166ab9a4ea06b3f6ceb163f0d6b1")
    assert gate.code_commit_url("d" * 40) == COMMIT
    assert all(len(row["sha256"]) == 64 for row in gate.OFFICIAL_SOURCES.values())
    for invalid in ("main", "d" * 39, "D" * 40, True):
        with pytest.raises(ValueError): gate.code_commit_url(invalid)


@pytest.mark.parametrize("args", [[], ["--episode", "30"], ["--ep", "15"], ["--episode", "15", "--episode", "16"], ["--episode", "15", "--commit", "main"]])
def test_explicit_episode_only_no_implicit_commit_or_path_override(gate, args):
    with pytest.raises(SystemExit): gate.parser().parse_args(args)


def test_fresh_actual_consumer_peer_signature_no_model_import():
    source = f"""
import inspect,sys
sys.path[:0]=[{str(ROOT / 'src')!r},{str(ROOT / 'infra')!r}]
import official_track1_pack_gate as gate
assert 'numpy' not in sys.modules and 'pandas' not in sys.modules
import cari_shared_episode_loader as loader
assert str(inspect.signature(loader.load_shared_track1_episode))=='(root, code, spec, export_pins)'
assert 'torch' not in sys.modules and 'joblib' not in sys.modules
assert gate.OFFICIAL_SOURCES['v2dlb/mhr_metrics.py']['bytes']==25647
print('actual CPU consumer/packer-gate API PASS')
"""
    result = subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "actual CPU consumer/packer-gate API PASS"


def test_bash_syntax_exact_selected_mounts_no_GPU_originalsample_official_reader():
    path = ROOT / "infra/run_official_track1_pack_gate.sh"; subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "--gpus" not in text and "--network none --memory 4g --cpus 2" in text
    assert "303s docker run" in text and "BASH_SOURCE[0]" in text and "source-sha256" in text
    assert "world-reward/official-pack-cpu:0.1" in text and "official_pack_runtime_pins.json" in text
    assert text.index("load_runtime(root,code)") < text.index('mkdir "$OUT"')
    assert "src=$ROOT/results/official-pack-image-build.json" in text
    assert "data/track_1_sample_submission.parquet" in text and "v2dlb/mhr_metrics.py" in text
    assert "src=$ROOT/data" not in text and "src=$ROOT/vendor/v2d_submission_kit,dst=" not in text
    assert "prepare forward refined export" in text and "src=$OUT,dst=$OUT" in text
    assert "source_paths(spec,object_source=source_profile(pins))" in text


def test_original_export_source_flag_fresh_namespace_and_no_arbitrary_paths(gate):
    assert gate.parser().parse_args(['--episode','21','--original-export-source']).original_export_source is True
    assert gate.output_relative(21,revision='a'*40)=='outputs/episode_000021/official_track1_pack_smoke_source_'+'a'*40
    with pytest.raises(SystemExit):gate.parser().parse_args(['--episode','21','--original-export-source','--original-export-source'])
    with pytest.raises(ValueError):gate.output_relative(21,revision='../old')


def test_host_source_paths_bootstrap_stdlib_only():
    text = (ROOT / "infra/run_official_track1_pack_gate.sh").read_text()
    source = text.split("<<'PYPATHS'\n", 1)[1].split("\nPYPATHS", 1)[0]
    program = f"""
import builtins,sys
sys.path[:0]=[{str(ROOT / 'src')!r},{str(ROOT / 'infra')!r}]
real=builtins.__import__
def guarded(name,*a,**k):
 if name.split('.')[0] in {{'numpy','torch','joblib','pyarrow','pandas','trimesh'}}:raise AssertionError(name)
 return real(name,*a,**k)
builtins.__import__=guarded
sys.argv=['bootstrap',{str(ROOT / 'configs/cari_clip_000015_input_pins.json')!r},'15']
exec({source!r},{{}})
"""
    result = subprocess.run([sys.executable, "-S", "-c", program], check=True, capture_output=True, text=True)
    assert len(result.stdout.splitlines()) == 15


def test_actual_runtime_archive_closure_control_cap():
    spec = importlib.util.spec_from_file_location("official_packer_archive", ROOT / "infra/azure_job.py")
    launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    files = {str(p.relative_to(ROOT)): p.read_bytes() for base in ("infra", "src", "configs") for p in (ROOT / base).rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    selected = launcher.runtime_bundle_paths(files, "infra/run_official_track1_pack_gate.sh")
    assert set(selected) >= {"infra/official_track1_pack_gate.py", "infra/run_official_track1_pack_gate.sh", "infra/cari_shared_episode_loader.py", "infra/cari_full_export.py", "infra/cari_full_refine.py", "infra/cari_full_forward.py", "infra/run_cari96_prepare.sh", "src/world_reward/submission.py", "infra/official_pack_image_build.py", "infra/run_official_pack_image_build.sh"}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in selected:
            entry = tarfile.TarInfo(name); entry.size = len(files[name]); entry.mode = 0o444
            archive.addfile(entry, io.BytesIO(files[name]))
    raw = stream.getvalue()
    compressed = lzma.compress(raw, preset=6)
    digest = hashlib.sha256(compressed).hexdigest()
    # Real large source closures use the existing bounded GitHub transport,
    # not a provenance-trimming waiver of the independent inline payload cap.
    descriptor = launcher.github_archive_descriptor(raw, "a" * 40, digest)
    assert {row["path"] for row in descriptor["files"]} == set(selected)
    assert len(json.dumps(descriptor, separators=(",", ":")).encode()) <= launcher.STAGED_SCRIPT_BYTES
    commands = launcher.transport_commands("", digest, raw, "a" * 40,
        "infra/run_official_track1_pack_gate.sh", "official-packer-source-test", ["--episode", "15"], github_source=True)
    assert len(commands) == 1 and len(commands[0][2].encode()) <= launcher.STAGED_SCRIPT_BYTES
    if (len(compressed) + 2) // 3 * 4 > launcher.MAX_CODE_CONTROL_BYTES:
        with pytest.raises(RuntimeError, match="256KB"):
            launcher.encoded_runtime_archive(raw)
    else:
        encoded, observed = launcher.encoded_runtime_archive(raw)
        assert len(encoded) <= launcher.MAX_CODE_CONTROL_BYTES and observed == digest


def test_exact_readonly_bind_mounts_not_filemode_assumption(gate, tmp_path):
    mounts = tmp_path / "mountinfo"
    first = tmp_path / "code"; second = tmp_path / "source with space.py"
    encoded = str(second).replace(" ", r"\040")
    mounts.write_text(f"1 0 0:1 / {first} ro,relatime - ext4 /dev/disk rw\n2 0 0:1 / {encoded} ro - ext4 /dev/disk rw\n")
    gate.require_readonly_mounts((first, second), mountinfo=mounts)
    for invalid in (tmp_path / "notmounted", first / "unmounted_child.py"):
        with pytest.raises(ValueError): gate.require_readonly_mounts((invalid,), mountinfo=mounts)
    mounts.write_text(f"1 0 0:1 / {first} rw - ext4 /dev/disk rw\n")
    with pytest.raises(ValueError): gate.require_readonly_mounts((first,), mountinfo=mounts)


def test_original_futureannotated_pack_signature_and_lazy_module_binding(gate, tmp_path, monkeypatch):
    kit = tmp_path / "vendor/v2d_submission_kit"
    sources = {}
    for name in gate.OFFICIAL_SOURCES:
        path = kit / name; path.parent.mkdir(parents=True, exist_ok=True)
        text = "# own empty external-source ABI fixture\n"
        if name == "tools/pack_reconstruction.py":
            text = "from __future__ import annotations\ndef pack_track1(args) -> None:\n    from v2dlb import report_schema, mesh_budget, mhr_submission, mesh_common, mhr_metrics\n"
        path.write_text(text); sources[name] = gate.identity(path, immutable=False)
    monkeypatch.setattr(gate, "OFFICIAL_SOURCES", sources)
    for name in tuple(sys.modules):
        if name == "v2dlb" or name.startswith("v2dlb."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in ("torch", "joblib"): monkeypatch.delitem(sys.modules, name, raising=False)
    original_path = list(sys.path)
    try:
        original = gate.load_official_packer(tmp_path)
        assert original.__annotations__["return"] == "None"
        with pytest.raises(ValueError, match="Every original lazy"):
            gate.verify_official_imports(tmp_path, complete=True)
        original(SimpleNamespace())
        assert len(gate.verify_official_imports(tmp_path, complete=True)) == 5
        path = kit / "v2dlb/mhr_metrics.py"; path.write_bytes(b"changed")
        with pytest.raises(ValueError): gate.verify_official_imports(tmp_path, complete=True)
    finally:
        sys.path[:] = original_path
        for name in tuple(sys.modules):
            if name == "v2dlb" or name.startswith("v2dlb."):
                sys.modules.pop(name)


def test_runtime_template_strict_new_image_build_and_unchanged_parent(gate, tmp_path, monkeypatch):
    pins, receipt, _ = runtime_fixture(gate, tmp_path, monkeypatch)
    gate.validate_runtime_pins(pins)
    summary = gate.validate_runtime(pins, receipt)
    assert summary["image_id"] == pins["image_id"] != gate.BASE_ID
    assert summary["versions"] == gate.RUNTIME_VERSIONS and summary["GPU_used"] is False


@pytest.mark.parametrize("fault", ["baseimage", "imagesecret", "schema", "extra", "revision", "script", "wheelbytes", "wheelsha", "helper", "arrowversion", "parentversion", "flagint", "parentarrow", "noarrow", "GPU", "Torch", "Joblib", "changedparent", "dependencies", "network", "sourcehash", "wrongstage", "failed", "python"])
def test_runtime_provenance_failclosed_no_fake_image_or_version(gate, tmp_path, monkeypatch, fault):
    pins, receipt, _ = runtime_fixture(gate, tmp_path, monkeypatch)
    if fault == "baseimage": pins["image_id"] = gate.BASE_ID
    elif fault == "imagesecret": pins["image_id"] = "Bearer do-not-accept-secret-looking-image"
    elif fault == "schema": pins["schema"] = "old"
    elif fault == "extra": pins["token"] = "forbidden"
    elif fault == "revision": pins["build_receipt"]["producer_revision"] = "a" * 40
    elif fault == "script": pins["build_receipt"]["script_sha256"] = "a" * 64
    elif fault == "wheelbytes": pins["wheel"]["bytes"] = True
    elif fault == "wheelsha": pins["wheel"]["sha256"] = "a" * 64
    elif fault == "helper": pins["source_helpers"] = {}
    elif fault == "arrowversion": pins["versions"]["pyarrow"] = "19.0.2"
    elif fault == "parentversion": receipt["parent_versions"]["versions"]["numpy"] = "2.0.0"
    elif fault == "flagint": receipt["only_pyarrow_added"] = 1
    elif fault == "parentarrow": receipt["parent_versions"]["pyarrow"] = "19.0.1"
    elif fault == "noarrow": receipt["CPU_probe"]["pyarrow"] = None
    elif fault in ("GPU", "Torch", "Joblib"): receipt[fault + "_used" if fault == "GPU" else fault + "_loaded"] = True
    elif fault == "changedparent": receipt["parent_unchanged_verified"] = False
    elif fault == "dependencies": receipt["runtime_dependency_installation"] = True
    elif fault == "network": receipt["build_network"] = "host"
    elif fault == "sourcehash": receipt["source_helpers"] = {}
    elif fault == "wrongstage": receipt["stage"] = "different_build"
    elif fault == "failed": receipt["status"] = "fail"
    else: receipt["CPU_probe"]["python"] = "3.11.12"
    with pytest.raises(ValueError): gate.validate_runtime(pins, receipt)


def test_actual_package_metadata_check_no_model_import(gate, monkeypatch):
    import importlib.metadata
    calls = []
    def version(name): calls.append(name); return gate.RUNTIME_VERSIONS[name]
    monkeypatch.setattr(importlib.metadata, "version", version)
    for name in ("torch", "joblib"): monkeypatch.delitem(sys.modules, name, raising=False)
    assert gate.check_runtime_packages(gate.RUNTIME_VERSIONS) == gate.RUNTIME_VERSIONS
    assert calls == list(gate.RUNTIME_VERSIONS)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "wrong")
    with pytest.raises(ValueError): gate.check_runtime_packages(gate.RUNTIME_VERSIONS)


def test_runtime_pin_and_receipt_hash_before_json_no_builder_execution(gate, tmp_path, monkeypatch):
    root, out, code, *_ = lifecycle_fixture(gate, tmp_path, monkeypatch)
    runtime = gate.load_runtime(root, code)
    assert runtime["summary"]["only_pyarrow_added"]
    path = root / gate.BUILD_RECEIPT; path.chmod(0o644); path.write_bytes(b"changed"); path.chmod(0o444)
    read = gate.strict_json
    def guarded(path):
        if path == root / gate.BUILD_RECEIPT: pytest.fail("Receipt JSON read before exact hash")
        return read(path)
    monkeypatch.setattr(gate, "strict_json", guarded)
    with pytest.raises(ValueError): gate.load_runtime(root, code)
    assert "official_pack_image_build" not in sys.modules


def test_missing_runtime_config_fails_before_receipt_or_callback(gate, tmp_path, monkeypatch):
    root, out, code, _, _, _, _, consumer, packer, reader = lifecycle_fixture(gate, tmp_path, monkeypatch)
    (code / gate.RUNTIME_PINS).unlink()
    with pytest.raises(FileNotFoundError):
        gate.execute(root, out, code, 15, "d" * 40, consumer=consumer, packer=packer, template_reader=reader)
    assert not list(out.iterdir())


def test_actual_runtime_host_bootstrap_all_stdlib_no_builder_import(gate, tmp_path, monkeypatch):
    root, out, code, *_ = lifecycle_fixture(gate, tmp_path, monkeypatch)
    text = (ROOT / "infra/run_official_track1_pack_gate.sh").read_text()
    source = text.split("<<'PYRUNTIME'\n", 1)[1].split("\nPYRUNTIME", 1)[0]
    # Execute the actual snippet against the owned helper with only its pure
    # source maps swapped for generated fixture identities, no fake producer.
    import builtins
    real = builtins.__import__
    def guarded(name, *a, **k):
        if name.split('.')[0] in {'numpy', 'scipy', 'torch', 'joblib', 'pyarrow', 'pandas', 'trimesh', 'official_pack_image_build'}:
            raise AssertionError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setitem(sys.modules, 'official_track1_pack_gate', gate)
    monkeypatch.setattr(sys, 'argv', ['bootstrap', str(root), str(code)])
    namespace = {}
    exec(source, namespace)
    assert namespace['load_runtime'] is gate.load_runtime


def test_isolated_original_source_function_does_not_execute_module_imports(gate, tmp_path):
    path = tmp_path / 'own_original_function.py'
    path.write_text('from __future__ import annotations\nimport forbidden_model\n\ndef selected(path: Path):\n    return path, np\n')
    marker = object(); namespace = dict(Path=Path, np=marker)
    selected = gate.isolated_source_function(path, 'selected', namespace)
    assert selected('owned') == ('owned', marker)
    assert selected.__code__.co_filename == str(path)
    assert selected.__code__.co_firstlineno == 4
    assert 'forbidden_model' not in sys.modules
    for invalid in ['@decorator\ndef selected(path):\n return path\n',
                    'def selected(path, other):\n return path\n',
                    'def selected(path):\n return path\ndef selected(path):\n return path\n']:
        path.write_text(invalid)
        with pytest.raises(ValueError): gate.isolated_source_function(path, 'selected', namespace)


def scene_authority_fixture(gate, tmp_path, monkeypatch, fault=None):
    # Other suite modules may legitimately import Joblib. This fixture models
    # the production fresh CPU process without changing production import guards.
    for name in ("torch", "joblib"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    ep = episode(97)
    native = tmp_path / 'vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d'
    sources = {
        'lib_mhr/contact.py': '''from __future__ import annotations
import forbidden_native_models
def load_object_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh_or_scene = trimesh.load(path, force="scene", process=False)
    if isinstance(mesh_or_scene, trimesh.Scene):
        geometries = [geom for geom in mesh_or_scene.dump(concatenate=False) if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0]
        if not geometries:
            raise ValueError("has no mesh geometry")
        if len(geometries) == 1:
            return geometries[0]
        return trimesh.util.concatenate(geometries)
    return mesh_or_scene
''',
        'learning/training/mhr_opt_refineout.py': '''from __future__ import annotations
import forbidden_optimizer

def _load_object_vertices(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    mesh = load_object_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = None if getattr(mesh, "faces", None) is None else np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces
'''}
    observed = {}
    for name, source in sources.items():
        p = native / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(source)
        observed[name] = gate.identity(p, immutable=False)
    monkeypatch.setattr(gate, 'NATIVE_MESH_SOURCES', observed)
    meshpath = tmp_path / 'original.glb'; meshpath.write_bytes(b'own scene fixture'); meshpath.chmod(0o444)
    class Mesh:
        def __init__(self, vertices, faces): self.vertices, self.faces = vertices, faces
    first = Mesh(ep.object_vertices.astype(np.float64), ep.object_faces.copy())
    moved = Mesh(first.vertices + np.array([.5, 0, 0]), first.faces.copy())
    joined = Mesh(np.r_[first.vertices, moved.vertices], np.r_[first.faces, moved.faces + len(first.vertices)])
    ep = replace(ep, object_vertices=joined.vertices.astype(np.float32), object_faces=joined.faces)
    class Graph:
        nodes_geometry = ['original', 'translated']
        def __getitem__(self, name):
            transform = np.eye(4)
            if name == 'translated': transform[0, 3] = .5
            if fault == 'transform': transform[0, 0] = np.nan
            return transform, 'geometry'
    class Scene:
        geometry = {'geometry': first}
        graph = Graph()
        def dump(self, concatenate=False):
            assert concatenate is False
            return [first] if fault == 'missing_instance' else [first, moved]
    scene = Scene()
    if fault == 'orphan': scene.geometry = dict(scene.geometry, orphan=first)
    if fault == 'nontriangle': scene.geometry = {'geometry': object()}
    def concatenate(items):
        assert len(items) == 2
        return joined
    def load(path, *, force, process):
        assert Path(path) == meshpath and process is False
        if force == 'scene': return scene
        assert force == 'mesh'
        return first if fault == 'official_drops_instance' else joined
    fake = SimpleNamespace(Trimesh=Mesh, Scene=Scene, load=load, util=SimpleNamespace(concatenate=concatenate))
    monkeypatch.setitem(sys.modules, 'trimesh', fake)
    if fault == 'native_geometry': ep.object_vertices[0, 0] += .01
    return meshpath, ep


def test_full_scene_instances_and_actual_native_FP32_source_replay(gate, tmp_path, monkeypatch):
    path, ep = scene_authority_fixture(gate, tmp_path, monkeypatch)
    vertices, faces, proof = gate.load_mesh_authorities(tmp_path, path, ep)
    assert vertices.dtype == np.float64 and faces.dtype == np.int64
    assert proof['scene_instances'] == 2 and proof['scene_geometries'] == 1
    assert proof['full_scene_instances_verified'] and proof['native_geometry_byte_exact']
    assert proof['original_native_FP32_loader_replayed'] and proof['model_imports'] is False
    assert 'forbidden_native_models' not in sys.modules and 'forbidden_optimizer' not in sys.modules


@pytest.mark.parametrize('fault', ['missing_instance', 'orphan', 'nontriangle', 'official_drops_instance', 'native_geometry', 'transform'])
def test_scene_source_authorities_failclosed(gate, tmp_path, monkeypatch, fault):
    path, ep = scene_authority_fixture(gate, tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError): gate.load_mesh_authorities(tmp_path, path, ep)


def test_original_native_scene_source_hash_checked_before_execution(gate, tmp_path, monkeypatch):
    path, ep = scene_authority_fixture(gate, tmp_path, monkeypatch)
    source = tmp_path / 'vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py'
    source.write_text('raise RuntimeError("must not execute altered source")')
    with pytest.raises(ValueError, match='Exact original native'):
        gate.load_mesh_authorities(tmp_path, path, ep)


@pytest.mark.parametrize('fault', ['repeated_index', 'collinear', 'repeated_coordinates'])
def test_zero_area_tail_cannot_hide_nonfirst_vertex_padding(gate, fault):
    ep, layout, rows = output_rows(gate)
    values = rows.values.copy()
    vertices = np.flatnonzero(layout.keys[:, 2] == 6)
    faces = np.flatnonzero(layout.keys[:, 2] == 7)
    values[vertices[-3]] = [.7, .8, .9]
    values[vertices[-2]] = [.3, .2, .1]
    values[vertices[-1]] = 2 * values[vertices[-2]] - values[vertices[-3]]
    if fault == 'repeated_index': values[faces[-1]] = [4093, 4093, 4094]
    elif fault == 'repeated_coordinates':
        values[vertices[-2]] = values[vertices[-3]]
        values[faces[-1]] = [4093, 4094, 4095]
    else:
        # Exactly dyadic coordinates, so all three points are truly collinear.
        values[vertices[-3:]] = [[.25, .25, .25], [.5, .5, .5], [.75, .75, .75]]
        values[faces[-1]] = [4093, 4094, 4095]
    with pytest.raises(ValueError, match='padding face'):
        gate.verify_output(replace(rows, values=values), layout, ep, COMMIT)
