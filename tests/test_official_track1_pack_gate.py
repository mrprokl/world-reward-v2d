"""Tiny official-packer boundary fixtures: no real data/models/official source."""
from dataclasses import asdict, replace
import csv
import hashlib
import importlib.util
import io
import json
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


def lifecycle_fixture(gate, tmp_path, monkeypatch):
    for name in ("torch", "joblib"): monkeypatch.delitem(sys.modules, name, raising=False)
    root = tmp_path / "root"; root.mkdir(); code = tmp_path / "code"; code.mkdir()
    for name in ("infra/official_track1_pack_gate.py", "infra/run_official_track1_pack_gate.sh", "infra/cari_shared_episode_loader.py", "src/world_reward/submission.py"):
        p = code / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((ROOT / name).read_bytes()); p.chmod(0o444)
    mesh = root / "outputs/episode_000015/cari_shared_export_v1/object_aligned.glb"; mesh.parent.mkdir(parents=True)
    files = {}
    for name in ("report.json", "trajectory.npz", "native_parameters.npz", "target.npy", "object_aligned.glb"):
        p = mesh.parent / name; p.write_bytes(("opaque fixture " + name).encode()); p.chmod(0o444); files[name] = gate.identity(p)
    spec = dict(episode_index=15, total_frames=501, camera_name="front_stereo_camera_left", height=1152, width=1536)
    pins = dict(schema="world-reward-cari-shared-export-pins-v1", clip_spec=spec, export_files=files,
        export=files["report.json"] | dict(producer_revision="b" * 40, script_sha256="c" * 64))
    pin = code / "configs/cari_clip_000015_shared_export_pins.json"; pin.parent.mkdir(); pin.write_text(json.dumps(pins)); pin.chmod(0o444)
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
    assert "data/track_1_sample_submission.parquet" in text and "v2dlb/mhr_metrics.py" in text
    assert "src=$ROOT/data" not in text and "src=$ROOT/vendor/v2d_submission_kit,dst=" not in text
    assert "prepare forward refined export" in text and "src=$OUT,dst=$OUT" in text


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
    assert set(selected) >= {"infra/official_track1_pack_gate.py", "infra/run_official_track1_pack_gate.sh", "infra/cari_shared_episode_loader.py", "infra/cari_full_export.py", "infra/cari_full_refine.py", "infra/cari_full_forward.py", "infra/run_cari96_prepare.sh", "src/world_reward/submission.py"}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in selected:
            entry = tarfile.TarInfo(name); entry.size = len(files[name]); entry.mode = 0o444
            archive.addfile(entry, io.BytesIO(files[name]))
    encoded, _ = launcher.encoded_runtime_archive(stream.getvalue())
    assert len(encoded) <= 160000


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
