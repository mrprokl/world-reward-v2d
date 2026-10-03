"""All files are tiny fake RGB payloads; ffprobe is injected, never run on them."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1] / "infra/track1_readiness.py"
    spec = importlib.util.spec_from_file_location("test_world_reward_track1_readiness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded(value):
    return json.dumps(value).encode("utf-8")


def json_lines(rows):
    return b"\n".join(encoded(row) for row in rows) + b"\n"


def make_fixture(gate, root):
    lengths = [552] * 27 + [553] * 3
    assert sum(lengths) == gate.FRAMES
    metadata = [dict(episode_index=i, object_prompt="official procedural object") for i in range(30)]
    episodes = [dict(episode_index=i, length=n) for i, n in enumerate(lengths)]
    contents = {gate.METADATA[0]: json_lines(episodes), gate.METADATA[1]: json_lines(metadata)}
    contents.update({gate.video_relative(i): f"tiny encoded fixture {i}".encode() for i in range(30)})
    records = []
    for name, payload in contents.items():
        path = root / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append(dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    manifest = dict(track="track_1", repo_id=gate.REPOSITORY, revision=gate.DATASET_REVISION,
        episodes=30, frames=16563, files=records)
    path = root / "results/input-manifest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded(manifest))
    return SimpleNamespace(root=root, manifest=manifest, manifest_path=path, lengths=lengths,
        episodes=episodes, metadata=metadata, probes=[])


@pytest.fixture
def data(gate, tmp_path):
    return make_fixture(gate, tmp_path)


def stream(gate, length):
    return dict(streams=[dict(width=gate.WIDTH, height=gate.HEIGHT, nb_frames=str(length), r_frame_rate="30/1")])


def audit(gate, data, free=None):
    def probe(path, timeout):
        assert path.is_file() and timeout > 0
        index = int(path.stem.split("_")[-1])
        data.probes.append(index)
        return stream(gate, data.lengths[index])
    return gate.audit(data.root, probe=probe, disk_usage=lambda root: SimpleNamespace(free=300*gate.GIB if free is None else free))


def rewrite(data, relative, payload, *, update_hash=True):
    path = data.root / "data" / relative
    path.write_bytes(payload)
    if update_hash:
        row = next(row for row in data.manifest["files"] if row["path"] == relative)
        row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        data.manifest_path.write_bytes(encoded(data.manifest))


def test_complete_ordered_original_structural_report(gate, data):
    report = audit(gate, data)
    assert report["status"] == "pass" and report["structural_ready"] and report["resource_ready"]
    assert report["episodes"] == 30 and report["frames"] == 16563
    assert data.probes == list(range(30))
    assert report["selected_files_verified"] == 32 and report["manifest_files"] == 32
    assert [c["length"] for c in report["clips"]] == data.lengths
    assert all(c["width"] == 1536 and c["height"] == 1152 and c["fps"] == 30 for c in report["clips"])
    assert all(all(s["state"] == "absent" and s["trusted"] is False for s in c["frontend_outputs"].values()) for c in report["clips"])
    assert report["all_selected_hashes_verified_before_metadata_parse"] is True
    assert report["model_files_read"] == report["video_frames_decoded"] == report["inference_calls"] == 0
    assert report["ground_truth_used"] is report["frontend_outputs_verified"] is report["challenge_performance_verified"] is False
    assert report["submission_eligible"] is False
    assert report["metadata_files"][gate.METADATA[0]]["bytes"] > 0
    assert "object_prompt" not in json.dumps(report)


def test_actual_style_extra_allowlisted_files_are_not_opened(gate, data, monkeypatch):
    extras = [f"track_1/data/chunk-000/episode_{i:06d}.parquet" for i in range(30)]
    extras += ["track_1/meta/info.json", "track_1/meta/tasks.jsonl", "track_1/meta/episodes_stats.jsonl", "track_1/README.md", "track_1/.gitignore"]
    data.manifest["files"] += [dict(path=name, bytes=7, sha256="f"*64) for name in extras]
    data.manifest_path.write_bytes(encoded(data.manifest))
    original = gate.file_identity
    def selected_only(path, **kwargs):
        assert str(path) not in {str(data.root / "data" / name) for name in extras}
        return original(path, **kwargs)
    monkeypatch.setattr(gate, "file_identity", selected_only)
    report = audit(gate, data)
    assert report["manifest_files"] == 67 and report["unselected_allowlisted_files_not_read"] == 35
    assert report["selected_files_verified"] == 32


@pytest.mark.parametrize("name", ["track_2/meta/episodes.jsonl", "track_3/videos/a.mp4", "/track_1/meta/info.json",
    "track_1/../track_2/meta/info.json", "track_1/meta//info.json", "track_1/meta/./info.json",
    "track_1\\meta\\info.json", "track_1/meta/ground_truth.jsonl", "track_1/multiview/a.mp4",
    "track_1/videos/chunk-000/observation.images.exo_camera/episode_000030.mp4"])
def test_disallowed_or_extra_video_manifest_path(gate, data, name):
    data.manifest["files"].append(dict(path=name, bytes=7, sha256="f"*64))
    with pytest.raises(ValueError):
        gate.validate_manifest(data.manifest)


@pytest.mark.parametrize("key,value", [("track", "track_2"), ("repo_id", "wrong/repo"), ("revision", "main"),
    ("episodes", True), ("episodes", 29), ("frames", 16563.0), ("frames", 16562)])
def test_manifest_provenance_and_typed_totals(gate, data, key, value):
    data.manifest[key] = value
    with pytest.raises(ValueError):
        gate.validate_manifest(data.manifest)


@pytest.mark.parametrize("key,value", [("bytes", True), ("bytes", 1.0), ("bytes", 0), ("bytes", -1),
    ("sha256", "a"*63), ("sha256", "A"*64), ("sha256", None)])
def test_manifest_records_strict_scalar_types(gate, data, key, value):
    data.manifest["files"][0][key] = value
    with pytest.raises(ValueError):
        gate.validate_manifest(data.manifest)


def test_duplicate_or_missing_selected_path(gate, data):
    data.manifest["files"].append(copy.deepcopy(data.manifest["files"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        gate.validate_manifest(data.manifest)
    data.manifest["files"].pop()
    data.manifest["files"].pop(5)
    with pytest.raises(ValueError, match="one of each"):
        gate.validate_manifest(data.manifest)


@pytest.mark.parametrize("role", ["metadata", "last_video"])
def test_all_hashes_gate_before_metadata_parse_or_probe(gate, data, monkeypatch, role):
    relative = gate.METADATA[0] if role == "metadata" else gate.video_relative(29)
    rewrite(data, relative, b"malformed JSON only if parsed", update_hash=False)
    monkeypatch.setattr(gate, "parse_episode_metadata", lambda *_: pytest.fail("Metadata parsed before all32 verified"))
    with pytest.raises(ValueError, match="hash/bytes"):
        audit(gate, data)
    assert not data.probes


@pytest.mark.parametrize("kind", ["symlink_file", "symlink_ancestor", "fifo", "directory", "root_symlink"])
def test_input_paths_regular_with_no_symlink_ancestors(gate, data, tmp_path, kind):
    path = data.root / "data" / gate.video_relative(0)
    root = data.root
    if kind == "symlink_file":
        target = tmp_path / "other-file"; target.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(target)
    elif kind == "symlink_ancestor":
        parent = path.parent; target = parent.with_name("elsewhere"); parent.rename(target); parent.symlink_to(target, target_is_directory=True)
    elif kind == "root_symlink":
        root = tmp_path / "root-link"; root.symlink_to(data.root, target_is_directory=True)
    elif kind == "directory":
        path.unlink(); path.mkdir()
    else:
        import os
        path.unlink(); os.mkfifo(path)
    with pytest.raises(ValueError):
        gate.audit(root, probe=lambda *_: pytest.fail("unsafe source probed"), disk_usage=lambda _: SimpleNamespace(free=0))


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e400}', 'not json'])
def test_strict_json_duplicate_and_nonfinite(gate, text):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        gate.strict_json(text)


@pytest.mark.parametrize("role", ["episodes", "metadata"])
@pytest.mark.parametrize("change", ["duplicate", "missing", "bool_index", "out_of_range", "negative_index"])
def test_metadata_one_each_exact_identity(gate, data, role, change):
    rows = copy.deepcopy(getattr(data, role))
    if change == "duplicate": rows.append(copy.deepcopy(rows[0]))
    elif change == "missing": rows.pop()
    elif change == "bool_index": rows[0]["episode_index"] = False
    elif change == "out_of_range": rows[0]["episode_index"] = 30
    else: rows[0]["episode_index"] = -1
    episodes = rows if role == "episodes" else data.episodes
    metadata = rows if role == "metadata" else data.metadata
    with pytest.raises(ValueError):
        gate.parse_episode_metadata(json_lines(episodes).decode(), json_lines(metadata).decode())


@pytest.mark.parametrize("length", [True, 552.0, 95, 0, -96, "552", None])
def test_original_minimum_integer_lengths_no_padding(gate, data, length):
    data.episodes[0]["length"] = length
    with pytest.raises(ValueError):
        gate.parse_episode_metadata(json_lines(data.episodes).decode(), json_lines(data.metadata).decode())


@pytest.mark.parametrize("prompt", [None, "", "   ", 1, True, []])
def test_official_nonempty_prompts_only(gate, data, prompt):
    data.metadata[0]["object_prompt"] = prompt
    with pytest.raises(ValueError):
        gate.parse_episode_metadata(json_lines(data.episodes).decode(), json_lines(data.metadata).decode())


def test_wrong_total_and_unordered_metadata(gate, data):
    data.episodes[0]["length"] += 1
    with pytest.raises(ValueError, match="16563"):
        gate.parse_episode_metadata(json_lines(data.episodes).decode(), json_lines(data.metadata).decode())
    data.episodes[0]["length"] -= 1
    assert gate.parse_episode_metadata(json_lines(data.episodes[::-1]).decode(), json_lines(data.metadata[::-1]).decode()) == data.lengths


@pytest.mark.parametrize("key,value", [("width", True), ("width", 1536.0), ("height", 1080),
    ("nb_frames", 552), ("nb_frames", "N/A"), ("nb_frames", "0552"), ("nb_frames", "553"),
    ("r_frame_rate", "29/1"), ("r_frame_rate", "30000/1001"), ("r_frame_rate", "30/0"),
    ("r_frame_rate", "30"), ("r_frame_rate", 30), ("r_frame_rate", "-30/1")])
def test_video_exact_container_fields_and_no_frame_fallback(gate, key, value):
    data = stream(gate, 552)
    data["streams"][0][key] = value
    with pytest.raises(ValueError):
        gate.validate_video_metadata(data, 552)


@pytest.mark.parametrize("value", [{}, {"streams": []}, {"streams": [{}, {}]}, {"streams": {}}, {"streams": [{}]}, []])
def test_video_only_selected_stream_inventory(gate, value):
    with pytest.raises(ValueError):
        gate.validate_video_metadata(value, 552)


def test_actual_azure_ffprobe_empty_program_envelope(gate):
    value = {"programs": [], "streams": [{"width": 1536, "height": 1152,
        "r_frame_rate": "30/1", "nb_frames": "790"}]}
    assert gate.validate_video_metadata(value, 790) == dict(width=1536, height=1152, nb_frames=790, fps=30)


@pytest.mark.parametrize("programs", [None, False, {}, "", [{"streams": []}], [1]])
def test_program_metadata_not_empty_envelope_rejected(gate, programs):
    value = stream(gate, 552); value["programs"] = programs
    with pytest.raises(ValueError):
        gate.validate_video_metadata(value, 552)


def test_ffprobe_unknown_envelope_fields_rejected(gate):
    value = stream(gate, 552); value["unexpected"] = []
    with pytest.raises(ValueError):
        gate.validate_video_metadata(value, 552)


def test_exact_rational_30_hz(gate):
    value = stream(gate, 552); value["streams"][0]["r_frame_rate"] = "30000/1000"
    assert gate.validate_video_metadata(value, 552)["fps"] == 30


def test_ffprobe_metadata_only_command(gate, monkeypatch):
    calls = []
    monkeypatch.setattr(gate.shutil, "which", lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None)
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=encoded(stream(gate, 552)), stderr=b"")
    monkeypatch.setattr(gate.subprocess, "run", run)
    assert gate.probe_video(Path("/only/original.mp4"), 7) == stream(gate, 552)
    command, kwargs = calls[0]
    assert command == ["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate", "-of", "json", "/only/original.mp4"]
    assert not any("count_frames" in part or "show_frames" in part for part in command)
    assert kwargs["timeout"] == 7 and kwargs["check"] is True


def test_missing_ffprobe_fail_no_fallback(gate, monkeypatch):
    monkeypatch.setattr(gate.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="unavailable"):
        gate.probe_video(Path("/none.mp4"), 1)


def test_frontend_occupied_opaque_report_untrusted(gate, data):
    stage = data.root / "outputs/episode_000000/cari_inputs"
    stage.mkdir(parents=True)
    (stage / "report.json").write_text('malformed + claimed status pass is irrelevant')
    empty = stage.parent / "body_full"; empty.mkdir()
    unsafe = stage.parent / "depth_full"; unsafe.symlink_to(stage, target_is_directory=True)
    special = stage.parent / "automatic_masks"; special.write_bytes(b"not a directory")
    report = audit(gate, data)
    states = report["clips"][0]["frontend_outputs"]
    assert states["cari_inputs"] == dict(state="occupied", kind="directory", empty=False, trusted=False)
    assert states["body_full"]["empty"] is True
    assert states["depth_full"]["state"] == states["automatic_masks"]["state"] == "unverified"
    assert report["resources"]["next_clean_frontend_episode_index"] == 1


def test_partial_unsafe_or_empty_frontends_never_clean_next_clip(gate, data):
    (data.root / "outputs/episode_000000/body_full").mkdir(parents=True)
    stage = data.root / "outputs/episode_000001"; stage.mkdir(parents=True)
    (stage / "automatic_masks").symlink_to(data.root / "results", target_is_directory=True)
    report = audit(gate, data)
    assert report["resources"]["next_clean_frontend_episode_index"] == 2
    assert report["resources"]["execution_authorized"] is False


@pytest.mark.parametrize("sparse", ["body_smoke", "depth_smoke", "body_full/cari_adapter"])
def test_sparse_or_adapter_output_prevents_clean_frontend_route(gate, data, sparse):
    (data.root / "outputs/episode_000000" / sparse).mkdir(parents=True)
    report = audit(gate, data)
    assert len(report["clips"][0]["frontend_outputs"]) == 10
    assert report["resources"]["next_clean_frontend_episode_index"] == 1


def test_actual_frontend_launcher_exact_same_target_inventory(gate):
    root = Path(__file__).resolve().parents[1]
    text = (root / "infra/run_track1_frontends.sh").read_text()
    python = text.split("<<'PYSAFE'", 1)[1].split("\nPYSAFE", 1)[0]
    tree = ast.parse(python)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "targets" for target in node.targets))
    assert ast.literal_eval(assignment.value) == gate.FRONTENDS


def test_structural_pass_separate_low_resource_flags(gate, data):
    report = audit(gate, data, free=1)
    assert report["status"] == "pass" and report["structural_ready"] is True
    assert report["resource_ready"] is False
    assert report["resources"]["all_30_budget_available"] is False
    report = audit(gate, data, free=100*gate.GIB)
    assert report["resource_ready"] is True and report["resources"]["all_30_budget_available"] is False
    r = report["resources"]
    assert r["next_clip_required_free_bytes"] == max(20*gate.GIB, 2*r["next_clip_estimated_working_bytes"])
    assert r["measured_disk_requirement_verified"] is False


def test_all_frontends_occupied_never_claims_launch_ready(gate, data):
    for i in range(30):
        (data.root / f"outputs/episode_{i:06d}/cari_inputs").mkdir(parents=True)
    report = audit(gate, data)
    assert report["resource_ready"] is False and report["structural_ready"] is True
    assert report["resources"]["next_clip_required_free_bytes"] is None


def test_whole_stage_deadline(gate, data):
    with pytest.raises(TimeoutError):
        gate.audit(data.root, budget_seconds=-1, probe=lambda *_: pytest.fail("no probe afterdeadline"))


def test_video_changes_during_probe_rejected(gate, data):
    def probe(path, _timeout):
        path.write_bytes(b"changed")
        return stream(gate, data.lengths[0])
    with pytest.raises(ValueError, match="changed during"):
        gate.audit(data.root, probe=probe)


def test_video_changes_after_hash_before_probe_rejected(gate, data, monkeypatch):
    original = gate.parse_episode_metadata
    def parse(*args):
        result = original(*args)
        (data.root / "data" / gate.video_relative(0)).write_bytes(b"changed after all32hashes")
        return result
    monkeypatch.setattr(gate, "parse_episode_metadata", parse)
    with pytest.raises(ValueError, match="after selected"):
        audit(gate, data)
    assert not data.probes


def test_bounded_immutable_regular_source_identity(gate, tmp_path):
    path = tmp_path / "source.py"; path.write_bytes(b"stdlib fixture")
    with pytest.raises(ValueError, match="immutable"):
        gate.file_identity(path, immutable=True)
    path.chmod(0o444)
    assert gate.file_identity(path, immutable=True)["sha256"] == hashlib.sha256(b"stdlib fixture").hexdigest()
    with pytest.raises(ValueError, match="bounded"):
        gate.file_identity(path, capture=True, max_bytes=1)


def test_source_has_stdlib_only_no_stage_imports_or_writes(gate):
    source = Path(gate.__file__).read_text(); tree = ast.parse(source)
    modules = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert modules <= {"__future__", "hashlib", "json", "math", "os", "pathlib", "platform", "re", "shutil", "signal", "stat", "subprocess", "time"}
    assert not {"numpy", "torch", "body_smoke", "joblib"} & modules
    assert not any(isinstance(node, ast.Attribute) and node.attr in {"write_text", "write_bytes", "mkdir", "chmod", "unlink"} for node in ast.walk(tree))


def test_wrapper_isolated_stdlib_readonly_no_gpu():
    root = Path(__file__).resolve().parents[1]
    wrapper = root / "infra/run_track1_readiness.sh"
    result = subprocess.run(["bash", "-n", str(wrapper)], capture_output=True)
    assert result.returncode == 0
    text = wrapper.read_text()
    assert "python3 -I -B" in text and "123s" in text and "runpy.run_path" in text
    assert "docker" not in text and "--gpus" not in text and "PYTHONPATH" not in text


def test_actual_runtime_bundle_finds_entrypoint_and_no_stage_closure():
    root = Path(__file__).resolve().parents[1]
    path = root / "infra/azure_job.py"
    spec = importlib.util.spec_from_file_location("test_actual_readiness_azure_job", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    files = {str(path.relative_to(root)): path.read_bytes() for directory in ("infra", "src", "configs")
        for path in (root / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    files["pyproject.toml"] = (root / "pyproject.toml").read_bytes()
    selected = module.runtime_bundle_paths(files, "infra/run_track1_readiness.sh")
    infra = {path for path in selected if path.startswith("infra/")}
    assert infra == {"infra/run_track1_readiness.sh", "infra/track1_readiness.py"}
    assert all(path.startswith("configs/") or path in {*infra, "pyproject.toml", "src/world_reward/__init__.py"} for path in selected)
