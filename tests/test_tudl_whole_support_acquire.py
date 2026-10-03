"""Tiny new filename/stub-ZIP tests; no network, real media, models or GPUs."""
import hashlib
import importlib
import io
import json
from pathlib import Path
import struct
import zipfile
import zlib

import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra)); return importlib.import_module("tudl_whole_support_acquire")


def filenames(gate):
    result = {}
    for scene in (1, 2, 3):
        anchors = {0: gate.DEVELOPMENT_IDS[scene][0], 40: gate.PRIOR_IDS[scene][0], 80: gate.PRIOR_IDS[scene][1],
            100: gate.DEVELOPMENT_IDS[scene][1], 120: gate.PRIOR_IDS[scene][2], 160: gate.PRIOR_IDS[scene][3], 199: gate.DEVELOPMENT_IDS[scene][2]}
        ids = [None] * 200
        for index, value in anchors.items(): ids[index] = value
        points = sorted(anchors)
        for low, high in zip(points[:-1], points[1:]):
            for rank in range(low + 1, high):
                ids[rank] = anchors[low] + (anchors[high] - anchors[low]) * (rank - low) // (high - low)
        assert len(set(ids)) == 200
        for frame in reversed(ids): result[f"test/{scene:06d}/rgb/{frame:06d}.png"] = object()
    return result


def archives(gate):
    names = filenames(gate); selected = gate.select_rgb_names(names)
    header = struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + header + struct.pack(">I", zlib.crc32(b"IHDR" + header))
    base = {"tudl/camera.json": b"tiny private camera", "tudl/dataset_info.md": b"tiny CC-BY-SA4 source", "tudl/test_targets_bop19.json": b"[]"}
    models = {f"{folder}/{name}": b"tiny private mesh evidence" for folder in ("models", "models_eval")
        for name in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply")}
    frames = {name: png for name in names}
    for scene in (1, 2, 3):
        ids = [frame for s, frame, _ in selected if s == scene]; prefix = f"test/{scene:06d}/"
        for kind in ("camera", "gt", "gt_info"):
            values = {str(frame): ({"cam_K": ["private"], "depth_scale": 1.} if kind == "camera" else [{"obj_id": scene}, {"obj_id": scene}]) for frame in [*ids, 99999]}
            frames[prefix + f"scene_{kind}.json"] = json.dumps(values).encode()
        for frame in ids:
            frames[prefix + f"depth/{frame:06d}.png"] = b"tiny private sensor"
            for instance in range(2): frames[prefix + f"mask_visib/{frame:06d}_{instance:06d}.png"] = b"tiny private mask"
    result = {}
    for name, values in (("tudl_base.zip", base), ("tudl_models.zip", models), ("tudl_test_bop19.zip", frames)):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for member, value in values.items(): archive.writestr(member, value)
        archive = zipfile.ZipFile(io.BytesIO(stream.getvalue()))
        result[name] = (archive, gate.source.zip_inventory(archive, [0, 0]))
    return result, selected


def close(values):
    for archive, _ in values.values(): archive.close()


def destination(tmp_path):
    private, inputs = tmp_path / "eval_private", tmp_path / "inputs"
    private.mkdir(mode=0o700); inputs.mkdir(mode=0o755)
    (private / "source").mkdir(mode=0o700)
    (private / "acquisition-report.json").write_bytes(b"tiny report")
    for name in ("huggingface-README.md", "BOP-TUD-L-section.html", "attribution.json"):
        path = private / "source/licenses" / name; path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(b"tiny licence evidence"); path.chmod(0o400)
    return private, inputs


def test_new_filename_ranks_verify_all21_previous_pairs_without_opening_values(gate):
    names = filenames(gate); selected = gate.select_rgb_names(names)
    assert len(selected) == 12 and gate.select_rgb_names(dict(reversed(list(names.items())))) == selected
    for scene in (1, 2, 3):
        native = sorted(name for name in names if name.startswith(f"test/{scene:06d}/rgb/"))
        assert [row[2] for row in selected if row[0] == scene] == [native[index] for index in (30, 70, 110, 150)]
        assert not {row[1] for row in selected if row[0] == scene} & {*gate.DEVELOPMENT_IDS[scene], *gate.PRIOR_IDS[scene]}


@pytest.mark.parametrize("fault", ["missing", "extra", "prior_id", "development_id", "nonstring", "notdict", "ranks"])
def test_selection_fail_fast_on_changed_inventory_or_exclusions(gate, monkeypatch, fault):
    names = filenames(gate)
    if fault == "missing": names.pop(next(iter(names)))
    elif fault == "extra": names["test/000001/rgb/999999.png"] = None
    elif fault in ("prior_id", "development_id"):
        scene, old = 1, gate.PRIOR_IDS[1][0] if fault == "prior_id" else gate.DEVELOPMENT_IDS[1][0]
        key = f"test/{scene:06d}/rgb/{old:06d}.png"; names[key.replace(f"{old:06d}.png", f"{old + 1:06d}.png")] = names.pop(key)
    elif fault == "nonstring": names[True] = object()
    elif fault == "notdict": names = list(names)
    else: monkeypatch.setattr(gate, "INDICES", (40, 70, 110, 150))
    with pytest.raises(ValueError): gate.select_rgb_names(names)


def test_entire_protocol_native_helpers_pinned_and_not_global_name_rewrite(gate):
    root = Path(__file__).resolve().parents[1]; raw = (root / gate.PROTOCOL).read_bytes(); protocol = json.loads(raw)
    assert len(raw) == 5217 and hashlib.sha256(raw).hexdigest() == gate.PROTOCOL_SHA
    assert gate.identity is gate.prior.identity and gate.prior.source is gate.source
    assert gate.prior.INDICES == (40, 80, 120, 160) and gate.INDICES == (30, 70, 110, 150)
    for name, digest in gate.FROZEN.items(): assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert protocol["frozen_method"]["minimum_native_valid_pairs"] == 1024
    assert protocol["frozen_method"]["minimum_native_valid_full_grid_coverage"] == .25
    assert protocol["future_private_evaluation"]["samples_per_frame"] == 8192
    assert protocol["future_private_evaluation"]["minimum_median_relative_gain"] == .05
    assert protocol["future_private_evaluation"]["maximum_any_scene_relative_regression"] == .05


def test_selection_precedes_any_private_reads_and_all_instances_retained(gate, tmp_path, monkeypatch):
    values, selected = archives(gate); calls = []; native = gate.select_rgb_names
    def select(names): result = native(names); calls.append(result); return result
    monkeypatch.setattr(gate, "select_rgb_names", select)
    for archive, _ in values.values():
        read = archive.read
        def checked(name, original=read): assert calls and len(calls[-1]) == 12; return original(name)
        monkeypatch.setattr(archive, "read", checked)
    private, inputs = destination(tmp_path)
    try: manifest = gate.retain_subset(values, private, inputs, selected)
    finally: close(values)
    assert len(calls) == 1 and len(manifest["images"]) == 12 and set(manifest) == {"schema", "revision", "license", "selection", "images"}
    gate.source.save_bytes(inputs / "manifest.json", (json.dumps(manifest) + "\n").encode(), 0o444)
    public, retained = gate.output_inventory(tmp_path, manifest, selected)
    assert len(public) == 13 and len(retained) == 60
    assert all(row["bytes"] > 0 for row in retained)
    for scene in (1, 2, 3):
        gt = json.loads((private / f"source/test/{scene:06d}/scene_gt.json").read_bytes())
        assert len(gt) == 4 and all(len(instances) == 2 for instances in gt.values()) and "99999" not in gt


@pytest.mark.parametrize("fault", ["reverse", "bool", "old_record", "truncated"])
def test_invalid_selected_records_fail_before_any_private_read(gate, tmp_path, fault):
    values, selected = archives(gate)
    if fault == "reverse": selected.reverse()
    elif fault == "bool": selected[0] = (True, selected[0][1], selected[0][2])
    elif fault == "old_record": selected[0] = gate.prior.select_rgb_names(values["tudl_test_bop19.zip"][1])[0]
    else: selected.pop()
    for archive, _ in values.values(): archive.read = lambda *_: pytest.fail("No private reads before exact filename selection")
    try:
        with pytest.raises(ValueError): gate.retain_subset(values, tmp_path / "private", tmp_path / "inputs", selected)
    finally: close(values)


@pytest.mark.parametrize("fault", ["mask", "camera", "license"])
def test_missing_private_evidence_never_silently_dropped(gate, tmp_path, fault):
    values, selected = archives(gate); _, frame, _ = selected[0]
    if fault == "mask": del values["tudl_test_bop19.zip"][1][f"test/000001/mask_visib/{frame:06d}_000001.png"]
    else:
        archive = values["tudl_test_bop19.zip" if fault == "camera" else "tudl_base.zip"][0]; read = archive.read
        def changed(name):
            raw = read(name)
            if fault == "camera" and name == "test/000001/scene_camera.json":
                content = json.loads(raw); del content[str(frame)]; return json.dumps(content).encode()
            return b"Non-commercial only" if fault == "license" and name.endswith(".md") else raw
        archive.read = changed
    try:
        with pytest.raises(ValueError): gate.retain_subset(values, tmp_path / "private", tmp_path / "inputs", selected)
    finally: close(values)


def test_wrapper_complete_stdlib_closure_fresh_output_and_terminal_source_check(gate):
    root = Path(__file__).resolve().parents[1]; launcher = importlib.import_module("azure_job")
    files = {str(path.relative_to(root)): path.read_bytes() for folder in ("infra", "src", "configs") for path in (root / folder).rglob("*") if path.is_file()}
    files["pyproject.toml"] = (root / "pyproject.toml").read_bytes()
    closure = set(launcher.runtime_bundle_paths(files, "infra/run_tudl_whole_support_acquire.sh"))
    assert set(gate.HELPERS) <= closure and gate.PROTOCOL in closure
    assert {name for name in closure if name.startswith("infra/")} == set(gate.HELPERS)
    wrapper = (root / "infra/run_tudl_whole_support_acquire.sh").read_text()
    assert "603s runuser" in wrapper and "--kill-after=10s" in wrapper and "trap finish EXIT" in wrapper
    assert 'WR_TUDL_WHOLE_SUPPORT_RESERVED=1' in wrapper and "no overwrite/resume/cleanup" in wrapper
    assert "docker run" not in wrapper and "--gpus" not in wrapper


def configure_mock_main(gate, tmp_path, monkeypatch):
    """Tiny archives and mock dispatch, never valid production provenance."""
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE", str(tmp_path / "mock-code"))
    monkeypatch.setenv("WR_CODE_REVISION", "a" * 40); monkeypatch.setenv("WR_TUDL_WHOLE_SUPPORT_RESERVED", "1")
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    helpers = {name: dict(sha256="b" * 64, bytes=1) for name in gate.HELPERS}
    monkeypatch.setattr(gate, "bound_source", lambda *_: (helpers, dict(sha256="c" * 64, bytes=1), {}))
    output = tmp_path / gate.NAMESPACE; output.mkdir(parents=True)
    values, _ = archives(gate); blobs = {}
    for name, (archive, _) in values.items():
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as target:
            for member in archive.namelist(): target.writestr(member, archive.read(member))
        blobs[name] = stream.getvalue()
    close(values)
    monkeypatch.setattr(gate.source, "ARCHIVES", {name: (len(raw), hashlib.sha256(raw).hexdigest()) for name, raw in blobs.items()})
    def download(url, path, size, digest):
        assert url == gate.source.BASE_URL + path.name and size == len(blobs[path.name])
        assert digest == hashlib.sha256(blobs[path.name]).hexdigest()
        path.write_bytes(blobs[path.name]); path.chmod(0o400)
    monkeypatch.setattr(gate.source, "download", download)
    def licenses(private):
        for name in ("huggingface-README.md", "BOP-TUD-L-section.html", "attribution.json"):
            gate.source.save_bytes(private / "source/licenses" / name, b"tiny mock license source")
        return []
    monkeypatch.setattr(gate.source, "license_evidence", licenses)
    return output


def test_mock_main_pass_records_complete_immutable_inventory_and_no_resume(gate, tmp_path, monkeypatch):
    output = configure_mock_main(gate, tmp_path, monkeypatch); gate.main([])
    path = output / "eval_private/acquisition-report.json"; receipt = json.loads(path.read_bytes())
    assert receipt["status"] == "pass" and receipt["images_completed"] == 12
    assert receipt["all_previously_observed_frame_ids_disjoint"] is True and receipt["selection_before_private_annotation_values"] is True
    assert len(receipt["public_files"]) == 13 and len(receipt["retained_files"]) == 60
    assert receipt["source_helpers"] == receipt["source_helpers_after"] and receipt["source_helpers_unchanged"] is True
    assert receipt["retained_outputs_unchanged"] is True and receipt["disposable_archives_removed"] is True
    assert path.stat().st_mode & 0o777 == 0o400
    assert all(receipt[key] is False for key in ("gpu_used", "inference_performed", "challenge_inputs_used", "accuracy_verified",
        "ground_truth_used_for_inference", "training_overlap_verified", "challenge_overlap_verified", "independent_scenes_or_objects"))
    frozen = path.read_bytes()
    with pytest.raises(ValueError, match="empty canonical"): gate.main([])
    assert path.read_bytes() == frozen


@pytest.mark.parametrize("failure", ["download", "support_selection", "postcheck"])
def test_mock_main_failure_sealed_with_cleanup_or_failed_postcheck(gate, tmp_path, monkeypatch, failure):
    output = configure_mock_main(gate, tmp_path, monkeypatch)
    if failure == "download":
        def failed(*_): raise TimeoutError("mock bounded download failure")
        monkeypatch.setattr(gate.source, "download", failed)
    elif failure == "support_selection":
        def failed(*_): raise ValueError("mock pinned filename inventory differs")
        monkeypatch.setattr(gate, "select_rgb_names", failed)
    else:
        original = gate.bound_source; calls = []
        def changed(*args):
            calls.append(1)
            if len(calls) == 2: raise ValueError("mock source changed after acquisition")
            return original(*args)
        monkeypatch.setattr(gate, "bound_source", changed)
    with pytest.raises((ValueError, TimeoutError)): gate.main([])
    path = output / "eval_private/acquisition-report.json"; receipt = json.loads(path.read_bytes())
    assert receipt["status"] == "fail" and receipt["disposable_archives_removed"] is True
    assert not (output / "eval_private/.downloads").exists() and path.stat().st_mode & 0o777 == 0o400
    assert receipt["inference_performed"] is False and receipt["accuracy_verified"] is False
    if failure == "postcheck": assert receipt["source_helpers_unchanged"] is False
