"""Tiny opaque/native-shaped artifacts; no external pixels, labels or models."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "infra"), str(REPO / "src")]
SPEC = importlib.util.spec_from_file_location("dex_identity_calibrate_test", REPO / "infra/dexycb_identity_calibrate.py")
gate = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(gate)
from world_reward.automatic_candidate_bank import build_automatic_candidate_bank


def seal(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw); path.chmod(0o444)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write_json(path, value):
    return seal(path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def detector(rgb, query):
    boxes = [[10, 10, 110, 110]] if query == "hand." else [[120, 140, 220, 240], [330, 140, 430, 240]]
    return np.asarray(boxes, np.float32), np.full(len(boxes), .8, np.float32)


class Predictor:
    def set_image(self, rgb): pass
    def predict(self, *, box, multimask_output):
        masks = np.zeros((len(box), 1, 480, 640), bool)
        for i, b in enumerate(box.astype(int)): masks[i, 0, b[1]:b[3], b[0]:b[2]] = True
        return masks, np.ones((len(box), 1), np.float32), None


def bank(frames=3):
    built = build_automatic_candidate_bank(np.zeros((480, 640, 3), np.uint8), detector, detector, Predictor())
    arrays, candidates = gate.infer.bank_arrays(built, frames)
    return arrays, dict(candidates=candidates, background_query_count=len(built.background_query_points))


def native(data):
    queries, t = data["query_points"], len(data["video"])
    raw = np.broadcast_to(queries[:, [2, 1]][:, None], (len(queries), t, 2)).copy()
    scaled = raw.astype(np.float32) * np.asarray([256/640, 256/480], np.float32)
    return dict(tracks=(scaled * np.array([640, 480], np.float32)) / np.float32(256), tracks_256=scaled,
        occlusion=np.full((len(queries), t), -2, np.float32), expected_dist=np.full((len(queries), t), -2, np.float32),
        visible=np.ones((len(queries), t), bool), query_points=queries.copy(),
        point_indices=data["point_indices"].copy(), frame_index=np.arange(t, dtype=np.int64),
        static_tracks=raw, static_visible=np.ones((len(queries), t), bool))


def feature_inputs():
    arrays, row = bank()
    tracks = gate.infer.track_clip(np.zeros((3, 480, 640, 3), np.uint8), arrays, row, native)
    return arrays, row, tracks


def test_mean_uses_all_supported_hands_intervals_not_best_hand():
    arrays, row, tracks = feature_inputs()
    tracks["motion_pair_supported"] = np.array([[[True, True], [True, False]], [[True, True], [False, True]]])
    values = np.arange(24, dtype=np.float64).reshape(2, 2, 2, 3)
    values[~tracks["motion_pair_supported"]] = np.nan
    tracks["motion_features"] = values
    result = gate.summarize_clip(arrays, row, tracks)
    assert result["support_count"].tolist() == [3, 3]
    for j in range(2): np.testing.assert_array_equal(result["features"][j], values[:, :, j][tracks["motion_pair_supported"][:, :, j]].mean(0))
    tracks["motion_pair_supported"][:, :, 1] = False; tracks["motion_features"][:, :, 1] = np.nan
    result = gate.summarize_clip(arrays, row, tracks)
    assert result["supported"].tolist() == [True, False] and np.isnan(result["features"][1]).all()


def test_baseline_original_ambiguity_gate_not_hand_or_gt_selection():
    arrays, row = bank()
    assert gate.baseline_selection(arrays, row)[0] is None
    arrays["raw_object_scores"] = np.array([.9, .4], np.float32)
    assert gate.baseline_selection(arrays, row) == (row["candidates"][1]["stable_id"], None)
    arrays["boxes"][1] += 1
    with pytest.raises(ValueError, match="exactly one"): gate.baseline_selection(arrays, row)


def test_complete_tracking_recompute_native_source_math_and_no_static():
    arrays, row, tracks = feature_inputs()
    gate.validate_tracks(tracks, arrays, row, 3)
    tracks["motion_features"] = tracks["motion_features"].copy()
    tracks["motion_features"][0, 0, 0, 0] += .01
    with pytest.raises(ValueError, match="movement evidence"): gate.validate_tracks(tracks, arrays, row, 3)


@pytest.mark.parametrize("fault", ["frames", "dtype", "queries", "static", "visible", "conversion"])
def test_raw_tracks_and_full_query_bank_cannot_change(fault):
    arrays, row, tracks = feature_inputs()
    if fault == "frames": tracks["frame_index"][-1] = 0
    if fault == "dtype": tracks["tracks"] = tracks["tracks"].astype(np.float64)
    if fault == "queries": tracks["query_points"][0, 2] += 1
    if fault == "static": tracks["static_tracks"] = np.zeros(1)
    if fault == "visible": tracks["visible"][0, 0] = False
    if fault == "conversion": tracks["tracks"][0, 0, 0] += 1
    with pytest.raises(ValueError): gate.validate_tracks(tracks, arrays, row, 3)


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "root"; revision, observed = "a" * 40, "b" * 40
    code = root / "jobs" / revision / gate.ENTRY / "code"
    old = root / "jobs" / observed / gate.infer.ENTRY / "code"
    helpers = (*gate.infer.HELPERS, "infra/dexycb_identity_calibrate.py", "src/world_reward/identity_calibration.py")
    for folder, names in ((code, helpers), (old, gate.infer.HELPERS)):
        for name in names: seal(folder / name, (REPO / name).read_bytes())
        seal(folder.parent / "revision", (folder.parent.parent.name + "\n").encode())
        seal(folder.parent / "source-sha256", b"c" * 64 + b"\n")
        for p in reversed(sorted(folder.rglob("*"))):
            if p.is_dir(): p.chmod(0o555)
        folder.chmod(0o555)
    monkeypatch.setattr(gate, "ROOT", root); monkeypatch.setattr(gate.infer, "ROOT", root)
    monkeypatch.setattr(gate, "__file__", str(code / "infra/dexycb_identity_calibrate.py"))
    monkeypatch.setattr(gate.infer, "__file__", str(code / "infra/dexycb_identity_infer.py"))
    monkeypatch.setattr(gate.policy, "__file__", str(code / "src/world_reward/identity_calibration.py"))
    for module, name in ((gate.binding, "bridge_frontend_bindings"), (gate.binding.selected, "frontend_selected_assets"),
            (gate.infer.detector_policy, "hand_synthetic_masks"), (gate.infer.boots, "robotap_boots_infer"),
            (gate.infer.boots.source, "robotap_boots_acquire")):
        monkeypatch.setattr(module, "__file__", str(code / f"infra/{name}.py"))
    for name in ("relational_motion", "prompt_selection"):
        monkeypatch.setattr(sys.modules["world_reward." + name], "__file__", str(code / f"src/world_reward/{name}.py"))
    monkeypatch.setattr(gate.infer, "acquisition_proof", lambda *a: a[0])
    model = {"manufactured_model_proof": "no models executed in this fixture"}
    monkeypatch.setattr(gate.infer, "frontend_proof", lambda *a: ({}, model))
    monkeypatch.setattr(gate.infer, "boots_proof", lambda *a: model)
    inputs = root / gate.BASE / "inputs"; inputs.mkdir(parents=True)
    b = io.BytesIO(); Image.new("RGB", (640, 480), (40, 50, 60)).save(b, format="JPEG")
    clips, images = [], []
    for i in range(12):
        idx = gate.infer.INDICES[i % 6]
        clip = dict(subject=gate.infer.SUBJECTS[i // 6], sequence=f"20200813_{i:06d}",
            sequence_lex_index=idx, camera=gate.infer.CAMERA, frames=3,
            partition="blind_evaluation" if i < 6 else "fit" if idx in (0, 32, 64) else "decision")
        clips.append(clip)
        for t in range(3):
            name = f"subject_{i//6+1:02d}_sequence_{idx:03d}_frame_{t:06d}.jpg"
            pin = seal(inputs / name, b.getvalue())
            images.append(dict(file=name, **pin, width=640, height=480, subject=clip["subject"],
                sequence=clip["sequence"], camera=clip["camera"], frame_position=t, source_frame_id=t))
    manifest = dict(schema="world-reward-dexycb-identity-rgb-v1", license="CC-BY-NC-4.0", sequences=clips, images=images,
        timestamps_available=False, training_overlap_verified=False, challenge_overlap_verified=False,
        frame_count_evidence="all_RGB_and_label_member_headers_no_meta_values",
        source_archives={s: dict(bytes=100, sha256="d" * 64) for s in gate.infer.SUBJECTS})
    pins = dict(manifest=write_json(inputs / "manifest.json", manifest))
    pins["acquisition"] = write_json(inputs.parent / "report.json", {"retained_files": {}})
    source = gate.closure(old, observed, gate.infer.ENTRY, gate.infer.HELPERS)
    bank_arrays, row, track = feature_inputs(); frames = 3
    for stage, arrays in (("masks", bank_arrays), ("tracks", track)):
        folder = inputs.parent / f"identity_{stage}_{observed}"; folder.mkdir(); seal(folder / ".container.cid", b"e" * 64)
        rows = []
        for i in range(12):
            name = f"clip_{i:03d}.npz"; pin = gate.infer.save_arrays(folder / name, arrays)
            row_data = copy.deepcopy(row) if stage == "masks" else dict(query_count=len(bank_arrays["query_points"]),
                pair_supported=int(track["motion_pair_supported"].sum()), camera_supported=int(track["motion_camera_supported"].sum()))
            rows.append(dict(clip_index=i, frames=frames, file=name, **pin, **row_data))
        receipt = dict(schema="world-reward-dexycb-identity-infer-v1", stage=stage, status="pass", phase="complete",
            producer_revision=observed, script_sha256=source["helpers"][gate.infer.HELPERS[0]]["sha256"],
            source_binding=source, public_manifest=pins["manifest"], image_id=gate.infer.IMAGES[stage], network="none", device="cuda",
            budget_seconds=gate.infer.BUDGETS[stage], original_rehashed_after=True, private_annotations_read=False,
            challenge_inputs_used=False, oracle_modes=[], quality_verified=False, identity_accepted=False,
            clips=rows, model_assets=model)
        if stage == "masks": receipt.update(detector_attempts=24, detector_calls=24, sam2_encoder_attempts=12,
            sam2_encoder_calls=12, sam2_batch_attempts=12, sam2_batch_calls=12)
        else: receipt.update(native_calls_attempted=12, native_calls_returned=12, native_calls_completed=12,
            masks_report={k: pins["masks"][k] for k in ("bytes", "sha256")})
        pin = write_json(folder / "report.json", receipt)
        pins[stage] = dict(**pin, producer_revision=observed, script_sha256=receipt["script_sha256"])
    return root, code, revision, pins


def test_real_twelve_typed_public_artifacts_and_source_before_any_private(tmp_path, monkeypatch):
    root, code, revision, pins = fixture(tmp_path, monkeypatch)
    def forbidden(*a): raise AssertionError("No private decoding in public features")
    monkeypatch.setattr(gate, "private_anchor", forbidden)
    gate.validate_pins(pins)
    report = gate.run("public_features", code, revision, pins)
    assert report["status"] == "pass" and report["all_twelve_features_frozen"]
    assert report["original_rehashed_after"] and report["private_annotations_read"] is False
    folder = root / gate.BASE / gate.FOLDERS["public_features"]
    assert len(list(folder.glob("clip_*.npz"))) == 12 and (folder / "report.json").stat().st_mode & 0o777 == 0o444
    pin = dict(**gate.binding.identity(folder / "report.json"), producer_revision=revision,
        script_sha256=report["script_sha256"])
    privatepins = dict(pins, features=pin)
    clips, arrays, _, _ = gate.public_evidence(code, privatepins)
    assert len(gate.frozen_features(privatepins, clips, arrays)) == 13


@pytest.mark.parametrize("fault", ["incomplete_calls", "wrong_order", "source", "model", "bad_native", "changed_feature"])
def test_public_tampering_fails_before_private_read(tmp_path, monkeypatch, fault):
    root, code, revision, pins = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(gate, "private_anchor", lambda *a: (_ for _ in ()).throw(AssertionError("No labels")))
    stage = "tracks"; folder = root / gate.BASE / f"identity_tracks_{pins[stage]['producer_revision']}"
    path = folder / "report.json"; report = json.loads(path.read_bytes())
    if fault == "incomplete_calls": report["native_calls_completed"] = 11
    if fault == "wrong_order": report["clips"][0]["clip_index"] = 1
    if fault == "source": report["source_binding"]["closure_sha256"] = "0" * 64
    if fault == "model": report["model_assets"] = {}
    if fault in ("bad_native", "changed_feature"):
        row = report["clips"][0]; p = folder / row["file"]; data = gate.load_arrays(p)
        data["tracks" if fault == "bad_native" else "motion_features"].flat[0] += .1
        p.unlink(); row.update(gate.infer.save_arrays(p, data))
    path.chmod(0o600); pin = write_json(path, report); pins[stage].update(pin)
    with pytest.raises(ValueError): gate.run("public_features", code, revision, pins)


def labeled_arrays(tmp_path, monkeypatch, good=True):
    root = tmp_path / "root"; monkeypatch.setattr(gate, "ROOT", root)
    bank_arrays, row = bank(); clips, triples, retained = [], [], {}
    seg = np.zeros((480, 640), np.uint8)
    seg[bank_arrays["initial_masks"][1]] = 4; seg[bank_arrays["initial_masks"][2]] = 7
    for i in range(12):
        idx = gate.infer.INDICES[i % 6]
        clip = dict(clip_index=i, subject=gate.infer.SUBJECTS[i//6], sequence=f"20200813_{i:06d}",
            sequence_lex_index=idx, camera=gate.infer.CAMERA, frames=3)
        clips.append(clip)
        prefix = f"{clip['subject']}/{clip['sequence']}"
        private = root / gate.BASE / "eval_private"
        retained[prefix + "/meta.yml"] = seal(private / prefix / "meta.yml", b"ycb_ids: [4, 7]\nycb_grasp_ind: 0\npose_never_interpreted: secret\n")
        buffer = io.BytesIO(); np.savez(buffer, seg=seg, pose_y=np.array([object()], dtype=object))
        retained[prefix + f"/{gate.infer.CAMERA}/labels_000000.npz"] = seal(private / prefix / gate.infer.CAMERA / "labels_000000.npz", buffer.getvalue())
        data = dict(frame_index=np.arange(3, dtype=np.int64), candidate_index=np.array([1, 2], np.int64),
            features=np.array([[2., .1, .4], [-2., .4, .1]]) if good or i in (6, 8, 10) else np.zeros((2, 3)),
            supported=np.ones(2, bool), support_count=np.ones(2, np.int64))
        triples.append((bank_arrays, row, data))
    acquisition = {"retained_files": retained}
    pin = write_json(root / gate.BASE / "report.json", acquisition)
    return root, clips, triples, acquisition, dict(acquisition=pin)


def test_private_only_seg_payload_and_meta_target_fields_no_3d_or_mano_decode(tmp_path, monkeypatch):
    _, clips, _, acquisition, _ = labeled_arrays(tmp_path, monkeypatch)
    original = np.lib.npyio.NpzFile.__getitem__
    def segmentation_only(archive, key):
        assert key == "seg"  # pose_y object payload would fail if accidentally decoded.
        return original(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", segmentation_only)
    seg, target = gate.private_anchor(clips[6], acquisition, {})
    assert target == 4 and seg.dtype == np.uint8


def test_unknown_yaml_values_are_never_constructed(tmp_path, monkeypatch):
    import yaml
    root, clips, _, acquisition, _ = labeled_arrays(tmp_path, monkeypatch)
    clip = clips[6]; name = f"{clip['subject']}/{clip['sequence']}/meta.yml"
    path = root / gate.BASE / "eval_private" / name; path.chmod(0o600)
    acquisition["retained_files"][name] = seal(path, b"ycb_ids: [4, 7]\nycb_grasp_ind: 0\nprivate_ignored: [1.234, {camera_unknown: 9.87}]\n")
    def forbidden_float(*args): raise AssertionError("Unknown private values must not be constructed")
    monkeypatch.setitem(yaml.SafeLoader.yaml_constructors, "tag:yaml.org,2002:float", forbidden_float)
    assert gate.private_anchor(clip, acquisition, {})[1] == 4


def test_unsupported_ignored_yaml_tag_is_rejected_without_constructor(tmp_path, monkeypatch):
    import yaml
    root, clips, _, acquisition, _ = labeled_arrays(tmp_path, monkeypatch)
    clip = clips[6]; name = f"{clip['subject']}/{clip['sequence']}/meta.yml"
    path = root / gate.BASE / "eval_private" / name; path.chmod(0o600)
    acquisition["retained_files"][name] = seal(path, b"ycb_ids: [4, 7]\nycb_grasp_ind: 0\nignored: !unsupported anything\n")
    def forbidden(*args, **kwargs): raise AssertionError("No value constructor before unsupported-tag rejection")
    monkeypatch.setattr(yaml.SafeLoader, "construct_object", forbidden)
    with pytest.raises(ValueError, match="safe native YAML tags"):
        gate.private_anchor(clip, acquisition, {})


def test_failed_decision_gate_never_opens_evaluation_subject_labels(tmp_path, monkeypatch):
    root, clips, triples, _, pins = labeled_arrays(tmp_path, monkeypatch, good=False)
    out = root / gate.BASE / "testcalib"; out.mkdir(); report, frozen, visited = {}, {}, []
    original = gate.private_anchor
    def ordered(clip, acquisition, checked):
        assert clip["clip_index"] >= 6; visited.append(clip["clip_index"])
        if clip["clip_index"] in (7, 9, 11):
            assert (out / "model.npz").stat().st_mode & 0o777 == 0o444
            assert (out / "predictions.json").stat().st_mode & 0o777 == 0o444
        return original(clip, acquisition, checked)
    monkeypatch.setattr(gate, "private_anchor", ordered)
    gate.private_calibration(out, clips, triples, report, frozen, pins)
    assert visited == [6, 8, 10, 7, 9, 11]
    assert report["decision_gate"] is False and report["evaluation_subject_read"] is False
    assert report["hypothesis_supported"] is False


def test_fitted_model_and_all_public_scores_frozen_before_decision_and_heldout(tmp_path, monkeypatch):
    root, clips, triples, _, pins = labeled_arrays(tmp_path, monkeypatch)
    out = root / gate.BASE / "testcalib"; out.mkdir(); report, frozen, visited = {}, {}, []
    original = gate.private_anchor
    def ordered(clip, acquisition, checked):
        visited.append(clip["clip_index"])
        if clip["clip_index"] not in (6, 8, 10):
            assert set(p.name for p in out.iterdir()) == {"model.npz", "predictions.json"}
            assert len(json.loads((out / "predictions.json").read_bytes())) == 12
        return original(clip, acquisition, checked)
    monkeypatch.setattr(gate, "private_anchor", ordered)
    gate.private_calibration(out, clips, triples, report, frozen, pins)
    assert visited == [6, 8, 10, 7, 9, 11, 0, 1, 2, 3, 4, 5]
    assert report["decision_gate"] and report["evaluation_subject_read"] and report["evaluation_success_gain"] == 6
    assert report["hypothesis_supported"] and report["predictions_frozen_before_decision"]
    pred = json.loads((out / "predictions.json").read_bytes())
    assert all(row["baseline_proposal"] is None for row in pred)
    assert all(row["learned_proposal"] == "object:000001" for row in pred)


@pytest.mark.parametrize("fault", ["object_yaml", "duplicate_yaml", "wrong_seg", "changed_pin"])
def test_private_decoder_fail_closed_without_unapproved_annotation_parsing(tmp_path, monkeypatch, fault):
    root, clips, _, acquisition, _ = labeled_arrays(tmp_path, monkeypatch)
    clip = clips[6]; prefix = f"{clip['subject']}/{clip['sequence']}"
    if fault in ("object_yaml", "duplicate_yaml"):
        path = root / gate.BASE / "eval_private" / prefix / "meta.yml"; path.chmod(0o600)
        raw = b"!!python/object:builtins.dict {}" if fault == "object_yaml" else b"ycb_ids: [4]\nycb_grasp_ind: 0\nycb_grasp_ind: 0\n"
        acquisition["retained_files"][prefix + "/meta.yml"] = seal(path, raw)
    elif fault == "wrong_seg":
        path = root / gate.BASE / "eval_private" / prefix / gate.infer.CAMERA / "labels_000000.npz"; path.chmod(0o600)
        b = io.BytesIO(); np.savez(b, seg=np.zeros((480, 640), np.int64))
        acquisition["retained_files"][prefix + f"/{gate.infer.CAMERA}/labels_000000.npz"] = seal(path, b.getvalue())
    else: acquisition["retained_files"][prefix + "/meta.yml"]["sha256"] = "0" * 64
    with pytest.raises(Exception): gate.private_anchor(clip, acquisition, {})


def test_inclusive_postverify_and_unknown_output_no_overwrite(tmp_path, monkeypatch):
    root, code, revision, pins = fixture(tmp_path, monkeypatch)
    out = root / gate.BASE / gate.FOLDERS["public_features"]; out.mkdir()
    seal(out / "user.txt", b"preserve")
    with pytest.raises(FileExistsError): gate.run("public_features", code, revision, pins)
    assert (out / "user.txt").read_bytes() == b"preserve"


def test_mutated_public_feature_output_seals_fail_not_pass(tmp_path, monkeypatch):
    root, code, revision, pins = fixture(tmp_path, monkeypatch)
    original = gate.public_features
    def mutate(out, clips, arrays, report):
        original(out, clips, arrays, report)
        path = out / "clip_000.npz"; path.chmod(0o600); path.write_bytes(b"changed"); path.chmod(0o444)
    monkeypatch.setattr(gate, "public_features", mutate)
    with pytest.raises(ValueError): gate.run("public_features", code, revision, pins)
    report = json.loads((root / gate.BASE / gate.FOLDERS["public_features"] / "report.json").read_bytes())
    assert report["status"] == "fail" and report["original_rehashed_after"] is False


def test_private_run_replays_all_public_features_before_first_private_access(tmp_path, monkeypatch):
    root, code, revision, pins = fixture(tmp_path, monkeypatch)
    report = gate.run("public_features", code, revision, pins)
    folder = root / gate.BASE / gate.FOLDERS["public_features"]
    pins["features"] = dict(**gate.binding.identity(folder / "report.json"), producer_revision=revision,
        script_sha256=report["script_sha256"])
    visited = []
    def private(out, clips, arrays, result, frozen, checked):
        assert len([p for p in frozen if p.parent == folder]) == 13
        assert len(clips) == len(arrays) == 12
        visited.append(True)
        # No learned run/quality is claimed in this orchestration-only fixture.
        for name in ("model.npz", "predictions.json"): frozen[out / name] = seal(out / name, b"fixture only")
    monkeypatch.setattr(gate, "private_calibration", private)
    result = gate.run("private_calibration", code, revision, pins)
    assert visited == [True] and result["original_rehashed_after"]


def test_invalid_frozen_features_prevents_first_private_read(tmp_path, monkeypatch):
    root, code, revision, pins = fixture(tmp_path, monkeypatch)
    report = gate.run("public_features", code, revision, pins)
    folder = root / gate.BASE / gate.FOLDERS["public_features"]
    pins["features"] = dict(**gate.binding.identity(folder / "report.json"), producer_revision=revision,
        script_sha256=report["script_sha256"])
    path = folder / "clip_000.npz"; path.chmod(0o600); path.write_bytes(b"changed"); path.chmod(0o444)
    monkeypatch.setattr(gate, "private_anchor", lambda *a: (_ for _ in ()).throw(AssertionError("No private read")))
    with pytest.raises(ValueError): gate.run("private_calibration", code, revision, pins)
    assert not (root / gate.BASE / gate.FOLDERS["private_calibration"]).exists()


def test_no_extra_pins_or_protocol_fallback_before_io():
    with pytest.raises(ValueError): gate.validate_pins({"untrusted": {}})
    with pytest.raises(ValueError): gate.run("unknown", None, None, {})


def test_fixed_protocol_has_no_best_hand_tuning_or_full_timeline_quality_claim():
    assert gate.BUDGET == 120 and gate.RULE["minimum_support_count"] == 1
    assert gate.RULE["minimum_raw_gap"] == gate.RULE["minimum_raw_score"] == 0
    assert gate.RULE["decision_required_successes"] == 3 and gate.RULE["evaluation_minimum_success_gain"] == 1
    args = ["--stage", "public_features"]
    for name in ("manifest", "acquisition", "masks", "tracks"):
        args += ["--" + name + "-sha256", "a" * 64, "--" + name + "-bytes", "100"]
        if name in ("masks", "tracks"): args += ["--" + name + "-producer-revision", "b" * 40, "--" + name + "-script-sha256", "c" * 64]
    assert gate.parser().parse_args(args).stage == "public_features"
    with pytest.raises(SystemExit): gate.parser().parse_args(args + ["--minimum-gap", "1"])
