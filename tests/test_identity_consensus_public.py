"""Small real numerical/provenance contracts; no model, CUDA or media download."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec = importlib.util.spec_from_file_location("own_public_consensus", infra/"identity_consensus_public.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_frozen_protocol_indices_and_bounds(gate):
    assert gate.ANCHORS == tuple(np.rint(np.linspace(0, 789, 12)).astype(int))
    assert gate.MIDPOINTS == tuple((np.array(gate.ANCHORS[:-1])+gate.ANCHORS[1:])//2)
    assert len(gate.SELECTED) == 23 and not set(gate.ANCHORS)&set(gate.MIDPOINTS)
    assert gate.SELECTED == tuple(sorted(gate.SELECTED))
    assert gate.PARITY_BOUND == 1e-5 and gate.MIDPOINT_REGRESSION_BOUND == 1e-4
    assert gate.BUDGET == 180


def meshes(gate, monkeypatch):
    monkeypatch.setattr(gate, "VERTICES", 4)
    base = np.array([[0, 0, 0], [1, .5, .2], [.3, 1, .4], [.6, .1, 1]], np.float32)
    return np.stack([base+i for i in range(12)])


def test_medoid_uses_full_vertices_unaligned_and_first_exact_tie(gate, monkeypatch):
    value = meshes(gate, monkeypatch)
    position, distance, costs = gate.neutral_medoid(value)
    assert position == 5  # Two central raw anchors tie, not an averaged identity.
    assert distance[0, 11] == pytest.approx(np.sqrt(3)*11)
    assert np.array_equal(distance, distance.T) and np.array_equal(np.diag(distance), np.zeros(12))
    assert np.array_equal(costs, distance.sum(axis=1))
    changed = value.copy(); changed[5, 3, 2] += 100
    assert gate.neutral_medoid(changed)[1][0, 5] > distance[0, 5]
    assert np.array_equal(value, meshes(gate, monkeypatch))


@pytest.mark.parametrize("fault", ["nan", "masked", "dtype", "count", "vertices", "collapse"])
def test_medoid_fail_closed(gate, monkeypatch, fault):
    value = meshes(gate, monkeypatch)
    if fault == "nan": value[0, 0, 0] = np.nan
    elif fault == "masked": value = np.ma.array(value, mask=False)
    elif fault == "dtype": value = value.astype(np.float64)
    elif fault == "count": value = value[:11]
    elif fault == "vertices": value = value[:, :3]
    else: value[0] = 0
    with pytest.raises(ValueError): gate.neutral_medoid(value)


def blocks(gate):
    return {k: np.repeat(np.arange(len(gate.SELECTED), dtype=np.float32)[:, None], n, axis=1) for k, n in gate.BLOCKS.items()}


def test_candidate_preserves_pose_hands_shape_pca_bytes_and_nonmutation(gate, tmp_path):
    raw = blocks(gate); before = {k: v.copy() for k, v in raw.items()}
    raw["expr_params"][:] = 0; before["expr_params"][:] = 0
    raw["scale_params"][gate.SELECTED.index(gate.ANCHORS[3]), 0] = -0.
    before["scale_params"] = raw["scale_params"].copy()
    candidate, (path, digest) = gate.freeze_candidate(tmp_path, raw, 3, np.zeros((12, 12)), np.zeros(12))
    assert not path.stat().st_mode & 0o222 and gate.sha256(path) == digest
    for k in raw:
        assert raw[k].tobytes() == before[k].tobytes() and not np.shares_memory(raw[k], candidate[k])
        if k in ("shape_params", "scale_params"):
            row = raw[k][gate.SELECTED.index(gate.ANCHORS[3])].tobytes()
            assert all(v.tobytes() == row for v in candidate[k])
        else: assert candidate[k].tobytes() == raw[k].tobytes()
    with pytest.raises(FileExistsError): gate.freeze_candidate(tmp_path, raw, 3, np.zeros((12, 12)), np.zeros(12))


@pytest.mark.parametrize("index", [-1, 12, True, 3., np.int64(3)])
def test_candidate_requires_explicit_anchor_position(gate, index):
    with pytest.raises(ValueError): gate.candidate_blocks(blocks(gate), index)


def test_actual_forward_argument_contract_neutral_and_posed(gate):
    raw = blocks(gate); raw["expr_params"][:] = 0
    torch = SimpleNamespace(float32=np.float32, tensor=lambda v, **kw: np.asarray(v, np.float32).copy(),
        zeros_like=np.zeros_like, zeros=lambda shape, **kw: np.zeros(shape, np.float32))
    calls = []
    head = SimpleNamespace(mhr_forward=lambda **kw: calls.append(kw) or kw)
    posed = gate.forward_blocks(head, raw, torch)
    neutral = gate.forward_blocks(head, raw, torch, neutral=True)
    assert len(calls) == 2 and not posed["global_trans"].any()
    assert posed["return_model_params"] is True and neutral["return_model_params"] is False
    for k in ("global_rot", "body_pose_params", "hand_pose_params", "expr_params"):
        assert not neutral[k].any() and np.array_equal(posed[k], raw[k])
    for k in ("shape_params", "scale_params"):
        assert neutral[k].tobytes() == posed[k].tobytes() == raw[k].tobytes()
    assert all(c["return_keypoints"] is c["return_joint_coords"] is c["return_joint_rotations"] is False for c in calls)


@pytest.fixture
def masks(gate, monkeypatch):
    monkeypatch.setattr(gate, "WIDTH", 8); monkeypatch.setattr(gate, "HEIGHT", 8)
    human = np.zeros((8, 8), bool); human[:6] = True
    obj = np.zeros_like(human); obj[:1] = True
    return human, obj


def test_same_object_exclusion_full_region_falsepositives_not_ignored(gate, masks):
    human, obj = masks; raw = human.copy(); candidate = human.copy()
    candidate[0] = False  # Occluded changes must affect neither mode.
    evidence = gate.mask_evidence(raw, candidate, human, obj, gate.MIDPOINTS[0])
    assert evidence["raw_iou"] == evidence["candidate_iou"] == 1.
    assert evidence["observed_human_pixels_outside_object"] == 40
    assert evidence["comparison_region_pixels"] == 56
    candidate[-1] = True  # Outside-human false positives remain in full union.
    assert gate.mask_evidence(raw, candidate, human, obj, 36)["candidate_iou"] == pytest.approx(40/48)


@pytest.mark.parametrize("fault", ["empty", "exact32", "dtype", "shape", "masked"])
def test_masks_fail_without_frame_dropping(gate, masks, fault):
    human, obj = masks; candidate = human.copy()
    if fault == "empty": human[:] = False
    elif fault == "exact32": human[:] = False; human[1:5] = True
    elif fault == "dtype": candidate = candidate.astype(np.uint8)
    elif fault == "shape": candidate = candidate[:7]
    else: candidate = np.ma.array(candidate, mask=False)
    with pytest.raises(ValueError): gate.mask_evidence(human, candidate, human, obj, 36)


def records(gate, delta=0.):
    return [{"frame_index": i, "raw_iou": .5, "candidate_iou": .5+delta,
             "delta_iou": (.5+delta)-.5, "observed_human_pixels_outside_object": 40} for i in gate.SELECTED]


def test_decision_midpoints_only_strict_mean_and_perframe_no_adoption(gate):
    rows = records(gate)
    for row in rows:
        if row["frame_index"] in gate.ANCHORS:
            row.update(candidate_iou=0., delta_iou=-.5)
    result = gate.decision(rows)
    assert result["consensus_gate_pass"] and result["midpoint_mean_delta_iou"] == 0.
    assert result["adoption_authorized"] is result["accuracy_verified"] is False
    assert not gate.decision(records(gate, -1e-8))["consensus_gate_pass"]
    rows = records(gate, .1)
    rows[gate.SELECTED.index(gate.MIDPOINTS[0])].update(candidate_iou=.4998, delta_iou=.4998-.5)
    assert not gate.decision(rows)["consensus_gate_pass"]  # Good mean cannot hide one >1e-4 regression.


@pytest.mark.parametrize("fault", ["missing", "duplicate", "order", "nan", "unlinked", "invalidrange", "support", "integeriou"])
def test_decision_rejects_incomplete_or_forged_evidence(gate, fault):
    rows = records(gate)
    if fault == "missing": rows.pop()
    elif fault == "duplicate": rows[-1] = rows[0].copy()
    elif fault == "order": rows.reverse()
    elif fault == "nan": rows[0]["delta_iou"] = np.nan
    elif fault == "unlinked": rows[0]["delta_iou"] = 1.
    elif fault == "invalidrange": rows[0].update(candidate_iou=1.2, delta_iou=.7)
    elif fault == "support": rows[0]["observed_human_pixels_outside_object"] = 32
    else: rows[0]["raw_iou"] = 0
    with pytest.raises(ValueError): gate.decision(rows)


def receipts(gate):
    common = dict(status="pass", episode_index=0, input_track="track_1", input_sha256=gate.VIDEO_SHA,
        input_dataset_revision=gate.body.DATASET_REVISION, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    raw = dict(common, stage="sam3d_body_full_video_initializer", total_video_frames=gate.FRAMES,
        frame_indices=list(range(gate.FRAMES)), script_sha256=gate.BODY_SCRIPT_SHA, predictions_sha256=gate.PREDICTIONS_SHA,
        network="none", mask_report_sha256=gate.MASK_REPORT_SHA, inference_type="body", body_revision=gate.body.BODY_REVISION,
        upstream_revision=gate.body.UPSTREAM_REVISION, camera_intrinsics="RGB_size_default_FOV", human_mask_id=0,
        prompt_mode="automatic_mask_and_derived_bbox_no_fallback", model_mask_range="uint8_0_1_matches_CARI_prepare_batch",
        vertices=gate.VERTICES, faces=gate.FACES, geometry_units="metres", geometry_frame="SAM3D_camera_x_right_y_down_z_forward",
        translation="vertices_camera_m = vertices_root_camera_m + pred_cam_t (exactly once)", mhr_geometry_forward_verified=True,
        mhr_geometry_forward_basis="original_body133_hand108_global_rot3_scale28_shape45_expr72_not_raw_logits",
        frames=[dict(frame_index=i, mask_sha256="a"*64, decoded_rgb_sha256="b"*64) for i in range(gate.FRAMES)],
        checkpoint_loading=dict(mode="strict_network_and_head_state_with_explicit_asset_buffer_retention",
            parameter_tensors_loaded=1101, unexpected_keys=[], retained_mhr_asset_buffer_names=[str(i) for i in range(113)]))
    masks = dict(common, stage="automatic_masks", frames=gate.FRAMES, script_sha256=gate.MASK_SCRIPT_SHA)
    return raw, masks


def test_receipts_actual_legacy_missing_revision_and_mask_network_not_invented(gate):
    raw, masks = receipts(gate); gate.validate_receipts(raw, masks)
    assert "producer_revision" not in raw and "network" not in masks
    assert gate.BODY_SCRIPT_SHA == gate.sha256(Path(gate.body.__file__))


@pytest.mark.parametrize("fault", ["track", "video", "dataset", "episodebool", "gtmissing", "gttrue", "labels", "oracle",
    "source", "fullcount", "frameorder", "network", "handmode", "maskhash", "checkpoint", "assetcount"])
def test_receipts_fail_closed(gate, fault):
    raw, masks = receipts(gate)
    if fault == "track": raw["input_track"] = "track_2"
    elif fault == "video": masks["input_sha256"] = "c"*64
    elif fault == "dataset": masks["input_dataset_revision"] = "x"
    elif fault == "episodebool": raw["episode_index"] = False
    elif fault == "gtmissing": del masks["ground_truth_used"]
    elif fault == "gttrue": raw["ground_truth_used"] = True
    elif fault == "labels": raw["hand_labeled_test"] = True
    elif fault == "oracle": raw["oracle_modes"] = ["human"]
    elif fault == "source": raw["script_sha256"] = "d"*64
    elif fault == "fullcount": masks["frames"] -= 1
    elif fault == "frameorder": raw["frames"][-1]["frame_index"] = 0
    elif fault == "network": raw["network"] = "host"
    elif fault == "handmode": raw["inference_type"] = "full"
    elif fault == "maskhash": raw["mask_report_sha256"] = "e"*64
    elif fault == "checkpoint": raw["checkpoint_loading"]["unexpected_keys"] = ["unknown"]
    else: raw["checkpoint_loading"]["retained_mhr_asset_buffer_names"].pop()
    with pytest.raises(ValueError): gate.validate_receipts(raw, masks)


def small_dimensions(gate, monkeypatch):
    for key, value in dict(FRAMES=5, VERTICES=4, FACES=2, WIDTH=8, HEIGHT=8,
        ANCHORS=(0, 2, 4), MIDPOINTS=(1, 3), SELECTED=(0, 1, 2, 3, 4)).items(): monkeypatch.setattr(gate, key, value)


def prediction_arrays(gate):
    arrays = {k: np.zeros((gate.FRAMES, *v), np.float32) for k, v in gate.body.PARAMETER_SHAPES.items() if k != "pred_vertices"}
    verts = np.array([[0, 0, .1], [1, 0, .1], [0, 1, .1], [.2, .3, 1]], np.float32)
    arrays.update(vertices_root_camera_m=np.repeat(verts[None], gate.FRAMES, axis=0),
        faces=np.array([[0, 1, 2], [0, 2, 3]], np.int64), frame_index=np.arange(gate.FRAMES, dtype=np.int64))
    arrays["pred_cam_t"][:, 2] = 2
    arrays["vertices_camera_m"] = arrays["vertices_root_camera_m"]+arrays["pred_cam_t"][:, None]
    arrays["focal_length"][:] = np.hypot(gate.WIDTH, gate.HEIGHT)
    return arrays


def test_prediction_loader_full_schema_and_exact_translation(gate, monkeypatch, tmp_path):
    small_dimensions(gate, monkeypatch); arrays = prediction_arrays(gate)
    path = tmp_path/"predictions.npz"; np.savez_compressed(path, **arrays)
    loaded = gate.load_predictions(path)
    assert loaded["vertices_camera_m"].tobytes() == arrays["vertices_camera_m"].tobytes()
    assert "pred_pose_raw" not in loaded and "pred_global_rots" not in loaded


@pytest.mark.parametrize("fault", ["extra", "missing", "dtype", "index", "expr", "focal", "translation", "faces", "nanunused"])
def test_prediction_loader_validates_unused_rows_and_no_repairs(gate, monkeypatch, tmp_path, fault):
    small_dimensions(gate, monkeypatch); arrays = prediction_arrays(gate)
    if fault == "extra": arrays["oracle_mask"] = np.zeros(1)
    elif fault == "missing": del arrays["pred_pose_raw"]
    elif fault == "dtype": arrays["shape_params"] = arrays["shape_params"].astype(np.float64)
    elif fault == "index": arrays["frame_index"][-1] = 0
    elif fault == "expr": arrays["expr_params"][-1, 0] = .1
    elif fault == "focal": arrays["focal_length"][-1] += 1
    elif fault == "translation": arrays["vertices_camera_m"][-1, 0, 2] += 1
    elif fault == "faces": arrays["faces"][0, 2] = 0
    else: arrays["pred_global_rots"][-1, 0, 0, 0] = np.nan
    path = tmp_path/"bad.npz"; np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError): gate.load_predictions(path)


def test_regular_old_0644_accepted_but_hash_and_symlink_fail(gate, tmp_path):
    path = tmp_path/"old.json"; path.write_text("{}"); path.chmod(0o644)
    before = path.stat().st_mode; frozen = []
    gate.freeze_input(path, frozen, digest=gate.sha256(path), size=2)
    assert path.stat().st_mode == before and len(frozen) == 1
    with pytest.raises(ValueError): gate.freeze_input(path, [], digest="f"*64)
    with pytest.raises(ValueError): gate.freeze_input(path, [], size=3)
    link = tmp_path/"link"; link.symlink_to(path)
    with pytest.raises(ValueError): gate.regular(link)
    with pytest.raises(ValueError): gate.regular(tmp_path)


@pytest.fixture
def sources(gate, monkeypatch, tmp_path):
    small_dimensions(gate, monkeypatch)
    root = tmp_path; base = root/"outputs/episode_000000"; masks_dir = base/"automatic_masks"
    for folder in (root/"results", root/"data/track_1/meta", base/"body_full", masks_dir/"masks/0", masks_dir/"masks/1"):
        folder.mkdir(parents=True)
    raw, mask_report = receipts(gate)
    for i in range(5):
        human = np.zeros((8, 8), np.uint8); human[:6] = 255
        for label, value in ((0, human), (1, np.zeros_like(human))):
            target = masks_dir/f"masks/{label}/{i:06d}.png"; Image.fromarray(value).save(target)
        raw["frames"][i]["mask_sha256"] = gate.sha256(masks_dir/f"masks/0/{i:06d}.png")
    prompts = masks_dir/"prompts.json"; prompts.write_text(json.dumps({"prompts": [{"object_id": 0}, {"object_id": 1}]}))
    raw["prompts_sha256"] = gate.sha256(prompts)
    meta = root/"data/track_1/meta/episodes.jsonl"; meta.write_text('{"episode_index":0,"length":5}\n')
    manifest = root/"results/input-manifest.json"
    manifest.write_text(json.dumps(dict(track="track_1", repo_id="nvidia/video_to_data_challenge", revision=gate.body.DATASET_REVISION,
        files=[dict(path="track_1/meta/episodes.jsonl", bytes=meta.stat().st_size, sha256=gate.sha256(meta)),
            dict(path="track_1/videos/chunk-000/observation.images.exo_camera/episode_000000.mp4", sha256=gate.VIDEO_SHA)])))
    assets = root/"weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3"; (assets/"assets").mkdir(parents=True)
    ids = {}
    for name in ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt", "LICENSE"):
        path = assets/name; path.write_text(name); ids[name] = dict(bytes=path.stat().st_size, sha256=gate.sha256(path))
    raw["body_assets"] = ids; raw["inference_source_identity"] = {"python_files": 1, "sha256": "f"*64}
    (root/"results/weights-acquisition.json").write_text('{}')
    monkeypatch.setattr(gate, "CHECKPOINT_SHA", ids["model.ckpt"]["sha256"])
    monkeypatch.setattr(gate, "CHECKPOINT_BYTES", ids["model.ckpt"]["bytes"])
    monkeypatch.setattr(gate.body, "_body_assets", lambda p: (assets, ids))
    monkeypatch.setattr(gate.body, "_source_identity", lambda p: raw["inference_source_identity"])
    predictions = base/"body_full/predictions.npz"; np.savez_compressed(predictions, **prediction_arrays(gate))
    monkeypatch.setattr(gate, "PREDICTIONS_SHA", gate.sha256(predictions)); raw["predictions_sha256"] = gate.PREDICTIONS_SHA
    monkeypatch.setattr(gate, "PREDICTIONS_BYTES", predictions.stat().st_size)
    mask_path = masks_dir/"report.json"; mask_path.write_text(json.dumps(mask_report))
    monkeypatch.setattr(gate, "MASK_REPORT_SHA", gate.sha256(mask_path)); raw["mask_report_sha256"] = gate.MASK_REPORT_SHA
    monkeypatch.setattr(gate, "MASK_REPORT_BYTES", mask_path.stat().st_size)
    raw_path = base/"body_full/report.json"; raw_path.write_text(json.dumps(raw))
    monkeypatch.setattr(gate, "BODY_REPORT_SHA", gate.sha256(raw_path)); monkeypatch.setattr(gate, "BODY_REPORT_BYTES", raw_path.stat().st_size)
    return root, raw_path, masks_dir, meta, manifest


def test_real_source_consumer_hashes_selected_both_masks_no_video_or_private(gate, sources):
    root, _, _, _, _ = sources
    arrays, masks, mask_records, assets, ids, installed, frozen = gate.source_inputs(root)
    assert len(masks) == len(mask_records) == 5 and len(arrays["global_rot"]) == 5
    assert len([p for p, h in frozen if p.suffix == ".png"]) == 10
    assert all(gate.sha256(p) == h for p, h in frozen)
    assert not any(".mp4" in str(p) or "eval_private" in str(p) for p, _ in frozen)
    assert all("decoded_rgb_sha256" in r and "object_mask_sha256" in r for r in mask_records)


@pytest.mark.parametrize("fault", ["bodyreceipt", "objectbytes", "humansha", "maskindex", "maskextra", "objectsymlink",
    "manualprompt", "metadata", "manifest", "predictions", "asset"])
def test_real_source_consumer_tampering_fail_before_native(gate, sources, fault):
    root, raw_path, masks_dir, meta, manifest = sources
    if fault == "bodyreceipt": raw_path.write_text('{}')
    elif fault == "objectbytes": (masks_dir/"masks/1/000000.png").write_text("notPNG")
    elif fault == "humansha": Image.fromarray(np.ones((8, 8), np.uint8)*255).save(masks_dir/"masks/0/000000.png")
    elif fault == "maskindex": (masks_dir/"masks/1/000004.png").unlink()
    elif fault == "maskextra": (masks_dir/"masks/0/extra.txt").write_text('x')
    elif fault == "objectsymlink":
        p = masks_dir/"masks/1/000000.png"; p.unlink(); p.symlink_to(masks_dir/"masks/1/000001.png")
    elif fault == "manualprompt": (masks_dir/"prompts.json").write_text('{"prompts":[{"object_id":0,"points":[1,2]},{"object_id":1}]}')
    elif fault == "metadata": meta.write_text('{"episode_index":0,"length":4}\n')
    elif fault == "manifest":
        value = json.loads(manifest.read_text()); value["track"] = "track_3"; manifest.write_text(json.dumps(value))
    elif fault == "predictions": (root/"outputs/episode_000000/body_full/predictions.npz").write_text("bad")
    else: (root/"weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt").write_text("bad")
    with pytest.raises((ValueError, OSError)): gate.source_inputs(root)


def test_wrapper_syntax_exclusive_mounts_no_gt_or_heavy_unused_inputs(gate):
    path = Path(gate.__file__).with_name("run_identity_consensus_public.sh")
    subprocess.run(["bash", "-n", str(path)], check=True)
    source = path.read_text()
    assert "--network none" in source and "--memory 32g --cpus 4" in source and "183s" in source
    assert gate.IMAGE in source and "! -e \"$OUT\"" in source and "(( $# == 0 ))" in source
    assert "--env CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    mounts = [line for line in source.splitlines() if '--mount "' in line]
    assert len(mounts) == 13 and all("readonly" in line for line in mounts[:-1])
    assert "src=$OUT,dst=$OUT" in mounts[-1]
    assert not any(x in source for x in ("eval_private", "videos/", "cari_forward", "weights/mhr", "src=$ROOT/outputs,dst"))
    # Candidate freezes before ANY raw/candidate rasterization; gate never fits.
    tree = ast.parse(Path(gate.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    text = ast.unparse(run)
    assert text.index("freeze_candidate(") < text.index("raster_camera_mesh(")
    assert "official" not in text and "convert(" not in text


def test_unknown_cli_argument_fails_before_runtime(gate):
    with pytest.raises(SystemExit) as exc: gate.main(["--episode", "1"])
    assert exc.value.code == 2
