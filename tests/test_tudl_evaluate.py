"""Real numeric/private firewall tests on tiny own arrays, never dataset assets."""
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def evaluate(monkeypatch):
    source = Path(__file__).parents[1]/"infra/tudl_evaluate.py"
    spec = importlib.util.spec_from_file_location("tudl_eval_tests", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WIDTH", 8); monkeypatch.setattr(module, "HEIGHT", 6)
    return module


def native_points(z, K):
    y, x = np.mgrid[:z.shape[0], :z.shape[1]]
    return np.stack(((x+.5-K[0, 2])*z/K[0, 0], (y+.5-K[1, 2])*z/K[1, 1], z), axis=-1).astype(np.float32)


def predictions(scene, frame, focal=10.):
    K = np.array([[focal, 0., 4.], [0., focal, 3.], [0., 0., 1.]])
    arrays = {"scene_id": np.array(scene, np.int64), "frame_id": np.array(frame, np.int64)}
    for mode, z in (("fixed", 2.2), ("learned", 2.)):
        depth = np.full((6, 8), z, np.float32)
        arrays.update({mode+"_depth": depth, mode+"_points": native_points(depth, K), mode+"_K": K.copy(), mode+"_validity": np.ones((6, 8), bool)})
    return arrays


def write_json(path, value):
    path.write_text(json.dumps(value)); return path


@pytest.fixture
def frozen(evaluate, tmp_path, monkeypatch):
    base = tmp_path/"validation/tudl_rgb_v1"; public = base/"inputs"; pred = base/"predictions_v1"; private = base/"eval_private"
    for p in (public, pred, private): p.mkdir(parents=True)
    images, outputs, data = [], [], []
    for scene, frame in evaluate.FRAMES:
        stem = f"scene_{scene:06d}_frame_{frame:06d}"; rgb = public/(stem+".png"); rgb.write_bytes(b"tiny own RGB identity")
        arrays = predictions(scene, frame); np.savez(pred/(stem+".npz"), **arrays); data.append(arrays)
        images.append(dict(scene_id=scene, frame_id=frame, file=rgb.name, sha256=evaluate.sha256(rgb), width=8, height=6))
        outputs.append(dict(scene_id=scene, frame_id=frame, file=stem+".npz", sha256=evaluate.sha256(pred/(stem+".npz")), rgb_sha256=images[-1]["sha256"]))
    mp = write_json(public/"manifest.json", dict(schema="world-reward-tudl-rgb-v1", revision=evaluate.REVISION,
                    license="CC-BY-SA-4.0", selection=evaluate.SELECTION, images=images))
    calls = [{"scene_id": s, "frame_id": f, "file": f"scene_{s:06d}_frame_{f:06d}.png", "pass": phase,
              "focal_prior_supplied": phase != "camera_candidate", "original_solver_returned": True, "native_nearest64_valid_pixels": 40}
             for phase in ("fixed_prior", "camera_candidate", "shared_camera") for s, f in evaluate.FRAMES]
    receipt = dict(stage="public_tudl_rgb_native_camera_depth_predictions", status="pass", producer_revision="a"*40, image_id="sha256:"+"b"*64,
                   private_truth_read=False, challenge_inputs_used=False, ground_truth_used=False, hand_labeled_test=False,
                   oracle_modes=[], scale_fit=False, network="none", actual_MoGe_inference=True, native_camera_support_verified=True,
                   apply_mask=False, force_projection=True, pointmap_geometry_filled=False, original_frame_coverage_verified=True,
                   focal_geometry_source_sha256=evaluate.GEOMETRY_SHA, model_asset={"sha256": evaluate.MODEL_SHA}, model_source={"revision": evaluate.MODEL_SOURCE},
                   public_records=images, public_input_manifest_sha256=evaluate.sha256(mp), focal_candidates_pixels=[10.]*9, scene_focal_pixels=[10.]*3,
                   native_focal_solver_calls=calls, outputs=outputs, MoGe_calls_completed=27, prior_calls_completed=18, no_prior_calls_completed=9,
                   fixed_prior_calls_completed=9, camera_candidate_calls_completed=9, shared_camera_calls_completed=9, outputs_completed=9)
    rp = write_json(pred/"report.json", receipt)
    for scene in (1, 2, 3):
        directory = private/f"source/test/{scene:06d}"; directory.mkdir(parents=True)
        ids = [f for s, f in evaluate.FRAMES if s == scene]
        # Integer GT rays equivalent to unchanged MoGe+.5 rays, not oracle recalibration.
        camera = {str(f): dict(cam_K=[10., 0., 3.5, 0., 10., 2.5, 0., 0., 1.], depth_scale=1.) for f in ids}
        write_json(directory/"scene_camera.json", camera); write_json(directory/"scene_gt.json", {str(f): [{"obj_id": scene}] for f in ids})
        for f in ids:
            for sub, name in (("depth", f"{f:06d}.png"), ("mask_visib", f"{f:06d}_000000.png")):
                p = directory/sub/name; p.parent.mkdir(exist_ok=True); p.write_bytes(b"tiny private sensor/mask identity")
    retained = [dict(file=str(p.relative_to(private)), sha256=evaluate.sha256(p), bytes=p.stat().st_size) for p in sorted((private/"source").rglob("*")) if p.is_file()]
    acquisition = dict(status="pass", dataset_revision=evaluate.REVISION, license="CC-BY-SA-4.0", public_manifest_sha256=evaluate.sha256(mp),
                       selection_before_private_annotation_values=True, private_annotations_exported_as_inference_inputs=False,
                       retained_files=retained, selected_records=[dict(scene_id=r["scene_id"], frame_id=r["frame_id"], file=r["file"], rgb_sha256=r["sha256"]) for r in images])
    ap = write_json(private/"acquisition-report.json", acquisition)
    monkeypatch.setattr(evaluate, "read_png", lambda p, depth=False: np.full((6, 8), 2000.) if depth else np.ones((6, 8), bool))
    def sync():
        for arrays, output in zip(data, outputs):
            np.savez(pred/output["file"], **arrays); output["sha256"] = evaluate.sha256(pred/output["file"])
        write_json(rp, receipt)
    return tmp_path, data, receipt, acquisition, sync, ap


def test_native_pixel_convention_integer_GT_keeps_prediction_XYZ(evaluate):
    z = np.full((6, 8), 2.); K = np.array([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]])
    integer_K = K.copy(); integer_K[:2, 2] -= .5
    np.testing.assert_allclose(evaluate.sensor_points(z, integer_K, np.ones_like(z, bool)), native_points(z, K).reshape(-1, 3), atol=3e-8)


def test_same_pixel_depth_perfect_but_wrong_focal_not_erased(evaluate):
    data = predictions(1, 0); data["learned_depth"][:] = 2.; data["fixed_depth"][:] = 2.
    K = data["fixed_K"]; wrong = K.copy(); wrong[0, 0] = wrong[1, 1] = 5.
    data["fixed_points"] = native_points(data["fixed_depth"], wrong)
    gtK = K.copy(); gtK[:2, 2] -= .5
    score = evaluate.score_frame(data, np.full((6, 8), 2.), gtK, np.ones((6, 8), bool))
    assert score["fixed"]["sensor_camera_Z_absrel"] == 0.
    assert score["fixed"]["sensor_visible_camera_chamfer_half_cm"] > 15.
    assert score["learned"]["sensor_visible_camera_chamfer_half_cm"] < 2e-6


def test_real_full_numeric_flow_all9_beforeprivate_no_alignment(evaluate, frozen):
    root, _, _, _, _, _ = frozen; report = {}; evaluate.run(root, report)
    assert report["status"] == "pass" and len(report["frames"]) == 9
    assert report["predictions_frozen_before_private_truth_read"]
    assert report["decision"]["real_object_camera_hypothesis_supported"]
    assert report["decision"]["median_paired_scene_relative_gain"] > .99999
    assert not report["alignment_performed"] and not report["predicted_XYZ_camera_or_scale_changed"]
    assert not report["human_quality_verified"] and not report["training_overlap_excluded"]


@pytest.mark.parametrize("fault", ["private", "partial", "sha", "source", "manifest", "focal", "nativefallback", "nativeorder", "missing", "nan", "K", "indices", "extra"])
def test_bad_public_predictions_rejected_before_any_private_file(evaluate, frozen, monkeypatch, fault):
    root, arrays, receipt, _, sync, _ = frozen
    if fault == "private": receipt["private_truth_read"] = True
    elif fault == "partial": receipt["MoGe_calls_completed"] = 26
    elif fault == "sha": receipt["public_input_manifest_sha256"] = "c"*64
    elif fault == "source": receipt["model_source"]["revision"] = "c"*40
    elif fault == "manifest": receipt["public_records"] = []
    elif fault == "focal": receipt["scene_focal_pixels"][0] = 11.
    elif fault == "nativefallback": receipt["native_focal_solver_calls"][0]["native_nearest64_valid_pixels"] = 1
    elif fault == "nativeorder": receipt["native_focal_solver_calls"][0]["pass"] = "shared_camera"
    elif fault == "missing": receipt["outputs"] = receipt["outputs"][:-1]
    elif fault == "nan": arrays[-1]["learned_points"][0, 0, 0] = np.nan
    elif fault == "K": arrays[-1]["learned_K"][0, 0] = 11.
    elif fault == "indices": arrays[-1]["frame_id"] = np.array(3, np.int64)
    elif fault == "extra": arrays[-1]["gt_camera"] = np.eye(3)
    sync(); original = evaluate.regular_hash
    def guard(path, digest=None):
        assert "eval_private" not in str(path), "Private truth opened before rejected public outputs"
        return original(path, digest)
    monkeypatch.setattr(evaluate, "regular_hash", guard)
    with pytest.raises(ValueError): evaluate.run(root, {})


def test_validity_discard_cannot_turn_shared_support_into_passing_gain(evaluate, frozen):
    root, arrays, _, _, sync, _ = frozen
    arrays[-1]["learned_validity"][0, 0] = False; sync(); report = {}; evaluate.run(root, report)
    assert len(report["frames"]) == 9 and report["frames"][-1]["learned"]["sensor_valid_object_coverage"] > .95
    assert not report["decision"]["coverage_gate_pass"]
    assert not report["decision"]["real_object_camera_hypothesis_supported"]


def test_one_scene_regression_and_zero_baseline_cannot_hide(evaluate):
    frames = []
    for i, (s, f) in enumerate(evaluate.FRAMES):
        frames.append(dict(scene_id=s, frame_id=f, fixed={"sensor_visible_camera_chamfer_half_cm": 10.},
                           learned={"sensor_visible_camera_chamfer_half_cm": (8., 9., 11.)[i//3]}, coverage_gate_pass=True))
    assert not evaluate.decision(frames)["real_object_camera_hypothesis_supported"]
    frames[0]["fixed"]["sensor_visible_camera_chamfer_half_cm"] = 0.
    frames[1]["fixed"]["sensor_visible_camera_chamfer_half_cm"] = 0.
    frames[2]["fixed"]["sensor_visible_camera_chamfer_half_cm"] = 0.
    assert evaluate.decision(frames)["median_paired_scene_relative_gain"] is None
    with pytest.raises(ValueError): evaluate.decision(frames[:-1])


def test_private_perimage_depth_scale_consumed_not_global_assumed(evaluate, frozen):
    root, _, _, acquisition, _, ap = frozen; private = ap.parent
    path = private/"source/test/000001/scene_camera.json"; camera = json.loads(path.read_text()); camera["0"]["depth_scale"] = 2.
    write_json(path, camera)
    for row in acquisition["retained_files"]:
        if row["file"] == str(path.relative_to(private)): row.update(sha256=evaluate.sha256(path), bytes=path.stat().st_size)
    write_json(ap, acquisition); report = {}; evaluate.run(root, report)
    assert report["frames"][0]["learned"]["sensor_camera_Z_mae_cm"] == 200.
    assert report["frames"][1]["learned"]["sensor_camera_Z_mae_cm"] == 0.


def test_eval_wrapper_cpu_private_readonly_no_models_or_parent_mount():
    root = Path(__file__).parents[1]; wrapper = root/"infra/run_tudl_evaluate.sh"; source = wrapper.read_text()
    assert "--gpus" not in source and "weights" not in source and "--network none" in source
    assert "src=$BASE/eval_private,dst=$BASE/eval_private,readonly" in source
    assert "src=$BASE,dst=$BASE" not in source
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    assert subprocess.run(["bash", str(wrapper), "--unknown"], capture_output=True).returncode == 2
