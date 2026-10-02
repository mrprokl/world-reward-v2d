"""Actual tiny numeric metrics/gauge/firewall tests; no models, RGB decoding or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from world_reward.metric_alignment import fit_shared_depth_scale


@pytest.fixture
def evaluate():
    path = Path(__file__).parents[1]/"infra/joint_rgb_evaluate.py"
    spec = importlib.util.spec_from_file_location("joint_eval_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def cloud():
    return np.array([[-.4, -.3, 3.], [.5, -.3, 3.1], [-.1, .6, 2.9], [.2, .1, 3.4]])


def test_exact_human_correspondence_sim3_proper_noncommuting_rotation(evaluate):
    x = cloud(); R = Rotation.from_rotvec([.2, -.3, .1]).as_matrix(); y = 1.3*x@R.T+[.1, -.2, .3]
    fit = evaluate.fit_human_sim3(x, y)
    np.testing.assert_allclose(evaluate.transform(x, fit), y, atol=2e-15)
    np.testing.assert_allclose(fit["rotation"], R, atol=1e-14)
    assert fit["scale"] == pytest.approx(1.3) and np.linalg.det(fit["rotation"]) == pytest.approx(1)


def test_one_human_scale_cancels_common_gamma_but_does_not_erase_object_only_bias(evaluate):
    human = cloud(); obj = cloud()+[.8, .1, -.2]; gamma = 1.2
    fit = evaluate.fit_human_sim3(human*gamma, human)
    np.testing.assert_allclose(evaluate.transform(obj*gamma, fit), obj, atol=2e-15)
    unchanged = evaluate.fit_human_sim3(human, human)
    score = evaluate.metrics(evaluate.transform(obj*gamma, unchanged), obj, human, human)
    assert score["visible_chamfer_half_cm"] > 20
    assert score["signed_object_median_Z_bias_cm"] > 50


def test_shared_depth_ratio_common_gamma_invariance_not_object_only_invariance():
    depths = np.full((3, 6, 6), 3.75); render = np.full_like(depths, 3.); mask = np.ones_like(depths, bool)
    a = fit_shared_depth_scale(depths, render, mask, [0, 1, 2])
    b = fit_shared_depth_scale(depths*1.2, render*1.2, mask, [0, 1, 2])
    assert a.shared_scale == pytest.approx(.8) and b.shared_scale == pytest.approx(a.shared_scale)


def test_alpha_identity_metrics_and_half_not_sum(evaluate):
    p = cloud(); identity = evaluate.metrics(p, p.copy()*1., p, p)
    assert identity["visible_chamfer_half_cm"] == identity["signed_object_median_Z_bias_cm"] == 0
    score = evaluate.metrics(p+[1., 0, 0], p, p, p)
    assert score["visible_chamfer_half_cm"] == pytest.approx(.5*(score["pred_to_true_cm"]+score["true_to_pred_cm"]))
    assert score["relative_visible_centroid_vector_error_cm"] == pytest.approx(100)


def test_visible_raster_camera_positive_z_half_pixel_and_human_occlusion(evaluate):
    ids = np.array([[-1, 0, 3], [2, -1, 1]])
    depth = np.array([[np.nan, 2., 4.], [3., np.nan, 2.]])
    K = np.array([[2., 0, 1.], [0, 4., 1.], [0, 0, 1.]])
    p, count = evaluate.visible_object_points(depth, ids, K, 2, 2)
    np.testing.assert_array_equal(p, [[3., -.5, 4.], [-.75, .375, 3.]])
    assert count == 2


@pytest.mark.parametrize("fault", ["noobject", "nanvisible", "negative", "badface", "floatids", "badK"])
def test_invalid_or_missing_visible_truth_never_dropped(evaluate, fault):
    ids = np.ones((2, 2), dtype=np.int64); z = np.ones((2, 2), dtype=float); K = np.eye(3)
    if fault == "noobject": ids[:] = 0
    if fault == "nanvisible": z[0, 0] = np.nan
    if fault == "negative": z[0, 0] = -1
    if fault == "badface": ids[0, 0] = 2
    if fault == "floatids": ids = ids.astype(float)
    if fault == "badK": K[0, 0] = np.nan
    with pytest.raises(ValueError): evaluate.visible_object_points(z, ids, K, 1, 1)


def test_wrong_focal_context_regression_cannot_hide_behind_median_gain(evaluate):
    cases = [{"raw_visible_chamfer_half_cm": 10., "aligned_visible_chamfer_half_cm": x} for x in [9., 8., 11.]]
    result = evaluate.decision(cases)
    assert result["median_paired_clip_relative_gain"] == pytest.approx(.1)
    assert not result["synthetic_hypothesis_supported"]
    cases[-1]["aligned_visible_chamfer_half_cm"] = 10.
    assert evaluate.decision(cases)["synthetic_hypothesis_supported"]
    cases[0]["raw_visible_chamfer_half_cm"] = 0
    result = evaluate.decision(cases)
    assert result["zero_baseline_gain_undefined"] and result["median_paired_clip_relative_gain"] is None
    with pytest.raises(ValueError): evaluate.decision(cases[:2])


def write_json(path, data):
    path.write_text(json.dumps(data)); return path


@pytest.fixture
def frozen(evaluate, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "WIDTH", 8); monkeypatch.setattr(evaluate, "HEIGHT", 6)
    root = tmp_path; base = root/"validation/joint_rgb_v1"
    public = base/"inputs"; private = base/"eval_private"; pred = base/"predictions"
    for p in (public, private, pred): p.mkdir(parents=True)
    rng = np.random.default_rng(0); human = rng.normal(0, .05, (18439, 3))+[0., 0., 3.]
    hf = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    y, x = np.mgrid[:6, :8]; raw = []; records = []; cases = []
    for i in range(9):
        clip, frame = divmod(i, 3); stem = f"clip_{clip:02d}_frame_{frame:03d}"
        rgb = public/(stem+".png"); rgb.write_bytes(b"public dummy RGB identity only")
        K = np.array([[(1280., 960., 1600.)[clip], 0., 4.], [0., (1280., 960., 1600.)[clip], 3.], [0., 0., 1.]])
        z = np.full((6, 8), 2.); ids = np.full((6, 8), len(hf), np.int64)
        xyz = np.stack(((x+.5-4)*z/K[0, 0], (y+.5-3)*z/K[1, 1], z), axis=-1)
        raw.append(xyz*1.25)
        truth_path = private/(stem+".npz")
        np.savez(truth_path, human_vertices_camera_m=human, human_faces=hf,
                 object_vertices_camera_m=np.array([[0., 0, 2.], [.1, 0, 2.], [0, .1, 2.]]),
                 object_faces=np.array([[0, 1, 2]], np.int64), camera_K=K, scene_depth_m=z,
                 visible_face_indices=ids, clip_index=np.array(clip), frame_index=np.array(frame))
        records.append({"file": rgb.name, "sha256": evaluate.sha256(rgb), "width": 8, "height": 6})
        cases.append({"file": rgb.name, "rgb_sha256": evaluate.sha256(rgb), "truth_sha256": evaluate.sha256(truth_path), "clip_index": clip, "frame_index": frame})
    raw = np.asarray(raw)
    arrays = dict(raw_points=raw, aligned_points=raw*.8, human_vertices_camera_m=np.tile(human, (9, 1, 1)),
                  human_faces=hf, object_masks=np.ones((9, 6, 8), bool), moge_validity=np.ones((9, 6, 8), bool),
                  shared_scale=np.full(3, .8), frame_index=np.tile(np.arange(3), 3), clip_index=np.repeat(np.arange(3), 3),
                  camera_K=np.array([[1280., 0, 4.], [0, 1280., 3.], [0, 0, 1.]]))
    np.savez(pred/"arrays.npz", **arrays)
    receipt = {"stage": "public_joint_rgb_shared_grounding_predictions", "status": "pass", "private_truth_read": False,
               "challenge_inputs_used": False, "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "arrays_sha256": evaluate.sha256(pred/"arrays.npz")}
    write_json(public/"manifest.json", {"schema": "world-reward-joint-rgb-v1", "images": records})
    receipt.update(public_inputs_sha=evaluate.sha256(public/"manifest.json"), mask_report_sha="b"*64,
                   public_records=[{"file": r["file"], "rgb_sha256": r["sha256"], "clip_index": i//3, "frame_index": i%3,
                                    "human_mask_sha256": "c"*64, "object_mask_sha256": "d"*64} for i, r in enumerate(records)])
    write_json(pred/"report.json", receipt)
    write_json(private/"render-report.json", {"stage": "own_joint_human_object_rgb_render", "status": "pass", "synthetic_truth_used_for_rendering_only": True,
                                            "inference_performed": False, "challenge_inputs_used": False, "cases": cases})
    return root, arrays, receipt


def test_complete_private_pipeline_real_metrics_all_frames_same_sim3_not_selected(evaluate, frozen):
    root, _, _ = frozen; report = {}
    evaluate.run(root, report)
    assert report["predictions_frozen_before_private_truth_read"] and report["original_frame_coverage_verified"]
    assert len(report["frames"]) == 9 and report["decision"]["synthetic_hypothesis_supported"]
    assert report["negative_diagnostic_used_for_selection"] is False
    for r in report["frames"]:
        assert r["raw_camera"]["raw"]["visible_chamfer_half_cm"] > 49
        assert r["raw_camera"]["aligned"]["visible_chamfer_half_cm"] < 1e-12
    assert report["shared_human_sim3_identical_for_both_methods"]


@pytest.mark.parametrize("fault", ["truthread", "sha", "scale", "indices", "mask", "K", "nan"])
def test_prediction_failure_before_private_read_no_repair(evaluate, frozen, monkeypatch, fault):
    root, arrays, receipt = frozen; pred = root/"validation/joint_rgb_v1/predictions"
    if fault == "truthread": receipt["private_truth_read"] = True
    if fault == "sha": receipt["arrays_sha256"] = "a"*64
    if fault == "scale": arrays["shared_scale"][0] = .7
    if fault == "indices": arrays["frame_index"][1] = 0
    if fault == "mask": arrays["object_masks"][0] = False
    if fault == "K": arrays["camera_K"][0, 0] = 960.
    if fault == "nan": arrays["raw_points"][0, 0, 0] = np.nan
    if fault not in ("truthread", "sha"):
        np.savez(pred/"arrays.npz", **arrays); receipt["arrays_sha256"] = evaluate.sha256(pred/"arrays.npz")
    write_json(pred/"report.json", receipt)
    original = evaluate.require_hash
    def guard(path, digest):
        assert "eval_private" not in str(path), "Private truth accessed before rejected predictions"
        return original(path, digest)
    monkeypatch.setattr(evaluate, "require_hash", guard)
    with pytest.raises(ValueError): evaluate.run(root, {})


def test_missing_private_visibility_and_wrong_focal_are_full_failure(evaluate, frozen):
    root, _, _ = frozen; private = root/"validation/joint_rgb_v1/eval_private"
    path = private/"clip_01_frame_001.npz"
    with np.load(path) as a: data = {k: a[k] for k in a.files}
    data["visible_face_indices"][:] = -1; data["scene_depth_m"][:] = np.nan
    np.savez(path, **data)
    report_path = private/"render-report.json"; receipt = json.loads(report_path.read_text())
    receipt["cases"][4]["truth_sha256"] = evaluate.sha256(path); write_json(report_path, receipt)
    with pytest.raises(ValueError, match="No finite positive"): evaluate.run(root, {})


@pytest.mark.parametrize("fault", ["manifest", "RGB", "maskSHA", "boolclip"])
def test_consumed_public_provenance_rejected_before_private_truth(evaluate, frozen, monkeypatch, fault):
    root, _, receipt = frozen; path = root/"validation/joint_rgb_v1/predictions/report.json"
    if fault == "manifest": receipt["public_inputs_sha"] = "a"*64
    if fault == "RGB": receipt["public_records"][0]["rgb_sha256"] = "a"*64
    if fault == "maskSHA": receipt["public_records"][0]["human_mask_sha256"] = "missing"
    if fault == "boolclip": receipt["public_records"][0]["clip_index"] = False
    write_json(path, receipt)
    original = evaluate.require_hash
    def guard(p, digest):
        assert "eval_private" not in str(p)
        return original(p, digest)
    monkeypatch.setattr(evaluate, "require_hash", guard)
    with pytest.raises(ValueError): evaluate.run(root, {})


def test_negative_permuted_alpha_remains_diagnostic_not_metric_selection(evaluate, frozen):
    root, arrays, receipt = frozen; pred = root/"validation/joint_rgb_v1/predictions"
    arrays["shared_scale"] = np.array([.8, .9, 1.1])
    arrays["aligned_points"] = arrays["raw_points"]*np.repeat(arrays["shared_scale"], 3)[:, None, None, None]
    np.savez(pred/"arrays.npz", **arrays); receipt["arrays_sha256"] = evaluate.sha256(pred/"arrays.npz")
    write_json(pred/"report.json", receipt)
    report = {}; evaluate.run(root, report)
    assert report["negative_diagnostic_used_for_selection"] is False
    first = report["frames"][0]
    assert first["alpha_permuted_negative_diagnostic"]["visible_chamfer_half_cm"] > first["raw_camera"]["aligned"]["visible_chamfer_half_cm"]
    assert report["clips"][0]["aligned_visible_chamfer_half_cm"] < 1e-12


def test_wrapper_private_readonly_no_gpu_models_and_reject_symlink_early(evaluate, tmp_path):
    wrapper = Path(evaluate.__file__).with_name("run_joint_rgb_evaluate.sh"); source = wrapper.read_text()
    assert "--gpus" not in source and "weights" not in source and "--network none" in source
    assert "src=$BASE/eval_private,dst=$BASE/eval_private,readonly" in source
    assert "src=$BASE/predictions,dst=$BASE/predictions,readonly" in source
    assert "src=$BASE,dst=$BASE" not in source
    assert subprocess.run(["bash", str(wrapper), "--unknown"], capture_output=True).returncode == 2
    real = tmp_path/"real"; real.mkdir(); link = tmp_path/"linked"; link.symlink_to(real, target_is_directory=True)
    env = dict(os.environ, WR_ROOT=str(link), WR_CODE=str(tmp_path), WR_CODE_REVISION="a"*40)
    assert subprocess.run(["bash", str(wrapper)], env=env, capture_output=True).returncode == 2


def test_frozen_hash_rejects_symlink_and_byte_tamper(evaluate, tmp_path):
    p = tmp_path/"artifact"; p.write_bytes(b"one"); digest = evaluate.sha256(p)
    link = tmp_path/"link"; link.symlink_to(p)
    with pytest.raises(ValueError): evaluate.require_hash(link, digest)
    p.write_bytes(b"two")
    with pytest.raises(ValueError): evaluate.require_hash(p, digest)


@pytest.fixture
def camera_frozen(evaluate, frozen):
    root, fixed, receipt = frozen; base = root/"validation/joint_rgb_v1"
    predicted = base/"predictions_camera_v1"; predicted.mkdir()
    learned = {k: v.copy() for k, v in fixed.items()}
    learned["camera_K"] = np.array([[[f, 0., 4.], [0., f, 3.], [0., 0., 1.]] for f in [1200., 1000., 1500.]])
    # Numerical fixture, not real model accuracy: learned depths are perfect.
    learned["raw_points"] *= .8
    learned["aligned_points"] = learned["raw_points"].copy()
    learned["shared_scale"] = np.ones(3)
    candidates = [1190., 1200., 1210., 990., 1000., 1010., 1490., 1500., 1510.]
    learned_receipt = dict(receipt, stage="public_joint_rgb_learned_camera_shared_grounding_predictions",
                           camera_source="learned", focal_fitted=True, focal_candidates_pixels=candidates,
                           actual_body_inference=True, actual_MoGe_inference=True, actual_predicted_human_render=True,
                           focal_geometry_source_sha256=evaluate.FOCAL_GEOMETRY_SHA,
                           native_camera_support_verified=True, body_calls_completed=9, MoGe_calls_completed=18,
                           MoGe_camera_calls_completed=9, MoGe_fixed_camera_calls_completed=9,
                           native_focal_solver_calls=[{"file": f"clip_{(i%9)//3:02d}_frame_{i%3:03d}.png",
                               "clip_index": (i%9)//3, "frame_index": i%3,
                               "pass": "camera_candidate" if i < 9 else "shared_camera",
                               "focal_prior_supplied": i >= 9, "original_solver_returned": True,
                               "native_nearest64_valid_pixels": 42} for i in range(18)])
    # Fixed-K aligned result has 10% residual scale to create a paired gate.
    fixed["shared_scale"] = np.full(3, .88)
    fixed["aligned_points"] = fixed["raw_points"] * .88
    np.savez(base/"predictions/arrays.npz", **fixed)
    receipt["arrays_sha256"] = evaluate.sha256(base/"predictions/arrays.npz")
    write_json(base/"predictions/report.json", receipt)
    def sync():
        np.savez(predicted/"arrays.npz", **learned)
        learned_receipt["arrays_sha256"] = evaluate.sha256(predicted/"arrays.npz")
        write_json(predicted/"report.json", learned_receipt)
    sync()
    return root, fixed, receipt, learned, learned_receipt, sync


def test_actual_paired_numeric_camera_eval_all_frames_and_baseline_freeze(evaluate, camera_frozen):
    root, _, _, _, _, _ = camera_frozen; report = {}
    evaluate.run(root, report, "learned")
    assert report["paired_camera_predictions_frozen_before_private_truth_read"]
    assert len(report["frames"]) == 9 and len(report["clips"]) == 3
    assert report["camera_comparison_decision"]["synthetic_hypothesis_supported"]
    assert report["camera_comparison_decision"]["median_paired_clip_relative_gain"] == pytest.approx(1.)
    assert not report["decision"]["synthetic_hypothesis_supported"]  # Identity alpha is a separate gate.
    if not report["decision"]["zero_baseline_gain_undefined"]:
        assert report["decision"]["median_paired_clip_relative_gain"] == pytest.approx(0.)
    for frame in report["frames"]:
        assert frame["fixed_camera_baseline"]["aligned"]["visible_chamfer_half_cm"] > 19
        assert frame["raw_camera"]["aligned"]["visible_chamfer_half_cm"] < 1e-12


@pytest.mark.parametrize("fault", ["fixed_sha", "private_flag", "candidate_median", "missing_candidate", "wrong_stage",
                                   "extra_arrays", "perframe_K", "offcenter", "nonisotropic", "mask_change", "source_change",
                                   "partial_calls", "solver_fallback", "solver_order", "body_missing"])
def test_camera_both_producers_checked_before_any_private_data(evaluate, camera_frozen, monkeypatch, fault):
    root, _, fixed_receipt, learned, receipt, sync = camera_frozen; base = root/"validation/joint_rgb_v1"
    if fault == "fixed_sha":
        fixed_receipt["arrays_sha256"] = "0"*64; write_json(base/"predictions/report.json", fixed_receipt)
    elif fault == "private_flag": receipt["private_truth_read"] = True
    elif fault == "candidate_median": receipt["focal_candidates_pixels"][1] = 1300.
    elif fault == "missing_candidate": receipt["focal_candidates_pixels"] = []
    elif fault == "wrong_stage": receipt["stage"] = "public_joint_rgb_shared_grounding_predictions"
    elif fault == "extra_arrays": learned["source_GT_K"] = learned["camera_K"].copy()
    elif fault == "perframe_K": learned["camera_K"] = np.repeat(learned["camera_K"], 3, axis=0)
    elif fault == "offcenter": learned["camera_K"][0, 0, 2] += 1.
    elif fault == "nonisotropic": learned["camera_K"][0, 1, 1] += 1.
    elif fault == "mask_change": learned["object_masks"][0, 0, 0] = False
    elif fault == "source_change": receipt["public_records"][0] = dict(receipt["public_records"][0], human_mask_sha256="e"*64)
    elif fault == "partial_calls": receipt["native_focal_solver_calls"] = receipt["native_focal_solver_calls"][:-1]
    elif fault == "solver_fallback": receipt["native_focal_solver_calls"][0]["native_nearest64_valid_pixels"] = 1
    elif fault == "solver_order": receipt["native_focal_solver_calls"][0]["focal_prior_supplied"] = True
    elif fault == "body_missing": receipt["body_calls_completed"] = 8
    sync()
    original = evaluate.require_hash
    def guard(path, digest):
        assert "eval_private" not in str(path), "Private data opened before paired producer rejection"
        return original(path, digest)
    monkeypatch.setattr(evaluate, "require_hash", guard)
    with pytest.raises(ValueError): evaluate.run(root, {}, "learned")


def test_learned_K_never_accepted_as_fixed_prior(evaluate, camera_frozen):
    with pytest.raises(ValueError): evaluate.validate_predictions(camera_frozen[3], "fixed")
    evaluate.validate_predictions(camera_frozen[3], "learned")


def test_camera_eval_wrapper_mounts_both_predictions_readonly_but_not_models():
    source = (Path(__file__).parents[1]/"infra/run_joint_rgb_camera_evaluate.sh").read_text()
    assert "--gpus" not in source and "src=$ROOT/weights" not in source
    assert "src=$BASE/predictions,dst=$BASE/predictions,readonly" in source
    assert "src=$BASE/predictions_camera_v1,dst=$BASE/predictions_camera_v1,readonly" in source
    assert "--camera-source learned" in source and "quality_camera_v1" in source
    subprocess.run(["bash", "-n", str(Path(__file__).parents[1]/"infra/run_joint_rgb_camera_evaluate.sh")], check=True)
