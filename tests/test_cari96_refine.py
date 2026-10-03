"""Actual pure native96 contracts; data-free fixtures, no model/GPU execution."""
import ast
import copy
from dataclasses import dataclass, asdict
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra")); monkeypatch.syspath_prepend(str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("cari96_refine_test", ROOT / "infra/cari96_refine.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Only fixture frame count is reduced; full native block/crop dimensions remain.
    monkeypatch.setattr(module, "FRAMES", 3)
    return module


@dataclass
class Config:
    num_steps: int = 300
    batch_size: int = 0
    frame_start: int = 0
    frame_limit: int = 0
    freeze_object_rotation: bool = True
    freeze_body_internal_translations: bool = True
    report_every: int = 100


def bundle(gate, mesh=Path("/own/aligned.glb")):
    n = gate.FRAMES
    params = {k: np.zeros((n, d), np.float32) for k, d in gate.inputs.PARAMETER_DIMS.items()}
    params["mhr_global_rot6d"][:] = [1, 0, 0, 1, 0, 0]
    params["mhr_trans"][:, 2] = 2
    params["pose_abs"] = np.broadcast_to(np.eye(4, dtype=np.float32), (n, 4, 4)).copy()
    params["contact_logits"] = np.ones((n, 2), np.float32)
    meta = dict(ground_truth_used=False, object_mesh=str(mesh), materialized_input_cache=None,
        object_pose_frame="centered_axis_aligned", object_pose_storage_frame="output_aligned_mesh_frame",
        object_pose_frame_revision="cari4d.object_pose_frame.centered_axis_aligned.v1",
        object_pose_storage_to_training_transform=np.eye(4).tolist(), object_mesh_to_training_transform=np.eye(4).tolist())
    obs = {k: np.ones((n, 2, 2), bool) for k in ("human_mask", "object_mask")}
    obs.update({k: np.ones((n, 256, 256), np.float32) for k in ("postopt_human_mask", "postopt_object_mask")})
    return dict(schema="cari4d.mhr_wild_inference.v1", frames=[f"{i:06d}" for i in range(n)], gt={},
        metadata=meta, pr=params, pr_initial=copy.deepcopy(params), **{"in": copy.deepcopy(params)},
        observations=obs, raw={"rot": np.zeros((n, 6), np.float32), "trans": np.zeros((n, 3), np.float32)},
        faces=np.array([[0, 1, 2]], np.int32), K_rois=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        bboxes=np.zeros((n, 4), np.float32), frame_meta=[dict(frame=f"{i:06d}", src_frame=i, kid=0) for i in range(n)])


def result(gate, source, cfg=None):
    out = copy.deepcopy(source)
    out["pr"]["mhr_body_pose_cont"][:, 0] += .02
    out["pr"]["pose_abs"][:, 0, 3] += .003
    out["pr"]["pose_abs_postopt"] = out["pr"]["pose_abs"].copy()
    out["postopt"] = dict(mode="smplh_parity", config=asdict(cfg or Config()),
        frame_indices=list(range(gate.FRAMES)), resolved_batch_size=gate.FRAMES, batch_sampling="full_clip_v1",
        optimized_parameters=gate.contract.OPTIMIZED_PARAMETERS, fixed_parameters=gate.contract.FIXED_PARAMETERS,
        history=[dict(iter=i, total_loss=.1) for i in (0, 100, 200, 300)], final_diagnostics=dict(contact=.01))
    return out


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.chmod(0o644)
    path.write_text(json.dumps(data)) if isinstance(data, dict) else path.write_bytes(data)
    path.chmod(0o444)


def producer_fixture(gate, root, base, role, report, files):
    directory = root / base
    for name, value in files.items(): write(directory / name, value)
    report = report | dict(producer_revision="a" * 40, script_sha256="b" * 64)
    write(directory / "report.json", report)
    identities = {str(p.relative_to(directory)): gate.identity(p) for p in directory.rglob("*") if p.is_file()}
    pins = dict(schema=f"world-reward-cari96-{role}-pins-v1",
        **{role: identities["report.json"] | {k: report[k] for k in ("producer_revision", "script_sha256")}, role + "_files": identities})
    return directory, pins, report


def source_fixture(gate, root):
    source_pins = json.loads((ROOT / "configs/cari96_input_pins.json").read_text())
    common = dict(status="pass", phase="complete", image_id=gate.IMAGE, frames=gate.FRAMES,
        input_track="track_1", ground_truth_used=False, private_truth_read=False,
        hand_labeled_test=False, oracle_modes=[], network="none")
    input_files = {"aligned_depth.h5": b"own encoded depth", "own_object_poses.pkl": b"own poses",
        "export/episode_000015/edex": b"own edex", "export/episode_000015/wild_export.json": b"own metadata",
        "export/episode_000015/object_mesh/output_aligned.glb": b"own GLB",
        **{f"export/episode_000015/{kind}/front_stereo_camera_left.h5": b"own encoded frames" for kind in ("images", "human_masks", "object_masks")}}
    manifest = dict(stage="world_reward_public_cari96_inputs_snapshot", status="pass", frames=gate.FRAMES,
        source_frames=501, original_frame_indices=list(range(gate.FRAMES)), source_files=source_pins["source_files"],
        source_inputs_report_sha256=source_pins["source_files"][gate.inputs.REPORT]["sha256"],
        initializer_modified=False, ground_truth_read=False, ground_truth_used=False,
        hand_labeled_test=False, oracle_modes=[])
    directory = root / gate.PREPARE
    for name, data in input_files.items(): write(directory / "inputs" / name, data)
    manifest["output_files"] = {name: gate.identity(directory / "inputs" / name) for name in input_files}
    files = {"inputs/" + name: data for name, data in input_files.items()} | {
        "inputs/manifest.json": manifest, "shared_initializer.pkl": b"own init", "target.npy": b"own target", "direct_parameters.npz": b"own controls"}
    prepare = common | dict(stage=gate.PREPARE_STAGE, shared_identity_verified=True, original_initializer_unchanged=True,
        source_inputs_assets_rehashed=True, frozen_outputs_rehashed_after_reference=True,
        reference_per_frame_mean_mm=[.001] * gate.FRAMES,
        **{k: 6 for k in ("native_geometry_calls", "native_direct_calls", "native_replay_calls", "reference_calls")})
    for name, data in files.items(): write(directory / name, data)
    prepare["output_files"] = {name: gate.identity(directory / name) for name in files}
    prepare["snapshot_manifest_sha256"] = gate.inputs.sha256(directory / "inputs/manifest.json")
    _, pp, prepare = producer_fixture(gate, root, gate.PREPARE, "prepare", prepare, files)
    fb = b"own native CoCoNet bundle"
    forward = common | dict(stage=gate.FORWARD_STAGE, prepare_report_sha256=pp["prepare"]["sha256"], prepare_files=pp["prepare_files"],
        checkpoint_sha256=gate.contract.CHECKPOINT_SHA256, bundle_sha256=__import__("hashlib").sha256(fb).hexdigest(),
        shared_identity_verified=True, raw_prediction_bytes_preserved=True, native_global_unchanged=True, hub_restored=True,
        **{k: 1 for k in ("forward_attempts", "forward_returns", "forward_validated", "composition_hook_calls",
            "composition_delegate_calls", "composition_delegate_returns", "composition_verified_calls")})
    _, fp, forward = producer_fixture(gate, root, gate.FORWARD, "forward", forward, {"coconet.pth": fb})
    return fp, pp, source_pins, forward, prepare


def test_complete_hash_bound_public_chain_without_reading_historical_data(gate, tmp_path):
    fp, pp, sp, _, _ = source_fixture(gate, tmp_path)
    forward, prepare, files = gate.source_inputs(tmp_path, fp, pp, sp)
    assert len(files) == 15 and forward["composition_delegate_returns"] == 1
    assert prepare["reference_per_frame_mean_mm"] == [.001] * 3
    assert not (tmp_path / "outputs").exists()  # No old source/private/cache dependency.


@pytest.mark.parametrize("fault", ["missing", "extra", "traversal", "absolute", "boolbytes", "hash", "revision", "source", "schema", "writable", "symlink", "tamper"])
def test_pinned_producer_inventory_fail_closed(gate, tmp_path, fault):
    _, pp, *_ = source_fixture(gate, tmp_path)
    path = tmp_path / gate.PREPARE / "target.npy"
    if fault == "missing": pp["prepare_files"].pop("target.npy")
    elif fault == "extra": write(tmp_path / gate.PREPARE / "cache.pth", b"unexpected")
    elif fault in ("traversal", "absolute"): pp["prepare_files"]["../private" if fault == "traversal" else "/private"] = pp["prepare_files"]["target.npy"]
    elif fault == "boolbytes": pp["prepare_files"]["target.npy"]["bytes"] = True
    elif fault == "hash": pp["prepare_files"]["target.npy"]["sha256"] = "x" * 64
    elif fault == "revision": pp["prepare"]["producer_revision"] = "a" * 39
    elif fault == "source": pp["prepare"]["script_sha256"] = "c" * 64
    elif fault == "schema": pp["schema"] = "another-stage"
    elif fault == "writable": path.chmod(0o644)
    elif fault == "symlink": path.unlink(); path.symlink_to(path.with_name("direct_parameters.npz"))
    else: write(path, b"changed")
    with pytest.raises(ValueError): gate.pinned_report(tmp_path, gate.PREPARE, pp, "prepare")


@pytest.mark.parametrize("fault", ["GT", "timeline", "checkpoint", "delegate", "boolcount", "snapshotfiles", "reference", "snapshotGT"])
def test_lineage_and_every_frame_guards_before_optimizer(gate, tmp_path, fault):
    fp, pp, sp, fr, pr = source_fixture(gate, tmp_path)
    path = tmp_path / gate.FORWARD / "report.json"
    if fault in ("snapshotfiles", "snapshotGT"):
        mp = tmp_path / gate.PREPARE / "inputs/manifest.json"; m = json.loads(mp.read_text())
        if fault == "snapshotfiles": m["output_files"].pop("aligned_depth.h5")
        else: m["ground_truth_read"] = True
        write(mp, m); pp["prepare_files"]["inputs/manifest.json"] = gate.identity(mp)
        pr["output_files"]["inputs/manifest.json"] = gate.identity(mp); pr["snapshot_manifest_sha256"] = gate.inputs.sha256(mp)
    elif fault == "reference": pr["reference_per_frame_mean_mm"][-1] = 2.00001
    elif fault == "GT": fr["ground_truth_used"] = True
    elif fault == "timeline": fr["frames"] = 2
    elif fault == "checkpoint": fr["checkpoint_sha256"] = "c" * 64
    elif fault == "delegate": fr["composition_verified_calls"] = 0
    else: fr["forward_attempts"] = True
    rp = tmp_path / gate.PREPARE / "report.json"; write(rp, pr)
    pp["prepare_files"]["report.json"] = gate.identity(rp); pp["prepare"].update(gate.identity(rp))
    fr["prepare_files"] = pp["prepare_files"]; fr["prepare_report_sha256"] = pp["prepare"]["sha256"]
    write(path, fr); fp["forward_files"]["report.json"] = gate.identity(path); fp["forward"].update(gate.identity(path))
    with pytest.raises(ValueError): gate.source_inputs(tmp_path, fp, pp, sp)


def test_native_observations_retain_fully_occluded_frames(gate):
    source = bundle(gate)
    for a in source["observations"].values(): a[1] = 0
    params, pose = gate.validate_source_bundle(source, Path("/own/aligned.glb"))
    assert pose.shape == (3, 4, 4) and len(params) == 7
    metadata = gate.validate_result(source, result(gate, source))
    assert metadata["effective_optimizer_updates"] == 301
    assert metadata["fixed_parameters"] == gate.contract.FIXED_PARAMETERS


@pytest.mark.parametrize("fault", ["GT", "timeline", "dtype", "shape", "scale", "expression", "mesh", "cache", "maskmissing", "maskdtype", "masknan", "postgrid", "contact"])
def test_native_source_no_repair_no_frame_drop(gate, fault):
    source = bundle(gate)
    if fault == "GT": source["gt"] = {"labels": 1}
    elif fault == "timeline": source["frames"][-1] = "000000"
    elif fault == "dtype": source["pr"]["mhr_trans"] = source["pr"]["mhr_trans"].astype(np.float64)
    elif fault in ("shape", "scale"): source["pr"]["mhr_" + fault][1, 0] = .01
    elif fault == "expression": source["pr"]["mhr_face"][0, 0] = .01
    elif fault == "mesh": source["metadata"]["object_mesh"] = "/different.glb"
    elif fault == "cache": source["metadata"]["materialized_input_cache"] = "old-cache"
    elif fault == "maskmissing": source["observations"].pop("human_mask")
    elif fault == "maskdtype": source["observations"]["human_mask"] = source["observations"]["human_mask"].astype(np.float32)
    elif fault == "masknan": source["observations"]["postopt_object_mask"][0, 0, 0] = np.nan
    elif fault == "postgrid": source["observations"]["postopt_human_mask"] = np.ones((3, 2, 2), np.float32)
    else: source["pr"]["contact_logits"] = np.ones((3, 1), np.float32)
    with pytest.raises(ValueError): gate.validate_source_bundle(source, Path("/own/aligned.glb"))


@pytest.mark.parametrize("fault", ["root", "trans", "hand", "shape", "scale", "face", "internal", "objectR", "mesh", "raw", "camera", "mask", "topology", "frames", "updates", "partial", "diag", "contact", "initial", "pose_copy", "extra"])
def test_native_result_only_original_two_parameter_groups(gate, fault):
    source = bundle(gate); out = result(gate, source)
    if fault in ("root", "trans", "hand", "shape", "scale", "face"):
        key = "mhr_global_rot6d" if fault == "root" else "mhr_" + fault; out["pr"][key][0, 0] += .01
    elif fault == "internal": out["pr"]["mhr_body_pose_cont"][0, 254] = .01
    elif fault == "objectR": out["pr"]["pose_abs"][0, :3, :3] = np.diag([-1, -1, 1])
    elif fault == "mesh": out["metadata"]["object_mesh"] = "another"
    elif fault == "raw": out["raw"]["rot"][0, 0] = .01
    elif fault == "camera": out["K_rois"][0, 0, 0] = 2
    elif fault == "mask": out["observations"]["human_mask"][1, 0, 0] = False
    elif fault == "topology": out["faces"][0, 0] = 1
    elif fault == "contact": out["pr"]["contact_logits"][0, 0] += .1
    elif fault == "initial": out["pr_initial"]["mhr_shape"][0, 0] += .1
    elif fault == "pose_copy": out["pr"]["pose_abs_postopt"][0, 0, 3] += .1
    elif fault == "extra": out["pr"]["unexpected_prediction"] = np.zeros(3, np.float32)
    elif fault == "frames": out["postopt"]["frame_indices"] = [0, 1]
    elif fault == "updates": out["postopt"]["config"]["num_steps"] = 299
    elif fault == "partial": out["postopt"]["history"][-1]["iter"] = 299
    else: out["postopt"]["final_diagnostics"]["contact"] = np.nan
    with pytest.raises(ValueError): gate.validate_result(source, out)


@pytest.mark.parametrize("outcome", ["success", "exception", "mutated", "config"])
def test_actual_attempt_return_validated_accounting_and_config(gate, outcome):
    source = bundle(gate); report = dict(optimizer_attempts=0, optimizer_returns=0, optimizer_validated=0); seen = []
    def callback(s, v, f, cfg, *, mhr_layer):
        assert report == dict(optimizer_attempts=1, optimizer_returns=0, optimizer_validated=0)
        assert mhr_layer == "native layer"
        if outcome == "exception": raise RuntimeError("native failure")
        out = result(gate, s, cfg)
        if outcome == "mutated": s["raw"]["rot"][0, 0] = .1
        if outcome == "config": out["postopt"]["config"]["report_every"] = 50
        return out
    call = lambda: gate.invoke_optimizer(source, np.zeros((3, 3)), np.array([[0, 1, 2]]), Config(),
        "native layer", callback, report, lambda: seen.append(report.copy()))
    if outcome == "success":
        call(); assert report["optimizer_validated"] == 1
        assert report["metadata"]["effective_optimizer_updates"] == 301
    else:
        with pytest.raises(RuntimeError if outcome == "exception" else ValueError): call()
        assert report["optimizer_validated"] == 0
    assert report["optimizer_attempts"] == 1 and report["optimizer_returns"] == (outcome != "exception")
    assert seen[0]["optimizer_attempts"] == 1 and seen[0]["optimizer_returns"] == 0


def test_complete_fingerprint_checks_all_array_bytes_and_masked_arrays(gate):
    a = bundle(gate); b = copy.deepcopy(a)
    assert gate.fingerprint(a) == gate.fingerprint(b)
    b["raw"]["rot"][0, 0] = -0.0
    assert gate.fingerprint(a) != gate.fingerprint(b)
    with pytest.raises(ValueError): gate.fingerprint(np.ma.array([1], mask=[False]))


def completed_report(gate):
    report = dict(stage=gate.STAGE, status="pass", phase="complete", image_id=gate.IMAGE, frames=gate.FRAMES,
        input_track="track_1", ground_truth_used=False, private_truth_read=False, hand_labeled_test=False,
        oracle_modes=[], network="none", ground_truth_read=False, optimizer_attempts=1, optimizer_returns=1,
        optimizer_validated=1, requested_steps=300, effective_optimizer_updates=301, source_frames=501,
        original_frame_indices=list(range(gate.FRAMES)), source_inputs_assets_rehashed=True,
        saved_bundle_reloaded_verified=True, frozen_raw_inputs_byte_preserved=True, object_mesh_unchanged=True,
        native_refinement_verified=True, learned_inference_calls=0, quality_verified=False, adoption_authorized=False,
        submission_produced=False, numerical_bit_determinism_claimed=False, optimizer_sha256=gate.contract.OPTIMIZER_SHA256,
        refinement_assets=gate.contract.REFINEMENT_ASSETS, bundle_bytes=100,
        metadata=dict(frozen_parameters_bit_identical=True, effective_optimizer_updates=301,
            optimized_parameters=gate.contract.OPTIMIZED_PARAMETERS, fixed_parameters=gate.contract.FIXED_PARAMETERS))
    report.update({k: "c" * 64 for k in ("forward_report_sha256", "prepare_report_sha256", "source_bundle_sha256", "bundle_sha256", "object_mesh_sha256")})
    report["output_files"] = {"refined.pth": {"sha256": report["bundle_sha256"], "bytes": 100}}
    return report


def test_completed_report_is_execution_only_not_accuracy_or_full501(gate, monkeypatch):
    monkeypatch.setattr(gate, "FRAMES", 96)
    report = completed_report(gate); gate.validate_refinement_report(report)
    assert len(report["original_frame_indices"]) == 96 and report["source_frames"] == 501
    assert report["quality_verified"] is False and report["adoption_authorized"] is False


@pytest.mark.parametrize("fault", ["updates", "attempts", "validated", "boolcount", "coverage", "GT", "quality", "sha", "extra", "identity", "mesh"])
def test_completed_consumer_requires_actual_protocol_and_frozen_chain(gate, fault):
    r = completed_report(gate)
    if fault == "updates": r["effective_optimizer_updates"] = 300
    elif fault == "attempts": r["optimizer_attempts"] = 2
    elif fault == "validated": r["optimizer_validated"] = 0
    elif fault == "boolcount": r["optimizer_returns"] = True
    elif fault == "coverage": r["original_frame_indices"][-1] = 1
    elif fault == "GT": r["ground_truth_read"] = True
    elif fault == "quality": r["quality_verified"] = True
    elif fault == "sha": r["bundle_sha256"] = "unknown"
    elif fault == "extra": r["output_files"]["old_cache.pth"] = r["output_files"]["refined.pth"]
    elif fault == "identity": r["metadata"]["frozen_parameters_bit_identical"] = False
    else: r["object_mesh_unchanged"] = False
    with pytest.raises(ValueError): gate.validate_refinement_report(r)


def test_wrapper_shell_syntax_and_no_private_or_historical_data_mounts():
    shell = ROOT / "infra/run_cari96_refine.sh"
    subprocess.run(["bash", "-n", str(shell)], check=True)
    text = shell.read_text()
    assert '--network none' in text and '1203s docker run' in text and '--memory 64g --cpus 4' in text
    assert 'validation/cari96_forward_v1' in text and 'validation/cari96_public_v1' in text
    assert '(( $# == 0 )) || exit 2' in text and 'Never overwrite complete or incomplete refinement' in text
    assert 'eval_private' not in text and 'outputs/episode_' not in text and '--privileged' not in text
    tree = ast.parse((ROOT / "infra/cari96_refine.py").read_text())
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and ('torch' in ast.unparse(n)) for n in tree.body)
