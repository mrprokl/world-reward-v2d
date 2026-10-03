"""Private metric/firewall tests with own procedural arrays and fake rasters."""
import ast
import copy
from dataclasses import asdict
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

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "infra")); monkeypatch.syspath_prepend(str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("human_photometric_evaluate_test", REPO / "infra/human_photometric_evaluate.py")
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def pins():
    return dict(schema="world-reward-human-photometric-quality-pins-v1", **{role:dict(producer_revision=("a" if role == "render" else "b") * 40,
        report_sha256="c" * 64, report_bytes=123, script_sha256=("d" if role == "render" else "e") * 64) for role in ("render", "masks", "body")})


@pytest.mark.parametrize("fault", [None, "schema", "missing", "extra", "roleextra", "emptySHA", "rev", "boolbytes", "zero", "splitrev", "splitsha"])
def test_explicit_actual_pins_no_empty_or_fallback(gate, tmp_path, fault):
    value = pins()
    if fault == "schema": value["schema"] = "other"
    elif fault == "missing": value.pop("body")
    elif fault == "extra": value["private"] = "unneeded"
    elif fault == "roleextra": value["body"]["fallback"] = True
    elif fault == "emptySHA": value["body"]["report_sha256"] = ""
    elif fault == "rev": value["render"]["producer_revision"] = "main"
    elif fault == "boolbytes": value["body"]["report_bytes"] = True
    elif fault == "zero": value["body"]["report_bytes"] = 0
    elif fault == "splitrev": value["body"]["producer_revision"] = "f" * 40
    elif fault == "splitsha": value["body"]["script_sha256"] = "f" * 64
    path = tmp_path / "pins.json"; path.write_text(json.dumps(value)); path.chmod(0o444)
    if fault is None:
        loaded, receipt = gate.load_pins(path); assert loaded == value and receipt["bytes"] == path.stat().st_size
    else:
        with pytest.raises(ValueError): gate.load_pins(path)


def truth_fixture(gate, monkeypatch):
    monkeypatch.setattr(gate, "COHORT", SimpleNamespace(height=12, width=24, fixed_K=((1280., 0., 512.), (0., 1280., 384.), (0., 0., 1.))))
    idx = np.linspace(0., 1., 18439); v = np.c_[.3 * np.sin(idx * 37), idx * 1.5 - .75, 3. + .1 * np.cos(idx * 19)]
    faces = np.tile([0, 1, 2], (36874, 1)).astype(np.int32); monkeypatch.setattr(gate.manufacture, "FACE_SHA", gate.manufacture.array_sha(faces))
    left, right = np.zeros(18439, bool), np.zeros(18439, bool); left[:100] = True; right[-100:] = True
    rig = dict(human_faces=faces.astype(np.int64), hand_mask_left=left, hand_mask_right=right, camera_K=np.asarray(gate.COHORT.fixed_K))
    visibility = np.zeros((12, 24), bool); visibility[:4] = True
    data = dict(human_vertices_camera_m=v, human_faces=faces, human_joints_camera_m=v[:127].copy(), camera_K=rig["camera_K"].copy(),
        scene_depth_m=np.where(visibility, 3., np.nan).astype(np.float32), visible_face_indices=np.where(visibility, 0, -1).astype(np.int64),
        human_visibility=visibility, group_index=np.array(0, np.int64), frame_index=np.array(0, np.int64))
    return data, rig, dict(group_index=0, frame_index=0)


@pytest.mark.parametrize("fault", [None, "extra", "masked", "int64faces", "floatfaces", "bounds", "reordered", "K", "negativeZ", "index", "visibility", "roomdepth", "nanvisible"])
def test_truth_exact_native_i32_topology_full_geometry_and_visibility(gate, monkeypatch, fault):
    data, rig, record = truth_fixture(gate, monkeypatch)
    if fault == "extra": data["object_vertices"] = np.zeros((1, 3))
    elif fault == "masked": data["human_visibility"] = np.ma.array(data["human_visibility"], mask=False)
    elif fault == "int64faces": data["human_faces"] = data["human_faces"].astype(np.int64)
    elif fault == "floatfaces": data["human_faces"] = data["human_faces"].astype(np.float64)
    elif fault == "bounds": data["human_faces"][0, 0] = 18439
    elif fault == "reordered": data["human_faces"][0] = [0, 2, 1]
    elif fault == "K": data["camera_K"][0, 0] += 1
    elif fault == "negativeZ": data["human_joints_camera_m"][0, 2] = -.1
    elif fault == "index": data["frame_index"] = np.array(1, np.int64)
    elif fault == "visibility": data["human_visibility"][0, 0] = False
    elif fault == "roomdepth": data["scene_depth_m"][-1, -1] = 2.
    elif fault == "nanvisible": data["scene_depth_m"][0, 0] = np.nan
    if fault is None: assert gate.validate_truth(data, record, rig) is data
    else:
        with pytest.raises(ValueError): gate.validate_truth(data, record, rig)


def test_shared_first_baseline_sim3_not_each_method_or_pose_alignment(gate, monkeypatch):
    truth, rig, _ = truth_fixture(gate, monkeypatch); q = truth["human_vertices_camera_m"]
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    p = (q - [2., 1., .4]) @ R / 1.3
    sim = gate.shared_group_alignment(p, q)
    assert sim["scale"] == pytest.approx(1.3) and np.linalg.det(sim["rotation"]) == pytest.approx(1.)
    baseline = dict(vertices_camera_m=p, joints_camera_m=p[:127]); tta = dict(vertices_camera_m=p + [.1, 0., 0.], joints_camera_m=p[:127] + [.1, 0., 0.])
    before, _ = gate.frame_metrics(baseline, truth, rig, sim); after, _ = gate.frame_metrics(tta, truth, rig, sim)
    assert before["aligned_human_pve_cm"] < 1e-10 and after["aligned_human_pve_cm"] == pytest.approx(13.)
    assert after["aligned_hand_pve_cm"] == pytest.approx([13., 13.])


def test_invalid_public_predictions_do_not_open_private_or_import_torch(gate, tmp_path, monkeypatch):
    touched = []
    monkeypatch.setattr(gate, "public_predictions", lambda *_: (_ for _ in ()).throw(ValueError("public rejected")))
    monkeypatch.setattr(gate, "rendering", lambda *_: touched.append("private"))
    report = dict(private_truth_read=False)
    with pytest.raises(ValueError, match="public rejected"): gate.run(tmp_path, pins(), report, lambda: None)
    assert not touched and not report["private_truth_read"]


def test_source_producer_canonical_no_code_aliases(gate, tmp_path):
    data = pins()
    assert gate.producer_source(tmp_path, data["render"], "render") == tmp_path / "jobs" / ("a" * 40) / "run_human_photometric_prepare/code/infra/human_photometric_render.py"
    assert gate.producer_source(tmp_path, data["body"], "body") == tmp_path / "jobs" / ("b" * 40) / "run_human_photometric_observe/code/infra/human_photometric_observe.py"


def test_no_truth_or_model_from_public_consumer_source(gate):
    node = next(n for n in ast.parse(Path(gate.__file__).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "public_predictions")
    source = ast.unparse(node)
    assert "eval_private" not in source and "np.load" in source and "public.frozen_frames" in source and "public.completed_body" in source
    assert "load_model" not in source and not any(isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names) for n in ast.walk(node))
    assert "body._source_identity" in source and "body._body_assets" in source


def test_complete24_private_run_keeps_all_modes_shared_sim_and_72_rasters(gate, tmp_path, monkeypatch):
    data, rig, _ = truth_fixture(gate, monkeypatch); records = []; frames = []; rows = []; cases = []; contrasts = []
    monkeypatch.setattr(gate.manufacture, "HEIGHT", 12); monkeypatch.setattr(gate.manufacture, "WIDTH", 24)
    from PIL import Image
    automatic = np.r_[np.full(100, 255, np.uint8), np.zeros(188, np.uint8)].reshape(12, 24)
    Image.fromarray(automatic).save(tmp_path / "mask.png")
    monkeypatch.setattr(gate.public.native.joint, "HEIGHT", 12); monkeypatch.setattr(gate.public.native.joint, "WIDTH", 24)
    assert gate.public.native.joint.read_mask(tmp_path / "mask.png", Image).dtype == np.uint8
    for i in range(24):
        group, pose = divmod(i, 3); record = dict(file=f"group_{group:02d}_frame_{pose:03d}.png", group_index=group, frame_index=pose, human_mask_path=tmp_path / "mask.png", human_mask_pixels=100)
        value = {k:v.copy() for k, v in data.items()}; value["group_index"] = np.array(group, np.int64); value["frame_index"] = np.array(pose, np.int64)
        if group % 2: value["human_visibility"][4:8] = True; value["visible_face_indices"][4:8] = 0; value["scene_depth_m"][4:8] = 3.
        path = tmp_path / f"truth{i}.npz"; np.savez_compressed(path, **value); path.chmod(0o400)
        baseline = dict(vertices_camera_m=(value["human_vertices_camera_m"] + [.04, .01, .05]).astype(np.float32), joints_camera_m=(value["human_joints_camera_m"] + [.04, .01, .05]).astype(np.float32))
        frames.append(dict(baseline=baseline, sham={k:v.copy() for k,v in baseline.items()}, tta={k:v.copy() for k,v in baseline.items()}))
        records.append(record); rows.append(gate.regular(path)); geometry = dict(vertices_sha256=gate.manufacture.array_sha(value["human_vertices_camera_m"]), joints_sha256=gate.manufacture.array_sha(value["human_joints_camera_m"]))
        cases.append(dict(geometry=geometry, visible_human_pixels=int(value["human_visibility"].sum())))
        if group % 2:
            contrasts.append(dict(morphology_index=group//4, appearance_index=(group//2)%2, frame_index=pose,
                newly_visible_human_pixels=96, front_human_pixels=96, back_human_pixels=192))
    monkeypatch.setattr(gate, "public_predictions", lambda *_: (records, frames, rig, {}, {"manifest": {"sha256": "a" * 64}}, []))
    monkeypatch.setattr(gate, "rendering", lambda *_: (dict(cases=cases, occlusion_evidence=contrasts), rows, rows))
    monkeypatch.setattr(gate, "recheck", lambda rows: None)
    def raster_iou(_prediction, _rig, target):
        assert target.dtype == np.bool_ and np.count_nonzero(target) == 100
        return .8
    monkeypatch.setattr(gate, "raw_iou", raster_iou)
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True), use_deterministic_algorithms=lambda *a, **k:None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)), cudnn=SimpleNamespace(allow_tf32=True)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    # The real entrypoint requires a fresh interpreter; this fixture supplies the runtime only at its import boundary.
    original_modules = gate.sys.modules
    class Modules(dict):
        def __contains__(self, key): return False if key == "torch" else super().__contains__(key)
    monkeypatch.setattr(gate.sys, "modules", Modules(original_modules))
    report = dict(raster_attempts=0, raster_returns=0, private_truth_read=False); progress = []
    gate.run(tmp_path, pins(), report, lambda:progress.append(report.get("phase")))
    assert report["status"] == "pass" and report["frames_scored"] == 24 and report["raster_attempts"] == report["raster_returns"] == 72
    assert len(report["frame_metrics"]) == 24 and len(report["shared_group_Sim3"]) == 8 and report["decision"]["SHAM_metrics_exact"]
    assert report["decision"]["adoption_authorized"] is False and report["learned_inference_calls"] == report["optimizer_calls"] == 0


def test_wrapper_private_only_in_final_stage_with_readonly_public_assets(gate):
    path = REPO / "infra/run_human_photometric_evaluate.sh"; text = path.read_text(); subprocess.run(["bash", "-n", str(path)], check=True)
    assert subprocess.run(["bash", str(path), "--resume"], capture_output=True).returncode == 2
    assert text.count("docker run") == 1 and "--gpus all --network none --memory 32g --cpus 4" in text and "123s" in text
    assert "quality_pins.json" in text and "$BASE/eval_private" in text and "$BASE/predictions_v1" in text
    assert "src=$path,dst=$path,readonly" in text and "src=$OUT,dst=$OUT" in text and "weights/mhr" not in text
    assert "run_human_photometric_prepare" in text and "run_human_photometric_observe" in text and "src=$BASE,dst=$BASE" not in text


def test_complete_runtime_closure_keeps_sources_and_small_config_only(gate):
    spec = importlib.util.spec_from_file_location("h101_quality_bundle", REPO / "infra/azure_job.py"); launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)
    paths = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "infra", "src", "configs", "pyproject.toml"], cwd=REPO, text=True).splitlines()
    files = {p:(REPO/p).read_bytes() for p in set(paths) if (REPO/p).is_file() and not (REPO/p).is_symlink()}
    selected = launch.runtime_bundle_paths(files, "infra/run_human_photometric_evaluate.sh"); output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as target:
        for path in selected:
            member = tarfile.TarInfo(path); member.size = len(files[path]); target.addfile(member, io.BytesIO(files[path]))
    assert len(launch.encoded_runtime_archive(output.getvalue())[0]) <= launch.MAX_CODE_CONTROL_BYTES
    assert "infra/human_photometric_evaluate.py" in selected and "src/world_reward/human_photometric_metrics.py" in selected


def actual_public_fixture(gate, monkeypatch, tmp_path):
    """360 real serialized native-shaped artifacts; only learned/source assets are mocked."""
    spec = importlib.util.spec_from_file_location("h101_observer_fixture", REPO / "tests/test_human_photometric_observe.py")
    tools = importlib.util.module_from_spec(spec); spec.loader.exec_module(tools)
    base = tmp_path / gate.BASE; base.mkdir(parents=True)
    observer_out, records, rows, _ = tools.frame_fixture(gate.public, monkeypatch, base)
    out = base / "predictions_v1"; observer_out.rename(out)
    for row in rows:
        for artifact in row["artifacts"]:
            artifact["identity"] = gate.public.core.identity(out / Path(row["file"]).stem / artifact["file"])
    for record in records:
        record.update(path=tmp_path / record["file"], human_mask_path=tmp_path / (Path(record["file"]).stem + "_human.png"))
    inputs = dict(manifest=dict(sha256="a"*64, bytes=2000), mask_report=dict(sha256="b"*64, bytes=1000))
    monkeypatch.setattr(gate.public, "public_masks", lambda *_: (records, inputs))
    monkeypatch.setattr(gate.public.masks, "ASSETS", {})
    (tmp_path / "results").mkdir()
    (tmp_path / "results/image-grounding.json").write_text(json.dumps(dict(Id=gate.public.core.MASK_IMAGE)))
    (tmp_path / "results/weights-acquisition.json").write_text("{}")
    semantic = dict(source_image_id=gate.IMAGE, joint_names=["joint"+str(i) for i in range(127)])
    (tmp_path / "results/mhr-finger-semantics-v4.json").write_text(json.dumps(semantic))
    monkeypatch.setattr(gate.public.native.regions_helper, "require_semantic_report", lambda _: None)
    source = dict(source_revision="ownfixture"); assets = {"model.ckpt":dict(sha256=gate.public.native.human.BODY_SHA, bytes=gate.public.native.human.BODY_BYTES),
        "assets/mhr_model.pt":dict(sha256=gate.manufacture.MODEL_SHA, bytes=gate.manufacture.MODEL_BYTES)}
    monkeypatch.setattr(gate.public.native.human.body, "_source_identity", lambda _: source)
    monkeypatch.setattr(gate.public.native.human.body, "_body_assets", lambda _: (tmp_path / "body_fixture", assets))
    with np.load(out / "rig_metadata.npz", allow_pickle=False) as data: rig = {k:data[k] for k in data.files}
    body = dict(stage=gate.public.STAGES["body"], status="pass", phase="complete", cohort=asdict(gate.public.COHORT), frames=24,
        producer_revision="b"*40, image_id=gate.IMAGE, script_sha256=gate.sha256(Path(gate.public.__file__)), source_helpers=gate.public.helper_hashes(),
        network="none", private_truth_read=False, ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[],
        fitting_performed=False, camera_fit_performed=False, geometry_averaged=False, all_cases_retained=True, source_inputs_assets_rehashed=True,
        quality_verified=False, accuracy_verified=False, adoption_authorized=False, full_HOI_verified=False, public_inputs=inputs,
        seed=0, torch_version="2.5.1+cu124", CUDA_version="12.4", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        body_model=dict(body_revision=gate.public.native.human.body.BODY_REVISION, upstream_revision=gate.public.native.human.body.UPSTREAM_REVISION,
            dinov3_revision=gate.public.native.human.body.DINOV3_REVISION, inference_source_identity=source, body_assets=assets,
            checkpoint_loading=dict(mode="strict_network_and_head_state_with_explicit_asset_buffer_retention", parameter_tensors_loaded=1101,
                unexpected_keys=[], retained_mhr_asset_buffer_names=["buffer"+str(i) for i in range(113)])),
        semantic_report_sha256="c"*64, joint_names=semantic["joint_names"], rig_metadata=gate.public.core.identity(out / "rig_metadata.npz"),
        hand_region_sha256={side:gate.manufacture.array_sha(rig["hand_mask_"+side]) for side in ("left", "right")},
        actual_body_inference=True, all_artifacts_frozen_and_reloaded=True, SHAM_exact_replay_verified=True, original_native_replay_verified=True,
        empirical_same_process_reproducibility_only=True, hand_regions_native_verified=True, native_topology=gate.public.rig_metadata(),
        execution_policy_instrumented=True, native_operations_modified=False, native_arguments_modified=False,
        scoped_MHR_execution="strictTrue_warnFalse_JITunoptimized", native_head_method_restored=True,
        scoped_MHR_attempts=1392, scoped_MHR_returns=1392, scoped_MHR_validated=1392, deterministic_algorithms=False, warn_only=False, TF32=False,
        scoped_MHR_calls=[dict(index=i, input_guard_enabled=False, input_warn_only=False, strict_enabled=True, warn_only=False, JIT_optimized=False,
            delegated_original=True, returned=True, validated=True, restored=True, synchronized=True) for i in range(1,1393)], records=rows)
    body.update({name+suffix:count for name,count in gate.public.COUNTS.items() for suffix in ("_attempts", "_completed")})
    pin_data = pins(); pin_data["render"]["script_sha256"] = gate.sha256(Path(gate.manufacture.__file__))
    pin_data["masks"]["script_sha256"] = pin_data["body"]["script_sha256"] = body["script_sha256"]
    checked = []
    def source_identity(path, expected=None, immutable=True):
        checked.append(str(path)); return dict(path=str(path), sha256=expected or "a"*64, bytes=123)
    monkeypatch.setattr(gate, "regular", source_identity)
    monkeypatch.setattr(gate, "pinned_report", lambda path,pin: (dict(records=[{}]*24) if "automatic_masks" in str(path) else body, source_identity(path)))
    monkeypatch.setattr(gate.public, "completed_masks", lambda _: None)
    return pin_data, body, out, checked


def test_entire_public360_completion_replay_before_private_io(gate, tmp_path, monkeypatch):
    pin_data, body, out, checked = actual_public_fixture(gate, monkeypatch, tmp_path)
    records, frames, rig, producer, inputs, frozen = gate.public_predictions(tmp_path, pin_data)
    assert len(records) == len(frames) == 24 and len(list(out.rglob("*.npz"))) == 361
    assert producer is body and len(frozen) > 400 and all("eval_private" not in p for p in checked)
    assert all(frame["baseline"]["vertices_camera_m"].tobytes() == frame["sham"]["vertices_camera_m"].tobytes() for frame in frames)
    assert len(body["scoped_MHR_calls"]) == 1392 and body["scoped_MHR_validated"] == 1392


@pytest.mark.parametrize("fault", ["count", "strict", "restoration", "source", "loading", "selection"])
def test_bad_complete_native_public_contract_fails_before_private(gate, tmp_path, monkeypatch, fault):
    pin_data, body, out, checked = actual_public_fixture(gate, monkeypatch, tmp_path)
    if fault == "count": body["body_completed"] -= 1
    elif fault == "strict": body["scoped_MHR_calls"][0]["strict_enabled"] = False
    elif fault == "restoration": body["native_head_method_restored"] = False
    elif fault == "source": body["body_model"]["inference_source_identity"] = {}
    elif fault == "loading": body["body_model"]["checkpoint_loading"]["parameter_tensors_loaded"] = 1100
    else: body["records"][0]["selection"]["selected_gamma_index"] = 2
    with pytest.raises(ValueError): gate.public_predictions(tmp_path, pin_data)
    assert all("eval_private" not in p for p in checked)
