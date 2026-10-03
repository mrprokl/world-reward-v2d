"""Data-free public/native308 contracts; no local model, fitting or rendering."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

INFRA = Path(__file__).parents[1]/"infra"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA)); monkeypatch.syspath_prepend(str(INFRA.parent/"src"))
    spec = importlib.util.spec_from_file_location("keypoint_baseline_test", INFRA/"keypoint_rgb_baseline.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def public_fixture(gate, root, monkeypatch):
    base = root/gate.BASE; inputs, masks = base/"inputs", base/"automatic_masks"; inputs.mkdir(parents=True); masks.mkdir()
    images, rows = [], []
    for i in range(15):
        clip, frame = divmod(i, 5); name = f"clip_{clip:02d}_frame_{frame:03d}.png"
        rgb = inputs/name; rgb.write_bytes(f"own-fresh-RGB-{i}".encode()); rgb.chmod(0o444)
        images.append(dict(file=name, sha256=gate.sha256(rgb), width=1024, height=768))
        row = dict(file=name, rgb_sha256=gate.sha256(rgb), clip_index=clip, frame_index=frame)
        for label, query in gate.masks.QUERIES:
            path = masks/(Path(name).stem+"_"+label+".png"); path.write_bytes(f"own-automatic-{i}-{label}".encode()); path.chmod(0o444)
            row.update({label+"_query": query, label+"_mask_file": path.name, label+"_mask_sha256": gate.sha256(path),
                        label+"_mask_bytes": path.stat().st_size, label+"_mask_pixels": 100})
        rows.append(row)
    manifest = dict(schema=gate.masks.SCHEMA, images=images); manifest_path = inputs/"manifest.json"
    manifest_path.write_text(json.dumps(manifest)); assert manifest_path.stat().st_size == 2199
    monkeypatch.setattr(gate, "MANIFEST_SHA", gate.sha256(manifest_path)); manifest_path.chmod(0o444)
    report = dict(stage=gate.masks.STAGE, status="pass", phase="complete", frames=15, network="none", private_truth_read=False,
        ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[],
        actual_automatic_inference_verified=True, all_cases_retained=True, human_query="person.", object_query="bottle.",
        input_manifest_sha256=gate.MANIFEST_SHA, input_manifest_bytes=2199, producer_revision=gate.MASK_REVISION,
        script_sha256=gate.MASK_SCRIPT_SHA, shared_mask_helper_sha256=gate.sha256(Path(gate.masks.masks.__file__)),
        detector_revision=gate.masks.masks.DETECTOR_REVISION, sam2_weights_revision=gate.masks.masks.SAM2_REVISION,
        actual_detector_calls=30, actual_sam2_calls=30, actual_sam2_image_encoder_calls=15, records=rows,
        model_assets={n: {"path": str(root/"weights"/n), "sha256": p[0], "bytes": p[1]} for n, p in gate.masks.masks.ASSETS.items()})
    def save():
        path = masks/"report.json"; path.chmod(0o644) if path.exists() else None
        path.write_text(json.dumps(report)); monkeypatch.setattr(gate, "MASK_SHA", gate.sha256(path)); path.chmod(0o444)
    save(); return report, save, base


def test_full_fresh_public_before_any_native_inputs_or_private(gate, tmp_path, monkeypatch):
    _, _, base = public_fixture(gate, tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}
    records, receipt = gate.public_inputs(tmp_path)
    assert len(records) == 15 and receipt["public_inputs_bytes"] == 2199 and all("human_mask_path" in r for r in records)
    assert gate.native.BASE == "validation/identity_rgb_v2" and gate.BASE != gate.native.BASE
    assert before == {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["private", "stage", "producer", "script", "RGB", "masksha", "maskbytes", "count", "assets", "extra", "writable"])
def test_mask_integrity_and_fresh_producer_fail_closed(gate, tmp_path, monkeypatch, fault):
    report, save, base = public_fixture(gate, tmp_path, monkeypatch)
    if fault == "private": report["private_truth_read"] = True
    elif fault == "stage": report["stage"] = gate.native.masks.STAGE
    elif fault == "producer": report["producer_revision"] = "0"*40
    elif fault == "script": report["script_sha256"] = "0"*64
    elif fault == "RGB": report["records"][0]["rgb_sha256"] = "0"*64
    elif fault == "masksha": report["records"][0]["human_mask_sha256"] = "0"*64
    elif fault == "maskbytes": report["records"][0]["human_mask_bytes"] += 1
    elif fault == "count": report["actual_sam2_calls"] = 29
    elif fault == "assets": report["model_assets"] = {}
    elif fault == "extra": (base/"automatic_masks/extra.npz").write_bytes(b"unapproved")
    save()
    if fault == "writable": (base/"inputs/clip_00_frame_000.png").chmod(0o644)
    with pytest.raises(ValueError): gate.public_inputs(tmp_path)


def raw_fixture(gate, monkeypatch, clip=0, frame=0):
    monkeypatch.setattr(gate.native, "WIDTH", 4); monkeypatch.setattr(gate.native, "HEIGHT", 3)
    monkeypatch.setattr(gate.native, "VERTICES", 100); monkeypatch.setattr(gate.native, "JOINTS", 127)
    y, x = np.indices((3, 4)); z = np.full((3, 4), 2., np.float32)
    points = np.stack(((x+.5-512)/1280*z, (y+.5-384)/1280*z, z), -1).astype(np.float32)
    data = {k: np.zeros(s, np.float32) for k, s in gate.native.BLOCKS.items()}; data["pred_cam_t"][:] = [0, 0, 2]
    data.update(raw_depth=z, raw_points=points, validity=np.ones((3, 4), bool), rendered_depth=z.copy(), silhouette=np.ones((3, 4), bool),
        human_mask=np.ones((3, 4), bool), object_mask=np.ones((3, 4), bool), raw_vertices_camera_m=np.ones((100, 3), np.float32),
        raw_joints_camera_m=np.ones((127, 3), np.float32), raw_keypoints_camera_m=np.ones((308, 3), np.float32),
        raw_joint_global_rotations=np.tile(np.eye(3, dtype=np.float32), (127, 1, 1)), human_faces=np.zeros((36874, 3), np.int64),
        camera_K=gate.CAMERA_K.copy(), clip_index=np.array(clip, np.int64), frame_index=np.array(frame, np.int64))
    record = dict(file=f"clip_{clip:02d}_frame_{frame:03d}.png", clip_index=clip, frame_index=frame, sha256="a"*64)
    return data, record


def pair_fixture(raw):
    left = np.arange(100) < 50
    data = dict(human_faces=raw["human_faces"].copy(), hand_mask_left=left, hand_mask_right=~left, camera_K=raw["camera_K"].copy(),
        pred_cam_t=raw["pred_cam_t"].copy(), raw_model_controls=raw["mhr_model_params"].copy(), shared_model_controls=raw["mhr_model_params"].copy(),
        raw_shape_params=raw["shape_params"].copy(), shared_shape_params=raw["shape_params"].copy(), raw_scale_params=raw["scale_params"].copy(),
        shared_scale_params=raw["scale_params"].copy(), expression=raw["expr_params"].copy(), clip_index=raw["clip_index"].copy(), frame_index=raw["frame_index"].copy())
    for mode in ("raw", "shared"):
        for suffix in ("vertices_camera_m", "joints_camera_m", "keypoints_camera_m", "joint_global_rotations"):
            data[mode+"_"+suffix] = raw["raw_"+suffix].copy()
    return data


def test_extended_validators_use_real_old_contract_without_global_mutation(gate, monkeypatch):
    raw, record = raw_fixture(gate, monkeypatch); pair = pair_fixture(raw); before = {k: v.tobytes() for k, v in raw.items()}
    assert gate.validate_raw(raw, record) is raw and gate.validate_pair(pair, record, raw) is pair
    assert {k: v.tobytes() for k, v in raw.items()} == before
    assert "raw_keypoints_camera_m" not in gate.native.RAW_KEYS and len(raw["raw_keypoints_camera_m"]) != len(raw["raw_joints_camera_m"])


@pytest.mark.parametrize("fault", ["70keypoints", "nan", "dtype", "masked", "improper", "rawKP", "rawRot", "pose", "root", "translation", "extra"])
def test_extended_geometry_keypoints_rotation_and_frozen_source_contract(gate, monkeypatch, fault):
    raw, record = raw_fixture(gate, monkeypatch); pair = pair_fixture(raw)
    if fault == "70keypoints": pair["shared_keypoints_camera_m"] = np.zeros((70, 3), np.float32)
    elif fault == "nan": pair["shared_keypoints_camera_m"][0, 0] = np.nan
    elif fault == "dtype": pair["shared_keypoints_camera_m"] = pair["shared_keypoints_camera_m"].astype(np.float64)
    elif fault == "masked": pair["shared_joint_global_rotations"] = np.ma.array(pair["shared_joint_global_rotations"], mask=False)
    elif fault == "improper": pair["shared_joint_global_rotations"][0, 0, 0] = -1
    elif fault == "rawKP": pair["raw_keypoints_camera_m"][0, 0] += 1
    elif fault == "rawRot": pair["raw_joint_global_rotations"][0] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    elif fault == "pose": pair["shared_model_controls"][10] += .1
    elif fault == "root": pair["shared_model_controls"][0] = .1
    elif fault == "translation": pair["pred_cam_t"][0] += 1
    elif fault == "extra": pair["private_GT"] = np.zeros(1)
    with pytest.raises(ValueError): gate.validate_pair(pair, record, raw)


@pytest.mark.parametrize("fault", [None, "shape", "PCA28", "scale68", "signedzero", "missing"])
def test_first_frame_identity_and68_are_byteconstant_not_averaged(gate, monkeypatch, fault):
    raw_frames, pairs = [], []
    for i in range(15):
        raw, _ = raw_fixture(gate, monkeypatch, i//5, i%5); raw["shape_params"][:] = i//5; raw["scale_params"][:] = i//5
        pair = pair_fixture(raw); raw_frames.append(raw); pairs.append(pair)
    if fault == "shape": pairs[3]["shared_shape_params"][0] += .1
    elif fault == "PCA28": pairs[8]["shared_scale_params"][0] += .1
    elif fault == "scale68": pairs[13]["shared_model_controls"][150] = .1
    elif fault == "signedzero": pairs[2]["shared_model_controls"][150] = np.float32(-0.)
    elif fault == "missing": pairs.pop()
    if fault is None: gate.validate_clip_constants(pairs, raw_frames)
    else:
        with pytest.raises(ValueError): gate.validate_clip_constants(pairs, raw_frames)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    @property
    def shape(self): return self.value.shape
    def __getitem__(self, key): return Tensor(self.value[key])
    def __mul__(self, other): return Tensor(self.value * other)
    def detach(self): return self
    def float(self): return Tensor(self.value.astype(np.float32))
    def cpu(self): return self
    def numpy(self): return self.value


def native_values():
    return tuple(Tensor(v) for v in (np.full((1, 18439, 3), .5, np.float32), np.full((1, 308, 3), .6, np.float32),
        np.full((1, 127, 3), .7, np.float32), np.zeros((1, 204), np.float32), np.tile(np.eye(3, dtype=np.float32), (1, 127, 1, 1))))


def fake_torch():
    return SimpleNamespace(inference_mode=nullcontext, as_tensor=lambda v, **kw: Tensor(v), isfinite=lambda v: np.isfinite(v.value))


def test_shared_head_all_returns_exact_params_flip_translation_once(gate):
    raw = {k: np.full(s, .2, np.float32) for k, s in gate.native.BLOCKS.items()}; raw["pred_cam_t"] = np.array([1., 2., 3.], np.float32)
    first = {"shape_params": np.full(45, .9, np.float32), "scale_params": np.full(28, -.3, np.float32)}; calls = []
    def forward(**kwargs): calls.append(kwargs); return native_values()
    v, kp, j, c, r = gate.shared_decode(fake_torch(), SimpleNamespace(mhr_forward=forward), raw, first)
    assert np.array_equal(v[0], [1.5, 1.5, 2.5]) and np.allclose(kp[0], [1.6, 1.4, 2.4]) and np.allclose(j[0], [1.7, 1.3, 2.3])
    assert c.shape == (204,) and r.shape == (127, 3, 3) and len(calls) == 1
    for flag in ("return_keypoints", "return_joint_coords", "return_model_params", "return_joint_rotations"): assert calls[0][flag] is True
    assert not calls[0]["global_trans"].value.any() and not calls[0]["expr_params"].value.any()
    assert np.array_equal(calls[0]["shape_params"].value[0], first["shape_params"]) and np.array_equal(calls[0]["scale_params"].value[0], first["scale_params"])


def test_extra_raw_keypoint_head_uses_original_prediction_and_parity(gate, monkeypatch):
    outputs = native_values(); seen = []
    monkeypatch.setattr(gate.native.human.body, "_native_forward_from_blocks", lambda head, prediction: seen.append(prediction) or outputs)
    prediction = {"own": "original"}; t = np.array([1, 2, 3], np.float32); flip = np.array([1, -1, -1], np.float32)
    values = {"pred_cam_t": t, "vertices_camera_m": outputs[0].value[0]*flip+t,
              "joints_camera_m": outputs[2].value[0]*flip+t, "mhr_model_params": outputs[3].value[0]}
    kp, r = gate.raw_keypoint_decode(fake_torch(), None, prediction, values)
    assert seen == [prediction] and kp.shape == (308, 3) and r.shape == (127, 3, 3)
    values["vertices_camera_m"][0, 0] += .01
    with pytest.raises(ValueError, match="changed native"): gate.raw_keypoint_decode(fake_torch(), None, prediction, values)


def test_real_frozen_all15_raw_and15_pairs_before_reference(gate, monkeypatch, tmp_path):
    (tmp_path/"raw").mkdir(); (tmp_path/"paired").mkdir(); rows, records, raw_frames = [], [], []
    for i in range(15):
        raw, record = raw_fixture(gate, monkeypatch, i//5, i%5)
        gate.validate_raw(raw, record); rows.append(gate.native.save(tmp_path, "raw", record, raw)); records.append(record)
    raw_frames = gate.native.frozen_rows(tmp_path, rows, records, "raw", gate.validate_raw)
    paired_rows = []
    for raw, record in zip(raw_frames, records): paired_rows.append(gate.native.save(tmp_path, "paired", record, pair_fixture(raw)))
    pairs = gate.native.frozen_rows(tmp_path, paired_rows, records, "paired", gate.validate_pair)
    gate.validate_clip_constants(pairs, raw_frames)
    assert len(raw_frames) == len(pairs) == 15
    target = tmp_path/rows[0]["artifact"]; target.chmod(0o644); target.write_bytes(b"tampered")
    with pytest.raises(ValueError): gate.native.frozen_rows(tmp_path, rows, records, "raw", gate.validate_raw)


def test_official_reference_full15_no_perframe_alignment(gate, monkeypatch):
    monkeypatch.setattr(gate.native, "VERTICES", 2); monkeypatch.setattr(gate.native, "JOINTS", 127)
    v = np.zeros((15, 2, 3), np.float64); j = np.zeros((15, 127, 3), np.float64); t = np.tile([0, 0, 2], (15, 1)).astype(np.float32)
    shared_v = np.broadcast_to(t[:, None], (15, 2, 3)).copy(); shared_j = np.broadcast_to(t[:, None], (15, 127, 3)).copy()
    result = gate.native.reference_errors(v, j, shared_v, shared_j, t)
    assert result["per_frame_mean_mm"] == [0.] * 15
    v[14, :, 0] = 2.01
    with pytest.raises(ValueError, match="exceeds2mm"): gate.native.reference_errors(v, j, shared_v, shared_j, t)


def test_actual_cpu_native_producer_bindings_with_tiny_asset_files(gate, monkeypatch, tmp_path):
    """Only external package/rig readers mocked; real report equality/hash gates."""
    n = gate.native; checkpoint = {"model.ckpt": {"sha256": n.human.BODY_SHA, "bytes": n.human.BODY_BYTES}}
    source = {"python_files": 46, "sha256": "a"*64}
    monkeypatch.setattr(n.human.body, "_source_identity", lambda root: source)
    monkeypatch.setattr(n.human.body, "_body_assets", lambda root: (root/"weights/body", checkpoint))
    receipt, asset = {"sha256": "b"*64}, {"sha256": "c"*64, "bytes": 10}
    monkeypatch.setattr(n.depth_model, "model_asset", lambda root: (root/"weights/moge", receipt, asset))
    directory = tmp_path/"installed_moge"; (directory/"utils").mkdir(parents=True)
    geometry = directory/"utils/geometry_torch.py"; geometry.write_text("# own source fixture\n")
    monkeypatch.setattr(n.joint, "GEOMETRY_SHA", gate.sha256(geometry))
    moge_source = {"revision": "d"*40, "sha256": "e"*64}
    monkeypatch.setattr(n.depth_model, "installed_source", lambda path, direct: moge_source)
    monkeypatch.setattr(gate.metadata, "distribution", lambda name: SimpleNamespace(read_text=lambda file: None))
    monkeypatch.setattr(gate.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(directory)]))
    model = tmp_path/"weights/mhr/mhr_model.pt"; model.parent.mkdir(parents=True); model.write_bytes(b"own reference")
    tool = tmp_path/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"; tool.parent.mkdir(parents=True); tool.write_text("# own reference tool")
    monkeypatch.setattr(n, "REFERENCE_MODEL_SHA256", gate.sha256(model)); monkeypatch.setattr(n, "CONVERTER_SHA256", gate.sha256(tool))
    semantic_path = tmp_path/"results/mhr-finger-semantics-v4.json"; semantic_path.parent.mkdir()
    semantic = {"joint_names": [f"joint_{i}" for i in range(127)], "source_image_id": n.IMAGE_ID}; semantic_path.write_text(json.dumps(semantic))
    monkeypatch.setattr(n.regions_helper, "require_semantic_report", lambda data: None)
    report = {"script_sha256": gate.sha256(Path(gate.__file__)), "helper_source_sha256": gate.helper_identities(),
        "model_sha256": n.REFERENCE_MODEL_SHA256, "converter_sha256": n.CONVERTER_SHA256,
        "body_model": {"body_revision": n.human.body.BODY_REVISION, "upstream_revision": n.human.body.UPSTREAM_REVISION,
            "dinov3_revision": n.human.body.DINOV3_REVISION, "body_assets": checkpoint, "inference_source_identity": source,
            "checkpoint_loading": {"mode": "strict_network_and_head_state_with_explicit_asset_buffer_retention", "unexpected_keys": [],
                "retained_mhr_asset_buffer_names": [f"buffer_{i}" for i in range(113)], "parameter_tensors_loaded": 1101}},
        "acquisition_report": receipt, "MoGe_model_asset": asset, "MoGe_source": moge_source,
        "semantic_report_sha256": gate.sha256(semantic_path), "joint_names": semantic["joint_names"], "image_id": n.IMAGE_ID}
    assert gate.validate_producer_bindings(tmp_path, report) is True
    for fault in ("script", "helper", "retained", "checkpoint", "source", "MoGe", "image", "joints", "tool"):
        import copy
        bad = copy.deepcopy(report)
        if fault == "script": bad["script_sha256"] = "0"*64
        elif fault == "helper": bad["helper_source_sha256"] = {}
        elif fault == "retained": bad["body_model"]["checkpoint_loading"]["retained_mhr_asset_buffer_names"][1] = "buffer_0"
        elif fault == "checkpoint": bad["body_model"]["body_assets"]["model.ckpt"]["sha256"] = "0"*64
        elif fault == "source": bad["body_model"]["inference_source_identity"] = {}
        elif fault == "MoGe": bad["MoGe_source"] = {}
        elif fault == "image": bad["image_id"] = "sha256:"+"0"*64
        elif fault == "joints": bad["joint_names"] = ["wrong"]
        elif fault == "tool": bad["converter_sha256"] = "0"*64
        with pytest.raises(ValueError): gate.validate_producer_bindings(tmp_path, bad)


def test_source_closure_under100kb_and_no_private_evaluator(gate):
    import base64
    import io
    import lzma
    import tarfile
    import azure_job
    root = INFRA.parent
    files = {str(p.relative_to(root)): p.read_bytes() for base in (root/"infra", root/"src", root/"configs") for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"] = (root/"pyproject.toml").read_bytes()
    selected = azure_job.runtime_bundle_paths(files, "infra/run_keypoint_rgb_baseline.sh")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in selected:
            info = tarfile.TarInfo(path); info.size = len(files[path]); archive.addfile(info, io.BytesIO(files[path]))
    assert len(base64.b64encode(lzma.compress(buffer.getvalue(), preset=6))) < 100000
    assert "infra/keypoint_rgb_baseline.py" in selected and "infra/identity_rgb_evaluate.py" not in selected
    assert "infra/keypoint_rgb_render.py" not in selected and "infra/dwpose_smoke.py" not in selected


def test_cpu_wrapper_only_public_inputs_no_quality_or_dwpose(gate):
    wrapper = INFRA/"run_keypoint_rgb_baseline.sh"; subprocess.run(["bash", "-n", str(wrapper)], check=True)
    assert subprocess.run(["bash", str(wrapper), "--GT"], capture_output=True).returncode == 2
    text = wrapper.read_text(); assert "eval_private" not in text and "quality" not in text and "dwpose" not in text.lower()
    assert "600" not in text or "603s" in text
    assert text.count("docker run") == 1 and "--gpus all --memory 32g" in text and "--network none" in text
    assert "src=$BASE/inputs,dst=$BASE/inputs,readonly" in text and "src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly" in text
    assert 'src=$BASE,dst=$BASE' not in text and native_image(gate) in text


def native_image(gate): return gate.native.IMAGE_ID
