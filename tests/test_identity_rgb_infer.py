"""Public firewall/shared-identity ABI tests, without models, images or CUDA."""
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec = importlib.util.spec_from_file_location("own_identity_infer", infra/"identity_rgb_infer.py")
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


@pytest.fixture
def public(module, tmp_path):
    base = tmp_path/module.BASE; inputs = base/"inputs"; masks = base/"automatic_masks"
    inputs.mkdir(parents=True); masks.mkdir(); images = []; rows = []
    for i in range(15):
        clip, frame = divmod(i, 5); name = f"clip_{clip:02d}_frame_{frame:03d}.png"
        rgb = inputs/name; rgb.write_bytes(f"public_rgb{i}".encode())
        digest = module.sha256(rgb); images.append(dict(file=name, sha256=digest, width=1024, height=768))
        row = dict(file=name, rgb_sha256=digest, clip_index=clip, frame_index=frame)
        for label, query in module.masks.QUERIES:
            mask = masks/(Path(name).stem+"_"+label+".png"); mask.write_bytes(f"automatic_{i}_{label}".encode()); mask.chmod(0o444)
            row.update({label+"_query": query, label+"_mask_file": mask.name, label+"_mask_sha256": module.sha256(mask),
                        label+"_mask_bytes": mask.stat().st_size, label+"_mask_pixels": 100})
        rows.append(row)
    path = inputs/"manifest.json"; path.write_text(json.dumps(dict(schema=module.masks.SCHEMA, images=images)))
    report = dict(stage=module.masks.STAGE, status="pass", phase="complete", frames=15, network="none", private_truth_read=False,
        ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[],
        actual_automatic_inference_verified=True, all_cases_retained=True, human_query="person.", object_query="bottle.",
        input_manifest_sha256=module.sha256(path), input_manifest_bytes=path.stat().st_size,
        script_sha256=module.sha256(Path(module.masks.__file__)), shared_mask_helper_sha256=module.sha256(Path(module.masks.masks.__file__)),
        detector_revision=module.masks.masks.DETECTOR_REVISION, sam2_weights_revision=module.masks.masks.SAM2_REVISION,
        producer_revision="a"*40, image_id="sha256:"+"b"*64, records=rows, actual_detector_calls=30,
        actual_sam2_calls=30, actual_sam2_image_encoder_calls=15,
        model_assets={name: {"path": str(tmp_path/"weights"/name), "sha256": value[0], "bytes": value[1]}
                      for name, value in module.masks.masks.ASSETS.items()})
    def save():
        target = masks/"report.json"; target.chmod(0o644) if target.exists() else None
        target.write_text(json.dumps(report)); target.chmod(0o444)
    save(); return tmp_path, report, save


def test_public_masks_and_all_fifteen_original_frames(module, public):
    root, _, _ = public; records, receipt = module.public_inputs(root)
    assert len(records) == 15 and records[-1]["frame_index"] == 4
    assert receipt["public_manifest_sha256"] == receipt["public_inputs_sha256"]
    assert "inference" in module.helper_identities()  # importlib alias is intentionally not in sys.modules


@pytest.mark.parametrize("fault", [None, "missing", "extra", "path", "hash", "bytes", "boolbytes", "extra_field"])
def test_actual_asset_producer_receipt_not_oversimplified_mock(module, public, monkeypatch, fault):
    """Run the unmodified real producer with tiny files, not invented receipts."""
    root, report, save = public; results = root/"results"; results.mkdir(); pins = {}
    for name in module.masks.masks.ASSETS:
        path = root/"weights"/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(("own_tiny_asset:"+name).encode())
        actual = module.masks.masks.identity(path); pins[name] = (actual["sha256"], actual["bytes"])
    monkeypatch.setattr(module.masks.masks, "ASSETS", pins)
    (results/"image-grounding.json").write_text(json.dumps({"Id": report["image_id"]}))
    (results/"weights-acquisition.json").write_text(json.dumps({"assets": [
        {"repo_id": "IDEA-Research/grounding-dino-base", "revision": module.masks.masks.DETECTOR_REVISION, "path": str(root/"weights/grounding_dino")},
        {"repo_id": "facebook/sam2.1-hiera-large", "revision": module.masks.masks.SAM2_REVISION, "path": str(root/"weights/sam2")},
    ]}))
    report["model_assets"] = module.masks.masks.validate_assets(root, report["image_id"])
    assert len(report["model_assets"]) == 9
    name = next(iter(pins)); value = report["model_assets"][name]
    assert set(value) == {"path", "sha256", "bytes"} and value == module.masks.masks.identity(root/"weights"/name)
    if fault == "missing": report["model_assets"].pop(name)
    elif fault == "extra": report["model_assets"]["unapproved/file.pt"] = value.copy()
    elif fault == "path": value["path"] = str(root/"weights/../eval_private/file.pt")
    elif fault == "hash": value["sha256"] = "0"*64
    elif fault == "bytes": value["bytes"] += 1
    elif fault == "boolbytes": value["bytes"] = True
    elif fault == "extra_field": value["unverified"] = True
    save()
    if fault is None: assert len(module.public_inputs(root)[0]) == 15
    else:
        with pytest.raises(ValueError, match="exact detector/SAM2 assets"): module.public_inputs(root)


@pytest.mark.parametrize("fault", ["calls", "status", "private", "source", "masksha", "rgbsha", "assets", "query", "extra", "writable", "boolframes"])
def test_mask_integrity_fails_before_private_or_network(module, public, fault):
    root, report, save = public; masks = root/module.BASE/"automatic_masks"
    if fault == "calls": report["actual_sam2_calls"] = 29
    elif fault == "status": report["status"] = "running"
    elif fault == "private": report["private_truth_read"] = True
    elif fault == "source": report["script_sha256"] = "0"*64
    elif fault == "masksha": report["records"][0]["human_mask_sha256"] = "0"*64
    elif fault == "rgbsha": report["records"][0]["rgb_sha256"] = "0"*64
    elif fault == "assets": report["model_assets"] = {}
    elif fault == "query": report["records"][0]["object_query"] = "bottle from truth"
    elif fault == "extra": (masks/"private.npz").write_bytes(b"forbidden")
    elif fault == "boolframes": report["frames"] = True
    save()
    if fault == "writable": (masks/report["records"][0]["human_mask_file"]).chmod(0o644)
    with pytest.raises(ValueError): module.public_inputs(root)


@pytest.fixture
def raw(module, monkeypatch):
    monkeypatch.setattr(module, "WIDTH", 4); monkeypatch.setattr(module, "HEIGHT", 3)
    monkeypatch.setattr(module, "VERTICES", 100); monkeypatch.setattr(module, "JOINTS", 3)
    normalized = np.diag([1/4, 1/3, 1.]) @ module.CAMERA_K
    # Own pinhole points, including the unchanged half-pixel centre convention.
    y, x = np.indices((3, 4)); z = np.full((3, 4), 2., np.float32)
    points = np.stack(((x+.5-512)/1280*z, (y+.5-384)/1280*z, z), axis=-1).astype(np.float32)
    row = {key: np.zeros(shape, np.float32) for key, shape in module.BLOCKS.items()}
    row["pred_cam_t"][:] = [0., 0., 2.]
    row.update(raw_depth=z, raw_points=points, validity=np.ones((3, 4), bool), rendered_depth=z.copy(), silhouette=np.ones((3, 4), bool),
        human_mask=np.ones((3, 4), bool), object_mask=np.ones((3, 4), bool), raw_vertices_camera_m=np.ones((100, 3), np.float32),
        raw_joints_camera_m=np.ones((3, 3), np.float32), human_faces=np.zeros((36874, 3), np.int64), camera_K=module.CAMERA_K.copy(),
        clip_index=np.array(0, np.int64), frame_index=np.array(0, np.int64))
    record = dict(file="clip_00_frame_000.png", clip_index=0, frame_index=0, sha256="a"*64)
    assert normalized.shape == (3, 3)
    return row, record


def pair_from(raw):
    left = np.arange(100) < 50
    return dict(raw_vertices_camera_m=raw["raw_vertices_camera_m"].copy(), shared_vertices_camera_m=raw["raw_vertices_camera_m"].copy(),
        raw_joints_camera_m=raw["raw_joints_camera_m"].copy(), shared_joints_camera_m=raw["raw_joints_camera_m"].copy(),
        human_faces=raw["human_faces"].copy(), hand_mask_left=left, hand_mask_right=~left, camera_K=raw["camera_K"].copy(),
        pred_cam_t=raw["pred_cam_t"].copy(), raw_model_controls=raw["mhr_model_params"].copy(), shared_model_controls=raw["mhr_model_params"].copy(),
        raw_shape_params=raw["shape_params"].copy(), shared_shape_params=raw["shape_params"].copy(),
        raw_scale_params=raw["scale_params"].copy(), shared_scale_params=raw["scale_params"].copy(), expression=raw["expr_params"].copy(),
        clip_index=raw["clip_index"].copy(), frame_index=raw["frame_index"].copy())


def test_raw_and_shared_schema_no_scale_or_root_repair(module, raw):
    data, record = raw; module.validate_raw(data, record); candidate = pair_from(data)
    candidate["shared_shape_params"][0] = .3; candidate["shared_scale_params"][0] = -.2
    candidate["shared_model_controls"][136] = .12
    module.validate_pair(candidate, record, data)
    assert not data["shape_params"].any() and not data["mhr_model_params"].any()


@pytest.mark.parametrize("fault", ["extra", "pose", "root", "translation", "rawV", "expr", "hand", "dtype", "frame", "K", "nonfinite"])
def test_paired_invariants_fail_closed(module, raw, fault):
    data, record = raw; candidate = pair_from(data)
    if fault == "extra": candidate["truth"] = np.zeros(1)
    elif fault == "pose": candidate["shared_model_controls"][5] = .1
    elif fault == "root": candidate["shared_model_controls"][0] = 1.
    elif fault == "translation": candidate["pred_cam_t"][0] = 1.
    elif fault == "rawV": candidate["raw_vertices_camera_m"][0, 0] += 1.
    elif fault == "expr": candidate["expression"][0] = .01
    elif fault == "hand": candidate["hand_mask_right"][:] = True
    elif fault == "dtype": candidate["shared_shape_params"] = candidate["shared_shape_params"].astype(np.float64)
    elif fault == "frame": candidate["frame_index"] = np.array(1, np.int64)
    elif fault == "K": candidate["camera_K"][0, 0] = 1000
    else: candidate["shared_vertices_camera_m"][0, 0] = np.nan
    with pytest.raises(ValueError): module.validate_pair(candidate, record, data)


def test_invalid_pointmap_preserved_but_not_silently_filled(module, raw):
    data, record = raw; data["validity"][0, 0] = False; data["raw_points"][0, 0] = np.nan; data["raw_depth"][0, 0] = np.nan
    module.validate_raw(data, record)
    assert np.isnan(data["raw_points"][0, 0]).all()
    data["validity"][0, 0] = True
    with pytest.raises(ValueError): module.validate_raw(data, record)


def test_reference_units_camera_flip_already_applied_and_translation_once(module, raw):
    translations = np.repeat([[.1, -.2, 2.]], 15, axis=0).astype(np.float32)
    root_v = np.full((15, 100, 3), .25, np.float64); root_j = np.full((15, 3, 3), .125, np.float64)
    shared_v = (root_v+translations[:, None]).astype(np.float32); shared_j = (root_j+translations[:, None]).astype(np.float32)
    result = module.reference_errors(root_v*1000, root_j, shared_v, shared_j, translations)
    assert max(result["per_frame_mean_mm"]) < .001
    with pytest.raises(ValueError): module.reference_errors(root_v*1000+translations[:, None]*1000, root_j, shared_v, shared_j, translations)
    bad = shared_v.copy(); bad[8] += .004
    with pytest.raises(ValueError): module.reference_errors(root_v*1000, root_j, bad, shared_j, translations)


def test_native_head_shared_only_two_identity_blocks_and_no_inputs_mutated(module, raw):
    original, _ = raw; first = {k: v.copy() for k, v in original.items()}; first["shape_params"][0] = .4; first["scale_params"][0] = -.2
    class Tensor:
        def __init__(self, value): self.value = np.asarray(value)
        def __getitem__(self, item): return Tensor(self.value[item])
        def __mul__(self, value): return Tensor(self.value*value)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
    class Torch:
        inference_mode = staticmethod(nullcontext)
        as_tensor = staticmethod(lambda value, **_: Tensor(value))
    class Head:
        def mhr_forward(self, **kw):
            assert not kw["global_trans"].value.any() and not kw["expr_params"].value.any()
            np.testing.assert_array_equal(kw["shape_params"].value[0], first["shape_params"])
            np.testing.assert_array_equal(kw["scale_params"].value[0], first["scale_params"])
            np.testing.assert_array_equal(kw["body_pose_params"].value[0], original["body_pose_params"])
            assert kw["return_keypoints"] is False and kw["return_joint_rotations"] is False
            return Tensor(np.ones((1, 100, 3), np.float32)), Tensor(np.ones((1, 3, 3), np.float32)), Tensor(np.zeros((1, 204), np.float32))
    v, j, _ = module.shared_decode(Torch, Head(), original, first)
    np.testing.assert_array_equal(v[0], np.array([1., -1., 1.], np.float32))
    np.testing.assert_array_equal(j[0], v[0]); assert not original["shape_params"].any()


def test_frozen_all_fifteen_before_shared_branch_and_complete_inventory(module, raw, tmp_path):
    data, record = raw; output = tmp_path/"predictions"; (output/"raw").mkdir(parents=True)
    records = []; rows = []
    for i in range(15):
        clip, frame = divmod(i, 5); item = dict(record, file=f"clip_{clip:02d}_frame_{frame:03d}.png", clip_index=clip, frame_index=frame)
        values = dict(data, clip_index=np.array(clip, np.int64), frame_index=np.array(frame, np.int64))
        records.append(item); rows.append(module.save(output, "raw", item, values))
    assert len(module.frozen_rows(output, rows, records, "raw", module.validate_raw)) == 15
    with pytest.raises(ValueError): module.frozen_rows(output, rows[:14], records, "raw", module.validate_raw)
    target = output/rows[0]["artifact"]; target.chmod(0o644)
    with pytest.raises(ValueError): module.frozen_rows(output, rows, records, "raw", module.validate_raw)


def test_source_binding_rejects_forged_receipt_before_assets(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module.human.body, "_source_identity", lambda _: pytest.fail("No assets should be inspected yet"))
    with pytest.raises(ValueError): module.validate_producer_bindings(tmp_path, {})


def test_paired_region_masks_bound_to_native_byte_evidence(module, raw):
    pair = pair_from(raw[0]); hashes = {side: module.hashlib.sha256(pair["hand_mask_"+side].tobytes()).hexdigest() for side in ("left", "right")}
    module.validate_region_masks([pair], {"hand_region_sha256": hashes})
    pair["hand_mask_left"][0] = False
    with pytest.raises(ValueError): module.validate_region_masks([pair], {"hand_region_sha256": hashes})


def test_runtime_lazy_torch_no_private_and_official_one_chunk15_z_supported(module):
    source = Path(module.__file__).read_text(); tree = ast.parse(source)
    names = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)} | {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    assert "torch" not in names and "eval_private" not in source
    assert source.index('raw_frozen_before_shared=True') < source.index('v, j, controls = shared_decode')
    assert 'chunk=16, precision="float32"' in source and 'reference.run' in source and 'official.convert(' not in source
    assert 'shared_vertices.astype(np.float64)*1000' in source
    assert 'inference_type="body"' in source and 'inference_type="full"' not in source


def test_wrapper_private_mount_separate_cpu_after_public_and_scoped_chown(module):
    source = Path(module.__file__).with_name("run_identity_rgb_validate.sh").read_text()
    gpu, cpu = source.split('# The public producer has terminated', 1)
    assert "eval_private" not in gpu and "eval_private,readonly" in cpu and "--gpus" not in cpu
    assert 'chown "$UID_SCENE:$GID_SCENE" "$OUT" "$QUALITY"' in source and 'chown -R' not in source
    assert '603s' in gpu and '123s' in cpu and '--network none' in source
    assert 'src=$BASE,dst=$BASE' not in source and 'src=$ROOT,dst=$ROOT' not in source


def test_unknown_args_rejected_before_runtime(module):
    with pytest.raises(SystemExit): module.main(["--ground-truth-identity"])


def test_decoupled_official_pins_identical_without_runtime_inverse_driver(module):
    import cari_converter
    assert module.CONVERTER_SHA256 == cari_converter.CONVERTER_SHA256
    assert module.REFERENCE_MODEL_SHA256 == cari_converter.REFERENCE_MODEL_SHA256
    assert "from cari_converter" not in Path(module.__file__).read_text()
