"""D88 manufacturing/public firewalls; no models, media or CUDA executed."""
from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec = importlib.util.spec_from_file_location("own_depth_render", infra/"depth_rgb_render.py")
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def metadata():
    names = [f"unused_{i}" for i in range(249)]
    required = [f"{side}_{suffix}" for side in "lr" for suffix in
                ("uparm_ry", "elbow_bend", "wrist_ry", "index1_rz", "middle1_rz", "ring1_rz", "pinky1_rz")]
    for index, name in enumerate(required): names[index] = name
    bounds = np.tile([-1., 1.], (249, 1)); bounds[136:204] = [-.1, .1]
    bounds[136+np.array([0, 1, 2, 11, 12, 15, 16])] = 0
    return names, bounds


def test_new_frozen_recipe_is_not_rejected_identity_cohort(module):
    p = module.protocol.D88
    assert p.base == "validation/depth_rgb_v1" and p.schema == "world-reward-depth-rgb-v1"
    assert (p.width, p.height, p.clips, p.frames) == (1024, 768, 3, 5)
    assert p.focals == (1200., 1500., 1800.) and p.scales == (-.03, .025, .06)
    assert p.shapes == ((-.21, -.09), (.26, .11), (.43, -.12))
    assert p.bottle_depth_offsets == (-.12, -.06, 0., .065, .13)
    assert len(p.digest()) == 64 and p.digest() == module.protocol.RGBProtocol().digest()
    assert p.digest() != replace(p, focals=(1201., 1500., 1800.)).digest()
    with pytest.raises(FrozenInstanceError): p.frames = 10
    assert not np.array_equal(np.asarray(p.shapes), np.asarray(module.primitives.SHAPES))
    assert p.scales != module.primitives.SCALES


@pytest.mark.parametrize("change", [dict(width=True), dict(frames=0), dict(focals=(0., 1500., 1800.)),
    dict(scales=(float("nan"), .025, .06)), dict(shapes=((0.,), (0., 0.), (0., 0.))),
    dict(base="../eval_private"), dict(focals=[1200., 1500., 1800.])])
def test_invalid_protocol_contract_fails(module, change):
    with pytest.raises(ValueError): replace(module.protocol.D88, **change)


@pytest.fixture
def public(module, tmp_path):
    folder = tmp_path/"inputs"; folder.mkdir(); digests = []
    for clip in range(3):
        for frame in range(5):
            path = folder/f"clip_{clip:02d}_frame_{frame:03d}.png"
            path.write_bytes(f"own_rgb_{clip}_{frame}".encode()); path.chmod(0o444); digests.append(module.sha256(path))
    manifest = module.protocol.public_manifest(digests); path = folder/"manifest.json"
    def write():
        if path.exists(): path.chmod(0o644)
        path.write_text(json.dumps(manifest)); path.chmod(0o444)
    write(); return folder, manifest, write


def test_exact_readonly_public_rgb_schema_and_order(module, public):
    folder, manifest, _ = public; records, receipt = module.protocol.public_inputs(folder)
    assert len(records) == 15 and records[-1]["clip_index"] == 2 and records[-1]["frame_index"] == 4
    assert set(manifest) == {"schema", "images"} and all(set(r) == {"file", "sha256", "width", "height"} for r in manifest["images"])
    assert receipt == {"sha256": module.sha256(folder/"manifest.json"), "bytes": (folder/"manifest.json").stat().st_size}
    assert set(records[0]) == {"file", "sha256", "width", "height", "path", "clip_index", "frame_index"}
    assert np.array_equal(module.protocol.FIXED_K, [[1280, 0, 512], [0, 1280, 384], [0, 0, 1]])


@pytest.mark.parametrize("fault", ["count", "order", "hash", "grid", "bool_grid", "label", "extra", "writable", "symlink"])
def test_public_reader_never_repairs_or_accepts_private_input(module, public, fault):
    folder, manifest, write = public
    if fault == "count": manifest["images"].pop()
    elif fault == "order": manifest["images"][0]["file"] = "clip_00_frame_001.png"
    elif fault == "hash": manifest["images"][0]["sha256"] = "0"*64
    elif fault == "grid": manifest["images"][0]["width"] = 512
    elif fault == "bool_grid": manifest["images"][0]["height"] = True
    elif fault == "label": manifest["images"][0]["camera_K"] = [1280]
    elif fault == "extra": (folder/"truth.npz").write_bytes(b"private")
    elif fault == "writable": (folder/manifest["images"][0]["file"]).chmod(0o644)
    else:
        path = folder/manifest["images"][0]["file"]; target = folder.parent/"other.png"
        target.write_bytes(path.read_bytes()); target.chmod(0o444); path.unlink(); path.symlink_to(target)
    write()
    with pytest.raises(ValueError): module.protocol.public_inputs(folder)


def test_three_native_identities_all249_legal_and_clip_constant(module):
    names, bounds = metadata(); controls, identity, changes = module.protocol.named_controls(names, bounds)
    assert controls.shape == (15, 204) and identity.shape == (15, 45) and controls.dtype == identity.dtype == np.float32
    locked = np.all(bounds[136:204] == 0, axis=1)
    for clip in range(3):
        rows = slice(5*clip, 5*(clip+1)); assert not controls[rows, 136:][:, locked].any()
        assert np.all(controls[rows, 136:][:, ~locked] == np.float32(module.protocol.D88.scales[clip]))
        assert np.all(identity[rows, :2] == np.asarray(module.protocol.D88.shapes[clip], np.float32))
        assert not identity[rows, 2:].any() and not np.array_equal(controls[5*clip, :136], controls[5*clip+4, :136])
    full = np.c_[controls, identity]; neutral = full[::5].copy(); neutral[:, :136] = 0
    assert np.all(full >= bounds[:, 0]) and np.all(full <= bounds[:, 1])
    assert np.all(neutral >= bounds[:, 0]) and np.all(neutral <= bounds[:, 1])
    assert len(changes) == 120 and all(r["column"] < 136 and r["name"] == names[r["column"]] for r in changes)
    old, _, _ = module.primitives.named_controls(names, bounds); assert not np.array_equal(old[:, :136], controls[:, :136])


@pytest.mark.parametrize("fault", ["locked_all", "free_narrow", "unused_shape", "zero_pose", "motion", "missing", "duplicate", "masked"])
def test_native_bound_gates_include_all_zeros_and_refuse_clipping(module, fault):
    names, bounds = metadata()
    if fault == "locked_all": bounds[136:204] = 0
    elif fault == "free_narrow": bounds[153] = [-.01, .01]
    elif fault == "unused_shape": bounds[210] = [.1, 1.]
    elif fault == "zero_pose": bounds[30] = [.1, 1.]
    elif fault == "motion": bounds[7] = [-.1, .1]
    elif fault == "missing": names[0] = "unknown"
    elif fault == "duplicate": names[-1] = names[0]
    else: bounds = np.ma.array(bounds, mask=False)
    with pytest.raises(ValueError): module.protocol.named_controls(names, bounds)


def test_new_scene_moves_rigidly_with_private_focal_and_unchanged_topology(module):
    human = np.tile([[-.1, -.5, 0], [.1, .5, .1], [0, .3, -.1]], (40, 1)); faces = np.array([[0, 1, 2]], np.int64)
    mask = np.ones(len(human), bool); regions = {side: {"vertex_mask": mask} for side in "lr"}
    first = module.scene(human, faces, regions, np.zeros(3), 4., 1., 0, 0)
    last = module.scene(human, faces, regions, np.zeros(3), 4., 1., 0, 4)
    assert not np.array_equal(first[0], last[0]) and not np.array_equal(first[1], last[1])
    assert np.array_equal(first[4], last[4]) and first[-1][0, 0] == 1200.
    assert np.min(first[3][:, 2]) > .01 and np.array_equal(first[4][:len(faces)], faces)
    assert np.array_equal(first[4][len(faces):len(faces)+len(first[2])], first[2]+len(human))
    assert np.linalg.norm(first[0][0]-first[0][1]) == pytest.approx(np.linalg.norm(human[0]-human[1]))


def test_framing_uses_three_neutral_meshes_not_framewise_private_camera_fit(module):
    mesh = np.array([[-.4, -.9, -.1], [.4, .9, .1], [0, 0, 0]])
    neutral = np.stack([mesh, mesh*1.02, mesh*.95]); result = module.fixed_framing(neutral)
    shifted = module.fixed_framing(neutral+[2, 3, 4])
    assert shifted[0] == pytest.approx(result[0]+[2, 3, 4]) and shifted[1:] == pytest.approx(result[1:])


def test_no_local_runtime_or_truth_input_cli_and_thin_primitive_reuse(module):
    source = Path(module.__file__).read_text()
    assert '"torch" in sys.modules' in source and source.index('CUBLAS_WORKSPACE_CONFIG') < source.index('import torch\n')
    assert source.count('actual_reference_forward_calls"] += 1') == 2
    assert 'primitives.foreground_truth' in source and 'primitives.background' in source and 'primitives.bottle_mesh' in source
    assert 'primitives.scene(' not in source and 'primitives.named_controls(' not in source
    assert 'private.mkdir(mode=0o700)' in source and 'png.chmod(0o444)' in source and 'truth.chmod(0o400)' in source
    assert 'full_HOI_verified=False' in source and 'inference_performed=False' in source
    assert module.TRUTH_KEYS == {"human_vertices_camera_m", "human_faces", "object_vertices_camera_m", "object_faces", "camera_K",
                                 "scene_depth_m", "visible_face_indices", "clip_index", "frame_index"}
    with pytest.raises(SystemExit): module.main(["--focal", "1280"])
