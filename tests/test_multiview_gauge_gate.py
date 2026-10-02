"""Tiny source/algebra/provenance contracts; no models, native imports or media."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


INFRA = Path(__file__).parents[1] / "infra"
spec = importlib.util.spec_from_file_location("multiview_gauge_tests", INFRA / "multiview_gauge_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


EXPORTER = '''
def to_glb(app_rep: UnavailableGaussian, mesh: UnavailableMesh, with_mesh_postprocess=True,
           with_texture_baking=True, use_vertex_color=False) -> UnavailableTrimesh:
    vertices = mesh.vertices.float().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    vert_colors = mesh.vertex_attrs[:, :3].cpu().numpy()
    if with_mesh_postprocess:
        raise RuntimeError("Forbidden geometric repair")
    if with_texture_baking:
        raise RuntimeError("Forbidden model rendering")
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    if not with_mesh_postprocess and not with_texture_baking and use_vertex_color:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = vert_colors
    return mesh
'''


def texts():
    return {gate.POST: EXPORTER, gate.LAYOUT: '''
def get_mesh(Mesh, tfm_ori, device):
    mesh_vertices = Mesh.vertices.copy()
    mesh_vertices = mesh_vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    points_world = tfm_ori.transform_points(mesh_vertices.unsqueeze(0))
    return points_world
''', gate.TRANSFORMS: '''
def compose_transform(scale, rotation, translation):
    return tfm.scale(scale).rotate(rotation).translate(translation)
''', gate.INFERENCE: "tfm_ori = compose_transform(scale=Scale, rotation=Rotation, translation=Translation)"}


def test_literal_source_bound_rotation_and_native_scale_order():
    report = gate.source_contract(texts())
    assert report["exporter_row_rotation"] == gate.A.tolist()
    assert report["native_module_imported"] is False
    assert "GLB@A.T -> component_scale" in report["native_layout_order"]
    np.testing.assert_array_equal(gate.A.T @ gate.A, np.eye(3))
    assert np.linalg.det(gate.A) == 1


@pytest.mark.parametrize("kind", ["rotation", "rotation_duplicate", "native_inverse", "scale_order", "pose_constructor"])
def test_modified_source_contract_does_not_silently_adapt(kind):
    source = texts()
    if kind == "rotation": source[gate.POST] = source[gate.POST].replace("[0, 0, -1]", "[0, 0, 1]")
    elif kind == "rotation_duplicate": source[gate.POST] = source[gate.POST].replace("    return mesh", "    vertices = vertices @ np.array([[1,0,0],[0,0,-1],[0,1,0]])\n    return mesh")
    elif kind == "native_inverse": source[gate.LAYOUT] = source[gate.LAYOUT].replace("]).T", "])")
    elif kind == "scale_order": source[gate.TRANSFORMS] = source[gate.TRANSFORMS].replace(".scale(scale).rotate(rotation)", ".rotate(rotation).scale(scale)")
    else: source[gate.INFERENCE] = "unrelated native helper"
    with pytest.raises(ValueError): gate.source_contract(source)


def test_exact_function_only_no_imports_and_disabled_model_repair_branches():
    called = []
    class Mesh:
        def __init__(self, vertices, faces, process):
            called.append(process)
            self.vertices, self.faces = vertices, faces
            self.visual = SimpleNamespace()
    v = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]], dtype=np.float32)
    f = np.array([[0, 1, 2]], dtype=np.int64)
    saved = v.copy(), f.copy()
    source = "raise RuntimeError('Module import forbidden')\n" + EXPORTER
    mesh = gate.execute_exporter(source, v, f, SimpleNamespace(Trimesh=Mesh))
    np.testing.assert_array_equal(mesh.vertices, v @ gate.A)
    np.testing.assert_array_equal(mesh.faces, f)
    np.testing.assert_array_equal(v, saved[0]); np.testing.assert_array_equal(f, saved[1])
    assert called == [False]


def pose(scale=(.8, 1.1, 1.3)):
    return {"rotation": [1., 0., 0., 0.], "translation": [.1, -.2, 2.], "scale": list(scale)}


def test_anisotropic_noncommutation_explicit_not_silently_uniformized():
    result = gate.gauge_math()
    assert result["source_order_permuted_scale_max_error"] < 1e-12
    assert result["unpermuted_scale_noncommutation_max_error"] > .01
    assert "no native decoder/camera parity" in result["interpretation"]
    v = np.array([[.1, .2, .3]])
    p = pose((1.2, 1.2, 1.2))
    r, t, s = gate.pose_arrays(p)
    r_net = gate.N @ r.T @ gate.A
    np.testing.assert_allclose(gate.native_order_points(v, p), (v * s) @ r_net.T + t @ gate.N, atol=1e-12)


@pytest.mark.parametrize("kind", ["missing", "extra", "nonunit", "nonfinite", "negative_scale", "wrong_scale"])
def test_invalid_native_pose_rejected(kind):
    p = pose()
    if kind == "missing": p.pop("rotation")
    elif kind == "extra": p["gt_pose"] = []
    elif kind == "nonunit": p["rotation"][0] = 2
    elif kind == "nonfinite": p["translation"][0] = np.nan
    elif kind == "negative_scale": p["scale"][0] = -1
    else: p["scale"] = [1]
    with pytest.raises(ValueError): gate.pose_arrays(p)


def full(tmp_path):
    source = tmp_path / "vendor"
    path = source / gate.POST
    path.parent.mkdir(parents=True); path.write_text(EXPORTER)
    image = "sha256:" + "a" * 64
    data = {"stage": "native_mv_sam3d_full_execution_only", "status": "pass", "vendor_revision": gate.PIN,
            "image_id": image, "procedural_inputs_only": True, "challenge_inputs_used": False,
            "native_decode_verified": True, "native_constructor_verified": True, "accuracy_evaluated": False,
            "imported_source": {"sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils": gate.identity(path)},
            "proposals": {"single": {}, "three_view": {}}}
    report = tmp_path / "results/multiview-full-execution-gate-v2.json"
    report.parent.mkdir(); report.write_text(json.dumps(data))
    return source, image, report, data


def test_actual_frozen_full_pass_source_exporter_binding(tmp_path):
    source, image, path, data = full(tmp_path)
    result, receipt = gate.validate_full(tmp_path, source, image)
    assert result == data and receipt == gate.identity(path)


@pytest.mark.parametrize("kind", ["status", "image", "integer_bool", "sourcehash", "sourcepath", "proposal"])
def test_wrong_actual_full_source_or_provenance_blocks_partial_gate(tmp_path, kind):
    source, image, path, data = full(tmp_path)
    if kind == "status": data["status"] = "fail"
    elif kind == "image": data["image_id"] = "sha256:" + "b" * 64
    elif kind == "integer_bool": data["native_decode_verified"] = 1
    elif kind in ("sourcehash", "sourcepath"):
        entry = data["imported_source"]["sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils"]
        entry["sha256" if kind == "sourcehash" else "path"] = "tampered"
    else: data["proposals"].pop("three_view")
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError): gate.validate_full(tmp_path, source, image)


def test_gate_is_explicitly_partial_no_weights_rgb_gpu_mounts_or_adoption():
    source = (INFRA / "multiview_gauge_gate.py").read_text()
    for flag in ("actual_raw_decoder_parity_verified", "camera_bridge_verified", "metric_scale_verified", "adoption_authorized"):
        assert f'"{flag}": False' in source
    wrapper = (INFRA / "run_multiview_gauge_gate.sh").read_text()
    assert "--gpus" not in wrapper and "--network none" in wrapper
    assert "src=$ROOT/weights" not in wrapper and "src=$ROOT/data" not in wrapper
    result = subprocess.run(["bash", str(INFRA / "run_multiview_gauge_gate.sh"), "--unknown"], capture_output=True)
    assert result.returncode == 2
