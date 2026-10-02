"""CPU, source-bound partial GLB gauge audit; no inferred raw-decoder parity.

Executes only the exact pinned to_glb function on an own tetrahedron with its
model-dependent branches disabled. Actual proposal inversion is a roundtrip,
not evidence of the original decoded vertices, metric correctness or adoption.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
from types import SimpleNamespace

import numpy as np


PIN = "abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd"
A = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
N = np.diag([-1., -1., 1.])
POST = "sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py"
LAYOUT = "sam3d_objects/pipeline/layout_post_optimization_utils.py"
TRANSFORMS = "sam3d_objects/data/dataset/tdfy/transforms_3d.py"
INFERENCE = "sam3d_objects/pipeline/inference_utils.py"
SOURCES = {
    POST: ("bf422f54c3df621e5caaf362fa60156a1ce7f77e6174cb308a9cf52a6800c875", 30950),
    LAYOUT: ("b19be16a96855a630ec2a4ba160957959f25b4313919002d0b5e2f66a93bb2f4", 14638),
    TRANSFORMS: ("496d4e39974c188bd56464d4d303b22fe3e20c6d544563b67e365bf6f9cba6fb", 1572),
    INFERENCE: ("93a6f93cc5e00b78447556f2091b5e0047758eb67385aeddd7feac4a70d9e2ca", 31728),
}


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ValueError("Require a canonical, regular nonsymlink artifact")
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def source_function(text, name):
    matches = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("Require exactly one pinned source function")
    return matches[0]


def source_contract(texts):
    post = source_function(texts[POST], "to_glb")
    rotations = [node for node in ast.walk(post) if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "vertices" for target in node.targets)
        and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult)]
    if len(rotations) != 1 or not isinstance(rotations[0].value.right, ast.Call):
        raise ValueError("Expected one explicit native exporter rotation")
    if not np.array_equal(np.array(ast.literal_eval(rotations[0].value.right.args[0])), A):
        raise ValueError("Pinned exporter canonical rotation changed")
    layout = ast.unparse(source_function(texts[LAYOUT], "get_mesh"))
    compose = ast.unparse(source_function(texts[TRANSFORMS], "compose_transform"))
    inference = texts[INFERENCE]
    if ("mesh_vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T" not in layout
            or "tfm_ori.transform_points(mesh_vertices.unsqueeze(0))" not in layout
            or ".scale(scale).rotate(rotation).translate(translation)" not in compose
            or "compose_transform(scale=Scale, rotation=Rotation, translation=Translation)" not in inference):
        raise ValueError("Pinned native layout scale/rotation order changed")
    return {"exporter_row_rotation": A.tolist(), "exporter_function_lines": [post.lineno, post.end_lineno],
            "native_layout_order": "GLB@A.T -> component_scale -> row_Rq -> translation -> OpenCV_Nxy",
            "layout_source_function": "get_mesh", "pose_constructor": "compose_transform.scale.rotate.translate",
            "native_module_imported": False}


class ArrayTensor:
    """Own NumPy adapter solely for this pinned function's CPU accessors."""
    def __init__(self, array): self.array = np.asarray(array)
    def float(self): return ArrayTensor(self.array.astype(np.float32))
    def cpu(self): return self
    def numpy(self): return self.array
    def __getitem__(self, key): return ArrayTensor(self.array[key])


def execute_exporter(text, vertices, faces, trimesh):
    node = copy.deepcopy(source_function(text, "to_glb"))
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs): arg.annotation = None
    namespace = {"np": np, "trimesh": trimesh}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<pinned-to_glb-only>", "exec"), namespace)
    raw = SimpleNamespace(vertices=ArrayTensor(vertices), faces=ArrayTensor(faces),
                          vertex_attrs=ArrayTensor(np.ones((len(vertices), 3), dtype=np.float32)))
    return namespace["to_glb"](None, raw, with_mesh_postprocess=False, with_texture_baking=False, use_vertex_color=True)


def pose_arrays(pose):
    from scipy.spatial.transform import Rotation
    if not isinstance(pose, dict) or set(pose) != {"rotation", "translation", "scale"}:
        raise ValueError("Require exactly native WXYZ quaternion, translation and scale")
    q, t, s = (np.asarray(pose[key], dtype=float) for key in ("rotation", "translation", "scale"))
    if (q.shape != (4,) or t.shape != (3,) or s.shape != (3,) or not np.isfinite(np.concatenate((q, t, s))).all()
            or not np.isclose(np.linalg.norm(q), 1., atol=1e-3) or np.any(s <= 0)):
        raise ValueError("Native pose must be finite, unit quaternion and positive scales")
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), t, s


def native_order_points(glb_vertices, pose):
    rotation, translation, scale = pose_arrays(pose)
    return ((np.asarray(glb_vertices) @ A.T) * scale) @ rotation @ N + translation @ N


def gauge_math():
    from scipy.spatial.transform import Rotation
    v = np.array([[.13, -.19, .31], [-.23, .17, .07], [.29, .05, -.11]])
    q = Rotation.from_rotvec([.21, -.32, .13]).as_quat()[[3, 0, 1, 2]]
    pose = {"rotation": q.tolist(), "translation": [.1, -.2, 2.], "scale": [.8, 1.1, 1.3]}
    r, t, s = pose_arrays(pose)
    native = native_order_points(v, pose)
    r_net = N @ r.T @ A
    permuted = (v * s[[0, 2, 1]]) @ r_net.T + t @ N
    unchanged = (v * s) @ r_net.T + t @ N
    error = float(np.abs(native - permuted).max())
    if error > 1e-12 or not np.allclose(r_net.T @ r_net, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(r_net), 1.):
        raise RuntimeError("Own analytic source-order bridge failed")
    return {"source_order_permuted_scale_max_error": error,
            "unpermuted_scale_noncommutation_max_error": float(np.abs(native - unchanged).max()),
            "interpretation": "conditional source-order algebra only; no native decoder/camera parity proof"}


def validate_full(root, source, image):
    path = root / "results/multiview-full-execution-gate-v2.json"
    data = json.loads(path.read_text())
    required = {"stage": "native_mv_sam3d_full_execution_only", "status": "pass", "vendor_revision": PIN,
                "image_id": image, "procedural_inputs_only": True, "challenge_inputs_used": False,
                "native_decode_verified": True, "native_constructor_verified": True, "accuracy_evaluated": False}
    if any(type(data.get(k)) is not type(v) or data.get(k) != v for k, v in required.items()):
        raise RuntimeError("Require actual full native V2 execution on same source/image")
    imported = data.get("imported_source", {}).get("sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils", {})
    actual = identity(source / POST)
    if imported.get("path") != actual["path"] or imported.get("sha256") != actual["sha256"]:
        raise RuntimeError("Actual full exporter imported source differs from pinned bytes")
    if set(data.get("proposals", {})) != {"single", "three_view"}:
        raise RuntimeError("Require both actual frozen proposals")
    return data, identity(path)


def run(root, report):
    import trimesh
    source = root / "vendor/mv-sam3d" / PIN
    texts, records = {}, {}
    for relative, expected in SOURCES.items():
        records[relative] = identity(source / relative)
        if (records[relative]["sha256"], records[relative]["bytes"]) != expected:
            raise RuntimeError("Independent pinned source identity failed")
        texts[relative] = (source / relative).read_text()
    report.update(source_identity=records, source_contract=source_contract(texts), analytic_order=gauge_math())
    full, report["full_execution_report"] = validate_full(root, source, report["source_image_id"])
    v = np.array([[.1, .2, .3], [.1, -.2, -.3], [-.1, .2, -.3], [-.1, -.2, .3]], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)
    mesh = execute_exporter(texts[POST], v, f, trimesh)
    if not isinstance(mesh, trimesh.Trimesh) or not np.array_equal(mesh.vertices, v @ A) or not np.array_equal(mesh.faces, f):
        raise RuntimeError("Pinned function subset changed own raw vertices/faces")
    original = trimesh.Trimesh(v, f, process=False)
    if not np.isclose(mesh.volume, original.volume, atol=1e-12):
        raise RuntimeError("Proper exporter rotation changed signed volume")
    serialized = trimesh.load(io.BytesIO(mesh.export(file_type="glb")), file_type="glb", force="mesh", process=False)
    if (not np.array_equal(serialized.faces, f) or not np.array_equal(serialized.vertices, v @ A)
            or not np.isclose(serialized.volume, original.volume, atol=1e-12)):
        raise RuntimeError("Own float32 GLB serialization changed topology or coordinates")
    report["own_pinned_function_subset_executed"] = True
    report["own_glb_serialization_roundtrip_verified"] = True
    report["proposals"] = {}
    for name, evidence in full["proposals"].items():
        directory = root / "results/multiview-full-proposals-v2"
        path, pose_path = directory / (name + ".glb"), directory / (name + "-pose.json")
        receipt = identity(path)
        if receipt != evidence["artifact"]:
            raise RuntimeError("Actual proposal path/SHA/bytes changed")
        pose_data = json.loads(pose_path.read_text())
        if (pose_data.get("schema") != "native_mv_proposal_unadopted" or pose_data.get("anchor_pose") != evidence["anchor_pose"]
                or pose_data.get("pose_applied_to_glb") is not False or pose_data.get("coordinate_conversion_verified") is not False):
            raise RuntimeError("Native unadopted pose receipt changed")
        pose_arrays(pose_data["anchor_pose"])
        actual_mesh = trimesh.load(path, force="mesh", process=False)
        if (not isinstance(actual_mesh, trimesh.Trimesh) or not np.isfinite(actual_mesh.vertices).all()
                or len(actual_mesh.vertices) != evidence["vertices"] or len(actual_mesh.faces) != evidence["faces"]):
            raise RuntimeError("GLB decoder/reload count or finiteness differs")
        raw_hypothesis = actual_mesh.vertices @ A.T
        roundtrip = float(np.abs(raw_hypothesis @ A - actual_mesh.vertices).max())
        if roundtrip > 1e-12:
            raise RuntimeError("Algebraic proper-rotation roundtrip failed")
        report["proposals"][name] = {"glb": receipt, "pose": identity(pose_path), "vertices": len(actual_mesh.vertices),
            "faces": len(actual_mesh.faces), "watertight": bool(actual_mesh.is_watertight),
            "winding_consistent": bool(actual_mesh.is_winding_consistent), "volume_glb_units_cubed": float(actual_mesh.volume),
            "algebraic_roundtrip_max_error": roundtrip, "roundtrip_is_raw_decoder_parity": False}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require isolated Linux CPU container")
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_SOURCE_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise RuntimeError("Require immutable code and actual producer image IDs")
    output = root / "results/multiview-gauge-partial-gate.json"
    if output.exists(): raise FileExistsError("Gauge audit reports are frozen")
    report = {"stage": "native_mv_glb_source_bound_partial_gauge", "status": "fail", "code_revision": revision,
              "vendor_revision": PIN, "source_image_id": image, "script": identity(Path(__file__)),
              "ground_truth_used": False, "challenge_inputs_used": False, "models_loaded": False,
              "gpu_used": False, "actual_raw_decoder_parity_verified": False, "camera_bridge_verified": False,
              "full_gauge_code_proof_verified": False, "native_pose_scale_axes_verified": False,
              "metric_scale_verified": False, "adoption_authorized": False, "submission_eligible": False,
              "license_status": "SAM_custom_competition_eligibility_unresolved"}
    try:
        run(root, report)
        report["status"] = "pass"
    except Exception as exc:
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        raise
    finally:
        with output.open("x") as handle: json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__": main()
