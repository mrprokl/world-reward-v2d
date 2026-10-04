"""Read-only GLB precision attribution; no historical pre-export arrays assumed."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import struct
import time
import tempfile

import numpy as np
import object_budget_endpoint as endpoint
from world_reward.data import sha256

STAGE = "source_mesh_precision_diagnostic"
PIN_SCHEMA = "world-reward-mesh-precision-diagnostic-pins-v1"
PIN_FILE = "configs/mesh_precision_diagnostic_v1.json"
BUDGET = 60
MAX_GLB = 256 * 1024 * 1024
ROLES = {"glb", "object_report", "original_producer", "image_evidence", "failed_report"}


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result: raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant")))


def item(rows, index):
    if type(index) is not int or not 0 <= index < len(rows): raise ValueError("Nonnegative in-range GLB index required")
    return rows[index]


def identity(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink input forbidden")
    before = path.stat()
    if not path.is_file() or before.st_size > MAX_GLB or before.st_nlink != 1:
        raise ValueError("Require bounded regular single-link input")
    result = {"bytes": before.st_size, "sha256": sha256(path)}
    after = path.stat()
    if any(getattr(before, k) != getattr(after, k) for k in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("Input changed while hashing")
    return result


def bindings(root, episode, pin_path):
    pin_identity = identity(pin_path); pins = strict_json(pin_path.read_text())
    if (set(pins) != {"schema", "episode_index", "diagnostic_image_id", "image_source_id", "original_producer_revision", "image_evidence_format", "files"}
            or pins["schema"] != PIN_SCHEMA or type(pins["episode_index"]) is not int or pins["episode_index"] != episode
            or set(pins["files"]) != ROLES or pins["image_evidence_format"] not in ("json_image_identity", "dispatcher_log")
            or not re.fullmatch("[0-9a-f]{40}", pins["original_producer_revision"])
            or any(not re.fullmatch("sha256:[0-9a-f]{64}", pins[k]) for k in ("diagnostic_image_id", "image_source_id"))):
        raise ValueError("Explicit complete provenance pins required")
    paths = {}; measured = {}
    for role, row in pins["files"].items():
        if (set(row) != {"path", "bytes", "sha256"} or type(row["bytes"]) is not int or not 0 < row["bytes"] <= MAX_GLB
                or not re.fullmatch("[0-9a-f]{64}", row["sha256"]) or not isinstance(row["path"], str)):
            raise ValueError("Invalid file pin")
        relative = Path(row["path"])
        if (relative.is_absolute() or str(relative) != row["path"] or ".." in relative.parts
                or relative.parts[0] not in ("outputs", "jobs", "results", "logs") or "eval_private" in relative.parts):
            raise ValueError("Unexpected provenance path")
        paths[role] = root / relative; measured[role] = identity(paths[role])
        if measured[role] != {k: row[k] for k in ("bytes", "sha256")}:
            raise ValueError("Frozen provenance hash differs")
    base = root / f"outputs/episode_{episode:06d}"
    if (paths["glb"] != base / "object_grounded/object.glb" or paths["object_report"] != base / "object_grounded/report.json"
            or paths["failed_report"] not in [base / f"{name}/report.json" for name in ("object_budget_endpoint", "object_budget_guarded", "object_budget_volume")]
            or not re.fullmatch(r"jobs/" + pins["original_producer_revision"] + r"/run_(track1_frontends|track1_initializers_only|episode_initializers)/code/infra/object_smoke\.py", pins["files"]["original_producer"]["path"])):
        raise ValueError("Canonical source paths differ")
    obj = strict_json(paths["object_report"].read_text()); old = strict_json(paths["failed_report"].read_text())
    if (obj.get("stage") != "sam3d_objects_grounded_fixed_frame" or obj.get("status") != "pass"
            or type(obj.get("episode_index")) is not int or obj["episode_index"] != episode or obj.get("frame_index") != 0
            or obj.get("object_sha256") != measured["glb"]["sha256"] or obj.get("script_sha256") != measured["original_producer"]["sha256"]
            or any(obj.get(k) is not False for k in ("ground_truth_used", "hand_labeled_test")) or obj.get("oracle_modes") != []
            or old.get("status") != "fail" or type(old.get("episode_index")) is not int or old["episode_index"] != episode or old.get("error_type") != "ValueError"
            or old.get("stage") != {"object_budget_endpoint": "world_reward_cpu_object_budget_endpoint", "object_budget_guarded": "world_reward_cpu_guarded_object_mesh", "object_budget_volume": "world_reward_cpu_volume_constrained_object_mesh"}[paths["failed_report"].parent.name]
            or any(old.get(k) is not False for k in ("ground_truth_used", "hand_labeled_test", "adoption_performed")) or old.get("oracle_modes") != []
            or not re.fullmatch("[0-9a-f]{64}", obj.get("input_sha256", "")) or old.get("input_sha256") != obj["input_sha256"]
            or old.get("error") not in ("Collapsed/numerically zero-area faces are forbidden", "Source contains collapsed triangles; no deletion or healing")
            or old.get("source_hashes", {}).get("object.glb") != measured["glb"]["sha256"]
            or old.get("source_hashes", {}).get("object_report") != measured["object_report"]["sha256"]):
        raise ValueError("Original source/failure lineage differs")
    evidence = paths["image_evidence"].read_text()
    if pins["image_evidence_format"] == "json_image_identity":
        data = strict_json(evidence)
        if pins["image_source_id"] not in (data.get("Id"), data.get("image_id")):
            raise ValueError("Original image identity differs")
    elif pins["image_source_id"] not in evidence:
        raise ValueError("Pinned dispatcher log lacks original image identity")
    if obj.get("image_id", pins["image_source_id"]) != pins["image_source_id"]:
        raise ValueError("Object report image differs")
    return paths, {"pins": pin_identity, "files": measured, "image_source_id": pins["image_source_id"],
                   "diagnostic_image_id": pins["diagnostic_image_id"], "original_producer_revision": pins["original_producer_revision"],
                   "image_evidence_format": pins["image_evidence_format"], "original_object_report_records_image_id": "image_id" in obj,
                   "original_producer_image_execution_directly_proven": "image_id" in obj, "image_evidence_only_when_producer_omits_image": "image_id" not in obj}


def raw_glb(path):
    """Decode only in-file uncompressed triangle accessors and scene transforms."""
    import trimesh
    raw = path.read_bytes()
    if len(raw) > MAX_GLB or len(raw) < 20 or struct.unpack_from("<4sII", raw) != (b"glTF", 2, len(raw)):
        raise ValueError("Invalid bounded GLB2 header")
    chunks = []; offset = 12
    while offset < len(raw):
        if offset + 8 > len(raw): raise ValueError("Truncated GLB chunk")
        size, kind = struct.unpack_from("<II", raw, offset); offset += 8
        if size % 4 or offset + size > len(raw): raise ValueError("Invalid GLB chunk bounds")
        chunks.append((kind, raw[offset:offset + size])); offset += size
    if [k for k, _ in chunks] != [0x4E4F534A, 0x004E4942]: raise ValueError("Require JSON plus BIN only")
    doc = strict_json(chunks[0][1]); binary = chunks[1][1]
    if (set(doc.get("extensionsRequired", [])) - {"KHR_materials_unlit"} or set(doc.get("extensionsUsed", [])) - {"KHR_materials_unlit"} or doc.get("animations") or doc.get("skins") or len(doc.get("buffers", [])) != 1
            or "uri" in doc["buffers"][0] or not len(binary) - 3 <= doc["buffers"][0]["byteLength"] <= len(binary)):
        raise ValueError("External/compressed/extended GLB unsupported")
    def accessor(index, position=False):
        a = item(doc["accessors"], index); view = item(doc["bufferViews"], a["bufferView"])
        types = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}
        if ("sparse" in a or a.get("normalized") or view.get("buffer", 0) != 0 or a["componentType"] not in types
                or a["type"] != ("VEC3" if position else "SCALAR") or (position != (a["componentType"] == 5126))):
            raise ValueError("Unsupported accessor representation")
        dt = np.dtype(types[a["componentType"]]); columns = 3 if position else 1; count = a["count"]
        stride = view.get("byteStride", columns * dt.itemsize); start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        end = start + (count - 1) * stride + columns * dt.itemsize
        if (type(count) is not int or count < 1 or stride < columns * dt.itemsize or stride % dt.itemsize
                or start % dt.itemsize or start < view.get("byteOffset", 0) or end > view.get("byteOffset", 0) + view["byteLength"]
                or end > doc["buffers"][0]["byteLength"]): raise ValueError("Accessor bounds/stride differ")
        return np.ndarray((count, columns), dt, binary, start, (stride, dt.itemsize)).copy()
    local = []; world = []; records = []; seen = set(); used = set()
    def visit(index, parent, active):
        if index in active or index in seen: raise ValueError("Cyclic/repeated scene node")
        seen.add(index); node = item(doc["nodes"], index)
        if "skin" in node: raise ValueError("Skinned meshes unsupported")
        if "matrix" in node:
            if any(k in node for k in ("rotation", "scale", "translation")): raise ValueError("Mixed node transforms")
            matrix = np.asarray(node["matrix"], np.float64).reshape(4, 4, order="F")
        else:
            q = np.asarray(node.get("rotation", [0, 0, 0, 1]), np.float64)
            if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-5: raise ValueError("Invalid node rotation")
            matrix = trimesh.transformations.quaternion_matrix(q[[3, 0, 1, 2]])
            matrix[:3, :3] *= np.asarray(node.get("scale", [1, 1, 1]), np.float64)[None, :]
            matrix[:3, 3] = node.get("translation", [0, 0, 0])
        if not np.isfinite(matrix).all() or not np.array_equal(matrix[3], [0, 0, 0, 1]): raise ValueError("Invalid affine scene transform")
        matrix = parent @ matrix
        if "mesh" in node:
            used.add(node["mesh"])
            for primitive in item(doc["meshes"], node["mesh"])["primitives"]:
                if primitive.get("mode", 4) != 4 or primitive.get("extensions") or primitive.get("targets"): raise ValueError("Require plain triangles")
                v = accessor(primitive["attributes"]["POSITION"], True)
                indices = accessor(primitive["indices"]) if "indices" in primitive else np.arange(len(v), dtype=np.int64)[:, None]
                f = indices.reshape(-1, 3).astype(np.int64)
                if not np.isfinite(v).all() or np.min(f) < 0 or np.max(f) >= len(v): raise ValueError("Invalid raw triangles")
                local.append((v, f)); world.append(trimesh.transform_points(v, matrix)[f])
                records.append({"position_component_type": 5126, "position_dtype": str(v.dtype), "vertices": len(v), "faces": len(f),
                                "node_transform_identity": bool(np.array_equal(matrix, np.eye(4))), "position_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
                                "indices_component_type": item(doc["accessors"], primitive["indices"])["componentType"] if "indices" in primitive else None,
                                "indices_dtype": str(indices.dtype), "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()})
        for child in node.get("children", []): visit(child, matrix, active | {index})
    scenes = doc["scenes"]; selected = doc.get("scene", 0)
    for node in item(scenes, selected)["nodes"]: visit(node, np.eye(4), set())
    if not world or used != set(range(len(doc["meshes"]))): raise ValueError("Unreferenced or missing triangle geometry")
    return local, np.concatenate(world), records


def triangle_hash(triangles):
    v = np.asarray(triangles, np.float64).reshape(-1, 3); f = np.arange(len(v)).reshape(-1, 3)
    rows = endpoint._triangle_rows(v, f).astype("<f8"); rows[rows == 0] = 0
    return hashlib.sha256(rows.tobytes()).hexdigest()


def area_metrics(vertices, faces):
    v = np.asarray(vertices); f = np.asarray(faces); eps = float(np.finfo(v.dtype).eps)
    extent = float(np.linalg.norm(np.ptp(v, axis=0))); threshold = eps * extent ** 2 * 32
    a, b = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    cross = np.cross(a, b); area = np.linalg.norm(cross, axis=1)
    vf = v.astype(np.float64); af, bf = vf[f[:, 1]] - vf[f[:, 0]], vf[f[:, 2]] - vf[f[:, 0]]; recomputed = np.linalg.norm(np.cross(af, bf), axis=1)
    product = np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1)
    sine = np.divide(recomputed, product, out=np.zeros_like(product), where=product > 0)
    if not np.isfinite(np.r_[extent, threshold, area, recomputed, sine]).all(): raise ValueError("Area arithmetic nonfinite")
    positive = area > 0
    return {"vertices": len(v), "faces": len(f), "dtype": str(v.dtype), "diagonal": extent, "global_double_area_threshold": threshold,
            "arithmetic_zero_faces": int(np.count_nonzero(~positive)), "cross_component_zero_faces": int(np.count_nonzero(np.all(cross == 0, axis=1))),
            "norm_underflow_zero_faces": int(np.count_nonzero((~positive) & np.any(cross != 0, axis=1))), "float64_recomputed_zero_faces": int(np.count_nonzero(recomputed == 0)),
            "positive_below_global_threshold": int(np.count_nonzero(positive & (area <= threshold))),
            "positive_angular_roundoff_faces": int(np.count_nonzero((recomputed > 0) & (sine <= 32 * eps))),
            "minimum_positive_double_area_over_diagonal_squared": float(np.min(area[positive]) / extent ** 2) if positive.any() and extent > 0 else None,
            "minimum_positive_sin_angle": float(np.min(sine[recomputed > 0])) if (recomputed > 0).any() else None,
            "global_area_guard_would_reject": bool(extent <= 0 or np.any(area <= threshold)), "angular_threshold_is_diagnostic_only": True}


def measure(path):
    local, raw_triangles, accessors = raw_glb(path); v, f = endpoint._load_mesh(path)
    raw_hash = triangle_hash(raw_triangles); loaded_hash = triangle_hash(v[f])
    if raw_hash != loaded_hash: raise ValueError("Raw scene/loader oriented triangle identity differs")
    unique, inverse = np.unique(v, axis=0, return_inverse=True); welded = inverse[f]
    if not np.array_equal(unique[welded], v[f]) or triangle_hash(unique[welded]) != loaded_hash: raise ValueError("Exact welding changed triangles")
    accepted = True
    try: endpoint.exact_weld(v, f)
    except ValueError as error:
        if str(error) != "Source contains collapsed triangles; no deletion or healing": raise
        accepted = False
    return {"accessors": accessors, "raw_accessor_area": [area_metrics(*x) for x in local], "loaded_area": area_metrics(v, f),
            "welded_area": area_metrics(unique, welded), "raw_scene_triangle_sha256": raw_hash, "loaded_triangle_sha256": loaded_hash,
            "welded_triangle_sha256": loaded_hash, "raw_scene_loader_triangles_identical": True, "welding_triangles_identical": True,
            "original_exact_weld_accepts": accepted, "vertices_merged_by_exact_weld": len(v) - len(unique), "faces_preserved": len(f)}


def selftest():
    import trimesh
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    with tempfile.TemporaryDirectory(prefix="mesh-precision-", dir="/tmp") as directory:
        p = Path(directory) / "control.glb"
        p.write_bytes(trimesh.Trimesh(v, f, process=False).export(file_type="glb"))
        if not measure(p)["original_exact_weld_accepts"]: raise ValueError("Real loader control failed")
        shifted = v + 1e8
        if area_metrics(shifted, f)["arithmetic_zero_faces"]: raise ValueError("F64 reference control collapsed")
        p.write_bytes(trimesh.Trimesh(shifted, f, process=False).export(file_type="glb"))
        loss = measure(p)
        if loss["original_exact_weld_accepts"] or loss["raw_accessor_area"][0]["arithmetic_zero_faces"] != 4:
            raise ValueError("Real F32 serialization loss control failed")
    tiny = area_metrics(np.array([[0., 0., 0.], [1e-8, 0., 0.], [0., 1e-8, 0.], [1., 1., 1.]]), np.array([[0, 1, 2]]))
    if tiny["arithmetic_zero_faces"] or tiny["positive_below_global_threshold"] != 1 or tiny["positive_angular_roundoff_faces"]:
        raise ValueError("Tiny well-shaped arithmetic control failed")
    return {"passed": True, "fixture_names": ["regular_glb_loader", "f64_to_glb_f32_loss", "tiny_well_shaped_global_threshold"],
            "f32_loss_zero_faces": 4, "tiny_positive_below_global_threshold": 1}


def source_helpers():
    code = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(code)): identity(p) for p in sorted(code.rglob("*")) if p.is_file() and "__pycache__" not in p.parts} | {"dispatch/" + name: identity(code.parent / name) for name in ("revision", "source-sha256")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    class Once(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            if getattr(namespace, self.dest, None) is not None: parser.error("Episode occurs exactly once")
            setattr(namespace, self.dest, value)
    parser.add_argument("--episode", required=True, choices=[str(x) for x in range(30)], action=Once); args = parser.parse_args(argv); episode = int(args.episode)
    root = Path(os.environ["WR_ROOT"]); code = Path(__file__).resolve().parents[1]; output = root / f"diagnostic/episode{episode:06d}-mesh-precision-v1"
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}: raise RuntimeError("Require remote CPU network-none")
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()): raise FileExistsError("Require fresh diagnostic output")
    report = {"stage": STAGE, "status": "fail", "phase": "integrity", "episode_index": episode, "budget_seconds": BUDGET,
              "producer_revision": os.environ["WR_CODE_REVISION"], "image_id": os.environ["WR_IMAGE_ID"], "device": "cpu", "network": "none",
              "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "adoption_performed": False, "prediction_exported": False, "geometry_changed": False,
              "production_gate_pass_claimed": False, "historical_preexport_arrays_available": False, "historical_preexport_identity_provable": False,
              "arithmetic_zero_is_not_exact_real_arithmetic_proof": True, "thresholds_changed": False, "uid": os.getuid(),
              "historical_failure_unchanged": False, "gpu_used": False, "models_loaded": False, "arrays_exported": False, "learned_predictions_performed": False}
    start = time.perf_counter(); before = None; helpers = None
    def expired(*unused): raise TimeoutError("Whole precision diagnostic exceeded60s")
    old_alarm = signal.signal(signal.SIGALRM, expired); old_term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try:
        if not re.fullmatch("[0-9a-f]{40}", report["producer_revision"]): raise ValueError("Immutable diagnostic revision required")
        helpers = source_helpers(); paths, before = bindings(root, episode, code / PIN_FILE)
        if before["diagnostic_image_id"] != report["image_id"]: raise ValueError("Diagnostic image differs from explicit pins")
        report.update(provenance=before, source_helpers=helpers, versions={"numpy": np.__version__, "trimesh": importlib.metadata.version("trimesh")}, phase="precision_measurement")
        report["real_loader_selftest"] = selftest()
        report["measurement"] = measure(paths["glb"]); report.update(status="measurement_complete", phase="complete")
    except Exception as error: report.update(error_type=type(error).__name__, error=str(error))
    finally:
        try:
            if before is not None and bindings(root, episode, code / PIN_FILE)[1] != before: raise ValueError("Original input/provenance changed")
            if helpers is not None and source_helpers() != helpers: raise ValueError("Diagnostic helpers changed")
            report["inputs_sources_rehashed_after"] = before is not None and helpers is not None
            report["historical_failure_unchanged"] = before is not None
        except Exception as error: report.update(status="fail", error_type=type(error).__name__, error=str(error))
        report["elapsed_seconds"] = time.perf_counter() - start
        if report["elapsed_seconds"] > BUDGET: report.update(status="fail", error_type="TimeoutError", error="Whole precision diagnostic exceeded60s")
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term)
        with (output / "report.json").open("x") as handle: json.dump(report, handle, sort_keys=True, allow_nan=False); handle.write("\n")
        (output / "report.json").chmod(0o444)
    print(json.dumps({k: report[k] for k in ("stage", "status", "episode_index", "elapsed_seconds")}))
    if report["status"] != "measurement_complete": raise RuntimeError("Precision diagnostic failed; sealed receipt retained")


if __name__ == "__main__": main()
