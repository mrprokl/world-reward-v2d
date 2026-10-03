"""One Azure-only author reference manufacture, not a prediction or quality gain.

Extract the complete rest surface once, then use identical topology, fixed rest
weights and identity at three predetermined states. The unchanged conservative
whole-surface certificate decides; failure stops before saving any geometry or
making RGB. No external assets, native human model, fit, mesh repair or sweep.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import time

import numpy as np

from world_reward import author_human_field as author
from world_reward.cross_surface import audit_closed_surface


ROOT = Path("/srv/scenesmith/world-reward")
OUTPUT = "validation/author_human_feasibility_v1"
IMAGE = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
BUDGET_SECONDS = 1200
HELPERS = ("infra/author_human_feasibility.py", "infra/run_author_human_feasibility.sh",
    "src/world_reward/author_human_field.py", "src/world_reward/cross_surface.py",
    "src/world_reward/__init__.py")
CROSS_SURFACE_SHA = "d808600c4a71d7ad19b1054bbb5be6b2cb079a1d4a37974725b99412fad2f47f"
MESHING_SOURCE_PINS = {
    "skimage/measure/_marching_cubes_lewiner.py": dict(bytes=12872,
        sha256="f482cdb5c9996c1a465a1fc84ffea9f88eec012200b61ca67e8ac7eab964f7fa"),
    "scikit_image-0.26.0.dist-info/LICENSE.txt": dict(bytes=6435,
        sha256="611d3207504dcb4a808df209d2e7c9b4f2e89fe690f84735c0d548160966b527"),
}


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + repr(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def file_identity(path, *, immutable=True):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Canonical unsymlinked source required")
    before = path.stat()
    if (not stat.S_ISREG(before.st_mode) or before.st_size <= 0
            or immutable and before.st_mode & 0o222):
        raise ValueError("Nonempty immutable regular source required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_mode) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_mode):
        raise ValueError("Source changed during hash")
    return dict(sha256=digest.hexdigest(), bytes=after.st_size)


def source_snapshot(root, code, revision):
    root, code = Path(root), Path(code)
    if (not re.fullmatch(r"[0-9a-f]{40}", revision)
            or code != root / "jobs" / revision / "run_author_human_feasibility" / "code"
            or not code.is_dir() or code.resolve() != code
            or any(p.is_symlink() for p in (code, *code.parents))):
        raise ValueError("Exact immutable author feasibility code namespace required")
    for path in (code, *code.rglob("*")):
        if path.is_symlink() or path.stat().st_mode & 0o222 or not (
                path.is_dir() or stat.S_ISREG(path.stat().st_mode)):
            raise ValueError("Complete dispatched code must remain readonly")
    helpers = {name: file_identity(code / name) for name in HELPERS}
    if helpers["src/world_reward/cross_surface.py"]["sha256"] != CROSS_SURFACE_SHA:
        raise ValueError("Original whole-surface certificate must not change")
    markers = {name: file_identity(code.parent / name, immutable=False)
        for name in ("revision", "source-sha256")}
    if (code.parent / "revision").read_bytes() != (revision + "\n").encode():
        raise ValueError("Actual producer revision marker differs")
    archive = (code.parent / "source-sha256").read_bytes()
    if not re.fullmatch(b"[0-9a-f]{64}\n", archive):
        raise ValueError("Actual dispatch archive marker required")
    return dict(helpers=helpers, markers=markers, archive_sha256=archive.decode().strip())


def require_certificate(certificate, vertices, faces):
    """Fail closed on the original complete certificate, never a sampled check."""
    if type(certificate) is not dict:
        raise ValueError("Original full surface certificate required")
    expected = dict(schema="world-reward-own-closed-surface-certificate-v1", status="pass",
        tolerance_m=1e-8, max_candidate_pairs=2_000_000, budget_seconds=30.,
        full_original_faces_retained=True, exact_arithmetic_proof=False)
    if any(type(certificate.get(k)) is not type(v) or certificate[k] != v for k, v in expected.items()):
        raise ValueError("Full original author surface certificate failed; no repair/retry")
    mesh, embedding = certificate.get("mesh"), certificate.get("embedding")
    if type(mesh) is not dict or type(embedding) is not dict:
        raise ValueError("Complete original mesh and embedding records required")
    required = dict(vertices=len(vertices), faces=len(faces), components=1,
        original_face_coverage=len(faces), topology_closed_oriented=True, embedding_verified=True,
        original_vertices_sha256=array_hash(vertices), original_faces_sha256=array_hash(faces))
    if any(type(mesh.get(k)) is not type(v) or mesh[k] != v for k, v in required.items()):
        raise ValueError("Complete single connected original author surface required")
    if (embedding.get("verified") is not True or embedding.get("forbidden_intersections") != []
            or embedding.get("nested_or_ambiguous_components") != []):
        raise ValueError("Full original embedding failed; no repair/retry")


def meshing_source_snapshot():
    distribution = importlib.metadata.distribution("scikit-image")
    if distribution.version != "0.26.0":
        raise ValueError("Exact installed and primary-audited meshing distribution required")
    result = {name: file_identity(distribution.locate_file(name), immutable=False)
        for name in MESHING_SOURCE_PINS}
    if result != MESHING_SOURCE_PINS:
        raise ValueError("Exact audited Lewiner source and BSD3 license required")
    return result


def author_semantics(reference):
    """Coverage sanity for every authored primitive/digit, not an anatomy proof."""
    rig, vertices, weights = reference.rig, reference.rest_vertices, reference.weights
    if len(author.FINGERTIP_NAMES) != 10 or not set(author.FINGERTIP_NAMES) <= set(rig.names):
        raise ValueError("All ten named original fingertip endpoints required")
    points = np.asarray([p.end for p in rig.primitives], dtype=np.float64)
    if not np.all(author.evaluate_field(points, rig) < 0):
        raise ValueError("Every declared author primitive endpoint must be inside the original field")
    endpoint_records = []
    for primitive, point in zip(rig.primitives, points):
        distance = float(np.linalg.norm(vertices - point, axis=1).min())
        limit = max(primitive.radii) + 2 * author.RESOLUTION_M
        if distance > limit:
            raise ValueError("Original surface lacks declared primitive support; no repair/resweep")
        endpoint_records.append(dict(name=primitive.name, nearest_vertex_distance_m=distance,
            maximum_distance_m=limit))
    digit_records = []
    for name in author.FINGERTIP_NAMES:
        tip = rig.rest_joints[rig.names.index(name)]
        bone = name.replace("_tip", "_dip")
        maximum_weight = float(weights[:, rig.names.index(bone)].max())
        distance = float(np.linalg.norm(vertices - tip, axis=1).min())
        radius = max(max(p.radii) for p in rig.primitives if p.bone == bone)
        if maximum_weight <= .1 or distance > radius + 2 * author.RESOLUTION_M:
            raise ValueError("All ten original digits require surface/rig support; no repair/resweep")
        digit_records.append(dict(name=name, maximum_distal_rest_weight=maximum_weight,
            nearest_vertex_distance_m=distance))
    return dict(primitive_endpoint_checks=endpoint_records, ten_digit_checks=digit_records,
        complete_anatomical_geometry_proof=False)


def manufacture(report, *, deadline):
    if author.RESOLUTION_M != .012 or tuple(author.POSE_NAMES) != ("rest", "carry", "fingerbend"):
        raise ValueError("One frozen resolution and three exact predetermined states required")
    rig = author.author_rig()
    report["phase"] = "extract_original_author_rest_surface"
    reference = author.extract_rest_surface(rig, deadline=deadline)
    rest = np.asarray(reference.rest_vertices)
    faces = np.asarray(reference.faces)
    weights = np.asarray(reference.weights)
    if (rest.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or rest.ndim != 2 or rest.shape[1:] != (3,) or len(rest) < 4
            or not np.isfinite(rest).all() or faces.dtype.kind not in "iu"
            or faces.ndim != 2 or faces.shape[1:] != (3,)
            or weights.shape != (len(rest), len(rig.names))
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError("Complete finite rest geometry and normalized frozen author weights required")
    rest = rest.copy(); faces = faces.copy(); weights = weights.copy()
    rest_hash, face_hash, weight_hash = map(array_hash, (rest, faces, weights))
    rig_hashes = dict(rest_joints=array_hash(rig.rest_joints), parents=array_hash(rig.parents),
        local_rotations=array_hash(rig.local_rotations))
    rig_names = tuple(rig.names)
    report["reference"] = dict(vertices=len(rest), faces=len(faces), joint_names=list(rig.names),
        rest_vertices_sha256=rest_hash, original_faces_sha256=face_hash,
        fixed_weights_sha256=weight_hash, rig_hashes=rig_hashes)
    report["phase"] = "original_author_primitive_digit_support"
    report["reference_support"] = author_semantics(reference)
    report["states"] = []
    states, joint_states = [], []
    for name in author.POSE_NAMES:
        if time.monotonic() >= deadline:
            raise TimeoutError("Original author manufacture overall deadline exceeded")
        report["phase"] = "certify_original_author_" + name
        vertices, returned_faces, joints = author.deform_reference(reference, name)
        vertices, returned_faces, joints = map(np.asarray, (vertices, returned_faces, joints))
        if (array_hash(reference.faces) != face_hash
                or tuple(reference.rig.names) != rig_names or tuple(rig.names) != rig_names):
            raise ValueError("Original reference faces or joint semantics changed")
        if (vertices.shape != rest.shape or vertices.dtype != rest.dtype or not np.isfinite(vertices).all()
                or joints.shape != rig.rest_joints.shape or not np.isfinite(joints).all()
                or array_hash(returned_faces) != face_hash
                or array_hash(reference.rest_vertices) != rest_hash
                or array_hash(reference.weights) != weight_hash
                or dict(rest_joints=array_hash(rig.rest_joints), parents=array_hash(rig.parents),
                    local_rotations=array_hash(rig.local_rotations)) != rig_hashes):
            raise ValueError("Original identity, topology, rig or frozen weights changed")
        if name == "rest" and (not np.array_equal(vertices, rest) or not np.array_equal(joints, rig.rest_joints)):
            raise ValueError("Rest replay must preserve original exact geometry and joints")
        certificate = audit_closed_surface(vertices, returned_faces, tolerance_m=1e-8,
            max_candidate_pairs=2_000_000, budget_seconds=30.)
        report["states"].append(dict(name=name, surface_certificate=certificate,
            joints_sha256=array_hash(joints)))
        require_certificate(certificate, vertices, returned_faces)
        states.append(vertices.copy()); joint_states.append(joints.copy())
    if time.monotonic() >= deadline:
        raise TimeoutError("Original author manufacture overall deadline exceeded")
    return dict(rest_vertices=rest, faces=faces, weights=weights,
        rest_joints=rig.rest_joints, parents=rig.parents, local_rotations=rig.local_rotations,
        vertices=np.stack(states), joints=np.stack(joint_states))


def write_private_report(path, report):
    data = json.dumps(report, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def main():
    if (platform.system() != "Linux" or os.environ.get("WR_ROOT") != str(ROOT)
            or os.environ.get("WR_IMAGE_ID") != IMAGE):
        raise RuntimeError("Author manufacture restricted to exact Azure CPU runtime")
    code, revision = Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    sources = source_snapshot(ROOT, code, revision)
    out = ROOT / OUTPUT
    if (not out.is_dir() or out.resolve() != out or any(p.is_symlink() for p in (out, *out.parents))
            or tuple(out.iterdir()) or out.stat().st_mode & 0o077):
        raise ValueError("Fresh private empty output required")
    started = time.monotonic()
    report = dict(schema="world-reward-author-human-feasibility-v1", status="fail",
        phase="runtime", producer_revision=revision, source_snapshot=sources, image_id=IMAGE,
        network="none", GPU_used=False, budget_seconds=BUDGET_SECONDS, resolution_m=.012,
        own_fresh_procedural_geometry_only=True, challenge_inputs_used=False, ground_truth_read=False,
        historical_poses_results_read=False, hand_labeled_test=False, native_model_used=False,
        mesh_repair=False, component_deletion=False, resolution_sweep=False, RGB_produced=False,
        quality_verified=False, challenge_performance_verified=False, submission_eligible=False,
        force_closure_verified=False, touching_certified=False, biomechanics_verified=False,
        adoption_authorized=False)
    def timeout_handler(signum, frame):
        raise TimeoutError("Original author manufacture overall deadline exceeded")
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(BUDGET_SECONDS)
    try:
        report["runtime"] = {name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-image")}
        if report["runtime"] != {"numpy": "1.26.3", "scipy": "1.16.3", "scikit-image": "0.26.0"}:
            raise ValueError("Actual audited exact numeric and meshing runtime required")
        report["meshing_sources"] = meshing_source_snapshot()
        arrays = manufacture(report, deadline=started + BUDGET_SECONDS)
        if (source_snapshot(ROOT, code, revision) != sources
                or meshing_source_snapshot() != report["meshing_sources"]):
            raise ValueError("Original source or dispatch markers changed")
        report["phase"] = "save_original_certified_author_geometry"
        fd = os.open(out / "geometry.npz", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        report["private_geometry"] = file_identity(out / "geometry.npz")
        report["status"] = "pass"
        report["phase"] = "done"
    except Exception as exc:
        report["status"] = "fail"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
    # A filesystem/provenance failure must never be hidden by a geometric PASS.
    if source_snapshot(ROOT, code, revision) != sources:
        report.update(status="fail", error_type="ValueError", error="Original immutable source changed")
    report["elapsed_seconds"] = time.monotonic() - started
    write_private_report(out / "report.json", report)
    print(json.dumps(dict(status=report["status"], phase=report["phase"],
        elapsed_seconds=report["elapsed_seconds"]), separators=(",", ":")), flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
