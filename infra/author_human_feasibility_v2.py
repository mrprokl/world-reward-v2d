"""Correct a buried-primitive measurement, never alter the failed reference.

V1 stays FAIL. Exactly the same extracted geometry, rig, weights and poses must
match its independently pinned receipt before the original surface certificate.
Only after that certificate do full continuous solid queries replace the invalid
assumption that every primitive endpoint is near the exposed union boundary.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np

import author_human_feasibility as original
from world_reward import author_human_field as author
from world_reward.cross_surface import audit_closed_surface
from world_reward.solid_queries import point_surface_query

ROOT = original.ROOT
OUTPUT = "validation/author_human_feasibility_v2"
ORIGINAL_OUTPUT = "validation/author_human_feasibility_v1/report.json"
ORIGINAL_RECEIPT = dict(bytes=3655, sha256="eef678c211714894934d6c0715efed2829cd809999c03e0225f59a9d6021f4c0")
ORIGINAL_REVISION = "00074fe8d7585219bdeb952cd305c466631c6932"
REFERENCE_PINS = dict(vertices=19982, faces=39960,
    rest_vertices_sha256="02aad0054fb20ef133910546127ddc363ab2de30db3f4b255cc07a64f0595246",
    original_faces_sha256="0806cae2669f813853ecac4f7eb22cf95ae228a33910dd502f31a2eb025d2ac7",
    fixed_weights_sha256="6e6b8913bebb574792703e27da44e95e32613d39b14b54c9bc4d07ee94e9f71b",
    rig_hashes=dict(rest_joints="820d59cf16b14b336c31a1f91fd718132c4622a758fcf4b9f13a3caa45782f60",
        parents="447003860ed6f1006f411453d0b16fcf226f5e91d439549f8872e1dfbc9b99f7",
        local_rotations="3e4f673c5cbb48bfca44749d5475a8c163661b26612c145c25d3a326c4a66847"))
ORIGINAL_SOURCES = {
    "infra/author_human_feasibility.py": dict(bytes=16260, sha256="3e078ff31df475087305540911793b91648482cd2870bfd37061feff799922bf"),
    "infra/run_author_human_feasibility.sh": dict(bytes=4804, sha256="232b9550d851f513927960bd72fa0cc67de34d5bcb4bb6305a5fd07c5ad2eb81"),
    "src/world_reward/author_human_field.py": dict(bytes=18215, sha256="385e9c6a3e98b43cccdca140be8215de64e243f7c0a73bff54339e4fa3b06423"),
    "src/world_reward/cross_surface.py": dict(bytes=25472, sha256=original.CROSS_SURFACE_SHA),
    "src/world_reward/__init__.py": dict(bytes=81, sha256="1ac8c64041277f0c3cf4d8838b3153210d7076dd86a5121d1e6ef494b64b36a1"),
}
HELPERS = (*ORIGINAL_SOURCES, "infra/author_human_feasibility_v2.py",
    "infra/run_author_human_feasibility_v2.sh", "src/world_reward/solid_queries.py")


def original_report(path):
    if original.file_identity(path) != ORIGINAL_RECEIPT:
        raise ValueError("Actual immutable original failed receipt required before parsing")
    d = json.loads(Path(path).read_text())
    expected = dict(schema="world-reward-author-human-feasibility-v1", status="fail",
        phase="original_author_primitive_digit_support", producer_revision=ORIGINAL_REVISION,
        error_type="ValueError", error="Original surface lacks declared primitive support; no repair/resweep",
        image_id=original.IMAGE, resolution_m=.012, budget_seconds=1200, RGB_produced=False,
        own_fresh_procedural_geometry_only=True, challenge_inputs_used=False, ground_truth_read=False,
        mesh_repair=False, component_deletion=False, resolution_sweep=False, quality_verified=False)
    if (type(d) is not dict or any(type(d.get(k)) is not type(v) or d[k] != v for k, v in expected.items())
            or any(k in d for k in ("states", "private_geometry", "reference_support"))
            or d.get("source_snapshot", {}).get("helpers") != ORIGINAL_SOURCES):
        raise ValueError("Original failed support protocol and complete source identity required")
    reference = d.get("reference", {})
    if (any(reference.get(k) != v for k, v in REFERENCE_PINS.items())
            or reference.get("joint_names") != list(author.author_rig().names)):
        raise ValueError("Actual original complete author geometry and rig receipt required")
    if original.file_identity(path) != ORIGINAL_RECEIPT:
        raise ValueError("Original failed receipt changed")
    return d


def source_snapshot(code, revision):
    code = Path(code)
    if (not re.fullmatch(r"[0-9a-f]{40}", revision)
            or code != ROOT / "jobs" / revision / "run_author_human_feasibility_v2" / "code"
            or code.resolve() != code or not code.is_dir()):
        raise ValueError("Exact separate immutable v2 namespace required")
    helpers = {name: original.file_identity(code / name) for name in HELPERS}
    if any(helpers[k] != v for k, v in ORIGINAL_SOURCES.items()):
        raise ValueError("Original field, geometry manufacture, poses and certificate must not change")
    markers = {name: original.file_identity(code.parent / name, immutable=False)
        for name in ("revision", "source-sha256")}
    if ((code.parent / "revision").read_text() != revision + "\n"
            or not re.fullmatch(r"[0-9a-f]{64}\n", (code.parent / "source-sha256").read_text())):
        raise ValueError("Actual source dispatch markers required")
    return dict(helpers=helpers, markers=markers)


def reference_identity(ref):
    return dict(vertices=len(ref.rest_vertices), faces=len(ref.faces),
        rest_vertices_sha256=original.array_hash(ref.rest_vertices),
        original_faces_sha256=original.array_hash(ref.faces),
        fixed_weights_sha256=original.array_hash(ref.weights),
        rig_hashes={k: original.array_hash(getattr(ref.rig, k))
            for k in ("rest_joints", "parents", "local_rotations")})


def solid_support(ref, *, deadline):
    """67 fixed complete surface queries, only after the original rest certificate."""
    rig, v, f = ref.rig, ref.rest_vertices, ref.faces
    queries = []
    def check(name, point, state):
        q = point_surface_query(np.asarray(point, np.float64), v, f,
            expected_state=state, tolerance_m=1e-8, deadline=deadline)
        queries.append(dict(name=name, expected_state=state, query=q))
        if q["status"] != "pass":
            raise ValueError("Certified original author solid support query failed: " + name)
    for p in rig.primitives:
        check(p.name + "_endpoint", p.end, "inside")
    for name in author.FINGERTIP_NAMES:
        tip = rig.rest_joints[rig.names.index(name)]
        bone = name.replace("_tip", "_dip")
        if float(ref.weights[:, rig.names.index(bone)].max()) <= .1:
            raise ValueError("Original exposed digit requires nontrivial fixed distal weight")
        radius = max(max(p.radii) for p in rig.primitives if p.bone == bone)
        probe = tip + np.array([0., 0., radius + 2 * author.RESOLUTION_M])
        if author.evaluate_field(probe[None], rig)[0] <= 0:
            raise ValueError("Predeclared analytic fingertip bracket is not exterior")
        check(name + "_anterior_exposure", probe, "outside")
    for side in ("l", "r"):
        for first, last in zip(author.FINGER_NAMES[:-1], author.FINGER_NAMES[1:]):
            a, b = (rig.rest_joints[rig.names.index(f"{side}_{n}_tip")] for n in (first, last))
            point = (a + b) / 2
            if author.evaluate_field(point[None], rig)[0] <= 0:
                raise ValueError("Predeclared analytic digit gap must be exterior")
            check(side + "_" + first + "_" + last + "_gap", point, "outside")
    if len(queries) != 67:
        raise ValueError("All original primitive endpoints/exposure brackets/digit gaps required")
    return dict(queries=queries, query_count=67, continuous_distance_to_all_original_faces=True,
        anatomical_accuracy_verified=False, embedding_checked_separately=True)


def manufacture(report, *, deadline):
    if author.RESOLUTION_M != .012 or author.POSE_NAMES != ("rest", "carry", "fingerbend"):
        raise ValueError("Original single resolution and fixed three poses required")
    report["phase"] = "hash_identical_original_author_rest"
    ref = author.extract_rest_surface(author.author_rig(), deadline=deadline)
    identity = reference_identity(ref)
    report["reference"] = identity
    if identity != REFERENCE_PINS:
        raise ValueError("Re-extracted full reference must match original v1 V/F/weights/rig exactly")
    report["original_reference_byte_identical"] = True
    report["states"] = []; vertices, joints = [], []
    for name in author.POSE_NAMES:
        if time.monotonic() >= deadline:
            raise TimeoutError("Original v2 author overall deadline exceeded")
        report["phase"] = "original_full_surface_" + name
        v, f, j = author.deform_reference(ref, name)
        if (reference_identity(ref) != identity or original.array_hash(f) != identity["original_faces_sha256"]
                or v.dtype != ref.rest_vertices.dtype or v.shape != ref.rest_vertices.shape
                or not np.isfinite(v).all() or j.shape != ref.rig.rest_joints.shape or not np.isfinite(j).all()):
            raise ValueError("Complete original reference identity/topology/rig changed")
        if name == "rest" and (not np.array_equal(v, ref.rest_vertices) or not np.array_equal(j, ref.rig.rest_joints)):
            raise ValueError("Original rest replay must be byte-exact")
        c = audit_closed_surface(v, f, tolerance_m=1e-8, max_candidate_pairs=2_000_000, budget_seconds=30.)
        report["states"].append(dict(name=name, surface_certificate=c))
        original.require_certificate(c, v, f)
        if name == "rest":
            report["phase"] = "corrected_original_rest_solid_support"
            report["reference_support"] = solid_support(ref, deadline=deadline)
        vertices.append(v.copy()); joints.append(j.copy())
    return dict(rest_vertices=ref.rest_vertices, faces=ref.faces, weights=ref.weights,
        rest_joints=ref.rig.rest_joints, parents=ref.rig.parents, local_rotations=ref.rig.local_rotations,
        vertices=np.stack(vertices), joints=np.stack(joints))


def main():
    if platform.system() != "Linux" or os.environ.get("WR_ROOT") != str(ROOT) or os.environ.get("WR_IMAGE_ID") != original.IMAGE:
        raise RuntimeError("Corrected author measurement remains exact Azure CPU-only")
    code, revision = Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    sources = source_snapshot(code, revision); path = ROOT / ORIGINAL_OUTPUT; original_report(path)
    out = ROOT / OUTPUT
    if (not out.is_dir() or out.resolve() != out or any(p.is_symlink() for p in (out, *out.parents))
            or tuple(out.iterdir()) or out.stat().st_mode & 0o077):
        raise ValueError("Exclusive separate private v2 output required")
    started = time.monotonic()
    report = dict(schema="world-reward-author-human-feasibility-v2", status="fail", phase="runtime",
        producer_revision=revision, source_snapshot=sources, original_failed_receipt=ORIGINAL_RECEIPT,
        image_id=original.IMAGE, network="none", GPU_used=False, budget_seconds=1200, resolution_m=.012,
        correction="buried_endpoint_containment_not_exposed_boundary_proximity", original_v1_status="fail",
        own_fresh_procedural_geometry_only=True, mesh_repair=False, component_deletion=False,
        geometry_retuned=False, poses_retuned=False, resolution_sweep=False, challenge_inputs_used=False,
        ground_truth_read=False, hand_labeled_test=False, RGB_produced=False, quality_verified=False,
        anatomical_accuracy_verified=False, motion_verified=False, force_closure_verified=False,
        touching_certified=False, challenge_performance_verified=False, submission_eligible=False,
        adoption_authorized=False)
    def expired(*_): raise TimeoutError("Corrected original author overall20min budget exceeded")
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(1200)
    try:
        report["runtime"] = {name: original.importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-image")}
        if report["runtime"] != {"numpy": "1.26.3", "scipy": "1.16.3", "scikit-image": "0.26.0"}:
            raise ValueError("Actual original numeric and meshing runtime must remain unchanged")
        report["meshing_sources"] = original.meshing_source_snapshot()
        arrays = manufacture(report, deadline=started + 1200)
        if (source_snapshot(code, revision) != sources or original.file_identity(path) != ORIGINAL_RECEIPT
                or original.meshing_source_snapshot() != report["meshing_sources"]):
            raise ValueError("Original source or failed receipt changed")
        report["phase"] = "save_identical_certified_reference"
        fd = os.open(out / "geometry.npz", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(fd, "wb") as stream: np.savez_compressed(stream, **arrays)
        report["private_geometry"] = original.file_identity(out / "geometry.npz")
        report.update(status="pass", phase="complete")
    except Exception as exc:
        report.update(status="fail", error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
    if source_snapshot(code, revision) != sources or original.file_identity(path) != ORIGINAL_RECEIPT:
        report.update(status="fail", error_type="ValueError", error="Original source or failed receipt changed")
    report["elapsed_seconds"] = time.monotonic() - started
    original.write_private_report(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "phase", "elapsed_seconds")}, separators=(",", ":")), flush=True)
    if report["status"] != "pass": raise SystemExit(1)


if __name__ == "__main__": main()
