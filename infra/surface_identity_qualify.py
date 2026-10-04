"""Data-free open-surface identity/real loader/official budget qualification only.

No QEM, body construction, rendering, dataset, model, metric gauge or adoption.
The three positive sources and negative domains are fixed before runtime calls.
Host provenance is stdlib-only; numerical/native imports happen offline on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path("/srv/scenesmith/world-reward")
ENTRY = "run_surface_identity_qualify"
STAGE = "world_reward_surface_identity_runtime_qualification_v1"
BUDGET = 60
OUTER_SECONDS = 63
IMAGE = "sha256:1a04b1930f713ef9ffb411489e80ddebbce59a5ce26e713add4095cd9b5303f0"
NAMES = ("curved_sheet", "open_square_tube", "sheet_and_isolated_triangle")
HELPERS = ("infra/surface_identity_qualify.py", "infra/run_surface_identity_qualify.sh",
    "infra/mediapipe_cpu_runtime_verify.py", "infra/official_track1_pack_gate.py",
    "infra/official_pack_geometry.py", "src/world_reward/surface_identity.py",
    "src/world_reward/raw_shape_proposal.py", "src/world_reward/exact_triangle_predicates.py",
    "configs/official_pack_runtime_pins.json", "infra/official_pack_image_build.py",
    "infra/run_official_pack_image_build.sh")


def modules():
    import mediapipe_cpu_runtime_verify as rt
    import official_track1_pack_gate as official
    return rt, official


def environment():
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision = os.environ["WR_CODE_REVISION"]
    if (root != ROOT or not re.fullmatch("[0-9a-f]{40}", revision)
            or code != root / "jobs" / revision / ENTRY / "code"
            or Path(__file__).resolve() != code / HELPERS[0]):
        raise ValueError("Exact actual immutable qualification namespace required")
    return root, code, revision


def output(root, revision):
    return root / "results" / ("surface-identity-qualify-" + revision)


def host_proof(root, code, revision):
    rt, official = modules()
    binding = rt.source(root, code, revision, ENTRY, HELPERS)
    runtime = official.load_runtime(root, code)
    rt.require(runtime["pins"]["image_id"] == IMAGE, "Independent actual official CPU image required")
    sources = official.official_sources(root)
    _, native = official.native_mesh_sources(root)
    # Historical native/kit permissions can be writable; mounts are RO and hashes
    # stable pre/post. Only our source and independently sealed receipt require RO.
    leaves = {path: rt.identity(path, 2 << 20, readonly=False)
              for path in mount_paths(root, code) if path != code.parent}
    return dict(source_binding=binding, runtime=runtime, official_sources=sources,
                native_mesh_sources=native, leaf_identities={str(p): pin for p, pin in leaves.items()})


def mount_paths(root, code):
    _, official = modules()
    native = root / "vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d"
    return [code.parent, root / official.BUILD_RECEIPT,
            *(root / "vendor/v2d_submission_kit" / name for name in official.OFFICIAL_SOURCES),
            *(native / name for name in official.NATIVE_MESH_SOURCES)]


def fixtures(np):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, .25], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    tube = np.array([[x, y, z] for z in (0, 1)
                     for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))], np.float32)
    tube_faces = np.array([face for i in range(4) for face in (
        (i, (i + 1) % 4, (i + 1) % 4 + 4), (i, (i + 1) % 4 + 4, i + 4))], np.int64)
    combined = np.vstack((v, np.array([[4, 0, 0], [5, 0, 0], [4, 1, 0]], np.float32)))
    combined_faces = np.vstack((f, [[4, 5, 6]])).astype(np.int64)
    return tuple(zip(NAMES, ((v, f), (tube, tube_faces), (combined, combined_faces))))


def surface_signature(surface):
    import numpy as np
    from official_pack_geometry import canonical_oriented_triangles
    components = []
    for index in range(len(surface.component_keys)):
        triangles = canonical_oriented_triangles(surface.vertices, surface.faces[surface.face_components == index])
        loops = []
        for loop, component in zip(surface.boundary_loops, surface.boundary_components):
            if component != index:
                continue
            xyz = surface.vertices[list(loop)].astype(np.float64)
            rotations = [np.roll(xyz, -i, axis=0) for i in range(len(xyz))]
            canonical = min(rotations, key=lambda row: tuple(row.ravel()))
            loops.append(hashlib.sha256(canonical.tobytes()).hexdigest())
        components.append((hashlib.sha256(triangles.tobytes()).hexdigest(), tuple(sorted(loops))))
    return tuple(sorted(components))


def _array_manifest(rows):
    result = []
    for name, v, f in rows:
        result.append(dict(name=name, arrays=[dict(dtype=a.dtype.str, shape=list(a.shape),
            bytes=a.nbytes, sha256=hashlib.sha256(a.tobytes()).hexdigest()) for a in (v, f)]))
    return result


def controls(np, trimesh, budget_mesh, load_native, authorities, scratch, remaining, progress=None):
    rows = fixtures(np)
    negatives = (("invalid_index", rows[0][1][0], np.array([[0, 1, 99]], np.int64)),
        ("collinear", np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32), np.array([[0, 1, 2]], np.int64)),
        ("nonmanifold_edge", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], np.float32),
         np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], np.int64)))
    # Freeze ALL six complete source arrays before the first domain/predicate/
    # loader measurement. Immutable bytes backing cannot be made writable.
    frozen = [(name, *(np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape) for a in (v, f)))
              for name, v, f in (*[(n, *pair) for n, pair in rows], *negatives)]
    manifest = _array_manifest(frozen)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    proof = {} if progress is None else progress
    proof.update(fixture_manifest=manifest, fixture_manifest_sha256=digest,
                 all_fixture_sources_frozen_before_measurement=True)
    try:
        return _controls(np, trimesh, budget_mesh, load_native, authorities, scratch, remaining,
                         frozen[:3], frozen[3:], proof)
    finally:
        if _array_manifest(frozen) != manifest:
            raise ValueError("Complete fixed fixture source changed")
        proof["all_fixture_sources_rehashed_after"] = True


def _controls(np, trimesh, budget_mesh, load_native, authorities, scratch, remaining, rows, negatives, progress):
    from official_pack_geometry import (canonical_oriented_triangles, nonzero_triangle_mask,
                                        verify_exact_dual_surfaces)
    from world_reward.surface_identity import SurfaceIdentity
    rejected = []
    for name, v, f in negatives:
        before = v.tobytes(), f.tobytes()
        try:
            SurfaceIdentity(v, f)
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError("Negative surface domain unexpectedly accepted")
        if before != (v.tobytes(), f.tobytes()):
            raise ValueError("Negative inputs mutated")
    results, counts = [], dict(exports=0, native_loader_attempts=0, native_loader_returns=0,
        authority_attempts=0, authority_returns=0, budget_attempts=0, budget_returns=0)
    if progress is not None:
        progress.update(records=results, rejected_domain_controls=rejected, counts=counts, QEM_calls=0)
    for name, v, f in rows:
        progress.update(current_case=name, current_phase="source_predicate")
        remaining()
        source = SurfaceIdentity(v, f)
        before = v.tobytes(), f.tobytes()
        path = scratch / (name + ".glb")
        if path.exists():
            raise ValueError("Exclusive manufactured scratch path required")
        trimesh.Trimesh(vertices=v.copy(), faces=f.copy(), process=False).export(path)
        path.chmod(0o444)  # Only this newly created owned fixture; original authority requires immutable GLB.
        counts["exports"] += 1
        rt, _ = modules()
        glb_id = rt.identity(path, 1 << 20, readonly=False)
        counts["native_loader_attempts"] += 1
        progress["current_phase"] = "native_loader"
        nv, nf = load_native(path)
        counts["native_loader_returns"] += 1
        if (not isinstance(nv, np.ndarray) or not isinstance(nf, np.ndarray)
                or nv.dtype != np.float32 or nf.dtype != np.int64):
            raise ValueError("Actual native loader must return F32 vertices/I64 faces")
        native = SurfaceIdentity(nv, nf)
        counts["authority_attempts"] += 1
        progress["current_phase"] = "full_scene_authorities"
        rv, rf, proof = authorities(path, SimpleNamespace(object_vertices=nv, object_faces=nf))
        counts["authority_returns"] += 1
        if not (proof.get("original_native_FP32_loader_replayed") is True
                and proof.get("native_geometry_byte_exact") is True
                and proof.get("full_scene_instances_verified") is True and proof.get("model_imports") is False):
            raise ValueError("Actual native/full-scene authority replay required")
        counts["budget_attempts"] += 1
        progress["current_phase"] = "official_budget_mesh"
        pv, pf = budget_mesh(str(path), faces=4096, vertices=4096)
        counts["budget_returns"] += 1
        if pv.shape != (4096, 3) or pf.shape != (4096, 3):
            raise ValueError("Exact official4096-row budgets required")
        exact = verify_exact_dual_surfaces(nv, nf, rv, rf, pv, pf)
        if not np.array_equal(canonical_oriented_triangles(v, f), canonical_oriented_triangles(nv, nf)):
            raise ValueError("Export/native loader changed original manufactured surface")
        active = nonzero_triangle_mask(pv, pf)
        if not np.all(pf[~active] == 0):
            raise ValueError("Only official all-zero padding is excluded from diagnostic view")
        ids, inverse = np.unique(pf[active], return_inverse=True)
        packed = SurfaceIdentity(pv[ids], inverse.reshape(-1, 3).astype(np.int64))
        if (surface_signature(source) != surface_signature(native)
                or surface_signature(source) != surface_signature(packed)):
            raise ValueError("Component/boundary surface changed during identity packing")
        if (before != (v.tobytes(), f.tobytes()) or rt.identity(path, 1 << 20, readonly=False) != glb_id):
            raise ValueError("Manufactured source/GLB changed")
        remaining()
        results.append(dict(name=name, source_vertices=len(v), source_faces=len(f),
            components=len(source.component_keys), boundary_loops=len(source.boundary_loops),
            GLB_identity=glb_id, source_arrays=source.report()["vertices"],
            source_faces_identity=source.report()["faces"], exact_dual_surfaces=exact,
            meaningful_faces_preserved=True, component_boundary_geometry_exact=True,
            full_scene_instances=proof["scene_instances"], simplification_performed=False))
    progress.update(records=results, rejected_domain_controls=rejected, counts=counts,
        native_loader_calls_including_authority_replays=counts["native_loader_returns"] + counts["authority_returns"],
        native_counter_scope="direct_calls_plus_source_authenticated_one_call_authority_replays",
        under_budget_sources=True, QEM_calls=0, negative_GLBS_written=False, current_phase="complete")
    return progress


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o444)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def native(root, code, revision):
    started = time.monotonic()
    def remaining():
        if time.monotonic() - started >= BUDGET:
            raise TimeoutError("Inclusive60-second qualification exhausted")
    def timeout(*_):
        raise TimeoutError("Inclusive60-second qualification exhausted")
    old = signal.signal(signal.SIGALRM, timeout); term = signal.signal(signal.SIGTERM, timeout); signal.alarm(BUDGET)
    rt, official = modules(); out = output(root, revision)
    report = dict(stage=STAGE, status="fail", phase="preflight", producer_revision=revision,
        budget_seconds=BUDGET, GPU_used=False, datasets_read=False, models_read=False, private_values_read=False,
        body_model_constructed=False, reconstruction_accuracy_verified=False, adoption=False)
    before = None; scratch = None
    try:
        rt.require(sys.platform == "linux" and os.geteuid() == 1000
            and {p.name for p in Path("/sys/class/net").iterdir()} == {"lo"}
            and os.environ.get("WR_IMAGE_ID") == IMAGE
            and rt.canonical(out) == out and out.is_dir() and not list(out.iterdir())
            and out.stat().st_uid == 1000 and out.stat().st_mode & 0o777 == 0o755,
            "Restricted offline CPU/empty reserved output required")
        before = host_proof(root, code, revision); report["source_proof"] = before
        report["runtime_versions"] = official.check_runtime_packages(before["runtime"]["pins"]["versions"])
        import numpy as np
        import trimesh
        native_root, _ = official.native_mesh_sources(root)
        namespace = dict(np=np, trimesh=trimesh, Path=Path)
        namespace["load_object_mesh"] = official.isolated_source_function(
            native_root / "lib_mhr/contact.py", "load_object_mesh", namespace)
        load_native = official.isolated_source_function(native_root / "learning/training/mhr_opt_refineout.py",
                                                        "_load_object_vertices", namespace)
        # Only the verified original mesh-budget module is imported; no sample/
        # pack_track1 call, body decoder, native optimizer or model imports.
        kit = root / "vendor/v2d_submission_kit"
        if any(n == "v2dlb" or n.startswith("v2dlb.") for n in sys.modules):
            raise ValueError("Fresh original official namespace required")
        sys.path.insert(0, str(kit))
        from v2dlb.mesh_budget import budget_mesh
        if Path(budget_mesh.__code__.co_filename) != kit / "v2dlb/mesh_budget.py":
            raise ValueError("Actual original budget helper source required")
        report["phase"] = "controls"
        with tempfile.TemporaryDirectory(prefix="wr-surface-identity-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            report["controls"] = {}
            report["controls"] = controls(np, trimesh, budget_mesh, load_native,
                lambda path, ep: official.load_mesh_authorities(root, path, ep), scratch, remaining, report["controls"])
        report["owned_scratch_removed"] = not scratch.exists()
        official.check_runtime_packages(report["runtime_versions"])
        remaining()
        report.update(status="pass", phase="complete")
    except Exception as error:
        report.update(status="fail", error_type=type(error).__name__)
    finally:
        report["owned_scratch_removed"] = scratch is None or not scratch.exists()
        try:
            remaining()
            rt.require(before is not None and host_proof(root, code, revision) == before, "Source/runtime changed")
            report["source_runtime_rehashed_after"] = True
            remaining()
        except Exception as error:
            report.update(status="fail", postflight_error_type=type(error).__name__)
        report["elapsed_seconds"] = time.monotonic() - started
        if report["elapsed_seconds"] > BUDGET:
            report.update(status="fail", budget_exhausted=True)
        try:
            write(out / "native.json", report)
            remaining()  # Receipt serialization/write remains inside the active deadline.
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old); signal.signal(signal.SIGTERM, term)
    return 0 if report["status"] == "pass" else 1


def container_absence(root, revision, *, failed=False):
    rt, _ = modules(); path = Path(str(output(root, revision)) + ".container.cid")
    if failed and not path.exists() and not path.is_symlink():
        return dict(verified=False, CID_identity=None)
    pin = rt.identity(path, 65, readonly=False)
    rt.require(path.stat().st_uid == 0 and pin["bytes"] in (64, 65), "Actual host-root owned CID required")
    raw = path.read_bytes()
    rt.require(re.fullmatch(b"[0-9a-f]{64}\n?", raw), "Exact owned CID bytes required")
    cid = raw.decode().strip()
    result = subprocess.run(["docker", "inspect", cid, "--format", "{{.Id}}"], capture_output=True, timeout=5)
    errors = (f"Error: No such object: {cid}", f"Error: No such container: {cid}",
              f"Error response from daemon: No such container: {cid}")
    rt.require(result.returncode == 1 and not result.stdout.strip()
        and result.stderr.decode().strip() in errors and rt.identity(path, 65, readonly=False) == pin,
        "Independent exact owned-container absence required; daemon failure is not absence")
    return dict(verified=True, CID_identity=pin)


def seal(root, code, revision, before, status, *, cleanup_verified=False):
    rt, _ = modules(); out = output(root, revision)
    proof = rt.strict(before.encode()); after = host_proof(root, code, revision)
    rt.require(proof == after, "Complete original source/authorities changed")
    absence = container_absence(root, revision, failed=status != 0)
    raw = rt.strict((out / "native.json").read_bytes()) if (out / "native.json").exists() else None
    good = status == 0 and cleanup_verified is True and absence["verified"] is True and type(raw) is dict
    if good:
        expected = dict(stage=STAGE, status="pass", phase="complete", producer_revision=revision,
            source_proof=proof, source_runtime_rehashed_after=True, owned_scratch_removed=True,
            GPU_used=False, datasets_read=False, models_read=False, private_values_read=False,
            body_model_constructed=False, reconstruction_accuracy_verified=False, adoption=False)
        rt.require(all(type(raw.get(k)) is type(v) and raw[k] == v for k, v in expected.items()), "Actual complete native scope required")
        c = raw["controls"]
        rt.require([r["name"] for r in c["records"]] == list(NAMES)
            and c["counts"] == dict(exports=3, native_loader_attempts=3, native_loader_returns=3,
                authority_attempts=3, authority_returns=3, budget_attempts=3, budget_returns=3)
            and c["native_loader_calls_including_authority_replays"] == 6 and c["QEM_calls"] == 0
            and c["negative_GLBS_written"] is False and c["under_budget_sources"] is True
            and c["all_fixture_sources_frozen_before_measurement"] is True
            and c["all_fixture_sources_rehashed_after"] is True and len(c["fixture_manifest"]) == 6
            and c["rejected_domain_controls"] == ["invalid_index", "collinear", "nonmanifold_edge"]
            and type(raw["elapsed_seconds"]) in (int, float) and 0 < raw["elapsed_seconds"] <= BUDGET,
            "All fixed positives/negatives and real call counts required")
        rt.require(c["fixture_manifest_sha256"] == hashlib.sha256(json.dumps(c["fixture_manifest"], sort_keys=True).encode()).hexdigest()
            and all(r.get("component_boundary_geometry_exact") is True
                    and r.get("meaningful_faces_preserved") is True
                    and r.get("simplification_performed") is False
                    for r in c["records"]), "Frozen positive fixture identities/results differ")
    rt.require({p.name for p in out.iterdir()} <= {"native.json"}, "No geometry/predictions may escape scratch")
    report = dict(stage=STAGE + "_host", status="pass" if good else "fail", producer_revision=revision,
        source_proof=proof, native_report=raw, native_identity=rt.identity(out / "native.json", 1 << 20) if raw else None,
        source_runtime_rehashed_after=True, owned_container_removed=absence["verified"], container_absence=absence,
        native_budget_seconds=BUDGET, docker_outer_seconds=OUTER_SECONDS,
        budget_scope="60s inclusive native; bounded host pre/post and CID cleanup separate", GPU_used=False,
        datasets_read=False, models_read=False, reconstruction_accuracy_verified=False, adoption=False)
    write(out / "report.json", report)
    rt.require({p.name for p in out.iterdir()} == ({"native.json", "report.json"} if raw else {"report.json"})
        and all(p.stat().st_mode & 0o777 == 0o444 for p in out.iterdir()), "Only sealed actual receipts required")
    out.chmod(0o555)
    return 0 if good else (status or 1)


def main():
    code = Path(os.environ["WR_CODE"])
    sys.path[:0] = [str(code / "infra"), str(code / "src")]
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("action", choices=("proof", "mounts", "native", "seal"))
    parser.add_argument("--before"); parser.add_argument("--status", type=int)
    parser.add_argument("--cleanup-verified", type=int, choices=(0, 1))
    args = parser.parse_args(); root, code, revision = environment()
    if args.action == "proof": print(json.dumps(host_proof(root, code, revision), sort_keys=True)); return 0
    if args.action == "mounts": print("\n".join(map(str, mount_paths(root, code)))); return 0
    if args.action == "native": return native(root, code, revision)
    if args.before is None or args.status is None or not 0 <= args.status <= 255 or args.cleanup_verified is None:
        parser.error("Exact before proof/actual native status required")
    return seal(root, code, revision, args.before, args.status, cleanup_verified=args.cleanup_verified == 1)


if __name__ == "__main__":
    raise SystemExit(main())
