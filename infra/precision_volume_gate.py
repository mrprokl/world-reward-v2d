"""One frozen data-free precision/backend gate; no challenge input or adoption."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import subprocess
import tempfile
import time

import numpy as np
import exact_mesh_geometry as geometry
import object_budget_endpoint as endpoint
from mesh_link_gate import mesh_topology as historical_topology
from object_budget_guarded import export_birth_mapping
from mesh_endpoint_gate import source_intersections
from mesh_link_gate import _array_hash, _write
import volume_mesh_gate as volume
from world_reward.exact_triangle_predicates import ExactTriangleDegeneracyError
from world_reward.data import sha256

STAGE = "own_exact_predicate_volume_qem_controls_v1"
OUT = "validation/precision_volume_v1"
BUDGET, NATIVE_BUDGET = 3600, 900
IMAGE = "sha256:c8fb1632a6908a82aeeeb73c36a00f17a26b81f95f4d2d498c53f75894e21137"
BUILD_SHA = "8c5441162603260fb768dad688ed7d47ddf29a79de6de53fde603c84994c28ec"
BUILD_BYTES = 3429
BINARY_SHA = "eb606febe824f8f0a308e1d78c7677b2ecddc6ffdd0cdfc8f9a3a04fcd8055e7"
FIXTURES = ["graded_tetra_identity", "graded_sphere_cavity_native"]


def runtime_identity(root, code, revision):
    if (root != Path('/srv/scenesmith/world-reward') or not re.fullmatch('[0-9a-f]{40}',revision)
            or code != root/'jobs'/revision/'run_precision_volume_gate'/'code'
            or Path(__file__) != code/'infra/precision_volume_gate.py'):
        raise ValueError("Exact dispatched runtime required")
    rows = {}
    for p in (code,*sorted(code.rglob('*')),code.parent/'revision',code.parent/'source-sha256',root/'results/image-volume-qem.json',root/'vendor/v2d_submission_kit/v2dlb/mesh_budget.py',volume.BINARY):
        if p.resolve()!=p or any(a.is_symlink() for a in(p,*p.parents)): raise ValueError("Canonical runtime artifacts required")
        before=p.stat()
        if stat.S_ISDIR(before.st_mode):
            if before.st_mode&0o222: raise ValueError("Readonly runtime directories required")
            continue
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or (p.is_relative_to(code) and before.st_mode&0o222): raise ValueError("Original regular runtime files required")
        raw=p.read_bytes();after=p.stat();fields=('st_dev','st_ino','st_mode','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns')
        if any(getattr(before,k)!=getattr(after,k) for k in fields): raise ValueError("Runtime changed while hashing")
        if p==code.parent/'revision' and raw!=(revision+'\n').encode(): raise ValueError("Original revision marker differs")
        if p==code.parent/'source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',raw): raise ValueError("Original source marker differs")
        rows[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'identity':[getattr(before,k) for k in fields]}
    if rows[str(root/'vendor/v2d_submission_kit/v2dlb/mesh_budget.py')]['sha256']!=endpoint.BUDGET_HELPER_SHA: raise ValueError("Official helper changed")
    return rows


def validate_build(root):
    p = root / "results/image-volume-qem.json"
    if p.is_symlink() or not p.is_file() or p.stat().st_size != BUILD_BYTES or sha256(p) != BUILD_SHA:
        raise ValueError("Original volume build receipt changed")
    if os.environ.get("WR_IMAGE_ID") != IMAGE or BINARY_SHA is None:
        raise ValueError("Exact original image/binary pin required; prepared gate is nonexecutable")
    result = volume.validate_build(root)
    if result["binary_sha256"] != BINARY_SHA:
        raise ValueError("Original native binary changed")
    return result


def graded_corner(vertices, faces, vertex=0):
    """Retriangulate one procedural corner without moving its original surface."""
    v, f = vertices.copy(), faces.copy(); v -= v[vertex]
    adjacent = np.flatnonzero(np.any(f == vertex, axis=1))
    neighbors = np.unique(f[adjacent]); neighbors = neighbors[neighbors != vertex]
    ring = {int(n): len(v) + i for i, n in enumerate(neighbors)}
    result_v = np.r_[v, v[neighbors] * 2.**-30]
    replacement = []
    for face_index in adjacent:
        tri = np.roll(f[face_index], -int(np.flatnonzero(f[face_index] == vertex)[0]))
        _, a, b = map(int, tri); p, q = ring[a], ring[b]
        replacement.extend(((vertex, p, q), (p, a, b), (p, b, q)))
    return result_v, np.r_[f[~np.any(f == vertex, axis=1)], np.asarray(replacement)]


def fixtures():
    import trimesh
    v = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    tiny = graded_corner(v, f)
    sphere = trimesh.creation.icosphere(subdivisions=4)
    outer_v, outer_f = graded_corner(sphere.vertices, sphere.faces)
    inner = trimesh.creation.icosphere(subdivisions=2)
    center = -sphere.vertices[0] + [.1,0.,0.]
    inner_v = inner.vertices * [.22,.176,.198] + center
    hollow = (np.r_[outer_v, inner_v], np.r_[outer_f, inner.faces[:, ::-1] + len(outer_v)])
    return [(FIXTURES[0], tiny, False), (FIXTURES[1], hollow, True)]


def native_readiness(mesh):
    v, f = mesh
    # The native OBJ parser reads every represented position into double,
    # including exact float32 promotion; emulate its arithmetic, not GLB dtype.
    v = v.astype(np.float64, copy=False)
    with np.errstate(all="ignore"):
        cross = np.cross(v[f[:,1]] - v[f[:,0]], v[f[:,2]] - v[f[:,0]])
        squared = np.einsum('ij,ij->i', cross, cross)
    if not np.isfinite(cross).all() or not np.isfinite(squared).all() or np.any(squared <= 0):
        raise ValueError("Exact-positive triangle is not ready for unchanged native arithmetic")


def topology_and_embedding(mesh, cavity):
    top = geometry.exact_mesh_topology(*mesh); native_readiness(mesh)
    intersections = source_intersections(*mesh)
    if intersections: raise ValueError("Whole original geometry intersects; no healing")
    if cavity: geometry.true_hollow_containment(*mesh, self_intersecting_faces=intersections)
    return top


def simplify(source):
    with tempfile.TemporaryDirectory(prefix="precision-volume-native-") as temporary:
        a, b, m = (Path(temporary) / name for name in ("input.obj", "output.obj", "mapping.json"))
        geometry.write_obj(a, *source)
        child = subprocess.run([str(volume.BINARY), str(a), str(b), str(m)], capture_output=True, text=True, timeout=NATIVE_BUDGET)
        if child.returncode: raise RuntimeError("Unchanged native QEM failed fixed budget: " + child.stderr[-800:])
        candidate = geometry.read_obj(b); mapping = json.loads(m.read_text())
        expected = {"final_shell_volumes_verified": True, "volume_relative_limit": .05,
                    "native_cost_and_placement_unchanged": True, "cost_normalization": False}
        if any(type(mapping.get(k)) is not type(v) or mapping[k] != v for k,v in expected.items()):
            raise ValueError("Native volume protection/configuration differs")
    return candidate, mapping


def identity_mapping(mesh):
    v, f = mesh
    return dict(source_vertices=len(v), source_faces=len(f), output_vertices=len(v), output_faces=len(f),
                target_reached=True, mapping_complete=True, I=list(range(len(v))), J=list(range(len(f))))


def packed_mapping(exported, compact, mapping):
    """Reorder only exact cyclic oriented faces; no reversal, removal or birth inference."""
    ev, ef = exported; cv, cf = compact
    if not np.array_equal(ev, cv): raise ValueError("Packed canonical vertex order/values differ")
    key = lambda face: min(tuple(face), tuple(np.roll(face, 1)), tuple(np.roll(face, 2)))
    lookup = {key(face): i for i, face in enumerate(ef)}
    if len(lookup) != len(ef) or len(cf) != len(ef) or len({key(face) for face in cf}) != len(cf):
        raise ValueError("Oriented face birth bijection is ambiguous")
    try: order = [lookup[key(face)] for face in cf]
    except KeyError as exc: raise ValueError("Packed face has no exact exported birth") from exc
    return mapping | {"J": [mapping["J"][i] for i in order]}


def measure(source, candidate, mapping):
    record = geometry.mapped_geometry(source, candidate, mapping)
    return {k: record[k] for k in ("sampled_bidirectional_chamfer_diagonal_ratio", "net_volume_relative_error", "birthface_matched_shells")}


def pipeline(source, native, helper, work):
    cavity = native; original = [_array_hash(a) for a in source]
    topology_and_embedding(source, cavity)
    candidate, mapping = simplify(source) if native else (tuple(a.copy() for a in source), identity_mapping(source))
    topology_and_embedding(candidate, cavity); result = {"candidate": measure(source, candidate, mapping)}
    import trimesh
    fixed = work / "object.glb"; trimesh.Trimesh(*candidate, process=False).export(fixed)
    exported = endpoint.exact_weld(*endpoint._load_mesh(fixed))[:2]
    emap = export_birth_mapping(candidate, exported, mapping)
    topology_and_embedding(exported, cavity); result["exported"] = measure(source, exported, emap)
    pv, pf = helper.budget_mesh(str(fixed), faces=4096, vertices=4096)
    compact, fidelity = geometry.verify_pack_fidelity(exported, pv, pf)
    pmap = packed_mapping(exported, compact, emap)
    topology_and_embedding(compact, cavity); result["packed"] = measure(source, compact, pmap)
    scale = .375; metric = pv * scale
    metric_compact, _ = geometry.verify_pack_fidelity((compact[0].astype(pv.dtype) * scale, compact[1]), metric, pf)
    topology_and_embedding(metric_compact, cavity)
    result["metric_baked"] = measure((source[0] * scale, source[1]), metric_compact, pmap)
    if original != [_array_hash(a) for a in source]: raise ValueError("Source arrays changed")
    result.update(source_array_sha256=original, source_arrays_unchanged=True, official_pack_fidelity=fidelity,
                  metric_scale_baked_once=scale, native_binary_invoked=native, faces=len(compact[1]), vertices=len(compact[0]))
    return result


def run(root, report, path):
    report["legacy_sources"] = geometry.validate_legacy_sources(); report["build"] = validate_build(root)
    helper_path = endpoint.regular(root, root / "vendor/v2d_submission_kit/v2dlb/mesh_budget.py")
    if sha256(helper_path) != endpoint.BUDGET_HELPER_SHA: raise ValueError("Official budget helper changed")
    spec = importlib.util.spec_from_file_location("wr_precision_official_budget", helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    for name, source, native in fixtures():
        # A collapsed control is rejected as a whole, never repaired for replay.
        broken = source[0].copy(); broken[source[1][0,1]] = broken[source[1][0,0]]
        try: geometry.exact_mesh_topology(broken, source[1])
        except ExactTriangleDegeneracyError: pass
        else: raise ValueError("Strict collapsed control unexpectedly passed")
        with tempfile.TemporaryDirectory(prefix="precision-volume-control-") as temporary:
            try: historical_topology(*source)
            except ValueError as error:
                if str(error) != "Collapsed/numerically zero-area faces are forbidden": raise
            else: raise ValueError("Procedural control does not isolate the old area false positive")
            if native and len(source[1]) <= 4096: raise ValueError("Native control must exercise real mesh reduction")
            record = {"fixture": name, "status": "fail", "collapsed_control_rejected": True, "old_global_area_control_rejected": True}
            report["fixtures"].append(record); _write(path, report)
            record.update(pipeline(source, native, helper, Path(temporary)), status="pass"); _write(path, report)
    report['status']='pass'


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path('/sys/class/net').iterdir()} != {"lo"}:
        raise RuntimeError("Require isolated remote CPU network-none executor")
    root = Path(os.environ["WR_ROOT"]); out = root / OUT; path = out / "report.json"
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()): raise FileExistsError("Require reserved fresh precision controls")
    report = dict(stage=STAGE, status="fail", code_revision=os.environ["WR_CODE_REVISION"], image_id=os.environ["WR_IMAGE_ID"],
                  script_sha256=sha256(Path(__file__)), exact_geometry_sha256=sha256(Path(geometry.__file__)),
                  challenge_inputs_used=False, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], adoption_performed=False,
                  budget_seconds=BUDGET, native_budget_seconds=NATIVE_BUDGET, target_faces=4096, target_vertices=4096,
                  source_shell_volume_relative_limit=.05, embedding_exact_universal_proof=False, fixtures=[])
    start = time.perf_counter(); signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Frozen3600s precision gate deadline"))); signal.alarm(BUDGET)
    before=None;code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    try: _write(path, report);before=runtime_identity(root,code,revision);run(root, report, path)
    except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0)
        if before is not None:
            try:
                if runtime_identity(root,code,revision)!=before or geometry.validate_legacy_sources()!=report.get('legacy_sources') or validate_build(root)!=report.get('build'): raise ValueError("Original runtime/build/helper/binary changed during controls")
                report.update(runtime_identity_before_after_unchanged=True,runtime_files=len(before),runtime_identity_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest())
            except Exception as error: report.update(status='fail',runtime_revalidation_error=str(error))
        report["elapsed_seconds"] = time.perf_counter() - start; _write(path, report); path.chmod(0o444)
    if report['status']!='pass': raise RuntimeError("Precision runtime revalidation failed")
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds")}))

if __name__ == "__main__": main()
