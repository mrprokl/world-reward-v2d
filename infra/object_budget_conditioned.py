"""One qualified clip-constant mesh proposal; no repair, fitting or adoption."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time

import numpy as np
import exact_mesh_geometry as geometry
import object_budget_endpoint as endpoint
from mesh_serialization_geometry import (array_hashes, identity, math_mesh, require, stage,
                                         topology_and_embedding, export_birth_mapping, packed_mapping)
from world_reward.mesh_conditioning import prepare_conditioning
from world_reward.mesh_serialization import serialization_preflight, POLICY_SHA256

ROOT = Path("/srv/scenesmith/world-reward")
IMAGE = "sha256:c8fb1632a6908a82aeeeb73c36a00f17a26b81f95f4d2d498c53f75894e21137"
STAGE = "world_reward_cpu_conditioned_object_mesh"
PROTOCOL = "configs/object_budget_conditioned_protocol_v1.json"
PINS = "configs/mesh_conditioned_cache_qualification_pins.json"
CACHE_PROTOCOL = "configs/mesh_conditioned_cache_protocol_v1.json"
PREVIOUS_PINS = "configs/mesh_conditioned_qem_qualification_pins.json"
CPP = "infra/mesh_conditioned_qem.cpp"
REUSED = ("infra/exact_mesh_geometry.py", "infra/mesh_serialization_geometry.py",
          "src/world_reward/mesh_conditioning.py", "src/world_reward/mesh_serialization.py",
          "src/world_reward/exact_triangle_predicates.py")
TOTAL_SECONDS, NATIVE_SECONDS = 1800, 1200
STAGES = ("physical_source", "native_candidate", "float32_glb", "default8_exact_weld",
          "unmodified_official_pack", "original_grounding_metric_bake")
IDENTITY_FIELDS = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
VOLUME, BASE = Path("/opt/world-reward/volume-qem"), Path("/opt/world-reward/guarded-qem")
BOOST = Path("/usr/include/boost")
SCRATCH_FILES = {"input.obj", "candidate.obj", "mapping.json", "object_fixed_canonical.glb", "geometry.npz"}


def strict(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    json.dumps(value, allow_nan=False)  # Also rejects finite-token overflow such as 1e400.
    return value


def pin(value):
    require(type(value) is dict and set(value) == {"bytes", "sha256"}
            and type(value["bytes"]) is int and value["bytes"] > 0
            and type(value["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", value["sha256"]),
            "Exact nonempty byte/SHA identity required")
    return value


def source_binding(root, code, revision):
    require(root == ROOT and re.fullmatch("[0-9a-f]{40}", revision)
            and code == root/"jobs"/revision/"run_object_budget_volume"/"code"
            and Path(__file__) == code/"infra/object_budget_conditioned.py", "Exact dispatched source required")
    rows = {}
    for path in (code, *sorted(code.rglob("*"))):
        meta = path.lstat()
        require(path.resolve() == path and not path.is_symlink() and not meta.st_mode & 0o222,
                "Complete canonical readonly source required")
        if stat.S_ISDIR(meta.st_mode):
            continue
        require(stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1 and meta.st_size <= 32 << 20,
                "Regular bounded source required")
        raw = path.read_bytes(); after = path.lstat()
        require(all(getattr(meta, k) == getattr(after, k) for k in IDENTITY_FIELDS), "Source changed during hashing")
        rows[str(path.relative_to(code))] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    markers = {n: identity(code.parent/n) for n in ("revision", "source-sha256")}
    require((code.parent/"revision").read_bytes() == (revision+"\n").encode()
            and re.fullmatch(b"[0-9a-f]{64}\n", (code.parent/"source-sha256").read_bytes()), "Actual source markers required")
    require(set((PROTOCOL, PINS, CPP, CACHE_PROTOCOL, PREVIOUS_PINS, *REUSED)) <= set(rows), "Complete producer closure required")
    return dict(producer_revision=revision, source_files=len(rows), markers=markers,
                source_files_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                helpers=rows)


def protocol(code):
    value = strict((code/PROTOCOL).read_bytes())
    expected = dict(schema="world_reward.object_budget_conditioned_protocol.v1", backend="qualified_cached_fixed_chart_qem",
                    total_seconds=TOTAL_SECONDS, native_seconds=NATIVE_SECONDS, target_faces=4096, target_vertices=4096,
                    chamfer_diagonal_limit=.01, each_shell_and_net_volume_limit=.05, source_f32_triangles_required=True)
    require(all(type(value.get(k)) is type(v) and value[k] == v for k, v in expected.items())
            and value["full_stages"] == list(STAGES)
            and set(value["claims"]) == {"adoption", "reconstruction_accuracy_verified", "challenge_performance_verified",
                "metric_scale_accuracy_verified", "source_calibration_used", "ground_truth_used", "hand_labeled_test", "oracle_modes"}
            and all(v is False for k, v in value["claims"].items() if k != "oracle_modes")
            and value["claims"]["oracle_modes"] == [], "Frozen production numeric/scope protocol differs")
    return value


def cache_qualification(root, code):
    """SHA-bound completed cache receipts; no importing/executing historical code."""
    pins = strict((code/PINS).read_bytes())
    identities = ("source_cpp", "protocol", "host_report", "native_report", "retained_binary", "previous_qualification_pins")
    keys = {"schema", "producer_revision", "source_files", "source_archive_sha256", "source_files_sha256", *identities,
            "binary_mode", "receipt_mode", "directory_mode", "exact_implementation_regression_verified",
            "qualified_procedural_controls", "native_calls", "production_mesh_validated", "challenge_performance_verified", "adoption"}
    require(type(pins) is dict and set(pins) == keys and pins["schema"] == "world_reward.mesh_conditioned_cache_qualification.v1"
            and type(pins["producer_revision"]) is str and re.fullmatch("[0-9a-f]{40}", pins["producer_revision"])
            and type(pins["source_files"]) is int and pins["source_files"] > 0
            and all(type(pins[k]) is str and re.fullmatch("[0-9a-f]{64}", pins[k]) for k in ("source_archive_sha256", "source_files_sha256"))
            and all(type(pins[k]) is int and pins[k] == v for k, v in (("qualified_procedural_controls", 4), ("native_calls", 8)))
            and pins["exact_implementation_regression_verified"] is True
            and all(pins[k] is False for k in ("production_mesh_validated", "challenge_performance_verified", "adoption"))
            and (pins["binary_mode"], pins["receipt_mode"], pins["directory_mode"]) == ("0555", "0444", "0555"),
            "Actual completed cache qualification required")
    for name in identities:
        pin(pins[name])
    out = root/"results"/("mesh-conditioned-cache-"+pins["producer_revision"])
    require(out.resolve() == out and not out.is_symlink() and stat.S_IMODE(out.lstat().st_mode) == 0o555,
            "Qualified cache directory must be immutable 0555")
    binary = out/"mesh_conditioned_qem"
    for path, field, mode in ((out/"report.json", "host_report", 0o444), (out/"native.json", "native_report", 0o444),
                              (binary, "retained_binary", 0o555)):
        require(identity(path) == pins[field] and stat.S_IMODE(path.lstat().st_mode) == mode, "Qualified retained artifact differs")
    require(identity(code/CPP) == pins["source_cpp"] and identity(code/CACHE_PROTOCOL) == pins["protocol"]
            and identity(code/PREVIOUS_PINS) == pins["previous_qualification_pins"], "Current qualified implementation/protocol differs")
    host, native = (strict((out/n).read_bytes()) for n in ("report.json", "native.json"))
    bound = host["source_binding"]; markers = bound["markers"]
    require(host["stage"] == "mesh_conditioned_cache_host_v1" and native["stage"] == "mesh_conditioned_cache_native_v1"
            and host["status"] == native["status"] == "pass" and native["phase"] == "complete"
            and bound == native["source_binding"] and bound["producer_revision"] == pins["producer_revision"]
            and type(bound["source_files"]) is int and bound["source_files"] == pins["source_files"]
            and bound["source_files_sha256"] == pins["source_files_sha256"]
            and markers == {n: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for n, raw in
                (("revision", (pins["producer_revision"]+"\n").encode()), ("source-sha256", (pins["source_archive_sha256"]+"\n").encode()))}
            and bound["helpers"][CPP] == pins["source_cpp"] and bound["helpers"][CACHE_PROTOCOL] == pins["protocol"]
            and host["cache_protocol_identity"] == native["cache_protocol_identity"] == pins["protocol"]
            and host["retained_binary"] == native["retained_binary"] == native["build"]["binary"] == pins["retained_binary"]
            and host["native_identity"] == pins["native_report"] and host["original_image_id"] == IMAGE
            and host["geometry_qualification_status"] == "pass"
            and all(host[k] is True for k in ("source_rehashed_after", "original_build_rehashed_after", "owned_container_removed"))
            and all(native[k] is True for k in ("originals_rehashed_after", "owned_scratch_removed"))
            and all(r[k] is False for r in (host, native) for k in ("production_mesh_used", "challenge_performance_verified", "adoption", "gpu_used",
                                                                   "simplification_validated", "geometry_quality_validated", "predicate_parity_only"))
            and host["physical_coordinate_cache"] is native["physical_coordinate_cache"] is True,
            "Complete host/native source-bound cache receipts required")
    q = native["conditioned_qualification"]; previous = strict((code/PREVIOUS_PINS).read_bytes())
    require(q == host["conditioned_qualification"] and q["pins_identity"] == pins["previous_qualification_pins"]
            and q["source_cpp"] == previous["source_cpp"] and q["host_report"] == previous["host_report"]
            and q["native_report"] == previous["native_report"]
            and q["measured_binary"] == previous["measured_temporary_binary"] == native["slow_build"]["binary"]
            and native["original_runtime"] == q["original_runtime"]
            and native["build"]["compiler"] == native["slow_build"]["compiler"] == q["qualified_build"]["compiler"]
            and all(identity(code/n) == q["reused_sources"][n] for n in REUSED), "Previously qualified physical math/build differs")
    info = native["build"]["build_info"]
    require(info["physical_coordinate_cache"] is True and info["source_sha256"] == pins["source_cpp"]["sha256"]
            and info["volume_source_sha256"] == identity(code/"infra/mesh_volume_qem.cpp")["sha256"]
            and info["base_source_sha256"] == identity(code/"infra/mesh_guarded_qem.cpp")["sha256"]
            and info["serialization_source_sha256"] == identity(code/"infra/mesh_serialization_qem.cpp")["sha256"]
            and info["target_faces"] == 4096 and info["volume_relative_limit"] == .05
            and info["physical_geometry_rescaled"] is False and info["cost_normalization"] is True
            and info["new_numeric_algorithm"] is True and info["native_cost_and_placement_unchanged"] is False,
            "Actual retained compiler ABI differs")
    g = native["geometry"]; fixtures = g["paired_fixtures"]
    require(g["stage"] == "mesh_conditioned_cache_controls_v1" and g["status"] == "pass"
            and all(g[k] is True for k in ("sources_rehashed_after", "owned_scratch_removed", "previous_successful_sources_only", "exact_implementation_regression_verified"))
            and g["failed_controls_replayed"] is False and g["adoption"] is False
            and type(g["maximum_native_calls"]) is int and g["maximum_native_calls"] == 8 and g["native_budget_seconds"] == 450
            and len(fixtures) == 4 and {f["fixture"]: f["source_array_sha256"] for f in fixtures} == q["expected_sources"]
            and all(f["candidate_and_mapping_byte_exact"] is True and f["physical_stage_evidence_equal"] is True
                and [r["implementation"] for r in f["comparisons"]] == ["slow", "cached"]
                and all(r["method"] == "conditioned" and r["status"] == "pass" and type(r["native_attempts"]) is int
                    and r["native_attempts"] == 1 and r["native_returned"] is True and type(r["committed_collapses"]) is int
                    and r["committed_collapses"] > 0 for r in f["comparisons"]) for f in fixtures),
            "All eight completed byte-exact procedural regression calls required")
    require(native["parity"] == dict(native["parity"], cases=128, mismatches=0, policy_sha256=POLICY_SHA256)
            and native["orientation_parity"]["cases"] == 128 and native["orientation_parity"]["mismatches"] == 0,
            "Qualified independent predicate parity differs")
    return binary, dict(pins_identity=identity(code/PINS), host_report=pins["host_report"], native_report=pins["native_report"],
                        retained_binary=pins["retained_binary"], source_binding=bound, build_info=info,
                        original_runtime=native["original_runtime"], cache_protocol_identity=pins["protocol"])


def runtime_identity(root, code, qualification, remaining):
    """Read-only original CPU layers and one qualified executable ABI query."""
    remaining()
    config = strict((code/"configs/mesh_serialization_compiler_protocol_v1.json").read_bytes())["source_authentication"]
    volume, base = VOLUME, BASE
    image_path = root/"results/image-volume-qem.json"
    require(identity(image_path) == config["original_build_receipt"], "Original CPU image receipt differs")
    image = strict(image_path.read_bytes())
    require(image["status"] == "pass" and image["image_id"] == IMAGE
            and image["source_cpp_sha256"] == config["original_volume_cpp_sha256"]
            and image["binary_sha256"] == config["original_binary_sha256"], "Original CPU image/build differs")
    built, inherited = (strict(p.read_bytes()) for p in (volume/"build.json", base/"build.json"))
    require(built["status"] == inherited["status"] == "pass"
            and built["source_cpp_sha256"] == config["original_volume_cpp_sha256"]
            and inherited["libigl_revision"] == config["libigl_revision"]
            and inherited["eigen_revision"] == config["eigen_revision"]
            and built["binary_sha256"] == config["original_binary_sha256"]
            and identity(volume/"mesh_volume_qem")["sha256"] == config["original_binary_sha256"]
            and identity(base/"mesh_guarded_qem")["sha256"] == inherited["binary_sha256"]
            and identity(volume/"mesh_volume_qem.cpp") == identity(code/"infra/mesh_volume_qem.cpp")
            and identity(base/"mesh_guarded_qem.cpp") == identity(code/"infra/mesh_guarded_qem.cpp"),
            "Original native layer/source/build changed")
    inventory = {str(p.relative_to(base/"source")): identity(p)["sha256"]
                 for p in sorted((base/"source").rglob("*")) if not p.is_dir()}
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(digest == inherited["source_inventory_sha256"] == built["source_inventory_sha256"]
            and all(inventory.get(n) == d for n, d in inherited["pinned_primary_sha256"].items()),
            "Original libigl/Eigen header inventory differs")
    boost = BOOST
    require(identity(boost/"multiprecision/cpp_int.hpp")["sha256"] == config["boost_cpp_int_header_sha256"],
            "Original exact arithmetic header changed")
    rows = {str(p.relative_to(boost)): identity(p)["sha256"] for p in sorted(boost.rglob("*")) if not p.is_dir()}
    actual = dict(original_build=identity(volume/"build.json"), base_build=identity(base/"build.json"),
                  original_binary=identity(volume/"mesh_volume_qem"), inherited_inventory_sha256=digest,
                  boost_files=len(rows), boost_inventory_sha256=hashlib.sha256(json.dumps(
                      rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    require(actual == qualification["original_runtime"], "Qualified original runtime no longer matches")
    binary = root/"results"/("mesh-conditioned-cache-"+qualification["source_binding"]["producer_revision"])/"mesh_conditioned_qem"
    output = subprocess.check_output([str(binary), "--build-info"], timeout=min(10., remaining()))
    require(strict(output) == qualification["build_info"], "Retained fast binary ABI differs")
    remaining()
    return actual | dict(image_receipt=identity(image_path), actual_fast_build_info=qualification["build_info"])


def component_policy(mesh):
    top = geometry.exact_mesh_topology(*math_mesh(mesh))
    signs = sorted(c["volume_sign"] for c in top["components"])
    require(all(s == 1 for s in signs) or signs == [-1, 1], "Unqualified nested-component arrangement")
    cavity = signs == [-1, 1]
    checked = topology_and_embedding(math_mesh(mesh), cavity)
    if not cavity and len(signs) > 1:
        components = geometry._components(*math_mesh(mesh))
        for i, (ids, _, _) in enumerate(components):
            for j, (_, triangles, _) in enumerate(components):
                if i == j:
                    continue
                winding, _ = geometry.solid_winding_and_distances(mesh[0][ids], triangles)
                require(np.isfinite(winding).all() and np.all(np.abs(winding) <= geometry.WINDING_ATOL),
                        "Positive shells must be independent, not nested or overlapping")
        checked = checked | dict(independent_positive_shells_verified=True)
    return cavity, checked


def validate_mapping(source, candidate, document, chart):
    require(set(document) == {"conditioning", "serialization", "native_volume"}, "Native mapping schema differs")
    c, s, m = (document[k] for k in ("conditioning", "serialization", "native_volume"))
    require(all(c.get(k) is v for k, v in dict(chart_scale_positive=True, source_roundtrip_numerically_exact=True,
        physical_geometry_rescaled=False, new_numeric_algorithm=True, native_qslim_implementation_reused=True,
        chart_refitted=False, adopted=False).items()) and c["origin"] == chart.origin.tolist()
        and type(c["scale"]) in (int, float) and c["scale"] == chart.scale
        and type(c["scale_exponent"]) is int and c["scale_exponent"] == chart.scale_exponent
        and type(c["source_roundtrip_vertices"]) is int and c["source_roundtrip_vertices"] == len(source[0]),
        "Native chart was refitted or differs from exact source chart")
    count = s["committed_collapses"]
    require(s["serialization_safe"] is True and type(count) is int and count >= 0
            and type(s["serialization_vetoes"]) is int and s["serialization_vetoes"] >= 0
            and m["committed_collapses"] == count and type(m["committed_collapses"]) is int
            and (count > 0 or (max(len(source[0]), len(source[1])) <= 4096
                 and array_hashes(source) == array_hashes(candidate)))
            and m["native_cost_and_placement_unchanged"] is False and m["cost_normalization"] is True
            and m["final_shell_volumes_verified"] is True and m["volume_relative_limit"] == .05,
            "Require real legal collapses or exact admissible under-budget identity")
    return m, dict(conditioning=c, serialization=s, native_volume={k: v for k, v in m.items() if type(v) in (bool, int, float, str)})


def physical_stage(source, mesh, mapping, cavity, remaining):
    evidence = stage(source, mesh, mapping, cavity, remaining)
    return evidence | dict(candidate_topology=geometry.exact_mesh_topology(*math_mesh(mesh)),
                           scale_or_pose_fitted=False)


def cleanup_scratch(work, owner):
    current = work.lstat()
    require(stat.S_ISDIR(current.st_mode) and current.st_dev == owner.st_dev
            and current.st_ino == owner.st_ino and current.st_uid == owner.st_uid,
            "Owned scratch directory identity changed")
    paths = list(work.iterdir())
    require(all(item.name in SCRATCH_FILES and item.resolve() == item and stat.S_ISREG(item.lstat().st_mode)
                and item.lstat().st_nlink == 1 and item.lstat().st_uid == owner.st_uid for item in paths),
            "Refuse unknown or linked scratch artifacts")
    for path in paths:
        path.unlink()
    work.rmdir()


def native_once(source, binary, work, remaining, record):
    a, b, m = (work/n for n in ("input.obj", "candidate.obj", "mapping.json"))
    geometry.write_obj(a, *source); before = identity(a)
    record.update(phase="native", native_attempts=1, native_returned=False, native_input=before)
    started = time.monotonic()
    try:
        child = subprocess.run([str(binary), str(a), str(b), str(m)], capture_output=True,
                               timeout=min(NATIVE_SECONDS, remaining()))
    finally:
        record["native_elapsed_seconds"] = time.monotonic()-started
        require(identity(a) == before, "Native input OBJ changed")
    record.update(native_returned=True, native_exit_code=child.returncode)
    if child.returncode:
        record["native_stderr_tail"] = child.stderr[-500:].decode(errors="replace")
    require(child.returncode == 0, "One qualified conditioned native call rejected source")
    candidate = geometry.read_obj(b); document = strict(m.read_bytes())
    chart = prepare_conditioning(*source)
    mapping, evidence = validate_mapping(source, candidate, document, chart)
    record.update(evidence, native_artifacts={p.name: identity(p) for p in (a, b, m)})
    return candidate, mapping


def produce(root, episode, binary, official_helper, work, remaining, report):
    inputs, sources, scale = endpoint.prerequisites(root, episode)
    report.update(source_hashes=sources, input_sha256=inputs["video_sha256"])
    raw = endpoint._load_mesh(root/f"outputs/episode_{episode:06d}/object_grounded/object.glb")
    v, f, weld = endpoint.exact_weld(*raw); source = math_mesh((v, f)); before = array_hashes(source)
    report.update(phase="source_geometry", exact_source_welding=weld, source_array_sha256=before)
    cavity, top = component_policy(source); report["source_topology"] = top
    report["stages"] = {"physical_source": dict(topology=top, stored_array_sha256=before)}
    chart = prepare_conditioning(*source)
    require(chart.diagnostics["roundtrip_numerically_exact"] and np.array_equal(chart.decode(chart.encode(v)), v),
            "Whole physical source must roundtrip numerically exactly")
    preflight = dict(serialization_preflight(*source))
    require(preflight["float32_triangles_exactly_active"], "Physical source has inactive binary32 triangles")
    report.update(conditioning=dict(chart.diagnostics), source_serialization=preflight,
                  source_float32_topology=topology_and_embedding(math_mesh((v.astype(np.float32), f)), cavity),
                  source_array_sha256=before)
    remaining(); candidate, mapping = native_once(source, binary, work, remaining, report)
    report["phase"] = "native_candidate"
    report["stages"]["native_candidate"] = physical_stage(source, candidate, mapping, cavity, remaining)
    require(serialization_preflight(*candidate)["position_weld_admissible"], "Native physical candidate is serialization-unsafe")
    import trimesh  # Azure runtime only; no processing/healing or local installation.
    report["phase"] = "float32_glb"
    glb = work/"object_fixed_canonical.glb"
    trimesh.Trimesh(*candidate, process=False).export(glb)
    loaded = endpoint._load_mesh(glb)
    require(np.array_equal(loaded[0], loaded[0].astype(np.float32).astype(np.float64)), "GLB positions are not exact float32 promotion")
    exported = endpoint.exact_weld(*loaded)[:2]; emap = export_birth_mapping(candidate, exported, mapping)
    report["stages"]["float32_glb"] = physical_stage(source, exported, emap, cavity, remaining)
    require(serialization_preflight(*exported)["position_weld_admissible"], "Default8 weld would merge unequal positions")
    report["phase"] = "default8_exact_weld"
    report["stages"]["default8_exact_weld"] = physical_stage(source, exported, emap, cavity, remaining)
    spec = importlib.util.spec_from_file_location("wr_conditioned_official_single_helper", official_helper)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    report["phase"] = "unmodified_official_pack"
    pv, pf = helper.budget_mesh(str(glb), faces=4096, vertices=4096)
    compact, fidelity = geometry.verify_pack_fidelity(exported, pv, pf)
    pmap = packed_mapping(exported, compact, emap)
    report["stages"]["unmodified_official_pack"] = physical_stage(source, compact, pmap, cavity, remaining)
    report["phase"] = "original_grounding_metric_bake"
    metric = pv*scale
    require(np.isfinite(metric).all(), "Original metric grounding overflow")
    expected = (compact[0].astype(pv.dtype)*scale, compact[1])
    metric_compact, _ = geometry.verify_pack_fidelity(expected, metric, pf)
    report["stages"]["original_grounding_metric_bake"] = physical_stage((source[0]*scale, source[1]), metric_compact, pmap, cavity, remaining)
    require(list(report["stages"]) == list(STAGES) and array_hashes(source) == before, "Source/stage invariants changed")
    remaining()
    report.update(source_arrays_unchanged=True, official_pack_fidelity=fidelity, metric_scale_baked_once=scale,
                  frame_poses_changed=False, object_scale=1., candidate_serialization=dict(serialization_preflight(*candidate)))
    npz = work/"geometry.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, vertices=metric, faces=pf, episode_index=np.array(episode),
                            object_scale=np.array(1.), grounded_scale_baked=np.array(scale))
    with np.load(npz, allow_pickle=False) as saved:
        require(set(saved.files) == {"vertices", "faces", "episode_index", "object_scale", "grounded_scale_baked"}
                and np.array_equal(saved["vertices"], metric) and np.array_equal(saved["faces"], pf)
                and saved["episode_index"].item() == episode and saved["object_scale"].item() == 1.
                and saved["grounded_scale_baked"].item() == scale, "Saved metric geometry differs")
    return (glb, npz)


def main(episode):
    require(type(episode) is int and 0 <= episode < 30, "Explicit Track1 episode integer0..29 required")
    require(platform.system() == "Linux" and os.geteuid() == 1000
            and {p.name for p in Path("/sys/class/net").iterdir()} == {"lo"}, "Offline Linux CPU UID1000 required")
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision = os.environ["WR_CODE_REVISION"]
    require(root == ROOT and os.environ["WR_IMAGE_ID"] == IMAGE, "Exact original CPU image/root required")
    out = root/f"outputs/episode_{episode:06d}/object_budget_conditioned"
    require(out.resolve() == out and not out.is_symlink() and out.is_dir() and out.lstat().st_uid == 1000
            and not any(out.iterdir()), "Reserved exclusive conditioned output required")
    report = dict(stage=STAGE, status="fail", episode_index=episode, producer_revision=revision, image_id=IMAGE,
                  script_sha256=identity(Path(__file__))["sha256"],
                  input_track="track_1", ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
                  adoption_performed=False, challenge_performance_verified=False, metric_scale_accuracy_verified=False,
                  components_deleted=False, holes_filled=False, normals_repaired=False, frame_poses_changed=False,
                  native_cost_and_placement_unchanged=False, new_numeric_algorithm=True, cost_normalization=True,
                  budget_seconds=TOTAL_SECONDS, native_budget_seconds=NATIVE_SECONDS, target_faces=4096, target_vertices=4096,
                  native_attempts=0, native_returned=False, phase="authentication")
    started = time.monotonic()
    def remaining():
        seconds = TOTAL_SECONDS-(time.monotonic()-started)
        if seconds <= 0:
            raise TimeoutError("Inclusive conditioned proposal deadline")
        return seconds
    def timeout(*_):
        raise TimeoutError("Inclusive conditioned proposal deadline")
    previous_handler = signal.signal(signal.SIGALRM, timeout); signal.alarm(TOTAL_SECONDS)
    before = qualification = legacy = helper_before = runtime = initializers = None
    published = []; failure = None
    try:
        before = source_binding(root, code, revision); report["source_binding"] = before
        protocol(code); binary, qualification = cache_qualification(root, code); report["cache_qualification"] = qualification
        runtime = runtime_identity(root, code, qualification, remaining); report["runtime_identity"] = runtime
        legacy = geometry.validate_legacy_sources()
        initializers = endpoint.prerequisites(root, episode)
        helper = endpoint.regular(root, root/"vendor/v2d_submission_kit/v2dlb/mesh_budget.py")
        helper_before = identity(helper)
        require(helper_before == dict(bytes=2031, sha256=endpoint.BUDGET_HELPER_SHA), "Unmodified single official helper required")
        report["official_helper_identity"] = helper_before; report["phase"] = "geometry"
        work = Path(tempfile.mkdtemp(prefix="wr-conditioned-proposal-", dir="/tmp")).resolve(); scratch_owner = work.lstat()
        try:
            files = produce(root, episode, binary, helper, work, remaining, report)
            remaining()
            for source in files:
                path = out/source.name
                with path.open("xb") as stream:
                    published.append((path, os.fstat(stream.fileno())))
                    with source.open("rb") as origin:
                        shutil.copyfileobj(origin, stream, 1 << 20)
                    stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
                require(identity(path) == identity(source), "Published candidate differs from validated bytes")
            report.update(geometry_sha256=identity(out/"geometry.npz")["sha256"],
                          canonical_glb_sha256=identity(out/"object_fixed_canonical.glb")["sha256"],
                          outputs={p.name: identity(p) for p, _ in published}, owned_scratch_removed=True)
        finally:
            cleanup_scratch(work, scratch_owner)
            report["owned_scratch_removed"] = True
    except Exception as error:
        failure = error; report.update(error_type=type(error).__name__, error=str(error)[-500:])
    finally:
        try:
            remaining()
            if before is not None:
                require(source_binding(root, code, revision) == before, "Full producer source changed")
                report["source_rehashed_after"] = True
            if qualification is not None:
                require(cache_qualification(root, code)[1] == qualification, "Qualified cache changed")
                report["cache_rehashed_after"] = True
            if runtime is not None:
                require(runtime_identity(root, code, qualification, remaining) == runtime, "Runtime changed")
                report["runtime_rehashed_after"] = True
            if legacy is not None:
                require(geometry.validate_legacy_sources() == legacy, "Frozen geometry helpers changed")
            if helper_before is not None:
                require(identity(helper) == helper_before, "Official helper changed")
                report["helpers_rehashed_after"] = True
            if initializers is not None:
                require(endpoint.prerequisites(root, episode) == initializers, "Frozen input/source reports changed")
                report["inputs_rehashed_after"] = True
            remaining()
        except Exception as error:
            failure = error; report.update(error_type=type(error).__name__, error=str(error)[-500:])
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)
        if time.monotonic()-started >= TOTAL_SECONDS and failure is None:
            failure = TimeoutError("Inclusive conditioned proposal deadline")
            report.update(error_type="TimeoutError", error=str(failure))
        if failure is not None:
            try:
                for path, owner in published:
                    meta = path.lstat()
                    require(stat.S_ISREG(meta.st_mode) and meta.st_dev == owner.st_dev and meta.st_ino == owner.st_ino
                            and meta.st_nlink == 1 and meta.st_uid == owner.st_uid,
                            "Do not remove replaced or linked candidate output")
                    path.unlink()
                report.pop("outputs", None); report["own_candidate_outputs_removed"] = True
            except Exception as cleanup_error:
                report.update(own_candidate_outputs_removed=False, cleanup_error_type=type(cleanup_error).__name__)
        else:
            report.update(status="pass", phase="complete")
        if failure is not None:
            for field in ("canonical_glb_sha256", "geometry_sha256"):
                report.pop(field, None)
        report["elapsed_seconds"] = time.monotonic()-started
        with (out/"report.json").open("x") as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    if failure is not None:
        raise failure
    print(json.dumps({k: report[k] for k in ("stage", "status", "episode_index", "elapsed_seconds")}))
    return report
