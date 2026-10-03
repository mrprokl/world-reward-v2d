"""One hash-identical native neutral decode and bounded private exact witnesses.

The original v2 FAIL is immutable. This additive diagnostic never fits, repairs,
renders, reclassifies that gate or claims mesh embedding/separation/contact.
Only the first eight original nonadjacent transverse flags are inspected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np
import own_grasp_capability as original
from world_reward.exact_triangle_witness import exact_triangle_witness


ROOT = Path("/srv/scenesmith/world-reward")
OUTPUT = "validation/own_neutral_exact_witness_v1"
ORIGINAL_REPORT = "validation/own_grasp_capability_v2/report.json"
IMAGE = original.IMAGE
BUDGET = 120
PAIR_BUDGET = 2.0
MAX_PAIRS = 8
ORIGINAL_REVISION = "a14860393d109d73f814b979218065ea0ab4e1aa"
ORIGINAL_SCRIPT_SHA = "3ca1565892fc0d88eff7d3144fa0df05c841a4c0b2e69f6a08d371cc8a44aa07"
ORIGINAL_RECEIPT = dict(sha256="38c56607f2abfbae7f53a1fce55e55970f97b2f4aca740773ddcc161f9ff5391", bytes=18294)
VERTICES_SHA = "4f862130b7bafffd983cc776cf19a82de4d27085a7c58c501d55d88269bc00de"
FACES_SHA = "51d08a7d2e92893ca42c7525bda768afd5c223a33d8302b6674c832e943e8788"
EXACT_HELPER_SHA = "1881af80f85144585379ca4118fef81d524a595b5cd909349cdbedc5bef75812"
HELPERS = ("infra/own_neutral_exact_witness.py", "infra/run_own_neutral_exact_witness.sh",
    "infra/own_grasp_capability.py", "infra/run_own_grasp_capability.sh",
    "src/world_reward/cross_surface.py", "src/world_reward/exact_triangle_witness.py",
    "src/world_reward/__init__.py")


def require_fields(value, expected, context):
    if type(value) is not dict or any(type(value.get(key)) is not type(item) or value[key] != item
            for key, item in expected.items()):
        raise ValueError("Exact " + context + " fields required")


def bound_code(root, code, revision):
    root, code = Path(root), Path(code)
    if (not re.fullmatch("[0-9a-f]{40}", revision)
            or code != root / "jobs" / revision / "run_own_neutral_exact_witness" / "code"
            or not code.is_dir() or code.resolve() != code
            or any(path.is_symlink() for path in (code, *code.parents))):
        raise ValueError("Exact immutable dispatched neutral-witness namespace required")
    return code


def validate_original_report(report):
    require_fields(report, dict(stage="world_reward_own_native_surface_grasp_capability", status="fail",
        phase="neutral_full_surface_prerequisite", producer_revision=ORIGINAL_REVISION,
        script_sha256=ORIGINAL_SCRIPT_SHA, image_id=IMAGE, network="none", budget_seconds=300,
        max_native_forwards=100, own_fresh_procedural_geometry_only=True, historical_poses_results_read=False,
        challenge_inputs_used=False, ground_truth_read=False, hand_labeled_test=False,
        RGB_produced=False, public_manifest_produced=False, quality_verified=False,
        challenge_performance_verified=False, force_closure_verified=False, biomechanics_verified=False,
        touching_certified=False, motion_verified=False, adoption_authorized=False,
        fixed_identity="zero45", fixed_scales="zero68", fixed_expression="zero72", object_scale=1.,
        target_positive_gap_m=.0005, geometry_frame="native_cm_to_metres_proper_diag_1_minus1_minus1",
        error_type="ValueError", error="Fresh full original neutral closed/outward/self-embedded surface prerequisite failed"),
        "original failed v2 provenance")
    if any(key in report for key in ("bottle_surface", "surface_selection", "native_gradient_probe",
            "evaluated_states", "private_outputs", "cross_surface", "saved_cross_surface")):
        raise ValueError("Original gate must stop before bottle/selection/gradient/solve/output")
    require_fields(report.get("model"), dict(sha256=original.MODEL_SHA, bytes=original.MODEL_BYTES), "original model")
    require_fields(report.get("runtime"), dict(torch="2.5.1+cu124", CUDA="12.4", deterministic_algorithms=True,
        warn_only=False, JIT_optimized=False, TF32=False, seed=0, threads=4), "original native runtime")
    calls = report.get("native_calls")
    if type(calls) is not list or len(calls) != 1:
        raise ValueError("Exactly one original validated native forward required")
    require_fields(calls[0], dict(index=1, phase="fresh_neutral", batch_size=1,
        attempted=True, returned=True, validated=True), "original native call")
    neutral = report.get("neutral_surface")
    require_fields(neutral, dict(schema="world-reward-own-closed-surface-certificate-v1", status="fail",
        tolerance_m=1e-8, full_original_faces_retained=True, exact_arithmetic_proof=False,
        max_candidate_pairs=2000000, budget_seconds=30.), "original failed neutral audit")
    require_fields(neutral.get("mesh"), dict(vertices=18439, faces=36874, components=1,
        original_face_coverage=36874, topology_closed_oriented=True, embedding_verified=False,
        original_vertices_sha256=VERTICES_SHA, original_faces_sha256=FACES_SHA), "original full geometry")
    embedding = neutral.get("embedding")
    require_fields(embedding, dict(verified=False, candidate_pairs=240760,
        permitted_shared_simplex_pairs=226454, nested_or_ambiguous_components=[]), "original embedding")
    flags = embedding.get("forbidden_intersections")
    if type(flags) is not list or len(flags) != 114:
        raise ValueError("All original114 forbidden relations required")
    seen, kinds = set(), Counter()
    for row in flags:
        if type(row) is not dict or set(row) != {"faces", "kind"}:
            raise ValueError("Original face-index/type records required")
        pair = row["faces"]
        if (type(pair) is not list or len(pair) != 2 or any(type(item) is not int for item in pair)
                or not 0 <= pair[0] < pair[1] < 36874 or tuple(pair) in seen
                or row["kind"] not in ("proper_intersection", "boundary_contact")):
            raise ValueError("Unique ordered original full-face records required")
        seen.add(tuple(pair)); kinds[row["kind"]] += 1
    if kinds != {"proper_intersection": 110, "boundary_contact": 4}:
        raise ValueError("Original110 transverse/four boundary records required")
    helpers = report.get("source_helpers")
    if type(helpers) is not dict or set(helpers) != set(original.HELPERS):
        raise ValueError("Complete original immutable source helper inventory required")
    for row in helpers.values():
        if (type(row) is not dict or set(row) != {"sha256", "bytes"}
                or type(row["sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}", row["sha256"])
                or type(row["bytes"]) is not int or row["bytes"] <= 0):
            raise ValueError("Original helper SHA/size required")
    if helpers["infra/own_grasp_capability.py"]["sha256"] != ORIGINAL_SCRIPT_SHA:
        raise ValueError("Original v2 driver hash differs")
    metadata = report.get("metadata")
    if type(metadata) is not dict or set(metadata) != {"parameter_names", "joint_names", "parents_sha256",
            "transform_sha256", "bounds_sha256", "faces_sha256", "lbs_sha256"}:
        raise ValueError("Full original native metadata snapshot required")
    return flags


def read_original_report(path):
    # Exact external receipt identity is checked BEFORE parsing any JSON.
    identity = original.identity(Path(path), True)
    if identity != ORIGINAL_RECEIPT:
        raise ValueError("Independently pinned original v2 receipt SHA/bytes mismatch")
    report = json.loads(Path(path).read_text())
    validate_original_report(report)
    if original.identity(Path(path), True) != identity:
        raise ValueError("Original receipt changed during readonly validation")
    return report, identity


def select_original_pairs(report, faces):
    validate_original_report(report)
    if (type(faces) is not np.ndarray or faces.dtype != np.int32 or faces.shape != (36874, 3)
            or original.array_id(faces) != FACES_SHA):
        raise ValueError("Exact original full native int32 face identity required")
    counts, selected = Counter(), []
    for index, row in enumerate(report["neutral_surface"]["embedding"]["forbidden_intersections"]):
        first, last = row["faces"]
        common = sorted(set(map(int, faces[first])) & set(map(int, faces[last])))
        counts[(row["kind"], len(common))] += 1
        if row["kind"] == "proper_intersection" and not common and len(selected) < MAX_PAIRS:
            selected.append(dict(original_flag_index=index, faces=[first, last],
                original_kind=row["kind"], common_vertex_ids=[],
                original_vertex_ids=[faces[first].tolist(), faces[last].tolist()]))
    expected = {("proper_intersection", 0): 108, ("proper_intersection", 1): 2,
        ("boundary_contact", 0): 2, ("boundary_contact", 1): 2}
    if counts != expected or len(selected) != MAX_PAIRS:
        raise ValueError("Exact original nonadjacent flag classification differs")
    return selected, [dict(kind=kind, shared_vertex_count=shared, count=count)
        for (kind, shared), count in sorted(counts.items())]


def one_native_forward(report, invoke, validate, persist):
    if report.get("native_calls"):
        raise RuntimeError("Exactly one native forward permitted; no retry/replay")
    row = dict(index=1, phase="hash_identical_fresh_neutral", batch_size=1,
        attempted=True, returned=False, validated=False)
    report["native_calls"] = [row]; persist()
    values = invoke(); row["returned"] = True; persist()
    validate(values); row["validated"] = True; persist()
    return values


def verify_native_arrays(vertices, faces):
    if (type(vertices) is not np.ndarray or vertices.dtype != np.float32 or vertices.shape != (18439, 3)
            or not np.isfinite(vertices).all() or original.array_id(vertices) != VERTICES_SHA
            or type(faces) is not np.ndarray or faces.dtype != np.int32 or faces.shape != (36874, 3)
            or original.array_id(faces) != FACES_SHA):
        raise ValueError("Full neutral vertices/faces must bit-match original v2 before witnesses")


def inspect_pairs(vertices, faces, selected, report, persist):
    verify_native_arrays(vertices, faces)
    if (type(selected) is not list or len(selected) != MAX_PAIRS or report.get("pairs")
            or selected != report.get("selected_pairs")):
        raise ValueError("Exactly the first eight original nonadjacent pairs, once only")
    report["pairs"] = []
    for row in selected:
        first, last = row["faces"]
        if (row["original_kind"] != "proper_intersection" or row["common_vertex_ids"]
                or set(map(int, faces[first])) & set(map(int, faces[last]))
                or row["original_vertex_ids"] != [faces[first].tolist(), faces[last].tolist()]):
            raise ValueError("Consumer must independently reject adjacent pairs")
        result = exact_triangle_witness(vertices[faces[first]], vertices[faces[last]], budget_seconds=PAIR_BUDGET)
        report["pairs"].append(dict(row, exact=result)); persist()
    report["strict_transverse_witness_count"] = sum(row["exact"]["strict_transverse_witness"] for row in report["pairs"])
    verify_native_arrays(vertices, faces)


def run(root, code, out, report, persist):
    bound_code(root, code, report["producer_revision"])
    receipt_path, model_path = root / ORIGINAL_REPORT, root / "weights/mhr/mhr_model.pt"
    mounts = {str(path): original.exact_mount(Path("/proc/self/mountinfo").read_text(), path, ro)
        for path, ro in ((code, True), (model_path, True), (receipt_path, True), (out, False))}
    source_report, receipt = read_original_report(receipt_path)
    helpers = {name: original.identity(code / name, True) for name in HELPERS}
    if ({name: helpers[name] for name in original.HELPERS} != source_report["source_helpers"]
            or helpers["src/world_reward/exact_triangle_witness.py"]["sha256"] != EXACT_HELPER_SHA):
        raise ValueError("Unchanged original producer helpers and exact witness helper required")
    source = original.identity(model_path)
    if source != dict(sha256=original.MODEL_SHA, bytes=original.MODEL_BYTES):
        raise ValueError("Exact original MHR reference required")
    report.update(original_receipt=receipt, original_producer_revision=ORIGINAL_REVISION,
        original_script_sha256=ORIGINAL_SCRIPT_SHA, source_helpers=helpers, model=source, exact_mounts=mounts,
        phase="native_metadata"); persist()
    if "torch" in sys.modules: raise ValueError("Fresh original CUBLAS-before-Torch policy required")
    import torch
    if str(torch.__version__) != "2.5.1+cu124" or torch.version.cuda != "12.4" or not torch.cuda.is_available():
        raise ValueError("Original pinned CUDA ABI required; no CPU/alternate-image fallback")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    original._strict_runtime(torch)
    report["runtime"] = dict(torch=str(torch.__version__), CUDA=torch.version.cuda,
        deterministic_algorithms=True, warn_only=False, JIT_optimized=False, TF32=False,
        seed=0, threads=4, apply_correctives=True, neutral_controls_require_grad=True)
    persist()
    with torch.jit.optimized_execution(False): model = torch.jit.load(str(model_path), map_location="cuda").float().eval()
    original.freeze_native_weights(model)
    cpu = lambda value: value.detach().cpu().numpy().copy()
    def metadata_snapshot():
        names, joints = list(model.get_parameter_names()), list(model.get_joint_names())
        bounds, faces = cpu(model.get_parameter_limits()), cpu(model.character_torch.mesh.faces)
        parents, transform = cpu(model.character_torch.skeleton.joint_parents), cpu(model.get_parameter_transform())
        lbs = model.get_lbsw()
        if not isinstance(lbs, tuple) or len(lbs) != 2: raise ValueError("Actual original LBS tuple required")
        indices, weights = map(cpu, lbs)
        original.metadata(names, joints, parents, transform, bounds, indices, weights, faces)
        if (names != list(model.character_torch.parameter_transform.parameter_names)
                or joints != list(model.character_torch.skeleton.joint_names)
                or not np.array_equal(transform, cpu(model.character_torch.parameter_transform.parameter_transform))
                or (model.get_num_identity_blendshapes(), model.get_num_face_expression_blendshapes()) != (45, 72)):
            raise ValueError("Actual original submodule/getter metadata differs")
        snapshot = original.native_metadata_identity(names, joints, parents, transform, bounds, faces, indices, weights)
        if snapshot != source_report["metadata"]: raise ValueError("Full native metadata must match original v2")
        return snapshot, bounds, faces
    snapshot, bounds, faces = metadata_snapshot()
    selected, classes = select_original_pairs(source_report, faces)
    report.update(metadata=snapshot, flag_classification=classes, selected_pairs=selected); persist()
    schema = model._c._get_method("forward").schema
    if ([(arg.name, str(arg.type)) for arg in schema.arguments[1:]] != [("identity_coeffs", "Tensor"),
            ("model_parameters", "Tensor"), ("face_expr_coeffs", "Tensor"), ("apply_correctives", "bool")]
            or str(schema.returns[0].type) != "Tuple[Tensor, Tensor]"):
        raise ValueError("Actual original MHRDemo forward ABI required")
    q = torch.zeros(204, device="cuda", requires_grad=True)
    identity = torch.zeros(1, 45, device="cuda"); expression = torch.zeros(1, 72, device="cuda")
    original.legal_controls(cpu(q), bounds)
    if cpu(q).tobytes() != np.zeros(204, np.float32).tobytes(): raise ValueError("Literal native neutral controls required")
    before = original.native_input_identity(cpu(q), cpu(identity), cpu(expression))
    report["phase"] = "single_native_neutral"; persist()
    def invoke():
        original._strict_runtime(torch)
        with torch.jit.optimized_execution(False): result = model(identity, q[None], expression, True)
        torch.cuda.synchronize(); return result
    def validate(values):
        if (not isinstance(values, tuple) or len(values) != 2 or tuple(values[0].shape) != (1,18439,3)
                or tuple(values[1].shape) != (1,127,8) or any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in values)
                or torch.any(values[1][...,7] <= 0)
                or torch.max(torch.abs(torch.linalg.vector_norm(values[1][...,3:7], dim=-1)-1)) > 1e-5):
            raise ValueError("Original full geometry/global-skeleton ABI required")
        if original.native_input_identity(cpu(q), cpu(identity), cpu(expression)) != before:
            raise ValueError("Original native neutral input bytes changed")
        original._strict_runtime(torch)
    native_vertices, _ = one_native_forward(report, invoke, validate, persist)
    vertices = cpu(native_vertices[0] * torch.tensor([1.,-1.,-1.], device="cuda") / 100)
    verify_native_arrays(vertices, faces)
    report.update(phase="exact_pair_witnesses", original_neutral_vertex_hash_matched=True,
        original_native_face_hash_matched=True, vertices_sha256=original.array_id(vertices), faces_sha256=original.array_id(faces)); persist()
    inspect_pairs(vertices, faces, selected, report, persist)
    if (metadata_snapshot()[0] != snapshot or any(parameter.requires_grad for parameter in model.parameters())
            or original.identity(receipt_path, True) != receipt or original.identity(model_path) != source
            or {name: original.identity(code / name, True) for name in HELPERS} != helpers
            or {str(path): original.exact_mount(Path("/proc/self/mountinfo").read_text(), path, ro)
                for path, ro in ((code, True), (model_path, True), (receipt_path, True), (out, False))} != mounts):
        raise ValueError("Original receipt/model/metadata/helpers/mounts changed")
    original._strict_runtime(torch)
    if len(report["native_calls"]) != 1 or not all(report["native_calls"][0][key] for key in ("attempted","returned","validated")):
        raise ValueError("Exactly one complete native call required")
    report.update(status="pass", phase="complete", native_attempts=1, native_returns=1, native_validated=1,
        source_model_receipt_rehashed=True, diagnostic_complete=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision, out = os.environ["WR_CODE_REVISION"], root / OUTPUT
    bound_code(root, code, revision)
    if (platform.system() != "Linux" or root != ROOT or os.geteuid() != 1000
            or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}
            or os.environ.get("WR_IMAGE_ID") != IMAGE or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or not out.is_dir() or any(out.iterdir()) or out.stat().st_mode & 0o077
            or any(path.resolve() != path or any(parent.is_symlink() for parent in (path,*path.parents)) for path in (root,code,out))
            or Path(__file__).resolve() != code / "infra/own_neutral_exact_witness.py"):
        raise ValueError("Exclusive source-bound offline original-image private Azure runtime required")
    report = dict(stage="world_reward_own_neutral_exact_self_crossing_witness", status="fail", phase="source_integrity",
        producer_revision=revision, script_sha256=original.identity(Path(__file__), True)["sha256"], image_id=IMAGE,
        network="none", budget_seconds=BUDGET, max_native_forwards=1, max_pairs=MAX_PAIRS, pair_budget_seconds=PAIR_BUDGET,
        diagnostic_complete=False, original_v2_reclassified=False, original_v2_failure_preserved=True,
        geometry_embedding_certified=False, full_separation_proven=False, challenge_inputs_used=False, ground_truth_read=False,
        optimization_performed=False, model_training_performed=False, RGB_produced=False, geometry_payload_produced=False,
        touching_certified=False, force_closure_verified=False, motion_verified=False, quality_verified=False,
        adoption_authorized=False, native_calls=[])
    receipt = out / "report.json"
    with receipt.open("x") as stream:
        def persist():
            stream.seek(0); json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Frozen120s additive neutral exact-witness budget")
        alarm, term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired)
        started = time.monotonic(); signal.alarm(BUDGET)
        try: persist(); run(root, code, out, report, persist)
        except BaseException as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
            report["elapsed_seconds"] = time.monotonic() - started; persist(); receipt.chmod(0o400)
            if json.loads(receipt.read_text()) != report: raise ValueError("Saved additive report changed on reread")
            if {path.name for path in out.iterdir()} != {"report.json"}: raise ValueError("Report-only private inventory required")


if __name__ == "__main__": main()
