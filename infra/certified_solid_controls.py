"""Fresh procedural CGAL controls, not production geometry or HOI quality."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from world_reward.oriented_solid_forest import adjudicate_oriented_solid_forest

CALL_SECONDS, TOTAL_SECONDS, MAX_CALLS = 60, 900, 15
MAX_OUTPUT_BYTES = 1 << 20
TRUE_FLAGS = ("closed_oriented_vertex_manifold_verified", "all_original_faces_retained",
              "all_original_vertices_referenced", "exact_nondegenerate_triangles_verified",
              "component_self_intersections_absent", "inter_component_surface_contacts_absent")
FALSE_FLAGS = ("geometry_repaired", "orientation_changed", "qem_executed",
               "forest_adjudicated", "reconstruction_accuracy_verified")
FIELDS = set(("schema", "status", "source_sha256", "cgal_version", "vertices", "faces",
              "component_count", "components", "inside", "represented_coordinates") + TRUE_FLAGS + FALSE_FLAGS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _owned(array, dtype):
    value = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, eq=False)
class Control:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    labels: np.ndarray
    signs: tuple[int, ...]
    parents: tuple[int, ...]
    expected: str = "material"
    rejection: str | None = None
    input_override: bytes | None = None

    def __post_init__(self):
        for name, dtype in (("vertices", np.float64), ("faces", np.int64), ("labels", np.int64)):
            object.__setattr__(self, name, _owned(getattr(self, name), dtype))


def box(lo, hi, sign=1):
    """Own eight-vertex box, all twelve outward triangles; reverse only fixture sign."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    require(lo.shape == hi.shape == (3,) and np.all(hi > lo) and sign in (-1, 1), "Invalid box fixture")
    v = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                  [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                  [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]], dtype=np.int64)
    return v, f if sign == 1 else f[:, ::-1]


def _control(name, bounds, signs, parents, **kwargs):
    meshes = [box(lo, hi, s) for (lo, hi), s in zip(bounds, signs)]
    vertices = np.concatenate([v for v, _ in meshes])
    faces = np.concatenate([f + 8 * i for i, (_, f) in enumerate(meshes)])
    labels = np.repeat(np.arange(len(meshes), dtype=np.int64), 12)
    return Control(name, vertices, faces, labels, tuple(signs), tuple(parents), **kwargs)


def control_input(control):
    """All original coordinates use 17 digits: binary64 decimal roundtrip, no repair."""
    if control.input_override is not None:
        return control.input_override
    lines = [f"WR_SOLID_QUERY_V1 {len(control.vertices)} {len(control.faces)} {len(control.signs)}"]
    lines.extend(" ".join(format(float(x), ".17g") for x in row) for row in control.vertices)
    lines.extend(" ".join(str(int(x)) for x in (*row, label))
                 for row, label in zip(control.faces, control.labels))
    return ("\n".join(lines) + "\n").encode("ascii")


def controls():
    """Six valid similarities, eight native rejections, one invalid material forest."""
    specs = (
        ("two_cavities", [((-4, -4, -4), (4, 4, 4)), ((-3, -1, -1), (-1, 1, 1)),
                          ((1, -1, -1), (3, 1, 1))], (1, -1, -1), (-1, 0, 0)),
        ("disconnected_hollows", [((-5, -2, -2), (-1, 2, 2)), ((-4, -1, -1), (-2, 1, 1)),
                                  ((1, -2, -2), (5, 2, 2)), ((2, -1, -1), (4, 1, 1))],
         (1, -1, 1, -1), (-1, 0, -1, 2)),
        ("island_in_void", [((-4, -4, -4), (4, 4, 4)), ((-3, -3, -3), (3, 3, 3)),
                            ((-1, -1, -1), (1, 1, 1))], (1, -1, 1), (-1, 0, 1)),
    )
    result = []
    for name, bounds, signs, parents in specs:
        source = _control(name + "_identity", bounds, signs, parents)
        result.append(source)
        # Cyclic axis permutation is a proper rotation (det+1); dyadic transform exact.
        moved = source.vertices[:, [2, 0, 1]] * 8 + np.array([16., -8., 4.])
        result.append(Control(name + "_similarity", moved, source.faces, source.labels, signs, parents))
    small = _control("base", [((-1, -1, -1), (1, 1, 1))], (1,), (-1,))
    for name, bounds in (
        ("contact", [((-1, -1, -1), (1, 1, 1)), ((1, -1, -1), (3, 1, 1))]),
        ("crossing", [((-1, -1, -1), (1, 1, 1)), ((0, 0, 0), (2, 2, 2))]),
    ):
        result.append(_control(name, bounds, (1, 1), (-1, -1), expected="native_reject",
                               rejection="Components intersect or touch"))
    crossed = small.vertices.copy(); crossed[6] = (0., 0., -2.)
    result.append(Control("self_crossing", crossed, small.faces, small.labels, (1,), (-1,),
                          "native_reject", "Exact component self-intersection"))
    result.append(Control("open", small.vertices, small.faces[:-1], small.labels[:-1], (1,), (-1,),
                          "native_reject", "Closed opposite edge incidences required"))
    result.append(Control("duplicate", small.vertices, np.vstack((small.faces, small.faces[:1])),
                          np.r_[small.labels, 0], (1,), (-1,), "native_reject", "Duplicate indexed triangle"))
    degenerate = small.vertices.copy(); degenerate[2] = degenerate[1]
    result.append(Control("degenerate", degenerate, small.faces, small.labels, (1,), (-1,),
                          "native_reject", "Exact degenerate triangle"))
    result.append(Control("trailing", small.vertices, small.faces, small.labels, (1,), (-1,),
                          "native_reject", "Unexpected trailing input", control_input(small) + b"extra\n"))
    invalid = small.faces.copy(); invalid[0, 0] = len(small.vertices)
    result.append(Control("bounds", small.vertices, invalid, small.labels, (1,), (-1,),
                          "native_reject", "Integer outside input bounds"))
    result.append(_control("nested_positive", [((-3, -3, -3), (3, 3, 3)),
                                              ((-1, -1, -1), (1, 1, 1))],
                           (1, 1), (-1, 0), expected="forest_reject"))
    return tuple(result)


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and
            not any(p.is_symlink() for p in (path, *path.parents)), "Canonical source/binary required")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= 64 << 20,
            "Bounded single-link source/binary required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(all(getattr(before, f) == getattr(after, f) for f in fields), "Source/binary changed while hashed")
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate native JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite native JSON")))


def validate_pass(raw, control, source_sha256):
    require(len(raw) <= MAX_OUTPUT_BYTES, "Native output exceeds bound")
    report = strict_json(raw)
    require(type(report) is dict and set(report) == FIELDS and
            report["schema"] == "world_reward.certified_solid_query.v1" and report["status"] == "pass" and
            report["source_sha256"] == source_sha256 and report["cgal_version"] == "6.0.1",
            "Native schema/source/CGAL differs")
    for key, value in (("vertices", len(control.vertices)), ("faces", len(control.faces)),
                       ("component_count", len(control.signs))):
        require(type(report[key]) is int and report[key] == value, "Native original counts differ")
    require(all(report[k] is True for k in TRUE_FLAGS) and all(report[k] is False for k in FALSE_FLAGS),
            "Native geometric scope flags differ")
    require(report["represented_coordinates"] ==
            "EPECK exact values of original parsed IEEE754 binary64; no perturbation",
            "Native represented coordinates differ")
    rows = report["components"]
    require(type(rows) is list and len(rows) == len(control.signs), "Native components differ")
    for i, row in enumerate(rows):
        ids = np.unique(control.faces[control.labels == i])
        expected = dict(original_component_id=i, original_vertices=len(ids),
                        original_faces=int(np.count_nonzero(control.labels == i)),
                        witness_original_vertex=int(ids[0]), exact_volume_sign=control.signs[i])
        require(type(row) is dict and set(row) == set(expected) | {"diagnostic_signed_volume"} and
                all(type(row[k]) is int and row[k] == v for k, v in expected.items()), "Native lineage/sign differs")
        require(type(row["diagnostic_signed_volume"]) in (int, float) and
                math.isfinite(row["diagnostic_signed_volume"]), "Tiny control volume diagnostic unavailable")
    inside = report["inside"]
    count = len(control.signs)
    require(type(inside) is list and len(inside) == count and
            all(type(row) is list and len(row) == count and all(type(v) is bool for v in row) for row in inside),
            "Native containment is not a full bool matrix")
    expected_inside = np.zeros((count, count), dtype=bool)
    for i, parent in enumerate(control.parents):
        while parent != -1:
            expected_inside[i, parent] = True; parent = control.parents[parent]
    require(np.array_equal(inside, expected_inside), "Native exact containment differs")
    return report


class ControlError(RuntimeError):
    def __init__(self, report):
        super().__init__("Certified solid procedural controls failed")
        self.report = report


def run_controls(binary, source_sha256):
    started = time.monotonic()
    require(type(source_sha256) is str and re.fullmatch("[0-9a-f]{64}", source_sha256), "Source SHA256 required")
    binary = Path(binary)
    sources = (Path(__file__).resolve(), CODE / "src/world_reward/oriented_solid_forest.py",
               CODE / "infra/certified_solid_query.cpp")
    before = {str(p): identity(p) for p in (binary, *sources)}
    require(before[str(sources[-1])]["sha256"] == source_sha256, "Compiled native source pin differs")
    report = dict(stage="certified_solid_procedural_controls_v1", status="fail", records=[],
                  native_source_sha256=source_sha256, artifacts_before=before,
                  call_seconds=CALL_SECONDS, inclusive_seconds=TOTAL_SECONDS, maximum_calls=MAX_CALLS,
                  challenge_inputs_used=False, gt_used=False, qem_executed=False, adoption=False,
                  production_mesh_validated=False, reconstruction_accuracy_verified=False)
    try:
        bank = controls(); require(len(bank) == MAX_CALLS, "Frozen control count differs")
        for control in bank:
            remaining = TOTAL_SECONDS - (time.monotonic() - started)
            require(remaining > 0, "Inclusive control budget exhausted")
            raw = control_input(control)
            row = dict(name=control.name, expected=control.expected, input_bytes=len(raw),
                       input_sha256=hashlib.sha256(raw).hexdigest(), status="fail")
            report["records"].append(row); call_started = time.monotonic()
            try:
                child = subprocess.run([str(binary)], input=raw, capture_output=True,
                                       timeout=min(CALL_SECONDS, remaining), check=False)
            finally:
                row["elapsed_seconds"] = time.monotonic() - call_started
            row.update(returncode=child.returncode, stdout_bytes=len(child.stdout),
                       stdout_sha256=hashlib.sha256(child.stdout).hexdigest(), stderr_bytes=len(child.stderr),
                       stderr_sha256=hashlib.sha256(child.stderr).hexdigest())
            require(len(child.stdout) <= MAX_OUTPUT_BYTES and len(child.stderr) <= 512, "Native output bound differs")
            if control.expected == "native_reject":
                require(child.returncode == 1 and not child.stdout and
                        child.stderr == ("certified_solid_query FAIL: " + control.rejection + "\n").encode(),
                        "Negative control did not exercise expected rejection")
                row["native_rejection_verified"] = True
            else:
                require(child.returncode == 0 and not child.stderr, "Native positive control failed")
                actual = validate_pass(child.stdout, control, source_sha256)
                try:
                    forest = adjudicate_oriented_solid_forest(
                        tuple(f"original-component-{i}" for i in range(len(control.signs))),
                        np.array(control.signs, dtype=np.int64), np.array(actual["inside"], dtype=bool))
                except ValueError:
                    require(control.expected == "forest_reject", "Valid material forest was rejected")
                    row["invalid_orientation_forest_rejected"] = True
                else:
                    require(control.expected == "material" and tuple(forest.parents) == control.parents,
                            "Invalid forest accepted or parent changed")
                    row.update(parents=forest.parents.tolist(), signs=forest.signs.tolist(), depths=forest.depths.tolist())
            row["status"] = "pass"
        report["phase"] = "complete"
    except Exception as error:
        report["failure_type"] = type(error).__name__
        report["failure_reason"] = str(error)[:300]
    finally:
        try:
            report["artifacts_after"] = {str(p): identity(p) for p in (binary, *sources)}
            require(report["artifacts_after"] == before, "Source/binary changed during controls")
            report["source_binary_rehashed_after"] = True
        except Exception as error:
            report["failure_type"] = type(error).__name__
        report["elapsed_seconds"] = time.monotonic() - started
        if report["elapsed_seconds"] > TOTAL_SECONDS:
            report["failure_type"] = "InclusiveDeadline"
    if "failure_type" in report:
        raise ControlError(report)
    report["status"] = "pass"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    try:
        report = run_controls(args.binary, args.source_sha256)
    except ControlError as error:
        report = error.report
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(len(encoded) < MAX_OUTPUT_BYTES, "Control receipt exceeds bound")
    print(encoded)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
