"""Read-only position/weld preflight, not a mesh compiler or quality gate.

Simulate GLB float32 positions followed by the plain vertex/face constructor's
Trimesh 5.1.0 default weld. Keys round to even at eight decimal digits; the first
referenced vertex represents each key, without snapping. No geometry is returned
or repaired. Admissibility here says nothing about orientation, topology,
embedding, metric fidelity, downstream simplification or challenge performance.
"""
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import numpy as np

from world_reward.exact_triangle_predicates import (
    ExactTriangleDegeneracyError,
    validate_exact_triangle_non_degeneracy,
)

SCHEMA = "world-reward-mesh-serialization-preflight-v1"
WELD_DIGITS = 8
GROUPING_SOURCE_URL = "https://raw.githubusercontent.com/mikedh/trimesh/5.1.0/trimesh/grouping.py"
GROUPING_SOURCE_SHA256 = "b69118dd902372cd94d17338a1bf187643295e46b1957f34e5788b56c99ecb89"
OFFICIAL_HELPER_SHA256 = "42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0"
_POLICY = {"version": "trimesh-5.1.0", "digits": WELD_DIGITS,
           "keys": "np.round(float32_positions.astype(float64)*1e8).astype(int64)",
           "referenced_only": True, "representative": "first original vertex index",
           "snaps_positions": False, "grouping_sha256": GROUPING_SOURCE_SHA256,
           "official_helper_sha256": OFFICIAL_HELPER_SHA256}
POLICY_SHA256 = hashlib.sha256(json.dumps(
    _POLICY, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def _weld_keys(stored: np.ndarray) -> np.ndarray:
    """Exact default position keys, rejecting undefined int64 conversion."""
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = (stored * (10 ** WELD_DIGITS)).round()
    # float64(int64.max) is 2**63, not int64.max: the upper bound is exclusive.
    if (not np.isfinite(rounded).all() or np.any(rounded < -(2. ** 63))
            or np.any(rounded >= 2. ** 63)):
        raise ValueError("Default weld key outside supported int64 range")
    return rounded.astype(np.int64)


def _active(vertices: np.ndarray, faces: np.ndarray) -> bool:
    if len(vertices) < 3:
        return False
    try:
        validate_exact_triangle_non_degeneracy(vertices, faces)
    except ExactTriangleDegeneracyError:
        return False
    return True


def serialization_preflight(vertices, faces) -> MappingProxyType:
    """Return immutable scalar/hash evidence, never modified or filtered arrays.

    Finite float32/float64 positions and valid integer triangle indices are
    required. Unsupported input/key ranges fail, rather than emulating an
    undefined overflow cast. Orphans do not participate in keys or triangles.
    Exact seams (including signed zero) differ from distinct positions merged
    by float32 storage or rounded keys. Degeneracy/collisions return a negative
    preparatory verdict; every face remains present in the simulated hashes.

    Float32 changes without collisions can still harm fidelity/orientation;
    those require separate gates. Actual loader/library versions are not
    inspected, and unavailable pre-export geometry is never reconstructed.
    """
    if np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces):
        raise ValueError("Require explicit unmasked geometry")
    v, f = np.asarray(vertices), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 3
            or v.dtype.kind != "f" or v.dtype.itemsize not in (4, 8)
            or not np.isfinite(v).all()):
        raise ValueError("Require finite float32/float64 Vx3 positions")
    if (f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or f.dtype.kind not in "iu" or int(f.min()) < 0
            or int(f.max()) >= len(v)):
        raise ValueError("Require nonempty valid integer Fx3 faces")
    referenced = np.unique(f).astype(np.int64)
    original = v[referenced].astype(np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stored32 = original.astype(np.float32)
    if not np.isfinite(stored32).all():
        raise ValueError("Referenced positions cannot be stored as finite float32")
    stored = stored32.astype(np.float64)
    keys = _weld_keys(stored)
    inverse_ref = np.full(len(v), -1, dtype=np.int64)
    inverse_ref[referenced] = np.arange(len(referenced))
    ref_faces = inverse_ref[f]

    unique_source, source_counts = np.unique(original, axis=0, return_counts=True)
    unique_stored, stored_ids = np.unique(stored, axis=0, return_inverse=True)
    _, first, key_ids = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    representatives = first[order]
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    welded = stored[representatives]
    welded_faces = rank[key_ids][ref_faces]
    key_position_pairs = np.unique(np.column_stack((key_ids, stored_ids)), axis=0)
    position_counts = np.bincount(key_position_pairs[:, 0], minlength=len(first))

    source_active = _active(original, ref_faces)
    stored_active = _active(stored, ref_faces)
    welded_active = _active(welded, welded_faces)
    storage_merges = len(unique_source) - len(unique_stored)
    key_merges = len(unique_stored) - len(first)
    source_triangles, stored_triangles = original[ref_faces], stored[ref_faces]
    welded_triangles = welded[welded_faces]
    return MappingProxyType({
        "schema": SCHEMA, "phase": "prepared", "policy_sha256": POLICY_SHA256,
        "grouping_source_sha256": GROUPING_SOURCE_SHA256,
        "official_helper_sha256": OFFICIAL_HELPER_SHA256,
        "weld_digits": WELD_DIGITS, "rounding": "ties-to-even",
        "referenced_only": True, "representative": "first original vertex index",
        "vertices": len(v), "faces": len(f), "referenced_vertices": len(referenced),
        "ignored_orphan_vertices": len(v) - len(referenced),
        "source_exact_seam_duplicates": len(referenced) - len(unique_source),
        "source_exact_seam_groups": int(np.count_nonzero(source_counts > 1)),
        "float32_collapsed_distinct_positions": storage_merges,
        "weld_merged_distinct_float32_positions": key_merges,
        "nonexact_weld_collision_groups": int(np.count_nonzero(position_counts > 1)),
        "welded_vertices": len(first),
        "source_negative_zero_coordinates": int(np.count_nonzero((original == 0) & np.signbit(original))),
        "float32_negative_zero_coordinates": int(np.count_nonzero((stored == 0) & np.signbit(stored))),
        "quantization_changed_referenced_vertices": int(np.count_nonzero(np.any(original != stored, axis=1))),
        "source_triangles_exactly_active": source_active,
        "float32_triangles_exactly_active": stored_active,
        "welded_triangles_exactly_active": welded_active,
        "serialized_triangles_numerically_preserved_by_weld": bool(np.array_equal(stored_triangles, welded_triangles)),
        "serialized_triangles_byte_preserved_by_weld": stored_triangles.tobytes() == welded_triangles.tobytes(),
        "position_weld_admissible": bool(source_active and stored_active and welded_active
                                         and storage_merges == 0 and key_merges == 0),
        "input_vertices_sha256": _hash(v), "input_faces_sha256": _hash(f),
        "referenced_float32_positions_sha256": _hash(stored32),
        "weld_keys_sha256": _hash(keys),
        "representative_original_indices_sha256": _hash(referenced[representatives]),
        "source_oriented_triangles_sha256": _hash(source_triangles),
        "serialized_oriented_triangles_sha256": _hash(stored_triangles),
        "welded_oriented_triangles_sha256": _hash(welded_triangles),
        "input_arrays_modified": False, "faces_deleted_or_reordered": False,
        "positions_snapped_or_rescaled": False, "geometry_returned": False,
        "area_or_length_tolerance_used": False, "pre_export_geometry_recovered": False,
        "orientation_verified": False, "topology_verified": False, "embedding_verified": False,
        "metric_fidelity_verified": False, "simplification_verified": False,
        "runtime_library_verified": False, "challenge_performance_verified": False,
        "adopted": False,
    })
