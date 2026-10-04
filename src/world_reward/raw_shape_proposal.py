"""Freeze a whole raw generator mesh before admissibility or metric conversion.

No generator, Trimesh processing, normalization, compaction, repair, component
selection, scale/pose inference, I/O or quality scoring occurs here. Degenerate,
duplicate and unused geometry remains in the proposal, for a separate validator
to judge. Metadata references and automatic-anchor declarations are NOT verified
source, model, observation, licence or training-overlap evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

import numpy as np

from world_reward.shared_scene import ArtifactRef, _freeze, _text, _thaw

SCHEMA = "world_reward.raw_shape_proposal.v1"
REFERENCE_ROLES = ("source", "model", "config")


def _raw_array(value, name, dtypes):
    if (not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value)
            or value.dtype not in dtypes or value.ndim != 2 or value.shape[1:] != (3,)
            or not len(value) or not np.isfinite(value).all()):
        raise ValueError(f"{name}: nonempty finite raw Nx3 with exact declared dtype required")
    # An immutable bytes backing cannot be made writable via ndarray.setflags.
    # This keeps each value's C-order bytes/dtype, including FP32 and signed zero.
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def _ref(value):
    if type(value) is not ArtifactRef:
        raise ValueError("Declared ArtifactRef required; no artifact is read or authenticated")
    return ArtifactRef(value.path, value.bytes, value.sha256)


def _identity(array):
    return {"dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.nbytes,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _ref_record(value):
    return {"path": value.path, "bytes": value.bytes, "sha256": value.sha256}


@dataclass(frozen=True)
class AutomaticShapeAnchor:
    """Original frame0 automatic RGB/mask references, not verified observations."""
    frame_index: int
    image: ArtifactRef
    mask: ArtifactRef
    method: str

    def __post_init__(self):
        if type(self.frame_index) is not int or self.frame_index != 0:
            raise ValueError("Explicit original frame0 anchor required, not a selected later frame")
        object.__setattr__(self, "image", _ref(self.image))
        object.__setattr__(self, "mask", _ref(self.mask))
        _text(self.method, "automatic anchor method declaration")


@dataclass(frozen=True, eq=False)
class RawShapeProposal:
    vertices: np.ndarray
    faces: np.ndarray
    backend: str
    canonical_frame: str
    linear_unit: str
    gauge_convention: str
    anchor: AutomaticShapeAnchor
    references: Mapping
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        vertices = _raw_array(self.vertices, "vertices", (np.dtype(np.float32), np.dtype(np.float64)))
        faces = _raw_array(self.faces, "faces", (np.dtype(np.int64),))
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("In-range original face indices required; collapsed geometry is preserved")
        for name in ("backend", "canonical_frame", "linear_unit", "gauge_convention"):
            _text(getattr(self, name), name)
        if type(self.anchor) is not AutomaticShapeAnchor:
            raise ValueError("Explicit AutomaticShapeAnchor required")
        a = self.anchor
        anchor = AutomaticShapeAnchor(a.frame_index, a.image, a.mask, a.method)
        if not isinstance(self.references, Mapping) or set(self.references) != set(REFERENCE_ROLES):
            raise ValueError("Exactly source/model/config reference roles required")
        references = {}
        for role in REFERENCE_ROLES:
            refs = self.references[role]
            if type(refs) is not tuple or not refs:
                raise ValueError("Each reference role requires a nonempty tuple of ArtifactRef")
            references[role] = tuple(_ref(item) for item in refs)
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Provenance must be tiny JSON metadata")
        provenance = _freeze(self.provenance)
        metadata = {"references": {key: [_ref_record(item) for item in refs]
                                    for key, refs in references.items()},
                    "provenance": _thaw(provenance)}
        if len(json.dumps(metadata, allow_nan=False).encode()) > 16384:
            raise ValueError("Reference/provenance metadata must remain <=16KB")
        for name, value in (("vertices", vertices), ("faces", faces), ("anchor", anchor),
                            ("references", MappingProxyType(references)), ("provenance", provenance)):
            object.__setattr__(self, name, value)

    def report(self):
        """Fresh JSON diagnostics/fingerprints, not a solid or metric certificate.

        Cross products use unnormalized FP64 arithmetic only for diagnostics.
        Overflow/underflow is not a verdict on exact geometry; no face is dropped.
        Fingerprints bind supplied raw bytes and declarations, not upstream code.
        """
        referenced = np.unique(self.faces)
        repeated = ((self.faces[:, 0] == self.faces[:, 1])
                    | (self.faces[:, 0] == self.faces[:, 2])
                    | (self.faces[:, 1] == self.faces[:, 2]))
        zero = nonfinite = 0
        for start in range(0, len(self.faces), 4096):
            triangles = self.vertices[self.faces[start:start + 4096]].astype(np.float64, copy=False)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            finite = np.isfinite(cross).all(axis=1)
            zero += int(np.count_nonzero(finite & np.all(cross == 0, axis=1)))
            nonfinite += int(np.count_nonzero(~finite))
        result = {"schema": SCHEMA, "status": "proposal_uncertified", "backend": self.backend,
                  "canonical_frame": self.canonical_frame, "linear_unit": self.linear_unit,
                  "gauge_convention": self.gauge_convention,
                  "vertices": _identity(self.vertices), "faces": _identity(self.faces),
                  "anchor": {"frame_index": 0, "image": _ref_record(self.anchor.image),
                             "mask": _ref_record(self.anchor.mask), "method": self.anchor.method},
                  "references": {role: [_ref_record(item) for item in self.references[role]]
                                 for role in REFERENCE_ROLES}, "provenance": _thaw(self.provenance),
                  "diagnostics": {"referenced_vertices": len(referenced),
                      "unused_vertices_preserved": len(self.vertices) - len(referenced),
                      "repeated_index_faces_preserved": int(np.count_nonzero(repeated)),
                      "zero_cross_product_faces_fp64": zero, "nonfinite_cross_product_faces_fp64": nonfinite,
                      "triangle_diagnostics_are_exact_certification": False},
                  "supplied_raw_arrays_byte_preserved": True, "freeze_geometry_operations_applied": False,
                  "source_geometry_operations_verified": False,
                  "geometry_operation_scope": "this_freeze_only_upstream_unverified",
                  "source_references_authenticated": False, "automatic_anchor_authenticated": False,
                  "metric_gauge_verified": False, "solid_geometry_certified": False,
                  "reconstruction_accuracy_verified": False, "adoption_authorized": False}
        result["proposal_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return result
