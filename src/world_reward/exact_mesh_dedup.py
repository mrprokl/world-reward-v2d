"""Frame-local exact raster-input reuse, never hypothesis/evidence filtering.

Only byte-identical finite meshes of the same dtype/shape share a computation.
No cast, rounding, tolerance, topology change or cross-frame cache is allowed.
The caller keeps faces/camera/grid/settings and the current observed mask fixed;
restore scores to every original slot before unchanged selection/acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral

import numpy as np


def _fingerprint(array):
    return hashlib.blake2b(memoryview(array).cast('B'), digest_size=32).digest()


@dataclass(frozen=True)
class ExactMeshDedup:
    unique: tuple[np.ndarray, ...]
    original_to_unique: np.ndarray
    first_original_indices: tuple[int, ...]

    @property
    def counts(self):
        return dict(original_meshes=len(self.original_to_unique), unique_meshes=len(self.unique),
                    reused_meshes=len(self.original_to_unique)-len(self.unique))

    def restore(self, unique_values):
        """Restore any score/result for all slots in their original order."""
        if len(unique_values) != len(self.unique):
            raise ValueError('Exactly one result per unique full mesh required')
        return [unique_values[int(i)] for i in self.original_to_unique]

    def batches(self, batch_size=4):
        """Stable batches of at most four full geometries; no sampling."""
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral) or not 1 <= batch_size <= 4:
            raise ValueError('Explicit raster batch_size integer in [1,4] required')
        for start in range(0, len(self.unique), int(batch_size)):
            yield self.unique[start:start+int(batch_size)]


def deduplicate_meshes(meshes) -> ExactMeshDedup:
    """Build a NEW plan for ONE frame, keeping stable first occurrence order.

    Hashes only find candidate matches: even a digest collision must pass complete
    byte comparison. Signed zero, sub-ULP differences and dtype differences never
    merge. Arrays remain original (no precision/geometry mutation); keep them
    unchanged until rendering/restoration finishes, as with the original batch.
    """
    unique = []; canonical = []; first = []; mapping = []; buckets = {}; shape = None
    for i, mesh in enumerate(meshes):
        a = np.asarray(mesh)
        if (a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 3
                or a.dtype.kind != 'f' or a.dtype.itemsize not in (4, 8) or not np.isfinite(a).all()):
            raise ValueError('Finite complete float32/64 [V,3] meshes required')
        if shape is None: shape = a.shape
        if a.shape != shape: raise ValueError('All full meshes must share unchanged vertex count/order')
        c = np.ascontiguousarray(a)
        key = (a.dtype.str, a.shape, _fingerprint(c))
        match = next((j for j in buckets.get(key, ())
                      if np.array_equal(c.view(np.uint8), canonical[j].view(np.uint8))), None)
        if match is None:
            match = len(unique); unique.append(a); canonical.append(c); first.append(i)
            buckets.setdefault(key, []).append(match)
        mapping.append(match)
    if not mapping: raise ValueError('At least one original mesh required')
    indices = np.asarray(mapping, dtype=np.int64); indices.flags.writeable = False
    return ExactMeshDedup(tuple(unique), indices, tuple(first))
