"""Adjudicate supplied exact containment, never certify or repair geometry.

The caller must independently certify closed, connected, oriented embedded
components and no inter-component contact/intersection. ``inside[i,j]`` then
means the entire boundary of i lies in the bounded side of j. This module has
no coordinates, native predicates, certificate authentication or tolerances.
A valid result proves the supplied relation/signs describe a binary oriented
material forest, conditional on those geometric preconditions, not that the
input mesh satisfies them. Stable keys remain in the caller's original order;
their provenance/birth lineage is neither invented nor independently verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _readonly(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class OrientedSolidForest:
    """Owned readonly full-component relation and its unique parent forest.

    Parent indices reference ``component_keys`` without sorting or dropping;
    -1 denotes a root. Depth is the number of strict ancestors. Roots have
    sign +1, then signs alternate (-1 for cavities, +1 for islands). No
    geometric separation, volume, reconstruction quality or identity is claimed.
    """
    component_keys: tuple[str, ...]
    signs: np.ndarray
    inside: np.ndarray
    parents: np.ndarray = field(init=False)
    depths: np.ndarray = field(init=False)

    def __post_init__(self):
        keys = self.component_keys
        if (type(keys) is not tuple or not keys
                or any(type(key) is not str or not key.strip() or len(key) > 256
                       or any(ord(c) < 32 or ord(c) == 127 for c in key) for key in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError("Unique bounded component keys in original tuple order required")
        if np.ma.isMaskedArray(self.signs) or np.ma.isMaskedArray(self.inside):
            raise ValueError("Explicit unmasked signs and containment required")
        signs, inside = np.asarray(self.signs), np.asarray(self.inside)
        count = len(keys)
        if (signs.dtype != np.int64 or signs.shape != (count,)
                or not np.isin(signs, [-1, 1]).all()
                or inside.dtype != np.bool_ or inside.shape != (count, count)):
            raise ValueError("Require int64[K] signs +/-1 and bool[K,K] containment")
        # Own immutable copies before adjudication; caller arrays are never edited.
        signs, inside = _readonly(signs), _readonly(inside)
        if np.diag(inside).any():
            raise ValueError("A component cannot strictly contain itself")
        if np.any(inside & inside.T):
            raise ValueError("Mutual containment is forbidden")
        depths = inside.sum(axis=1, dtype=np.int64)
        parents = np.full(count, -1, dtype=np.int64)
        for i in range(count):
            ancestors = np.flatnonzero(inside[i])
            for j in ancestors:
                if np.any(inside[j] & ~inside[i]):
                    raise ValueError("Containment must include every transitive ancestor")
            for offset, j in enumerate(ancestors):
                for k in ancestors[offset + 1:]:
                    if not inside[j, k] and not inside[k, j]:
                        raise ValueError("Incomparable ancestors cannot define a material forest")
            if len(ancestors):
                closest = ancestors[depths[ancestors] == depths[i] - 1]
                if len(closest) != 1:
                    raise ValueError("Each non-root requires one unique minimal parent")
                parents[i] = closest[0]
        expected = np.where(depths % 2 == 0, 1, -1)
        if not np.array_equal(signs, expected):
            raise ValueError("Roots must be positive and orientation must alternate with depth")
        object.__setattr__(self, "signs", signs)
        object.__setattr__(self, "inside", inside)
        object.__setattr__(self, "parents", _readonly(parents))
        object.__setattr__(self, "depths", _readonly(depths))


def adjudicate_oriented_solid_forest(component_keys, signs, inside) -> OrientedSolidForest:
    """Fail closed on inconsistent supplied containment; never infer/repair it.

    All original components are retained, including inward boundaries. A
    native predicate failure (crossing/contact/uncertified geometry) must abort
    upstream, not be encoded as an all-false ``inside`` matrix. Graph checks
    are O(K^3) worst case for K components; no face/vertex queries happen here.
    """
    return OrientedSolidForest(component_keys, signs, inside)
