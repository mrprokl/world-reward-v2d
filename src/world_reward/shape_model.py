"""Five-DOF, clip-constant affine shape proposals, not a reconstruction model.

A symmetric trace-free log-stretch exponentiates to an SPD matrix with
determinant one. Apply it once to canonical vertices about a fixed supplied
pivot, and retain the original faces for every video frame. In exact arithmetic
the invertible, orientation-preserving map preserves closure, connectivity,
component/cavity winding, and signed volume. It cannot repair invalid input
topology; no runtime watertightness or reconstruction accuracy is asserted.

This deliberately small family changes proportions/shear but not uniform
metric scale. Thickness changes therefore require compensating stretches.
It cannot add local curvature, articulation, holes, or missing surfaces.
Canonical orientation and an optimizer's latent object rotation remain coupled:
the caller must keep one canonical frame and shared shape, not fit per-frame
deformations. No weights, observations, GT, scoring, or fitting are used here.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


DEFAULT_MAX_ABS_LOG_STRETCH = float(np.log(1.5))


def _real_array(value, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} cannot hide invalid entries behind a mask")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array.astype(np.float64, copy=False)


def deformation(params5, *, max_abs_log_stretch: float = DEFAULT_MAX_ABS_LOG_STRETCH) -> np.ndarray:
    """Return an SPD, determinant-one canonical deformation from five controls.

    ``params5 = [a, b, xy, xz, yz]`` specifies the symmetric trace-free matrix
    ``L = [[a,xy,xz], [xy,b,yz], [xz,yz,-a-b]]``; output is ``exp(L)``.
    The bound applies to eigenvalues of L, not individual parameters. Its
    default limits each principal stretch to [1/1.5, 1.5]. Invalid/extreme
    proposals fail without clipping, projecting, absorbing metric scale,
    or introducing a separate rotation/reflection. All arithmetic is float64.
    """
    parameters = _real_array(params5, "params5", (5,))
    limit = _real_array(max_abs_log_stretch, "max_abs_log_stretch", ())
    if limit < 0:
        raise ValueError("max_abs_log_stretch must be nonnegative")
    a, b, xy, xz, yz = parameters
    with np.errstate(over="ignore", invalid="ignore"):
        log_stretch = np.array([[a, xy, xz], [xy, b, yz], [xz, yz, -a - b]], dtype=np.float64)
    if not np.isfinite(log_stretch).all():
        raise ValueError("Trace-free log-stretch exceeds finite float64 range")
    eigenvalues = np.linalg.eigvalsh(log_stretch)
    if not np.isfinite(eigenvalues).all() or np.any(np.abs(eigenvalues) > limit):
        raise ValueError("Principal log-stretch exceeds the declared bound; no clipping")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = expm(log_stretch)
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)
            or np.any(np.linalg.eigvalsh(matrix) <= 0)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-12, rtol=1e-12)):
        raise ValueError("Deformation lost finite SPD determinant-one numerical invariants")
    return matrix


def apply_fixed_shape(
    vertices,
    params5,
    *,
    centroid,
    max_abs_log_stretch: float = DEFAULT_MAX_ABS_LOG_STRETCH,
) -> np.ndarray:
    """Deform canonical ``[V,3]`` vertices once about exactly ``centroid``.

    The caller supplies a generic, fixed mesh pivot, e.g. ``vertices.mean(0)``.
    It is never estimated from camera poses or recomputed after deformation.
    ``v_out = centroid + exp(L) @ (v_in - centroid)``. The supplied pivot is
    invariant; if it is the arithmetic vertex centroid, that centroid is also
    invariant. Inputs/order/padding are untouched; output is a fresh float64
    array. Reuse the input faces unchanged and this one shaped mesh throughout
    the clip. There is no per-frame input, learned parameter, pose or scale.
    """
    positions = _real_array(vertices, "vertices")
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions):
        raise ValueError("vertices must have nonempty shape [V,3]")
    pivot = _real_array(centroid, "centroid", (3,))
    matrix = deformation(params5, max_abs_log_stretch=max_abs_log_stretch)
    with np.errstate(over="ignore", invalid="ignore"):
        # Fixed per-row arithmetic also preserves identical padding vertices
        # bit-for-bit; BLAS can use different rounding in its final row block.
        shaped = np.einsum("ij,kj->ki", matrix, positions - pivot, optimize=False) + pivot
    if not np.isfinite(shaped).all():
        raise ValueError("Fixed shape arithmetic exceeds finite float64 range")
    return shaped
