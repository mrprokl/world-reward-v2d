"""Prospective, authored conditional point study; no native/anatomical certification.

The SAME-SAM scale mean is supplied unchanged by the caller. Its soft-limit
violations are retained explicitly; hard articulated/identity bounds remain
mandatory. This policy does not change the closed authored_point_study cohort.
Rendering, pad/attachment geometry, PCA reconstruction and native decoding reuse
the original helpers in that module/caller, not new approximations here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .authored_point_study import NamedRecipe

FRAMES = 24
KNOTS = (0, 5, 11, 17, 23)
SCENES = (("knot_front", "development", "l"),
          ("knot_cross", "development", "r"),
          ("knot_double", "reserved", "l"),
          ("knot_return", "reserved", "r"),
          ("knot_delayed", "reserved", "l"),
          ("knot_alternate", "reserved", "r"))
KNOT_VALUES = (
    ((0., 1., .25, .75, .2), (.2, .5, 1., .25, .6)),
    ((.2, .75, 1., .25, .5), (.8, .2, .5, 1., .1)),
    ((.1, .9, .2, .8, .3), (.7, .3, .9, .1, .6)),
    ((.8, .2, .7, .1, .5), (.1, .8, .3, .9, .2)),
    ((.2, .3, .9, .6, .1), (.9, .6, .2, .4, .8)),
    ((.6, .1, .8, .3, .9), (.3, .9, .1, .7, .4)))
CONTROLS = (("uparm_ry", .10, .04, 0), ("elbow_bend", .40, .06, 1),
            ("thumb2_rz", .08, .04, 2), ("index1_rz", .16, .06, 2),
            ("index2_rz", .10, .05, 1), ("middle1_rz", .14, .05, 0))
LIMIT_POLICY = "hard_articulation_and_identity_report_unchanged_sam_scale_mean"


@dataclass(frozen=True)
class ScaleViolation:
    """One clip-constant soft-limit excess, in original native249 coordinates."""
    parameter_index: int
    actual: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ArticulatedPointRecipe(NamedRecipe):
    perturbation: np.ndarray
    scale_violations: tuple[ScaleViolation, ...]


def _readonly(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _array(value, shape, name):
    a = np.asarray(value)
    if (np.ma.isMaskedArray(value) or a.dtype != np.float32
            or a.shape != shape or not np.isfinite(a).all()):
        raise ValueError(name + ": finite original float32 shape required")
    return a


def _bounds(value):
    b = np.asarray(value)
    if (np.ma.isMaskedArray(value) or b.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or b.shape != (249, 2) or np.isnan(b).any() or np.any(b[:, 0] > b[:, 1])):
        raise ValueError("Original249 ordered float bounds required; infinities denote open limits")
    return b


def validate_reconstituted_controls(original249, bounds):
    """Check actual post-PCA controls, not a guessed hand mapping/anatomy gate.

    All24 frames and all249 finite controls are mandatory. Scales136:204 and
    identity204:249 must be clip-constant. Returns at most68 immutable records;
    each records an excess shared by ALL24 frames, never an ignored-value mask.
    The caller must additionally bind reconstructed values to its original
    recipe and actual decoder/source identities.
    """
    x = _array(original249, (FRAMES, 249), "reconstituted controls")
    b = _bounds(bounds)
    if not np.array_equal(x[:, 136:], np.broadcast_to(x[0, 136:], (FRAMES, 113))):
        raise ValueError("Clip-constant original scale and identity required")
    for lo, hi in ((0, 136), (204, 249)):
        if np.any(x[:, lo:hi] < b[lo:hi, 0]) or np.any(x[:, lo:hi] > b[lo:hi, 1]):
            raise ValueError("Hard articulation/root/identity limit exceeded; no clipping")
    indices = np.flatnonzero((x[0, 136:204] < b[136:204, 0])
                             | (x[0, 136:204] > b[136:204, 1])) + 136
    return tuple(ScaleViolation(int(i), float(x[0, i]), float(b[i, 0]), float(b[i, 1]))
                 for i in indices)


def _curves(index):
    values = np.asarray(KNOT_VALUES[index], np.float64)
    result = np.empty((2, FRAMES), np.float64)
    for segment, (start, stop) in enumerate(zip(KNOTS[:-1], KNOTS[1:])):
        u = np.arange(stop - start + 1, dtype=np.float64) / (stop - start)
        ease = 3. * u ** 2 - 2. * u ** 3
        result[:, start:stop + 1] = (values[:, segment, None]
            + (values[:, segment + 1] - values[:, segment])[:, None] * ease)
    result[:, KNOTS] = values  # Literal source knot values, not rounded endpoint subtraction.
    return result[0], result[1], (result[0] + result[1]) / 2.


def named_recipe(names, bounds, scene_index, *, scale68, identity45):
    """Fixed native204 recipe plus new original-frame translation perturbation.

    Scale68 is the externally authenticated SAM mean, not a fitted/clamped scale.
    No interpolation/subsetting changes, native/PCA guesses or future evidence.
    """
    if type(scene_index) is not int or not 0 <= scene_index < len(SCENES):
        raise ValueError("Frozen scene index0..5 required")
    if (not isinstance(names, (list, tuple)) or len(names) != 249
            or any(type(n) is not str or not n for n in names) or len(set(names)) != 249):
        raise ValueError("249 unique actual parameter names required")
    b = _bounds(bounds)
    scale = _array(scale68, (68,), "actual SAM scale mean")
    identity = _array(identity45, (45,), "identity45")
    scene, split, side = SCENES[scene_index]
    curves = _curves(scene_index)
    q = np.zeros((FRAMES, 204), np.float32)
    for suffix, base, amplitude, curve in CONTROLS:
        name = side + "_" + suffix
        if name not in names or not 6 <= names.index(name) < 136:
            raise ValueError("Actual named articulation absent or outside pose ABI: " + name)
        q[:, names.index(name)] = base + amplitude * curves[curve]
    q[:, 136:] = scale
    shape = np.broadcast_to(identity, (FRAMES, 45))
    violations = validate_reconstituted_controls(np.column_stack((q, shape)), b)
    a, bv, c = curves
    sign = -1. if side == "l" else 1.
    delta = np.column_stack((sign * .018 * (a - a[0]),
                             .012 * (bv - bv[0]), .016 * (c - c[0]))).astype(np.float32)
    delta[0] = 0.  # Byte-exact unperturbed initializer, including the left-hand signed zero.
    return ArticulatedPointRecipe(scene, split, side, _readonly(q), _readonly(shape),
        _readonly(np.zeros((FRAMES, 72), np.float32)),
        _readonly(np.arange(FRAMES, dtype=np.int64)), _readonly(delta), violations)
