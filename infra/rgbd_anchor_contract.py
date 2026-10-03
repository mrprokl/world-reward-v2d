"""Pure native-grid contract for the frozen border-proxy depth hypothesis.

Grid stores an immutable centered square-pixel RGB-only camera; K returns a
copy. Border v1 requires dimensions divisible by10: 1536x1152 MUST stop there,
although camera/DA3 resize calculations support that grid. No model/private
imports. Torch is supplied only to runtime wrappers; no calibration is fitted.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np
from world_reward.pointmap import validate_camera_pointmap

MIN_PAIRS, MIN_COVERAGE = 1024, .95
PROCESSED_HW = (392, 518)
ARRAY_KEYS = frozenset({"moge_depth", "moge_points", "moge_validity", "da3_depth", "anchored_depth",
    "anchored_points", "anchored_validity", "K", "scene_id", "frame_id", "scene_anchor"})


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    camera_K: object

    def __post_init__(self):
        if any(type(value) is not int or value < 2 for value in (self.width, self.height)):
            raise ValueError("Original integer RGB grid dimensions>=2 required")
        if np.ma.isMaskedArray(self.camera_K): raise ValueError("Unmasked original camera required")
        raw = np.asarray(self.camera_K)
        if raw.shape != (3, 3) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
            raise ValueError("Finite real3x3 original camera required")
        k = raw.astype(np.float64)
        expected = np.array([[k[0, 0], 0., self.width / 2], [0., k[0, 0], self.height / 2], [0., 0., 1.]])
        if k[0, 0] <= 0 or not np.array_equal(k, expected):
            raise ValueError("Fixed positive centered square-pixel RGB-only camera required")
        object.__setattr__(self, "camera_K", tuple(tuple(float(v) for v in row) for row in k))

    @property
    def K(self): return np.array(self.camera_K, np.float64)

    @property
    def shape(self): return self.height, self.width


def _array(value, shape, dtype, name):
    if np.ma.isMaskedArray(value) or not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
        raise ValueError(name + ": exact original unmasked array shape/dtype required")
    return value


def _positive(value):
    if not np.isfinite(value).all() or np.any(value <= 0): raise ValueError("All original camera-Z pixels must be finite positive; no fill/drop")


def _id(scene, frame):
    if type(scene) is not int or scene < 1 or type(frame) is not int or frame < 0:
        raise ValueError("Explicit original scene/frame integers required")


def border_mask(grid):
    if grid.width % 10 or grid.height % 10: raise ValueError("Border v1 requires RGB dimensions divisible by10; no implicit rounding")
    yy, xx = np.indices(grid.shape); bx, by = grid.width // 10, grid.height // 10
    return (xx < bx) | (xx >= grid.width - bx) | (yy < by) | (yy >= grid.height - by)


def camera_arrays(grid, depth, points, validity, normalized):
    _array(depth, grid.shape, np.float32, "depth"); _array(points, (*grid.shape, 3), np.float32, "points")
    _array(validity, grid.shape, np.bool_, "validity")
    checks = validate_camera_pointmap(depth, points, validity, normalized, grid.K)
    return {"depth": depth, "points": points, "validity": validity, "K": grid.K}, checks


def validate_baseline(grid, baseline):
    if type(baseline) is not dict or set(baseline) != {"depth", "points", "validity", "K"}:
        raise ValueError("Exact unchanged native MoGe array inventory required")
    _array(baseline["K"], (3, 3), np.float64, "K")
    if not np.array_equal(baseline["K"], grid.K): raise ValueError("Original fixed camera changed")
    return camera_arrays(grid, baseline["depth"], baseline["points"], baseline["validity"],
        np.diag([1 / grid.width, 1 / grid.height, 1.]) @ grid.K)[1]


def border_ratio(grid, moge_depth, moge_validity, da3_depth):
    _array(moge_depth, grid.shape, np.float32, "MoGe"); _array(da3_depth, grid.shape, np.float32, "DA3")
    _array(moge_validity, grid.shape, np.bool_, "validity"); _positive(da3_depth); _positive(moge_depth[moge_validity])
    border = border_mask(grid); paired = border & moge_validity; count, total = int(paired.sum()), int(border.sum())
    if count < MIN_PAIRS or count / total < MIN_COVERAGE: raise ValueError("Insufficient fixed border-proxy support; no fallback/sweep")
    ratios = moge_depth[paired].astype(np.float64) / da3_depth[paired].astype(np.float64); _positive(ratios)
    return {"border_pixels": total, "paired_valid_pixels": count, "paired_border_coverage": count / total,
        "minimum_paired_pixels": MIN_PAIRS, "minimum_border_coverage": MIN_COVERAGE, "ratio_median": float(np.median(ratios))}


def scene_anchors(grid, diagnostics, ordered_frames):
    """Return {scene: median4}; caller supplies complete exact ordered ID pairs."""
    if type(ordered_frames) is not list or not ordered_frames or len(ordered_frames) % 4 or type(diagnostics) is not list or len(diagnostics) != len(ordered_frames):
        raise ValueError("Complete caller-declared groups of four ordered frames required")
    for pair in ordered_frames:
        if type(pair) is not tuple or len(pair) != 2: raise ValueError("Explicit scene/frame pairs required")
        _id(*pair)
    if len(set(ordered_frames)) != len(ordered_frames): raise ValueError("No duplicate original frame IDs")
    result = {}; total = int(border_mask(grid).sum())
    for start in range(0, len(ordered_frames), 4):
        scene = ordered_frames[start][0]; values = []
        if scene in result or any(s != scene for s, _ in ordered_frames[start:start + 4]): raise ValueError("Exactly one contiguous four-frame group per scene required")
        for pair, row in zip(ordered_frames[start:start + 4], diagnostics[start:start + 4]):
            expected = dict(scene_id=pair[0], frame_id=pair[1], border_pixels=total, minimum_paired_pixels=MIN_PAIRS, minimum_border_coverage=MIN_COVERAGE)
            if (type(row) is not dict or set(row) != {*expected, "paired_valid_pixels", "paired_border_coverage", "ratio_median"}
                    or any(type(row.get(k)) is not type(v) or row[k] != v for k, v in expected.items())
                    or type(row["paired_valid_pixels"]) is not int or not MIN_PAIRS <= row["paired_valid_pixels"] <= total
                    or type(row["paired_border_coverage"]) is not float or row["paired_border_coverage"] != row["paired_valid_pixels"] / total
                    or row["paired_border_coverage"] < MIN_COVERAGE or type(row["ratio_median"]) is not float
                    or not math.isfinite(row["ratio_median"]) or row["ratio_median"] <= 0):
                raise ValueError("Exact fixed support/ratio/order diagnostics required")
            values.append(row["ratio_median"])
        result[scene] = float(np.median(np.asarray(values, np.float64)))
    return result


def array_identities(arrays):
    return {k: {"sha256": hashlib.sha256(v.tobytes(order="C")).hexdigest(), "dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()}


def _anchored(grid, raw, coefficient):
    if type(coefficient) is not float or not math.isfinite(coefficient) or coefficient <= 0: raise ValueError("One finite positive scene scalar required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        depth = (raw.astype(np.float64) * coefficient).astype(np.float32)
    _positive(depth); yy, xx = np.indices(grid.shape, dtype=np.float64); k = grid.K
    points = np.stack(((xx + .5 - k[0, 2]) / k[0, 0] * depth, (yy + .5 - k[1, 2]) / k[1, 1] * depth, depth), axis=-1).astype(np.float32)
    if not np.isfinite(points).all(): raise ValueError("Full candidate XYZ overflowed")
    return depth, points


def validate_prediction_arrays(grid, data, scene, frame, coefficient):
    _id(scene, frame)
    if type(data) is not dict or set(data) != ARRAY_KEYS: raise ValueError("Exact eleven prediction arrays required")
    for key, value in (("scene_id", scene), ("frame_id", frame)):
        if int(_array(data[key], (), np.int64, key)) != value: raise ValueError("Original scalar IDs changed")
    if float(_array(data["scene_anchor"], (), np.float64, "scene_anchor")) != coefficient: raise ValueError("Original scene coefficient changed")
    baseline = {"depth": data["moge_depth"], "points": data["moge_points"], "validity": data["moge_validity"], "K": data["K"]}
    original = validate_baseline(grid, baseline); _positive(_array(data["da3_depth"], grid.shape, np.float32, "DA3"))
    _array(data["anchored_validity"], grid.shape, np.bool_, "anchored_validity")
    if not np.array_equal(data["anchored_validity"], data["moge_validity"]): raise ValueError("Candidate validity must equal native MoGe, no drop/expansion")
    expected_depth, expected_points = _anchored(grid, data["da3_depth"], coefficient)
    for key, expected in (("anchored_depth", expected_depth), ("anchored_points", expected_points)):
        if _array(data[key], expected.shape, np.float32, key).tobytes(order="C") != expected.tobytes(order="C"):
            raise ValueError("Single-scene coefficient/full +.5 rays must replay byte-exactly")
    anchored = camera_arrays(grid, data["anchored_depth"], data["anchored_points"], np.ones(grid.shape, np.bool_),
        np.diag([1 / grid.width, 1 / grid.height, 1.]) @ grid.K)[1]
    return {"moge": original, "anchored": anchored}


def candidate_arrays(grid, baseline, raw_da3, coefficient, scene, frame):
    _id(scene, frame); validate_baseline(grid, baseline); _positive(_array(raw_da3, grid.shape, np.float32, "DA3"))
    before = array_identities(baseline); depth, points = _anchored(grid, raw_da3, coefficient)
    result = {"moge_depth": baseline["depth"].copy(), "moge_points": baseline["points"].copy(), "moge_validity": baseline["validity"].copy(),
        "da3_depth": raw_da3.copy(), "anchored_depth": depth, "anchored_points": points, "anchored_validity": baseline["validity"].copy(),
        "K": grid.K, "scene_id": np.array(scene, np.int64), "frame_id": np.array(frame, np.int64), "scene_anchor": np.array(coefficient, np.float64)}
    validate_prediction_arrays(grid, result, scene, frame, coefficient)
    if array_identities(baseline) != before: raise ValueError("Original baseline mutated")
    return result


def expected_processed_camera(grid):
    return np.diag([PROCESSED_HW[1] / grid.width, PROCESSED_HW[0] / grid.height, 1.]) @ grid.K


def metric_factor(grid, processed_camera):
    if (np.ma.isMaskedArray(processed_camera) or not isinstance(processed_camera, np.ndarray) or processed_camera.shape != (3, 3)
            or processed_camera.dtype not in (np.float32, np.float64) or not np.isfinite(processed_camera).all()
            or not np.allclose(processed_camera, expected_processed_camera(grid), atol=1e-3, rtol=1e-6)):
        raise ValueError("Actual native392x518 camera must match original RGB resize, no fitted K")
    return (float(processed_camera[0, 0]) + float(processed_camera[1, 1])) / 2 / 300


def resize_metric_depth(torch, grid, depth, processed_camera):
    """One original FP32 metric multiply then bilinear Z to native H,W."""
    _positive(_array(depth, (1, *PROCESSED_HW), np.float32, "native_DA3_depth"))
    metric = torch.from_numpy(depth[None]).cuda() * metric_factor(grid, processed_camera)
    return torch.nn.functional.interpolate(metric, size=grid.shape, mode="bilinear", align_corners=False)


def checked_moge_infer(torch, grid, module, original_alias, network, tensor, diagnostics, context):
    """Restore callback always; return (unchanged native result, camera checks)."""
    if module.recover_focal_shift is not original_alias or not callable(original_alias) or not torch.is_tensor(tensor) or tuple(tensor.shape) != (3, *grid.shape) or tensor.dtype != torch.float32:
        raise ValueError("Original native focal alias and RGB tensor required")
    before = len(diagnostics)
    def checked(points, mask=None, focal=None, downsample_size=(64, 64)):
        if (downsample_size != (64, 64) or not torch.is_tensor(mask) or mask.dtype != torch.bool
                or tuple(mask.shape) != (1, *grid.shape) or not torch.is_tensor(points) or tuple(points.shape) != (1, *grid.shape, 3)):
            raise RuntimeError("Original native full-grid focal support ABI changed")
        sampled = torch.nn.functional.interpolate(mask.float().unsqueeze(1), (64, 64), mode="nearest").squeeze(1) > 0
        count = int(sampled.sum().item()); entry = {**context, "native_nearest64_valid_pixels": count, "focal_prior_supplied": focal is not None, "original_solver_returned": False}; diagnostics.append(entry)
        if not 2 <= count <= 4096: raise ValueError("Native sampled64 fallback rejected")
        result = original_alias(points, mask, focal=focal, downsample_size=downsample_size); entry["original_solver_returned"] = True; return result
    module.recover_focal_shift = checked
    try:
        fov = float(np.degrees(2 * np.arctan(grid.width / (2 * grid.K[0, 0]))))
        with torch.inference_mode(): result = network.infer(tensor[None], fov_x=fov, force_projection=True, apply_mask=False)
        if len(diagnostics) != before + 1: raise RuntimeError("Exactly one original native focal callback required")
        arrays = [result[key][0].detach().cpu().numpy() for key in ("depth", "points", "mask", "intrinsics")]
        return result, camera_arrays(grid, *arrays)[1]
    finally:
        module.recover_focal_shift = original_alias
