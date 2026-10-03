"""New whole-native-support hypothesis; never a fallback for the border recipe.

Record ``whole_support_ratio`` in the run receipt BEFORE ``require_support``.
Failed diagnostics retain every available scalar, but must never make an
anchor. All camera, native inference and candidate numerics are genuine imports
of the unchanged original grid contract. No labels, models or semantic masks.
"""
import math

import numpy as np
from rgbd_anchor_contract import (Grid, ARRAY_KEYS, PROCESSED_HW, array_identities,
    camera_arrays, candidate_arrays, checked_moge_infer, expected_processed_camera,
    metric_factor, resize_metric_depth, validate_baseline, validate_prediction_arrays)

MIN_PAIRS, MIN_COVERAGE = 1024, .25
STAT_KEYS = ("ratio_median", "log_ratio_q10", "log_ratio_q50", "log_ratio_q90", "log_ratio_iqr")
DIAGNOSTIC_KEYS = frozenset({"passed", "reasons", "whole_grid_pixels", "native_valid_pixels",
    "paired_valid_pixels", "paired_grid_coverage", "minimum_paired_pixels", "minimum_grid_coverage", *STAT_KEYS})


def whole_support_ratio(grid, moge_depth, moge_validity, da3_depth):
    """JSON-safe diagnostics, including numeric/array failures; no early throw.

    Numeric subsets are used ONLY to describe a failed run: any bad native-valid
    MoGe pixel or any bad full-grid DA3 pixel fails, rather than being dropped.
    On PASS the median uses ALL original MoGe-valid pixels, with no weighting.
    """
    total = grid.width * grid.height if isinstance(grid, Grid) else None
    row = dict(passed=False, reasons=[], whole_grid_pixels=total, native_valid_pixels=None,
        paired_valid_pixels=None, paired_grid_coverage=None, minimum_paired_pixels=MIN_PAIRS,
        minimum_grid_coverage=MIN_COVERAGE, **dict.fromkeys(STAT_KEYS))
    if total is None:
        row["reasons"].append("original_grid_required")
        return row
    good = {}
    for name, value, dtype in (("moge_depth", moge_depth, np.float32),
            ("moge_validity", moge_validity, np.bool_), ("da3_depth", da3_depth, np.float32)):
        good[name] = (isinstance(value, np.ndarray) and not np.ma.isMaskedArray(value)
            and value.shape == grid.shape and value.dtype == dtype)
        if not good[name]: row["reasons"].append(name + "_original_shape_dtype_required")
    if good["moge_validity"]: row["native_valid_pixels"] = int(moge_validity.sum())
    if not all(good.values()): return row
    native_positive = np.isfinite(moge_depth) & (moge_depth > 0)
    da3_positive = np.isfinite(da3_depth) & (da3_depth > 0)
    if not native_positive[moge_validity].all(): row["reasons"].append("bad_native_valid_moge_depth_no_drop")
    if not da3_positive.all(): row["reasons"].append("bad_full_grid_da3_depth_no_fill")
    paired = moge_validity & native_positive & da3_positive
    count = int(paired.sum()); coverage = count / total
    row.update(paired_valid_pixels=count, paired_grid_coverage=coverage)
    if count:
        ratios = moge_depth[paired].astype(np.float64) / da3_depth[paired].astype(np.float64)
        q10, q25, q50, q75, q90 = np.quantile(np.log(ratios), [.1, .25, .5, .75, .9])
        row.update(ratio_median=float(np.median(ratios)), log_ratio_q10=float(q10),
            log_ratio_q50=float(q50), log_ratio_q90=float(q90), log_ratio_iqr=float(q75 - q25))
    if count < MIN_PAIRS: row["reasons"].append("insufficient_whole_grid_paired_pixels")
    if coverage < MIN_COVERAGE: row["reasons"].append("insufficient_whole_grid_coverage")
    row["passed"] = not row["reasons"]
    return row


def require_support(row):
    """Gate an already-recorded diagnostic; reject malformed/tampered PASS rows."""
    if type(row) is not dict or set(row) not in (DIAGNOSTIC_KEYS, DIAGNOSTIC_KEYS | {"scene_id", "frame_id"}):
        raise ValueError("Exact recorded whole-support diagnostics required")
    if type(row["passed"]) is not bool or type(row["reasons"]) is not list or any(type(x) is not str for x in row["reasons"]):
        raise ValueError("Explicit recorded pass/reasons required")
    if not row["passed"] or row["reasons"]:
        raise ValueError("Whole-native support failed; no fallback: " + ", ".join(row["reasons"]))
    total, native, paired = (row[k] for k in ("whole_grid_pixels", "native_valid_pixels", "paired_valid_pixels"))
    if (type(total) is not int or total < 4 or type(native) is not int or type(paired) is not int
            or not MIN_PAIRS <= paired == native <= total
            or type(row["minimum_paired_pixels"]) is not int or row["minimum_paired_pixels"] != MIN_PAIRS
            or type(row["minimum_grid_coverage"]) is not float or row["minimum_grid_coverage"] != MIN_COVERAGE
            or type(row["paired_grid_coverage"]) is not float or row["paired_grid_coverage"] != paired / total
            or row["paired_grid_coverage"] < MIN_COVERAGE
            or any(type(row[k]) is not float or not math.isfinite(row[k]) for k in STAT_KEYS)
            or row["ratio_median"] <= 0 or row["log_ratio_iqr"] < 0
            or not row["log_ratio_q10"] <= row["log_ratio_q50"] <= row["log_ratio_q90"]):
        raise ValueError("Recorded whole-support scalar contract changed")
    return row


def scene_anchors(grid, diagnostics, ordered_frames):
    """Return one positive median4 per scene, preserving the declared order."""
    if (not isinstance(grid, Grid) or type(ordered_frames) is not list or not ordered_frames
            or len(ordered_frames) % 4 or type(diagnostics) is not list or len(diagnostics) != len(ordered_frames)):
        raise ValueError("Complete declared groups of four ordered frames required")
    for pair in ordered_frames:
        if (type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not int or pair[0] < 1
                or type(pair[1]) is not int or pair[1] < 0):
            raise ValueError("Explicit original scene/frame integer pairs required")
    if len(set(ordered_frames)) != len(ordered_frames): raise ValueError("Duplicate original frame IDs")
    result = {}
    for start in range(0, len(ordered_frames), 4):
        scene = ordered_frames[start][0]; values = []
        if scene in result or any(s != scene for s, _ in ordered_frames[start:start + 4]):
            raise ValueError("One contiguous group of four frames per scene required")
        for pair, row in zip(ordered_frames[start:start + 4], diagnostics[start:start + 4]):
            require_support(row)
            if (set(row) != DIAGNOSTIC_KEYS | {"scene_id", "frame_id"}
                    or type(row["scene_id"]) is not int or type(row["frame_id"]) is not int
                    or (row["scene_id"], row["frame_id"]) != pair
                    or row["whole_grid_pixels"] != grid.width * grid.height):
                raise ValueError("Original whole-grid diagnostics/order changed")
            values.append(row["ratio_median"])
        result[scene] = float(np.median(np.asarray(values, np.float64)))
    return result
