"""Frozen scalar TUM sensor-Z scoring; no calibration, fitting or alignment.

Pure NumPy contract. The caller must freeze and validate prediction/source pins
before supplying private sensor values. All original pixels participate, with
only raw sensor zero and unchanged native MoGe missingness excluded.
"""
import math

import numpy as np

SHAPE = (480, 640)
MIN_PIXELS = 1024
MIN_SENSOR_COVERAGE = .25
MIN_NATIVE_SENSOR_COVERAGE = .95
MIN_MEDIAN_GAIN = .05
MAX_SEQUENCE_REGRESSION = .05


def _array(value, dtype, name):
    if (np.ma.isMaskedArray(value) or not isinstance(value, np.ndarray)
            or value.shape != SHAPE or value.dtype != dtype):
        raise ValueError("Exact original full-grid unmasked " + name + " required")
    return value


def score_frame(moge_depth, anchored_depth, moge_validity, anchored_validity, sensor_raw):
    baseline = _array(moge_depth, np.float32, "MoGe Z")
    candidate = _array(anchored_depth, np.float32, "candidate Z")
    native = _array(moge_validity, np.bool_, "MoGe validity")
    validity = _array(anchored_validity, np.bool_, "candidate validity")
    raw = _array(sensor_raw, np.uint16, "uint16 sensor Z")
    if not np.array_equal(native, validity):
        raise ValueError("Candidate validity must equal every original MoGe validity bit")
    for depth in (baseline, candidate):
        if not np.isfinite(depth[native]).all() or np.any(depth[native] <= 0):
            raise ValueError("Every original native-valid prediction must be finite positive")
    sensor_valid = raw > 0
    sensor_count = int(sensor_valid.sum())
    sensor_coverage = sensor_count / raw.size
    if sensor_count < MIN_PIXELS or sensor_coverage < MIN_SENSOR_COVERAGE:
        raise ValueError("Original sensor support below fixed 1024/25% gate")
    paired = sensor_valid & native
    count = int(paired.sum())
    coverage = count / sensor_count
    if coverage < MIN_NATIVE_SENSOR_COVERAGE:
        raise ValueError("Original MoGe covers less than fixed 95% of sensor-valid pixels")
    # Cast before arithmetic; the publisher has already corrected the depth.
    truth = raw[paired].astype(np.float64) / 5000.
    scores = {}
    for name, depth in (("baseline", baseline), ("candidate", candidate)):
        delta = depth[paired].astype(np.float64) - truth
        absolute = np.abs(delta)
        scores[name] = dict(AbsRel=float(np.mean(absolute / truth)),
            RMSE_m=float(np.sqrt(np.mean(delta * delta))),
            median_signed_error_m=float(np.median(delta)),
            median_absolute_error_m=float(np.median(absolute)))
        if any(not math.isfinite(value) for value in scores[name].values()):
            raise ValueError("Finite scalar sensor-Z scores required")
    return dict(sensor_valid_pixels=sensor_count, sensor_grid_coverage=sensor_coverage,
        paired_pixels=count, native_sensor_coverage=coverage,
        exact_candidate_validity_matches_baseline=True,
        sensor_factor=5000, alignment_performed=False, **scores)


def decision(rows):
    expected = [(scene, frame) for scene in (1, 2, 3) for frame in (40, 80, 120, 160)]
    if (type(rows) is not list or len(rows) != 12
            or [(row.get("scene_id"), row.get("frame_id")) for row in rows] != expected):
        raise ValueError("All twelve original ordered records must contribute")
    sequences = []
    for offset, scene in zip((0, 4, 8), (1, 2, 3)):
        group = rows[offset:offset + 4]
        means = {}
        for name in ("baseline", "candidate"):
            values = [row[name]["AbsRel"] for row in group]
            if any(type(value) is not float or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("Finite nonnegative per-frame AbsRel scores required")
            means[name] = float(np.mean(values))
        if means["baseline"] <= 0:
            raise ValueError("Zero baseline AbsRel cannot be divided or called a perfect gain")
        if any(row.get("exact_candidate_validity_matches_baseline") is not True
                or row.get("sensor_valid_pixels", 0) < MIN_PIXELS
                or row.get("sensor_grid_coverage", 0) < MIN_SENSOR_COVERAGE
                or row.get("native_sensor_coverage", 0) < MIN_NATIVE_SENSOR_COVERAGE
                or row.get("alignment_performed") is not False for row in group):
            raise ValueError("Every original coverage/validity/no-alignment gate must pass")
        sequences.append(dict(scene_id=scene, **means,
            relative_AbsRel_gain=(means["baseline"] - means["candidate"]) / means["baseline"]))
    gains = [row["relative_AbsRel_gain"] for row in sequences]
    median = float(np.median(gains))
    supported = median >= MIN_MEDIAN_GAIN and min(gains) >= -MAX_SEQUENCE_REGRESSION
    return dict(sequences=sequences, median_sequence_relative_AbsRel_gain=median,
        scientific_decision="SUPPORT_NARROW_NEW_RECORDING_DEPTH_HYPOTHESIS" if supported else "REJECT",
        depth_hypothesis_supported=supported, adoption_performed=False,
        training_overlap_verified=False, calibration_accuracy_verified=False,
        human_object_temporal_quality_verified=False, verified_victory_over_CARI4D=False)
