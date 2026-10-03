"""Tiny manufactured scalar depth/validity tests, never challenge labels."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("wr_tum_quality", Path(__file__).resolve().parents[1] / "infra/tum_depth_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def arrays():
    return [np.full(quality.SHAPE, 2, np.float32), np.full(quality.SHAPE, 1.5, np.float32),
        np.ones(quality.SHAPE, np.bool_), np.ones(quality.SHAPE, np.bool_),
        np.full(quality.SHAPE, 5000, np.uint16)]


def test_absolute_sensor_factor_no_alignment_or_second_correction():
    values = arrays(); before = [v.tobytes() for v in values]
    row = quality.score_frame(*values)
    assert row["baseline"]["AbsRel"] == row["baseline"]["RMSE_m"] == 1.
    assert row["candidate"]["AbsRel"] == row["candidate"]["RMSE_m"] == .5
    assert row["native_sensor_coverage"] == 1. and row["paired_pixels"] == 307200
    assert row["sensor_factor"] == 5000 and row["alignment_performed"] is False
    assert [v.tobytes() for v in values] == before


@pytest.mark.parametrize("index", range(5))
def test_original_grid_dtype_required(index):
    values = arrays(); values[index] = values[index].astype(np.float64)
    with pytest.raises(ValueError): quality.score_frame(*values)
    values = arrays(); values[index] = values[index][:-1]
    with pytest.raises(ValueError): quality.score_frame(*values)


@pytest.mark.parametrize("fault", ["validity", "missing_sensor", "low_sensor", "low_native", "nan", "negative", "masked"])
def test_no_drop_fill_clip_or_prediction_dependent_pixel_filter(fault):
    values = arrays()
    if fault == "validity": values[3][0, 0] = False
    elif fault == "missing_sensor": values[4].fill(0)
    elif fault == "low_sensor": values[4][100:] = 0
    elif fault == "low_native": values[2][100:] = False; values[3] = values[2].copy()
    elif fault == "nan": values[1][0, 0] = np.nan
    elif fault == "negative": values[0][0, 0] = -1
    else: values[4] = np.ma.array(values[4], mask=False)
    with pytest.raises(ValueError): quality.score_frame(*values)


def rows():
    row = quality.score_frame(*arrays())
    return [dict(row, scene_id=scene, frame_id=frame) for scene in (1, 2, 3) for frame in (40, 80, 120, 160)]


def test_paired_equal_four_mean_and_median_gate():
    result = quality.decision(rows())
    assert result["depth_hypothesis_supported"] is True
    assert result["median_sequence_relative_AbsRel_gain"] == .5
    assert not result["adoption_performed"] and not result["verified_victory_over_CARI4D"]


def test_one_sequence_regression_rejected_even_when_median_improves():
    values = rows()
    for row in values[:4]: row["candidate"] = dict(row["candidate"], AbsRel=1.06)
    assert quality.decision(values)["depth_hypothesis_supported"] is False


@pytest.mark.parametrize("fault", ["order", "missing", "zero", "nonfinite", "coverage", "aligned"])
def test_no_invalid_or_missing_record_can_manufacture_win(fault):
    values = rows()
    if fault == "order": values.reverse()
    elif fault == "missing": values.pop()
    elif fault == "zero":
        for row in values[:4]: row["baseline"] = dict(row["baseline"], AbsRel=0.)
    elif fault == "nonfinite": values[0]["candidate"] = dict(values[0]["candidate"], AbsRel=float("nan"))
    elif fault == "coverage": values[0]["native_sensor_coverage"] = .94
    else: values[0]["alignment_performed"] = True
    with pytest.raises(ValueError): quality.decision(values)
