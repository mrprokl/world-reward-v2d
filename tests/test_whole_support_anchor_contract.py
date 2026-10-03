"""Tiny pure whole-support fixtures; no data, models or private labels."""
import copy
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def contract(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "src")); monkeypatch.syspath_prepend(str(root / "infra"))
    return importlib.import_module("whole_support_anchor_contract")


def inputs(c, count=1024, width=64, height=64):
    g = c.Grid(width, height, [[800., 0., width / 2], [0., 800., height / 2], [0., 0., 1.]])
    moge = np.full(g.shape, np.nan, np.float32); valid = np.zeros(g.shape, np.bool_)
    valid.flat[:count] = True; moge[valid] = 2.
    return g, moge, valid, np.ones(g.shape, np.float32)


def test_exact1024_and25percent_support_pass_without_border(contract):
    g, moge, valid, da3 = inputs(contract)
    before = tuple(x.tobytes() for x in (moge, valid, da3))
    row = contract.whole_support_ratio(g, moge, valid, da3)
    assert row["passed"] is True and row["reasons"] == []
    assert row["whole_grid_pixels"] == 4096 and row["native_valid_pixels"] == row["paired_valid_pixels"] == 1024
    assert row["paired_grid_coverage"] == .25 and row["ratio_median"] == 2.
    assert row["log_ratio_q10"] == row["log_ratio_q50"] == row["log_ratio_q90"] == np.log(2.)
    assert row["log_ratio_iqr"] == 0. and contract.require_support(row) is row
    assert tuple(x.tobytes() for x in (moge, valid, da3)) == before


@pytest.mark.parametrize("count,width,height,reason", [(1023, 64, 64, "paired_pixels"),
    (1024, 65, 64, "coverage"), (0, 64, 64, "paired_pixels")])
def test_support_failure_records_all_available_stats_before_gate(contract, count, width, height, reason):
    row = contract.whole_support_ratio(*inputs(contract, count, width, height))
    assert not row["passed"] and any(reason in item for item in row["reasons"])
    assert row["paired_valid_pixels"] == count and row["paired_grid_coverage"] == count / (width * height)
    assert row["ratio_median"] == (2. if count else None)
    assert row["log_ratio_iqr"] == (0. if count else None)
    with pytest.raises(ValueError, match="no fallback"): contract.require_support(row)


@pytest.mark.parametrize("fault", ["native_nan", "native_zero", "native_negative", "da3_valid_nan", "da3_invalid_nan", "da3_zero"])
def test_numeric_corruption_fails_not_filtered_to_make_anchor(contract, fault):
    g, moge, valid, da3 = inputs(contract, 2048)
    if fault.startswith("native_"): moge.flat[0] = {"native_nan": np.nan, "native_zero": 0., "native_negative": -1.}[fault]
    elif fault == "da3_invalid_nan": da3.flat[-1] = np.nan
    else: da3.flat[0] = 0. if fault == "da3_zero" else np.nan
    row = contract.whole_support_ratio(g, moge, valid, da3)
    assert not row["passed"] and row["ratio_median"] == 2. and row["log_ratio_iqr"] == 0.
    assert row["native_valid_pixels"] == 2048
    assert row["paired_valid_pixels"] == (2048 if fault == "da3_invalid_nan" else 2047)
    with pytest.raises(ValueError, match="no fallback"): contract.require_support(row)


@pytest.mark.parametrize("fault", ["moge_dtype", "valid_dtype", "da3_shape", "moge_masked", "grid"])
def test_array_fault_diagnostics_are_recordable_before_throw(contract, fault):
    g, moge, valid, da3 = inputs(contract)
    if fault == "moge_dtype": moge = moge.astype(np.float64)
    elif fault == "valid_dtype": valid = valid.astype(np.uint8)
    elif fault == "da3_shape": da3 = da3[:-1]
    elif fault == "moge_masked": moge = np.ma.array(moge)
    else: g = None
    row = contract.whole_support_ratio(g, moge, valid, da3)
    assert row["passed"] is False and row["reasons"] and set(row) == contract.DIAGNOSTIC_KEYS
    with pytest.raises(ValueError): contract.require_support(row)


def test_dispersion_is_recorded_not_filtered_or_confidence_weighted(contract):
    g, moge, valid, da3 = inputs(contract, 2048)
    moge[valid] = np.linspace(.01, 100., 2048, dtype=np.float32)
    row = contract.whole_support_ratio(g, moge, valid, da3)
    ratios = moge[valid].astype(np.float64) / da3[valid].astype(np.float64)
    qs = np.quantile(np.log(ratios), [.1, .25, .5, .75, .9])
    assert row["passed"] and row["ratio_median"] == float(np.median(ratios))
    assert [row[k] for k in ("log_ratio_q10", "log_ratio_q50", "log_ratio_q90", "log_ratio_iqr")] == [qs[0], qs[2], qs[4], qs[3] - qs[1]]


@pytest.mark.parametrize("fault", ["count", "coverage", "threshold", "nan", "order", "reasons", "extra", "native_count"])
def test_gate_rejects_tampered_successful_diagnostics(contract, fault):
    row = contract.whole_support_ratio(*inputs(contract))
    if fault == "count": row["paired_valid_pixels"] -= 1
    elif fault == "coverage": row["paired_grid_coverage"] = .9
    elif fault == "threshold": row["minimum_grid_coverage"] = .20
    elif fault == "nan": row["ratio_median"] = float("nan")
    elif fault == "order": row["log_ratio_q10"] = 100.
    elif fault == "reasons": row["reasons"].append("failed")
    elif fault == "native_count": row["native_valid_pixels"] += 1
    else: row["confidence"] = 1.
    with pytest.raises(ValueError): contract.require_support(row)


def test_scene_equal_median4_order_exact_and_genuine_candidate_imports(contract):
    import rgbd_anchor_contract as original
    g, moge, valid, da3 = inputs(contract); order = [(7, 0), (7, 1), (7, 2), (7, 3)]
    rows = []
    for pair, value in zip(order, [1., 3., 8., 100.]):
        moge[valid] = value; row = contract.whole_support_ratio(g, moge, valid, da3)
        row.update(scene_id=pair[0], frame_id=pair[1]); rows.append(row)
    assert contract.scene_anchors(g, rows, order) == {7: 5.5}
    for bad_order in (list(reversed(order)), order[:-1], order[:2] * 2):
        with pytest.raises(ValueError): contract.scene_anchors(g, rows, bad_order)
    bad_rows = copy.deepcopy(rows); bad_rows[0]["passed"] = False
    with pytest.raises(ValueError): contract.scene_anchors(g, bad_rows, order)
    for name in ("Grid", "camera_arrays", "candidate_arrays", "validate_prediction_arrays", "checked_moge_infer", "resize_metric_depth"):
        assert getattr(contract, name) is getattr(original, name)
