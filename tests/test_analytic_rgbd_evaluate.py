"""Tiny own geometry and frozen byte contracts, no remote assets or inference."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "infra")); monkeypatch.syspath_prepend(str(root / "src"))
    spec = importlib.util.spec_from_file_location("wr_authored_quality", root / "infra/analytic_rgbd_evaluate.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def tiny(gate):
    grid = gate.contract.Grid(10, 8, ((10., 0., 5.), (0., 10., 4.), (0., 0., 1.)))
    yy, xx = np.indices(grid.shape); depth = np.full(grid.shape, 2., np.float32)
    points = np.stack(((xx + .5 - 5) * depth / 10, (yy + .5 - 4) * depth / 10, depth), -1).astype(np.float32)
    data = dict(moge_points=points, anchored_points=points.copy(), moge_depth=depth, anchored_depth=depth.copy(),
        moge_validity=np.ones(grid.shape, bool), anchored_validity=np.ones(grid.shape, bool))
    truth = dict(depth=depth.copy(), visible=np.ones(grid.shape, bool), K=grid.K, scene_id=np.array(1, np.int64),
        frame_id=np.array(0, np.int64), camera_R=np.eye(3), camera_t=np.zeros(3))
    return grid, data, truth


def test_camera_truth_plus_half_not_integer_pixels_no_alignment(gate):
    grid, data, truth = tiny(gate); result = gate.score_frame(grid, data, truth, 1, 0)
    assert result["moge"]["visible_camera_chamfer_half_cm"] < 1e-5
    assert result["coverage_gate_pass"] and result["samples"] == 80
    data["anchored_points"][..., 2] += .4; data["anchored_depth"] += .4
    result = gate.score_frame(grid, data, truth, 1, 0)
    assert result["anchored"]["visible_camera_chamfer_half_cm"] > 39
    assert result["anchored"]["camera_Z_mae_cm"] > 39


@pytest.mark.parametrize("fault", ["extra", "nan", "mask", "K", "scene", "pose", "empty", "candidate_drop"])
def test_truth_strict_full_grid_no_repairs(gate, fault):
    grid, data, truth = tiny(gate)
    if fault == "extra": truth["oracle"] = np.array(1)
    elif fault == "nan": truth["depth"][0, 0] = np.nan
    elif fault == "mask": truth["visible"] = truth["visible"].astype(np.uint8)
    elif fault == "K": truth["K"][0, 0] = 11.
    elif fault == "scene": truth["scene_id"] = np.array(True)
    elif fault == "pose": truth["camera_R"][0, 0] = 2.
    elif fault == "empty": truth["visible"][:] = False
    else: data["anchored_validity"][0, 0] = False
    with pytest.raises(ValueError): gate.score_frame(grid, data, truth, 1, 0)


def test_coverage_drop_fails_scientific_gate(gate):
    grid, data, truth = tiny(gate)
    data["moge_validity"][:2] = False; data["anchored_validity"][:2] = False
    result = gate.score_frame(grid, data, truth, 1, 0)
    assert not result["coverage_gate_pass"] and result["moge"]["object_coverage"] == .75


def test_frozen_scene_equalweight_gates_not_selected_frames(gate):
    rows = [dict(scene_id=s, frame_id=f, moge={"visible_camera_chamfer_half_cm": 100.},
        anchored={"visible_camera_chamfer_half_cm": 90.}, coverage_gate_pass=True) for s, f in gate.inputs.ORDERED_FRAMES]
    result = gate.decision(rows)
    assert result["analytic_object_depth_hypothesis_supported"] and result["median_paired_scene_relative_gain"] == .1
    assert not result["verified_victory_over_CARI4D"] and not result["adoption_performed"]
    altered = copy.deepcopy(rows)
    for row in altered[:4]: row["anchored"]["visible_camera_chamfer_half_cm"] = 106.
    assert not gate.decision(altered)["analytic_object_depth_hypothesis_supported"]
    rows[0]["coverage_gate_pass"] = False
    assert not gate.decision(rows)["analytic_object_depth_hypothesis_supported"]
    with pytest.raises(ValueError): gate.decision(rows[:-1])


def test_sampling_original_PCG64_no_sweep(gate):
    np.testing.assert_array_equal(gate.sample_indices(19), np.arange(19))
    selected = gate.sample_indices(10000)
    np.testing.assert_array_equal(selected, np.sort(np.random.default_rng(0).choice(10000, 8192, replace=False)))
    with pytest.raises(ValueError): gate.sample_indices(0)


@pytest.mark.parametrize("fault", ["schema", "bytes", "producer", "outputs", "coefficient", "extra"])
def test_independent_prediction_pins_failclosed(gate, fault):
    ident = dict(bytes=1, sha256="a" * 64)
    pins = dict(schema="world_reward.analytic_rgbd_prediction_pins.v1", report={**ident, "producer_revision": "b" * 40,
        "script_sha256": "c" * 64}, outputs={Path(n).stem + ".npz": ident.copy() for n in gate.inputs.filenames()}, coefficients=[1., 1., 1.])
    gate.validate_pins(pins)
    if fault == "schema": pins["schema"] = "TUD-L"
    elif fault == "bytes": pins["report"]["bytes"] = True
    elif fault == "producer": pins["report"]["producer_revision"] = "a" * 39
    elif fault == "outputs": pins["outputs"].pop(next(iter(pins["outputs"])))
    elif fault == "coefficient": pins["coefficients"][0] = np.nan
    else: pins["private_truth"] = "oracle"
    with pytest.raises(ValueError): gate.validate_pins(pins)


def test_actual_source_wrapper_no_gpu_model_or_private_prediction_access(gate):
    root = Path(__file__).resolve().parents[1]
    source = (root / "infra/run_analytic_rgbd_evaluate.sh").read_text()
    assert "--gpus" not in source and "weights/" not in source and "--network none" in source
    assert "183s docker run" in source and "--kill-after=10s" in source
    assert 'src=$BASE/anchor_predictions_v1,dst=$BASE/anchor_predictions_v1,readonly' in source
    assert 'src=$BASE/eval_private,dst=$BASE/eval_private,readonly' in source
    assert '--name "$CONTAINER"' in source and 'docker kill "$CONTAINER"' in source


def test_independent_analytic_geometry_and_truth_nearest_root(gate):
    # Tiny grid only. Geometry manufactured analytically here; no dataset/model.
    grid = gate.contract.Grid(10, 8, ((40., 0., 5.), (0., 40., 4.), (0., 0., 1.)))
    protocol = {"objects": [{"semiaxes_m": [.23, .17, .21], "material_seed": 101,
        "background_z_m": 3.3, "depth_m": 1.75, "start_angle_rad": -.22, "angle_extent_rad": .72}],
        "motion": {"instants": [0., 1/3, 2/3, 1.], "axis": [.2, 1., .3]},
        "appearance": {"background_extent_xy_m": [2., 1.6]}}
    geometry = {"semiaxes_m": np.array([.23,.17,.21]), "material_seed": np.array(101,np.int64),
        "plane_z_m": np.array(3.3), "plane_extent_xy_m": np.array([2.,1.6])}
    r,t = gate.expected_pose(protocol,1,0); yy,xx=np.indices(grid.shape)
    d=np.stack(((xx+.5-5)/40,(yy+.5-4)/40,np.ones(grid.shape)),axis=-1)
    v=(d@r)/geometry["semiaxes_m"];o=(-t@r)/geometry["semiaxes_m"]
    a=np.sum(v*v,axis=-1);b=2*np.sum(v*o,axis=-1);c=np.sum(o*o)-1
    disc=b*b-4*a*c;nearest=(-b-np.sqrt(np.maximum(disc,0)))/(2*a)
    visible=(disc>0)&(nearest>0)&(nearest<3.3)
    truth=dict(depth=np.where(visible,nearest,3.3).astype(np.float32),visible=visible,K=grid.K,
        camera_R=r,camera_t=t,scene_id=np.array(1,np.int64),frame_id=np.array(0,np.int64))
    result=gate.validate_analytic_truth(grid,truth,geometry,protocol,1,0)
    assert result["checked_pixels"]==80 and result["visible_pixels"]>0
    for mode in ("pose","far_root","mask","shape","material","plane","dtype","recipe"):
        altered=copy.deepcopy(truth); g=copy.deepcopy(geometry); p=copy.deepcopy(protocol)
        if mode=="pose":altered["camera_t"][0]+=.01
        elif mode=="far_root":altered["depth"][visible]+=.1
        elif mode=="mask":altered["visible"][0,0]=not altered["visible"][0,0]
        elif mode=="shape":g["semiaxes_m"][0]*=.9
        elif mode=="material":g["material_seed"]=np.array(102,np.int64)
        elif mode=="plane":g["plane_z_m"]=np.array(3.4)
        elif mode=="dtype":altered["depth"]=altered["depth"].astype(np.float64)
        else:p["objects"][0]["depth_m"]+=.02
        with pytest.raises(ValueError):gate.validate_analytic_truth(grid,altered,g,p,1,0)
