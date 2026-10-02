"""Experimental pinhole calibration from predicted up/latitude fields.

Own ray-projection mathematics and SciPy optimization; no GeoCalib, PF or LM
implementation imports. Confidence is not calibrated uncertainty. An accepted
field fit is not a physically verified camera; failed gates return the explicit
image-diagonal focal fallback, never an invented successful calibration.
"""
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares

STRIDE = 8
MIN_ESS = 256
MIN_CELL_ESS = 8
MIN_MEAN_CONFIDENCE = 1e-6  # Numerical evidence floor, not calibrated certainty.
ROBUST_SCALE_RAD = np.deg2rad(2.)


@dataclass(frozen=True)
class PerspectiveCalibration:
    accepted: bool
    reason: str
    focal_px: float
    gravity: tuple[float, float, float] | None
    K: tuple[tuple[float, float, float], ...]
    fallback_focal_px: float
    diagnostics: dict

    def to_dict(self):
        return {"schema": "world-reward-perspective-field-calibration-v1", **asdict(self),
                "accuracy_verified": False, "ground_truth_used": False,
                "confidence_is_calibrated_uncertainty": False,
                "pixel_convention": "integer field indices; original edge-center mapped then minus 0.5"}


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name+" requires an integer >= "+str(minimum))
    return int(value)


def _field(value, name):
    if np.ma.isMaskedArray(value): raise ValueError(name+" cannot be a masked array")
    array = np.asarray(value)
    if array.dtype.kind != "f" or not np.isfinite(array).all(): raise ValueError(name+" requires finite floating-point values")
    return array.astype(np.float64, copy=True)


def _gravity(parameters):
    pitch, roll = parameters[1:]
    return np.array([-np.sin(roll)*np.cos(pitch), -np.cos(roll)*np.cos(pitch), np.sin(pitch)])


def _project(parameters, pixels, sx, sy, center):
    focal = np.exp(parameters[0]); xy = (pixels-center)/[sx*focal, sy*focal]
    gravity = _gravity(parameters)
    projected = gravity[:2]-gravity[2]*xy
    length = np.linalg.norm(projected, axis=1)
    if np.any(length == 0): raise FloatingPointError("Up projection is undefined at a ray parallel to gravity")
    up = projected/length[:, None]
    sine = (xy@gravity[:2]+gravity[2])/np.sqrt(1+np.sum(xy*xy, axis=1))
    # Roundoff at a unit-vector dot product is bounded, not a geometry repair.
    if np.any(np.abs(sine) > 1+8*np.finfo(float).eps): raise FloatingPointError("Latitude projection overflow")
    return up, np.arcsin(np.clip(sine, -1., 1.))


def _support(weights, cells, selected):
    values = weights[selected]
    # ESS is scale-invariant; this avoids underflow before the independent mean
    # evidence floor rejects infinitesimal confidences (no evidence fabrication).
    scaled = values/values.max() if np.any(values) else values
    total = float(scaled.sum())
    ess = total*total/float(scaled@scaled) if total else 0.
    cell_weights = np.bincount(cells[selected], weights=scaled, minlength=64)
    cell_ess = total*total/float(cell_weights@cell_weights) if total else 0.
    mean = float(values.mean()) if len(values) else 0.
    return {"samples": len(values), "effective_samples": ess, "effective_spatial_cells": cell_ess,
            "mean_confidence": mean,
            "supported": bool(ess >= MIN_ESS and cell_ess >= MIN_CELL_ESS and mean >= MIN_MEAN_CONFIDENCE)}


def fit_perspective_calibration(up_field, latitude_field, up_confidence, latitude_confidence, *,
                                original_width, original_height, resized_width, resized_height,
                                crop_left=0, crop_top=0):
    """Fit a single original-image fx=fy focal and camera-frame gravity.

    Inputs are native CHW up [2,H,W], latitude [1,H,W] or [H,W], and two [H,W]
    confidences. Sampling is fixed stride8; checkerboard8px blocks reserve half
    the samples. Three diagonal-focal starts fit train only, selected by heldout
    angular residuals. Independent train half-grid fits check focal stability.
    Geometry and thresholds are frozen, not tuned against private evaluation.
    Returned K is for the supplied *processed integer-index field grid*.
    """
    ow = _integer(original_width, "original_width", 1); oh = _integer(original_height, "original_height", 1)
    rw = _integer(resized_width, "resized_width", 1); rh = _integer(resized_height, "resized_height", 1)
    left = _integer(crop_left, "crop_left", 0); top = _integer(crop_top, "crop_top", 0)
    up = _field(up_field, "up_field"); lat = _field(latitude_field, "latitude_field")
    uc = _field(up_confidence, "up_confidence"); lc = _field(latitude_confidence, "latitude_confidence")
    if lat.ndim == 3 and lat.shape[0] == 1: lat = lat[0]
    if up.ndim != 3 or up.shape[0] != 2 or min(up.shape[1:]) < 1:
        raise ValueError("up_field must have nonempty [2,H,W] shape")
    height, width = up.shape[1:]
    if lat.shape != (height,width) or uc.shape != lat.shape or lc.shape != lat.shape:
        raise ValueError("All fields/confidences must share the same pixel grid")
    if width+left > rw or height+top > rh: raise ValueError("Crop lies outside resized image")
    if not np.allclose(np.linalg.norm(up, axis=0), 1., rtol=0, atol=5e-5): raise ValueError("Up field must already be unit-normalized")
    if np.any(np.abs(lat) > np.pi/2) or np.any((uc<0)|(uc>1)|(lc<0)|(lc>1)):
        raise ValueError("Latitude/confidence values outside physical/native bounds")
    sx, sy = rw/ow, rh/oh; center = np.array([sx*ow/2-left-.5, sy*oh/2-top-.5])
    diagonal = float(np.hypot(ow,oh)); diagnostics = {"support": {}, "starts": [], "solver_max_nfev": 100,
        "sample_stride": STRIDE, "robust_scale_rad": float(ROBUST_SCALE_RAD),
        "min_mean_confidence": MIN_MEAN_CONFIDENCE, "fx_fy_original_shared": True,
        "residual_weighting": "equal total modality weight; up two chord components share its mass"}

    def outcome(accepted, reason, focal=diagonal, gravity=None):
        K = ((sx*focal,0.,float(center[0])), (0.,sy*focal,float(center[1])), (0.,0.,1.))
        return PerspectiveCalibration(accepted, reason, float(focal), gravity, K, diagonal, diagnostics)

    y,x = np.meshgrid(np.arange(0,height,STRIDE), np.arange(0,width,STRIDE), indexing="ij")
    pixels = np.c_[x.ravel(), y.ravel()].astype(float)
    observed_up = up[:,y,x].reshape(2,-1).T; observed_lat = lat[y,x].ravel()
    weights = (uc[y,x].ravel(),lc[y,x].ravel())
    bx, by = x.ravel()//8, y.ravel()//8
    train = (bx+by)%2 == 0; holdout = ~train
    halves = (train&(bx%2 == 0), train&(bx%2 == 1))
    cells = (y.ravel()*8//height)*8+x.ravel()*8//width
    for label, selection in (("train",train),("holdout",holdout),("half_0",halves[0]),("half_1",halves[1])):
        diagnostics["support"][label] = {name: _support(w,cells,selection) for name,w in zip(("up","latitude"),weights)}
    if any(not value["supported"] for row in diagnostics["support"].values() for value in row.values()):
        return outcome(False,"insufficient_confidence_effective_spatial_support")
    lower = np.array([np.log(.15*diagonal),-np.pi/2,-np.inf])
    upper = np.array([np.log(4*diagonal),np.pi/2,np.inf])

    def residual(parameters, selected):
        predicted_up, predicted_lat = _project(parameters,pixels[selected],sx,sy,center)
        wu,wl = (w[selected] for w in weights)
        ru = (predicted_up-observed_up[selected])*np.sqrt(wu/wu.mean()/2)[:,None]
        rl = (predicted_lat-observed_lat[selected])*np.sqrt(wl/wl.mean())
        return np.r_[ru.ravel(),rl]

    def fit(initial, selected):
        result = least_squares(residual,initial,args=(selected,),bounds=(lower,upper),loss="soft_l1",
                               f_scale=ROBUST_SCALE_RAD,max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        if not result.success or not all(np.isfinite(v).all() for v in (result.x,result.fun,result.jac)):
            raise FloatingPointError("Finite converged field fit required")
        return result

    def heldout_errors(parameters):
        pu,pl = _project(parameters,pixels[holdout],sx,sy,center)
        angle = np.arccos(np.clip(np.sum(pu*observed_up[holdout],axis=1),-1.,1.))
        return float(np.median(angle[weights[0][holdout]>0])), float(np.median(np.abs(pl-observed_lat[holdout])[weights[1][holdout]>0]))

    mean_up = np.sum(observed_up[train]*weights[0][train,None],axis=0)
    pitch = float(np.median(observed_lat[train])); roll = float(np.arctan2(-mean_up[0],-mean_up[1]))
    candidates = []
    for factor in (.5,1.,2.):
        entry = {"start_factor": factor}; diagnostics["starts"].append(entry)
        try:
            result = fit([np.log(factor*diagonal),pitch,roll],train); errors = heldout_errors(result.x)
            entry.update(converged=True,nfev=int(result.nfev),focal_px=float(np.exp(result.x[0])),heldout_up_deg=float(np.rad2deg(errors[0])),heldout_latitude_deg=float(np.rad2deg(errors[1])))
            candidates.append((sum(errors),factor,result,errors))
        except (ValueError,FloatingPointError,OverflowError): entry["converged"] = False
    if not candidates: return outcome(False,"no_finite_converged_fit")
    _,_,result,errors = min(candidates,key=lambda item: (item[0],item[1]))
    focal = float(np.exp(result.x[0])); singular = np.linalg.svd(result.jac,compute_uv=False)
    rank = int(np.linalg.matrix_rank(result.jac))
    with np.errstate(over="ignore",divide="ignore"):
        ratio = singular[0]/singular[-1] if singular[-1]>0 else np.inf
    condition = float(ratio) if np.isfinite(ratio) else None
    diagnostics.update(candidate_focal_px=focal,jacobian_rank=rank,jacobian_condition=condition,
                       heldout_up_median_deg=float(np.rad2deg(errors[0])),heldout_latitude_median_deg=float(np.rad2deg(errors[1])))
    if rank != 3 or condition is None or condition > 1e5: return outcome(False,"unidentifiable_or_ill_conditioned")
    if focal <= .15*diagonal*(1+1e-6) or focal >= 4*diagonal*(1-1e-6): return outcome(False,"focal_at_declared_bound")
    if errors[0] > np.deg2rad(10) or errors[1] > np.deg2rad(5): return outcome(False,"heldout_field_residual_gate")
    try:
        half_focals = [float(np.exp(fit(result.x,selection).x[0])) for selection in halves]
    except (ValueError,FloatingPointError,OverflowError): return outcome(False,"half_grid_fit_failed")
    disagreement = max(half_focals)/min(half_focals)-1
    diagnostics.update(half_grid_focals_px=half_focals,half_grid_relative_disagreement=disagreement)
    if disagreement > .1: return outcome(False,"half_grid_focal_instability")
    return outcome(True,"accepted_prediction_field_consistency_only",focal,tuple(float(v) for v in _gravity(result.x)))
