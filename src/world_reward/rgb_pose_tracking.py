"""Stateless RGB LK/PnP proposal for a fixed, already-scaled anchor mesh.

Integer OpenCV pixel centres and caller-supplied RGB-derived K are required;
X_current = R @ X_anchor + t. No shape, scale, camera or depth correction is fit.
Flow-to-PnP idea inspired by EgoInfinity; this is an own implementation, not its code.
Spatially reserved pixels are a consistency proxy, not independent temporal
validation or verified pose accuracy. Planar/symmetric pose ambiguity is unaudited.
OpenCV is lazy/optional; missing libraries and backend contract errors propagate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RGBPoseProposal:
    status: str
    reason: str
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    fit_indices: tuple[int, ...] = ()
    heldout_indices: tuple[int, ...] = ()
    inlier_indices: tuple[int, ...] = ()
    fit_rmse_px: float | None = None
    heldout_median_px: float | None = None
    heldout_max_px: float | None = None
    planar_support: bool = False


def _floats(value: object, shape: tuple, name: str, *, finite: bool = True) -> np.ndarray:
    a = np.asarray(value)
    if a.shape != shape or not np.issubdtype(a.dtype, np.floating) or np.ma.isMaskedArray(value):
        raise ValueError(f"{name}: floating array with shape {shape} required")
    a = a.astype(np.float64, copy=True)
    if finite and not np.isfinite(a).all():
        raise ValueError(f"{name}: finite values required")
    return a


def _bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    y, x = np.nonzero(mask)
    return (np.array([x.min(), y.min()]), np.array([np.ptp(x) + 1, np.ptp(y) + 1])) if len(x) else None


def _project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    homogeneous = points @ k.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def vetted_tracks(anchor: object, current: object, backward: object, forward_ok: object,
                  backward_ok: object, anchor_mask: np.ndarray, current_mask: np.ndarray) -> np.ndarray:
    """Boolean acceptance: both LK statuses, <=1px FB, 1px border, both RGB masks."""
    n = len(np.asarray(anchor))
    a = _floats(anchor, (n, 2), "anchor")
    b = _floats(current, (n, 2), "current", finite=False)
    c = _floats(backward, (n, 2), "backward", finite=False)
    statuses = [np.asarray(s) for s in (forward_ok, backward_ok)]
    if any(s.shape != (n,) or s.dtype != np.bool_ for s in statuses):
        raise ValueError("LK statuses must be bool [N]")
    if (anchor_mask.dtype != np.bool_ or current_mask.dtype != np.bool_
            or anchor_mask.ndim != 2 or current_mask.shape != anchor_mask.shape):
        raise ValueError("masks must be matching bool [H,W]")
    ok = statuses[0] & statuses[1] & np.isfinite(b).all(1) & np.isfinite(c).all(1)
    ok &= np.linalg.norm(c - a, axis=1) <= 1.0
    h, w = anchor_mask.shape
    for p, mask in ((a, anchor_mask), (b, current_mask)):
        inside = np.isfinite(p).all(1) & (p[:, 0] >= 1) & (p[:, 0] < w - 1)
        inside &= (p[:, 1] >= 1) & (p[:, 1] < h - 1)
        ids = np.flatnonzero(inside)
        xy = np.floor(p[ids] + .5).astype(np.int64)
        accepted = np.zeros(n, dtype=bool)
        accepted[ids] = mask[xy[:, 1], xy[:, 0]]
        ok &= accepted
    return ok


def track_rgb_pose(anchor_rgb: object, current_rgb: object, anchor_mask: object, current_mask: object,
                   anchor_pixels: object, anchor_points_camera: object, k: object) -> RGBPoseProposal:
    """One EPNP(200/.99/3px) -> inlier-only LM, never fit spatial holdout pixels.

    Fixed 8x8 anchor-bbox grid parity reserves >=8 tracks; fit needs >=16 tracks
    and inliers spanning >=20% of each mask bbox in both axes. Heldout median
    <=3px is required. Seed0 sets OpenCV's process-global RANSAC RNG (not thread
    isolated). An abstention has no transform; callers must not fabricate stasis.
    """
    images = [np.asarray(v) for v in (anchor_rgb, current_rgb)]
    masks = [np.asarray(v) for v in (anchor_mask, current_mask)]
    if (any(np.ma.isMaskedArray(v) for v in (anchor_rgb, current_rgb, anchor_mask, current_mask))
            or images[0].ndim != 3 or images[0].shape[2] != 3 or min(images[0].shape[:2]) < 3
            or any(v.dtype != np.uint8 or v.shape != images[0].shape for v in images)
            or any(v.dtype != np.bool_ or v.shape != images[0].shape[:2] for v in masks)):
        raise ValueError("matching uint8 RGB [H,W,3] and bool masks [H,W] required")
    images, masks = [v.copy() for v in images], [v.copy() for v in masks]
    n = len(np.asarray(anchor_pixels))
    pixels = _floats(anchor_pixels, (n, 2), "anchor_pixels")
    points = _floats(anchor_points_camera, (n, 3), "anchor_points_camera")
    k = _floats(k, (3, 3), "k")
    if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0, 0, 1])
            or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError("zero-skew OpenCV pinhole K with positive focal lengths required")
    if (np.any(points[:, 2] <= 0) or not np.array_equal(pixels, np.floor(pixels))
            or len(np.unique(pixels, axis=0)) != n):
        raise ValueError("positive-depth 3D points and unique integer-centre anchor pixels required")
    projected = _project(points, k)
    if not np.isfinite(projected).all() or (n and np.max(np.linalg.norm(projected - pixels, axis=1)) > 3.0):
        raise ValueError("anchor mesh projection is nonfinite or disagrees with supplied pixels by >3px")
    boxes = [_bbox(m) for m in masks]
    if n < 24 or any(b is None for b in boxes):
        return RGBPoseProposal("abstain", "insufficient_anchor_support")
    import cv2
    try:
        gray = [cv2.cvtColor(v, cv2.COLOR_RGB2GRAY) for v in images]
        options = dict(winSize=(21, 21), maxLevel=3, criteria=(3, 30, .01))
        p = pixels.astype(np.float32).reshape(-1, 1, 2)
        q, sf, _ = cv2.calcOpticalFlowPyrLK(gray[0], gray[1], p.copy(), None, **options)
        if q is None or sf is None:
            return RGBPoseProposal("abstain", "forward_lk_failed")
        if (np.asarray(q).shape != (n, 1, 2) or np.asarray(sf).shape != (n, 1)
                or np.asarray(sf).dtype != np.uint8 or not np.isin(sf, [0, 1]).all()):
            raise ValueError("invalid OpenCV forward LK result contract")
        q = _floats(q, (n, 1, 2), "LK current", finite=False).reshape(n, 2)
        candidates = np.flatnonzero(sf.reshape(n).astype(bool) & np.isfinite(q).all(1))
        if not len(candidates):
            return RGBPoseProposal("abstain", "no_forward_tracks")
        back_subset, sb_subset, _ = cv2.calcOpticalFlowPyrLK(gray[1], gray[0],
            q[candidates].astype(np.float32).reshape(-1, 1, 2), None, **options)
        back, sb = np.full((n, 2), np.nan), np.zeros(n, dtype=bool)
        if back_subset is None or sb_subset is None:
            return RGBPoseProposal("abstain", "backward_lk_failed")
        if (np.asarray(sb_subset).shape != (len(candidates), 1) or np.asarray(sb_subset).dtype != np.uint8
                or not np.isin(sb_subset, [0, 1]).all()):
            raise ValueError("invalid OpenCV backward LK status contract")
        back[candidates] = _floats(back_subset, (len(candidates), 1, 2), "LK backward", finite=False).reshape(-1, 2)
        sb[candidates] = np.asarray(sb_subset).reshape(-1).astype(bool)
        good = vetted_tracks(pixels, q, back, np.asarray(sf).reshape(n).astype(bool),
                             sb, *masks)
        ids = np.flatnonzero(good)
        if len(np.unique(np.floor(q[ids] + .5), axis=0)) != len(ids):
            return RGBPoseProposal("abstain", "colliding_current_pixels")
        cells = np.floor((pixels[ids] - boxes[0][0]) / boxes[0][1] * 8).astype(int)
        parity = cells.sum(axis=1) % 2
        fit, held = ids[parity == 0], ids[parity == 1]
        evidence = dict(fit_indices=tuple(map(int, fit)), heldout_indices=tuple(map(int, held)))
        def abstain(reason: str) -> RGBPoseProposal:
            return RGBPoseProposal("abstain", reason, **evidence)
        def span(indices: np.ndarray) -> bool:
            return all(np.all(np.ptp(p[indices], axis=0) >= .2 * b[1])
                       for p, b in zip((pixels, q), boxes))
        if len(fit) < 16 or len(held) < 8 or not span(fit):
            return abstain("weak_spatial_support")
        singular = np.linalg.svd(points[fit] - points[fit].mean(0), compute_uv=False)
        if singular[0] <= 0 or singular[1] <= 1e-8 * singular[0]:
            return abstain("underconstrained_3d_support")
        evidence["planar_support"] = bool(singular[2] <= 1e-8 * singular[0])
        cv2.setRNGSeed(0)
        solved, rv, tv, inliers = cv2.solvePnPRansac(points[fit], q[fit].astype(np.float64), k,
            np.zeros(4), iterationsCount=200, reprojectionError=3., confidence=.99, flags=cv2.SOLVEPNP_EPNP)
        if not solved or inliers is None:
            return abstain("pnp_failed")
        inliers = np.asarray(inliers)
        if (not np.issubdtype(inliers.dtype, np.integer) or inliers.size == 0
                or np.any(inliers < 0) or np.any(inliers >= len(fit))
                or len(np.unique(inliers)) != inliers.size):
            raise ValueError("OpenCV returned invalid PnP inlier indices")
        selected = fit[inliers.reshape(-1)]
        evidence["inlier_indices"] = tuple(map(int, selected))
        if len(selected) < 16 or not span(selected):
            return abstain("weak_inlier_support")
        rv, tv = cv2.solvePnPRefineLM(points[selected], q[selected].astype(np.float64), k,
                                    np.zeros(4), np.asarray(rv).copy(), np.asarray(tv).copy())
        r = _floats(cv2.Rodrigues(rv)[0], (3, 3), "rotation")
        t = _floats(np.asarray(tv).reshape(3), (3,), "translation")
        if (not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(r), 1., atol=1e-6, rtol=0)):
            raise ValueError("PnP rotation is not proper SO(3)")
        moved = points[ids] @ r.T + t
        if not np.isfinite(moved).all() or np.any(moved[:, 2] <= 0):
            return abstain("nonpositive_current_depth")
        errors = np.full(n, np.inf)
        errors[ids] = np.linalg.norm(_project(moved, k) - q[ids], axis=1)
        if not np.isfinite(errors[ids]).all():
            return abstain("nonfinite_reprojection")
        evidence.update(fit_rmse_px=float(np.sqrt(np.mean(errors[selected] ** 2))),
                        heldout_median_px=float(np.median(errors[held])), heldout_max_px=float(errors[held].max()))
        if evidence["fit_rmse_px"] > 3. or evidence["heldout_median_px"] > 3.:
            return abstain("reprojection_disagreement")
        r.setflags(write=False)
        t.setflags(write=False)
        return RGBPoseProposal("proposal", "spatial_holdout_consistent_not_accuracy", r, t, **evidence)
    except cv2.error as exc:
        if getattr(exc, "code", None) != -7:  # cv::Error::StsNoConv only, never an ABI/missing-library fallback.
            raise
        return RGBPoseProposal("abstain", "opencv_no_convergence")
