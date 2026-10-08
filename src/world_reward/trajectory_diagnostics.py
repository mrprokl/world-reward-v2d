"""Full-timeline diagnostics only: no fitting, filtering, labels or thresholds.

Lengths are supplied metres, times seconds. Surface RMS uses EVERY supplied
vertex with equal weight, not surface area; canonical official padding should
not be mistaken for an area-weighted metric. A supplied plane is a visualization
assumption, not calibrated ground truth. Reports retain original frame indices.
"""
from __future__ import annotations

import numpy as np


def _array(value, label):
    if np.ma.isMaskedArray(value):
        raise ValueError(label+" must not be masked")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(label+" must be finite real numeric data")
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(label+" exceeds finite float64 arithmetic")
    return result


def _timeline(frame_index, total, fps):
    if np.ma.isMaskedArray(frame_index):
        raise ValueError("Explicit unmasked full original indices required")
    ids = np.asarray(frame_index)
    if (total < 3 or ids.dtype.kind not in "iu" or ids.shape != (total,)
            or not np.array_equal(ids, np.arange(total)) or type(fps) not in (int, float)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("All original indices 0..T-1, T>=3 and positive finite fps required")
    return ids.copy(), float(fps)


def summarize(values):
    """Finite nonempty scalar series; linear quantiles, median/MAD and RMS."""
    x = _array(values, "series")
    if x.ndim != 1 or not len(x):
        raise ValueError("Nonempty one-dimensional series required")
    peak = float(np.max(np.abs(x)))
    rms = peak*float(np.sqrt(np.mean((x/peak)**2))) if peak else 0.
    median = float(np.median(x))
    report = dict(count=len(x), mean=float(np.mean(x)), rms=rms, median=median,
                mad=float(np.median(np.abs(x-median))),
                quantiles=dict(zip(("p0", "p50", "p90", "p95", "p99", "p100"),
                    np.quantile(x, [0, .5, .9, .95, .99, 1], method="linear").tolist())))
    if (not np.isfinite([report[k] for k in ("mean","rms","median","mad")]).all()
            or not np.isfinite(list(report['quantiles'].values())).all()):
        raise ValueError("Summary exceeds finite float64 arithmetic")
    return report


def _plane(plane):
    p = _array(plane, "plane")
    norm = np.hypot.reduce(p[:3]) if p.shape == (4,) else 0.
    if not np.isfinite(norm) or not norm > 0:
        raise ValueError("Plane [nx,ny,nz,d], n·x+d=0 with nonzero normal required")
    return _array(p/norm,"normalized plane")


def _report(ids, fps, centroid, series):
    return dict(frame_index=ids, fps=fps, centroid_xyz_m=_array(centroid,"world centroid"), series=series,
        series_frame_index={name:ids[1:-1] if "acceleration" in name else
                            ids if name.startswith("plane_") else ids[1:] for name in series},
        statistics={name:summarize(x) for name,x in series.items()},
        vertex_weighting="all_supplied_vertices_equal_not_surface_area",
        predictions_modified=False, ground_truth_used=False, quality_verified=False)


def object_motion(vertices, rotations, translations, frame_index, fps, *, plane=None):
    """Exact all-vertex rigid RMS via centred second moment; no origin bias.

    Step series corresponds to frame_index[1:], acceleration to [1:-1]. A plane
    gap is the minimum signed distance of ALL vertices for each original frame.
    """
    v, r, t = (_array(x, n) for x,n in ((vertices,"vertices"),
        (rotations,"rotations"),(translations,"translations")))
    if (v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or r.ndim != 3
            or r.shape[1:] != (3,3) or t.shape != (len(r),3)
            or not np.allclose(r@r.swapaxes(-1,-2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-5, rtol=0)):
        raise ValueError("Finite (V,3), proper (T,3,3) rotations and (T,3) translations required")
    ids, fps = _timeline(frame_index, len(r), fps)
    centre = v.mean(axis=0); u = v-centre; covariance = u.T@u/len(v)
    centroid = np.einsum("tij,j->ti", r, centre)+t
    def rms(order):
        a = np.diff(r,n=order,axis=0); d = np.diff(centroid,n=order,axis=0)
        square = np.einsum("tij,jk,tik->t",a,covariance,a)+np.sum(d*d,axis=1)
        return np.sqrt(np.maximum(square,0)) * fps**(2 if order == 2 else 0)
    relative = r[1:]@r[:-1].swapaxes(-1,-2)
    cosine = (np.trace(relative,axis1=1,axis2=2)-1)/2
    sine = np.linalg.norm(np.stack((relative[:,2,1]-relative[:,1,2],
        relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]),axis=1),axis=1)/2
    series = dict(translation_step_m=np.linalg.norm(np.diff(t,axis=0),axis=1),
        centroid_step_m=np.linalg.norm(np.diff(centroid,axis=0),axis=1),
        rotation_step_deg=np.degrees(np.arctan2(sine,np.clip(cosine,-1,1))),
        surface_rms_step_m=rms(1), surface_rms_acceleration_m_s2=rms(2),
        centroid_acceleration_m_s2=np.linalg.norm(np.diff(centroid,n=2,axis=0),axis=1)*fps**2)
    if plane is not None:
        p = _plane(plane)
        series["plane_min_signed_gap_m"] = np.min(v@np.einsum("tij,i->tj",r,p[:3]).T,axis=0)+t@p[:3]+p[3]
    return _report(ids,fps,centroid,series)


def body_motion(vertices, frame_index, fps, *, plane=None):
    """All (T,V,3) body vertices, including deformation; never subsample frames."""
    v = _array(vertices,"body vertices")
    if v.ndim != 3 or v.shape[2] != 3 or not v.shape[1]:
        raise ValueError("Full nonempty (T,V,3) body geometry required")
    ids, fps = _timeline(frame_index,len(v),fps); centroid = v.mean(axis=1)
    series = dict(centroid_step_m=np.linalg.norm(np.diff(centroid,axis=0),axis=1),
        surface_rms_step_m=np.sqrt(np.mean(np.sum(np.diff(v,axis=0)**2,axis=2),axis=1)),
        surface_rms_acceleration_m_s2=np.sqrt(np.mean(np.sum(np.diff(v,n=2,axis=0)**2,axis=2),axis=1))*fps**2,
        centroid_acceleration_m_s2=np.linalg.norm(np.diff(centroid,n=2,axis=0),axis=1)*fps**2)
    if plane is not None:
        p = _plane(plane); series["plane_min_signed_gap_m"] = np.min(v@p[:3]+p[3],axis=1)
    return _report(ids,fps,centroid,series)


def project_centroids(centroids, K):
    """Project supplied (T,3) 3D centroids, not the mean of projected vertices."""
    c, k = _array(centroids,"centroids"), _array(K,"K")
    if (c.ndim != 2 or c.shape[1] != 3 or not len(c) or (c[:,2] <= 0).any()
            or k.shape != (3,3) or not np.array_equal(k[2],[0,0,1])
            or k[0,0] <= 0 or k[1,1] <= 0 or k[1,0] != 0):
        raise ValueError("Positive-z nonempty centroids and valid fixed pinhole K required")
    projected = c@k.T
    return _array(projected[:,:2]/projected[:,2:],"projected centroids")


def mask_geometry(mask):
    """Binary arbitrary-size mask: area pixels, (x,y) centroid, half-open bbox."""
    if np.ma.isMaskedArray(mask): raise ValueError("Unmasked binary mask required")
    x = np.asarray(mask)
    if (x.ndim != 2 or not all(x.shape) or x.dtype.kind not in "biu"
            or not (np.isin(x,[0,1]).all() or np.isin(x,[0,255]).all())):
        raise ValueError("Nonempty 2D bool, 0/1 or 0/255 integer mask required")
    yy, xx = np.nonzero(x); area = len(xx)
    return dict(area_pixels=area, area_fraction=area/x.size,
        centroid_xy=None if not area else [float(xx.mean()),float(yy.mean())],
        bbox_xyxy=None if not area else [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)])
