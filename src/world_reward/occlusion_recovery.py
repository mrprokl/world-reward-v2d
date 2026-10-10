"""Conservative fixed-identity, two-direction visual-observation recovery.

This is a controller, not a second segmentation model or a pose interpolator.
One automatic final-frame box can seed one fresh native reverse pass. A lone
nonempty reverse mask is NOT recovery: seed-linked RGB correspondences and an
independent geometric prediction must support it. Two absent SAM predictions
do not prove occlusion. Uncertain/occluded frames keep their original indices;
the downstream full-T 3D estimator must infer latent poses, not delete objects.

Native callbacks and automatic RGB evidence are configured/authenticated on
Azure by the caller. New raw presence logits must be retained; old boolean-only
tracking receipts can keep the original sign with raw_logit=None, never a
fabricated numerical confidence or calibrated model evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable

import numpy as np


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Nonnegative original integer {name} required")


def _sha(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Exact lowercase original RGB/request SHA256 required")


def _owned(value, dtype, name, shape=None):
    if type(value) is not np.ndarray or value.dtype != np.dtype(dtype):
        raise ValueError(f"Explicit {name} dtype required")
    if shape is not None and value.shape != shape:
        raise ValueError(f"Original-grid {name} shape required")
    # Immutable owned bytes, not merely a caller-reversible ndarray write flag.
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class RecoveryPolicy:
    """Global gates frozen on external data, never selected per challenge clip.

    No operating thresholds are supplied by default. Unit manufactured cases
    qualify mathematical contracts only, not tracking/generalization quality.
    Native SAM presence/mask decisions remain the unmodified zero-logit sign.
    """
    agreement_iou: float
    geometry_iou: float
    identity_confidence: float
    identity_support: float
    calibration_source: str

    def __post_init__(self):
        for name in ("agreement_iou", "geometry_iou", "identity_support"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value < 1:
                raise ValueError("Finite nontrivial global recovery gates required")
        if (type(self.identity_confidence) not in (int, float)
                or not np.isfinite(self.identity_confidence)
                or not .5 < self.identity_confidence < 1):
            raise ValueError("Declared statistical identity confidence required")
        if (type(self.calibration_source) is not str
                or not self.calibration_source.startswith("external:")
                or len(self.calibration_source) <= len("external:")):
            raise ValueError("External calibration provenance required")


@dataclass(frozen=True, eq=False)
class NativeCandidate:
    frame_index: int
    object_id: int
    rgb_sha256: str
    branch: str
    mask: np.ndarray
    presence_logit: float | None
    native_presence: bool | None = None
    presence_source: str = "native_raw_logit"

    def __post_init__(self):
        _integer(self.frame_index, "frame index"); _integer(self.object_id, "fixed ID")
        _sha(self.rgb_sha256)
        if self.branch not in ("forward", "reverse"):
            raise ValueError("Literal native tracking branch required")
        if self.presence_logit is None:
            if (type(self.native_presence) is not bool
                    or not self.presence_source.startswith("saved_native_presence_sign:")):
                raise ValueError("Raw-unavailable saved native sign needs explicit provenance")
            _sha(self.presence_source.split(":", 1)[1])
        else:
            if (type(self.presence_logit) not in (float, int)
                    or not np.isfinite(self.presence_logit)
                    or self.presence_source != "native_raw_logit"):
                raise ValueError("Finite raw native presence logit required")
            sign = bool(self.presence_logit > 0)
            if self.native_presence is not None and (type(self.native_presence) is not bool or self.native_presence != sign):
                raise ValueError("Native presence must agree with unmodified raw logit sign")
            object.__setattr__(self, "native_presence", sign)
        mask = _owned(self.mask, np.bool_, "native mask")
        if mask.ndim != 2 or not all(mask.shape):
            raise ValueError("Nonempty original image grid required")
        if not self.native_presence and mask.any():
            raise ValueError("Native absence cannot carry an invented visible mask")
        object.__setattr__(self, "mask", mask)


def native_candidate(*, frame_index, object_id, rgb_sha256, branch,
                     mask_logits, presence_logit):
    """Exact original HW float logits -> native sign mask and raw confidence."""
    if (type(mask_logits) is not np.ndarray or mask_logits.dtype.kind != "f"
            or mask_logits.ndim != 2 or not np.isfinite(mask_logits).all()
            or type(presence_logit) not in (float, int)
            or not np.isfinite(presence_logit)):
        raise ValueError("Original HW finite float native logits required")
    mask = (mask_logits > 0) & (presence_logit > 0)
    return NativeCandidate(frame_index, object_id, rgb_sha256, branch, mask, presence_logit)


def saved_native_candidate(*, frame_index, object_id, rgb_sha256, branch,
                           mask, native_presence, tracking_report_sha256):
    """Reuse authenticated old native sign, explicitly WITHOUT a raw logit.

    Caller verifies mask inventory/PNG/report before invoking this pure codec.
    The tracking report hash identifies the literal saved presence-sign source;
    no ±1 pseudo-logit/confidence is synthesized and no model pass is required.
    """
    _sha(tracking_report_sha256)
    return NativeCandidate(frame_index, object_id, rgb_sha256, branch, mask, None,
                           native_presence, "saved_native_presence_sign:" + tracking_report_sha256)


@dataclass(frozen=True, eq=False)
class RgbIdentityEvidence:
    """Automatic seed-linked RGB tracks, independent of candidate selection.

    xy are ORIGINAL pixel-centre coordinates of persistent seed-object features.
    matched is independently appearance-qualified and geometrically round-trip
    consistent; visible is the point tracker's visibility result. Invalid/hidden
    points never become in-mask votes. geometry_mask is a separately predicted
    original-grid silhouette/warp, not a copy/union of candidate SAM masks.

    Sources/checkpoints, seed feature ownership and geometric predictor closure
    must be pinned by the caller. The operator checks the supplied observation
    binding, but cannot certify checkpoint training overlap or RGB quality.
    """
    frame_index: int
    object_id: int
    rgb_sha256: str
    xy: np.ndarray
    matched: np.ndarray
    visible: np.ndarray
    geometry_mask: np.ndarray | None
    source: str
    occlusion_confirmed: bool = False
    occlusion_source: str | None = None

    def __post_init__(self):
        _integer(self.frame_index, "frame index"); _integer(self.object_id, "fixed ID")
        _sha(self.rgb_sha256)
        xy = _owned(self.xy, np.float64, "seed RGB track coordinates")
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("Finite original XY seed-linked point coordinates required")
        matched = _owned(self.matched, np.bool_, "automatic match support", (len(xy),))
        visible = _owned(self.visible, np.bool_, "automatic point visibility", (len(xy),))
        if type(self.source) is not str or not self.source.startswith("automatic_rgb:"):
            raise ValueError("Explicit automatic seed-linked RGB evidence required")
        if type(self.occlusion_confirmed) is not bool:
            raise ValueError("Literal automatic occlusion evidence required")
        if self.occlusion_confirmed:
            if (type(self.occlusion_source) is not str
                    or not self.occlusion_source.startswith("automatic_occlusion:")
                    or np.any(matched & visible)):
                raise ValueError("Occlusion needs independent evidence and no visible matched seed points")
        elif self.occlusion_source is not None:
            raise ValueError("No unsupported occlusion-source assertion")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "matched", matched)
        object.__setattr__(self, "visible", visible)
        if self.geometry_mask is not None:
            mask = _owned(self.geometry_mask, np.bool_, "independent geometry mask")
            if mask.ndim != 2 or not all(mask.shape):
                raise ValueError("Original-grid geometry prediction required")
            object.__setattr__(self, "geometry_mask", mask)


def _iou(a, b):
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else None


def _lower_bound(successes, total, confidence):
    """Wilson-form support score, NOT calibrated confidence (RGB points correlate)."""
    if not total:
        return 0.0
    z = NormalDist().inv_cdf(confidence)
    p = successes / total; zz = z * z
    return (p + zz / (2 * total) - z * np.sqrt(p * (1-p) / total + zz / (4*total*total))) / (1 + zz / total)


def _support(candidate, evidence, policy):
    if evidence is None:
        return dict(qualified=False, identity_points=0, in_mask_points=0,
                    identity_support_lower_bound=0.0, geometry_iou=None)
    h, w = candidate.mask.shape
    # Same floor(XY) pixel-membership convention as point_mask_association;
    # off-image coordinates never clamp to an edge vote.
    # Repeated seed-point pixels must not artificially multiply support. RGB
    # points can remain correlated, so Wilson-form score is only a global gate,
    # not a statistical probability or model confidence certificate.
    active = evidence.matched & evidence.visible
    points = np.unique(evidence.xy[active], axis=0)
    # Bound before int conversion to avoid overflowing a finite off-image float.
    convertible = ((points[:, 0] >= 0) & (points[:, 0] < w)
                   & (points[:, 1] >= 0) & (points[:, 1] < h))
    on_image = points[convertible]
    ix = np.floor(on_image[:, 0]).astype(np.int64)
    iy = np.floor(on_image[:, 1]).astype(np.int64)
    pixels = np.unique(np.stack((ix, iy), axis=1), axis=0)
    off_image_count = int(np.count_nonzero(~convertible))
    success = candidate.mask[pixels[:, 1], pixels[:, 0]]
    support_count = len(success) + off_image_count
    lower = _lower_bound(int(success.sum()), support_count, policy.identity_confidence)
    geometry = None if evidence.geometry_mask is None else _iou(candidate.mask, evidence.geometry_mask)
    return dict(qualified=bool(candidate.native_presence and candidate.mask.any()
            and lower > policy.identity_support and geometry is not None
            and geometry >= policy.geometry_iou and not evidence.occlusion_confirmed),
        identity_points=support_count, in_mask_points=int(success.sum()),
        identity_support_lower_bound=float(lower), geometry_iou=geometry)


@dataclass(frozen=True, eq=False)
class RecoveryFrame:
    frame_index: int
    object_id: int
    rgb_sha256: str
    observed_mask: np.ndarray
    state: str  # observed | occluded | uncertain; never a trajectory deletion.
    source: str
    diagnostics: dict


def fuse_frame(forward, reverse, evidence, policy):
    """Retain literal qualified native mask; never union/fill/shrink predictions."""
    if type(forward) is not NativeCandidate or forward.branch != "forward":
        raise ValueError("Exact forward native fixed-ID observation required")
    if type(policy) is not RecoveryPolicy:
        raise ValueError("Explicit globally frozen recovery policy required")
    binding = (forward.frame_index, forward.object_id, forward.rgb_sha256)
    if reverse is not None:
        if (type(reverse) is not NativeCandidate or reverse.branch != "reverse"
                or (reverse.frame_index, reverse.object_id, reverse.rgb_sha256) != binding
                or reverse.mask.shape != forward.mask.shape):
            raise ValueError("Reverse must preserve exact original frame/RGB/grid/physical ID")
    if evidence is not None:
        if (type(evidence) is not RgbIdentityEvidence
                or (evidence.frame_index, evidence.object_id, evidence.rgb_sha256) != binding
                or (evidence.geometry_mask is not None
                    and evidence.geometry_mask.shape != forward.mask.shape)):
            raise ValueError("Independent evidence must bind exact original frame/RGB/grid/ID")
    f = _support(forward, evidence, policy)
    r = None if reverse is None else _support(reverse, evidence, policy)
    overlap = None if reverse is None else _iou(forward.mask, reverse.mask)
    selected = None; state = "uncertain"; source = "unqualified_observations"
    if f["qualified"] and r is not None and r["qualified"]:
        if overlap is not None and overlap >= policy.agreement_iou:
            selected = forward; state = "observed"; source = "bidirectional_rgb_verified"
        else:
            source = "qualified_branches_disagree"
    elif f["qualified"]:
        selected = forward; state = "observed"; source = "forward_rgb_verified"
    elif r is not None and r["qualified"]:
        selected = reverse; state = "observed"; source = "reverse_rgb_recovery"
    elif evidence is not None and evidence.occlusion_confirmed:
        state = "occluded"; source = "independent_automatic_occlusion"
    # Empty observation means "do not treat a hallucination as visible data";
    # downstream must retain and infer this object's full original 3D timeline.
    mask = np.zeros(forward.mask.shape, np.bool_) if selected is None else selected.mask.copy()
    mask = _owned(mask, np.bool_, "selected original-grid observation")
    diagnostics = dict(forward=f, reverse=r, bidirectional_iou=overlap,
        native_forward_presence_logit=None if forward.presence_logit is None else float(forward.presence_logit),
        native_reverse_presence_logit=None if reverse is None or reverse.presence_logit is None else float(reverse.presence_logit),
        native_forward_presence=forward.native_presence,
        native_reverse_presence=None if reverse is None else reverse.native_presence,
        forward_presence_source=forward.presence_source,
        reverse_presence_source=None if reverse is None else reverse.presence_source,
        native_forward_visible_pixels=int(forward.mask.sum()),
        native_reverse_visible_pixels=None if reverse is None else int(reverse.mask.sum()),
        evidence_source=None if evidence is None else evidence.source,
        occlusion_source=None if evidence is None else evidence.occlusion_source,
        observation_interpolated=False, quality_verified=False,
        full_T_latent_pose_required=state != "observed",
        calibration_source=policy.calibration_source)
    return RecoveryFrame(*binding, mask, state, source, diagnostics)


@dataclass(frozen=True)
class AutomaticBoxAnchor:
    frame_index: int
    object_id: int
    rgb_sha256: str
    box: tuple[float, float, float, float]
    request_sha256: str

    def __post_init__(self):
        _integer(self.frame_index, "anchor frame"); _integer(self.object_id, "fixed ID")
        _sha(self.rgb_sha256); _sha(self.request_sha256)
        if (type(self.box) is not tuple or len(self.box) != 4
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in self.box)):
            raise ValueError("Finite original XYXY automatic box required")


def recover_sequence(*, frame_indices, rgb_sha256, image_size, object_id,
                     needs_recovery, forward_at: Callable, evidence_at: Callable,
                     resolve_final_anchor: Callable, prefix_anchors,
                     init_state: Callable, seed_box: Callable, propagate: Callable,
                     emit: Callable, policy: RecoveryPolicy,
                     forward_state=None, release_state=None):
    """At most one final-anchor resolution and one fresh full reverse pass.

    forward_at(original_index), evidence_at(original_index) return typed rows
    bound to this exact video. The caller can load one frozen PNG/logit record
    at a time; the controller never stacks RGB/video/masks. Prefix anchors are
    existing automatic boxes and are ALL retained in the fresh singleton state.

    resolve_final_anchor(frame_index=last, object_id=fixed, rgb_sha256=hash)
    returns one automatic box or None. No search/retry/new semantic identity is
    performed here. Infrastructure exceptions propagate; they are not converted
    into masks or hidden as object absence. The callback must qualify anchor
    identity using saved description/action and independent RGB evidence.

    propagate(state, start_frame_idx=last_position, reverse=True) yields typed
    NativeCandidate rows in exact reverse order over the full original timeline.
    The caller prepares native backbone features in that same reverse direction.
    Fresh state is an experiment/control boundary, not a claimed SAM API fix.

    emit receives every original frame exactly once (reverse order when a
    reverse pass is used). Unavailable anchor still emits full-T forward-only
    observation states, allowing downstream latent 3D fallback rather than a
    failed/dropped clip. Success means controller completion, not model accuracy.
    """
    if (type(frame_indices) is not tuple or not frame_indices
            or any(type(i) is not int or i < 0 for i in frame_indices)
            or any(a >= b for a, b in zip(frame_indices, frame_indices[1:]))):
        raise ValueError("Complete strictly increasing original frame-index tuple required")
    if type(rgb_sha256) is not tuple or len(rgb_sha256) != len(frame_indices):
        raise ValueError("Complete original RGB hash timeline required")
    for value in rgb_sha256: _sha(value)
    if (type(image_size) is not tuple or len(image_size) != 2
            or any(type(v) is not int or v <= 0 for v in image_size)):
        raise ValueError("Exact original (height,width) integer image grid required")
    _integer(object_id, "fixed ID")
    if type(needs_recovery) is not bool or type(prefix_anchors) is not tuple:
        raise ValueError("Literal recovery decision and all saved prefix anchors required")
    if type(policy) is not RecoveryPolicy:
        raise ValueError("Declared externally frozen policy required")
    callbacks = (forward_at, evidence_at, resolve_final_anchor, init_state, seed_box, propagate, emit)
    if not all(callable(c) for c in callbacks) or (release_state is not None and not callable(release_state)):
        raise ValueError("Explicit automatic/native callbacks and sink required")
    by_frame = dict(zip(frame_indices, rgb_sha256)); h, w = image_size

    def valid_anchor(anchor):
        if (type(anchor) is not AutomaticBoxAnchor or anchor.object_id != object_id
                or anchor.frame_index not in by_frame
                or anchor.rgb_sha256 != by_frame[anchor.frame_index]):
            raise ValueError("Automatic anchors must retain exact saved original frame/RGB/ID")
        x0, y0, x1, y1 = anchor.box
        if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
            raise ValueError("Original exclusive XYXY bounds required; no box clamping")

    if len({a.frame_index for a in prefix_anchors if type(a) is AutomaticBoxAnchor}) != len(prefix_anchors):
        raise ValueError("All unique saved prefix anchors required")
    for anchor in prefix_anchors: valid_anchor(anchor)
    final = None
    if needs_recovery:
        final = resolve_final_anchor(frame_index=frame_indices[-1], object_id=object_id,
                                     rgb_sha256=rgb_sha256[-1])
        if final is not None:
            valid_anchor(final)
            if final.frame_index != frame_indices[-1]:
                raise ValueError("One fixed final original-frame anchor required")
            if any(a.frame_index == final.frame_index for a in prefix_anchors):
                raise ValueError("Final anchor must not overwrite a saved prefix prompt")
    counts = dict(observed=0, occluded=0, uncertain=0); recovered = 0; native_frames = 0

    def output(index, reverse):
        nonlocal recovered
        forward = forward_at(index)
        if (type(forward) is not NativeCandidate or forward.frame_index != index
                or forward.object_id != object_id or forward.rgb_sha256 != by_frame[index]
                or forward.mask.shape != image_size):
            raise ValueError("Forward stream must preserve full original frame/RGB/grid/ID")
        row = fuse_frame(forward, reverse, evidence_at(index), policy)
        counts[row.state] += 1; recovered += row.source == "reverse_rgb_recovery"
        emit(row)

    state = None
    if final is None:
        for index in frame_indices: output(index, None)
    else:
        state = init_state()
        if state is None or state is forward_state:
            raise ValueError("One independent fresh native singleton state required")
        try:
            positions = {i: p for p, i in enumerate(frame_indices)}
            for anchor in prefix_anchors + (final,):
                seed_box(state, frame_idx=positions[anchor.frame_index], obj_id=object_id,
                         box=anchor.box, normalize_coords=True)
            stream = iter(propagate(state, start_frame_idx=len(frame_indices)-1, reverse=True))
            for index in reversed(frame_indices):
                try:
                    reverse = next(stream)
                except StopIteration as error:
                    raise ValueError("Native reverse pass omitted an original frame") from error
                if (type(reverse) is not NativeCandidate or reverse.branch != "reverse"
                        or reverse.frame_index != index):
                    raise ValueError("Exact full-T native reverse frame order required")
                output(index, reverse); native_frames += 1
            try:
                next(stream)
            except StopIteration:
                pass
            else:
                raise ValueError("Native reverse pass duplicated/added an original frame")
        finally:
            if release_state is not None: release_state(state)
    return dict(schema="world_reward.occlusion_recovery.v1", status="complete_controller_not_quality_pass",
        object_id=object_id, original_frame_indices=list(frame_indices), states=counts,
        recovered_observation_frames=int(recovered), native_reverse_frames=native_frames,
        final_anchor_resolutions=int(needs_recovery), additional_reverse_passes=int(final is not None),
        final_anchor_status="not_needed" if not needs_recovery else "unavailable" if final is None else "available",
        mask_interpolation=False, semantic_identity_changed=False, quality_verified=False,
        ground_truth_used=False, manual_labels=False, full_T_latent_pose_required=counts["occluded"]+counts["uncertain"] > 0,
        calibration_source=policy.calibration_source)
