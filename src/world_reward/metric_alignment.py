"""One robust positive depth scale, anchored to an *estimated* rendered human.

Neither source is ground truth: this makes their camera-Z estimates consistent,
not metrically correct. There is no affine offset, per-frame scale/translation or
camera-intrinsic fit. An additive depth bias, wrong human size/translation,
misregistered rasterization or predicted visibility error remains unidentifiable
with this scale-only model and is exposed by residual diagnostics, not repaired.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class FrameDepthSupport:
    frame_index: int
    visible_pixels: int
    valid_correspondences: int
    nonfinite_pairs: int
    nonpositive_finite_pairs: int
    supported: bool
    median_ratio: float | None
    median_absolute_relative_error: float | None


@dataclass(frozen=True)
class DepthScaleAlignment:
    shared_scale: float
    median_absolute_relative_error: float
    ratio_p10: float
    ratio_p90: float
    correspondences: int
    supported_frames: int
    frames: tuple[FrameDepthSupport, ...]

    def to_dict(self) -> dict:
        return {
            "schema": "world-reward-shared-depth-scale-v1",
            "interpretation": "Consistency with estimated human camera-Z, not ground-truth metric accuracy",
            "model": "human_render_Z = shared_positive_scale * estimated_depth_Z; offset fixed zero",
            "weighting": "equal total weight per supported frame; equal pixel weights within each frame",
            **asdict(self),
        }


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _iterator(values: Iterable[np.ndarray], name: str):
    if isinstance(values, np.ndarray) and values.ndim != 3:
        raise ValueError(f"{name} must have shape [T,H,W]")
    try:
        return iter(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of [H,W] arrays") from exc


def _array(value: object, name: str, frame_index: int, *, mask: bool = False) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} frame {frame_index}: masked arrays require an explicit validity mask")
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} frame {frame_index}: invalid array") from exc
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} frame {frame_index}: expected nonempty [H,W]")
    if mask:
        if result.dtype != np.dtype(bool):
            raise ValueError(f"{name} frame {frame_index}: explicit boolean visibility mask required")
    elif not (np.issubdtype(result.dtype, np.integer) or np.issubdtype(result.dtype, np.floating)):
        raise ValueError(f"{name} frame {frame_index}: real numeric camera-Z depth required")
    return result


def _quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    """Weighted empirical quantile; midpoint at an exact cumulative boundary."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    cumulative /= cumulative[-1]
    position = min(int(np.searchsorted(cumulative, probability, side="left")), len(values) - 1)
    value = float(sorted_values[position])
    if position + 1 < len(values) and np.isclose(cumulative[position], probability, rtol=0, atol=1e-12):
        value = (value + float(sorted_values[position + 1])) / 2
    return value


def _positive_exp(value: float, name: str) -> float:
    with np.errstate(over="ignore", under="ignore"):
        result = float(np.exp(value))
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} is outside finite positive floating-point range")
    return result


def fit_shared_depth_scale(
    depths: Iterable[np.ndarray], human_render_depths: Iterable[np.ndarray],
    visible_human_masks: Iterable[np.ndarray], frame_indices: Sequence[int], *,
    min_correspondences_per_frame: int = 32, min_supported_frames: int = 3,
) -> DepthScaleAlignment:
    """Fit one log-ratio weighted median across all supported original frames.

    Inputs are camera-axis Z, **not Euclidean ray range**, in the same full-image
    pixel grid. The visibility mask must come from automatic human rendering and
    occlusion reasoning, not GT. Valid correspondences are explicitly
    ``mask & finite(depth) & finite(human_Z) & (depth>0) & (human_Z>0)``. Invalid
    selected pairs are counted and excluded, never silently filled or clipped.
    Frames lacking pixel support contribute no weight; all others get equal
    total weight, so large human masks cannot dominate smaller ones.

    The weighted median minimizes absolute log-ratio residuals. No residual-based
    outlier/occlusion/motion rejection is applied and no per-frame fit is returned.
    Per-frame ratio medians are diagnostics only; frame indices remain original.
    Arrays [T,H,W] and streaming iterables of [H,W] arrays are both supported.
    """
    minimum_pairs = _integer(min_correspondences_per_frame, "min_correspondences_per_frame", 1)
    minimum_frames = _integer(min_supported_frames, "min_supported_frames", 3)
    if not isinstance(frame_indices, (Sequence, np.ndarray)):
        raise ValueError("frame_indices must be a sequence of original integer indices")
    indices = [_integer(index, "frame_index", 0) for index in frame_indices]
    if not indices or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("frame_indices must be nonempty and strictly increasing")
    statistics = []
    per_frame_logs: list[np.ndarray | None] = []
    dimensions = None
    try:
        for index, estimated, rendered, visibility in zip(
            indices, _iterator(depths, "depths"), _iterator(human_render_depths, "human_render_depths"),
            _iterator(visible_human_masks, "visible_human_masks"), strict=True,
        ):
            estimated = _array(estimated, "depths", index)
            rendered = _array(rendered, "human_render_depths", index)
            visibility = _array(visibility, "visible_human_masks", index, mask=True)
            if (estimated.shape != rendered.shape or estimated.shape != visibility.shape
                    or (dimensions is not None and dimensions != estimated.shape)):
                raise ValueError(f"Depth/render/visibility grids must share fixed [H,W] at frame {index}")
            dimensions = estimated.shape
            finite = np.isfinite(estimated) & np.isfinite(rendered)
            positive = (estimated > 0) & (rendered > 0)
            valid = visibility & finite & positive
            count = int(np.count_nonzero(valid))
            supported = count >= minimum_pairs
            statistics.append((
                index, int(np.count_nonzero(visibility)), count,
                int(np.count_nonzero(visibility & ~finite)),
                int(np.count_nonzero(visibility & finite & ~positive)), supported,
            ))
            if supported:
                # Difference of logs avoids overflowing the raw depth ratio.
                logs = np.log(rendered[valid].astype(np.float64)) - np.log(estimated[valid].astype(np.float64))
                if not np.isfinite(logs).all():
                    raise ValueError(f"Camera-Z log ratio is nonfinite at frame {index}")
                per_frame_logs.append(logs)
            else:
                per_frame_logs.append(None)
    except ValueError as exc:
        if str(exc).startswith("zip() argument"):
            raise ValueError("Depth/render/mask counts must match original frame_indices") from exc
        raise
    supported_logs = [logs for logs in per_frame_logs if logs is not None]
    if len(supported_logs) < minimum_frames:
        raise ValueError(
            f"Insufficient supported frames: {len(supported_logs)} < {minimum_frames}; "
            f"per-frame valid correspondences={[(row[0], row[2]) for row in statistics]}"
        )
    values = np.concatenate(supported_logs)
    weights = np.concatenate([np.full(len(logs), 1.0 / len(logs)) for logs in supported_logs])
    log_scale = _quantile(values, weights, 0.5)
    scale = _positive_exp(log_scale, "shared_scale")
    p10 = _positive_exp(_quantile(values, weights, 0.1), "ratio_p10")
    p90 = _positive_exp(_quantile(values, weights, 0.9), "ratio_p90")
    with np.errstate(over="ignore"):
        residuals = np.abs(np.expm1(log_scale - values))
    error = _quantile(residuals, weights, 0.5)
    if not np.isfinite(error):
        raise ValueError("Median relative alignment residual is nonfinite")
    frames = []
    for statistics_row, logs in zip(statistics, per_frame_logs, strict=True):
        ratio = frame_error = None
        if logs is not None:
            equal = np.ones(len(logs))
            ratio = _positive_exp(_quantile(logs, equal, 0.5), "per-frame median_ratio")
            with np.errstate(over="ignore"):
                residual = np.abs(np.expm1(log_scale - logs))
            frame_error = _quantile(residual, equal, 0.5)
            if not np.isfinite(frame_error):
                raise ValueError("Per-frame median relative alignment residual is nonfinite")
        frames.append(FrameDepthSupport(*statistics_row, ratio, frame_error))
    return DepthScaleAlignment(scale, error, p10, p90, len(values), len(supported_logs), tuple(frames))
