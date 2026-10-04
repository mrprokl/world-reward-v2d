"""The native 24+1 ICP/image candidate pool, independent of point-query costs.

Only one original frame pointmap is consumed at a time. The caller supplies one
physical fixed mesh, its native8192 surface sample (seed0) and mesh.centroid;
these numeric arrays cannot prove sampling provenance, units or reconstruction.
Raster callbacks return NumPy bool[H,W], using the unchanged native camera and
near-plane behavior. Runtime/backend failures propagate, never become evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from world_reward.rigid_alignment import align_observed_points

SURFACE_SAMPLES, MAX_OBSERVATIONS, MIN_OBSERVATIONS = 8192, 2048, 40
HYPOTHESES = 25


def _readonly(array):
    array = np.array(array, copy=True)
    array.flags.writeable = False
    return array


def _real(value, shape, name):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or array.shape != shape or not np.isfinite(array).all()):
        raise ValueError(f'{name}: finite unmasked float32/float64{shape} required')
    return array


def _proper(rotation):
    if (not np.isfinite(rotation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError('Candidate rotation must be proper SO(3)')


def _iou(predicted, observed):
    # Same scalar math as camera_render.silhouette_iou; no torch/infra imports.
    if np.ma.isMaskedArray(predicted):
        raise ValueError('Raster mask must not be a masked array')
    predicted = np.asarray(predicted)
    if predicted.shape != observed.shape or predicted.dtype != np.bool_:
        raise ValueError('Raster callback must return original-grid boolean mask')
    union = np.count_nonzero(predicted | observed)
    if not union:
        raise ValueError('IoU is undefined for an empty union')
    return float(np.count_nonzero(predicted & observed) / union)


@dataclass(frozen=True)
class FrameCandidates:
    frame_index: int
    rotations: np.ndarray
    translations: np.ndarray
    image_costs: np.ndarray
    valid_candidates: np.ndarray
    initial_iou: np.ndarray
    fitted_iou: np.ndarray
    selected_iou: np.ndarray
    initial_residual: np.ndarray
    fitted_residual: np.ndarray
    selected_residual: np.ndarray
    icp_accepted: np.ndarray
    greedy_index: int
    visible_point_pixels: int
    sampled_observations: int
    rejected: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class CandidatePool:
    frame_index: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    image_costs: np.ndarray
    valid_candidates: np.ndarray
    greedy_indices: np.ndarray


class NativeCandidatePool:
    """Sequential native pool builder; never takes tracks, queries, GT or costs.

    ``raster(vertices @ R.T + t, faces, K, width, height) -> bool[H,W]``.
    ``align`` is optional dependency injection for tiny tests; the default is
    align_observed_points(sampled, observed, R, t), with no changed kwargs.
    Invalid slots are identity/zero/cost0 and MUST keep the explicit valid mask.
    A fitted raster ValueError rejects the whole slot, including its initial pose.
    Future seed25 uses the native greedy winner, not Viterbi/point-cost decisions.
    """
    def __init__(self, frame_index, vertices, faces, surface_points, mesh_centroid,
                 initial_rotation, initial_translation, camera_K, width, height):
        indices = np.asarray(frame_index)
        if (np.ma.isMaskedArray(frame_index) or indices.dtype != np.int64 or indices.ndim != 1
                or len(indices) < 3 or not np.array_equal(indices, np.arange(len(indices), dtype=np.int64))):
            raise ValueError('Exact full original int64 arange(T), T>=3 required')
        if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0):
            raise ValueError('Original positive integer image dimensions required')
        vertices = np.asarray(vertices) if not np.ma.isMaskedArray(vertices) else vertices
        if np.ndim(vertices) != 2 or np.shape(vertices)[1:] != (3,) or len(vertices) < 3:
            raise ValueError('Fixed nonempty canonical mesh vertices required')
        vertices = _real(vertices, vertices.shape, 'vertices')
        f = np.asarray(faces)
        if (np.ma.isMaskedArray(faces) or f.dtype.kind not in 'iu' or f.ndim != 2
                or f.shape[1:] != (3,) or not len(f) or np.any(f < 0) or np.any(f >= len(vertices))
                or np.any(f[:, 0] == f[:, 1]) or np.any(f[:, 0] == f[:, 2]) or np.any(f[:, 1] == f[:, 2])):
            raise ValueError('Unmodified triangular integer faces required')
        normals = np.cross(vertices[f[:, 1]]-vertices[f[:, 0]], vertices[f[:, 2]]-vertices[f[:, 0]])
        if not np.isfinite(normals).all() or np.any(np.all(normals == 0, axis=1)):
            raise ValueError('Shared mesh has zero-area or nonfinite triangles')
        surface = _real(surface_points, (SURFACE_SAMPLES, 3), 'native surface sample')
        center = _real(mesh_centroid, (3,), 'native mesh.centroid')
        r = _real(initial_rotation, (3, 3), 'initial rotation'); _proper(r)
        t = _real(initial_translation, (3,), 'initial translation')
        k = _real(camera_K, (3, 3), 'camera K')
        if k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0 or not np.array_equal(k[2], [0, 0, 1]):
            raise ValueError('One original positive-focal zero-skew pinhole K required')
        self.frame_index, self.vertices, self.faces, self.surface_points = map(_readonly, (indices, vertices, f, surface))
        self.mesh_centroid, self.initial_rotation, self.initial_translation, self.camera_K = map(_readonly, (center, r, t, k))
        self.width, self.height = width, height
        self._orientations = _readonly(Rotation.create_group('O').as_matrix())
        self._previous_rotation = self.initial_rotation.copy()
        self._frames: list[FrameCandidates] = []
        self._finalized = False

    def add_frame(self, index, pointmap, automatic_mask, raster, *, align=None):
        if self._finalized or type(index) is not int or index != len(self._frames) or index >= len(self.frame_index):
            raise ValueError('Consume every original frame once in order before finalization')
        mask = np.asarray(automatic_mask)
        if np.ma.isMaskedArray(automatic_mask) or mask.dtype != np.bool_ or mask.shape != (self.height, self.width):
            raise ValueError('Original automatic boolean mask grid required')
        points = np.asarray(pointmap)
        if (np.ma.isMaskedArray(pointmap) or points.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or points.shape != (self.height, self.width, 3)):
            raise ValueError('Original inferred float32/float64 pointmap required')
        if not callable(raster) or (align is not None and not callable(align)):
            raise ValueError('Native raster and alignment callbacks required')
        visible = mask & np.isfinite(points).all(-1) & (points[..., 2] > 0)
        observed = points[visible]
        if len(observed) < MIN_OBSERVATIONS:
            raise ValueError('Insufficient inferred visible object points (<40); no fallback')
        count = len(observed)
        rng = np.random.default_rng(0)
        if len(observed) > MAX_OBSERVATIONS:
            observed = observed[rng.choice(len(observed), MAX_OBSERVATIONS, replace=False)]
        rotations = np.broadcast_to(np.eye(3), (HYPOTHESES, 3, 3)).copy()
        translations = np.zeros((HYPOTHESES, 3)); costs = np.zeros(HYPOTHESES)
        valid = np.zeros(HYPOTHESES, bool); accepted = np.zeros(HYPOTHESES, bool)
        initial_iou, fitted_iou, selected_iou, initial_residual, fitted_residual, selected_residual = [np.zeros(HYPOTHESES) for _ in range(6)]
        rejected = []
        seeds = [self.initial_rotation @ hypothesis for hypothesis in self._orientations]
        seeds.append(self._previous_rotation)
        aligner = align_observed_points if align is None else align
        for number, seed_r in enumerate(seeds):
            seed_t = self.initial_translation if index == 0 else np.median(observed, axis=0) - self.mesh_centroid @ seed_r.T
            try:
                initial = _iou(raster(self.vertices @ seed_r.T + seed_t, self.faces, self.camera_K, self.width, self.height), mask)
                fit = aligner(self.surface_points, observed, seed_r, seed_t)
                fitted = _iou(raster(self.vertices @ fit.rotation.T + fit.translation, self.faces, self.camera_K, self.width, self.height), mask)
                accept = fitted >= initial and fit.final_residual <= fit.initial_residual
                chosen_r, chosen_t = (fit.rotation, fit.translation) if accept else (seed_r, seed_t)
                chosen_iou = fitted if accept else initial
                residual = fit.final_residual if accept else fit.initial_residual
                # Explicit finite proper slots, without changing acceptance/ties.
                chosen_r = _real(chosen_r, (3, 3), 'chosen rotation'); _proper(chosen_r)
                chosen_t = _real(chosen_t, (3,), 'chosen translation')
                if not np.isfinite([fit.initial_residual, fit.final_residual]).all() or min(fit.initial_residual, fit.final_residual) < 0:
                    raise ValueError('Finite nonnegative native residuals required')
                rotations[number], translations[number] = chosen_r, chosen_t
                costs[number], valid[number], accepted[number] = 1-chosen_iou, True, bool(accept)
                initial_iou[number], fitted_iou[number], selected_iou[number] = initial, fitted, chosen_iou
                initial_residual[number], fitted_residual[number], selected_residual[number] = fit.initial_residual, fit.final_residual, residual
            except ValueError as exc:
                rejected.append((number, str(exc)))
        if not valid.any():
            raise ValueError(f'No finite supported hypothesis at original frame {index}')
        greedy = max(np.flatnonzero(valid), key=lambda slot: (selected_iou[slot], -selected_residual[slot], -int(slot)))
        arrays = map(_readonly, (rotations, translations, costs, valid, initial_iou, fitted_iou, selected_iou,
                                initial_residual, fitted_residual, selected_residual, accepted))
        row = FrameCandidates(index, *arrays, int(greedy), count, len(observed), tuple(rejected))
        self._previous_rotation = row.rotations[greedy].copy()
        self._frames.append(row)
        return row

    def finalize(self):
        if self._finalized or len(self._frames) != len(self.frame_index):
            raise ValueError('Complete original frame coverage required, finalized once')
        result = CandidatePool(*map(_readonly, (self.frame_index,
            np.stack([row.rotations for row in self._frames]), np.stack([row.translations for row in self._frames]),
            np.stack([row.image_costs for row in self._frames]), np.stack([row.valid_candidates for row in self._frames]),
            np.array([row.greedy_index for row in self._frames], dtype=np.int64))))
        self._finalized = True
        return result
