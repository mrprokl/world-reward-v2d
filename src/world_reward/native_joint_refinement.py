"""Explicit image-anchored extension of the pinned native joint MHR solver.

The original decoder, contact, penetration, silhouette, body-pose prior,
optimizer loop and schedule stay native. The only new degree of freedom is
human camera translation, independent of object translation. No model/label I/O.
"""
from dataclasses import asdict, dataclass
import numpy as np

from .joint_point_objective import _native_binding
from .root_refit import COCO_TO_MHR, TRAIN_COCO, HELDOUT_COCO


def frozen(value, shape, kind, name, *, finite=True):
    if type(value) is not np.ndarray or value.shape != shape or value.dtype.kind not in kind:
        raise ValueError(name + ': original plain array shape/dtype required')
    if finite and not np.isfinite(value).all():
        raise ValueError(name + ': finite values required')
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(shape)


@dataclass(frozen=True)
class JointImageEvidence:
    """All source-bound material points and native DWPose rows, without a cap.

    Caller proves actual mesh attachment/source ancestry. This numeric transport
    cannot establish material identity, detector correctness or training rights.
    Missing object observations remain NaN; DWPose raw invalid values survive.
    K/coordinates share one unchanged integer-centred RGB grid.
    """
    points: np.ndarray
    object_xy: np.ndarray
    object_supported: np.ndarray
    human_xy: np.ndarray
    human_scores: np.ndarray
    K: np.ndarray
    frame_index: np.ndarray
    allow_empty_object_evidence: bool = False

    def __post_init__(self):
        n, q = len(self.frame_index), len(self.points)
        if (n < 3 or type(self.allow_empty_object_evidence) is not bool
                or (q == 0 and not self.allow_empty_object_evidence)):
            raise ValueError('Full trajectory and nonempty original material bank required')
        rows = dict(points=frozen(self.points, (q, 3), 'f', 'points'),
            object_xy=frozen(self.object_xy, (n, q, 2), 'f', 'object XY', finite=False),
            object_supported=frozen(self.object_supported, (n, q), 'b', 'object support'),
            human_xy=frozen(self.human_xy, (n, 133, 2), 'f', 'native DWPose XY', finite=False),
            human_scores=frozen(self.human_scores, (n, 133), 'f', 'native DWPose scores'),
            K=frozen(self.K, (3, 3), 'f', 'K'),
            frame_index=frozen(self.frame_index, (n,), 'i', 'original frame index'))
        valid = rows['object_supported']; xy = rows['object_xy']; k = rows['K']
        if (not np.array_equal(rows['frame_index'], np.arange(n))
                or not np.isfinite(xy[valid]).all() or not np.isnan(xy[~valid]).all()
                or not np.isfinite(rows['human_xy'][rows['human_scores'] > 0]).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0
                or not np.array_equal(k[2], [0, 0, 1])):
            raise ValueError('Original chronology/support and fixed pinhole camera required')
        for key, value in rows.items(): object.__setattr__(self, key, value)

    def object_counts(self):
        return self.object_supported[1:].sum(0)


@dataclass(frozen=True)
class NativeJointConfig:
    image_sigma_px: float
    object_image_weight: float
    human_image_weight: float
    human_translation_prior_weight: float
    human_translation_temporal_weight: float
    human_translation_lr: float
    development_reference: str

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'development_reference':
                if type(value) is not str or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
                    raise ValueError('Explicit bounded non-challenge development reference required')
            elif type(value) not in (float, int) or not np.isfinite(value) or value < 0:
                raise ValueError('Explicit finite nonnegative coefficient required')
        if self.image_sigma_px <= 0 or self.human_translation_lr <= 0:
            raise ValueError('Positive image scale and translation learning rate required')


def reprojection_statistics(xyz, K, observed, supported, sigma, *, omit_initializer=False):
    """Tiny NumPy DEV/QA reference to exactly the Torch pseudo-Huber objective."""
    xyz = np.asarray(xyz); observed = np.asarray(observed); support = supported.copy()
    if (xyz.shape != (*support.shape, 3) or observed.shape != (*support.shape, 2)
            or support.dtype != bool or not np.isfinite(xyz).all() or (xyz[..., 2] <= 0).any()
            or not np.isfinite(observed[support]).all() or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError('Positive full-T geometry and literal supported observations required')
    if omit_initializer: support[0] = False
    projected = xyz[..., :2] / xyz[..., 2:] * np.diag(K)[:2] + K[:2, 2]
    squared = np.zeros(support.shape)
    squared[support] = np.square((projected[support] - observed[support]) / sigma).sum(-1)
    rho = squared / (np.sqrt(1 + squared) + 1)
    counts = support.sum(0)
    # All original tracks remain; unobserved tracks contribute no invented data.
    per_point = np.divide(rho.sum(0), counts, out=np.zeros(len(counts)), where=counts > 0)
    errors = np.linalg.norm(projected[support] - observed[support], axis=-1)
    return dict(loss=float(per_point.mean()), observations=int(support.sum()),
        unsupported_points=int((counts == 0).sum()),
        mean_px=float(errors.mean()) if len(errors) else None,
        p95_px=float(np.quantile(errors, .95)) if len(errors) else None)


def _image_loss(torch, xyz, K, observed, support, sigma):
    if not torch.isfinite(xyz).all() or (xyz[..., 2] <= 0).any():
        raise ValueError('Every predicted point must have positive finite camera Z')
    xy = xyz[..., :2] / xyz[..., 2:] * torch.stack((K[0, 0], K[1, 1])) + K[:2, 2]
    # Do not evaluate residuals at missing/NaN observations (0*NaN is still NaN).
    residual = torch.where(support[..., None], xy - observed, torch.zeros_like(xy)) / sigma
    squared = residual.square().sum(-1)
    rho = squared / (torch.sqrt(1 + squared) + 1)
    if xyz.shape[1] == 0: return xyz.sum() * 0
    return (rho.sum(0) / support.sum(0).clamp_min(1)).mean()


def native_joint_optimizer_class(native, evidence, config, *, enabled=True):
    """Return an explicit subclass, never mutate upstream methods or globals.

    enabled=False delegates the original loss/decode/parameter groups exactly.
    Caller authenticates transitive native assets. This adapter supports only
    the original full-T/fixed-R/fixed-internal-body-translations parity protocol.
    """
    if type(evidence) is not JointImageEvidence or type(config) is not NativeJointConfig or type(enabled) is not bool:
        raise ValueError('Explicit evidence/config/boolean mode required')
    _, base, check = _native_binding(native)

    class NativeJointOptimizer(base):
        def _optimizer_parameter_groups(self):
            groups = super()._optimizer_parameter_groups()
            if enabled:
                self.human_translation = self.params_fixed['mhr_trans'].detach().clone().requires_grad_(True)
                groups.append(dict(params=[self.human_translation], lr=config.human_translation_lr))
            return groups

        def _decode_body(self, indices):
            if not enabled or not hasattr(self, 'human_translation'):
                return super()._decode_body(indices)
            params = {key: value.index_select(0, indices) for key, value in self.params_fixed.items()}
            params['mhr_body_pose_cont'] = self._body_pose_for_indices(indices)
            params['mhr_trans'] = self.human_translation.index_select(0, indices)
            self._joint_last_body = self.mhr_layer.mhr_forward(params)
            return self._joint_last_body

        def __init__(self, bundle, vertices, faces, cfg, *, mhr_layer):
            check()
            if any(getattr(cfg, key, None) != value for key, value in dict(num_steps=300,
                    batch_size=0, frame_start=0, frame_limit=0, freeze_object_rotation=True,
                    freeze_body_internal_translations=True, checkpoint_path=None).items()):
                raise ValueError('Unchanged full-T native300/fixed-R protocol required')
            if (bundle.get('gt') != {} or bundle['metadata'].get('ground_truth_used') is not False
                    or bundle.get('frames') != [f'{i:06d}' for i in evidence.frame_index]):
                raise ValueError('Exact automatic no-GT original full-T bundle required')
            super().__init__(bundle, vertices, faces, cfg, mhr_layer=mhr_layer)
            if not np.array_equal(self.frame_indices, evidence.frame_index):
                raise ValueError('Native full chronology changed')
            if enabled:
                torch = native._torch()
                constant = lambda a: torch.tensor(np.array(a, copy=True), dtype=self.dtype, device=self.device)
                self._joint_constants = dict(points=constant(evidence.points), K=constant(evidence.K),
                    object_xy=constant(evidence.object_xy),
                    human_xy=constant(evidence.human_xy[:, TRAIN_COCO]),
                    object_support=torch.tensor(np.array(evidence.object_supported, copy=True), device=self.device),
                    human_support=torch.tensor(evidence.human_scores[:, TRAIN_COCO] > 0, device=self.device),
                    human_ids=torch.tensor([COCO_TO_MHR[i] for i in TRAIN_COCO], device=self.device))
                self._joint_constants['object_support'][0] = False

        def loss(self, indices, step, *, include_diagnostics=True):
            check()
            original = super().loss(indices, step, include_diagnostics=include_diagnostics)
            if not enabled: return original
            torch = native._torch()
            if not torch.equal(indices, torch.arange(len(evidence.frame_index), device=self.device)):
                raise ValueError('Ordered full-T loss required; no subsampling')
            c = self._joint_constants
            rotation, translation, _, _ = self._object_state(indices, include_surface=False)
            xyz = c['points'][None] @ rotation.transpose(-1, -2) + translation[:, None]
            object_loss = _image_loss(torch, xyz, c['K'], c['object_xy'], c['object_support'], config.image_sigma_px)
            keypoints = native._layer_output_keypoints(self._joint_last_body).index_select(1, c['human_ids'])
            human_loss = _image_loss(torch, keypoints, c['K'], c['human_xy'], c['human_support'], config.image_sigma_px)
            prior = (self.human_translation - self.params_fixed['mhr_trans']).square().mean()
            temporal = native._acceleration_loss(self.human_translation)
            extra = dict(loss_joint_object_RGB=object_loss * config.object_image_weight,
                loss_joint_human_RGB=human_loss * config.human_image_weight,
                loss_joint_human_translation_prior=prior * config.human_translation_prior_weight,
                loss_joint_human_translation_temporal=temporal * config.human_translation_temporal_weight)
            total = original[0] + sum(extra.values())
            self._joint_last_body = None  # Do not retain a second full-clip autograd graph.
            return total, {**original[1], **extra, 'loss_total': total}

        def result(self):
            result = super().result()
            if not enabled: return result
            result['pr']['mhr_trans'] = self.human_translation.detach().cpu().numpy().copy()
            post = result['postopt']
            post['mode'] = 'native_joint_image_translation_extension_v1'
            post['fixed_parameters'].remove('mhr_trans')
            post['optimized_parameters'].append('mhr_trans')
            post['joint_extension'] = dict(config=asdict(config), points=len(evidence.points),
                human_training_COCO=list(TRAIN_COCO), human_reserved_COCO=list(HELDOUT_COCO),
                native_source_modified=False, image_compute_dtype='native_FP32_raw_evidence_preserved',
                independent_human_object_translation=True, depth_used=False,
                calibrated_probabilities=False, production_adopted=False, heldout_4D_accuracy_verified=False)
            check()
            return result

    return NativeJointOptimizer
