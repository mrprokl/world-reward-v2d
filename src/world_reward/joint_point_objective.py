"""Explicit persistent-point extension of native joint HOI refinement.

No tracker, query selection, camera/shape fit, label I/O or native monkeypatch.
Torch is lazy; callers authenticate upstream assets and calibrate the REQUIRED
scale/weight externally. Source compatibility is not runtime/quality qualification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path
import types

import numpy as np

from .exact_triangle_predicates import validate_exact_triangle_non_degeneracy

NATIVE_SOURCE_SHA256 = "84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b"
FRAME_CONVENTION = "native_aligned_object_to_shared_opencv_camera"


def _text(value, name):
    if (type(value) is not str or not value.strip() or len(value) > 256
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError(f"{name}: bounded explicit reference required")
    return value


def _array(value, name, *, dtype=None, shape=None):
    a = np.asarray(value)
    if (np.ma.isMaskedArray(value) or (dtype is not None and a.dtype != dtype)
            or (dtype is None and a.dtype not in (np.dtype('float32'), np.dtype('float64')))
            or (shape is not None and a.shape != shape) or not np.isfinite(a).all()):
        raise ValueError(f"{name}: exact finite unmasked array required")
    # Immutable bytes backing: even flags.writeable=True cannot mutate a source.
    return np.frombuffer(a.tobytes(order='C'), dtype=a.dtype).reshape(a.shape)


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.dtype.str.encode()+b'\0'+str(a.shape).encode()+b'\0'); h.update(a.tobytes(order='C'))
    return h.hexdigest()


@dataclass(frozen=True)
class PointObjectiveConfig:
    residual_scale_256_px: float
    weight: float
    calibration_reference: str

    def __post_init__(self):
        for value, name, positive in ((self.residual_scale_256_px, 'scale', True), (self.weight, 'weight', False)):
            if (type(value) not in (float, int) or not math.isfinite(value)
                    or value < 0 or (positive and value == 0)):
                raise ValueError(f"Externally fixed finite {'positive' if positive else 'nonnegative'} {name} required")
        _text(self.calibration_reference, 'calibration reference')


@dataclass(frozen=True, eq=False)
class FixedTriangleTracks:
    """All frozen attachments are in the ACTUAL native loaded mesh frame.

    Indices are original positional arange(T); source IDs remain separate.
    Query points are original (time=0,Y,X), tracks are native continuous XY256.
    Visibility is support, not confidence. Unsupported but finite out-of-grid
    native coordinates reject the whole input too; nothing is clipped/dropped.
    Caller supplies triangle IDs/barycentrics selected BEFORE future tracking.
    This object binds bytes, not the truth of that producer/source statement.
    """
    vertices: np.ndarray
    faces: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray
    K: np.ndarray
    image_size: tuple[int, int]
    frame_index: np.ndarray
    source_frame_ids: np.ndarray
    query_ids: np.ndarray
    query_points: np.ndarray
    tracks_256: np.ndarray
    native_visible: np.ndarray
    native_frame_names: tuple[str, ...]
    source_references: tuple[str, ...]
    frame_convention: str

    def __post_init__(self):
        if (type(self.image_size) is not tuple or len(self.image_size) != 2
                or any(type(x) is not int or x <= 0 for x in self.image_size)
                or self.frame_convention != FRAME_CONVENTION
                or type(self.source_references) is not tuple or not self.source_references):
            raise ValueError('Explicit original grid/native frame/source references required')
        for ref in self.source_references: _text(ref, 'source reference')
        v = _array(self.vertices, 'native vertices', dtype=np.float32)
        f = _array(self.faces, 'native faces', dtype=np.int64)
        if (v.ndim != 2 or v.shape[1:] != (3,) or f.ndim != 2 or f.shape[1:] != (3,)
                or len(v) < 3 or not len(f) or np.any(f < 0) or np.any(f >= len(v))):
            raise ValueError('Complete native F32 Vx3 / I64 Fx3 required')
        ids = _array(self.face_indices, 'attachment face IDs', dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= len(f)):
            raise ValueError('All fixed attachment triangle IDs required')
        q = len(ids); bary = _array(self.barycentric, 'barycentrics', shape=(q, 3))
        if np.any(bary < 0) or np.any(bary > 1) or np.any(np.abs(bary.sum(-1)-1) > 8*np.finfo(bary.dtype).eps):
            raise ValueError('Supplied convex barycentrics required; never renormalized')
        validate_exact_triangle_non_degeneracy(v.astype(np.float64), f[ids])
        points = np.sum(v[f[ids]].astype(np.float64)*bary[:, :, None], axis=1)
        if len(np.unique(points, axis=0)) != q: raise ValueError('Duplicated attached points cannot gain weight')
        with np.errstate(over='ignore', invalid='ignore'):
            native_points = (v[f[ids]]*bary.astype(np.float32)[:, :, None]).sum(axis=1)
        if not np.isfinite(native_points).all() or len(np.unique(native_points, axis=0)) != q:
            raise ValueError('Native F32 attachment coordinates collapse/overflow; no query replacement')
        index = _array(self.frame_index, 'frame_index', dtype=np.int64)
        if index.ndim != 1 or len(index) < 3 or not np.array_equal(index, np.arange(len(index), dtype=np.int64)):
            raise ValueError('Full original arange(T), T>=3 required')
        t = len(index); source = _array(self.source_frame_ids, 'source IDs', dtype=np.int64, shape=(t,))
        if (type(self.native_frame_names) is not tuple or len(self.native_frame_names) != t
                or len(set(self.native_frame_names)) != t):
            raise ValueError('Exact full ordered native frame names required')
        for name in self.native_frame_names: _text(name, 'native frame name')
        query_ids = _array(self.query_ids, 'query IDs', dtype=np.int64, shape=(q,))
        if np.any(source < 0) or np.any(source[1:] <= source[:-1]) or not np.array_equal(query_ids, np.arange(q)):
            raise ValueError('Unique ordered source IDs and exact original query slots required')
        queries = _array(self.query_points, 'initial queries', shape=(q, 3))
        h, w = self.image_size
        if (np.any(queries[:, 0] != 0) or np.any(queries[:, 1:] < 0)
                or np.any(queries[:, 1] >= h) or np.any(queries[:, 2] >= w)):
            raise ValueError('Every query initializes at original time0 inside original RGB')
        if len(np.unique(queries, axis=0)) != q: raise ValueError('Duplicate query slots are not independent evidence')
        tracks = _array(self.tracks_256, 'native tracks', shape=(t, q, 2))
        if np.any(tracks < 0) or np.any(tracks >= 256): raise ValueError('Out-of-grid track evidence is unsupported; no clipping')
        visible = _array(self.native_visible, 'native support', dtype=np.bool_, shape=(t, q))
        k = _array(self.K, 'original RGB-derived K', shape=(3, 3))
        if (k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0
                or not np.array_equal(k[2], [0, 0, 1])):
            raise ValueError('Fixed positive-focal OpenCV pixel K required')
        for name, value in dict(vertices=v, faces=f, face_indices=ids, barycentric=bary, K=k,
                frame_index=index, source_frame_ids=source, query_ids=query_ids, query_points=queries,
                tracks_256=tracks, native_visible=visible).items(): object.__setattr__(self, name, value)

    @property
    def mesh_sha256(self): return _digest(self.vertices, self.faces)

    @property
    def evidence_sha256(self):
        h = hashlib.sha256(_digest(*(getattr(self, f.name) for f in fields(self)
            if isinstance(getattr(self, f.name), np.ndarray))).encode())
        h.update(repr((self.image_size, self.native_frame_names, self.source_references, self.frame_convention)).encode())
        return h.hexdigest()

    def support_counts(self): return self.native_visible[1:].sum(axis=0, dtype=np.int64)


def point_reprojection_loss(rotation, translation, evidence, config, frame_index):
    """Track-equal isotropic smooth pseudo-Huber, dimensionless XY256/scale.

    Every geometry projection must have positive Z, even if occluded. Projected
    points may leave the image and incur loss; input native tracks may not.
    Zero-supported frames remain in full T. A wholly unsupported track rejects
    positive-weight fitting rather than silently removing/reweighting it.
    Weight0 returns connected zero without altering the native baseline loss.
    """
    import torch
    if type(evidence) is not FixedTriangleTracks or type(config) is not PointObjectiveConfig:
        raise ValueError('Explicit frozen point evidence/config required')
    if (not torch.is_tensor(rotation) or not torch.is_tensor(translation)
            or rotation.dtype not in (torch.float32, torch.float64) or translation.dtype != rotation.dtype
            or rotation.device != translation.device or tuple(rotation.shape) != (len(evidence.frame_index), 3, 3)
            or tuple(translation.shape) != (len(evidence.frame_index), 3)
            or not torch.isfinite(rotation).all() or not torch.isfinite(translation).all()
            or not torch.is_tensor(frame_index) or frame_index.dtype != torch.int64
            or frame_index.device != translation.device
            or not torch.equal(frame_index, torch.arange(len(evidence.frame_index), device=translation.device))):
        raise ValueError('Finite full-T native transforms and unchanged ordered indices required')
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    if not torch.allclose(rotation @ rotation.transpose(-1, -2), eye, atol=1e-5, rtol=0) or not torch.allclose(
            torch.linalg.det(rotation), torch.ones(len(rotation), dtype=rotation.dtype, device=rotation.device), atol=1e-5, rtol=0):
        raise ValueError('Proper rotations required; never repaired')
    if config.weight == 0: return translation.sum()*0
    counts = evidence.support_counts()
    if np.any(counts == 0): raise ValueError('Track without post-initializer support: no credible point fit')
    def constant(a): return torch.tensor(np.array(a, copy=True), dtype=rotation.dtype, device=rotation.device)
    triangles = constant(evidence.vertices[evidence.faces[evidence.face_indices]])
    points = (triangles*constant(evidence.barycentric)[:, :, None]).sum(dim=1)
    if not torch.isfinite(points).all() or len(torch.unique(points, dim=0)) != len(points):
        raise ValueError('Actual compute-dtype attachments duplicate/overflow; no witness filtering')
    xyz = points[None] @ rotation.transpose(-1, -2) + translation[:, None]
    if not torch.isfinite(xyz).all() or (xyz[..., 2] <= 0).any(): raise ValueError('Nonpositive/overflowing projection: no witness dropping')
    k = constant(evidence.K); h, w = evidence.image_size
    xy = (xyz[..., :2]/xyz[..., 2, None]*torch.stack((k[0, 0], k[1, 1]))+k[:2, 2])
    residual = (xy*constant(np.array([256/w, 256/h]))-constant(evidence.tracks_256))/config.residual_scale_256_px
    squared = residual.square().sum(-1)
    rho = squared/(torch.sqrt(1+squared)+1)  # sqrt(1+||e/s||²)-1; smooth, unsaturated.
    support = torch.tensor(np.array(evidence.native_visible, copy=True), device=rotation.device)
    support[0] = False  # Forced initializer does not count as validation evidence.
    value = config.weight*((rho*support).sum(0)/constant(counts)).mean()
    if not torch.isfinite(value): raise ValueError('Point-loss arithmetic overflowed; no replacement zero')
    return value


def _code_signature(code):
    return (code.co_code, tuple(_code_signature(c) if isinstance(c, types.CodeType) else c for c in code.co_consts),
            code.co_names, code.co_varnames, code.co_argcount, code.co_kwonlyargcount, code.co_flags,
            code.co_freevars, code.co_cellvars)


def _native_binding(native):
    """Bind original file AND live class bytecode without executing source/imports."""
    path = Path(native.__file__)
    if not path.is_file() or path.is_symlink(): raise ValueError('Actual original native source file required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NATIVE_SOURCE_SHA256: raise ValueError('Unqualified native source SHA')
    compiled = compile(raw, str(path), 'exec', dont_inherit=True)
    codes = {c.co_name: c for c in compiled.co_consts if isinstance(c, types.CodeType)}
    cls = native.MHRParityPostOptimizer
    if cls.__module__ != native.__name__ or cls.__bases__ != (object,): raise ValueError('Original native class identity required')
    expected = {c.co_name: c for c in codes['MHRParityPostOptimizer'].co_consts if isinstance(c, types.CodeType)}
    def check():
        live = {k: v for k, v in vars(cls).items() if isinstance(v, types.FunctionType)}
        if (set(live) != set(expected) or any(live[k].__globals__ is not vars(native) or
                _code_signature(live[k].__code__) != _code_signature(v) for k, v in expected.items())):
            raise ValueError('Native optimizer live methods changed; no patch permitted')
        if _code_signature(native._torch.__code__) != _code_signature(codes['_torch']):
            raise ValueError('Native Torch accessor changed')
    check()
    return path, cls, check


def native_point_optimizer_class(native, evidence, point_config):
    """An executable explicit subclass, not mutation of native globals/classes.

    Only fixed-R/full-batch parity is supported. Caller authenticates all native
    transitive helpers/model/data before constructing it. Existing unchanged
    full-refine receipt validators must NOT label this new objective as baseline.
    """
    if type(evidence) is not FixedTriangleTracks or type(point_config) is not PointObjectiveConfig:
        raise ValueError('Frozen evidence and externally calibrated config required')
    path, base, check = _native_binding(native); digest = evidence.evidence_sha256
    config_dict = asdict(point_config)

    class JointPointOptimizer(base):
        def __init__(self, bundle, object_vertices, object_faces, cfg, *, mhr_layer):
            check()
            if cfg is None or any(type(getattr(cfg, k, None)) is not type(v) or getattr(cfg, k) != v for k, v in
                    dict(num_steps=300, batch_size=0, frame_start=0, frame_limit=0,
                        freeze_object_rotation=True, freeze_body_internal_translations=True).items()):
                raise ValueError('Explicit unchanged fixed-R/full-T/native control configuration required')
            v = _array(object_vertices, 'actual native vertices', dtype=np.float32)
            f = _array(object_faces, 'actual native faces', dtype=np.int64)
            if (_digest(v, f) != evidence.mesh_sha256 or len(bundle['pr']['pose_abs']) != len(evidence.frame_index)
                    or tuple(bundle.get('frames', ())) != evidence.native_frame_names):
                raise ValueError('Exact native mesh and original full timeline must bind point evidence')
            if point_config.weight > 0 and np.any(evidence.support_counts() == 0): raise ValueError('Unsupported track blocks point fitting')
            self._point_native_config = asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else dict(vars(cfg))
            super().__init__(bundle, object_vertices, object_faces, cfg, mhr_layer=mhr_layer)
            if not np.array_equal(self.frame_indices, evidence.frame_index): raise ValueError('Native timeline selection changed')

        def loss(self, indices, step, *, include_diagnostics=True):
            check()
            current_cfg = asdict(self.cfg) if hasattr(self.cfg, '__dataclass_fields__') else dict(vars(self.cfg))
            if evidence.evidence_sha256 != digest or asdict(point_config) != config_dict or current_cfg != self._point_native_config:
                raise ValueError('Frozen point evidence/config changed')
            torch = native._torch()
            if (not torch.is_tensor(indices) or indices.dtype != torch.int64
                    or not torch.equal(indices, torch.arange(len(evidence.frame_index), device=indices.device))
                    or _digest(self.object_vertices.detach().cpu().numpy(), self.object_faces.detach().cpu().numpy()) != evidence.mesh_sha256
                    or not self.cfg.freeze_object_rotation or self.object_axis.requires_grad):
                raise ValueError('Native fixed geometry/rotation contract changed')
            total, metrics = super().loss(indices, step, include_diagnostics=include_diagnostics)
            if point_config.weight == 0: return total, metrics  # Exact baseline objects, no +0 rounding.
            if not torch.equal(self._object_rotation_for_indices(indices), self.object_rotation_initial):
                raise ValueError('Object rotation changed; unlock requires separate experiment')
            rotation, translation, _, _ = self._object_state(indices, include_surface=False)
            point = point_reprojection_loss(rotation, translation, evidence, point_config, indices)
            combined = total+point
            return combined, {**metrics, 'loss_point_reprojection': point, 'loss_total': combined}

        def run(self):
            if hashlib.sha256(path.read_bytes()).hexdigest() != NATIVE_SOURCE_SHA256: raise ValueError('Native source changed before fit')
            result = super().run()
            check()
            current_cfg = asdict(self.cfg) if hasattr(self.cfg, '__dataclass_fields__') else dict(vars(self.cfg))
            if (hashlib.sha256(path.read_bytes()).hexdigest() != NATIVE_SOURCE_SHA256 or evidence.evidence_sha256 != digest
                    or asdict(point_config) != config_dict or current_cfg != self._point_native_config):
                raise ValueError('Native source/evidence changed during fit')
            result['postopt']['point_objective'] = dict(config=asdict(point_config), evidence_sha256=digest,
                mesh_sha256=evidence.mesh_sha256, after_initializer_support=evidence.support_counts().tolist(),
                native_source_sha256=NATIVE_SOURCE_SHA256, object_rotation_fixed=True, track_equal=True,
                support_is_confidence=False, calibration_reference_authenticated=False,
                calibrated_probabilities=False, quality_verified=False, adoption=False)
            return result

    return JointPointOptimizer


def run_joint_point_refinement(native, bundle, vertices, faces, cfg, *, mhr_layer, evidence, point_config):
    """Native callback-shaped entrypoint; no new CLI, renderer or fit loop."""
    cls = native_point_optimizer_class(native, evidence, point_config)
    return cls(bundle, vertices, faces, cfg, mhr_layer=mhr_layer).run()
