"""Own procedural reference, not MHR output, predictions or anatomical ground truth.

Primitives manufacture ONE rest boundary; they are never concatenated as meshes.
The sole 12 mm grid and three authored poses are fixed before any evaluation.
Meshing/skinning do not certify embedding: the caller must audit every original
face at rest, both extreme poses and every subsequently manufactured frame.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

RESOLUTION_M = .012
PADDING_M = .060
SMOOTH_UNION_M = .006
WEIGHT_BLEND_M = .018
POSE_NAMES = ("rest", "carry", "fingerbend")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGERTIP_NAMES = tuple(f"{side}_{finger}_tip" for side in ("l", "r") for finger in FINGER_NAMES)


def _array(value, dtype, shape, name):
    if np.ma.isMaskedArray(value):
        raise ValueError(f"Masked {name} forbidden")
    original = np.asarray(value)
    if original.dtype != np.dtype(dtype) or original.shape != shape or not np.isfinite(original).all():
        raise ValueError(f"Finite exact {dtype} {name} with shape {shape} required")
    result = original.copy()
    result.setflags(write=False)
    return result


def _triple(value, name, positive=False):
    if (type(value) is not tuple or len(value) != 3 or any(type(v) not in (int, float)
            or not math.isfinite(v) or positive and v <= 0 for v in value)):
        raise ValueError(f"Finite {'positive ' if positive else ''}three-tuple {name} required")


@dataclass(frozen=True)
class Primitive:
    name: str
    kind: str
    bone: str
    start: tuple
    end: tuple
    radii: tuple

    def __post_init__(self):
        if not self.name or not self.bone or self.kind not in ("capsule", "ellipsoid"):
            raise ValueError("Named capsule/ellipsoid and bone required")
        _triple(self.start, "start"); _triple(self.end, "end"); _triple(self.radii, "radii", True)
        if self.kind == "capsule" and (self.start == self.end or len(set(self.radii)) != 1):
            raise ValueError("Nonzero segment and isotropic capsule radius required")
        if self.kind == "ellipsoid" and self.start != self.end:
            raise ValueError("Ellipsoid start/end must be its identical center")


@dataclass(frozen=True, eq=False)
class Rig:
    names: tuple
    parents: np.ndarray
    rest_joints: np.ndarray
    primitives: tuple
    local_rotations: np.ndarray

    def __post_init__(self):
        n = len(self.names)
        if (type(self.names) is not tuple or n < 2 or len(set(self.names)) != n
                or any(type(s) is not str or not s for s in self.names)):
            raise ValueError("Unique topologically ordered joint names required")
        parents = _array(self.parents, np.int64, (n,), "parents")
        if parents[0] != -1 or any(not 0 <= parents[i] < i for i in range(1, n)):
            raise ValueError("One first root and parent-before-child tree required")
        joints = _array(self.rest_joints, np.float64, (n, 3), "rest joints")
        rotations = _array(self.local_rotations, np.float64, (3, n, 3, 3), "local rotations")
        identity = np.broadcast_to(np.eye(3), (n, 3, 3))
        if (rotations[0].tobytes() != np.ascontiguousarray(identity).tobytes()
                or not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-12, rtol=0)
                or not np.allclose(np.linalg.det(rotations), 1., atol=1e-12, rtol=0)):
            raise ValueError("Exact neutral identity and proper rotations required")
        if (type(self.primitives) is not tuple or not self.primitives
                or any(type(p) is not Primitive or p.bone not in self.names for p in self.primitives)
                or len({p.name for p in self.primitives}) != len(self.primitives)):
            raise ValueError("Ordered unique authored primitives bound to named bones required")
        for name, value in (("parents", parents), ("rest_joints", joints), ("local_rotations", rotations)):
            object.__setattr__(self, name, value)


def _rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x": return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    if axis == "y": return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    if axis == "z": return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    raise ValueError("Authored rotation axis required")


def author_rig():
    """One fixed 1.8 m broad-adult reference with ten separately authored digits.

    x is lateral, y is up and z is anterior, in metres. Named fingertip joints
    are endpoints, not independently skinned bones. Fingers have 11--13 mm
    radii: the fixed 12 mm grid may fail to preserve them; never resolution-sweep.
    """
    names, parents, joints, primitives = [], [], [], []
    def joint(name, parent, position):
        parents.append(-1 if parent is None else names.index(parent)); names.append(name); joints.append(position)
    def ellipsoid(name, bone, center, radii):
        primitives.append(Primitive(name, "ellipsoid", bone, tuple(center), tuple(center), tuple(radii)))
    def capsule(name, bone, start, end, radius):
        a, b = joints[names.index(start)], joints[names.index(end)]
        primitives.append(Primitive(name, "capsule", bone, tuple(a), tuple(b), (radius,) * 3))

    joint("pelvis", None, (0., .95, 0.))
    joint("spine1", "pelvis", (0., 1.12, 0.))
    joint("spine2", "spine1", (0., 1.32, 0.))
    joint("neck", "spine2", (0., 1.53, 0.))
    joint("head", "neck", (0., 1.66, 0.))
    ellipsoid("pelvis", "pelvis", (0., .96, 0.), (.175, .145, .105))
    ellipsoid("abdomen", "spine1", (0., 1.145, 0.), (.152, .175, .102))
    ellipsoid("chest", "spine2", (0., 1.345, 0.), (.202, .165, .115))
    capsule("neck", "neck", "spine2", "head", .059)
    ellipsoid("head", "head", (0., 1.66, 0.), (.098, .135, .092))
    for side, sign in (("l", 1.), ("r", -1.)):
        def label(name): return side + "_" + name
        joint(label("hip"), "pelvis", (sign * .095, .925, 0.))
        joint(label("knee"), label("hip"), (sign * .105, .55, .015))
        joint(label("ankle"), label("knee"), (sign * .105, .12, 0.))
        joint(label("toe"), label("ankle"), (sign * .105, .055, .185))
        capsule(label("thigh"), label("hip"), label("hip"), label("knee"), .088)
        capsule(label("shin"), label("knee"), label("knee"), label("ankle"), .055)
        ellipsoid(label("foot"), label("ankle"), (sign * .105, .070, .080), (.065, .070, .160))
        joint(label("shoulder"), "spine2", (sign * .180, 1.445, 0.))
        joint(label("elbow"), label("shoulder"), (sign * .390, 1.290, .010))
        joint(label("wrist"), label("elbow"), (sign * .530, 1.125, .020))
        joint(label("palm"), label("wrist"), (sign * .530, 1.050, .025))
        capsule(label("upperarm"), label("shoulder"), label("shoulder"), label("elbow"), .052)
        capsule(label("forearm"), label("elbow"), label("elbow"), label("wrist"), .041)
        capsule(label("wrist_link"), label("wrist"), label("wrist"), label("palm"), .028)
        ellipsoid(label("palm"), label("palm"), (sign * .530, 1.050, .025), (.075, .073, .031))
        finger_points = {
            "thumb": ((.598, 1.063, .025), (.643, 1.035, .021), (.679, 1.010, .013), (.705, .993, .005)),
            "index": ((.479, 1.005, .025), (.472, .965, .025), (.465, .933, .025), (.461, .908, .025)),
            "middle": ((.513, 1.005, .025), (.511, .961, .025), (.509, .925, .025), (.508, .898, .025)),
            "ring": ((.547, 1.005, .025), (.550, .964, .025), (.553, .929, .025), (.555, .903, .025)),
            "pinky": ((.581, 1.005, .025), (.589, .972, .025), (.595, .945, .025), (.600, .925, .025)),
        }
        for finger in FINGER_NAMES:
            parent = label("palm")
            for part, (x, y, z) in zip(("mcp", "pip", "dip", "tip"), finger_points[finger]):
                name = label(finger + "_" + part)
                joint(name, parent, (sign * x, y, z)); parent = name
            radius = .013 if finger == "thumb" else .011
            for start, end in (("mcp", "pip"), ("pip", "dip"), ("dip", "tip")):
                bone = label(finger + "_" + start)
                capsule(bone, bone, bone, label(finger + "_" + end), radius)
    q = np.broadcast_to(np.eye(3), (3, len(names), 3, 3)).copy()
    for pose, elbow, flexion in ((1, -.45, -.30), (2, -.60, -.65)):
        for side, sign in (("l", 1.), ("r", -1.)):
            q[pose, names.index(side + "_shoulder")] = _rotation("y", -sign * .90) @ _rotation("z", sign * .42)
            q[pose, names.index(side + "_elbow")] = _rotation("x", elbow)
            q[pose, names.index(side + "_wrist")] = _rotation("x", -.10)
            for finger in FINGER_NAMES:
                for part, gain in (("mcp", 1.), ("pip", 1.08), ("dip", .70)):
                    q[pose, names.index(f"{side}_{finger}_{part}")] = _rotation("x", flexion * gain)
    return Rig(tuple(names), np.array(parents, np.int64), np.array(joints, np.float64), tuple(primitives), q)


def _points(points):
    if np.ma.isMaskedArray(points): raise ValueError("Masked points forbidden")
    p = np.asarray(points)
    if p.dtype not in (np.dtype("float32"), np.dtype("float64")) or p.ndim < 1 or p.shape[-1] != 3 or not np.isfinite(p).all():
        raise ValueError("Finite float32/64 points with last dimension three required")
    return p.astype(np.float64, copy=False)


def primitive_field(points, primitive):
    """Negative inside; ellipsoid field has exact boundary, not exact distance."""
    p = _points(points)
    if type(primitive) is not Primitive: raise ValueError("Authored primitive required")
    a, radii = np.asarray(primitive.start), np.asarray(primitive.radii)
    if primitive.kind == "ellipsoid":
        return (np.linalg.norm((p - a) / radii, axis=-1) - 1.) * radii.min()
    edge = np.asarray(primitive.end) - a
    t = np.clip(np.sum((p - a) * edge, axis=-1) / np.dot(edge, edge), 0., 1.)
    return np.linalg.norm(p - a - t[..., None] * edge, axis=-1) - radii[0]


def smooth_union(first, second):
    """Fixed-width polynomial smooth minimum, never a geometry clipping rule."""
    a, b = np.asarray(first), np.asarray(second)
    if (np.ma.isMaskedArray(first) or np.ma.isMaskedArray(second) or a.shape != b.shape
            or a.dtype.kind != "f" or b.dtype.kind != "f" or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Equal finite floating fields required")
    h = np.clip(.5 + .5 * (b - a) / SMOOTH_UNION_M, 0., 1.)
    return b * (1. - h) + a * h - SMOOTH_UNION_M * h * (1. - h)


def evaluate_field(points, rig=None):
    rig = author_rig() if rig is None else rig
    if type(rig) is not Rig: raise ValueError("Authored Rig required")
    p = _points(points)
    result = primitive_field(p, rig.primitives[0])
    for primitive in rig.primitives[1:]: result = smooth_union(result, primitive_field(p, primitive))
    return result


@dataclass(frozen=True)
class GridSpec:
    origin: tuple
    shape: tuple
    spacing_m: float = RESOLUTION_M

    def __post_init__(self):
        _triple(self.origin, "grid origin")
        if (type(self.shape) is not tuple or len(self.shape) != 3 or any(type(n) is not int or n < 3 for n in self.shape)
                or self.spacing_m != RESOLUTION_M):
            raise ValueError("Three-dimensional grid at sole 12 mm resolution required")


def manufacturing_grid(rig=None):
    rig = author_rig() if rig is None else rig
    if type(rig) is not Rig: raise ValueError("Authored Rig required")
    lower = np.min([np.minimum(p.start, p.end) - np.asarray(p.radii) for p in rig.primitives], axis=0)
    upper = np.max([np.maximum(p.start, p.end) + np.asarray(p.radii) for p in rig.primitives], axis=0)
    lo = np.floor((lower - PADDING_M) / RESOLUTION_M).astype(np.int64)
    hi = np.ceil((upper + PADDING_M) / RESOLUTION_M).astype(np.int64)
    return GridSpec(tuple(float(v) for v in lo * RESOLUTION_M), tuple(int(v) for v in hi - lo + 1))


def _deadline(deadline):
    if deadline is not None and (not math.isfinite(deadline) or time.monotonic() >= deadline):
        raise TimeoutError("Author-reference manufacture deadline exceeded")


@dataclass(frozen=True, eq=False)
class FieldGrid:
    spec: GridSpec
    values: np.ndarray

    def __post_init__(self):
        if type(self.spec) is not GridSpec: raise ValueError("GridSpec required")
        a = _array(self.values, np.float32, self.spec.shape, "grid values")
        boundary = (a[0], a[-1], a[:, 0], a[:, -1], a[:, :, 0], a[:, :, -1])
        if not np.any(a < 0) or any(np.any(face <= 0) for face in boundary):
            raise ValueError("Negative interior and strictly positive entire grid boundary required")
        object.__setattr__(self, "values", a)


def sample_field_grid(rig=None, *, deadline=None):
    """CPU manufacture in bounded slabs; no model/data/IO or alternate resolution."""
    rig = author_rig() if rig is None else rig
    spec = manufacturing_grid(rig)
    axes = [start + np.arange(n) * RESOLUTION_M for start, n in zip(spec.origin, spec.shape)]
    field = np.empty(spec.shape, np.float32)
    for first in range(0, spec.shape[0], 8):
        _deadline(deadline)
        points = np.stack(np.meshgrid(axes[0][first:first + 8], axes[1], axes[2], indexing="ij"), axis=-1)
        field[first:first + 8] = evaluate_field(points, rig).astype(np.float32)
    _deadline(deadline)
    return FieldGrid(spec, field)


def skin_weights(vertices, rig=None):
    """Frozen rest weights from authored primitive proximity; no fitted labels."""
    rig = author_rig() if rig is None else rig
    if type(rig) is not Rig: raise ValueError("Authored Rig required")
    v = _points(vertices)
    if v.ndim != 2: raise ValueError("Vertex matrix required")
    distances = np.stack([primitive_field(v, p) for p in rig.primitives], axis=1)
    relative = (distances - distances.min(axis=1, keepdims=True)) / WEIGHT_BLEND_M
    influence = np.exp(-relative * relative)
    weights = np.zeros((len(v), len(rig.names)), np.float64)
    for column, primitive in enumerate(rig.primitives): weights[:, rig.names.index(primitive.bone)] += influence[:, column]
    weights /= weights.sum(axis=1, keepdims=True)
    if not np.isfinite(weights).all(): raise ValueError("Finite normalized rest weights required")
    weights.setflags(write=False)
    return weights


@dataclass(frozen=True, eq=False)
class Reference:
    rest_vertices: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    rig: Rig

    def __post_init__(self):
        if type(self.rig) is not Rig: raise ValueError("Authored Rig required")
        v, f, w = map(np.asarray, (self.rest_vertices, self.faces, self.weights))
        if v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 4 or f.ndim != 2 or f.shape[1:] != (3,) or len(f) < 4:
            raise ValueError("Complete original vertex/triangle matrices required")
        v = _array(self.rest_vertices, np.float64, v.shape, "rest vertices")
        f = _array(self.faces, np.int64, f.shape, "faces")
        w = _array(self.weights, np.float64, (len(v), len(self.rig.names)), "weights")
        if np.any(f < 0) or np.any(f >= len(v)) or np.any(w < 0) or not np.allclose(w.sum(axis=1), 1., atol=1e-12, rtol=0):
            raise ValueError("Original valid indices and normalized nonnegative weights required")
        for name, value in (("rest_vertices", v), ("faces", f), ("weights", w)): object.__setattr__(self, name, value)


def extract_rest_surface(rig=None, *, deadline=None):
    """Azure-only caller: unmodified raw Lewiner boundary; NO post-mesh repairs.

    f<0 inside implies ascent/outward convention. ``allow_degenerate=False`` is
    the declared mesher option, not permission for any later face removal.
    Caller must validate closure, one component, conditioning and embedding.
    """
    rig = author_rig() if rig is None else rig
    grid = sample_field_grid(rig, deadline=deadline)
    _deadline(deadline)
    from skimage.measure import marching_cubes  # Lazy; never imported by host preflight.
    vertices, faces, _, _ = marching_cubes(grid.values.copy(), level=0., spacing=(RESOLUTION_M,) * 3,
        gradient_direction="ascent", step_size=1, allow_degenerate=False, method="lewiner", mask=None)
    _deadline(deadline)
    vertices = np.asarray(vertices, np.float64) + np.asarray(grid.spec.origin)
    faces = np.asarray(faces, np.int64)
    return Reference(vertices, faces, skin_weights(vertices, rig), rig)


def pose_joint_transforms(rig, pose_name):
    """Deterministic hierarchical forward kinematics, no inference or optimization."""
    if type(rig) is not Rig or pose_name not in POSE_NAMES: raise ValueError("Rig and one preregistered pose required")
    if pose_name == "rest": return np.broadcast_to(np.eye(3), (len(rig.names), 3, 3)).copy(), rig.rest_joints.copy()
    local = rig.local_rotations[POSE_NAMES.index(pose_name)]
    rotations = np.empty_like(local); joints = np.empty_like(rig.rest_joints)
    for bone, parent in enumerate(rig.parents):
        if parent == -1:
            rotations[bone] = local[bone]; joints[bone] = rig.rest_joints[bone]
        else:
            rotations[bone] = rotations[parent] @ local[bone]
            joints[bone] = joints[parent] + rotations[parent] @ (rig.rest_joints[bone] - rig.rest_joints[parent])
    return rotations, joints


def deform_reference(reference, pose_name):
    """Same full vertices/faces and frozen rest weights; NOT an embedding claim."""
    if type(reference) is not Reference: raise ValueError("Frozen authored Reference required")
    rotations, joints = pose_joint_transforms(reference.rig, pose_name)
    if pose_name == "rest": return reference.rest_vertices.copy(), reference.faces, joints
    vertices = np.zeros_like(reference.rest_vertices)
    for bone in range(len(reference.rig.names)):
        transformed = (reference.rest_vertices - reference.rig.rest_joints[bone]) @ rotations[bone].T + joints[bone]
        vertices += reference.weights[:, bone, None] * transformed
    if not np.isfinite(vertices).all(): raise ValueError("Finite full posed reference required")
    return vertices, reference.faces, joints
