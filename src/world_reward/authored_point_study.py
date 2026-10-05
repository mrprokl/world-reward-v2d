"""Pure manufactured component-study inputs, not video inference or physics.

Named204 controls use ACTUAL rig names/bounds. Native compact/PCA conversion,
decoder provenance and camera-basis conversion belong to the caller: this module
does not guess them. A pad follows one material hand triangle with a positive
gap; that is controlled proximity, NOT grasp/contact/force-closure certification.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import numpy as np

FRAMES = 24
SENTINELS = (0, 11, 23)
SCENES = (("dev_left", "development", "l"), ("dev_right", "development", "r"),
          ("reserved_left_slow", "reserved", "l"), ("reserved_right_slow", "reserved", "r"),
          ("reserved_left_reverse", "reserved", "l"), ("reserved_right_reverse", "reserved", "r"))
GAP_M = .0005
_FINGER = re.compile(r"([lr])_(thumb|index|middle|ring|pinky)[0-3]")


def _array(value, name, *, dtype=None, shape=None):
    a = np.asarray(value)
    if (np.ma.isMaskedArray(value) or (dtype is not None and a.dtype != dtype)
            or (dtype is None and a.dtype not in (np.dtype("float32"), np.dtype("float64")))
            or (shape is not None and a.shape != shape) or not np.isfinite(a).all()):
        raise ValueError(f"{name}: finite original dtype/shape required")
    return a


def _readonly(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _scene(index):
    if type(index) is not int or not 0 <= index < len(SCENES):
        raise ValueError("Fixed original scene index0..5 required")
    return SCENES[index]


@dataclass(frozen=True)
class NamedRecipe:
    scene_id: str
    split: str
    side: str
    parameters: np.ndarray
    identity: np.ndarray
    expression: np.ndarray
    frame_index: np.ndarray


def named204_recipe(names, bounds, scene_index, *, scale68, identity45):
    """Full24 poses; explicit external constant scale68/identity45, no clipping.

    Caller must derive scale68 from its actual decoder mean/PCA, not assume that
    zero28 equals zero68. Root204 entries0:6 remain zero; camera R/T are external.
    Unknown named/compact correspondence is never inferred here.
    """
    scene, split, side = _scene(scene_index)
    if (not isinstance(names, (list, tuple)) or len(names) != 249 or len(set(names)) != 249
            or any(type(n) is not str or not n for n in names)):
        raise ValueError("249 unique actual parameter names required")
    b = np.asarray(bounds)
    if (np.ma.isMaskedArray(bounds) or b.shape != (249, 2) or b.dtype.kind != "f"
            or np.isnan(b).any() or np.any(b[:, 0] > b[:, 1])):
        raise ValueError("Actual249 lower/upper limits required")
    scale = _array(scale68, "actual scale68", dtype=np.float32, shape=(68,))
    identity = _array(identity45, "identity45", dtype=np.float32, shape=(45,))
    s = np.linspace(0., 1., FRAMES)
    wave = np.sin(np.pi * s) if scene_index < 4 else -np.sin(np.pi * s)
    speed = 1. if scene_index < 2 else .75
    q = np.zeros((FRAMES, 204), np.float32)
    controls = (("uparm_ry", .12, .08), ("elbow_bend", .35, .10),
                ("thumb2_rz", .10, .06), ("index1_rz", .20, .10),
                ("index2_rz", .12, .05), ("middle1_rz", .18, -.07))
    for suffix, base, amplitude in controls:
        name = side + "_" + suffix
        if name not in names[:136]:
            raise ValueError("Actual named pose control absent: " + name)
        column = names.index(name)
        if column < 6:
            raise ValueError("Named articulation must not replace global root controls")
        q[:, column] = base + amplitude * speed * wave
    q[:, 136:] = scale
    full = np.column_stack((q, np.broadcast_to(identity, (FRAMES, 45))))
    if np.any(full < b[:, 0]) or np.any(full > b[:, 1]):
        raise ValueError("Frozen authored controls exceed actual limits; no sign rescue or clip")
    return NamedRecipe(scene, split, side, _readonly(q),
                       _readonly(np.broadcast_to(identity, (FRAMES, 45))),
                       _readonly(np.zeros((FRAMES, 72), np.float32)),
                       _readonly(np.arange(FRAMES, dtype=np.int64)))


@dataclass(frozen=True)
class PadMesh:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    contact_point: np.ndarray


def pad_mesh(scene_index):
    """Shared indexed8x8 box surfaces; canonical RGB never follows animation.

    x/y centered, z in[0,depth]; contact_point is the bottom grid vertex(4,4,0).
    Original integer grid keys generate shared seams directly, not posthoc weld.
    """
    _scene(scene_index)
    dims = np.asarray((.080, .060, .022)) * (1. + .025 * scene_index)
    vertices = []; faces = []; lookup = {}; cells = 8
    def vertex(key):
        key = tuple(key)
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append([(key[0] / cells - .5) * dims[0],
                             (key[1] / cells - .5) * dims[1], key[2] / cells * dims[2]])
        return lookup[key]
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for side in (0, cells):
            desired = np.zeros(3); desired[axis] = -1 if side == 0 else 1
            for i in range(cells):
                for j in range(cells):
                    keys = []
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        k = [0, 0, 0]; k[axis] = side; k[u] = i + du; k[v] = j + dv
                        keys.append(vertex(k))
                    a, b, c, d = keys
                    if np.cross(np.subtract(vertices[b], vertices[a]), np.subtract(vertices[c], vertices[a])) @ desired < 0:
                        faces.extend(((a, c, b), (a, d, c)))
                    else:
                        faces.extend(((a, b, c), (a, c, d)))
    xyz = np.asarray(vertices, np.float32)
    phase = .37 * scene_index
    x, y, z = xyz.astype(np.float64).T
    colors = np.column_stack((.45 + .28 * np.sin(620*x + 270*y + phase),
                              .47 + .26 * np.sin(510*y - 290*z + phase),
                              .43 + .25 * np.sin(570*z + 330*x + phase))).astype(np.float32)
    return PadMesh(*map(_readonly, (xyz, np.asarray(faces, np.int64), colors, np.zeros(3, np.float64))))


def select_hand_triangle(faces, hand_vertex_indices, lbs_indices, lbs_weights, joint_names, side):
    """All3 vertices in source-bound hand-spec; greatest named wrist LBS mass.

    Ties use original face ID. This picks a manufacturing anchor, not a grasp or
    observed interaction, and does not substitute an anatomical vertex ordinal.
    """
    f = _array(faces, "faces", dtype=np.int64)
    hand = _array(hand_vertex_indices, "hand spec", dtype=np.int64)
    idx = _array(lbs_indices, "LBS indices", dtype=np.int64)
    w = _array(lbs_weights, "LBS weights")
    if (side not in ("l", "r") or not isinstance(joint_names, (list, tuple))
            or len(set(joint_names)) != len(joint_names) or side + "_wrist" not in joint_names
            or f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or idx.ndim != 2
            or idx.shape != w.shape or not len(idx) or hand.ndim != 1 or not len(hand)
            or len(np.unique(hand)) != len(hand) or np.any(hand < 0) or np.any(hand >= len(idx))
            or np.any(f < 0) or np.any(f >= len(idx)) or np.any(idx < 0)
            or np.any(idx >= len(joint_names)) or np.any(w < 0)
            or not np.allclose(w.sum(1), 1., atol=1e-5, rtol=0)):
        raise ValueError("Actual hand-spec/topology/named wrist/normalized LBS required")
    wrist = joint_names.index(side + "_wrist")
    mass = np.where(idx == wrist, w, 0.).sum(1)
    candidates = np.flatnonzero(np.isin(f, hand).all(1))
    if not len(candidates):
        raise ValueError("No full original hand-spec triangle")
    score = mass[f[candidates]].sum(1)
    if score.max() <= 0:
        raise ValueError("No named wrist-supported material triangle")
    return int(candidates[np.argmax(score)])


@dataclass(frozen=True)
class PadAttachment:
    rotation: np.ndarray
    translation: np.ndarray
    surface_point: np.ndarray
    surface_normal: np.ndarray


def attach_pad(vertices_camera_m, faces, face_index):
    v = _array(vertices_camera_m, "actual decoded metres")
    f = _array(faces, "actual original faces", dtype=np.int64)
    if (v.ndim != 3 or v.shape[0] != FRAMES or v.shape[2] != 3 or f.ndim != 2 or f.shape[1:] != (3,)
            or np.any(f < 0) or np.any(f >= v.shape[1]) or type(face_index) is not int or not 0 <= face_index < len(f)):
        raise ValueError("Full24 original material triangles required")
    tri = v[:, f[face_index]].astype(np.float64)
    tangent = tri[:, 1] - tri[:, 0]
    normal = np.cross(tangent, tri[:, 2] - tri[:, 0])
    lengths = np.stack((np.linalg.norm(tangent, axis=1), np.linalg.norm(normal, axis=1)))
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Material frame collapsed; no repaired frame")
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    rotation = np.stack((tangent, np.cross(normal, tangent), normal), axis=-1)
    center = tri.mean(1)
    return PadAttachment(*map(_readonly, (rotation, center + GAP_M * normal, center, normal)))


def articulation_metrics(joint_positions_m, joint_rotations, joint_names, side):
    """Both arrays MUST share one basis; caller handles raw-JIT rotation flip.

    Wrist-relative finger displacement and relative rotations exclude rigid
    whole-body motion. Wrist rotation separately observes arm motion; no guessed
    elbow-joint ordinal. This measures supplied poses, not decoder authenticity.
    """
    if (side not in ("l", "r") or not isinstance(joint_names, (tuple, list))
            or len(set(joint_names)) != len(joint_names) or side + "_wrist" not in joint_names):
        raise ValueError("Unique actual named wrist/fingers required")
    p = _array(joint_positions_m, "joint metres", shape=(FRAMES, len(joint_names), 3))
    r = _array(joint_rotations, "joint rotations", shape=(FRAMES, len(joint_names), 3, 3))
    if (not np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-5, rtol=0)):
        raise ValueError("Proper supplied rotations required; no projection")
    wrist = joint_names.index(side + "_wrist")
    fingers = [j for j, name in enumerate(joint_names) if (m := _FINGER.fullmatch(name)) and m[1] == side]
    if len(fingers) < 2:
        raise ValueError("At least two actual named finger joints required")
    wr = r[:, wrist]
    local = np.einsum("tfi,tik->tfk", p[:, fingers] - p[:, wrist, None], wr)
    relrot = wr[:, None].swapaxes(-1, -2) @ r[:, fingers]
    def excursion(rot):
        d = rot[0].swapaxes(-1, -2) @ rot
        return float(np.arccos(np.clip((np.trace(d, axis1=-2, axis2=-1) - 1.) / 2., -1., 1.)).max())
    distance = float(np.linalg.norm(local - local[0], axis=-1).max())
    finger_angle, arm_angle = excursion(relrot), excursion(wr)
    return dict(finger_local_displacement_m=distance, finger_relative_rotation_rad=finger_angle,
                wrist_rotation_excursion_rad=arm_angle,
                passed=distance > .0001 and finger_angle > .005 and arm_angle > .005,
                decoder_or_contact_certified=False)


def attachment_proximity(attachment, pad):
    """Algebraic attachment consistency, NOT independent collision evidence."""
    if type(attachment) is not PadAttachment or type(pad) is not PadMesh:
        raise ValueError("Original typed material anchor and pad required")
    _array(pad.contact_point, "pad contact point", dtype=np.float64, shape=(3,))
    for name, shape in (("rotation", (FRAMES, 3, 3)), ("translation", (FRAMES, 3)),
                        ("surface_point", (FRAMES, 3)), ("surface_normal", (FRAMES, 3))):
        _array(getattr(attachment, name), name, dtype=np.float64, shape=shape)
    point = pad.contact_point @ attachment.rotation.swapaxes(-1, -2) + attachment.translation
    delta = point - attachment.surface_point
    signed = np.sum(delta * attachment.surface_normal, axis=-1)
    distance = np.linalg.norm(delta, axis=-1)
    opposition = np.sum(-attachment.rotation[:, :, 2] * attachment.surface_normal, axis=-1)
    if not np.isfinite(distance).all() or not np.isfinite(signed).all() or not np.isfinite(opposition).all():
        raise ValueError("Finite material-pair arithmetic required")
    return dict(max_material_gap_m=float(distance.max()), min_signed_gap_m=float(signed.min()),
                max_signed_gap_m=float(signed.max()), max_normal_dot=float(opposition.max()),
                passed=bool(np.all((signed > 0) & (distance <= .002)) and np.all(opposition <= -.99)),
                whole_surface_collision_checked=False, touching_certified=False, force_closure_verified=False)


def sentinel_texture_metrics(rgb, object_mask):
    images = _array(rgb, "three sentinel RGB", dtype=np.uint8)
    mask = _array(object_mask, "three sentinel masks", dtype=np.bool_)
    if images.shape != (3, 480, 640, 3) or mask.shape != (3, 480, 640):
        raise ValueError("Original sentinels0,11,23 on480x640 grid required")
    counts = mask.reshape(3, -1).sum(1); eigenvalues = []; edges = []
    for image, valid in zip(images, mask):
        common = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:]
        dx = (image[:-1, 1:].astype(np.float64) - image[:-1, :-1])[common]
        dy = (image[1:, :-1].astype(np.float64) - image[:-1, :-1])[common]
        edges.append(int(common.sum()))
        gram = np.array([[np.sum(dx*dx), np.sum(dx*dy)], [np.sum(dx*dy), np.sum(dy*dy)]])
        eigenvalues.append(float(np.linalg.eigvalsh(gram / max(1, common.sum())).min()))
    return dict(visible_pixels=counts.tolist(), gradient_pixels=edges, minimum_gradient_eigenvalue=eigenvalues,
                passed=bool(np.all(counts >= 64) and min(edges) >= 16 and min(eigenvalues) > 1e-6),
                tracking_accuracy_or_photorealism_verified=False)


def translation_observability(camera_points_m, K):
    points = _array(camera_points_m, "camera attachment points")
    k = _array(K, "original camera K", shape=(3, 3))
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or np.any(points[:, 2] <= 0)
            or k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0
            or not np.array_equal(k[2], [0, 0, 1])):
        raise ValueError("Positive-Z actual witnesses/OpenCV camera required")
    x, y, z = points.T
    j = np.zeros((len(points), 2, 3)); j[:, 0, 0] = k[0, 0]/z; j[:, 1, 1] = k[1, 1]/z
    j[:, 0, 2] = -k[0, 0]*x/z**2; j[:, 1, 2] = -k[1, 1]*y/z**2
    values = np.linalg.svd(j.reshape(-1, 3), compute_uv=False)
    rank = int(np.linalg.matrix_rank(j.reshape(-1, 3)))
    return dict(rank=rank, singular_values=values.tolist(), passed=rank == 3,
                statistical_uncertainty_or_accuracy_verified=False)


def translation_perturbation(scene_index):
    """Future SAME initial perturbation in both arms; exact initializer t0."""
    _scene(scene_index)
    s = np.linspace(0., 1., FRAMES); bump = np.sin(np.pi*s)
    sign = 1. if scene_index % 2 == 0 else -1.
    delta = np.column_stack((sign*.02*bump, .01*bump*np.sin(2*np.pi*s), .02*bump)).astype(np.float32)
    delta[0] = 0.
    return _readonly(delta)
