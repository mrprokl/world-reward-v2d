"""Tiny own fields/rig contracts only: never manufacture the full reference locally."""
import sys
import time
import types

import numpy as np
import pytest

from world_reward import author_human_field as a


def tiny_reference():
    rig = a.author_rig()
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    weights = np.zeros((4, len(rig.names))); weights[:, 0] = 1.
    return a.Reference(vertices, faces, weights, rig)


def test_fixed_protocol_anatomical_tree_proportions_and_ten_digits():
    r = a.author_rig()
    assert a.RESOLUTION_M == .012 and a.POSE_NAMES == ("rest", "carry", "fingerbend")
    assert len(r.names) == 61 and r.parents[0] == -1
    assert set(a.FINGERTIP_NAMES) <= set(r.names) and len(a.FINGERTIP_NAMES) == 10
    assert np.ptp(r.rest_joints[:, 1]) > 1.5
    assert not r.parents.flags.writeable and not r.rest_joints.flags.writeable
    for side in ("l", "r"):
        for finger in a.FINGER_NAMES:
            chain = [r.names.index(f"{side}_{finger}_{p}") for p in ("mcp", "pip", "dip", "tip")]
            assert all(r.parents[next_] == previous for previous, next_ in zip(chain, chain[1:]))
            assert all(any(p.bone == r.names[b] for p in r.primitives) for b in chain[:-1])
    rr = a.author_rig()
    assert r.primitives == rr.primitives
    assert r.rest_joints.tobytes() == rr.rest_joints.tobytes()
    assert r.local_rotations.tobytes() == rr.local_rotations.tobytes()


def test_capsule_and_ellipsoid_signs_and_real_boundaries():
    capsule = a.Primitive("own", "capsule", "bone", (0., 0., 0.), (0., 1., 0.), (.1,) * 3)
    x = np.array([[0., .5, 0.], [.1, .5, 0.], [.2, .5, 0.], [0., -1., 0.]])
    assert np.allclose(a.primitive_field(x, capsule), [-.1, 0., .1, .9])
    ellipsoid = a.Primitive("own", "ellipsoid", "bone", (0., 0., 0.), (0., 0., 0.), (.1, .2, .3))
    assert np.allclose(a.primitive_field(np.array([[0., 0., 0.], [.1, 0., 0.], [0., .2, 0.]]), ellipsoid), [-.1, 0., 0.])


def test_smooth_union_not_concatenation_and_only_fixed_width():
    x, y = np.array([-.1, .1, 0.]), np.array([.1, -.1, 0.])
    assert np.array_equal(a.smooth_union(x, y), a.smooth_union(y, x))
    assert np.array_equal(a.smooth_union(x, y)[:2], [-.1, -.1])
    assert a.smooth_union(x, y)[2] == -a.SMOOTH_UNION_M / 4


def test_ten_fingertip_centers_inside_and_distal_gaps_are_authored():
    r = a.author_rig()
    tips = r.rest_joints[[r.names.index(n) for n in a.FINGERTIP_NAMES]]
    assert (a.evaluate_field(tips, r) < 0).all()
    for side in ("l", "r"):
        first = r.rest_joints[r.names.index(side + "_index_tip")]
        second = r.rest_joints[r.names.index(side + "_middle_tip")]
        assert a.evaluate_field(((first + second) / 2)[None], r)[0] > 0
    assert a.evaluate_field(np.array([[0., 2., 0.], [2., 1., 0.]]), r).min() > 0


def test_grid_extent_fixed_padded_without_sampling_full_field():
    rig = a.author_rig(); spec = a.manufacturing_grid(rig)
    assert np.prod(spec.shape) < 2_000_000 and spec.spacing_m == .012
    lower = np.min([np.minimum(p.start, p.end) - p.radii for p in rig.primitives], axis=0)
    upper = np.max([np.maximum(p.start, p.end) + p.radii for p in rig.primitives], axis=0)
    assert np.all(np.asarray(spec.origin) <= lower - a.PADDING_M)
    assert np.all(np.asarray(spec.origin) + (np.asarray(spec.shape) - 1) * .012 >= upper + a.PADDING_M)
    assert a.manufacturing_grid(rig) == spec
    with pytest.raises(ValueError): a.GridSpec((0., 0., 0.), (3, 3, 3), .006)


def test_skin_weights_frozen_normalized_and_named_distal_contribution():
    r = a.author_rig()
    points = r.rest_joints[[r.names.index(n) for n in a.FINGERTIP_NAMES]]
    weights = a.skin_weights(points, r)
    assert weights.dtype == np.float64 and not weights.flags.writeable
    assert np.all(weights >= 0) and np.allclose(weights.sum(1), 1., atol=1e-12, rtol=0)
    for row, name in enumerate(a.FINGERTIP_NAMES):
        assert weights[row, r.names.index(name.replace("_tip", "_dip"))] > .1


@pytest.mark.parametrize("pose", a.POSE_NAMES)
def test_forward_preserves_all_named_bone_lengths_and_is_repeatable(pose):
    r = a.author_rig(); rotations, joints = a.pose_joint_transforms(r, pose)
    children = np.arange(1, len(r.names))
    expected = np.linalg.norm(r.rest_joints[children] - r.rest_joints[r.parents[children]], axis=1)
    actual = np.linalg.norm(joints[children] - joints[r.parents[children]], axis=1)
    assert np.allclose(expected, actual, atol=1e-14, rtol=0)
    assert np.allclose(np.linalg.det(rotations), 1., atol=1e-12, rtol=0)
    assert joints.tobytes() == a.pose_joint_transforms(r, pose)[1].tobytes()
    if pose != "rest": assert np.linalg.norm(joints - r.rest_joints) > .5


def test_exact_rest_replay_and_every_pose_full_face_vertex_conservation():
    ref = tiny_reference(); original = (ref.rest_vertices.tobytes(), ref.faces.tobytes(), ref.weights.tobytes())
    for pose in a.POSE_NAMES:
        v, f, j = a.deform_reference(ref, pose)
        assert v.shape == ref.rest_vertices.shape and f is ref.faces and f.tobytes() == original[1]
        assert j.shape == ref.rig.rest_joints.shape and np.isfinite(v).all()
        if pose == "rest":
            assert v.tobytes() == original[0] and j.tobytes() == ref.rig.rest_joints.tobytes()
    assert original == (ref.rest_vertices.tobytes(), ref.faces.tobytes(), ref.weights.tobytes())
    with pytest.raises(ValueError): a.deform_reference(ref, "retuned")


@pytest.mark.parametrize("bad", ["empty", "boundary", "nan", "dtype"])
def test_strict_exterior_field_grid_fails_closed(bad):
    spec = a.GridSpec((0., 0., 0.), (3, 3, 3)); field = np.ones(spec.shape, np.float32); field[1, 1, 1] = -1
    if bad == "empty": field[:] = 1
    elif bad == "boundary": field[0, 1, 1] = 0
    elif bad == "nan": field[1, 1, 1] = np.nan
    else: field = field.astype(np.float64)
    with pytest.raises(ValueError): a.FieldGrid(spec, field)


def test_raw_mesher_contract_lazy_import_no_repair_and_exact_faces(monkeypatch):
    spec = a.GridSpec((.1, .2, .3), (3, 3, 3)); field = np.ones(spec.shape, np.float32); field[1, 1, 1] = -1
    grid = a.FieldGrid(spec, field)
    monkeypatch.setattr(a, "sample_field_grid", lambda rig, deadline: grid)
    source_vertices = np.array([[0., 0., 0.], [.012, 0., 0.], [0., .012, 0.], [0., 0., .012]], np.float32)
    source_faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    called = []
    def mesher(values, **kwargs):
        called.append(kwargs)
        assert np.array_equal(values, field)
        return source_vertices, source_faces, None, None
    monkeypatch.setitem(sys.modules, "skimage.measure", types.SimpleNamespace(marching_cubes=mesher))
    ref = a.extract_rest_surface()
    assert called == [dict(level=0., spacing=(.012,) * 3, gradient_direction="ascent", step_size=1,
        allow_degenerate=False, method="lewiner", mask=None)]
    assert ref.faces.tobytes() == source_faces.astype(np.int64).tobytes()
    assert np.array_equal(ref.rest_vertices, source_vertices.astype(np.float64) + spec.origin)
    with pytest.raises(TimeoutError): a.extract_rest_surface(deadline=time.monotonic() - 1)


@pytest.mark.parametrize("value", [np.array([[np.nan, 0., 0.]]), np.array([[1, 2, 3]]), np.zeros((2, 2)), np.ma.array([[0., 0., 0.]])])
def test_invalid_points_never_enter_field_or_weights(value):
    with pytest.raises(ValueError): a.evaluate_field(value)
    with pytest.raises(ValueError): a.skin_weights(value)


def test_invalid_primitive_and_bad_rig_rejected():
    with pytest.raises(ValueError): a.Primitive("x", "capsule", "bone", (0., 0., 0.), (0., 0., 0.), (.1,) * 3)
    with pytest.raises(ValueError): a.Primitive("x", "ellipsoid", "bone", (0., 0., 0.), (0., 0., 0.), (-.1, .1, .1))
    r = a.author_rig(); parents = r.parents.copy(); parents[1] = 1
    with pytest.raises(ValueError): a.Rig(r.names, parents, r.rest_joints, r.primitives, r.local_rotations)
    q = r.local_rotations.copy(); q[1, 0, 0, 0] = 2
    with pytest.raises(ValueError): a.Rig(r.names, r.parents, r.rest_joints, r.primitives, q)
    with pytest.raises(ValueError): a.smooth_union(np.array([np.nan]), np.array([0.]))
    with pytest.raises(ValueError): a.pose_joint_transforms(r, "unknown")
