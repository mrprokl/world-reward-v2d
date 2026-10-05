"""Manufactured arithmetic only; no decoder/PCA/render/tracker/native qualification."""
from dataclasses import FrozenInstanceError
import hashlib

import numpy as np
import pytest

from world_reward import articulated_point_cohort as c
from world_reward import authored_point_study as old


def metadata():
    names = [f"parameter_{i}" for i in range(249)]
    for side, offset in (("l", 10), ("r", 30)):
        for i, (suffix, *_) in enumerate(c.CONTROLS):
            names[offset + i] = side + "_" + suffix
    return names, np.tile([-2., 2.], (249, 1))


def recipe(index=0, *, names=None, bounds=None, scale=None, identity=None):
    n, b = metadata()
    return c.named_recipe(n if names is None else names, b if bounds is None else bounds, index,
        scale68=np.linspace(-.1, .1, 68, dtype=np.float32) if scale is None else scale,
        identity45=np.zeros(45, np.float32) if identity is None else identity)


def original249(r):
    return np.column_stack((r.parameters, r.identity))


def reference_curves(index):
    knots = np.asarray(c.KNOT_VALUES[index], np.float64)
    rows = []
    for frame in range(24):
        if frame in c.KNOTS:
            rows.append(knots[:, c.KNOTS.index(frame)])
            continue
        j = next(j for j in range(4) if c.KNOTS[j] < frame < c.KNOTS[j+1])
        u = (frame - c.KNOTS[j]) / (c.KNOTS[j+1] - c.KNOTS[j])
        w = 3 * u * u - 2 * u * u * u
        rows.append(knots[:, j] + (knots[:, j+1] - knots[:, j]) * w)
    a, b = np.asarray(rows).T
    return a, b, (a + b) / 2


def test_literal_new_cohort_is_frozen_and_disjoint_from_closed_ids():
    assert c.FRAMES == 24 and c.KNOTS == (0, 5, 11, 17, 23)
    assert c.SCENES == (("knot_front", "development", "l"), ("knot_cross", "development", "r"),
        ("knot_double", "reserved", "l"), ("knot_return", "reserved", "r"),
        ("knot_delayed", "reserved", "l"), ("knot_alternate", "reserved", "r"))
    assert not ({s[0] for s in c.SCENES} & {s[0] for s in old.SCENES})
    assert c.KNOT_VALUES == (
        ((0., 1., .25, .75, .2), (.2, .5, 1., .25, .6)),
        ((.2, .75, 1., .25, .5), (.8, .2, .5, 1., .1)),
        ((.1, .9, .2, .8, .3), (.7, .3, .9, .1, .6)),
        ((.8, .2, .7, .1, .5), (.1, .8, .3, .9, .2)),
        ((.2, .3, .9, .6, .1), (.9, .6, .2, .4, .8)),
        ((.6, .1, .8, .3, .9), (.3, .9, .1, .7, .4)))
    assert c.CONTROLS == (("uparm_ry", .10, .04, 0), ("elbow_bend", .40, .06, 1),
        ("thumb2_rz", .08, .04, 2), ("index1_rz", .16, .06, 2),
        ("index2_rz", .10, .05, 1), ("middle1_rz", .14, .05, 0))


@pytest.mark.parametrize("index", range(6))
def test_full24_controls_perturbation_and_original_slots(index):
    r = recipe(index); names, bounds = metadata(); curves = reference_curves(index)
    assert isinstance(r, old.NamedRecipe)
    assert (r.scene_id, r.split, r.side) == c.SCENES[index]
    assert r.parameters.dtype == r.perturbation.dtype == np.float32
    assert r.parameters.shape == (24, 204) and r.perturbation.shape == (24, 3)
    assert np.array_equal(r.frame_index, np.arange(24, dtype=np.int64))
    assert not np.any(r.parameters[:, :6])
    for suffix, base, amplitude, curve in c.CONTROLS:
        expected = (base + amplitude * curves[curve]).astype(np.float32)
        assert np.array_equal(r.parameters[:, names.index(r.side + "_" + suffix)], expected)
    a, b, cv = curves; sign = -1 if r.side == "l" else 1
    delta = np.column_stack((sign * .018 * (a-a[0]), .012 * (b-b[0]), .016 * (cv-cv[0]))).astype(np.float32)
    delta[0] = 0.
    assert np.array_equal(r.perturbation, delta) and np.all(r.perturbation[0] == 0)
    assert r.perturbation[0].tobytes() == np.zeros(3, np.float32).tobytes()
    assert np.array_equal(r.parameters[:, 136:], np.broadcast_to(r.parameters[0, 136:], (24, 68)))
    assert not np.any(r.identity) and not np.any(r.expression)
    assert r.scale_violations == () and c.validate_reconstituted_controls(original249(r), bounds) == ()
    for array in (r.parameters, r.identity, r.expression, r.frame_index, r.perturbation):
        assert not array.flags.writeable
        with pytest.raises(ValueError): array.flags.writeable = True


def test_input_copies_no_mutation_and_same_old_global_geometry_policy():
    names, bounds = metadata(); scale = np.ones(68, np.float32); identity = np.full(45, .1, np.float32)
    before = tuple(hashlib.sha256(x.tobytes()).hexdigest() for x in (bounds, scale, identity))
    r = recipe(names=names, bounds=bounds, scale=scale, identity=identity)
    assert before == tuple(hashlib.sha256(x.tobytes()).hexdigest() for x in (bounds, scale, identity))
    scale[:] = 0; identity[:] = 0; bounds[:] = 0
    assert np.all(r.parameters[:, 136:] == 1) and np.all(r.identity == np.float32(.1))
    with pytest.raises(FrozenInstanceError): r.side = "r"
    # Consumers reuse old geometry functions, not six modified helper instances.
    assert isinstance(old.pad_mesh(0), old.PadMesh)


def test_named_column_permutation_preserves_the_original_pose_abi():
    names, bounds = metadata(); p = np.arange(249); p[6:136] = p[6:136][::-1]
    a = recipe(); b = recipe(names=[names[i] for i in p], bounds=bounds[p])
    assert np.array_equal(b.parameters, a.parameters[:, p[:204]])
    assert np.array_equal(a.perturbation, b.perturbation)


def test_soft_scale_excess_retains_original_values_and_old_policy_still_fails():
    names, bounds = metadata(); bounds[136:204] = 0; scale = np.zeros(68, np.float32)
    scale[11] = np.float32(.0125); scale[12] = np.float32(-.01)
    r = recipe(bounds=bounds, scale=scale)
    assert [v.parameter_index for v in r.scale_violations] == [147, 148]
    assert [(v.actual, v.lower, v.upper) for v in r.scale_violations] == [
        (float(scale[11]), 0., 0.), (float(scale[12]), 0., 0.)]
    assert np.array_equal(r.parameters[:, 136:], np.broadcast_to(scale, (24, 68)))
    assert c.validate_reconstituted_controls(original249(r), bounds) == r.scale_violations
    with pytest.raises(ValueError, match="exceed actual limits"):
        old.named204_recipe(names, bounds, 0, scale68=scale, identity45=np.zeros(45, np.float32))


@pytest.mark.parametrize("column", [0, 10, 67, 122, 135, 204, 248])
def test_post_pca_hard_limits_fail_without_any_soft_articulation_waiver(column):
    r = recipe(); x = original249(r); _, bounds = metadata(); x[:, column] = 3
    with pytest.raises(ValueError, match="Hard articulation"):
        c.validate_reconstituted_controls(x, bounds)


@pytest.mark.parametrize("column", [136, 203, 204, 248])
def test_changing_clipconstant_scale_or_identity_rejected(column):
    x = original249(recipe()); _, bounds = metadata(); x[23, column] += np.float32(.1)
    with pytest.raises(ValueError, match="Clip-constant"):
        c.validate_reconstituted_controls(x, bounds)


@pytest.mark.parametrize("bad", ["bool_index", "extra_index", "missing", "duplicate", "root",
    "scale_name", "nan_bounds", "unordered_bounds", "object_bounds", "bad_bounds_shape", "masked_bounds",
    "bad_scale_dtype", "nan_scale", "bad_identity_shape", "masked_identity", "angle_bound", "identity_bound"])
def test_bad_metadata_or_hard_recipe_contract_fails(bad):
    names, bounds = metadata(); index = 0; scale = np.zeros(68, np.float32); identity = np.zeros(45, np.float32)
    if bad == "bool_index": index = True
    elif bad == "extra_index": index = 6
    elif bad == "missing": names[10] = "unknown"
    elif bad == "duplicate": names[0] = names[1]
    elif bad == "root": names[0], names[10] = names[10], names[0]
    elif bad == "scale_name": names[136], names[10] = names[10], names[136]
    elif bad == "nan_bounds": bounds[0, 0] = np.nan
    elif bad == "unordered_bounds": bounds[3] = [2., 1.]
    elif bad == "object_bounds": bounds = bounds.astype(object)
    elif bad == "bad_bounds_shape": bounds = bounds[:-1]
    elif bad == "masked_bounds": bounds = np.ma.array(bounds, mask=False)
    elif bad == "bad_scale_dtype": scale = scale.astype(np.float64)
    elif bad == "nan_scale": scale[0] = np.nan
    elif bad == "bad_identity_shape": identity = identity[:-1]
    elif bad == "masked_identity": identity = np.ma.array(identity, mask=False)
    elif bad == "angle_bound": bounds[10] = [0., .01]
    elif bad == "identity_bound": bounds[204] = [.1, .2]
    with pytest.raises(ValueError): c.named_recipe(names, bounds, index, scale68=scale, identity45=identity)


@pytest.mark.parametrize("bad", ["dtype", "shape", "nan", "masked"])
def test_reconstituted_contract_requires_full_original_finite_float32(bad):
    x = original249(recipe()); _, bounds = metadata()
    if bad == "dtype": x = x.astype(np.float64)
    elif bad == "shape": x = x[:-1]
    elif bad == "nan": x[4, 148] = np.nan
    elif bad == "masked": x = np.ma.array(x, mask=False)
    with pytest.raises(ValueError): c.validate_reconstituted_controls(x, bounds)


def test_open_limits_are_not_nan_and_do_not_constrain_finite_controls():
    _, bounds = metadata(); bounds[:] = [-np.inf, np.inf]
    assert recipe(bounds=bounds).scale_violations == ()
