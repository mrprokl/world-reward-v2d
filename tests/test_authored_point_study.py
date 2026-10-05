"""Tiny manufactured arithmetic only; no native model, render, labels or GPU."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from world_reward import authored_point_study as s

ROOT = Path(__file__).resolve().parents[1]


def metadata():
    names = [f"parameter_{i}" for i in range(249)]
    for side, offset in (("l", 10), ("r", 30)):
        for i, suffix in enumerate(("uparm_ry", "elbow_bend", "thumb2_rz", "index1_rz", "index2_rz", "middle1_rz")):
            names[offset+i] = side + "_" + suffix
    bounds = np.tile([-2., 2.], (249, 1))
    return names, bounds


def recipe(index=0, names=None, bounds=None):
    n, b = metadata()
    return s.named204_recipe(n if names is None else names, b if bounds is None else bounds, index,
                             scale68=np.linspace(-.1, .1, 68, dtype=np.float32), identity45=np.zeros(45, np.float32))


def test_protocol_matches_pure_constants_and_original_source_pin():
    p = json.loads((ROOT/'configs/authored_point_study_protocol_v1.json').read_text())
    raw = (ROOT/p['base_protocol']['path']).read_bytes()
    assert p['base_protocol']['bytes'] == len(raw)
    assert p['base_protocol']['sha256'] == hashlib.sha256(raw).hexdigest()
    assert p['cohort']['scenes'] == [x[0] for x in s.SCENES]
    assert p['cohort']['splits'] == [x[1] for x in s.SCENES]
    assert p['cohort']['sentinels'] == list(s.SENTINELS)
    assert p['cohort']['frames_each'] == s.FRAMES
    assert p['pad']['positive_material_gap_m'] == s.GAP_M
    assert p['claims']['tracker_executed'] is False
    assert p['future_study']['reserved_native_fits'] == 8


def test_all_six_full_named_recipes_constant_geometry_and_original_indices():
    names, _ = metadata()
    for i in range(6):
        r = recipe(i)
        assert (r.scene_id, r.split, r.side) == s.SCENES[i]
        assert r.parameters.shape == (24, 204) and r.parameters.dtype == np.float32
        assert np.array_equal(r.frame_index, np.arange(24, dtype=np.int64))
        assert np.all(r.parameters[:, :6] == 0)
        assert np.array_equal(r.parameters[:, 136:], np.broadcast_to(r.parameters[0, 136:], (24, 68)))
        assert np.array_equal(r.identity, np.zeros((24, 45), np.float32))
        assert np.array_equal(r.expression, np.zeros((24, 72), np.float32))
        assert np.ptp(r.parameters[:, names.index(r.side+'_elbow_bend')]) > .05
        assert np.ptp(r.parameters[:, names.index(r.side+'_index1_rz')]) > .05
        assert all(not x.flags.writeable for x in (r.parameters, r.identity, r.expression, r.frame_index))


def test_parameter_name_permutation_not_guessed_compact_positions():
    names, bounds = metadata()
    # Permute only named pose columns; root/scale/identity ABI remains explicit.
    permutation = np.arange(249); permutation[6:136] = permutation[6:136][::-1]
    reordered = recipe(names=[names[j] for j in permutation], bounds=bounds[permutation])
    original = recipe()
    assert np.array_equal(reordered.parameters, original.parameters[:, permutation[:204]])


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'bounds', 'root', 'scale_dtype', 'index'])
def test_bad_named_metadata_fails_without_clipping_or_conversion(bad):
    names, bounds = metadata(); scale = np.zeros(68, np.float32); index = 0
    if bad == 'missing': names[names.index('l_elbow_bend')] = 'unknown'
    elif bad == 'duplicate': names[1] = names[0]
    elif bad == 'bounds': bounds[names.index('l_index1_rz'), 1] = .01
    elif bad == 'root': names[0], names[10] = names[10], names[0]
    elif bad == 'scale_dtype': scale = scale.astype(np.float64)
    elif bad == 'index': index = True
    with pytest.raises(ValueError):
        s.named204_recipe(names, bounds, index, scale68=scale, identity45=np.zeros(45, np.float32))


def test_recipe_has_owned_immutable_scale_not_input_alias():
    names, bounds = metadata(); scale = np.ones(68, np.float32)
    r = s.named204_recipe(names, bounds, 0, scale68=scale, identity45=np.zeros(45, np.float32))
    scale[:] = 0
    assert (r.parameters[:, 136:] == 1).all()
    with pytest.raises(ValueError): r.parameters.flags.writeable = True


def test_pad_dense_closed_outward_direct_indexed_and_material_fixed():
    for i in range(6):
        p = s.pad_mesh(i)
        assert p.vertices.shape == (386, 3) and p.faces.shape == (768, 3)
        assert p.vertices.dtype == p.colors.dtype == np.float32 and p.faces.dtype == np.int64
        assert len(np.unique(p.faces)) == len(p.vertices)
        edges = np.concatenate((p.faces[:, [0, 1]], p.faces[:, [1, 2]], p.faces[:, [2, 0]]))
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
        assert (counts == 2).all()
        signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
        assert (np.bincount(inverse, weights=signs) == 0).all()
        tri = p.vertices.astype(np.float64)[p.faces]
        normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        assert np.all(np.linalg.norm(normals, axis=1) > 0)
        center = (p.vertices.min(0)+p.vertices.max(0))/2
        assert (np.einsum('ij,ij->i', tri.mean(1)-center, normals) > 0).all()
        assert np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() > 0
        assert (np.ptp(p.colors, axis=0) > .4).all() and ((p.colors > 0) & (p.colors < 1)).all()
        assert np.array_equal(p.colors, s.pad_mesh(i).colors)
        assert all(not x.flags.writeable for x in vars(p).values())


def hand_fixture():
    f = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5]], np.int64)
    idx = np.tile([0, 1], (6, 1)).astype(np.int64)
    weights = np.array([[.7, .3], [.8, .2], [.8, .2], [.9, .1], [.1, .9], [.1, .9]])
    return f, np.arange(4, dtype=np.int64), idx, weights, ['l_wrist', 'l_index1']


def test_material_anchor_uses_hand_spec_and_named_lbs_not_vertex_ordinal():
    args = hand_fixture(); assert s.select_hand_triangle(*args, 'l') == 1
    permutation = np.array([5, 2, 0, 4, 1, 3]); inverse = np.argsort(permutation)
    f, hand, idx, w, names = args
    assert s.select_hand_triangle(inverse[f], inverse[hand], idx[permutation], w[permutation], names, 'l') == 1


def test_material_anchor_tie_is_original_face_order():
    f, hand, idx, w, names = hand_fixture(); w[:] = [.5, .5]
    assert s.select_hand_triangle(f, hand, idx, w, names, 'l') == 0


@pytest.mark.parametrize('bad', ['no_hand_face', 'no_wrist', 'no_mass', 'invalid_index', 'weights'])
def test_no_anchor_no_placeholder_or_nearest_fallback(bad):
    f, hand, idx, w, names = hand_fixture()
    if bad == 'no_hand_face': hand = np.array([0], np.int64)
    elif bad == 'no_wrist': names[0] = 'unknown'
    elif bad == 'no_mass': w[:] = [0., 1.]
    elif bad == 'invalid_index': f[0, 0] = 6
    elif bad == 'weights': w[0] = [.1, .2]
    with pytest.raises(ValueError): s.select_hand_triangle(f, hand, idx, w, names, 'l')


def geometry():
    theta = np.linspace(0, .1, 24); c, sn = np.cos(theta), np.sin(theta)
    r = np.zeros((24, 3, 3)); r[:, 0, 0] = c; r[:, 0, 1] = -sn; r[:, 1, 0] = sn; r[:, 1, 1] = c; r[:, 2, 2] = 1
    base = np.array([[0., 0, 4], [.03, 0, 4], [0, .03, 4]])
    v = base[None] @ r.swapaxes(-1, -2) + np.linspace(0, .04, 24)[:, None, None]*[1, 0, 0]
    return v, np.array([[0, 1, 2]], np.int64), r


def test_full_material_frame_barycentre_gap_rotation_and_proximity():
    v, f, r = geometry(); a = s.attach_pad(v, f, 0)
    assert np.allclose(a.rotation, r, atol=1e-14, rtol=0)
    assert np.array_equal(a.surface_point, v[:, f[0]].mean(1))
    assert np.allclose(a.translation-a.surface_point, a.surface_normal*s.GAP_M, atol=1e-15, rtol=0)
    p = s.attachment_proximity(a, s.pad_mesh(0))
    assert p['passed'] and not p['touching_certified'] and not p['whole_surface_collision_checked']
    assert all(not x.flags.writeable for x in vars(a).values())
    v[:] = 0; assert np.any(a.surface_point)


def test_collapsed_material_frame_fail_not_arbitrary_rotation():
    v, f, _ = geometry(); v[12, 1] = v[12, 0]
    with pytest.raises(ValueError, match='collapsed'): s.attach_pad(v, f, 0)


def test_material_frame_overflow_and_nonfinite_pair_fail_closed():
    v, f, _ = geometry(); v[0, 1, 0] = 1e300
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='collapsed'): s.attach_pad(v, f, 0)
    v, f, _ = geometry(); a = s.attach_pad(v, f, 0)
    bad = a.translation.copy(); bad[0, 0] = np.nan
    a = s.PadAttachment(a.rotation, bad, a.surface_point, a.surface_normal)
    with pytest.raises(ValueError, match='translation'): s.attachment_proximity(a, s.pad_mesh(0))


def articulated_fixture():
    theta = np.linspace(0, .1, 24); r = np.tile(np.eye(3), (24, 3, 1, 1))
    for i, t in enumerate(theta):
        c, sn = np.cos(t), np.sin(t)
        r[i, 0] = [[c, -sn, 0], [sn, c, 0], [0, 0, 1]]
        r[i, 1] = r[i, 0] @ np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
        r[i, 2] = r[i, 1]
    local = np.tile([[0, 0, 0], [.03, 0, 0], [.06, 0, 0]], (24, 1, 1)).astype(float)
    local[:, 2, 1] += theta*.02
    p = local @ r[:, 0].swapaxes(-1, -2) + [0, 0, 4]
    return p, r, ['l_wrist', 'l_index1', 'l_index2']


def test_articulation_real_local_finger_and_arm_motion():
    p, r, names = articulated_fixture(); result = s.articulation_metrics(p, r, names, 'l')
    assert result['passed'] and result['finger_local_displacement_m'] > .001
    assert not result['decoder_or_contact_certified']


def test_rigid_global_motion_not_finger_articulation():
    v, _, rotations = geometry()
    r = np.repeat(rotations[:, None], 3, axis=1)
    result = s.articulation_metrics(v, r, ['l_wrist', 'l_index1', 'l_index2'], 'l')
    assert not result['passed'] and result['finger_local_displacement_m'] < 1e-14


def test_articulation_basis_translation_invariance_and_invalid_rotations():
    p, r, names = articulated_fixture(); a = s.articulation_metrics(p, r, names, 'l')
    flip = np.diag([1., -1., -1.]); b = s.articulation_metrics(p@flip + [1, 2, 3], flip@r, names, 'l')
    for key in ('finger_local_displacement_m', 'finger_relative_rotation_rad', 'wrist_rotation_excursion_rad'):
        assert np.isclose(a[key], b[key], atol=1e-14, rtol=0)
    r[:, 0, 0, 0] = 2
    with pytest.raises(ValueError, match='Proper'): s.articulation_metrics(p, r, names, 'l')


def texture_fixture():
    rgb = np.zeros((3, 480, 640, 3), np.uint8); mask = np.zeros((3, 480, 640), bool)
    yy, xx = np.meshgrid(np.arange(20), np.arange(20), indexing='ij')
    rgb[:, 200:220, 300:320, 0] = (xx*9).astype(np.uint8)
    rgb[:, 200:220, 300:320, 1] = (yy*9).astype(np.uint8)
    mask[:, 200:220, 300:320] = True
    return rgb, mask


def test_sentinel_texture_two_directions_and_flat_absent_controls():
    rgb, mask = texture_fixture(); a = s.sentinel_texture_metrics(rgb, mask)
    assert a['passed'] and a['visible_pixels'] == [400]*3 and min(a['minimum_gradient_eigenvalue']) > 1
    assert not a['tracking_accuracy_or_photorealism_verified']
    rgb[..., 1] = 0
    assert not s.sentinel_texture_metrics(rgb, mask)['passed']
    mask[:] = False
    assert not s.sentinel_texture_metrics(rgb, mask)['passed']


def test_texture_dtype_and_original_grid_enforced():
    rgb, mask = texture_fixture()
    with pytest.raises(ValueError): s.sentinel_texture_metrics(rgb.astype(float), mask)
    with pytest.raises(ValueError): s.sentinel_texture_metrics(rgb[:, :64, :64], mask[:, :64, :64])


def test_translation_projection_jacobian_independent_finite_difference_reference():
    points = np.array([[-.03, -.04, 4], [.03, .04, 4], [.01, -.02, 4.01]])
    K = np.array([[800., 0, 320], [0, 800, 240], [0, 0, 1]])
    def project(p): return p[:, :2]/p[:, 2, None]*[800, 800]+[320, 240]
    eps = 1e-5
    j = np.stack([(project(points + np.eye(3)[i]*eps)-project(points-np.eye(3)[i]*eps))/(2*eps) for i in range(3)], axis=-1)
    a = s.translation_observability(points, K)
    assert a['rank'] == 3 and a['passed']
    assert np.allclose(a['singular_values'], np.linalg.svd(j.reshape(-1, 3), compute_uv=False), atol=1e-7, rtol=1e-8)
    assert not s.translation_observability(np.array([[0., 0, 4]]), K)['passed']
    with pytest.raises(ValueError): s.translation_observability(np.array([[0., 0, -4]]), K)


def test_future_perturbation_t0_exact_full_original_no_static_trajectory():
    for index in range(6):
        delta = s.translation_perturbation(index)
        assert delta.dtype == np.float32 and delta.shape == (24, 3)
        assert delta[0].tobytes() == np.zeros(3, np.float32).tobytes()
        assert np.linalg.norm(delta[1:], axis=1).max() > .02 and not delta.flags.writeable


def test_pure_source_no_model_or_file_or_execution_imports():
    import ast
    tree = ast.parse((ROOT/'src/world_reward/authored_point_study.py').read_text())
    names = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    names |= {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert names == {'__future__', 'dataclasses', 're', 'numpy'}
