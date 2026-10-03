"""Tiny NumPy geometry/receipt contracts only; no assets, GPU or media creation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def renderer():
    source = Path(__file__).parents[1]/"infra/authored_rgbd_render.py"
    spec = importlib.util.spec_from_file_location("authored_render_test", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def recipe(renderer):
    return renderer.protocol(Path(__file__).parents[1]/"configs/authored_rgbd_protocol.json")


def test_entire_recipe_hard_pinned_before_json_and_gates_preserved(renderer, tmp_path):
    frozen = recipe(renderer)
    assert frozen["namespace"] == "validation/authored_rgbd_holdout_v1"
    assert frozen["manufacture"]["runtime_image"].startswith("sha256:7ebfff18")
    assert frozen["grid"] == {"width": 640, "height": 480, "K": renderer.K.tolist(), "pixel_rays": "integer_indices_plus_0.5"}
    assert frozen["frozen_method"]["minimum_border_pairs"] == 1024
    assert frozen["frozen_method"]["minimum_border_coverage"] == .95
    assert frozen["private_evaluation"]["minimum_median_relative_gain"] == .05
    assert frozen["private_evaluation"]["samples_per_frame"] == 8192
    assert frozen["private_evaluation"]["minimum_visible_object_pixels"] == 32
    altered = tmp_path/"protocol.json"; altered.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="preregistered"): renderer.protocol(altered)
    link = tmp_path/"link"; link.symlink_to(Path(__file__).parents[1]/"configs/authored_rgbd_protocol.json")
    with pytest.raises(ValueError): renderer.protocol(link)


@pytest.mark.parametrize("scene", [0, 1, 2])
def test_three_new_closed_outward_convex_constant_ellipsoids(renderer, scene):
    record = recipe(renderer)["objects"][scene]
    v, f = renderer.ellipsoid_mesh(record["semiaxes_m"]); saved = v.copy(), f.copy()
    checked = renderer.mesh_gate(v, f)
    assert checked["vertices"] == 1106 and checked["faces"] == 2208
    assert checked["volume_m3"] > 0 and checked["closed_outward_convex_manifold"]
    np.testing.assert_allclose(np.sum((v/record["semiaxes_m"])**2, axis=1), 1, atol=1e-14)
    for frame in range(4):
        rotation, translation = renderer.pose(record, frame)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        assert np.linalg.det(rotation) == pytest.approx(1.)
        camera = v @ rotation.T+translation
        pixels = camera[:, :2]/camera[:, 2, None]*800+[320, 240]
        assert np.all(pixels > [8, 8]) and np.all(pixels < [632, 472])
        assert camera[:, 2].min() > .01
    np.testing.assert_array_equal(v, saved[0]); np.testing.assert_array_equal(f, saved[1])
    assert not np.array_equal(renderer.pose(record, 0)[1], renderer.pose(record, 3)[1])


@pytest.mark.parametrize("fault", ["open", "reverse", "collapse", "inactive", "nan", "nonconvex"])
def test_mesh_gate_never_repairs_or_drops(renderer, fault):
    v, f = renderer.ellipsoid_mesh([.25, .18, .22])
    if fault == "open": f = f[:-1]
    elif fault == "reverse": f = f[:, ::-1]
    elif fault == "collapse": v[f[0, 1]] = v[f[0, 0]]
    elif fault == "inactive": v = np.vstack((v, [0., 0., 0.]))
    elif fault == "nan": v[0, 0] = np.nan
    else: v[100] *= .5
    with pytest.raises(ValueError): renderer.mesh_gate(v, f)


@pytest.mark.parametrize("bad", [True, -1, 4, .0])
def test_original_four_uniform_frames_only(renderer, bad):
    with pytest.raises(ValueError): renderer.pose(recipe(renderer)["objects"][0], bad)


def tiny_plane(renderer, monkeypatch):
    monkeypatch.setattr(renderer, "WIDTH", 40); monkeypatch.setattr(renderer, "HEIGHT", 40)
    vertices = np.array([[-5., -5., 3.], [-5., 5., 3.], [5., -5., 3.]])
    return vertices, np.array([[0, 1, 2]], np.int64), np.zeros((40, 40), np.int64), np.full((40, 40), 3., np.float32)


def test_independent_ray_reference_all_pixels_front_surface_camera_z(renderer, monkeypatch):
    vertices, faces, indices, depth = tiny_plane(renderer, monkeypatch)
    checked = renderer.ray_gate(vertices, faces, indices, depth, 1)
    assert checked["checked_pixels"] == checked["visible_pixels"] == 1600
    assert checked["camera_z_max_error_m"] == 0 and checked["minimum_barycentric"] > 0
    ray = renderer.rays()
    assert ray[0, 0, 0] == (0.5-320)/800 and ray[0, 0, 1] == (0.5-240)/800
    assert np.max(np.linalg.norm(ray, axis=-1)*depth-depth) > .01


def test_convex_hull_independent_coverage_rejects_missing_or_extra_object_pixel(renderer, monkeypatch):
    monkeypatch.setattr(renderer, "WIDTH", 40); monkeypatch.setattr(renderer, "HEIGHT", 40)
    pixels = np.array([[5.25, 5.25], [34.25, 5.25], [34.25, 34.25], [5.25, 34.25], [20., 20.]])
    vertices = np.column_stack(((pixels-[320, 240])*2/800, np.full(5, 2.)))
    yy, xx = np.mgrid[:40, :40]
    visible = (xx+.5 > 5.25) & (xx+.5 < 34.25) & (yy+.5 > 5.25) & (yy+.5 < 34.25)
    checked = renderer.silhouette_gate(vertices, visible)
    assert checked["hull_interior_pixels"] == 29*29
    for index, value in (((20, 20), False), ((0, 0), True)):
        broken = visible.copy(); broken[index] = value
        with pytest.raises(ValueError): renderer.silhouette_gate(vertices, broken)


@pytest.mark.parametrize("fault", ["wrong_z", "back_face", "nan", "zero", "outside", "wrong_type", "missing_face"])
def test_ray_reference_rejects_wrong_depth_or_geometry(renderer, monkeypatch, fault):
    vertices, faces, indices, depth = tiny_plane(renderer, monkeypatch)
    if fault == "wrong_z": depth += .001
    elif fault == "back_face": faces = faces[:, ::-1]
    elif fault == "nan": depth[0, 0] = np.nan
    elif fault == "zero": depth[0, 0] = 0
    elif fault == "outside": vertices[:, 0] += 20
    elif fault == "wrong_type": depth = depth.astype(np.float64)
    else: indices[0, 0] = -1
    with pytest.raises(ValueError): renderer.ray_gate(vertices, faces, indices, depth, 1)


def test_public_exact_twelve_rgb_only_and_distinct_authored_namespace(renderer):
    public = renderer.public_manifest(["a"*64]*12)
    assert set(public) == {"schema", "images"}
    assert public["schema"] == "world_reward.authored_rgbd_public.v1"
    assert [(p["scene_id"], p["frame_id"]) for p in public["images"]] == [(s, f) for s in range(1, 4) for f in range(4)]
    assert public["images"][0]["file"] == "scene_000001_frame_000000.png"
    assert all(set(p) == {"scene_id", "frame_id", "file", "sha256", "width", "height"} for p in public["images"])


@pytest.mark.parametrize("hashes", [[], ["a"*64]*11, ["a"*64]*13, ["g"*64]*12, [None]*12])
def test_no_manifest_for_partial_cohort(renderer, hashes):
    with pytest.raises(ValueError): renderer.public_manifest(hashes)


def test_source_no_old_failed_human_or_mutated_renderer_globals(renderer):
    source = Path(renderer.__file__).read_text()
    assert not any(word in source for word in ("object_synthetic_render", "hand_synthetic_render", "MHR", "cari4d"))
    assert 'private.mkdir(mode=0o700)' in source and 'receipt.chmod(0o400)' in source
    assert 'signal.alarm(360)' in source and '"inference_performed": False' in source
    assert 'with (public/"manifest.json").open("x")' in source
    assert "arrays_removed=True" in source
