"""Tiny analytic arrays/contract checks only: no Azure, GPU, assets or PNGs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def renderer():
    source = Path(__file__).parents[1]/"infra/analytic_rgbd_render.py"
    spec = importlib.util.spec_from_file_location("analytic_render_test", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def recipe(renderer):
    return renderer.protocol(Path(__file__).parents[1]/"configs/analytic_rgbd_protocol.json")


def test_entire_preregistered_recipe_and_unchanged_downstream_gate(renderer, tmp_path):
    frozen = recipe(renderer)
    assert frozen["namespace"] == "validation/analytic_rgbd_holdout_v1"
    assert frozen["manufacture"]["device"] == "cpu" and frozen["manufacture"]["budget_seconds"] == 360
    assert frozen["grid"]["K"] == renderer.K.tolist()
    assert frozen["frozen_method"]["minimum_border_pairs"] == 1024
    assert frozen["frozen_method"]["minimum_border_coverage"] == .95
    assert frozen["private_evaluation"]["samples_per_frame"] == 8192
    assert frozen["private_evaluation"]["minimum_median_relative_gain"] == .05
    assert frozen["private_evaluation"]["maximum_any_scene_relative_regression"] == .05
    altered = tmp_path/"protocol.json"; altered.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="preregistered"): renderer.protocol(altered)
    link = tmp_path/"link"; link.symlink_to(Path(__file__).parents[1]/"configs/analytic_rgbd_protocol.json")
    with pytest.raises(ValueError): renderer.protocol(link)


@pytest.mark.parametrize("scene", [0, 1, 2])
def test_constant_analytic_closed_convex_shape_and_four_rigid_poses(renderer, scene):
    record = recipe(renderer)["objects"][scene]; axes = np.array(record["semiaxes_m"]); original = axes.copy()
    for frame in range(4):
        rotation, translation = renderer.pose(record, frame)
        checked = renderer.geometry_gate(axes, rotation, translation, record["background_z_m"])
        assert checked["closed_convex_positive_implicit_ellipsoid"] and checked["volume_m3"] > 0
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    np.testing.assert_array_equal(axes, original)
    assert not np.array_equal(renderer.pose(record, 0)[1], renderer.pose(record, 3)[1])
    assert record["material_seed"] not in [2026103101, 2026103102, 2026103103]
    assert record["semiaxes_m"] not in [[.25, .18, .22], [.19, .30, .16], [.32, .15, .24]]


@pytest.mark.parametrize("bad", [True, -1, 4, .0])
def test_exact_original_indices_no_repair(renderer, bad):
    with pytest.raises(ValueError): renderer.pose(recipe(renderer)["objects"][0], bad)


@pytest.mark.parametrize("fault", ["nonpositive", "reflection", "camera_inside", "behind_plane", "outside_frustum", "nan"])
def test_geometry_gate_no_fix_or_component_drop(renderer, fault):
    axes = np.array([.2, .2, .2]); rotation = np.eye(3); translation = np.array([0., 0., 2.]); plane = 4.
    if fault == "nonpositive": axes[0] = 0
    elif fault == "reflection": rotation[0, 0] = -1
    elif fault == "camera_inside": translation[2] = .1
    elif fault == "behind_plane": plane = 2.
    elif fault == "outside_frustum": translation[0] = 1.
    else: axes[0] = np.nan
    with pytest.raises(ValueError): renderer.geometry_gate(axes, rotation, translation, plane)


def test_stable_nearest_hit_physical_plane_and_independent_reference(renderer):
    axes = np.ones(3); rotation = np.eye(3); translation = np.array([0., 0., 3.])
    direction = np.array([[[0., 0., 1.], [.05, .1, 1.], [1., 0., 1.]]])
    depth64, depth, visible, checked = renderer.intersections(axes, rotation, translation, 6., direction)
    assert visible.tolist() == [[True, True, False]]
    assert depth64[0, 0] == depth[0, 0] == 2.
    assert depth64[0, 2] == 6.
    assert checked["checked_pixels"] == 3 and checked["visible_pixels"] == 2
    assert checked["independent_root_max_error_m"] < 1e-14
    assert checked["fp32_depth_cast_max_error_m"] < 1e-6
    assert checked["first_hit_entering_derivative"]
    assert np.linalg.norm(direction[0, 1]*depth64[0, 1]) != depth64[0, 1]


def test_tangency_ambiguity_stops_all_pixels_no_drop(renderer):
    direction = np.array([[[0., 0., 1.], [1/np.sqrt(3), 0., 1.]]])
    with pytest.raises(ValueError, match="nonambiguous"):
        renderer.intersections(np.ones(3), np.eye(3), np.array([0., 0., 2.]), 5., direction)


@pytest.mark.parametrize("fault", ["float32", "nan", "ray_norm_not_camera_z"])
def test_exact_finite_fp64_camera_z_rays(renderer, fault):
    direction = np.array([[[0., 0., 1.]]])
    if fault == "float32": direction = direction.astype(np.float32)
    elif fault == "nan": direction[0, 0, 0] = np.nan
    else: direction[0, 0, 2] = .9
    with pytest.raises(ValueError): renderer.intersections(np.ones(3), np.eye(3), np.array([0., 0., 3.]), 6., direction)


def test_decimal70_independent_probe_catches_wrong_saved_depth_and_visible(renderer):
    axes = np.array([2., 2., 1.]); rotation = np.eye(3); translation = np.array([0., 0., 3.])
    direction = renderer.rays(width=2, height=2)
    depth64, _, visible, _ = renderer.intersections(axes, rotation, translation, 6., direction)
    assert visible.all()  # Exercise high-precision first roots, not only the plane.
    checked = renderer.decimal_gate(axes, rotation, translation, 6., depth64, visible, [[0, 0], [1, 1]])
    assert checked["decimal_precision"] == 70 and checked["decimal_probes_checked"] == 2
    changed = depth64.copy(); changed[0, 0] += .001
    with pytest.raises(ValueError): renderer.decimal_gate(axes, rotation, translation, 6., changed, visible, [[0, 0]])
    changed_visible = ~visible
    with pytest.raises(ValueError): renderer.decimal_gate(axes, rotation, translation, 6., depth64, changed_visible, [[0, 0]])


def test_pixel_centres_and_seeded_same_hit_appearance_tiny_arrays(renderer):
    direction = renderer.rays(width=2, height=2)
    assert direction[0, 0].tolist() == [(.5-320)/800, (.5-240)/800, 1.]
    record = recipe(renderer)["objects"][0]; rotation, translation = renderer.pose(record, 0)
    depth64, _, visible, _ = renderer.intersections(np.array(record["semiaxes_m"]), rotation, translation,
                                                 record["background_z_m"], direction)
    first = renderer.appearance(record, rotation, translation, depth64, visible, direction)
    second = renderer.appearance(record, rotation, translation, depth64, visible, direction)
    assert first.shape == (2, 2, 3) and first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)


def test_public_manifest_exact_rgb_only_no_recipe_or_truth(renderer):
    manifest = renderer.public_manifest(["a"*64]*12)
    assert set(manifest) == {"schema", "images"}
    assert manifest["schema"] == "world_reward.analytic_rgbd_public.v1"
    assert [(p["scene_id"], p["frame_id"]) for p in manifest["images"]] == [(s, f) for s in range(1, 4) for f in range(4)]
    assert all(set(p) == {"scene_id", "frame_id", "file", "sha256", "width", "height"} for p in manifest["images"])
    with pytest.raises(ValueError): renderer.public_manifest(["a"*64]*11)


def test_cpu_exclusive_frozen_runtime_sources_and_owned_failure_cleanup(renderer):
    source = Path(renderer.__file__).read_text(); wrapper = (Path(renderer.__file__).parent/"run_analytic_rgbd_render.sh").read_text()
    assert not any(word in source for word in ("torch", "pytorch3d", "cari4d", "face_indices"))
    assert "--gpus" not in wrapper and "nvidia-smi" not in wrapper and "flock" not in wrapper
    assert "--name \"$CONTAINER\"" in wrapper and 'docker rm --force "$CONTAINER"' in wrapper
    assert 'timeout --signal=TERM --kill-after=2s 4s docker stop --time 2 "$CONTAINER"' in wrapper
    assert 'timeout --signal=TERM --kill-after=2s 4s docker kill "$CONTAINER"' in wrapper
    assert 'timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^${CONTAINER}$"' in wrapper
    assert 'cleanup || exit 1' in wrapper and "trap 'exit 143' TERM" in wrapper
    assert "363s docker run" in wrapper and "--kill-after=10s" in wrapper
    assert 'private.mkdir(mode=0o700)' in source and 'receipt.chmod(0o400)' in source
    assert "signal.alarm(360)" in source and '"inference_performed": False' in source
    assert len(source.splitlines()) <= 300 and len(wrapper.splitlines()) < 90
