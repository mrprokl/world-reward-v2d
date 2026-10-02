"""Pure procedural mesh/camera/public contract tests; no GPU, media or assets."""

import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def render(monkeypatch):
    infra = Path(__file__).parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("object_render_tests", infra / "object_synthetic_render.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("generator,euler,count", [("radial_mesh", 2, (1986, 3968)), ("ring_mesh", 0, (2048, 4096))])
def test_frozen_own_meshes_closed_oriented_manifold_no_overlapping_union(render, generator, euler, count):
    v, f = getattr(render, generator)(); saved = v.copy(), f.copy()
    result = render.mesh_contract(v, f, euler)
    assert (result["vertices"], result["faces"]) == count and result["euler_characteristic"] == euler
    assert result["volume_m3"] > 0 and result["closed_oriented_edge_manifold"] is True
    assert result["embedding_by_analytic_construction"] is True
    assert result["pair_triangle_intersection_test_performed"] is False
    np.testing.assert_array_equal(v, saved[0]); np.testing.assert_array_equal(f, saved[1])
    again = getattr(render, generator)()
    np.testing.assert_array_equal(v, again[0]); np.testing.assert_array_equal(f, again[1])


def test_radial_fixture_positive_polynomial_star_surface_not_old_spd_fit(render):
    v, f = render.radial_mesh()
    unit = v / [.28, .21, .24]
    radius = np.linalg.norm(unit, axis=1)
    direction = unit / radius[:, None]
    x, y, z = direction.T
    expected = 1+.12*(x**3-3*x*y*y)+.09*x*y+.08*z*x+.04*y**3
    np.testing.assert_allclose(radius, expected, atol=1e-14)
    assert radius.min() > .7 and radius.max()-radius.min() > .2


def test_thin_elliptic_ring_embedded_angle_and_non_circular_sections(render):
    v, _ = render.ring_mesh()
    p = v.reshape(64, 32, 3)
    u = np.arange(64)*2*np.pi/64
    angle = np.arctan2(p[..., 1]/.18, p[..., 0]/.29)
    np.testing.assert_allclose(np.sin(angle), np.repeat(np.sin(u)[:, None], 32, axis=1), atol=1e-14)
    np.testing.assert_allclose(np.cos(angle), np.repeat(np.cos(u)[:, None], 32, axis=1), atol=1e-14)
    assert np.linalg.norm(p[..., :2]/[.29, .18], axis=-1).min() > .7
    assert np.ptp(v[:, 2]) < .12 and np.ptp(v[:, 0]) > .6


@pytest.mark.parametrize("kind", ["nan", "inactive", "badface", "open", "winding", "collapsed", "genus"])
def test_invalid_mesh_never_repaired_or_dropped(render, kind):
    v, f = render.radial_mesh(); expected = 2
    if kind == "nan": v[0, 0] = np.nan
    elif kind == "inactive": v = np.vstack((v, [0., 0., 0.]))
    elif kind == "badface": f[0, 0] = len(v)
    elif kind == "open": f = f[:-1]
    elif kind == "winding": f[0] = f[0, ::-1]
    elif kind == "collapsed": v[f[0, 1]] = v[f[0, 0]]
    else: expected = 0
    with pytest.raises(ValueError): render.mesh_contract(v, f, expected)


def test_six_true_cameras_original_grid_fixed_public_hidden_calibration(render):
    assert render.WIDTH == 512 and render.HEIGHT == 384
    assert render.K[0, 0] == render.K[1, 1] == np.hypot(512, 384)
    assert render.TRAIN_VIEWS == (0, 2, 4) and render.HELDOUT_VIEWS == (1, 3, 5)
    for generator in (render.radial_mesh, render.ring_mesh):
        v, f = generator()
        for index in range(6):
            r, t = render.camera_pose(index)
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
            assert np.linalg.det(r) == pytest.approx(1.)
            camera = v @ r.T+t
            px = render.project_camera_points(camera, render.K)
            assert np.all(px > [8, 8]) and np.all(px < [504, 376]) and camera[:, 2].min() > .01
    for invalid in (True, -1, 6, 0.):
        with pytest.raises(ValueError): render.camera_pose(invalid)


def test_continuous_material_no_side_label_codes_and_background_contrast(render):
    v, _ = render.radial_mesh(); color = render.material(v)
    assert color.shape == v.shape and np.isfinite(color).all() and np.all(color > 0) and np.all(color < 1)
    assert np.ptp(color, axis=0).min() > .1
    for background in render.BACKGROUNDS:
        assert np.max(color[:, 0])+.08 < background[0]


def test_rgb_manifest_only_twelve_hash_grid_no_camera_gt_labels(render):
    manifest = render.public_manifest(["a"*64]*12)
    assert set(manifest) == {"schema", "images"}
    assert manifest["schema"] == "world-reward-objects-rgb-inputs-v1"
    assert [item["file"] for item in manifest["images"]] == [f"object_{i:02d}_view_{j:02d}.png" for i in range(2) for j in range(6)]
    assert all(set(item) == {"file", "sha256", "width", "height"} for item in manifest["images"])


@pytest.mark.parametrize("values", [[], ["a"*64]*11, ["a"*64]*13, ["x"*64]*12, [None]*12])
def test_manifest_not_created_for_partial_or_unhashed_render(render, values):
    with pytest.raises(ValueError): render.public_manifest(values)


def test_private_truth_exclusive_scoped_mounts_and_frozen_resource_limit(render):
    source = Path(render.__file__).read_text()
    wrapper = Path(render.__file__).with_name("run_object_synthetic_render.sh").read_text()
    assert 'private.mkdir(mode=0o700)' in source and 'private / "render-report.json"' in source
    assert 'with (public / "manifest.json").open("x")' in source
    assert '"inference_performed": False' in source and '"accuracy_verified": False' in source
    assert 'mkdir "$DEST"' in wrapper and 'chown scenesmith:scenesmith "$DEST"' in wrapper
    assert 'src=$DEST,dst=$DEST' in wrapper and 'src=$ROOT/validation,dst=' not in wrapper
    assert not any(f"src=$ROOT/{name}" in wrapper for name in ("weights", "data", "vendor", "outputs", "results"))
    assert "--network none" in wrapper and "123s docker run" in wrapper and "--memory 4g" in wrapper
    result = subprocess.run(["bash", str(Path(render.__file__).with_name("run_object_synthetic_render.sh")), "--unknown"], capture_output=True)
    assert result.returncode == 2
