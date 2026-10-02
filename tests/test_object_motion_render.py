"""Only own geometry/motion/public contracts, never rendering or model loading."""
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def motion(monkeypatch):
    infra = Path(__file__).parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("object_motion_tests", infra / "object_motion_render.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("index,euler", [(0, 2), (1, 0), (2, 2)])
def test_new_geometric_parameters_same_clean_topology_not_old_o2_mesh(motion, index, euler):
    v, f, evidence = motion.geometry(index)
    assert evidence["euler_characteristic"] == euler and evidence["closed_oriented_edge_manifold"] is True
    assert evidence["volume_m3"] > 0 and evidence["pair_triangle_intersection_test_performed"] is False
    previous, old_faces = motion.render.ring_mesh() if index == 1 else motion.render.radial_mesh()
    np.testing.assert_array_equal(f, old_faces)
    assert not np.allclose(v, previous)
    again = motion.geometry(index)
    np.testing.assert_array_equal(v, again[0]); np.testing.assert_array_equal(f, again[1])


def test_textureless_control_no_hidden_vertex_identity_codes(motion):
    v, _, _ = motion.geometry(2)
    rgb = motion.colors(v, 2)
    np.testing.assert_array_equal(rgb, np.tile([.27, .36, .45], (len(v), 1)))
    for index in (0, 1):
        v, _, _ = motion.geometry(index); color = motion.colors(v, index)
        assert color.shape == v.shape and np.isfinite(color).all()
        assert np.ptp(color, axis=0).min() > .15 and np.all(color > 0) and np.all(color < 1)


def test_adjacent_proper_motion_and_frozen_fast_transition(motion):
    from scipy.spatial.transform import Rotation
    rotations, translations = [], []
    for i in range(8):
        r, t = motion.motion(i)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1.)
        rotations.append(r); translations.append(t)
    angles = Rotation.from_matrix(np.asarray(rotations)[1:] @ np.asarray(rotations)[:-1].transpose(0, 2, 1)).magnitude()
    displacement = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    assert np.all(angles[np.arange(7) != 4] < .05) and .18 < angles[4] < .22
    assert np.all(displacement[np.arange(7) != 4] < .013) and .065 < displacement[4] < .075
    for invalid in (True, -1, 8, 0.):
        with pytest.raises(ValueError): motion.motion(invalid)


def test_all_own_geometry_inside_original_grid_no_camera_adjustment_per_case(motion):
    assert motion.render.K[0, 0] == motion.render.K[1, 1] == 640
    for obj in range(3):
        v, _, _ = motion.geometry(obj)
        for frame in range(8):
            r, t = motion.motion(frame); camera = v @ r.T+t
            px = motion.render.project_camera_points(camera, motion.render.K)
            assert np.all(px > [8, 8]) and np.all(px < [504, 376])
            assert camera[:, 2].min() > .01


def test_manifest_contains_24_original_rgb_sha_grid_and_nothing_private(motion):
    result = motion.public_manifest(["a"*64]*24)
    assert set(result) == {"schema", "images"} and result["schema"] == "world-reward-object-motion-rgb-v1"
    assert [r["file"] for r in result["images"]] == [f"object_{o:02d}_frame_{f:03d}.png" for o in range(3) for f in range(8)]
    assert all(set(r) == {"file", "sha256", "width", "height"} and (r["width"], r["height"]) == (512, 384) for r in result["images"])


@pytest.mark.parametrize("digests", [[], ["a"*64]*23, ["a"*64]*25, [None]*24, ["x"*64]*24])
def test_partial_or_invalid_rgb_never_get_public_manifest(motion, digests):
    with pytest.raises(ValueError): motion.public_manifest(digests)


@pytest.mark.parametrize("index", [True, -1, 3, .0])
def test_object_identity_not_coerced(motion, index):
    with pytest.raises(ValueError): motion.geometry(index)


def test_render_only_frozen_budget_private_truth_and_scoped_wrapper(motion):
    source = Path(motion.__file__).read_text(); wrapper = Path(motion.__file__).with_name("run_object_motion_render.sh").read_text()
    assert 'private.mkdir(mode=0o700)' in source and 'private / "render-report.json"' in source
    assert '"inference_performed": False' in source and '"relative_tracking_accuracy_verified": False' in source
    assert '"inherited_anchor_bias_and_relative_motion_must_be_evaluated_separately": True' in source
    assert 'mkdir "$DEST"' in wrapper and 'chown scenesmith:scenesmith "$DEST"' in wrapper
    assert 'src=$DEST,dst=$DEST' in wrapper and 'src=$ROOT/validation,dst=' not in wrapper
    assert not any(f"src=$ROOT/{name}" in wrapper for name in ("weights", "data", "vendor", "outputs", "results"))
    assert "123s docker run" in wrapper and "--network none" in wrapper
    result = subprocess.run(["bash", str(Path(motion.__file__).with_name("run_object_motion_render.sh")), "--unknown"], capture_output=True)
    assert result.returncode == 2


@pytest.mark.parametrize("linked", ["root", "validation"])
def test_wrapper_rejects_symlink_parent_before_docker_or_directory_creation(motion, tmp_path, linked):
    real = tmp_path / "real"; real.mkdir()
    if linked == "root":
        (real / "validation").mkdir()
        root = tmp_path / "root"; root.symlink_to(real, target_is_directory=True)
    else:
        root = real
        target = tmp_path / "private"; target.mkdir()
        (root / "validation").symlink_to(target, target_is_directory=True)
    env = dict(os.environ, WR_ROOT=str(root), WR_CODE=str(tmp_path), WR_CODE_REVISION="a"*40)
    wrapper = Path(motion.__file__).with_name("run_object_motion_render.sh")
    result = subprocess.run(["bash", str(wrapper)], env=env, capture_output=True)
    assert result.returncode == 2 and b"Require real root" in result.stderr
    assert not (root / "validation" / "object_motion_v1").exists()
