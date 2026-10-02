"""Tiny numerical/provenance guards only; no rendering, shape experiment or Torch."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def synthetic():
    path = Path(__file__).resolve().parents[1] / "infra/shape_synthetic.py"
    spec = importlib.util.spec_from_file_location("world_reward_test_shape_synthetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predeclared_cases_frames_and_controls_frozen_before_experiment(synthetic):
    assert synthetic.CASES == (
        ("asymmetric", (.40, .23, .31), (.06, -.05, .02, -.03, .01)),
        ("thin", (.10, .22, .35), (.08, -.04, -.02, .01, .025)),
        ("symmetric_control", (.30, .30, .30), (0., 0., 0., 0., 0.)),
    )
    assert synthetic.FIT_FRAMES == (0, 2, 4) and synthetic.HELDOUT_FRAMES == (1, 3, 5)
    assert set(synthetic.FIT_FRAMES).isdisjoint(synthetic.HELDOUT_FRAMES)
    assert sorted(synthetic.FIT_FRAMES + synthetic.HELDOUT_FRAMES) == list(range(6))
    assert synthetic.OCCLUSIONS == (0., .25, .5)
    assert synthetic.SURFACE_SAMPLES == 8192 and synthetic.MAX_OBSERVATIONS == 2048


@pytest.mark.parametrize("fraction", [0., .25, .5])
def test_rectangular_occlusion_hides_existing_foreground_without_filling(synthetic, fraction):
    mask = np.zeros((8, 12), dtype=bool)
    mask[2:6, 2:10] = True
    before = mask.copy()
    visible, occluder, actual = synthetic.visible_region(mask, fraction)
    np.testing.assert_array_equal(mask, before)
    np.testing.assert_array_equal(visible, mask & ~occluder)
    assert not np.any(visible & ~mask)
    assert actual == pytest.approx(fraction)
    if fraction:
        rows, columns = np.where(occluder)
        assert set(rows) == set(range(2, 6))
        assert occluder[rows.min():rows.max() + 1, :columns.max() + 1].all()
    else:
        assert not occluder.any()


def test_occlusion_fraction_is_reported_not_assumed_for_irregular_mask(synthetic):
    mask = np.zeros((3, 5), dtype=bool)
    mask[0, 0] = mask[1, 0] = mask[2, 4] = True
    visible, hidden, actual = synthetic.visible_region(mask, .5)
    assert actual == pytest.approx(2 / 3)
    assert visible.any() and np.count_nonzero(visible) == 1


@pytest.mark.parametrize("mask,fraction", [(np.zeros((3, 3), bool), .25), (np.ones((2, 2)), .25),
                                           (np.ones(3, bool), .25), (np.ones((3, 3), bool), True),
                                           (np.ones((3, 3), bool), np.nan), (np.ones((3, 3), bool), 1.),
                                           (np.ones((3, 3), bool), -.1)])
def test_invalid_mask_occlusion_fails_explicitly(synthetic, mask, fraction):
    with pytest.raises(ValueError):
        synthetic.visible_region(mask, fraction)


def test_camera_ray_uses_pixel_centres_camera_z_not_ray_distance(synthetic):
    depth = np.array([[2., np.nan], [4., 6.]])
    mask = np.array([[True, False], [True, True]])
    K = np.array([[2., 0., .25], [0., 4., .75], [0., 0., 1.]])
    points, indices = synthetic.camera_observations(depth, mask, K)
    np.testing.assert_array_equal(indices, [0, 2, 3])
    np.testing.assert_allclose(points, [[.25, -.125, 2.], [.5, .75, 4.], [3.75, 1.125, 6.]])
    np.testing.assert_array_equal(points[:, 2], depth[mask])
    assert np.linalg.norm(points[-1]) > depth[-1, -1]


@pytest.mark.parametrize("field", ["depth", "visible", "camera"])
def test_masked_arrays_cannot_hide_invalid_evidence(synthetic, field):
    values = {"depth": np.ones((2, 2)), "visible": np.ones((2, 2), bool), "camera": np.eye(3)}
    values[field] = np.ma.array(values[field], mask=False)
    with pytest.raises(ValueError, match="Masked arrays"):
        synthetic.camera_observations(values["depth"], values["visible"], values["camera"])
    with pytest.raises(ValueError):
        synthetic.visible_region(np.ma.array(np.ones((2, 2), bool), mask=False), .25)


def test_observation_selection_deterministic_original_row_major_no_mutation(synthetic):
    z, mask = np.full((3, 5), 2.), np.ones((3, 5), dtype=bool)
    old_z, old_mask = z.copy(), mask.copy()
    first, indices = synthetic.camera_observations(z, mask, np.eye(3), maximum=5)
    second, second_indices = synthetic.camera_observations(z, mask, np.eye(3), maximum=5)
    np.testing.assert_array_equal(indices, np.linspace(0, 14, 5, dtype=np.int64))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(indices, second_indices)
    np.testing.assert_array_equal(z, old_z)
    np.testing.assert_array_equal(mask, old_mask)


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_visible_invalid_depth_not_silently_filtered(synthetic, value):
    depth = np.ones((2, 2))
    depth[0, 0] = value
    with pytest.raises(ValueError, match="positive"):
        synthetic.camera_observations(depth, np.ones((2, 2), bool), np.eye(3))


@pytest.mark.parametrize("maximum", [True, 2, 2049, 3., "4"])
def test_observation_cap_requires_explicit_bounded_integer(synthetic, maximum):
    with pytest.raises(ValueError):
        synthetic.camera_observations(np.ones((2, 2)), np.ones((2, 2), bool), np.eye(3), maximum)


@pytest.mark.parametrize("K", [np.eye(4), np.ones((3, 3)), np.diag([0., 1., 1.]),
                              np.diag([1., 1., 2.]), np.full((3, 3), np.nan),
                              np.array([[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]])])
def test_camera_intrinsics_cannot_be_inferred_or_repaired(synthetic, K):
    with pytest.raises(ValueError):
        synthetic.camera_observations(np.ones((2, 2)), np.ones((2, 2), bool), K)


def test_controlled_and_perturbed_poses_deterministic_proper_rotation_fixed_metric_errors(synthetic):
    from scipy.spatial.transform import Rotation
    R, t = synthetic.controlled_poses()
    R1, t1 = synthetic.controlled_poses(True)
    np.testing.assert_allclose(np.linalg.det(R), 1., atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(R1), 1., atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(t1 - t, axis=1), .005, atol=1e-15)
    np.testing.assert_allclose(Rotation.from_matrix(R1 @ R.transpose(0, 2, 1)).magnitude(), .02, atol=1e-15)
    assert t[:, 2].tolist() == pytest.approx(np.linspace(3., 4., 6))
    R2, t2 = synthetic.controlled_poses(True)
    np.testing.assert_array_equal(R1, R2)
    np.testing.assert_array_equal(t1, t2)
    with pytest.raises(ValueError):
        synthetic.controlled_poses(1)


@pytest.fixture
def tetrahedron():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


def test_physical_metric_has_no_ground_truth_alignment_or_scale_normalization(synthetic, tetrahedron):
    vertices, faces = tetrahedron
    sample = np.array([[0., 0., 0.]])
    report = synthetic.physical_metrics(sample + [.3, .4, 0.], sample, vertices, vertices, faces)
    assert report["chamfer_m"] == pytest.approx(.5)
    assert report["pred_to_truth_m"] == pytest.approx(.5)
    assert report["truth_to_pred_m"] == pytest.approx(.5)
    assert report["signed_volume_m3"] == pytest.approx(1 / 6)
    assert report["relative_volume_error"] == 0
    json.dumps(report, allow_nan=False)


def test_signed_volume_does_not_reward_shrinking_or_repair_reversed_winding(synthetic, tetrahedron):
    vertices, faces = tetrahedron
    report = synthetic.physical_metrics(vertices * .5, vertices, vertices * .5, vertices, faces)
    assert report["relative_volume_error"] == pytest.approx(.875)
    assert report["chamfer_m"] > 0
    with pytest.raises(ValueError, match="positive"):
        synthetic.physical_metrics(vertices, vertices, vertices, vertices, faces[:, ::-1])


@pytest.mark.parametrize("kind", ["nan", "empty", "negative_index", "float_faces", "out_of_bounds"])
def test_metric_invalid_geometry_or_topology_fails(synthetic, tetrahedron, kind):
    vertices, faces = tetrahedron
    sample = vertices.copy()
    if kind == "nan": sample[0, 0] = np.nan
    elif kind == "empty": sample = np.empty((0, 3))
    elif kind == "negative_index": faces[0, 0] = -1
    elif kind == "float_faces": faces = faces.astype(float)
    else: faces[0, 0] = 4
    with pytest.raises(ValueError):
        synthetic.physical_metrics(sample, vertices, vertices, vertices, faces)


def test_fit_receives_only_base_observations_and_fixed_fit_frames_no_truth(synthetic, monkeypatch):
    import world_reward.shape_fit as fitter
    surface, centroid = np.arange(12).reshape(4, 3), np.array([1., 2., 3.])
    observations = [np.full((4, 3), i) for i in range(6)]
    rotations, translations = synthetic.controlled_poses()
    sentinel, calls = object(), []
    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    monkeypatch.setattr(fitter, "fit_shared_shape", fit)
    assert synthetic.fit_condition(surface, centroid, observations, rotations, translations) is sentinel
    args, kwargs = calls[0]
    assert len(args) == 5 and kwargs == {}
    assert args[0] is surface and args[1] is centroid
    for i, actual in zip(synthetic.FIT_FRAMES, args[2]):
        assert actual is observations[i]
    np.testing.assert_array_equal(args[3], rotations[[0, 2, 4]])
    np.testing.assert_array_equal(args[4], translations[[0, 2, 4]])


def test_rejected_proposal_keeps_exact_base_not_even_decoding_candidate(synthetic, monkeypatch):
    import world_reward.shape_model as model
    base = np.arange(12).reshape(4, 3).astype(float)
    monkeypatch.setattr(model, "apply_fixed_shape", lambda *_args, **_kwargs: pytest.fail("Rejected candidate must not deform source"))
    result = synthetic.adopted_vertices(base, np.zeros(3), SimpleNamespace(accepted=False, params5=np.full(5, np.nan)))
    np.testing.assert_array_equal(result, base)
    assert not np.shares_memory(result, base)


def test_main_existing_report_fails_before_torch_import_or_experiment(synthetic, monkeypatch, tmp_path):
    output = tmp_path / "results/shape-synthetic.json"
    output.parent.mkdir()
    output.write_text("frozen")
    monkeypatch.setattr(synthetic.platform, "system", lambda: "Linux")
    old = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter([Path("/sys/class/net/lo")]) if str(p) == "/sys/class/net" else old(p))
    monkeypatch.setenv("WR_ROOT", str(tmp_path))
    monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(synthetic, "_experiment", lambda *_: pytest.fail("Frozen output cannot run experiment"))
    with pytest.raises(FileExistsError, match="frozen"):
        synthetic.main()
    assert output.read_text() == "frozen"


def test_wrapper_only_immutable_code_and_scalar_results_no_data_weights_network(synthetic):
    script = Path(synthetic.__file__).with_name("run_shape_synthetic.sh").read_text()
    assert '--network none' in script and '--gpus all' in script
    assert 'src=$CODE,dst=$CODE,readonly' in script
    assert 'WR_CODE_REVISION' in script and 'PYTHONPATH="$CODE/src"' in script
    assert 'world-reward/cari4d-source:0.1' in script
    assert 'src=$ROOT/data' not in script and 'src=$ROOT/weights' not in script
    assert 'docker build' not in script and 'curl' not in script
