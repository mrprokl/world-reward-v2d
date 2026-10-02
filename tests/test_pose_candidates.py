"""Scalar/batch orchestration parity using tiny deterministic CPU-only stubs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from world_reward.rigid_alignment import RigidAlignment


@pytest.fixture
def helper(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_pose_candidates", infra / "pose_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    vertices = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]])
    faces = np.array([[0, 1, 2]])
    surface = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [.1, .1, 0.]])
    observed = surface + [0, 0, 3]
    mask = np.array([[True, True], [False, False]])
    K = np.array([[10., 0., 1.], [0., 10., 1.], [0., 0., 1.]])
    rotations = [np.eye(3) for _ in range(25)]
    translations = [np.array([float(number), 0., 3.]) for number in range(25)]
    return vertices, faces, surface, observed, mask, K, rotations, translations


class FakeCudaMask:
    def __init__(self, value, events):
        self.value, self.events = np.asarray(value), events
    def cpu(self):
        self.events.append("cpu")
        return self
    def numpy(self):
        self.events.append("numpy")
        return self.value


@pytest.fixture
def backends(helper, inputs, monkeypatch):
    events = []
    behavior = {"icp_value_error": set(), "fitted_near_plane": set(), "runtime": None,
                "batch_error": None, "bad_batch_shape": False}

    def image(posed):
        x = float(posed[0, 0])
        number, fitted = int(np.floor(x)), not x.is_integer()
        events.append(("render_fitted" if fitted else "render_initial", number))
        initial = np.array([[True, False], [False, False]])
        fitted_mask = np.array([[True, True], [False, False]])
        if number % 3 == 0:
            initial, fitted_mask = fitted_mask, initial
        return fitted_mask if fitted else initial

    def scalar(posed, faces, K, width, height):
        helper._mesh_inputs(posed, faces, K, width, height, 1e-4)
        if behavior["runtime"] == "scalar": raise RuntimeError("CUDA scalar backend failure")
        return FakeCudaMask(image(posed), events), None

    def batch(posed, faces, K, width, height):
        events.append(("batch_size", len(posed)))
        if behavior["batch_error"] is not None: raise behavior["batch_error"]
        for value in posed: helper._mesh_inputs(value, faces, K, width, height, 1e-4)
        masks = np.stack([image(value) for value in posed])
        if behavior["bad_batch_shape"]: masks = masks[:, :-1]
        return FakeCudaMask(masks, events), None

    def align(surface, observed, R, t):
        number = int(t[0])
        events.append(("icp", number))
        # No alternate mesh, sampled points, trim parameters or dtype supplied.
        assert surface is inputs[2] and observed is inputs[3]
        if behavior["runtime"] == "icp": raise RuntimeError("ICP global infrastructure failure")
        if number in behavior["icp_value_error"]: raise ValueError(f"Underconstrained candidate {number}")
        result_t = t.copy()
        result_t[0] += .25
        if number in behavior["fitted_near_plane"]: result_t[2] = 1e-5
        # One apparent image improvement still worsens residual (not accepted).
        final = .2 if number % 3 == 2 else .05
        return RigidAlignment(R.copy(), result_t, "improved", .1, final, 32, 2)

    monkeypatch.setattr(helper, "raster_camera_mesh", scalar)
    monkeypatch.setattr(helper, "raster_camera_mesh_batch", batch)
    monkeypatch.setattr(helper, "align_observed_points", align)
    return events, behavior


def original_scalar(helper, inputs):
    """Literal original producer loop; this reference never calls the new helper."""
    vertices, faces, surface, observed, mask, K, Rs, ts = inputs
    height, width = mask.shape
    candidates, rejected = [], []
    for number, (initial_R, initial_t) in enumerate(zip(Rs, ts, strict=True)):
        try:
            initial_mask, _ = helper.raster_camera_mesh(vertices @ initial_R.T + initial_t, faces, K, width, height)
            initial_iou = helper.silhouette_iou(initial_mask.cpu().numpy(), mask)
            fit = helper.align_observed_points(surface, observed, initial_R, initial_t)
            fitted_mask, _ = helper.raster_camera_mesh(vertices @ fit.rotation.T + fit.translation, faces, K, width, height)
            fitted_iou = helper.silhouette_iou(fitted_mask.cpu().numpy(), mask)
            accepted = fitted_iou >= initial_iou and fit.final_residual <= fit.initial_residual
            chosen_R, chosen_t = (fit.rotation, fit.translation) if accepted else (initial_R, initial_t)
            chosen_iou = fitted_iou if accepted else initial_iou
            chosen_residual = fit.final_residual if accepted else fit.initial_residual
            candidates.append({"hypothesis_index": number, "initial_silhouette_iou": initial_iou,
                               "fitted_silhouette_iou": fitted_iou, "icp_accepted_by_image_gate": bool(accepted),
                               "selected_silhouette_iou": chosen_iou, "selected_depth_residual_m": chosen_residual,
                               "rotation": chosen_R.tolist(), "translation": chosen_t.tolist(), "icp": fit.to_dict()})
        except ValueError as exc:
            rejected.append({"hypothesis_index": number, "reason": str(exc)})
    return candidates, rejected


@pytest.mark.parametrize("seed_count", [1, 7, 8, 9, 24, 25])
def test_all_candidate_and_report_bytes_match_original_scalar(helper, inputs, backends, seed_count):
    events, _ = backends
    inputs = (*inputs[:6], inputs[6][:seed_count], inputs[7][:seed_count])
    # Reuse original sample identities from the fixture when trimming seed lists.
    before = [value.copy() for value in inputs[:6]]
    original = original_scalar(helper, inputs)
    original_events = events.copy()
    events.clear()
    scalar = helper.evaluate_pose_candidates(*inputs)
    assert scalar == original and events == original_events
    events.clear()
    batch = helper.evaluate_pose_candidates(*inputs, render_batch_size=8)
    assert batch == original
    assert json.dumps(batch, sort_keys=True) == json.dumps(original, sort_keys=True)
    batch_calls = [event for event in events if isinstance(event, tuple) and event[0] == "batch_size"]
    assert all(1 <= event[1] <= 8 for event in batch_calls)
    assert len(batch_calls) == 2 * int(np.ceil(seed_count / 8))
    assert events.count("numpy") == events.count("cpu") == len(batch_calls)
    for value, old in zip(inputs[:6], before, strict=True): assert value.tobytes() == old.tobytes()


def test_mixed_bad_candidates_preserve_indices_reasons_and_fitted_failure_rejects_whole(helper, inputs, backends):
    events, behavior = backends
    behavior["icp_value_error"] = {5}
    behavior["fitted_near_plane"] = {7}
    inputs[7][2][2] = -1  # invalid initial pose, never enters ICP
    inputs[7][19][2] = np.nan
    inputs[6][23][:] = 0  # collapsed initial mesh
    original = original_scalar(helper, inputs)
    events.clear()
    scalar = helper.evaluate_pose_candidates(*inputs)
    scalar_events = events.copy()
    events.clear()
    batch = helper.evaluate_pose_candidates(*inputs, render_batch_size=8)
    assert scalar == original == batch
    assert json.dumps(batch) == json.dumps(original)
    assert [value["hypothesis_index"] for value in batch[1]] == [2, 5, 7, 19, 23]
    assert 7 not in [value["hypothesis_index"] for value in batch[0]]
    for number in (2, 19, 23):
        assert ("icp", number) not in events and ("icp", number) not in scalar_events
    assert ("icp", 5) in events and ("render_fitted", 5) not in events
    # Rejections from three phases must still return in original index order.
    assert [row["hypothesis_index"] for row in batch[0]] == sorted(row["hypothesis_index"] for row in batch[0])


def test_exact_acceptance_conjunction_not_residual_only_or_image_only(helper, inputs, backends):
    rows, _ = helper.evaluate_pose_candidates(*inputs, render_batch_size=8)
    assert rows[0]["icp_accepted_by_image_gate"] is False  # lower IoU, lower residual
    assert rows[1]["icp_accepted_by_image_gate"] is True   # higher IoU, lower residual
    assert rows[2]["icp_accepted_by_image_gate"] is False  # higher IoU, worse residual
    for number in (0, 2):
        assert rows[number]["rotation"] == inputs[6][number].tolist()
        assert rows[number]["translation"] == inputs[7][number].tolist()
        assert rows[number]["selected_depth_residual_m"] == .1


@pytest.mark.parametrize("mode", [1, 8])
def test_all_initial_invalid_returns_same_rejections_and_no_icp_or_raster(helper, inputs, backends, mode):
    events, _ = backends
    for translation in inputs[7]: translation[2] = -1
    expected = original_scalar(helper, inputs)
    events.clear()
    actual = helper.evaluate_pose_candidates(*inputs, render_batch_size=mode)
    assert actual == expected and not actual[0] and len(actual[1]) == 25
    assert not events  # Stub scalar/backend validator rejects before raster work.


@pytest.mark.parametrize("mode", [1, 8])
def test_all_icp_underconstrained_same_rejections_and_no_fitted_raster(helper, inputs, backends, mode):
    events, behavior = backends
    behavior["icp_value_error"] = set(range(25))
    expected = original_scalar(helper, inputs)
    events.clear()
    actual = helper.evaluate_pose_candidates(*inputs, render_batch_size=mode)
    assert actual == expected and not actual[0]
    assert not any(isinstance(event, tuple) and event[0] == "render_fitted" for event in events)


@pytest.mark.parametrize("mode", [1, 8])
@pytest.mark.parametrize("failure", ["topology", "zero_area", "intrinsics", "mask_dtype", "mask_shape"])
def test_shared_input_failures_propagate_before_candidate_or_gpu_work(helper, inputs, backends, mode, failure):
    events, _ = backends
    items = list(inputs)
    if failure == "topology": items[1] = np.array([[0, 1, 3]])
    elif failure == "zero_area": items[0] = np.zeros((3, 3))
    elif failure == "intrinsics": items[5][0, 0] = -1
    elif failure == "mask_dtype": items[4] = items[4].astype(np.uint8)
    elif failure == "mask_shape": items[4] = items[4][0]
    with pytest.raises(ValueError): helper.evaluate_pose_candidates(*items, render_batch_size=mode)
    assert not events


@pytest.mark.parametrize("exception", [RuntimeError("CUDA kernel failure"), ValueError("Batch invariant failure"),
                                       MemoryError("GPU batch memory pressure"), OSError("Global environment failure")])
def test_no_whole_batch_backend_exception_is_swallowed_or_retried(helper, inputs, backends, exception):
    events, behavior = backends
    behavior["batch_error"] = exception
    with pytest.raises(type(exception), match=str(exception)):
        helper.evaluate_pose_candidates(*inputs, render_batch_size=8)
    assert events == [("batch_size", 8)]


@pytest.mark.parametrize("mode", [1, 8])
def test_icp_runtime_is_not_candidate_rejection(helper, inputs, backends, mode):
    _, behavior = backends
    behavior["runtime"] = "icp"
    with pytest.raises(RuntimeError, match="global infrastructure"):
        helper.evaluate_pose_candidates(*inputs, render_batch_size=mode)


def test_scalar_cuda_runtime_is_not_candidate_rejection(helper, inputs, backends):
    _, behavior = backends
    behavior["runtime"] = "scalar"
    with pytest.raises(RuntimeError, match="CUDA scalar"):
        helper.evaluate_pose_candidates(*inputs)


def test_invalid_batch_output_fails_globally_no_partial_evidence(helper, inputs, backends):
    _, behavior = backends
    behavior["bad_batch_shape"] = True
    with pytest.raises(RuntimeError, match="original-resolution"):
        helper.evaluate_pose_candidates(*inputs, render_batch_size=8)


@pytest.mark.parametrize("size", [0, -1, 2, 16, True, 1., "8", None])
def test_only_declared_batch_sizes_allowed(helper, inputs, backends, size):
    events, _ = backends
    with pytest.raises(ValueError, match="exactly 1 or 8"):
        helper.evaluate_pose_candidates(*inputs, render_batch_size=size)
    assert not events


def test_mismatched_or_empty_seed_lists_not_silently_zipped(helper, inputs, backends):
    for rotations, translations in (([], []), (inputs[6], inputs[7][:-1])):
        with pytest.raises(ValueError, match="seed rotation/translation"):
            helper.evaluate_pose_candidates(*inputs[:6], rotations, translations)
