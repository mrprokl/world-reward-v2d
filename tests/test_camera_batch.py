import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("wr_batch_renderer_test", Path(__file__).resolve().parents[1] / "infra/camera_render.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
K = np.array([[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])
TRIANGLE = np.array([[0., 0., 2.], [.1, 0., 2.], [0., .1, 2.]])
FACES = np.array([[0, 1, 2]])


def test_batch_contract_same_as_scalar_no_mutation():
    value = np.stack([TRIANGLE, TRIANGLE + [0, 0, 1]])
    before = value.copy()
    actual, faces, matrix = renderer._mesh_batch_inputs(value, FACES, K, 1536, 1152, 1e-4)
    np.testing.assert_array_equal(value, before)
    np.testing.assert_array_equal(faces, FACES)
    np.testing.assert_array_equal(matrix, K)
    for index in range(2):
        scalar = renderer._mesh_inputs(value[index], FACES, K, 1536, 1152, 1e-4)
        np.testing.assert_array_equal(actual[index], scalar[0])


@pytest.mark.parametrize("value", [TRIANGLE, np.zeros((0, 3, 3)), np.zeros((1, 2, 3)),
                                  np.zeros((1, 3, 4)), np.ones((1, 3, 3), bool)])
def test_invalid_batch_shape_or_numeric_type_rejected(value):
    with pytest.raises(ValueError):
        renderer.raster_camera_mesh_batch(value, FACES, K)


@pytest.mark.parametrize("mutation", [lambda x: x.__setitem__((1, 0, 2), 0),
                                     lambda x: x.__setitem__((1, 0, 0), np.nan),
                                     lambda x: x.__setitem__((1, 0, 0), np.inf),
                                     lambda x: x.__setitem__((1, 1), x[1, 0])])
def test_one_bad_candidate_rejects_whole_batch_never_dropped(mutation):
    value = np.stack([TRIANGLE, TRIANGLE.copy()])
    mutation(value)
    with pytest.raises(ValueError):
        renderer.raster_camera_mesh_batch(value, FACES, K)


def test_local_batch_cannot_import_torch_as_processing_fallback(monkeypatch):
    monkeypatch.setattr(renderer.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="Azure Linux CUDA"):
        renderer.raster_camera_mesh_batch(TRIANGLE[None], FACES, K)
