import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward.contracts import require_rigid_transforms


def poses():
    value = np.repeat(np.eye(4)[None], 3, axis=0)
    value[:, :3, :3] = Rotation.from_rotvec([[.1, .2, -.3], [1., 0, 0], [0, 0, 0]]).as_matrix()
    value[:, :3, 3] = [[0, 0, 2], [1, -.4, 3], [.2, 1, 4]]
    return value


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_proper_camera_transforms_pass_without_modification(dtype):
    value = poses().astype(dtype)
    before = value.copy()
    require_rigid_transforms(value, np.int64(3))
    np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("count", [True, False, np.bool_(True), 0, -1, 3., "3", None])
def test_invalid_frame_count_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        require_rigid_transforms(poses(), count)


@pytest.mark.parametrize("mutation, message", [
    (lambda x: x.__setitem__((0, 0, 0), 2), "orthonormal"),
    (lambda x: x.__setitem__((0, 0, 1), .3), "orthonormal"),
    (lambda x: x.__setitem__((0, 3, 0), .1), "bottom row"),
    (lambda x: x.__setitem__((0, 3, 3), 2), "bottom row"),
    (lambda x: x.__setitem__((1, 0, 3), np.nan), "finite real"),
    (lambda x: x.__setitem__((1, 0, 3), np.inf), "finite real"),
])
def test_invalid_transforms_fail_not_repair(mutation, message):
    value = poses()
    mutation(value)
    before = value.copy()
    with pytest.raises(ValueError, match=message):
        require_rigid_transforms(value, 3)
    np.testing.assert_array_equal(value, before)


def test_reflection_rejected_despite_orthonormality():
    value = poses()
    value[0, :3, 0] *= -1
    with pytest.raises(ValueError, match="not reflections"):
        require_rigid_transforms(value, 3)


@pytest.mark.parametrize("value", [np.eye(4), np.zeros((2, 4, 4)), np.zeros((3, 3, 4)),
                                  np.ones((3, 4, 4), bool), np.ones((3, 4, 4), complex),
                                  np.ones((3, 4, 4), object)])
def test_shape_or_non_real_dtype_rejected(value):
    with pytest.raises(ValueError, match="finite real"):
        require_rigid_transforms(value, 3)


def test_masked_value_cannot_conceal_invalid_transform():
    with pytest.raises(ValueError, match="hide invalid"):
        require_rigid_transforms(np.ma.array(poses(), mask=False), 3)


def test_rotation_overflow_rejected_without_warning():
    value = poses()
    value[0, :3, :3] = 1e308
    with pytest.raises(ValueError, match="orthonormal"):
        require_rigid_transforms(value, 3)
