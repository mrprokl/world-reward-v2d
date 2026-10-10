import numpy as np
import pytest
from world_reward.mask_observation import alignment_masks


def test_object_absence_does_not_block_human_scale_fit():
    human = np.full((4, 5), 255, np.uint8); obj = np.zeros_like(human)
    h, o = alignment_masks(human, obj)
    assert h.all() and not o.any()
    np.testing.assert_array_equal(h & ~o, human > 0)


def test_present_masks_preserve_original_alignment_math():
    h = np.full((4, 5), 255, np.uint8); o = np.zeros_like(h); o[1:3, 2:4] = 255
    a, b = alignment_masks(h, o)
    np.testing.assert_array_equal(a & ~b, (h > 0) & ~(o > 0))


@pytest.mark.parametrize('fault', ['emptyhuman', 'shape', 'float', 'nonbinary'])
def test_invalid_inputs_remain_failures(fault):
    h = np.full((4, 5), 255, np.uint8); o = np.zeros_like(h)
    if fault == 'emptyhuman': h[:] = 0
    if fault == 'shape': o = o[:2]
    if fault == 'float': o = o.astype(float)
    if fault == 'nonbinary': o[1, 1] = 1
    with pytest.raises(ValueError): alignment_masks(h, o)
