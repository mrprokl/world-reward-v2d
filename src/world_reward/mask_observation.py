"""Automatic mask support is an observation, never an existence assertion."""
import numpy as np


def alignment_masks(human, obj):
    """Human anchors scale; an absent object observation cannot block that fit.

    Empty human support is invalid. Empty object support is retained literally:
    it only removes no pixels from the human fit. No object mask is imputed.
    """
    values = [np.asarray(v) for v in (human, obj)]
    if (any(np.ma.isMaskedArray(v) for v in (human, obj))
            or any(v.ndim != 2 or v.dtype != np.uint8 or not np.isin(v, [0, 255]).all() for v in values)
            or values[0].shape != values[1].shape or not (values[0] > 0).any()):
        raise ValueError('Matching binary original-grid masks and nonempty human required')
    return values[0] > 0, values[1] > 0
