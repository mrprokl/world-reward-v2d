"""Fixed RGB gamma views and a geometry-only choice of an existing prediction.

No observation confidence, private reference, alignment or identity averaging is
used. Consistency is a diagnostic proxy, not a claim of reconstruction accuracy.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np

GAMMAS = (1., .8, 1.2)


def gamma_rgb(rgb: np.ndarray, gamma: float) -> np.ndarray:
    """Preserve RGB channels/grid; use the fixed round-nearest uint8 LUT."""
    if (type(rgb) is not np.ndarray or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
        raise ValueError("Nonempty original uint8 HWC RGB array required")
    if (isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, Real)
            or float(gamma) not in GAMMAS):
        raise ValueError("One preregistered gamma value required")
    if float(gamma) == 1.:
        return rgb.copy()
    levels = np.arange(256, dtype=np.float64) / 255.
    lut = np.floor(255. * levels ** float(gamma) + .5).astype(np.uint8)
    return lut[rgb]


def gamma_variants(rgb: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return gamma1, gamma.8 and gamma1.2 in protocol order, independently."""
    return tuple(gamma_rgb(rgb, gamma) for gamma in GAMMAS)


@dataclass(frozen=True)
class GeometryMedoid:
    index: int
    pairwise_squared_distances: np.ndarray
    scores: np.ndarray
    chosen_vertices: np.ndarray


def geometric_medoid(vertices: np.ndarray) -> GeometryMedoid:
    """Choose a full-correspondence vertex medoid, never a blended prediction.

    Inputs follow GAMMAS order. Every pair uses mean squared Euclidean distance
    over all vertices in float64; scores sum the two pair distances. Exact ties
    select the first index (the original RGB). Returned arrays are readonly
    copies, and the chosen geometry retains the original input dtype/bytes.
    """
    if (type(vertices) is not np.ndarray or vertices.dtype not in (np.float32, np.float64)
            or vertices.ndim != 3 or vertices.shape[0] != 3 or vertices.shape[2] != 3
            or vertices.shape[1] < 1 or not np.isfinite(vertices).all()
            or np.any(vertices[:, :, 2] <= 0)):
        raise ValueError("Three finite full-vertex float32/64 positiveZ predictions required")
    values = vertices.astype(np.float64, copy=True)
    distances = np.zeros((3, 3), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            for i in range(3):
                for j in range(i + 1, 3):
                    delta = values[i] - values[j]
                    distances[i, j] = distances[j, i] = np.mean(np.sum(delta * delta, axis=1))
            scores = distances.sum(axis=1)
        except FloatingPointError as error:
            raise ValueError("Finite float64 geometry distances required") from error
    if not np.isfinite(scores).all():
        raise ValueError("Finite float64 geometry distances required")
    index = int(np.argmin(scores))
    chosen = vertices[index].copy()
    for value in (distances, scores, chosen):
        value.setflags(write=False)
    return GeometryMedoid(index, distances, scores, chosen)
