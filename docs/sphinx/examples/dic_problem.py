# SPDX-License-Identifier: BSD-3-Clause
"""Common set-up of the DIC examples: a speckle pair related by a pure translation."""

import numpy as np

from bspline_image import BSplineImage
from speckle import make_speckle, sample

SIZE = 96
HALF_SIZE = 15  # the subset is (2 * HALF_SIZE + 1) pixels wide
SPECKLE_SEED = 11

_speckle = make_speckle(n_spots=450, radius=2.0, extent=(SIZE - 1, SIZE - 1), seed=SPECKLE_SEED)
reference = sample(_speckle, SIZE, SIZE)

_centre = SIZE // 2
_rows, _cols = np.mgrid[
    _centre - HALF_SIZE : _centre + HALF_SIZE + 1, _centre - HALF_SIZE : _centre + HALF_SIZE + 1
]
subset_x = _cols.ravel().astype(np.float64)
subset_y = _rows.ravel().astype(np.float64)
reference_values = reference[_rows.ravel(), _cols.ravel()]


def translated_spline(u_star, degree, gain=1.0, offset=0.0):
    """B-spline of G(x) = gain * f(x - u*) + offset: the reference translated by u*.

    The default gain and offset leave the grey levels unchanged; other values add a
    contrast and brightness change that a pure translation model cannot represent.
    """
    ux, uy = u_star
    deformed = gain * sample(lambda x, y: _speckle(x - ux, y - uy), SIZE, SIZE) + offset
    return BSplineImage(deformed, degree=degree)
