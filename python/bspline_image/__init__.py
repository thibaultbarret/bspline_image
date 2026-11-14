# SPDX-License-Identifier: BSD-3-Clause
"""B-spline image interpolation for sub-pixel grey-level values and gradients."""

import numpy as np

from . import _core

__version__ = "0.1.0"
__all__ = ["BSplineImage", "__version__"]


class BSplineImage:
    """B-spline interpolant of a 2D image; coordinates are (x = column, y = row)."""

    def __init__(self, img, degree=3, dtype=np.float64):
        self._dtype = np.dtype(dtype)
        if self._dtype == np.float64:
            cls = _core.Spline64
        elif self._dtype == np.float32:
            cls = _core.Spline32
        else:
            raise ValueError("dtype must be float32 or float64")
        self._impl = cls(np.ascontiguousarray(img), degree)

    @property
    def degree(self):
        return self._impl.degree

    @property
    def width(self):
        return self._impl.width

    @property
    def height(self):
        return self._impl.height

    @property
    def dtype(self):
        return self._dtype

    def value(self, x, y):
        return self._impl.value(x, y)
