# SPDX-License-Identifier: BSD-3-Clause
"""B-spline image interpolation for sub-pixel grey-level values and gradients."""

import numpy as np

from . import _core

__version__ = "0.1.0"
__all__ = ["BSplineImage", "__version__"]


_NATIVE_PIXEL_DTYPES = tuple(
    np.dtype(t) for t in (np.uint8, np.uint16, np.float32, np.float64)
)


def _normalise_image(img):
    """Return a 2D array the core accepts: native supported dtype, unit column stride."""
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"image must be 2D, got {img.ndim} dimension(s)")
    if img.size == 0:
        raise ValueError("image must not be empty")
    if img.dtype not in _NATIVE_PIXEL_DTYPES or not img.dtype.isnative:
        img = img.astype(np.float64)
    if img.strides[1] != img.itemsize:
        img = np.ascontiguousarray(img)
    return img


class BSplineImage:
    """B-spline interpolant of a 2D image.

    Coordinates are (x = column, y = row): pixel ``img[y, x]`` sits at (x, y).

    Args:
        img: 2D array-like. uint8, uint16, float32 and float64 (native byte
            order) are used as is; any other dtype is converted to float64.
        degree: spline degree, 0 to 7 (``ValueError`` otherwise).
        dtype: coefficient/evaluation precision, float64 or float32.
    """

    def __init__(self, img, degree=3, dtype=np.float64):
        self._dtype = np.dtype(dtype)
        if self._dtype == np.float64:
            cls = _core.Spline64
        elif self._dtype == np.float32:
            cls = _core.Spline32
        else:
            raise ValueError("dtype must be float32 or float64")
        self._impl = cls(_normalise_image(img), degree)

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
        """Interpolated grey level at (x = column, y = row); NaN if non-finite."""
        return self._impl.value(x, y)

    def gradient(self, x, y):
        """Gradient ``(dx, dy)`` at (x = column, y = row), derivatives along x and y."""
        return self._impl.gradient(x, y)

    def value_and_gradient(self, x, y):
        """Return ``(value, dx, dy)`` at (x = column, y = row), bit-identical to
        ``value`` and ``gradient``."""
        return self._impl.value_and_gradient(x, y)
