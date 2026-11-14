# SPDX-License-Identifier: BSD-3-Clause
"""B-spline image interpolation for sub-pixel grey-level values and gradients."""

from typing import Tuple

import numpy as np
import numpy.typing as npt

from . import _core

__version__ = "0.1.0"
__all__ = ["BSplineImage", "__version__"]


_NATIVE_PIXEL_DTYPES = tuple(
    np.dtype(t) for t in (np.uint8, np.uint16, np.float32, np.float64)
)


def _normalise_image(img: npt.ArrayLike) -> np.ndarray:
    """Return a 2D array the core accepts: native supported dtype, unit column stride."""
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"image must be 2D, got {img.ndim} dimension(s)")
    if img.size == 0:
        raise ValueError("image must not be empty")
    if img.dtype not in _NATIVE_PIXEL_DTYPES or not img.dtype.isnative:
        img = img.astype(np.float64)
    if img.strides[1] != img.itemsize or img.strides[0] < img.shape[1] * img.itemsize:
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

    def __init__(
        self,
        img: npt.ArrayLike,
        degree: int = 3,
        dtype: npt.DTypeLike = np.float64,
    ) -> None:
        self._dtype = np.dtype(dtype)
        if self._dtype == np.float64:
            cls = _core.Spline64
        elif self._dtype == np.float32:
            cls = _core.Spline32
        else:
            raise ValueError("dtype must be float32 or float64")
        self._impl = cls(_normalise_image(img), degree)

    @property
    def degree(self) -> int:
        return self._impl.degree

    @property
    def width(self) -> int:
        return self._impl.width

    @property
    def height(self) -> int:
        return self._impl.height

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def value(self, x: float, y: float) -> float:
        """Interpolated grey level at (x = column, y = row); NaN if non-finite."""
        return self._impl.value(x, y)

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Gradient ``(dx, dy)`` at (x = column, y = row), derivatives along x and y."""
        return self._impl.gradient(x, y)

    def value_and_gradient(self, x: float, y: float) -> Tuple[float, float, float]:
        """Return ``(value, dx, dy)`` at (x = column, y = row), bit-identical to
        ``value`` and ``gradient``."""
        return self._impl.value_and_gradient(x, y)

    def _coords(
        self, xs: npt.ArrayLike, ys: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
        """Return flat contiguous coordinate arrays and the original shape."""
        xs = np.asarray(xs, dtype=self._dtype)
        ys = np.asarray(ys, dtype=self._dtype)
        if xs.shape != ys.shape:
            raise ValueError(f"xs and ys shapes differ: {xs.shape} vs {ys.shape}")
        return (
            np.ascontiguousarray(xs).ravel(),
            np.ascontiguousarray(ys).ravel(),
            xs.shape,
        )

    def values(self, xs: npt.ArrayLike, ys: npt.ArrayLike) -> np.ndarray:
        """Interpolated values at (xs = columns, ys = rows).

        ``xs`` and ``ys`` must have identical shapes (no broadcasting). Returns
        an array of that shape and of ``dtype``; 0-d inputs give a 0-d array.
        NaN where a coordinate is non-finite.
        """
        fx, fy, shape = self._coords(xs, ys)
        out = np.empty(fx.shape, dtype=self._dtype)
        self._impl.values(fx, fy, out)
        return out.reshape(shape)

    def gradients(
        self, xs: npt.ArrayLike, ys: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(dx, dy)`` arrays at (xs = columns, ys = rows).

        ``xs`` and ``ys`` must have identical shapes (no broadcasting); the
        outputs have that shape, and 0-d inputs give 0-d arrays.
        """
        fx, fy, shape = self._coords(xs, ys)
        dx = np.empty(fx.shape, dtype=self._dtype)
        dy = np.empty(fx.shape, dtype=self._dtype)
        self._impl.gradients(fx, fy, dx, dy)
        return dx.reshape(shape), dy.reshape(shape)

    def values_and_gradients(
        self, xs: npt.ArrayLike, ys: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(values, dx, dy)`` arrays at (xs = columns, ys = rows).

        ``xs`` and ``ys`` must have identical shapes (no broadcasting); the
        outputs have that shape, and 0-d inputs give 0-d arrays.
        """
        fx, fy, shape = self._coords(xs, ys)
        v = np.empty(fx.shape, dtype=self._dtype)
        dx = np.empty(fx.shape, dtype=self._dtype)
        dy = np.empty(fx.shape, dtype=self._dtype)
        self._impl.values_and_gradients(fx, fy, v, dx, dy)
        return v.reshape(shape), dx.reshape(shape), dy.reshape(shape)
