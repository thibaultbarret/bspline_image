# SPDX-License-Identifier: BSD-3-Clause
"""Analytic speckle patterns shared by the executed examples.

A speckle is a sum of isotropic Gaussian spots at random positions. Because it is an
analytic function of (x, y), it can be sampled on the pixel grid to build an image and
also evaluated exactly at any sub-pixel position, which gives a ground truth against
which interpolation can be checked.
"""

from collections.abc import Callable

import numpy as np

AMPLITUDE_RANGE = (0.5, 1.0)
SPOT_REACH = 3.0  # spots farther than this many radii outside the image are negligible

Field = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _draw_spots(
    n_spots: int, radius: float, extent: tuple[float, float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centres_x, centres_y, amplitudes) drawn from a seeded generator.

    Centres are uniform over the image extent enlarged by ``SPOT_REACH * radius`` on
    every side, with the count scaled so that ``n_spots`` is the number of spots per
    image area. Spots centred just outside the image still light its border, so the
    density of grey levels is uniform up to the edges.
    """
    rng = np.random.default_rng(seed)
    width, height = extent
    pad = SPOT_REACH * radius
    area_ratio = (width + 2 * pad) * (height + 2 * pad) / (width * height)
    n_total = round(n_spots * area_ratio)
    cx = rng.uniform(-pad, width + pad, n_total)
    cy = rng.uniform(-pad, height + pad, n_total)
    amplitude = rng.uniform(*AMPLITUDE_RANGE, n_total)
    return cx, cy, amplitude


def _spot_terms(cx, cy, amplitude, radius, x, y):
    """Yield (gaussian, dx, dy) for every spot, with dx = x - cx and dy = y - cy."""
    for cxi, cyi, ai in zip(cx, cy, amplitude):
        dx = x - cxi
        dy = y - cyi
        yield ai * np.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius)), dx, dy


def make_speckle(
    n_spots: int, radius: float, extent: tuple[float, float], seed: int
) -> Field:
    """Build f(x, y), a sum of ``n_spots`` Gaussians of standard deviation ``radius``.

    ``extent`` is the extent of the image, ``(width - 1, height - 1)``, and ``n_spots`` the
    number of spots over that area (centres also fall up to 3 radii outside it, see
    `_draw_spots`). Amplitudes are uniform in ``[0.5, 1]``. The returned function is
    vectorised over numpy arrays.
    """
    cx, cy, amplitude = _draw_spots(n_spots, radius, extent, seed)

    def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros(np.broadcast(x, y).shape)
        for gauss, _, _ in _spot_terms(cx, cy, amplitude, radius, x, y):
            total += gauss
        return total

    return field


def make_speckle_gradient(
    n_spots: int, radius: float, extent: tuple[float, float], seed: int
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build (df/dx, df/dy) of the speckle made by `make_speckle` (same arguments)."""
    cx, cy, amplitude = _draw_spots(n_spots, radius, extent, seed)
    inv_var = 1.0 / (radius * radius)

    def gradient(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(x, y).shape
        gx, gy = np.zeros(shape), np.zeros(shape)
        for gauss, dx, dy in _spot_terms(cx, cy, amplitude, radius, x, y):
            gx -= gauss * dx * inv_var
            gy -= gauss * dy * inv_var
        return gx, gy

    return gradient


def sample(f: Field, width: int, height: int) -> np.ndarray:
    """Sample ``f`` on the pixel grid: ``image[y, x] = f(x, y)`` (float64)."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    return np.asarray(f(xs, ys), dtype=np.float64)
