# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import bspline_image as bsi

scipy_ndimage = pytest.importorskip("scipy.ndimage")

WIDTH, HEIGHT = 31, 27
HALF_INTEGER_MARGIN = 1e-6


def smooth_image():
    j, i = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    return 100 + 50 * np.sin(0.21 * i + 0.13 * j) + 30 * np.cos(0.17 * j - 0.09 * i)


def random_points():
    """500 points in [-5, W+5] x [-5, H+5], away from half-integers (rounding ties)."""
    rng = np.random.default_rng(1)
    xs = rng.uniform(-5, WIDTH + 5, 500)
    ys = rng.uniform(-5, HEIGHT + 5, 500)
    keep = (np.abs(np.abs(xs - np.floor(xs)) - 0.5) > HALF_INTEGER_MARGIN) & (
        np.abs(np.abs(ys - np.floor(ys)) - 0.5) > HALF_INTEGER_MARGIN
    )
    assert keep.all()
    return xs, ys


@pytest.mark.parametrize("degree", range(6))
def test_matches_scipy_mirror(degree):
    img = smooth_image()
    xs, ys = random_points()
    s = bsi.BSplineImage(img, degree=degree)

    expected = scipy_ndimage.map_coordinates(
        img, [ys, xs], order=degree, mode="mirror"
    )

    np.testing.assert_allclose(s.values(xs, ys), expected, rtol=0, atol=1e-9)
