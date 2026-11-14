# SPDX-License-Identifier: BSD-3-Clause
import math

import numpy as np
import pytest

import bspline_image as bsi


def smooth(h, w):
    j, i = np.mgrid[0:h, 0:w].astype(np.float64)
    return 100 + 50 * np.sin(0.7 * i + 0.3 * j) + 30 * np.cos(0.45 * j - 0.2 * i)


@pytest.mark.parametrize("degree", range(8))
def test_value_and_gradient_consistent(degree):
    s = bsi.BSplineImage(smooth(17, 23), degree=degree)
    rng = np.random.default_rng(degree)
    for x, y in zip(rng.uniform(0, 22, 20), rng.uniform(0, 16, 20)):
        x, y = float(x), float(y)
        assert s.value_and_gradient(x, y) == (s.value(x, y), *s.gradient(x, y))


@pytest.mark.parametrize("degree", range(2, 8))
def test_gradient_finite_differences(degree):
    s = bsi.BSplineImage(smooth(17, 23), degree=degree)
    h = 1e-5
    for x, y in [(5.3, 7.7), (10.45, 3.2), (15.9, 11.1)]:
        dx, dy = s.gradient(x, y)
        fdx = (s.value(x + h, y) - s.value(x - h, y)) / (2 * h)
        fdy = (s.value(x, y + h) - s.value(x, y - h)) / (2 * h)
        assert abs(dx - fdx) < 1e-4
        assert abs(dy - fdy) < 1e-4


def test_non_finite_coordinates():
    s = bsi.BSplineImage(smooth(17, 23))
    assert math.isnan(s.value(float("nan"), 1))
    assert math.isnan(s.value(np.inf, 1))


@pytest.mark.parametrize("degree", range(8))
def test_batch_matches_pointwise(degree):
    s = bsi.BSplineImage(smooth(17, 23), degree=degree)
    rng = np.random.default_rng(0)
    xs = rng.uniform(-3, 26, 200)
    ys = rng.uniform(-3, 20, 200)
    v = s.values(xs, ys)
    dx, dy = s.gradients(xs, ys)
    v2, dx2, dy2 = s.values_and_gradients(xs, ys)
    for k in range(200):
        x, y = float(xs[k]), float(ys[k])
        gx, gy = s.gradient(x, y)
        assert v[k] == s.value(x, y)
        assert (dx[k], dy[k]) == (gx, gy)
        assert (v2[k], dx2[k], dy2[k]) == s.value_and_gradient(x, y)


def test_batch_preserves_shape():
    s = bsi.BSplineImage(smooth(17, 23))
    xs = np.full((4, 5), 3.5)
    ys = np.full((4, 5), 2.5)
    assert s.values(xs, ys).shape == (4, 5)
    assert all(a.shape == (4, 5) for a in s.gradients(xs, ys))
    assert all(a.shape == (4, 5) for a in s.values_and_gradients(xs, ys))


def test_batch_accepts_lists_and_ints():
    s = bsi.BSplineImage(smooth(17, 23))
    expected = s.values(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(s.values([1, 2], [3, 4]), expected)
    np.testing.assert_array_equal(
        s.values(np.array([1, 2], dtype=np.int64), np.array([3, 4], dtype=np.int64)),
        expected,
    )


def test_batch_empty():
    s = bsi.BSplineImage(smooth(17, 23))
    e = np.empty((0,))
    assert s.values(e, e).shape == (0,)
    assert all(a.shape == (0,) for a in s.gradients(e, e))
    assert all(a.shape == (0,) for a in s.values_and_gradients(e, e))


def test_batch_shape_mismatch():
    s = bsi.BSplineImage(smooth(17, 23))
    with pytest.raises(ValueError):
        s.values(np.zeros(3), np.zeros(4))


def test_core_rejects_size_mismatch():
    from bspline_image import _core

    c = _core.Spline64(smooth(5, 5), 3)
    with pytest.raises(ValueError):
        c.values(np.zeros(3), np.zeros(4), np.zeros(3))
    with pytest.raises(ValueError):
        c.values(np.zeros(3), np.zeros(3), np.zeros(2))


def test_batch_float32():
    s = bsi.BSplineImage(smooth(17, 23), dtype=np.float32)
    xs = np.array([1.5, 2.5])
    assert s.values(xs, xs).dtype == np.float32
    assert all(a.dtype == np.float32 for a in s.gradients(xs, xs))
    assert all(a.dtype == np.float32 for a in s.values_and_gradients(xs, xs))


def test_batch_releases_gil():
    from concurrent.futures import ThreadPoolExecutor

    s = bsi.BSplineImage(smooth(17, 23))
    rng = np.random.default_rng(1)
    xs = rng.uniform(0, 22, 200_000)
    ys = rng.uniform(0, 16, 200_000)
    expected = s.values(xs, ys)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: s.values(xs, ys), range(4)))
    for r in results:
        np.testing.assert_array_equal(r, expected)
