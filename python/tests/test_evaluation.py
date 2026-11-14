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
