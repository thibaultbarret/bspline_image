# SPDX-License-Identifier: BSD-3-Clause
import numpy as np

import bspline_image as bsi


def smooth(h, w):  # same formula as tests/support/test_images.hpp
    j, i = np.mgrid[0:h, 0:w].astype(np.float64)
    return 100 + 50 * np.sin(0.7 * i + 0.3 * j) + 30 * np.cos(0.45 * j - 0.2 * i)


def test_version():
    assert bsi.__version__ == "0.1.0"


def test_properties():
    s = bsi.BSplineImage(smooth(17, 23))
    assert s.width == 23
    assert s.height == 17
    assert s.degree == 3
    assert s.dtype == np.dtype(np.float64)


def test_interpolates_pixels_degree3():
    img = smooth(17, 23)
    s = bsi.BSplineImage(img, degree=3)
    for j in range(17):
        for i in range(23):
            assert abs(s.value(i, j) - img[j, i]) < 1e-9
