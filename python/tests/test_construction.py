# SPDX-License-Identifier: BSD-3-Clause
import gc

import numpy as np
import pytest

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


@pytest.mark.parametrize("pix", [np.uint8, np.uint16, np.float32, np.float64])
def test_pixel_dtypes(pix):
    img = np.clip(smooth(17, 23), 0, 255).astype(pix)
    s = bsi.BSplineImage(img, degree=3)
    for j in range(17):
        for i in range(23):
            assert abs(s.value(i, j) - float(img[j, i])) < 1e-9


def test_row_strided_view_no_copy_needed():
    big = smooth(40, 30)
    v = big[::2, 1:]
    got = bsi.BSplineImage(v)
    ref = bsi.BSplineImage(np.ascontiguousarray(v))
    for x, y in [(0.3, 0.7), (5.5, 4.25), (20.1, 10.9)]:
        assert got.value(x, y) == ref.value(x, y)


def test_column_strided_and_fortran_views():
    img = smooth(17, 23)
    big = smooth(17, 46)
    for view, base in [(big[:, ::2], big[:, ::2]), (np.asfortranarray(img), img)]:
        got = bsi.BSplineImage(view)
        ref = bsi.BSplineImage(np.ascontiguousarray(base))
        for x, y in [(0.3, 0.7), (5.5, 4.25), (20.1, 10.9)]:
            assert got.value(x, y) == ref.value(x, y)


@pytest.mark.parametrize("bad", [np.int32, np.bool_, ">f8"])
def test_other_dtypes_converted(bad):
    img = np.clip(smooth(17, 23), 0, 1).astype(bad) if bad is np.bool_ else smooth(17, 23).astype(bad)
    got = bsi.BSplineImage(img)
    ref = bsi.BSplineImage(img.astype(np.float64))
    for x, y in [(0.3, 0.7), (5.5, 4.25), (20.1, 10.9)]:
        assert abs(got.value(x, y) - ref.value(x, y)) < 1e-12


def test_image_lifetime():
    img = smooth(17, 23)
    s = bsi.BSplineImage(img)
    before = s.value(3.5, 2.25)
    del img
    gc.collect()
    assert s.value(3.5, 2.25) == before


def test_float32_spline():
    img = smooth(17, 23)
    s = bsi.BSplineImage(img, degree=3, dtype=np.float32)
    assert s.dtype == np.dtype(np.float32)
    for j in range(17):
        for i in range(23):
            assert abs(s.value(i, j) - img[j, i]) < 2e-3


@pytest.mark.parametrize(
    "args",
    [
        dict(img=np.zeros((2, 3, 4))),
        dict(img=np.zeros((0, 5))),
        dict(degree=-1),
        dict(degree=8),
        dict(dtype=np.int16),
    ],
)
def test_invalid_arguments(args):
    kwargs = dict(img=smooth(8, 8))
    kwargs.update(args)
    with pytest.raises(ValueError):
        bsi.BSplineImage(**kwargs)


@pytest.mark.parametrize("flip", [np.s_[::-1], np.s_[::-1, ::-1]])
def test_flipped_views_match_contiguous_copy(flip):
    view = smooth(17, 23)[flip]
    got = bsi.BSplineImage(view)
    ref = bsi.BSplineImage(np.ascontiguousarray(view))
    for x, y in [(0.3, 0.7), (5.5, 4.25), (20.1, 10.9)]:
        assert got.value(x, y) == ref.value(x, y)


def test_broadcast_zero_row_stride():
    row = smooth(1, 10)[0]
    s = bsi.BSplineImage(np.broadcast_to(row, (6, 10)))
    for j in range(6):
        for i in range(10):
            assert abs(s.value(i, j) - row[i]) < 1e-9


@pytest.mark.parametrize(
    "make",
    [
        lambda row: row[None, :],
        lambda row: np.atleast_2d(row),
        lambda row: row[:, None],
    ],
    ids=["row_none", "atleast_2d", "col_none"],
)
def test_zero_stride_single_row_or_column(make):
    row = np.linspace(1.0, 9.0, 6)
    img = make(row)
    s = bsi.BSplineImage(img)
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            assert s.value(i, j) == pytest.approx(img[j, i], abs=1e-9)


@pytest.mark.parametrize("bad", [3.0, None, True, 8, -1, "3"])
def test_invalid_degree_message(bad):
    with pytest.raises(ValueError, match="degree"):
        bsi.BSplineImage(smooth(5, 5), degree=bad)


def test_unaligned_image_is_copied():
    h, w = 5, 6
    ref = smooth(h, w)
    buf = bytearray(1 + ref.nbytes)
    buf[1:] = ref.tobytes()
    img = np.frombuffer(buf, dtype=np.float64, offset=1, count=h * w).reshape(h, w)
    assert not img.flags.aligned
    a = bsi.BSplineImage(img)
    b = bsi.BSplineImage(ref)
    assert a.value(2.3, 1.7) == b.value(2.3, 1.7)


def test_core_rejects_wrong_output_dtype():
    from bspline_image import _core

    core = _core.Spline64(smooth(5, 5), 3)
    xs = np.zeros(4)
    with pytest.raises(TypeError):
        core.values(xs, xs, np.empty(4, np.float32))
