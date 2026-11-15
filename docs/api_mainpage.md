<!-- SPDX-License-Identifier: BSD-3-Clause -->
bspline_image API reference {#mainpage}
=======================================

The library is organised in three layers:

- the free functions of `bsi/kernel/` (B-spline kernels and prefilters);
- the `bsi::BSplineImage` class template, which owns the coefficients of one image;
- the type-erased `bsi::AnyBSplineImage`, which selects dimension and degree at run time.

Include `bsi/bsi.hpp` to get everything.

[User guide, theory and examples](../index.html)
