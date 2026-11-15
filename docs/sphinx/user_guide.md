<!-- SPDX-License-Identifier: BSD-3-Clause -->
# User guide

## The three layers

- **Layer 0** (`bsi/kernel/`) holds free functions on plain views, usable on host and
  device (`BSI_HOST_DEVICE`); see {doc}`gpu`.
- **Layer 1** is {bsi}`bsi::BSplineImage`, an owning interpolator whose degree is a
  compile-time parameter.
- **Layer 2** is {bsi}`bsi::AnyBSplineImage`, a facade with a runtime degree and a
  `visit()` escape hatch for hot loops.

The Python class `bspline_image.BSplineImage` wraps layer 2
({bsi}`bsi::AnyBSplineImage`) behind a runtime-degree interface.

## Degrees and boundaries

Degrees 0 to 7 are supported, odd and even. For degree $n$, evaluation reads
$(n+1)^2$ coefficients.

| Degree | Interpolant | Prefilter | Poles | Support (samples) |
|---|---|---|---|---|
| 0 | nearest neighbour | no | 0 | 1 x 1 |
| 1 | bilinear | no | 0 | 2 x 2 |
| 2 to 7 | B-spline | yes | degree / 2 (integer division) | (degree + 1) x (degree + 1) |

The prefilter turns the samples into B-spline coefficients once, at construction (see
{doc}`theory`). Boundaries are mirrored: coordinates outside the image are folded back
by symmetry.

## Pixels and coordinates

Supported pixel types are `std::uint8_t`, `std::uint16_t`, `float` and `double`. A
{bsi}`bsi::ImageView` describes row-major data with a width, a height and a stride.

Pixel (i, j) sits at coordinate (x = i, y = j): x is the column and y the row. In
numpy, `image[y, x]` sits at (x, y).

## Evaluating

In C++:

```cpp
#include <vector>
#include "bsi/bsi.hpp"

int main() {
    std::vector<double> pixels(64 * 64, 0.0);  // row-major, width 64, height 64
    const bsi::ImageView<const double> view{pixels.data(), 64, 64, 64};  // data, w, h, stride

    const bsi::BSplineImage<double, 3> spline(view);          // cubic, prefiltered once
    const auto vg = spline.value_and_gradient(10.25, 20.75);  // sub-pixel position
    // vg.value, vg.dx, vg.dy
    const auto h = spline.value_gradient_hessian(10.25, 20.75);     // adds second derivatives
    // h.value, h.dx, h.dy, h.dxx, h.dxy, h.dyy
}
```

For a degree chosen at run time, use `bsi::AnyBSplineImage<T>(view, degree)`.

In Python:

```python
import numpy as np
import bspline_image as bsi

img = np.random.default_rng(0).uniform(0, 255, (64, 48))   # shape (height, width)
spline = bsi.BSplineImage(img, degree=3)                   # prefiltered once
print(spline.value(10.25, 20.75))                          # x = column, y = row
xs, ys = np.array([1.5, 2.5]), np.array([3.5, 4.5])
values, dx, dy = spline.values_and_gradients(xs, ys)
v, dx, dy, dxx, dxy, dyy = spline.values_gradients_hessians(xs, ys)
```

The point methods are `value`, `gradient`, `value_and_gradient`, `hessian` and
`value_gradient_hessian`. The batch methods are `values`, `gradients`,
`values_and_gradients`, `hessians` and `values_gradients_hessians`; in Python `xs` and
`ys` may have any shape, provided it is identical for both. In C++ the batch methods take
`std::span`s of identical length. Values match SciPy: `s.values(xs, ys)` equals
`scipy.ndimage.map_coordinates(img, [ys, xs], order=degree, mode='mirror')` for degrees 0
to 5.

Second derivatives are exact for degrees 0 to 2 but of limited use there (degree 1 has
$\partial_{xx} = \partial_{yy} = 0$ and a piecewise-constant $\partial_{xy}$). Use degree
3 or more, and 5 for a full Newton iteration.

## float or double

In C++ the evaluation type is the first template parameter. In Python,
`BSplineImage(img, degree=3, dtype=np.float64)` is the default; with `np.float32` the
coefficients are stored in single precision and the coordinates are rounded to `float32`
too, which limits precision to about $10^{-3}$ px around $x = 4000$. Use `float64` for
large images.

## Threads

All evaluation methods are `const`, so one interpolator can be read from several threads
at once. Parallelise over subsets or points in the caller.

(user-guide-dic)=
## Digital image correlation

In DIC, a subset of the reference image is matched to the deformed image by minimising a
correlation criterion over sub-pixel displacements. The criterion needs grey levels and
gradients at non-integer positions, so interpolation error becomes measurement error: a
systematic **interpolation bias** that depends on the sub-pixel part of the displacement.

### Choosing the degree

The bias decreases quickly as the degree grows, because high-degree splines approximate
the ideal band-limited interpolator better {cite}`thevenaz2000interpolation`. Degree 1
(bilinear) is cheap but biased and gives piecewise-constant gradients; degree 3 is a
common compromise; degrees 5 and 7 further reduce the bias on well-resolved speckle
patterns. Odd degrees (3, 5, 7) are the usual choice. Even degrees are supported (the
index convention is $\lfloor x + \tfrac12 \rfloor$), but offer no advantage in typical
DIC.

### Cost

The window grows as $(n+1)^2$ coefficients, so cost rises with the degree. See
[`bench/RESULTS.md`](https://github.com/thibaultbarret/bspline_image/blob/main/bench/RESULTS.md) for measured numbers: per-point time roughly doubles from degree 3 to
degree 5 and rises again towards degree 7, while degree 1 is the cheapest. Prefiltering
is paid once per image, so it is negligible compared with millions of evaluations. The
numbers in [`bench/RESULTS.md`](https://github.com/thibaultbarret/bspline_image/blob/main/bench/RESULTS.md) come from one noisy laptop; treat them qualitatively.

### The visit pattern

{bsi}`bsi::AnyBSplineImage` selects the degree at runtime, but calling `value()` through
it dispatches on every call. For a hot loop, dispatch **once** with `visit()` and run the
loop on the concrete {bsi}`bsi::BSplineImage` type, so the compiler can inline and unroll
the fixed-length evaluation:

```{literalinclude} ../../examples/dic_visit.cpp
:language: cpp
```

The callable passed to `visit()` must accept every supported degree and return the same
type for all of them. The batch methods (`values`, `gradients`, `values_and_gradients`)
are the alternative when points are already in arrays; they dispatch once per call.

### Hessian for Newton schemes

For a full Newton scheme, `value_gradient_hessian` and `values_gradients_hessians`
return $I$, $\nabla I$ and $\nabla^2 I$ in one pass. Use degree 3 or more (5 is
recommended): lower degrees have a trivial or discontinuous Hessian.

## Where next

- {doc}`theory` explains the model, the prefilter and the evaluation.
- {doc}`gpu` explains how to reuse the layer-0 kernels in device code.
