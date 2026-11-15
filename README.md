<!-- SPDX-License-Identifier: BSD-3-Clause -->
# bspline_image

[![CI](https://github.com/thibaultbarret/bspline_image/actions/workflows/ci.yml/badge.svg)](https://github.com/thibaultbarret/bspline_image/actions/workflows/ci.yml)
[![Docs](https://github.com/thibaultbarret/bspline_image/actions/workflows/docs.yml/badge.svg)](https://thibaultbarret.github.io/bspline_image/)

`bspline_image` is a header-only C++20 library (namespace `bsi`) for sub-pixel
interpolation of grey-level images with cardinal B-splines. It is designed for synthetic
image generation and digital image correlation (DIC), where the interpolated value and its
gradient are evaluated millions of times.

- Spline degrees 0 to 7, odd and even, chosen at compile time or at run time.
- Value, gradient and Hessian evaluation at arbitrary real coordinates.
- Mirror boundary conditions (same as SciPy `mode='mirror'`).
- No dependency beyond the C++20 standard library; header-only.
- GPU-ready kernels: the lowest layer is allocation-free, exception-free and annotated
  `BSI_HOST_DEVICE`, so it can be reused in CUDA/HIP/SYCL device code.


## Integration

With `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(bspline_image
  GIT_REPOSITORY https://github.com/thibaultbarret/bspline_image.git
  GIT_TAG main)
FetchContent_MakeAvailable(bspline_image)
target_link_libraries(my_app PRIVATE bsi::bspline_image)
```

With an installed copy (`cmake --install build --prefix <prefix>`):

```cmake
find_package(bspline_image 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE bsi::bspline_image)
```

## Example

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

Pixel (i, j) sits at coordinate (x = i, y = j). Second derivatives (`hessian`,
`value_gradient_hessian` and their batch forms `hessians`, `values_gradients_hessians`) are
exact for degrees 0 to 2 but of limited use there (degree 1 has dxx = dyy = 0 and a piecewise
constant dxy); use degree 3 or more,
and 5 for a full Newton iteration. Supported pixel types are `std::uint8_t`,
`std::uint16_t`, `float` and `double`. For a degree chosen at run time, use
`bsi::AnyBSplineImage<T>(view, degree)`; in hot loops, dispatch once with its `visit()`
method.

## Python

Python bindings (nanobind) are available for local use. The module is not published on
PyPI; install it from a checkout, preferably in a virtual environment (needs a C++20
compiler and CMake):

```sh
python3 -m venv .venv && .venv/bin/pip install .
```

```python
import numpy as np
import bspline_image as bsi

img = np.random.default_rng(0).uniform(0, 255, (64, 48))   # shape (height, width)
spline = bsi.BSplineImage(img, degree=3)                   # prefiltered once
print(spline.value(10.25, 20.75))                          # x = column, y = row
values, dx, dy = spline.values_and_gradients(np.array([1.5, 2.5]), np.array([3.5, 4.5]))
xs, ys = np.array([1.5, 2.5]), np.array([3.5, 4.5])
v, dx, dy, dxx, dxy, dyy = spline.values_gradients_hessians(xs, ys)   # degree >= 3 advised
```

Images are indexed `img[y, x]`: x is the column and y the row, so pixel `img[y, x]` sits at
coordinate (x, y). Batch methods take `xs` and `ys` arrays of identical shape. Values match
SciPy: `s.values(xs, ys)` equals
`scipy.ndimage.map_coordinates(img, [ys, xs], order=degree, mode='mirror')` for degrees 0
to 5.

## Degrees

| Degree | Interpolant | Prefilter | Poles | Support (samples) |
|---|---|---|---|---|
| 0 | nearest neighbour | no | 0 | 1 x 1 |
| 1 | bilinear | no | 0 | 2 x 2 |
| 2 to 7 | B-spline | yes | degree / 2 (integer division) | (degree + 1) x (degree + 1) |

For degree n, evaluation reads (n + 1)^2 coefficients.

## Documentation

The documentation (installation, user guide, theory, GPU kernels) is a Sphinx site published
at <https://thibaultbarret.github.io/bspline_image/>; the Doxygen API reference lives under
`/api/` at the same address. To build both locally (needs Doxygen):

```sh
python3 -m venv .venv
.venv/bin/pip install . -r docs/sphinx/requirements.txt
PATH="$PWD/.venv/bin:$PATH" cmake --preset docs
PATH="$PWD/.venv/bin:$PATH" cmake --build --preset docs --target docs
```

The site is written to `build/docs/docs/site/`. Code examples in the pages are executed
at build time.

## Build and test

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBSI_WARNINGS_AS_ERRORS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

In-source builds are refused: always configure into a separate directory.

CMake presets (`dev`, `release`, `python`, `docs`, `sanitize`; list them with
`cmake --list-presets`) provide ready-made configurations, also usable from IDEs:

```sh
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

Each preset builds in `build/<preset>`, so its options can be browsed and edited
interactively with `ccmake build/dev`.

Examples are built by default (`build/examples/synthetic_image`, `build/examples/dic_visit`).
A Release benchmark is available as `bsi_bench`; see `bench/RESULTS.md`.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
