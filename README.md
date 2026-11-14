<!-- SPDX-License-Identifier: BSD-3-Clause -->
# bspline_image

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![Docs](https://github.com/OWNER/REPO/actions/workflows/docs.yml/badge.svg)](https://OWNER.github.io/REPO/)

`bspline_image` is a header-only C++20 library (namespace `bsi`) for sub-pixel
interpolation of grey-level images with cardinal B-splines. It is designed for synthetic
image generation and digital image correlation (DIC), where the interpolated value and its
gradient are evaluated millions of times.

- Spline degrees 0 to 7, odd and even, chosen at compile time or at run time.
- Value and gradient evaluation at arbitrary real coordinates.
- Mirror boundary conditions (same as SciPy `mode='mirror'`).
- No dependency beyond the C++20 standard library; header-only.
- GPU-ready kernels: the lowest layer is allocation-free, exception-free and annotated
  `BSI_HOST_DEVICE`, so it can be reused in CUDA/HIP/SYCL device code.

> `OWNER/REPO` in the links above is a placeholder for the GitHub repository.

## Integration

With `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(bspline_image
  GIT_REPOSITORY https://github.com/OWNER/REPO.git
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
}
```

Pixel (i, j) sits at coordinate (x = i, y = j). Supported pixel types are `std::uint8_t`,
`std::uint16_t`, `float` and `double`. For a degree chosen at run time, use
`bsi::AnyBSplineImage<T>(view, degree)`; in hot loops, dispatch once with its `visit()`
method.

## Degrees

| Degree | Interpolant | Prefilter | Poles | Support (samples) |
|---|---|---|---|---|
| 0 | nearest neighbour | no | 0 | 1 x 1 |
| 1 | bilinear | no | 0 | 2 x 2 |
| 2 to 7 | B-spline | yes | degree / 2 (integer division) | (degree + 1) x (degree + 1) |

For degree n, evaluation reads (n + 1)^2 coefficients.

## Documentation

API reference and guides (theory, DIC usage, GPU kernels) are built with Doxygen and
published at <https://OWNER.github.io/REPO/>. To build them locally:

```sh
cmake -S . -B build -G Ninja -DBSI_BUILD_DOCS=ON
cmake --build build --target docs
```

## Build and test

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBSI_WARNINGS_AS_ERRORS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Examples are built by default (`build/examples/synthetic_image`, `build/examples/dic_visit`).
A Release benchmark is available as `bsi_bench`; see `bench/RESULTS.md`.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
