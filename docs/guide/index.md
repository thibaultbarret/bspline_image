<!-- SPDX-License-Identifier: BSD-3-Clause -->
# bspline_image {#mainpage}

`bspline_image` is a header-only C++20 library (namespace `bsi`) that interpolates
grey-level images at sub-pixel positions with cardinal B-splines. It targets synthetic
image generation and digital image correlation (DIC), where both the interpolated value
and its gradient are needed millions of times.

- **Three layers.** Layer 0 (`bsi/kernel/`) holds free functions on plain views, usable on
  host and device (`BSI_HOST_DEVICE`). Layer 1 is `bsi::BSplineImage`, an owning
  interpolator with a compile-time degree. Layer 2 is `bsi::AnyBSplineImage`, a facade with
  a runtime degree and a `visit()` escape hatch for hot loops.
- **Degrees 0 to 7**, odd and even.
- **Mirror boundaries**: coordinates outside the image are folded back by symmetry.
- **No dependency** beyond the C++20 standard library.

Pixels are `std::uint8_t`, `std::uint16_t`, `float` or `double`. Pixel (i, j) sits at
coordinate (x = i, y = j).

## Integration

With `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(bspline_image GIT_REPOSITORY https://github.com/OWNER/REPO.git GIT_TAG main)
FetchContent_MakeAvailable(bspline_image)
target_link_libraries(my_app PRIVATE bsi::bspline_image)
```

Or after `cmake --install`:

```cmake
find_package(bspline_image REQUIRED)
target_link_libraries(my_app PRIVATE bsi::bspline_image)
```

## Example

```cpp
#include <vector>
#include "bsi/bsi.hpp"

std::vector<double> pixels(64 * 64, 0.0);  // row-major, width 64, height 64
const bsi::ImageView<const double> view{pixels.data(), 64, 64, 64};  // data, w, h, stride

const bsi::BSplineImage<double, 3> spline(view);           // cubic, prefiltered once
const auto vg = spline.value_and_gradient(10.25, 20.75);   // sub-pixel position
// vg.value, vg.dx, vg.dy
```

## Where next

- @ref theory explains the model, the prefilter and the evaluation.
- @ref dic discusses the degree choice and the `visit` pattern.
- @ref gpu explains how to reuse the layer-0 kernels in device code.
