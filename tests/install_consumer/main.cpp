// SPDX-License-Identifier: BSD-3-Clause
// Install-consumer smoke test: interpolating at a pixel centre must return the pixel.

#include <cmath>
#include <cstdio>
#include <vector>

#include <bsi/bsi.hpp>

int main() {
    constexpr double kTolerance = 1e-9;
    const std::vector<double> pixels{1, 2, 3, 4, 5, 6, 7, 8, 9};
    const bsi::ImageView<const double> view{pixels.data(), 3, 3, 3};
    const bsi::BSplineImage<double, 3> spline(view);
    const double center = spline.value(1.0, 1.0);
    if (std::abs(center - 5.0) > kTolerance) {
        std::fprintf(stderr, "unexpected value %.12f (expected 5)\n", center);
        return 1;
    }
    return 0;
}
