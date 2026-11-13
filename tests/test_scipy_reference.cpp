// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "bsi/bspline_image.hpp"
#include "data/scipy_reference.inc"
#include "support/degrees.hpp"
#include "support/test_images.hpp"

namespace {

using Degrees0to5 =
    ::testing::Types<Degree<0>, Degree<1>, Degree<2>, Degree<3>, Degree<4>, Degree<5>>;

constexpr double kTolerance = 1e-9;
constexpr double kImageTolerance = 1e-12;

template <class D>
class ScipyReference : public ::testing::Test {};
TYPED_TEST_SUITE(ScipyReference, Degrees0to5);

TYPED_TEST(ScipyReference, MatchesMapCoordinatesMirror) {
    namespace ref = bsi_test::scipy_ref;
    constexpr int degree = TypeParam::value;

    std::vector<double> pixels(static_cast<std::size_t>(ref::ref_width * ref::ref_height));
    for (std::ptrdiff_t j = 0; j < ref::ref_height; ++j) {
        for (std::ptrdiff_t i = 0; i < ref::ref_width; ++i) {
            const auto index = static_cast<std::size_t>(j * ref::ref_width + i);
            ASSERT_NEAR(ref::ref_image[index], smooth_pixel(i, j), kImageTolerance)
                << "image mismatch at " << i << "," << j;
            pixels[index] = ref::ref_image[index];
        }
    }
    const bsi::BSplineImage<double, degree> spline(
        bsi::ImageView<const double>{pixels.data(), ref::ref_width, ref::ref_height, ref::ref_width});

    double max_error = 0.0;
    for (std::size_t k = 0; k < static_cast<std::size_t>(ref::ref_count); ++k) {
        const double expected = ref::ref_values[degree][k];
        const double actual = spline.value(ref::ref_x[k], ref::ref_y[k]);
        max_error = std::max(max_error, std::abs(actual - expected));
        EXPECT_NEAR(actual, expected, kTolerance)
            << "degree " << degree << " point (" << ref::ref_x[k] << ", " << ref::ref_y[k] << ")";
    }
    std::cout << "degree " << degree << " max abs error " << max_error << "\n";
}

}  // namespace
