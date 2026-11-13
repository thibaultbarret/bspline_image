// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <gtest/gtest.h>

#include <type_traits>

/// @brief Compile-time spline degree tag for typed tests.
/// @tparam D Spline degree.
template <int D>
using Degree = std::integral_constant<int, D>;

/// @brief All supported degrees (0 to 7), for use with TYPED_TEST_SUITE.
using AllDegrees = ::testing::Types<Degree<0>, Degree<1>, Degree<2>, Degree<3>,
                                    Degree<4>, Degree<5>, Degree<6>, Degree<7>>;
