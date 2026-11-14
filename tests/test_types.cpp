// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "bsi/types.hpp"

TEST(Types, ImageViewAtRespectsStride) {
    const std::uint16_t buf[] = {1, 2, 3, 99, 4, 5, 6, 99};   // 3x2, stride 4
    bsi::ImageView<const std::uint16_t> v{buf, 3, 2, 4};
    EXPECT_EQ(v.at(0, 1), 4);
    EXPECT_EQ(v.at(2, 1), 6);
}

TEST(Types, CoeffViewAtAddressesPadding) {
    double buf[5 * 5] = {};
    buf[0] = 7.0;                                              // 3x3 with pad 1, stride 5
    bsi::CoeffView<double> c{buf + 5 + 1, 3, 3, 5, 1};
    EXPECT_EQ(c.at(-1, -1), 7.0);
}

TEST(Types, CoeffViewConvertsToConstView) {
    double buf[5 * 5] = {};
    buf[0] = 7.0;
    bsi::CoeffView<double> mutable_view{buf + 5 + 1, 3, 3, 5, 1};
    bsi::CoeffView<const double> const_view = mutable_view;
    EXPECT_EQ(const_view.origin, mutable_view.origin);
    EXPECT_EQ(const_view.width, 3);
    EXPECT_EQ(const_view.height, 3);
    EXPECT_EQ(const_view.stride, 5);
    EXPECT_EQ(const_view.pad, 1);
    EXPECT_EQ(const_view.at(-1, -1), 7.0);
    static_assert(std::is_convertible_v<bsi::CoeffView<double>, bsi::CoeffView<const double>>);
    static_assert(!std::is_convertible_v<bsi::CoeffView<const double>, bsi::CoeffView<double>>);
}

TEST(Types, MaxDegreeIsSeven) { EXPECT_EQ(bsi::max_degree, 7); }

TEST(Types, ImageViewConvertsToConstView) {
    float buf[6] = {1, 2, 3, 4, 5, 6};
    bsi::ImageView<float> mutable_view{buf, 3, 2, 3};
    bsi::ImageView<const float> const_view = mutable_view;
    EXPECT_EQ(const_view.data, buf);
    EXPECT_EQ(const_view.width, 3);
    EXPECT_EQ(const_view.height, 2);
    EXPECT_EQ(const_view.stride, 3);
    static_assert(std::is_convertible_v<bsi::ImageView<float>, bsi::ImageView<const float>>);
    static_assert(!std::is_convertible_v<bsi::ImageView<const float>, bsi::ImageView<float>>);
}
