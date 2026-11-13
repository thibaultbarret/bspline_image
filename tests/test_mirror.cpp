// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "bsi/kernel/mirror.hpp"

namespace {

using bsi::fold_mirror;
using bsi::mirror_index;

TEST(Mirror, IndexSequenceN4) {
    const std::ptrdiff_t expected[] = {3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2};
    for (std::ptrdiff_t i = -3; i <= 8; ++i) {
        EXPECT_EQ(mirror_index(i, 4), expected[i + 3]) << "i=" << i;
    }
}

TEST(Mirror, IndexSingleSample) {
    for (const std::ptrdiff_t i : {-5, 0, 7}) {
        EXPECT_EQ(mirror_index(i, 1), 0);
    }
}

TEST(Mirror, IndexFarAway) {
    EXPECT_EQ(mirror_index(1'000'000'007, 5), mirror_index(1'000'000'007 % 8, 5));
}

TEST(Mirror, FoldMatchesIndexAtIntegers) {
    for (std::ptrdiff_t i = -20; i <= 20; ++i) {
        EXPECT_EQ(fold_mirror(double(i), 6).x, double(mirror_index(i, 6))) << "i=" << i;
        EXPECT_EQ(fold_mirror(float(i), 6).x, float(mirror_index(i, 6))) << "i=" << i;
    }
}

TEST(Mirror, FoldSignFlipsOnReflection) {
    const auto a = fold_mirror(-1.25, 6);
    EXPECT_DOUBLE_EQ(a.x, 1.25);
    EXPECT_DOUBLE_EQ(a.sign, -1.0);
    const auto b = fold_mirror(6.5, 6);
    EXPECT_DOUBLE_EQ(b.x, 3.5);
    EXPECT_DOUBLE_EQ(b.sign, -1.0);
    const auto c = fold_mirror(11.25, 6);
    EXPECT_DOUBLE_EQ(c.x, 1.25);
    EXPECT_DOUBLE_EQ(c.sign, 1.0);
    const auto f = fold_mirror(-1.25f, 6);
    EXPECT_FLOAT_EQ(f.x, 1.25f);
    EXPECT_FLOAT_EQ(f.sign, -1.0f);
}

TEST(Mirror, FoldHugeCoordinateIsFinite) {
    for (const double v : {1e15, -1e9}) {
        const auto r = fold_mirror(v, 6);
        EXPECT_TRUE(std::isfinite(r.x));
        EXPECT_GE(r.x, 0.0);
        EXPECT_LE(r.x, 5.0);
    }
}

TEST(Mirror, FoldPreservesNaN) {
    EXPECT_TRUE(std::isnan(fold_mirror(std::numeric_limits<double>::quiet_NaN(), 6).x));
    EXPECT_TRUE(std::isnan(fold_mirror(std::numeric_limits<float>::quiet_NaN(), 6).x));
}

TEST(Mirror, FoldSingleSample) {
    const auto r = fold_mirror(3.7, 1);
    EXPECT_DOUBLE_EQ(r.x, 0.0);
    EXPECT_DOUBLE_EQ(r.sign, 1.0);
}

TEST(Mirror, FillPaddingMirrorsInterior) {
    constexpr std::ptrdiff_t w = 3, h = 2, pad = 2;
    constexpr std::ptrdiff_t stride = w + 2 * pad;
    std::vector<double> storage(static_cast<std::size_t>(stride * (h + 2 * pad)), -1.0);
    bsi::CoeffView<double> c{storage.data() + pad * stride + pad, w, h, stride, pad};
    for (std::ptrdiff_t j = 0; j < h; ++j) {
        for (std::ptrdiff_t i = 0; i < w; ++i) c.at(i, j) = double(10 * j + i + 1);
    }
    bsi::fill_mirror_padding(c);
    EXPECT_EQ(c.at(-2, -1), c.at(2, 1));
    EXPECT_EQ(c.at(4, 2), c.at(0, 0));
    for (std::ptrdiff_t j = -pad; j < h + pad; ++j) {
        for (std::ptrdiff_t i = -pad; i < w + pad; ++i) {
            EXPECT_EQ(c.at(i, j), c.at(mirror_index(i, w), mirror_index(j, h)))
                << "i=" << i << " j=" << j;
        }
    }
}

}  // namespace
