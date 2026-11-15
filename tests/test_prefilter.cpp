// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "bsi/kernel/basis.hpp"
#include "bsi/kernel/mirror.hpp"
#include "bsi/kernel/poles.hpp"
#include "bsi/kernel/prefilter.hpp"
#include "support/degrees.hpp"
#include "support/test_images.hpp"

namespace {

template <class D>
class Prefilter : public ::testing::Test {
protected:
    static constexpr int degree = D::value;
};
TYPED_TEST_SUITE(Prefilter, AllDegrees);

std::vector<double> samples(std::ptrdiff_t n) {
    std::vector<double> f(static_cast<std::size_t>(n));
    for (std::ptrdiff_t k = 0; k < n; ++k) f[static_cast<std::size_t>(k)] = smooth_pixel(k, 0);
    return f;
}

TYPED_TEST(Prefilter, PolesAreRootsOfTheSymbol) {
    constexpr int D = TestFixture::degree;
    if constexpr (D >= 2) {
        static_assert(bsi::pole_count<D> == D / 2);
        for (int i = 0; i < bsi::pole_count<D>; ++i) {
            const double z = bsi::pole<D>(i);
            EXPECT_LT(std::abs(z), 1.0);
            double sum = 0.0;
            for (int k = -D / 2 - 1; k <= D / 2 + 1; ++k) {
                sum += bsi::bspline<double, D>(static_cast<double>(k)) * std::pow(z, k);
            }
            // Normalise by |z|^(D/2+1): the unscaled sum contains terms up to |z|^-(D/2+1)
            // (about 3.5e8 for degree 6), which amplifies rounding errors well beyond 1e-12.
            const double scaled = sum * std::pow(std::abs(z), D / 2 + 1);
            EXPECT_NEAR(scaled, 0.0, 1e-12) << "pole " << i;
        }
    } else {
        static_assert(bsi::pole_count<D> == 0);
    }
}

TYPED_TEST(Prefilter, ConstantStaysConstant) {
    constexpr int D = TestFixture::degree;
    std::vector<double> c(17, 42.0);
    bsi::prefilter_line<D>(c.data(), 17, 1);
    for (const double v : c) EXPECT_NEAR(v, 42.0, 1e-12);
}

TYPED_TEST(Prefilter, InterpolatesSamples) {
    constexpr int D = TestFixture::degree;
    for (const std::ptrdiff_t n : {2, 3, 5, 20, 64}) {
        const std::vector<double> f = samples(n);
        std::vector<double> c = f;
        bsi::prefilter_line<D>(c.data(), n, 1);
        for (std::ptrdiff_t m = 0; m < n; ++m) {
            double s = 0.0;
            for (std::ptrdiff_t k = m - D / 2 - 1; k <= m + D / 2 + 1; ++k) {
                s += c[static_cast<std::size_t>(bsi::mirror_index(k, n))] *
                     bsi::bspline<double, D>(static_cast<double>(m - k));
            }
            EXPECT_NEAR(s, f[static_cast<std::size_t>(m)], 1e-10) << "n=" << n << " m=" << m;
        }
    }
}

TYPED_TEST(Prefilter, RespectsStep) {
    constexpr int D = TestFixture::degree;
    constexpr std::ptrdiff_t step = 3;
    for (const std::ptrdiff_t n : {2, 3, 5, 20, 64}) {
        std::vector<double> ref = samples(n);
        bsi::prefilter_line<D>(ref.data(), n, 1);
        std::vector<double> inter(static_cast<std::size_t>(n * step), -7.0);
        for (std::ptrdiff_t k = 0; k < n; ++k) {
            inter[static_cast<std::size_t>(k * step)] = smooth_pixel(k, 0);
        }
        bsi::prefilter_line<D>(inter.data(), n, step);
        for (std::ptrdiff_t k = 0; k < n; ++k) {
            EXPECT_DOUBLE_EQ(inter[static_cast<std::size_t>(k * step)],
                             ref[static_cast<std::size_t>(k)]);
            if (k + 1 < n) {
                EXPECT_EQ(inter[static_cast<std::size_t>(k * step + 1)], -7.0);
            }
        }
    }
}

TYPED_TEST(Prefilter, SingleSampleUnchanged) {
    constexpr int D = TestFixture::degree;
    double c = 3.25;
    bsi::prefilter_line<D>(&c, 1, 1);
    EXPECT_EQ(c, 3.25);
}

}  // namespace
