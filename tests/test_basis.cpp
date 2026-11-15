// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "bsi/kernel/basis.hpp"
#include "support/degrees.hpp"
#include "support/reference_bspline.hpp"

namespace {
const std::vector<double> xs = {-3.7, -1.5, -0.5, 0.0, 0.25, 0.5, 0.999, 1.0, 2.5, 7.3};
}

template <class DegreeTag>
class Basis : public ::testing::Test {
public:
    static constexpr int D = DegreeTag::value;
};
TYPED_TEST_SUITE(Basis, AllDegrees);

TYPED_TEST(Basis, MatchesCoxDeBoorReference) {
    constexpr int D = TestFixture::D;
    for (int i = -500; i <= 500; ++i) {
        const double u = i * 0.01;
        EXPECT_NEAR((bsi::bspline<double, D>(u)), reference_bspline(D, u), 1e-14) << "u=" << u;
    }
}

TYPED_TEST(Basis, FloatMatchesReference) {
    constexpr int D = TestFixture::D;
    for (int i = -500; i <= 500; ++i) {
        const float uf = static_cast<float>(i * 0.01);
        EXPECT_NEAR(static_cast<double>(bsi::bspline<float, D>(uf)),
                    reference_bspline(D, static_cast<double>(uf)), 1e-5)
            << "u=" << uf;
    }
}

TEST(Basis, FirstIndexConvention) {
    EXPECT_EQ((bsi::first_index<double, 3>(2.7)), 1);
    EXPECT_EQ((bsi::first_index<double, 2>(2.5)), 2);
    EXPECT_EQ((bsi::first_index<double, 0>(2.5)), 3);
    EXPECT_EQ((bsi::first_index<double, 0>(2.49)), 2);
}

TYPED_TEST(Basis, WeightsMatchBasisAndSumToOne) {
    constexpr int D = TestFixture::D;
    for (const double x : xs) {
        double w[D + 1] = {};
        const std::ptrdiff_t i0 = bsi::weights<double, D>(x, w);
        EXPECT_EQ(i0, (bsi::first_index<double, D>(x)));
        double sum = 0.0;
        for (int k = 0; k <= D; ++k) {
            EXPECT_NEAR(w[k], (bsi::bspline<double, D>(x - static_cast<double>(i0 + k))), 1e-14)
                << "x=" << x << " k=" << k;
            sum += w[k];
        }
        EXPECT_NEAR(sum, 1.0, 1e-14) << "x=" << x;
    }
}

TYPED_TEST(Basis, DerivativesMatchFiniteDifferences) {
    constexpr int D = TestFixture::D;
    const double h = 1e-6;
    for (const double x : xs) {
        double w[D + 1] = {};
        double dw[D + 1] = {};
        const std::ptrdiff_t i0 = bsi::weights_and_derivatives<double, D>(x, w, dw);
        EXPECT_EQ(i0, (bsi::first_index<double, D>(x)));
        double sum_w = 0.0;
        double sum_dw = 0.0;
        for (int k = 0; k <= D; ++k) {
            const double u = x - static_cast<double>(i0 + k);
            EXPECT_NEAR(w[k], (bsi::bspline<double, D>(u)), 1e-14);
            sum_w += w[k];
            sum_dw += dw[k];
            if constexpr (D >= 2) {
                const double fd =
                    (bsi::bspline<double, D>(u + h) - bsi::bspline<double, D>(u - h)) / (2.0 * h);
                EXPECT_NEAR(dw[k], fd, 1e-6) << "x=" << x << " k=" << k;
            }
        }
        EXPECT_NEAR(sum_w, 1.0, 1e-14);
        EXPECT_NEAR(sum_dw, 0.0, 1e-13) << "x=" << x;
    }
}

TEST(Basis, LowDegreeDerivativeConventions) {
    double w0[1];
    double dw0[1];
    bsi::weights_and_derivatives<double, 0>(0.3, w0, dw0);
    EXPECT_EQ(w0[0], 1.0);
    EXPECT_EQ(dw0[0], 0.0);

    double w1[2];
    double dw1[2];
    const std::ptrdiff_t i0 = bsi::weights_and_derivatives<double, 1>(3.0, w1, dw1);
    EXPECT_EQ(i0, 3);
    EXPECT_EQ(w1[0], 1.0);
    EXPECT_EQ(w1[1], 0.0);
    EXPECT_EQ(dw1[0], -1.0);
    EXPECT_EQ(dw1[1], 1.0);
}

TEST(Basis, DegreeZeroIsHalfOpen) {
    EXPECT_EQ((bsi::bspline<double, 0>(-0.5)), 1.0);
    EXPECT_EQ((bsi::bspline<double, 0>(0.5)), 0.0);
    EXPECT_EQ((bsi::bspline<double, 0>(0.0)), 1.0);
    EXPECT_EQ((bsi::bspline<double, 0>(-0.51)), 0.0);
}

namespace {
const std::vector<double> second_derivative_xs = {0.13, 0.37, 0.71, 2.29, -1.6};
}

TYPED_TEST(Basis, SecondDerivativesMatchFiniteDifferences) {
    constexpr int D = TestFixture::D;
    if constexpr (D < 2) {
        GTEST_SKIP() << "second derivative is not a function for degree < 2";
    } else {
        const double h = 1e-5;
        for (const double x : second_derivative_xs) {
            double w[D + 1] = {};
            double dw[D + 1] = {};
            double ddw[D + 1] = {};
            double w_ref[D + 1] = {};
            double dw_ref[D + 1] = {};
            const std::ptrdiff_t i0 = bsi::weights_and_second_derivatives<double, D>(x, w, dw, ddw);
            EXPECT_EQ(i0, (bsi::weights_and_derivatives<double, D>(x, w_ref, dw_ref)));
            for (int k = 0; k <= D; ++k) {
                EXPECT_EQ(w[k], w_ref[k]) << "x=" << x << " k=" << k;
                EXPECT_EQ(dw[k], dw_ref[k]) << "x=" << x << " k=" << k;
            }
            // Derivative weights at x +- h, expressed on the same window (shift by i0).
            double wp[D + 1] = {};
            double dwp[D + 1] = {};
            double wm[D + 1] = {};
            double dwm[D + 1] = {};
            const std::ptrdiff_t ip = bsi::weights_and_derivatives<double, D>(x + h, wp, dwp);
            const std::ptrdiff_t im = bsi::weights_and_derivatives<double, D>(x - h, wm, dwm);
            ASSERT_EQ(ip, i0);
            ASSERT_EQ(im, i0);
            for (int k = 0; k <= D; ++k) {
                const double fd = (dwp[k] - dwm[k]) / (2.0 * h);
                EXPECT_NEAR(ddw[k], fd, 1e-6) << "x=" << x << " k=" << k;
            }
        }
    }
}

TYPED_TEST(Basis, SecondDerivativesSumToZero) {
    constexpr int D = TestFixture::D;
    for (const double x : second_derivative_xs) {
        double w[D + 1] = {};
        double dw[D + 1] = {};
        double ddw[D + 1] = {};
        bsi::weights_and_second_derivatives<double, D>(x, w, dw, ddw);
        double sum = 0.0;
        for (int k = 0; k <= D; ++k) sum += ddw[k];
        EXPECT_LT(std::abs(sum), 1e-12) << "x=" << x;
    }
}

TEST(Basis, LowDegreeSecondDerivatives) {
    double w0[1];
    double dw0[1];
    double ddw0[1] = {9.0};
    bsi::weights_and_second_derivatives<double, 0>(0.3, w0, dw0, ddw0);
    EXPECT_EQ(ddw0[0], 0.0);

    double w1[2];
    double dw1[2];
    double ddw1[2] = {9.0, 9.0};
    bsi::weights_and_second_derivatives<double, 1>(0.3, w1, dw1, ddw1);
    EXPECT_EQ(ddw1[0], 0.0);
    EXPECT_EQ(ddw1[1], 0.0);

    double w2[3];
    double dw2[3];
    double ddw2[3] = {};
    bsi::weights_and_second_derivatives<double, 2>(0.3, w2, dw2, ddw2);
    EXPECT_NEAR(ddw2[0], 1.0, 1e-14);
    EXPECT_NEAR(ddw2[1], -2.0, 1e-14);
    EXPECT_NEAR(ddw2[2], 1.0, 1e-14);
}
