// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "bsi/kernel/evaluate.hpp"
#include "bsi/kernel/mirror.hpp"
#include "support/degrees.hpp"
#include "support/test_images.hpp"

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// Padded W x H coefficient storage with mirror padding already filled.
template <class T, int D>
struct PaddedCoefficients {
    template <class F>
    PaddedCoefficients(std::ptrdiff_t w, std::ptrdiff_t h, F&& fill)
        : pad(bsi::required_pad<D>),
          stride(w + 2 * pad),
          storage(static_cast<std::size_t>(stride * (h + 2 * pad)), T(0)),
          view{storage.data() + pad * stride + pad, w, h, stride, pad} {
        for (std::ptrdiff_t j = 0; j < h; ++j) {
            for (std::ptrdiff_t i = 0; i < w; ++i) view.at(i, j) = static_cast<T>(fill(i, j));
        }
        bsi::fill_mirror_padding(view);
    }
    PaddedCoefficients(const PaddedCoefficients&) = delete;
    PaddedCoefficients& operator=(const PaddedCoefficients&) = delete;

    std::ptrdiff_t pad;
    std::ptrdiff_t stride;
    std::vector<T> storage;
    bsi::CoeffView<T> view;
};

template <class DegreeTag>
class Evaluate : public ::testing::Test {
  protected:
    static constexpr int D = DegreeTag::value;
};
TYPED_TEST_SUITE(Evaluate, AllDegrees);

TYPED_TEST(Evaluate, ConstantCoefficientsGiveConstant) {
    constexpr int D = TestFixture::D;
    PaddedCoefficients<double, D> c(16, 12, [](std::ptrdiff_t, std::ptrdiff_t) { return 3.0; });
    for (int p = 0; p < 20; ++p) {
        const double x = -9.3 + 1.37 * p;
        const double y = -7.1 + 1.11 * p;
        const auto r = bsi::evaluate_with_gradient<double, D>(c.view, x, y);
        EXPECT_NEAR(r.value, 3.0, 1e-14) << "p=" << p;
        EXPECT_NEAR(r.dx, 0.0, 1e-13) << "p=" << p;
        EXPECT_NEAR(r.dy, 0.0, 1e-13) << "p=" << p;
    }
}

TYPED_TEST(Evaluate, LinearCoefficientsReproduceX) {
    constexpr int D = TestFixture::D;
    if constexpr (D >= 1) {
        PaddedCoefficients<double, D> c(
            32, 8, [](std::ptrdiff_t i, std::ptrdiff_t) { return static_cast<double>(i); });
        for (double x = 8.0; x <= 24.0; x += 0.37) {
            const auto r = bsi::evaluate_with_gradient<double, D>(c.view, x, 3.3);
            EXPECT_NEAR(r.value, x, 1e-12) << "x=" << x;
            EXPECT_NEAR(r.dx, 1.0, 1e-12) << "x=" << x;
            EXPECT_NEAR(r.dy, 0.0, 1e-12) << "x=" << x;
        }
    }
}

TYPED_TEST(Evaluate, MirrorSymmetryOutside) {
    constexpr int D = TestFixture::D;
    constexpr std::ptrdiff_t W = 16;
    PaddedCoefficients<double, D> c(W, 10, smooth_pixel);
    for (double x = 0.13; x < 14.0; x += 0.91) {
        const auto a = bsi::evaluate_with_gradient<double, D>(c.view, x, 4.2);
        const auto b = bsi::evaluate_with_gradient<double, D>(c.view, -x, 4.2);
        const auto d = bsi::evaluate_with_gradient<double, D>(c.view, 2.0 * (W - 1) - x, 4.2);
        EXPECT_NEAR(b.value, a.value, 1e-12) << "x=" << x;
        EXPECT_NEAR(b.dx, -a.dx, 1e-12) << "x=" << x;
        EXPECT_NEAR(b.dy, a.dy, 1e-12) << "x=" << x;
        EXPECT_NEAR(d.value, a.value, 1e-12) << "x=" << x;
        EXPECT_NEAR(d.dx, -a.dx, 1e-12) << "x=" << x;
    }
}

TYPED_TEST(Evaluate, MirrorSymmetryInY) {
    constexpr int D = TestFixture::D;
    PaddedCoefficients<double, D> c(12, 10, smooth_pixel);
    for (double y = 0.13; y < 8.0; y += 0.77) {
        const auto a = bsi::evaluate_with_gradient<double, D>(c.view, 3.9, y);
        const auto b = bsi::evaluate_with_gradient<double, D>(c.view, 3.9, -y);
        EXPECT_NEAR(b.value, a.value, 1e-12) << "y=" << y;
        EXPECT_NEAR(b.dx, a.dx, 1e-12) << "y=" << y;
        EXPECT_NEAR(b.dy, -a.dy, 1e-12) << "y=" << y;
    }
}

TYPED_TEST(Evaluate, NonFiniteCoordinatesGiveNaN) {
    constexpr int D = TestFixture::D;
    PaddedCoefficients<double, D> c(8, 8, smooth_pixel);
    for (const double bad : {kNaN, kInf, -kInf}) {
        const auto rx = bsi::evaluate_with_gradient<double, D>(c.view, bad, 2.5);
        const auto ry = bsi::evaluate_with_gradient<double, D>(c.view, 2.5, bad);
        for (const auto& r : {rx, ry}) {
            EXPECT_TRUE(std::isnan(r.value));
            EXPECT_TRUE(std::isnan(r.dx));
            EXPECT_TRUE(std::isnan(r.dy));
        }
        EXPECT_TRUE(std::isnan(bsi::evaluate<double, D>(c.view, bad, 2.5)));
        EXPECT_TRUE(std::isnan(bsi::evaluate<double, D>(c.view, 2.5, bad)));
    }
}

TYPED_TEST(Evaluate, ValueMatchesValueOfEvaluateWithGradient) {
    constexpr int D = TestFixture::D;
    PaddedCoefficients<double, D> c(20, 15, smooth_pixel);
    for (int p = 0; p < 50; ++p) {
        const double x = -3.1 + 0.53 * p;
        const double y = -2.7 + 0.41 * p;
        const double v = bsi::evaluate<double, D>(c.view, x, y);
        const double g = bsi::evaluate_with_gradient<double, D>(c.view, x, y).value;
        EXPECT_EQ(std::memcmp(&v, &g, sizeof v), 0) << "p=" << p;
    }
}

TYPED_TEST(Evaluate, FloatPathMatchesDouble) {
    constexpr int D = TestFixture::D;
    PaddedCoefficients<float, D> cf(16, 12, smooth_pixel);
    PaddedCoefficients<double, D> cd(16, 12, smooth_pixel);
    for (int p = 0; p < 20; ++p) {
        const float x = -2.3f + 0.91f * static_cast<float>(p);
        const float y = -1.9f + 0.77f * static_cast<float>(p);
        const auto f = bsi::evaluate_with_gradient<float, D>(cf.view, x, y);
        const auto d = bsi::evaluate_with_gradient<double, D>(cd.view, double(x), double(y));
        EXPECT_NEAR(double(f.value), d.value, 1e-3) << "p=" << p;
        EXPECT_NEAR(double(f.dx), d.dx, 1e-2) << "p=" << p;
        EXPECT_NEAR(double(f.dy), d.dy, 1e-2) << "p=" << p;
        EXPECT_TRUE(std::isnan(
            bsi::evaluate<float, D>(cf.view, std::numeric_limits<float>::infinity(), y)));
    }
}

TYPED_TEST(Evaluate, ValueAndGradientOfHessianBitIdentical) {
    constexpr int D = TestFixture::D;
    PaddedCoefficients<double, D> c(20, 15, smooth_pixel);
    for (int p = 0; p < 50; ++p) {
        const double x = -3.1 + 0.53 * p;
        const double y = -2.7 + 0.41 * p;
        const auto g = bsi::evaluate_with_gradient<double, D>(c.view, x, y);
        const auto h = bsi::evaluate_with_hessian<double, D>(c.view, x, y);
        EXPECT_EQ(h.value, g.value) << "p=" << p;
        EXPECT_EQ(h.dx, g.dx) << "p=" << p;
        EXPECT_EQ(h.dy, g.dy) << "p=" << p;
    }
}

TYPED_TEST(Evaluate, HessianMatchesFiniteDifferencesOfGradient) {
    constexpr int D = TestFixture::D;
    if constexpr (D < 3) {
        GTEST_SKIP() << "Hessian is piecewise constant or zero for degree < 3";
    } else {
        constexpr double h = 1e-5;
        PaddedCoefficients<double, D> c(20, 15, smooth_pixel);
        const auto grad = [&](double x, double y) {
            return bsi::evaluate_with_gradient<double, D>(c.view, x, y);
        };
        const double pts[][2] = {{3.37, 4.21}, {9.71, 7.43}, {15.13, 2.67}, {-1.37, 3.29},
                                 {21.43, 16.71}, {6.29, -2.37}, {5.61, 18.13}};
        for (const auto& pt : pts) {
            const double x = pt[0];
            const double y = pt[1];
            const auto r = bsi::evaluate_with_hessian<double, D>(c.view, x, y);
            const double fd_xx = (grad(x + h, y).dx - grad(x - h, y).dx) / (2 * h);
            const double fd_yy = (grad(x, y + h).dy - grad(x, y - h).dy) / (2 * h);
            const double fd_xy_a = (grad(x, y + h).dx - grad(x, y - h).dx) / (2 * h);
            const double fd_xy_b = (grad(x + h, y).dy - grad(x - h, y).dy) / (2 * h);
            EXPECT_NEAR(r.dxx, fd_xx, 1e-6) << "x=" << x << " y=" << y;
            EXPECT_NEAR(r.dyy, fd_yy, 1e-6) << "x=" << x << " y=" << y;
            EXPECT_NEAR(r.dxy, fd_xy_a, 1e-6) << "x=" << x << " y=" << y;
            EXPECT_NEAR(r.dxy, fd_xy_b, 1e-6) << "x=" << x << " y=" << y;
        }
    }
}

TYPED_TEST(Evaluate, HessianMirrorSymmetry) {
    constexpr int D = TestFixture::D;
    constexpr std::ptrdiff_t W = 16;
    constexpr std::ptrdiff_t H = 10;
    PaddedCoefficients<double, D> c(W, H, smooth_pixel);
    for (double x = 0.13; x < 14.0; x += 0.91) {
        const auto a = bsi::evaluate_with_hessian<double, D>(c.view, x, 4.2);
        const auto b = bsi::evaluate_with_hessian<double, D>(c.view, -x, 4.2);
        const auto d = bsi::evaluate_with_hessian<double, D>(c.view, 2.0 * (W - 1) - x, 4.2);
        EXPECT_NEAR(b.dxx, a.dxx, 1e-12) << "x=" << x;
        EXPECT_NEAR(b.dyy, a.dyy, 1e-12) << "x=" << x;
        EXPECT_NEAR(b.dxy, -a.dxy, 1e-12) << "x=" << x;
        EXPECT_NEAR(d.dxx, a.dxx, 1e-12) << "x=" << x;
        EXPECT_NEAR(d.dyy, a.dyy, 1e-12) << "x=" << x;
        EXPECT_NEAR(d.dxy, -a.dxy, 1e-12) << "x=" << x;
    }
    for (double y = 0.13; y < 8.0; y += 0.77) {
        const auto a = bsi::evaluate_with_hessian<double, D>(c.view, 3.9, y);
        const auto b = bsi::evaluate_with_hessian<double, D>(c.view, 3.9, -y);
        const auto d = bsi::evaluate_with_hessian<double, D>(c.view, 3.9, 2.0 * (H - 1) - y);
        EXPECT_NEAR(b.dxx, a.dxx, 1e-12) << "y=" << y;
        EXPECT_NEAR(b.dyy, a.dyy, 1e-12) << "y=" << y;
        EXPECT_NEAR(b.dxy, -a.dxy, 1e-12) << "y=" << y;
        EXPECT_NEAR(d.dxx, a.dxx, 1e-12) << "y=" << y;
        EXPECT_NEAR(d.dyy, a.dyy, 1e-12) << "y=" << y;
        EXPECT_NEAR(d.dxy, -a.dxy, 1e-12) << "y=" << y;
    }
}

TYPED_TEST(Evaluate, HessianContinuousAcrossKnots) {
    constexpr int D = TestFixture::D;
    if constexpr (D < 3) {
        GTEST_SKIP() << "Hessian is discontinuous or zero for degree < 3";
    } else {
        constexpr double eps = 1e-7;
        PaddedCoefficients<double, D> c(16, 12, smooth_pixel);
        // Knots are at integers for odd degrees and at half-integers for even degrees.
        for (const double knot : {2.0, 2.5}) {
            const auto lo = bsi::evaluate_with_hessian<double, D>(c.view, knot - eps, 4.3);
            const auto hi = bsi::evaluate_with_hessian<double, D>(c.view, knot + eps, 4.3);
            EXPECT_NEAR(lo.dxx, hi.dxx, 1e-5) << "knot=" << knot;
            EXPECT_NEAR(lo.dxy, hi.dxy, 1e-5) << "knot=" << knot;
            EXPECT_NEAR(lo.dyy, hi.dyy, 1e-5) << "knot=" << knot;
        }
    }
}

TYPED_TEST(Evaluate, HessianNonFiniteGivesNaN) {
    constexpr int D = TestFixture::D;
    PaddedCoefficients<double, D> c(8, 8, smooth_pixel);
    for (const double bad : {kNaN, kInf, -kInf}) {
        const auto rx = bsi::evaluate_with_hessian<double, D>(c.view, bad, 2.5);
        const auto ry = bsi::evaluate_with_hessian<double, D>(c.view, 2.5, bad);
        for (const auto& r : {rx, ry}) {
            EXPECT_TRUE(std::isnan(r.value));
            EXPECT_TRUE(std::isnan(r.dx));
            EXPECT_TRUE(std::isnan(r.dy));
            EXPECT_TRUE(std::isnan(r.dxx));
            EXPECT_TRUE(std::isnan(r.dxy));
            EXPECT_TRUE(std::isnan(r.dyy));
        }
    }
}

TYPED_TEST(Evaluate, ConstantAndLinearHaveZeroHessian) {
    constexpr int D = TestFixture::D;
    PaddedCoefficients<double, D> constant(
        32, 16, [](std::ptrdiff_t, std::ptrdiff_t) { return 3.0; });
    PaddedCoefficients<double, D> linear(
        32, 16, [](std::ptrdiff_t i, std::ptrdiff_t) { return static_cast<double>(i); });
    for (double x = 8.0; x <= 24.0; x += 0.37) {
        for (const auto* c : {&constant.view, &linear.view}) {
            const auto r = bsi::evaluate_with_hessian<double, D>(
                bsi::CoeffView<const double>(*c), x, 6.3);
            EXPECT_NEAR(r.dxx, 0.0, 1e-12) << "x=" << x;
            EXPECT_NEAR(r.dxy, 0.0, 1e-12) << "x=" << x;
            EXPECT_NEAR(r.dyy, 0.0, 1e-12) << "x=" << x;
        }
    }
}

}  // namespace
