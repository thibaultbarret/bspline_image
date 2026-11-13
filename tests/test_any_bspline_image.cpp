// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "bsi/any_bspline_image.hpp"
#include "bsi/bsi.hpp"
#include "support/test_images.hpp"

namespace {

constexpr std::ptrdiff_t kW = 23;
constexpr std::ptrdiff_t kH = 17;
constexpr std::size_t kPoints = 50;

std::vector<double> make_image() {
    std::vector<double> img(static_cast<std::size_t>(kW * kH));
    for (std::ptrdiff_t j = 0; j < kH; ++j) {
        for (std::ptrdiff_t i = 0; i < kW; ++i) {
            img[static_cast<std::size_t>(j * kW + i)] = smooth_pixel(i, j);
        }
    }
    return img;
}

bsi::ImageView<const double> view_of(const std::vector<double>& v) {
    return bsi::ImageView<const double>{v.data(), kW, kH, kW};
}

template <class T>
bool bit_equal(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <class T>
void make_points(std::vector<T>& xs, std::vector<T>& ys) {
    for (std::size_t k = 0; k < kPoints; ++k) {
        xs.push_back(static_cast<T>(-2.0 + 0.53 * static_cast<double>(k)));
        ys.push_back(static_cast<T>(-1.5 + 0.41 * static_cast<double>(k)));
    }
}

template <int D>
void expect_matches_core() {
    const auto img = make_image();
    const bsi::AnyBSplineImage<double> any(view_of(img), D);
    const bsi::BSplineImage<double, D> core(view_of(img));
    std::vector<double> xs, ys;
    make_points(xs, ys);

    for (std::size_t k = 0; k < kPoints; ++k) {
        EXPECT_TRUE(bit_equal(any.value(xs[k], ys[k]), core.value(xs[k], ys[k])));
        const auto g = any.gradient(xs[k], ys[k]);
        const auto gc = core.gradient(xs[k], ys[k]);
        EXPECT_TRUE(bit_equal(g.dx, gc.dx));
        EXPECT_TRUE(bit_equal(g.dy, gc.dy));
        const auto vg = any.value_and_gradient(xs[k], ys[k]);
        const auto vgc = core.value_and_gradient(xs[k], ys[k]);
        EXPECT_TRUE(bit_equal(vg.value, vgc.value));
        EXPECT_TRUE(bit_equal(vg.dx, vgc.dx));
        EXPECT_TRUE(bit_equal(vg.dy, vgc.dy));
    }

    std::vector<double> v(kPoints), dx(kPoints), dy(kPoints);
    std::vector<double> vc(kPoints), dxc(kPoints), dyc(kPoints);
    any.values(xs, ys, v);
    core.values(xs, ys, vc);
    EXPECT_EQ(std::memcmp(v.data(), vc.data(), kPoints * sizeof(double)), 0);
    any.gradients(xs, ys, dx, dy);
    core.gradients(xs, ys, dxc, dyc);
    EXPECT_EQ(std::memcmp(dx.data(), dxc.data(), kPoints * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(dy.data(), dyc.data(), kPoints * sizeof(double)), 0);
    any.values_and_gradients(xs, ys, v, dx, dy);
    core.values_and_gradients(xs, ys, vc, dxc, dyc);
    EXPECT_EQ(std::memcmp(v.data(), vc.data(), kPoints * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(dx.data(), dxc.data(), kPoints * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(dy.data(), dyc.data(), kPoints * sizeof(double)), 0);
}

template <int... Ds>
void expect_all_degrees_match(std::integer_sequence<int, Ds...>) {
    (expect_matches_core<Ds>(), ...);
}

}  // namespace

TEST(AnyBSplineImage, MatchesCoreBitForBit) {
    expect_all_degrees_match(std::make_integer_sequence<int, bsi::max_degree + 1>{});
}

TEST(AnyBSplineImage, ReportsDegreeAndSize) {
    const auto img = make_image();
    for (int d = 0; d <= bsi::max_degree; ++d) {
        const bsi::AnyBSplineImage<double> any(view_of(img), d);
        EXPECT_EQ(any.degree(), d);
        EXPECT_EQ(any.width(), kW);
        EXPECT_EQ(any.height(), kH);
    }
}

TEST(AnyBSplineImage, RejectsInvalidDegree) {
    const auto img = make_image();
    EXPECT_THROW(bsi::AnyBSplineImage<double>(view_of(img), -1), std::invalid_argument);
    EXPECT_THROW(bsi::AnyBSplineImage<double>(view_of(img), bsi::max_degree + 1),
                 std::invalid_argument);
}

TEST(AnyBSplineImage, PropagatesInvalidImage) {
    const bsi::ImageView<const double> empty{nullptr, 0, 0, 0};
    EXPECT_THROW(bsi::AnyBSplineImage<double>(empty, 3), std::invalid_argument);
}

TEST(AnyBSplineImage, VisitExposesConcreteType) {
    const auto img = make_image();
    for (int d = 0; d <= bsi::max_degree; ++d) {
        const bsi::AnyBSplineImage<double> any(view_of(img), d);
        const int seen = any.visit([](const auto& s) { return std::decay_t<decltype(s)>::degree; });
        EXPECT_EQ(seen, d);
    }
}

TEST(AnyBSplineImage, FloatFacadeCallsEveryMethod) {
    std::vector<std::uint8_t> img(static_cast<std::size_t>(kW * kH));
    for (std::ptrdiff_t j = 0; j < kH; ++j) {
        for (std::ptrdiff_t i = 0; i < kW; ++i) {
            img[static_cast<std::size_t>(j * kW + i)] =
                static_cast<std::uint8_t>(smooth_pixel(i, j));
        }
    }
    const bsi::AnyBSplineImage<float> any(bsi::ImageView<const std::uint8_t>{img.data(), kW, kH, kW},
                                          3);
    const bsi::BSplineImage<float, 3> core(
        bsi::ImageView<const std::uint8_t>{img.data(), kW, kH, kW});
    std::vector<float> xs, ys;
    make_points(xs, ys);

    EXPECT_EQ(any.degree(), 3);
    EXPECT_EQ(any.width(), kW);
    EXPECT_EQ(any.height(), kH);
    EXPECT_TRUE(bit_equal(any.value(xs[7], ys[7]), core.value(xs[7], ys[7])));
    const auto g = any.gradient(xs[7], ys[7]);
    EXPECT_TRUE(bit_equal(g.dx, core.gradient(xs[7], ys[7]).dx));
    const auto vg = any.value_and_gradient(xs[7], ys[7]);
    EXPECT_TRUE(bit_equal(vg.dy, core.value_and_gradient(xs[7], ys[7]).dy));

    std::vector<float> v(kPoints), dx(kPoints), dy(kPoints);
    any.values(xs, ys, v);
    EXPECT_TRUE(bit_equal(v[9], core.value(xs[9], ys[9])));
    any.gradients(xs, ys, dx, dy);
    EXPECT_TRUE(bit_equal(dx[9], core.gradient(xs[9], ys[9]).dx));
    any.values_and_gradients(xs, ys, v, dx, dy);
    EXPECT_TRUE(bit_equal(dy[9], core.gradient(xs[9], ys[9]).dy));
    EXPECT_EQ(any.visit([](const auto& s) { return s.width(); }), kW);
}
