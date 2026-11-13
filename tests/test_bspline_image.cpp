// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bsi/bspline_image.hpp"
#include "bsi/kernel/mirror.hpp"
#include "support/degrees.hpp"
#include "support/test_images.hpp"

namespace {

constexpr std::ptrdiff_t kW = 23;
constexpr std::ptrdiff_t kH = 17;

template <class P>
std::vector<P> make_image(std::ptrdiff_t w, std::ptrdiff_t h, std::ptrdiff_t stride) {
    std::vector<P> img(static_cast<std::size_t>(stride * h), P(0));
    for (std::ptrdiff_t j = 0; j < h; ++j) {
        for (std::ptrdiff_t i = 0; i < w; ++i) {
            img[static_cast<std::size_t>(j * stride + i)] = static_cast<P>(smooth_pixel(i, j));
        }
    }
    return img;
}

template <class P>
bsi::ImageView<const P> view_of(const std::vector<P>& v, std::ptrdiff_t w, std::ptrdiff_t h,
                                std::ptrdiff_t stride) {
    return bsi::ImageView<const P>{v.data(), w, h, stride};
}

template <class T>
bool bit_equal(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <class D>
class BSplineImageT : public ::testing::Test {
public:
    static constexpr int degree = D::value;
};
TYPED_TEST_SUITE(BSplineImageT, AllDegrees);

}  // namespace

TYPED_TEST(BSplineImageT, InterpolatesPixelsAtIntegers) {
    const auto img = make_image<double>(kW, kH, kW);
    const bsi::BSplineImage<double, TestFixture::degree> s(view_of(img, kW, kH, kW));
    EXPECT_EQ(s.width(), kW);
    EXPECT_EQ(s.height(), kH);
    for (std::ptrdiff_t j = 0; j < kH; ++j) {
        for (std::ptrdiff_t i = 0; i < kW; ++i) {
            EXPECT_NEAR(s.value(double(i), double(j)), smooth_pixel(i, j), 1e-9);
        }
    }
}

TYPED_TEST(BSplineImageT, FloatInterpolatesPixels) {
    const auto img = make_image<float>(kW, kH, kW);
    const bsi::BSplineImage<float, TestFixture::degree> s(view_of(img, kW, kH, kW));
    for (std::ptrdiff_t j = 0; j < kH; ++j) {
        for (std::ptrdiff_t i = 0; i < kW; ++i) {
            EXPECT_NEAR(double(s.value(float(i), float(j))), double(img[std::size_t(j * kW + i)]),
                        2e-3);
        }
    }
}

TYPED_TEST(BSplineImageT, ReproducesPolynomialsInInterior) {
    constexpr int D = TestFixture::degree;
    constexpr std::ptrdiff_t n = 64;
    const auto poly = [](double u, double v) {
        double sum = 0.0;
        for (int a = 0; a <= D; ++a) {
            for (int b = 0; a + b <= D; ++b) {
                sum += std::pow(u, a) * std::pow(v, b) / double(a + b + 1);
            }
        }
        return sum;
    };
    std::vector<double> img(std::size_t(n * n));
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            img[std::size_t(j * n + i)] = poly(double(i) / 64.0, double(j) / 64.0);
        }
    }
    const bsi::BSplineImage<double, D> s(view_of(img, n, n, n));
    for (double y = 24.0; y <= 40.0; y += 0.37) {
        for (double x = 24.0; x <= 40.0; x += 0.37) {
            EXPECT_NEAR(s.value(x, y), poly(x / 64.0, y / 64.0), 1e-6);
        }
    }
}

TYPED_TEST(BSplineImageT, GradientMatchesFiniteDifferences) {
    constexpr int D = TestFixture::degree;
    if constexpr (D < 2) {
        GTEST_SKIP() << "gradient is only smooth for degree >= 2";
    } else {
        const auto img = make_image<double>(kW, kH, kW);
        const bsi::BSplineImage<double, D> s(view_of(img, kW, kH, kW));
        constexpr double h = 1e-5;
        for (int k = 0; k < 30; ++k) {
            const double x = 2.3 + 0.61 * k * 0.97;
            const double y = 1.7 + 0.43 * k * 0.9;
            const auto g = s.gradient(x, y);
            const double fdx = (s.value(x + h, y) - s.value(x - h, y)) / (2 * h);
            const double fdy = (s.value(x, y + h) - s.value(x, y - h)) / (2 * h);
            EXPECT_NEAR(g.dx, fdx, 1e-4);
            EXPECT_NEAR(g.dy, fdy, 1e-4);
        }
    }
}

TYPED_TEST(BSplineImageT, BatchEqualsPointwise) {
    constexpr int D = TestFixture::degree;
    const auto img = make_image<double>(kW, kH, kW);
    const bsi::BSplineImage<double, D> s(view_of(img, kW, kH, kW));
    constexpr std::size_t n = 100;
    std::vector<double> xs(n), ys(n), v(n), dx(n), dy(n), v2(n), dx2(n), dy2(n);
    for (std::size_t k = 0; k < n; ++k) {
        xs[k] = -2.0 + 0.27 * double(k);
        ys[k] = -1.5 + 0.19 * double(k);
    }
    s.values(xs, ys, v);
    s.gradients(xs, ys, dx, dy);
    s.values_and_gradients(xs, ys, v2, dx2, dy2);
    for (std::size_t k = 0; k < n; ++k) {
        const auto vg = s.value_and_gradient(xs[k], ys[k]);
        const auto g = s.gradient(xs[k], ys[k]);
        EXPECT_TRUE(bit_equal(v[k], s.value(xs[k], ys[k])));
        EXPECT_TRUE(bit_equal(dx[k], g.dx));
        EXPECT_TRUE(bit_equal(dy[k], g.dy));
        EXPECT_TRUE(bit_equal(v2[k], vg.value));
        EXPECT_TRUE(bit_equal(dx2[k], vg.dx));
        EXPECT_TRUE(bit_equal(dy2[k], vg.dy));
    }
}

TYPED_TEST(BSplineImageT, CopyAndMoveKeepResults) {
    constexpr int D = TestFixture::degree;
    const auto img = make_image<double>(kW, kH, kW);
    using Image = bsi::BSplineImage<double, D>;
    const auto other_img = make_image<double>(3, 2, 3);

    // Reference values come from an independently built object that is never copied or moved.
    const Image reference(view_of(img, kW, kH, kW));
    const double ref = reference.value(5.4, 7.7);
    const auto ref_g = reference.gradient(5.4, 7.7);
    const auto expect_same = [&](const Image& p) {
        EXPECT_TRUE(bit_equal(p.value(5.4, 7.7), ref));
        const auto g = p.gradient(5.4, 7.7);
        EXPECT_TRUE(bit_equal(g.dx, ref_g.dx));
        EXPECT_TRUE(bit_equal(g.dy, ref_g.dy));
    };
    const auto make_source = [&] { return Image(view_of(img, kW, kH, kW)); };

    // Copies must stay valid after their source is destroyed.
    std::vector<Image> keep;
    keep.reserve(4);
    {
        const Image source = make_source();
        Image copy_ctor(source);
        Image copy_assigned(view_of(other_img, 3, 2, 3));
        copy_assigned = source;
        keep.push_back(std::move(copy_ctor));
        keep.push_back(std::move(copy_assigned));
    }
    // Moves must stay valid after their source is destroyed or overwritten.
    {
        Image source = make_source();
        Image move_ctor(std::move(source));
        source = Image(view_of(other_img, 3, 2, 3));  // overwrite the moved-from object
        keep.push_back(std::move(move_ctor));
        Image source2 = make_source();
        Image move_assigned(view_of(other_img, 3, 2, 3));
        move_assigned = std::move(source2);
        source2 = Image(view_of(other_img, 3, 2, 3));
        keep.push_back(std::move(move_assigned));
    }
    // Copy from a moved-to object, then destroy the moved-to object.
    Image from_moved(view_of(other_img, 3, 2, 3));
    {
        Image moved_to = std::move(keep.back());
        keep.pop_back();
        from_moved = moved_to;
        const Image copied(moved_to);
        expect_same(copied);
    }
    for (const Image& p : keep) {
        expect_same(p);
    }
    expect_same(from_moved);
}

TYPED_TEST(BSplineImageT, StridedUint16EqualsContiguousDouble) {
    constexpr int D = TestFixture::degree;
    constexpr std::ptrdiff_t stride = kW + 5;
    std::vector<std::uint16_t> strided(std::size_t(stride * kH), 7);
    std::vector<double> contiguous(std::size_t(kW * kH));
    for (std::ptrdiff_t j = 0; j < kH; ++j) {
        for (std::ptrdiff_t i = 0; i < kW; ++i) {
            const std::ptrdiff_t raw = (i + j) % 3 == 0 ? 65535 : (i * 977 + j * 131) % 65536;
            const auto p = static_cast<std::uint16_t>(raw);
            strided[std::size_t(j * stride + i)] = p;
            contiguous[std::size_t(j * kW + i)] = double(p);
        }
    }
    const bsi::BSplineImage<double, D> a(view_of(strided, kW, kH, stride));
    const bsi::BSplineImage<double, D> b(view_of(contiguous, kW, kH, kW));
    for (double y = -1.0; y < double(kH); y += 0.83) {
        for (double x = -1.0; x < double(kW); x += 0.71) {
            EXPECT_NEAR(a.value(x, y), b.value(x, y), 1e-9);
        }
    }
}

TYPED_TEST(BSplineImageT, Uint8Input) {
    constexpr int D = TestFixture::degree;
    std::vector<std::uint8_t> img(std::size_t(kW * kH));
    for (std::size_t k = 0; k < img.size(); ++k) img[k] = static_cast<std::uint8_t>((k * 53) % 256);
    const bsi::BSplineImage<double, D> s(view_of(img, kW, kH, kW));
    for (std::ptrdiff_t j = 0; j < kH; ++j) {
        for (std::ptrdiff_t i = 0; i < kW; ++i) {
            EXPECT_NEAR(s.value(double(i), double(j)), double(img[std::size_t(j * kW + i)]), 1e-9);
        }
    }
}

TYPED_TEST(BSplineImageT, TinyImages) {
    constexpr int D = TestFixture::degree;
    const std::vector<double> one{42.0};
    const bsi::BSplineImage<double, D> s1(view_of(one, 1, 1, 1));
    for (double x : {-3.2, 0.0, 0.4, 9.9}) {
        EXPECT_NEAR(s1.value(x, x * 0.5), 42.0, 1e-9);
        const auto g = s1.gradient(x, x * 0.5);
        EXPECT_NEAR(g.dx, 0.0, 1e-9);
        EXPECT_NEAR(g.dy, 0.0, 1e-9);
    }
    const std::ptrdiff_t shapes[][2] = {{1, 5}, {5, 1}, {2, 2}};
    for (const auto& shape : shapes) {
        const auto img = make_image<double>(shape[0], shape[1], shape[0]);
        const bsi::BSplineImage<double, D> s(view_of(img, shape[0], shape[1], shape[0]));
        for (std::ptrdiff_t j = 0; j < shape[1]; ++j) {
            for (std::ptrdiff_t i = 0; i < shape[0]; ++i) {
                EXPECT_NEAR(s.value(double(i), double(j)), smooth_pixel(i, j), 1e-9);
            }
        }
    }
}

TYPED_TEST(BSplineImageT, FarCoordinatesEqualFolded) {
    constexpr int D = TestFixture::degree;
    const auto img = make_image<double>(kW, kH, kW);
    const bsi::BSplineImage<double, D> s(view_of(img, kW, kH, kW));
    const double far = 1e9 + 0.3;
    const double folded = bsi::fold_mirror<double>(far, kW).x;
    const double result = s.value(far, 5.2);
    EXPECT_TRUE(std::isfinite(result));
    EXPECT_NEAR(result, s.value(folded, 5.2), 1e-6);
}

TEST(BSplineImage, RejectsInvalidViews) {
    const std::vector<double> img(16, 1.0);
    using S = bsi::BSplineImage<double, 3>;
    EXPECT_THROW(S(bsi::ImageView<const double>{nullptr, 4, 4, 4}), std::invalid_argument);
    EXPECT_THROW(S(bsi::ImageView<const double>{img.data(), 0, 4, 4}), std::invalid_argument);
    EXPECT_THROW(S(bsi::ImageView<const double>{img.data(), 4, 0, 4}), std::invalid_argument);
    EXPECT_THROW(S(bsi::ImageView<const double>{img.data(), 4, 4, 3}), std::invalid_argument);
}
