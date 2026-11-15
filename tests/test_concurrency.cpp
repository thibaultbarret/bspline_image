// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "bsi/any_bspline_image.hpp"
#include "bsi/bsi.hpp"
#include "support/test_images.hpp"

namespace {

constexpr std::ptrdiff_t kSide = 96;
constexpr std::size_t kThreads = 8;
constexpr std::size_t kPointsPerThread = 4000;
constexpr double kMinCoord = 2.0;
constexpr double kMaxCoord = 93.0;
constexpr unsigned kSeed = 12345U;

std::vector<double> make_image() {
    std::vector<double> img(static_cast<std::size_t>(kSide * kSide));
    for (std::ptrdiff_t j = 0; j < kSide; ++j) {
        for (std::ptrdiff_t i = 0; i < kSide; ++i) {
            img[static_cast<std::size_t>(j * kSide + i)] = smooth_pixel(i, j);
        }
    }
    return img;
}

bsi::ImageView<const double> view_of(const std::vector<double>& img) {
    return bsi::ImageView<const double>{img.data(), kSide, kSide, kSide};
}

template <class T>
struct Points {
    std::vector<T> xs;
    std::vector<T> ys;
};

template <class T>
struct Outputs {
    explicit Outputs(std::size_t n) : v(n), dx(n), dy(n), dxx(n), dxy(n), dyy(n) {}
    std::vector<T> v, dx, dy, dxx, dxy, dyy;
    bool operator==(const Outputs&) const = default;
};

template <class T>
Points<T> make_points() {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<double> dist(kMinCoord, kMaxCoord);
    Points<T> p;
    for (std::size_t k = 0; k < kThreads * kPointsPerThread; ++k) {
        p.xs.push_back(static_cast<T>(dist(rng)));
        p.ys.push_back(static_cast<T>(dist(rng)));
    }
    return p;
}

/// Evaluates every block of points sequentially, then in kThreads concurrent threads (each with
/// its own output buffers), and checks that the results are identical.
/// `evaluate(xs, ys, out)` fills `out` for one block of points.
template <class T, class Evaluate>
void expect_concurrent_matches_sequential(const Evaluate& evaluate) {
    const auto pts = make_points<T>();
    const auto block = [&](std::size_t t) {
        const std::size_t begin = t * kPointsPerThread;
        return std::pair{std::span<const T>(pts.xs).subspan(begin, kPointsPerThread),
                         std::span<const T>(pts.ys).subspan(begin, kPointsPerThread)};
    };

    std::vector<Outputs<T>> sequential(kThreads, Outputs<T>(kPointsPerThread));
    for (std::size_t t = 0; t < kThreads; ++t) {
        const auto [xs, ys] = block(t);
        evaluate(xs, ys, sequential[t]);
    }

    std::vector<Outputs<T>> parallel(kThreads, Outputs<T>(kPointsPerThread));
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            const auto [xs, ys] = block(t);
            evaluate(xs, ys, parallel[t]);
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    for (std::size_t t = 0; t < kThreads; ++t) {
        EXPECT_TRUE(parallel[t] == sequential[t]) << "block " << t;
    }
}

template <class Spline, class T>
void pointwise_hessians(const Spline& s, std::span<const T> xs, std::span<const T> ys,
                        Outputs<T>& o) {
    for (std::size_t k = 0; k < xs.size(); ++k) {
        const auto r = s.value_gradient_hessian(xs[k], ys[k]);
        o.v[k] = r.value;
        o.dx[k] = r.dx;
        o.dy[k] = r.dy;
        o.dxx[k] = r.dxx;
        o.dxy[k] = r.dxy;
        o.dyy[k] = r.dyy;
    }
}

template <class T, int Degree>
void check_shared_spline() {
    const auto img = make_image();
    const bsi::BSplineImage<T, Degree> spline(view_of(img));
    expect_concurrent_matches_sequential<T>(
        [&](std::span<const T> xs, std::span<const T> ys, Outputs<T>& o) {
            spline.values_gradients_hessians(xs, ys, o.v, o.dx, o.dy, o.dxx, o.dxy, o.dyy);
        });
    expect_concurrent_matches_sequential<T>(
        [&](std::span<const T> xs, std::span<const T> ys, Outputs<T>& o) {
            pointwise_hessians(spline, xs, ys, o);
        });
}

}  // namespace

TEST(Concurrency, SharedBSplineImageMatchesSequential) {
    check_shared_spline<double, 5>();
    check_shared_spline<float, 3>();
}

TEST(Concurrency, SharedAnyBSplineImageMatchesSequential) {
    const auto img = make_image();
    const bsi::AnyBSplineImage<double> any(view_of(img), 5);

    expect_concurrent_matches_sequential<double>(
        [&](std::span<const double> xs, std::span<const double> ys, Outputs<double>& o) {
            any.values_and_gradients(xs, ys, o.v, o.dx, o.dy);
        });
    expect_concurrent_matches_sequential<double>(
        [&](std::span<const double> xs, std::span<const double> ys, Outputs<double>& o) {
            any.visit([&](const auto& s) { pointwise_hessians(s, xs, ys, o); });
        });
}
