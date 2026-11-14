// SPDX-License-Identifier: BSD-3-Clause
// Dependency-free evaluation benchmark. Prints a Markdown table of ns/point.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "bsi/bsi.hpp"

namespace {

constexpr std::ptrdiff_t kImageSize = 1024;
constexpr std::size_t kPointCount = 1'000'000;
constexpr int kRepetitions = 5;
constexpr unsigned kSeed = 12345;
constexpr double kMargin = 4.0;  // keep points away from the border

volatile double g_sink = 0.0;  // defeats dead-code elimination

template <class T>
struct Points {
    std::vector<T> xs;
    std::vector<T> ys;
};

/// Uniform points in the image interior, fixed seed.
template <class T>
Points<T> make_points() {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<double> coord(
        kMargin, static_cast<double>(kImageSize) - 1.0 - kMargin);
    Points<T> p;
    p.xs.reserve(kPointCount);
    p.ys.reserve(kPointCount);
    for (std::size_t k = 0; k < kPointCount; ++k) {
        p.xs.push_back(static_cast<T>(coord(rng)));
        p.ys.push_back(static_cast<T>(coord(rng)));
    }
    return p;
}

std::vector<float> make_image() {
    std::mt19937 rng(kSeed + 1);
    std::uniform_real_distribution<float> px(0.0F, 255.0F);
    std::vector<float> img(static_cast<std::size_t>(kImageSize * kImageSize));
    for (float& v : img) {
        v = px(rng);
    }
    return img;
}

/// One warm-up pass, then the minimum of kRepetitions timed passes, in ns/point.
template <class F>
double time_ns_per_point(F&& pass) {
    g_sink = g_sink + pass();
    double best = std::numeric_limits<double>::max();
    for (int r = 0; r < kRepetitions; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        const double acc = pass();
        const auto t1 = std::chrono::steady_clock::now();
        g_sink = g_sink + acc;
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        best = std::min(best, ns / static_cast<double>(kPointCount));
    }
    return best;
}

template <class T, int Degree>
void bench_row(const std::vector<float>& image) {
    const bsi::ImageView<const float> view{image.data(), kImageSize, kImageSize, kImageSize};
    const bsi::BSplineImage<T, Degree> core(view);
    const bsi::AnyBSplineImage<T> any(view, Degree);
    const Points<T> pts = make_points<T>();

    const double core_value = time_ns_per_point([&] {
        T acc{};
        for (std::size_t k = 0; k < kPointCount; ++k) {
            acc += core.value(pts.xs[k], pts.ys[k]);
        }
        return static_cast<double>(acc);
    });
    const double core_vg = time_ns_per_point([&] {
        T acc{};
        for (std::size_t k = 0; k < kPointCount; ++k) {
            const auto r = core.value_and_gradient(pts.xs[k], pts.ys[k]);
            acc += r.value + r.dx + r.dy;
        }
        return static_cast<double>(acc);
    });
    const double facade_point = time_ns_per_point([&] {
        T acc{};
        for (std::size_t k = 0; k < kPointCount; ++k) {
            acc += any.value(pts.xs[k], pts.ys[k]);
        }
        return static_cast<double>(acc);
    });
    const double facade_visit = time_ns_per_point([&] {
        return any.visit([&](const auto& spline) {
            T acc{};
            for (std::size_t k = 0; k < kPointCount; ++k) {
                acc += spline.value(pts.xs[k], pts.ys[k]);
            }
            return static_cast<double>(acc);
        });
    });
    const double facade_vg_point = time_ns_per_point([&] {
        T acc{};
        for (std::size_t k = 0; k < kPointCount; ++k) {
            const auto r = any.value_and_gradient(pts.xs[k], pts.ys[k]);
            acc += r.value + r.dx + r.dy;
        }
        return static_cast<double>(acc);
    });
    const double facade_vg_visit = time_ns_per_point([&] {
        return any.visit([&](const auto& spline) {
            T acc{};
            for (std::size_t k = 0; k < kPointCount; ++k) {
                const auto r = spline.value_and_gradient(pts.xs[k], pts.ys[k]);
                acc += r.value + r.dx + r.dy;
            }
            return static_cast<double>(acc);
        });
    });

    const double core_vgh = time_ns_per_point([&] {
        T acc{};
        for (std::size_t k = 0; k < kPointCount; ++k) {
            const auto r = core.value_gradient_hessian(pts.xs[k], pts.ys[k]);
            acc += r.value + r.dx + r.dy + r.dxx + r.dxy + r.dyy;
        }
        return static_cast<double>(acc);
    });
    const double facade_vgh = time_ns_per_point([&] {
        T acc{};
        for (std::size_t k = 0; k < kPointCount; ++k) {
            const auto r = any.value_gradient_hessian(pts.xs[k], pts.ys[k]);
            acc += r.value + r.dx + r.dy + r.dxx + r.dxy + r.dyy;
        }
        return static_cast<double>(acc);
    });

    const double overhead = 100.0 * (facade_visit - core_value) / core_value;
    std::printf("| %d | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %+.1f%% |\n",
                Degree, sizeof(T) == sizeof(float) ? "float" : "double", core_value, core_vg,
                core_vgh, facade_point, facade_vg_point, facade_vgh, facade_visit,
                facade_vg_visit, overhead);
}

template <int Degree>
void bench_degree(const std::vector<float>& image) {
    bench_row<float, Degree>(image);
    bench_row<double, Degree>(image);
}

}  // namespace

int main() {
    const std::vector<float> image = make_image();
    std::printf("Image %lldx%lld, %zu points, min of %d runs (ns/point)\n\n",
                static_cast<long long>(kImageSize), static_cast<long long>(kImageSize),
                kPointCount, kRepetitions);
    std::puts("| degree | type | core value | core value_and_gradient | core value_gradient_hessian | "
              "facade value | facade value_and_gradient | facade value_gradient_hessian | "
              "facade + visit value | "
              "facade + visit value_and_gradient | visit vs core value |");
    std::puts("|---|---|---|---|---|---|---|---|---|---|---|");
    bench_degree<1>(image);
    bench_degree<3>(image);
    bench_degree<5>(image);
    bench_degree<7>(image);
    std::printf("\nsink: %g\n", g_sink);
    return 0;
}
