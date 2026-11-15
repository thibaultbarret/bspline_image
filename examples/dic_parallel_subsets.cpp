// SPDX-License-Identifier: BSD-3-Clause
// Local DIC with full Newton, parallelised over subsets.
//
// A "subset" is a small square window of the reference image F. For each subset we look for the
// displacement u = (ux, uy) that minimises the sum of squared differences (SSD)
//     C(u) = 1/2 * sum_x ( G(x + u) - F(x) )^2
// where G is the deformed image, interpolated by a B-spline so it can be sampled at sub-pixel
// positions. With r = G(x + u) - F(x), the gradient is g = sum r * grad G and the exact Hessian is
//     H = sum ( grad G grad G^T + r * hess G ).
// The second term is dropped by Gauss-Newton; here we keep it, which is why the interpolator
// provides second derivatives. Each Newton step solves H du = -g.
//
// Subsets are independent and the interpolator is only read, so a single const interpolator is
// shared by all threads without any locking.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <random>
#include <thread>
#include <vector>

#include "bsi/bsi.hpp"

namespace {

constexpr int kSize = 256;
constexpr int kBlobCount = 400;
constexpr double kBlobRadius = 2.0;  // pixels
constexpr int kBlobReach = 10;  // truncation radius: the cut tails are below exp(-25) of the peak
constexpr unsigned kSeed = 42;
constexpr double kBlobAmplitude = 100.0;
constexpr double kTrueUx = 0.4;
constexpr double kTrueUy = 0.3;
constexpr int kSubsetHalf = 15;  // 31 x 31 pixels
constexpr int kStep = 24;
constexpr int kMargin = 24 + kSubsetHalf;  // subset edges stay 24 px from the border
constexpr int kMaxIterations = 20;
constexpr double kTolerance = 1e-8;
constexpr double kMaxAllowedError = 1e-3;

using Spline = bsi::BSplineImage<double, 5>;

struct Blob {
    double x, y, amplitude;
};

struct Displacement {
    double ux = 0.0;
    double uy = 0.0;
};

struct Subset {
    int cx, cy;
};

std::vector<Blob> make_blobs() {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<double> pos(0.0, static_cast<double>(kSize));
    std::uniform_real_distribution<double> amp(0.5, 1.0);
    std::vector<Blob> blobs;
    for (int i = 0; i < kBlobCount; ++i) {
        const double x = pos(rng);
        const double y = pos(rng);
        blobs.push_back({x, y, kBlobAmplitude * amp(rng)});
    }
    return blobs;
}

/// Speckle pattern f (a seeded sum of Gaussian blobs) sampled at integer pixels after a shift:
/// image(x) = f(x - shift). f is analytic, so this is exact up to truncating each blob at
/// kBlobReach pixels (relative error ~1e-11).
std::vector<double> render_speckle(const std::vector<Blob>& blobs, double shift_x, double shift_y) {
    std::vector<double> img(static_cast<std::size_t>(kSize) * kSize, 0.0);
    for (const Blob& b : blobs) {
        const int cx = static_cast<int>(b.x + shift_x);
        const int cy = static_cast<int>(b.y + shift_y);
        const int y_end = std::min(kSize - 1, cy + kBlobReach);
        const int x_end = std::min(kSize - 1, cx + kBlobReach);
        for (int y = std::max(0, cy - kBlobReach); y <= y_end; ++y) {
            for (int x = std::max(0, cx - kBlobReach); x <= x_end; ++x) {
                const double dx = x - shift_x - b.x;
                const double dy = y - shift_y - b.y;
                const auto index =
                    static_cast<std::size_t>(y) * kSize + static_cast<std::size_t>(x);
                img[index] +=
                    b.amplitude * std::exp(-(dx * dx + dy * dy) / (kBlobRadius * kBlobRadius));
            }
        }
    }
    return img;
}

/// Full Newton minimisation of the SSD for the subset centred on pixel (cx, cy).
Displacement solve_subset(const Spline& deformed, const std::vector<double>& reference,
                          const Subset& subset) {
    std::vector<double> ref_values, px, py;
    for (int j = -kSubsetHalf; j <= kSubsetHalf; ++j) {
        for (int i = -kSubsetHalf; i <= kSubsetHalf; ++i) {
            const auto index = static_cast<std::size_t>(subset.cy + j) * kSize +
                               static_cast<std::size_t>(subset.cx + i);
            ref_values.push_back(reference[index]);
            px.push_back(subset.cx + i);
            py.push_back(subset.cy + j);
        }
    }
    const std::size_t n = ref_values.size();
    std::vector<double> xs(n), ys(n), g(n), gx(n), gy(n), gxx(n), gxy(n), gyy(n);

    // Full Newton assumes a start close to the minimum (H may be indefinite far from it); here
    // u = (0, 0) is within 0.5 px of the true displacement.
    Displacement u;
    for (int it = 0; it < kMaxIterations; ++it) {
        for (std::size_t k = 0; k < n; ++k) {
            xs[k] = px[k] + u.ux;
            ys[k] = py[k] + u.uy;
        }
        deformed.values_gradients_hessians(xs, ys, g, gx, gy, gxx, gxy, gyy);

        double g1 = 0, g2 = 0, h11 = 0, h12 = 0, h22 = 0;
        for (std::size_t k = 0; k < n; ++k) {
            const double r = g[k] - ref_values[k];
            g1 += r * gx[k];
            g2 += r * gy[k];
            h11 += gx[k] * gx[k] + r * gxx[k];
            h12 += gx[k] * gy[k] + r * gxy[k];
            h22 += gy[k] * gy[k] + r * gyy[k];
        }
        const double det = h11 * h22 - h12 * h12;
        if (det == 0.0) {
            break;  // singular Hessian: keep the current estimate
        }
        const double dux = -(h22 * g1 - h12 * g2) / det;
        const double duy = -(h11 * g2 - h12 * g1) / det;
        u = {u.ux + dux, u.uy + duy};
        if (std::hypot(dux, duy) < kTolerance) {
            break;
        }
    }
    return u;
}

std::vector<Displacement> solve_all(const Spline& deformed, const std::vector<double>& reference,
                                    const std::vector<Subset>& subsets, unsigned thread_count) {
    std::vector<Displacement> results(subsets.size());
    // Thread t handles subsets t, t + n, t + 2n, ... and writes only its own result slots.
    // An exception in a worker (e.g. bad_alloc) or a failed thread creation terminates the
    // program, which is acceptable for an example.
    auto work = [&](std::size_t first) {
        for (std::size_t k = first; k < subsets.size(); k += thread_count) {
            results[k] = solve_subset(deformed, reference, subsets[k]);
        }
    };
    std::vector<std::thread> threads;
    for (unsigned t = 1; t < thread_count; ++t) {
        threads.emplace_back(work, static_cast<std::size_t>(t));
    }
    work(0);  // the calling thread takes its share too
    for (std::thread& t : threads) {
        t.join();
    }
    return results;
}

template <class Fn>
double seconds(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main() {
    try {
        const std::vector<Blob> blobs = make_blobs();
        const std::vector<double> reference = render_speckle(blobs, 0.0, 0.0);
        const std::vector<double> deformed_pixels = render_speckle(blobs, kTrueUx, kTrueUy);
        const bsi::ImageView<const double> view{deformed_pixels.data(), kSize, kSize, kSize};
        const Spline deformed(view);  // shared by all threads, read-only

        std::vector<Subset> subsets;
        for (int cy = kMargin; cy < kSize - kMargin; cy += kStep) {
            for (int cx = kMargin; cx < kSize - kMargin; cx += kStep) {
                subsets.push_back({cx, cy});
            }
        }

        const unsigned thread_count = std::max(1u, std::thread::hardware_concurrency());
        std::vector<Displacement> sequential, parallel;
        const double t_seq =
            seconds([&] { sequential = solve_all(deformed, reference, subsets, 1); });
        const double t_par =
            seconds([&] { parallel = solve_all(deformed, reference, subsets, thread_count); });

        double max_error = 0.0;
        bool identical = true;
        for (std::size_t k = 0; k < subsets.size(); ++k) {
            identical = identical && sequential[k].ux == parallel[k].ux &&
                        sequential[k].uy == parallel[k].uy;
            max_error = std::max({max_error, std::abs(sequential[k].ux - kTrueUx),
                                  std::abs(sequential[k].uy - kTrueUy)});
        }

        std::printf("subsets: %zu, threads: %u\n", subsets.size(), thread_count);
        std::printf("sequential: %.4f s, parallel: %.4f s, speedup: %.2fx\n", t_seq, t_par,
                    t_seq / t_par);
        std::printf("max |u - u*|: %.3e px, parallel identical to sequential: %s\n", max_error,
                    identical ? "yes" : "no");
        return identical && max_error < kMaxAllowedError ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
