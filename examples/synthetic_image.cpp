// SPDX-License-Identifier: BSD-3-Clause
// Synthetic image generation: a speckle pattern is warped by a smooth displacement
// field using B-spline interpolation, and both images are written as 8-bit PGM files.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bsi/bsi.hpp"

namespace {

constexpr int kSize = 256;
constexpr int kBlobCount = 400;
constexpr double kBlobRadius = 2.0;  // pixels
constexpr unsigned kSeed = 42;
constexpr double kAmplitudeX = 0.5;   // u = 0.5 sin(2 pi y / 256)
constexpr double kAmplitudeY = 0.25;  // v = 0.25 cos(2 pi x / 256)
constexpr int kBlobSupport = 3;       // blob is evaluated within kBlobSupport * radius
constexpr double kGrayMax = 255.0;

std::size_t index_of(int x, int y) {
    return static_cast<std::size_t>(y) * kSize + static_cast<std::size_t>(x);
}

/// Sum of Gaussian blobs of radius kBlobRadius at random positions, scaled to [0, 255].
std::vector<double> make_speckle() {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<double> pos(0.0, static_cast<double>(kSize));
    std::uniform_real_distribution<double> amp(0.5, 1.0);

    std::vector<double> img(static_cast<std::size_t>(kSize) * kSize, 0.0);
    const int reach = static_cast<int>(std::ceil(kBlobSupport * kBlobRadius));
    for (int b = 0; b < kBlobCount; ++b) {
        const double cx = pos(rng);
        const double cy = pos(rng);
        const double a = amp(rng);
        const int x0 = std::max(0, static_cast<int>(cx) - reach);
        const int x1 = std::min(kSize - 1, static_cast<int>(cx) + reach);
        const int y0 = std::max(0, static_cast<int>(cy) - reach);
        const int y1 = std::min(kSize - 1, static_cast<int>(cy) + reach);
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double dx = x - cx;
                const double dy = y - cy;
                const double r2 = (dx * dx + dy * dy) / (kBlobRadius * kBlobRadius);
                img[index_of(x, y)] += a * std::exp(-r2);
            }
        }
    }
    const double peak = *std::max_element(img.begin(), img.end());
    for (double& v : img) {
        v *= kGrayMax / peak;
    }
    return img;
}

std::uint8_t to_gray(double v) {
    return static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
}

/// Write a binary PGM (P5, 8 bits) image; values are rounded and clamped to [0, 255].
void write_pgm(const std::string& path, const std::vector<double>& img) {
    std::vector<std::uint8_t> bytes(img.size());
    std::transform(img.begin(), img.end(), bytes.begin(), to_gray);
    std::ofstream out(path, std::ios::binary);
    out << "P5\n" << kSize << ' ' << kSize << "\n255\n";
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
}

}  // namespace

int main() {
    try {
        const std::vector<double> reference = make_speckle();
        const bsi::ImageView<const double> view{reference.data(), kSize, kSize, kSize};
        const bsi::BSplineImage<double, 5> spline(view);

        const double two_pi_over_size = 2.0 * std::numbers::pi / kSize;
        std::vector<double> deformed(reference.size());
        for (int y = 0; y < kSize; ++y) {
            for (int x = 0; x < kSize; ++x) {
                const double u = kAmplitudeX * std::sin(two_pi_over_size * y);
                const double v = kAmplitudeY * std::cos(two_pi_over_size * x);
                deformed[index_of(x, y)] = spline.value(x - u, y - v);
            }
        }

        write_pgm("reference.pgm", reference);
        write_pgm("deformed.pgm", deformed);
        std::puts("wrote reference.pgm and deformed.pgm");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
