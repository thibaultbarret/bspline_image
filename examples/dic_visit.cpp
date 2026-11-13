// SPDX-License-Identifier: BSD-3-Clause
// DIC-style subset evaluation: build a runtime-degree interpolator, dispatch once with
// visit(), and evaluate a 21x21 subset with direct (non-dispatching) calls.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <numbers>
#include <string>
#include <vector>

#include "bsi/bsi.hpp"

namespace {

constexpr int kImageSize = 128;
constexpr int kSubsetSize = 21;
constexpr double kShiftX = 0.3;
constexpr double kShiftY = -0.2;
constexpr double kSubsetCenter = kImageSize / 2.0;
constexpr double kPatternPeriod = 16.0;
constexpr double kPatternMean = 128.0;
constexpr double kPatternAmplitude = 100.0;
constexpr double kPatternYRatio = 0.7;

struct SubsetResult {
    double ssd;       ///< Sum of squared differences between shifted and reference values.
    double grad_sum;  ///< Sum of gradient components (dx + dy) at the shifted points.
};

/// Evaluate the subset on a concrete spline type (no per-call dispatch).
template <class Spline>
SubsetResult ssd_subset(const Spline& spline, double cx, double cy, double shift_x,
                        double shift_y) {
    SubsetResult res{0.0, 0.0};
    const int half = kSubsetSize / 2;
    for (int j = -half; j <= half; ++j) {
        for (int i = -half; i <= half; ++i) {
            const double x = cx + i;
            const double y = cy + j;
            const double ref = spline.value(x, y);
            const auto shifted = spline.value_and_gradient(x + shift_x, y + shift_y);
            const double diff = shifted.value - ref;
            res.ssd += diff * diff;
            res.grad_sum += shifted.dx + shifted.dy;
        }
    }
    return res;
}

/// Smooth deterministic test image.
std::vector<double> make_image() {
    std::vector<double> img(static_cast<std::size_t>(kImageSize) * kImageSize);
    const double k = 2.0 * std::numbers::pi / kPatternPeriod;
    for (int y = 0; y < kImageSize; ++y) {
        for (int x = 0; x < kImageSize; ++x) {
            img[static_cast<std::size_t>(y) * kImageSize + static_cast<std::size_t>(x)] =
                kPatternMean +
                kPatternAmplitude * std::sin(k * x) * std::cos(k * kPatternYRatio * y);
        }
    }
    return img;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int degree = argc > 1 ? std::stoi(argv[1]) : 5;
        const std::vector<double> pixels = make_image();
        const bsi::ImageView<const double> view{pixels.data(), kImageSize, kImageSize,
                                                kImageSize};
        const bsi::AnyBSplineImage<double> any(view, degree);

        const SubsetResult res = any.visit([](const auto& spline) {
            return ssd_subset(spline, kSubsetCenter, kSubsetCenter, kShiftX, kShiftY);
        });
        std::printf("degree %d: SSD = %.9g, gradient sum = %.9g\n", degree, res.ssd,
                    res.grad_sum);
        return std::isfinite(res.ssd) && std::isfinite(res.grad_sum) ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
