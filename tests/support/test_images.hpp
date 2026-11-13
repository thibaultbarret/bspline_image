// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include <cstddef>

/// @brief Smooth deterministic test image value at pixel (i, j).
inline double smooth_pixel(std::ptrdiff_t i, std::ptrdiff_t j) {
    const double di = static_cast<double>(i);
    const double dj = static_cast<double>(j);
    return 100.0 + 50.0 * std::sin(0.7 * di + 0.3 * dj) + 30.0 * std::cos(0.45 * dj - 0.2 * di);
}
