// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Whole-sample-symmetric mirror boundary kernels.
///
/// The signal of \f$n\f$ samples is extended by reflection about its first and last
/// samples, without duplicating them. The extension is periodic with period
/// \f$p = 2n-2\f$ and even about 0: \f$e(i) = e(-i) = e(i+p)\f$. This is the
/// `mode='mirror'` boundary of `scipy.ndimage`.

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "bsi/config.hpp"
#include "bsi/types.hpp"

namespace bsi {

/// @brief Mirror an integer index into \f$[0, n-1]\f$ (period \f$2n-2\f$, edges not duplicated).
///
/// With \f$p = 2n-2\f$ and \f$r = i \bmod p \in [0, p)\f$, the result is \f$r\f$ if
/// \f$r < n\f$ and \f$p - r\f$ otherwise.
///
/// @param i Index of any sign and magnitude.
/// @param n Number of samples.
/// @return The mirrored index in \f$[0, n-1]\f$; always 0 when \f$n = 1\f$.
/// @pre \f$n \ge 1\f$.
BSI_HOST_DEVICE inline std::ptrdiff_t mirror_index(std::ptrdiff_t i, std::ptrdiff_t n) noexcept {
    if (n <= 1) return 0;
    const std::ptrdiff_t period = 2 * n - 2;
    const std::ptrdiff_t r = ((i % period) + period) % period;
    return r < n ? r : period - r;
}

/// @brief Result of folding a coordinate: position and orientation.
/// @tparam T Floating-point scalar type.
template <class T>
struct Folded {
    T x;     ///< Folded coordinate in \f$[0, n-1]\f$.
    T sign;  ///< \f$-1\f$ if the folding reverses orientation, \f$+1\f$ otherwise.
};

/// @brief Fold a real coordinate into \f$[0, n-1]\f$ with the mirror symmetry.
///
/// Computed in floating point (`fabs` then `fmod` by \f$2(n-1)\f$), so arbitrarily large
/// coordinates never overflow an integer. The returned `sign` must multiply derivatives of
/// the signal evaluated at the folded position. NaN stays NaN.
///
/// @tparam T Floating-point scalar type.
/// @param x Coordinate of any value.
/// @param n Number of samples.
/// @return `{x', s}` with \f$x' \in [0, n-1]\f$ (or NaN) and \f$s = \pm 1\f$; `{0, 1}` when
///         \f$n = 1\f$.
/// @pre \f$n \ge 1\f$.
template <class T>
BSI_HOST_DEVICE inline Folded<T> fold_mirror(T x, std::ptrdiff_t n) noexcept {
    static_assert(std::is_floating_point_v<T>, "fold_mirror: T must be floating point");
    if (n <= 1) return Folded<T>{T(0), T(1)};
    const T last = static_cast<T>(n - 1);
    const T period = T(2) * last;
    const T r = std::fmod(std::abs(x), period);
    const bool reflected = r > last;
    const bool negative = x < T(0);
    return Folded<T>{reflected ? period - r : r, (reflected != negative) ? T(-1) : T(1)};
}

/// @brief Fill the padding cells of a coefficient array by mirror extension.
///
/// Rows are processed first (padding columns of interior rows), then columns (padding rows
/// over the full padded width), so that every padded cell \f$(i, j)\f$ receives
/// `c.at(mirror_index(i, width), mirror_index(j, height))`.
///
/// @tparam T Coefficient type.
/// @param c View whose interior is filled; its `pad` cells on each side are overwritten.
/// @pre The storage behind `c` is padded by at least `c.pad` cells on each side,
///      `c.width >= 1` and `c.height >= 1`.
template <class T>
BSI_HOST_DEVICE inline void fill_mirror_padding(CoeffView<T> c) noexcept {
    for (std::ptrdiff_t j = 0; j < c.height; ++j) {
        for (std::ptrdiff_t k = 1; k <= c.pad; ++k) {
            c.at(-k, j) = c.at(mirror_index(-k, c.width), j);
            c.at(c.width - 1 + k, j) = c.at(mirror_index(c.width - 1 + k, c.width), j);
        }
    }
    for (std::ptrdiff_t i = -c.pad; i < c.width + c.pad; ++i) {
        for (std::ptrdiff_t k = 1; k <= c.pad; ++k) {
            c.at(i, -k) = c.at(i, mirror_index(-k, c.height));
            c.at(i, c.height - 1 + k) = c.at(i, mirror_index(c.height - 1 + k, c.height));
        }
    }
}

}  // namespace bsi
