// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Separable 2-D B-spline evaluation (value and gradient) from a coefficient array.
///
/// The interpolated function is
/// \f$f(x, y) = \sum_{i,j} c_{i,j}\, \beta^n(x - i)\, \beta^n(y - j)\f$ where only the
/// \f$(n+1)^2\f$ coefficients around \f$(x, y)\f$ contribute. Being separable, it is computed
/// as \f$n+1\f$ row sums along x followed by one weighted sum along y. Coordinates outside the
/// image are folded back with the mirror symmetry (see `fold_mirror`).

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "bsi/config.hpp"
#include "bsi/kernel/basis.hpp"
#include "bsi/kernel/fma.hpp"
#include "bsi/kernel/mirror.hpp"
#include "bsi/types.hpp"

namespace bsi {

/// @brief Padding (cells per side) a coefficient array needs for degree `Degree`.
///
/// The support of the interpolation window around a folded coordinate in
/// \f$[0, n-1]\f$ reaches at most \f$\lfloor n/2 \rfloor + 1\f$ cells beyond the image.
/// @tparam Degree Spline degree \f$n\f$.
template <int Degree>
inline constexpr std::ptrdiff_t required_pad = Degree / 2 + 1;

namespace detail {

/// @brief A quiet NaN of type `T`, built with `std::nan` of `<cmath>` only.
/// @tparam T Floating-point scalar type.
/// @return A NaN value.
template <class T>
BSI_HOST_DEVICE inline T make_nan() noexcept {
    return static_cast<T>(std::nan(""));
}

/// @brief Row sum \f$\sum_k a_k\, c(i_0 + k, j)\f$ over the x window.
/// @tparam T      Floating-point scalar type.
/// @tparam Degree Spline degree.
/// @param c Coefficient view.
/// @param i0 First column of the window.
/// @param j Row index.
/// @param a Weights (or weight derivatives), `Degree + 1` values.
/// @return The weighted sum of the window's coefficients in row `j`.
template <class T, int Degree>
BSI_HOST_DEVICE inline T row_sum(CoeffView<const T> c, std::ptrdiff_t i0, std::ptrdiff_t j,
                                 const T* a) noexcept {
    const T* row = &c.at(i0, j);
    T s = a[0] * row[0];
    for (int k = 1; k <= Degree; ++k) {
        s = mul_add(a[k], row[k], s);
    }
    return s;
}

/// @brief Fold a coordinate into \f$[0, n-1]\f$, skipping the floating-point fold when it is
///        already inside.
///
/// For \f$0 \le x \le n-1\f$ the fold is the identity with orientation \f$+1\f$ (this
/// includes `-0.0`, which folds to `+0.0` with the same weights), so `x` is returned as is.
/// Any other value (including NaN) goes through `fold_mirror`.
/// @tparam T Floating-point scalar type.
/// @param x Coordinate of any value.
/// @param n Number of samples.
/// @return Same result as `fold_mirror(x, n)` up to the sign of a zero.
/// @pre \f$n \ge 1\f$.
template <class T>
BSI_HOST_DEVICE inline Folded<T> fold_fast(T x, std::ptrdiff_t n) noexcept {
    if (x >= T(0) && x <= static_cast<T>(n - 1)) {
        return Folded<T>{x, T(1)};
    }
    return fold_mirror(x, n);
}

}  // namespace detail

/// @brief Evaluate the interpolated value at \f$(x, y)\f$.
///
/// Non-finite coordinates give NaN (checked before any conversion to an integer).
///
/// @tparam T      Floating-point scalar type.
/// @tparam Degree Spline degree \f$n\f$.
/// @param c Coefficients (output of the prefilter), mirror padding filled.
/// @param x Column coordinate (any value, pixel centers at integers).
/// @param y Row coordinate.
/// @return The interpolated value, or NaN if `x` or `y` is not finite.
/// @pre `c.pad >= required_pad<Degree>`, padding filled by `fill_mirror_padding`,
///      `c.width >= 1`, `c.height >= 1`.
template <class T, int Degree>
BSI_HOST_DEVICE inline T evaluate(CoeffView<const T> c, T x, T y) noexcept {
    static_assert(std::is_floating_point_v<T>, "evaluate: T must be floating point");
    if (!std::isfinite(x) || !std::isfinite(y)) {
        return detail::make_nan<T>();
    }
    const Folded<T> fx = detail::fold_fast(x, c.width);
    const Folded<T> fy = detail::fold_fast(y, c.height);

    T wx[Degree + 1];
    T wy[Degree + 1];
    const std::ptrdiff_t i0 = weights<T, Degree>(fx.x, wx);
    const std::ptrdiff_t j0 = weights<T, Degree>(fy.x, wy);

    // value = sum_l wy[l] * r_l with r_l = sum_k wx[k] * c(i0 + k, j0 + l).
    // Same initialisation and loop shape as evaluate_with_gradient, so that floating-point
    // contraction (FMA) treats both identically and the values stay bit-identical.
    T value = T(0);
    for (int l = 0; l <= Degree; ++l) {
        const T r = detail::row_sum<T, Degree>(c, i0, j0 + l, wx);
        value = detail::mul_add(wy[l], r, value);
    }
    return value;
}

/// @brief Evaluate the interpolated value and its gradient at \f$(x, y)\f$.
///
/// Shares the weight computation between value and gradient. With \f$r_l\f$ the row sums
/// of the weights and \f$r'_l\f$ the row sums of the weight derivatives along x:
/// \f$f = \sum_l w^y_l r_l\f$, \f$\partial_y f = s_y \sum_l dw^y_l r_l\f$ and
/// \f$\partial_x f = s_x \sum_l w^y_l r'_l\f$, where \f$s_x, s_y = \pm 1\f$ are the fold
/// orientations. The returned `value` is bit-identical to `evaluate`.
///
/// Non-finite coordinates give NaN for the value and both derivatives.
///
/// @tparam T      Floating-point scalar type.
/// @tparam Degree Spline degree \f$n\f$.
/// @param c Coefficients (output of the prefilter), mirror padding filled.
/// @param x Column coordinate.
/// @param y Row coordinate.
/// @return `{value, dx, dy}`; all NaN if `x` or `y` is not finite.
/// @pre `c.pad >= required_pad<Degree>`, padding filled by `fill_mirror_padding`,
///      `c.width >= 1`, `c.height >= 1`.
template <class T, int Degree>
BSI_HOST_DEVICE inline ValueAndGradient<T> evaluate_with_gradient(CoeffView<const T> c, T x,
                                                                  T y) noexcept {
    static_assert(std::is_floating_point_v<T>, "evaluate_with_gradient: T must be floating point");
    if (!std::isfinite(x) || !std::isfinite(y)) {
        const T nan = detail::make_nan<T>();
        return ValueAndGradient<T>{nan, nan, nan};
    }
    const Folded<T> fx = detail::fold_fast(x, c.width);
    const Folded<T> fy = detail::fold_fast(y, c.height);

    T wx[Degree + 1];
    T dwx[Degree + 1];
    T wy[Degree + 1];
    T dwy[Degree + 1];
    const std::ptrdiff_t i0 = weights_and_derivatives<T, Degree>(fx.x, wx, dwx);
    const std::ptrdiff_t j0 = weights_and_derivatives<T, Degree>(fy.x, wy, dwy);

    T value = T(0);
    T sum_dy = T(0);
    T sum_dx = T(0);
    for (int l = 0; l <= Degree; ++l) {
        const T r = detail::row_sum<T, Degree>(c, i0, j0 + l, wx);
        const T r_dx = detail::row_sum<T, Degree>(c, i0, j0 + l, dwx);
        value = detail::mul_add(wy[l], r, value);
        sum_dy = detail::mul_add(dwy[l], r, sum_dy);
        sum_dx = detail::mul_add(wy[l], r_dx, sum_dx);
    }
    return ValueAndGradient<T>{value, fx.sign * sum_dx, fy.sign * sum_dy};
}

/// @brief Evaluate the interpolated value, gradient and Hessian at \f$(x, y)\f$.
///
/// Extends `evaluate_with_gradient` with the second derivatives. With \f$r_l\f$, \f$r'_l\f$
/// and \f$r''_l\f$ the row sums of the weights and of their first and second derivatives
/// along x: \f$\partial_{yy} f = \sum_l ddw^y_l r_l\f$,
/// \f$\partial_{xy} f = s_x s_y \sum_l dw^y_l r'_l\f$ and
/// \f$\partial_{xx} f = \sum_l w^y_l r''_l\f$, where \f$s_x, s_y = \pm 1\f$ are the fold
/// orientations (second derivatives along one axis are even under the mirror fold). The
/// returned `value`, `dx` and `dy` are bit-identical to `evaluate_with_gradient`.
///
/// For degrees 0 to 2 the Hessian is exact for the piecewise representation: 0 for degree 0;
/// for degree 1 (bilinear) \f$\partial_{xx} = \partial_{yy} = 0\f$ and \f$\partial_{xy}\f$ is
/// piecewise constant and generally non-zero; piecewise constant for degree 2 (right
/// derivative at the knots, in folded coordinates). Degrees >= 3 give a continuous Hessian.
///
/// Non-finite coordinates give NaN for all six outputs.
///
/// @tparam T      Floating-point scalar type.
/// @tparam Degree Spline degree \f$n\f$.
/// @param c Coefficients (output of the prefilter), mirror padding filled.
/// @param x Column coordinate.
/// @param y Row coordinate.
/// @return `{value, dx, dy, dxx, dxy, dyy}`; all NaN if `x` or `y` is not finite.
/// @pre `c.pad >= required_pad<Degree>`, padding filled by `fill_mirror_padding`,
///      `c.width >= 1`, `c.height >= 1`.
template <class T, int Degree>
BSI_HOST_DEVICE inline ValueGradientHessian<T> evaluate_with_hessian(CoeffView<const T> c, T x,
                                                                     T y) noexcept {
    static_assert(std::is_floating_point_v<T>, "evaluate_with_hessian: T must be floating point");
    if (!std::isfinite(x) || !std::isfinite(y)) {
        const T nan = detail::make_nan<T>();
        return ValueGradientHessian<T>{nan, nan, nan, nan, nan, nan};
    }
    const Folded<T> fx = detail::fold_fast(x, c.width);
    const Folded<T> fy = detail::fold_fast(y, c.height);

    T wx[Degree + 1];
    T dwx[Degree + 1];
    T ddwx[Degree + 1];
    T wy[Degree + 1];
    T dwy[Degree + 1];
    T ddwy[Degree + 1];
    const std::ptrdiff_t i0 = weights_and_second_derivatives<T, Degree>(fx.x, wx, dwx, ddwx);
    const std::ptrdiff_t j0 = weights_and_second_derivatives<T, Degree>(fy.x, wy, dwy, ddwy);

    T value = T(0);
    T sum_dy = T(0);
    T sum_dx = T(0);
    T sum_dyy = T(0);
    T sum_dxy = T(0);
    T sum_dxx = T(0);
    for (int l = 0; l <= Degree; ++l) {
        const T r = detail::row_sum<T, Degree>(c, i0, j0 + l, wx);
        const T r_dx = detail::row_sum<T, Degree>(c, i0, j0 + l, dwx);
        const T r_dxx = detail::row_sum<T, Degree>(c, i0, j0 + l, ddwx);
        value = detail::mul_add(wy[l], r, value);
        sum_dy = detail::mul_add(dwy[l], r, sum_dy);
        sum_dx = detail::mul_add(wy[l], r_dx, sum_dx);
        sum_dyy = detail::mul_add(ddwy[l], r, sum_dyy);
        sum_dxy = detail::mul_add(dwy[l], r_dx, sum_dxy);
        sum_dxx = detail::mul_add(wy[l], r_dxx, sum_dxx);
    }
    return ValueGradientHessian<T>{
        value, fx.sign * sum_dx, fy.sign * sum_dy, sum_dxx, fx.sign * fy.sign * sum_dxy, sum_dyy};
}

}  // namespace bsi
