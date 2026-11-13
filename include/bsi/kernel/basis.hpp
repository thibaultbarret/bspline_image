// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Centered cardinal B-spline basis function, interpolation weights and derivatives.
///
/// The centered B-spline of degree \f$n\f$ is the \f$(n+1)\f$-fold convolution of the
/// indicator of \f$[-\frac12, \frac12)\f$ with itself. It is a piecewise polynomial of degree
/// \f$n\f$, symmetric, supported on \f$[-\frac{n+1}{2}, \frac{n+1}{2}]\f$, with unit integral.
/// Its integer translates form a partition of unity:
/// \f$\sum_k \beta^n(x-k) = 1\f$.

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "bsi/config.hpp"

namespace bsi {

namespace detail {

/// @brief Compile-time validation of the (scalar type, degree) template pair.
template <class T, int Degree>
inline constexpr bool valid_basis_params =
    std::is_floating_point_v<T> && 0 <= Degree && Degree <= max_degree;

}  // namespace detail

/// @brief Centered cardinal B-spline \f$\beta^n(x)\f$, written as explicit piecewise
///        polynomials in \f$a = |x|\f$.
///
/// Pieces for \f$a \ge 0\f$ (zero beyond the last one; the outermost piece is always
/// \f$\frac{1}{n!}\left(\frac{n+1}{2}-a\right)^n\f$):
///
/// - \f$n=0\f$: \f$1\f$ if \f$-\frac12 \le x < \frac12\f$, else 0 (half-open convention,
///   not based on \f$|x|\f$).
/// - \f$n=1\f$: \f$1-a\f$ for \f$a<1\f$.
/// - \f$n=2\f$: \f$\frac34-a^2\f$ for \f$a<\frac12\f$;
///   \f$\frac12(\frac32-a)^2\f$ for \f$a<\frac32\f$.
/// - \f$n=3\f$: \f$\frac23-a^2+\frac{a^3}{2}\f$ for \f$a<1\f$;
///   \f$\frac{(2-a)^3}{6}\f$ for \f$a<2\f$.
/// - \f$n=4\f$: \f$\frac{115}{192}-\frac58a^2+\frac14a^4\f$ for \f$a<\frac12\f$;
///   \f$\frac{1}{96}(55+20a-120a^2+80a^3-16a^4)\f$ for \f$a<\frac32\f$;
///   \f$\frac{(5/2-a)^4}{24}\f$ for \f$a<\frac52\f$.
/// - \f$n=5\f$: \f$\frac{11}{20}-\frac{a^2}{2}+\frac{a^4}{4}-\frac{a^5}{12}\f$ for \f$a<1\f$;
///   \f$\frac{17}{40}+\frac58a-\frac74a^2+\frac54a^3-\frac38a^4+\frac{a^5}{24}\f$ for
///   \f$a<2\f$; \f$\frac{(3-a)^5}{120}\f$ for \f$a<3\f$.
/// - \f$n=6\f$: \f$\frac{5887}{11520}-\frac{77}{192}a^2+\frac{7}{48}a^4-\frac{a^6}{36}\f$ for
///   \f$a<\frac12\f$;
///   \f$\frac{7861}{15360}-\frac{7}{768}a-\frac{91}{256}a^2-\frac{35}{288}a^3
///   +\frac{21}{64}a^4-\frac{7}{48}a^5+\frac{a^6}{48}\f$ for \f$a<\frac32\f$;
///   \f$\frac{1379}{7680}+\frac{1267}{960}a-\frac{329}{128}a^2+\frac{133}{72}a^3
///   -\frac{21}{32}a^4+\frac{7}{60}a^5-\frac{a^6}{120}\f$ for \f$a<\frac52\f$;
///   \f$\frac{(7/2-a)^6}{720}\f$ for \f$a<\frac72\f$.
/// - \f$n=7\f$: \f$\frac{151}{315}-\frac{a^2}{3}+\frac{a^4}{9}-\frac{a^6}{36}
///   +\frac{a^7}{144}\f$ for \f$a<1\f$;
///   \f$\frac{103}{210}-\frac{7}{90}a-\frac{a^2}{10}-\frac{7}{18}a^3+\frac{a^4}{2}
///   -\frac{7}{30}a^5+\frac{a^6}{20}-\frac{a^7}{240}\f$ for \f$a<2\f$;
///   \f$-\frac{139}{630}+\frac{217}{90}a-\frac{23}{6}a^2+\frac{49}{18}a^3-\frac{19}{18}a^4
///   +\frac{7}{30}a^5-\frac{a^6}{36}+\frac{a^7}{720}\f$ for \f$a<3\f$;
///   \f$\frac{(4-a)^7}{5040}\f$ for \f$a<4\f$.
///
/// @tparam T      Floating-point scalar type.
/// @tparam Degree Spline degree \f$n\f$, \f$0 \le n \le\f$ `max_degree`.
/// @param x Evaluation point.
/// @return \f$\beta^n(x)\f$.
template <class T, int Degree>
BSI_HOST_DEVICE inline T bspline(T x) noexcept {
    static_assert(detail::valid_basis_params<T, Degree>,
                  "bspline: T must be floating point and 0 <= Degree <= max_degree");
    if constexpr (Degree == 0) {
        return (x >= T(-0.5) && x < T(0.5)) ? T(1) : T(0);
    } else {
        const T a = std::abs(x);
        if constexpr (Degree == 1) {
            return a < T(1) ? T(1) - a : T(0);
        } else if constexpr (Degree == 2) {
            if (a < T(0.5)) return T(0.75) - a * a;
            if (a < T(1.5)) {
                const T d = T(1.5) - a;
                return T(0.5) * d * d;
            }
            return T(0);
        } else if constexpr (Degree == 3) {
            if (a < T(1)) return T(2) / T(3) + a * a * (a / T(2) - T(1));
            if (a < T(2)) {
                const T d = T(2) - a;
                return d * d * d / T(6);
            }
            return T(0);
        } else if constexpr (Degree == 4) {
            const T a2 = a * a;
            if (a < T(0.5)) return T(115) / T(192) + a2 * (T(-5) / T(8) + a2 / T(4));
            if (a < T(1.5)) {
                return (T(55) + a * (T(20) + a * (T(-120) + a * (T(80) - T(16) * a)))) / T(96);
            }
            if (a < T(2.5)) {
                const T d = T(2.5) - a;
                const T d2 = d * d;
                return d2 * d2 / T(24);
            }
            return T(0);
        } else if constexpr (Degree == 5) {
            const T a2 = a * a;
            if (a < T(1)) {
                return T(11) / T(20) + a2 * (T(-1) / T(2) + a2 * (T(1) / T(4) - a / T(12)));
            }
            if (a < T(2)) {
                return T(17) / T(40) +
                       a * (T(5) / T(8) +
                            a * (T(-7) / T(4) +
                                 a * (T(5) / T(4) + a * (T(-3) / T(8) + a / T(24)))));
            }
            if (a < T(3)) {
                const T d = T(3) - a;
                const T d2 = d * d;
                return d2 * d2 * d / T(120);
            }
            return T(0);
        } else if constexpr (Degree == 6) {
            const T a2 = a * a;
            if (a < T(0.5)) {
                return T(5887) / T(11520) +
                       a2 * (T(-77) / T(192) + a2 * (T(7) / T(48) - a2 / T(36)));
            }
            if (a < T(1.5)) {
                return T(7861) / T(15360) +
                       a * (T(-7) / T(768) +
                            a * (T(-91) / T(256) +
                                 a * (T(-35) / T(288) +
                                      a * (T(21) / T(64) + a * (T(-7) / T(48) + a / T(48))))));
            }
            if (a < T(2.5)) {
                return T(1379) / T(7680) +
                       a * (T(1267) / T(960) +
                            a * (T(-329) / T(128) +
                                 a * (T(133) / T(72) +
                                      a * (T(-21) / T(32) + a * (T(7) / T(60) - a / T(120))))));
            }
            if (a < T(3.5)) {
                const T d = T(3.5) - a;
                const T d3 = d * d * d;
                return d3 * d3 / T(720);
            }
            return T(0);
        } else {
            static_assert(Degree == 7);
            const T a2 = a * a;
            if (a < T(1)) {
                return T(151) / T(315) +
                       a2 * (T(-1) / T(3) + a2 * (T(1) / T(9) + a2 * (T(-1) / T(36) + a / T(144))));
            }
            if (a < T(2)) {
                return T(103) / T(210) +
                       a * (T(-7) / T(90) +
                            a * (T(-1) / T(10) +
                                 a * (T(-7) / T(18) +
                                      a * (T(1) / T(2) +
                                           a * (T(-7) / T(30) + a * (T(1) / T(20) - a / T(240)))))));
            }
            if (a < T(3)) {
                return T(-139) / T(630) +
                       a * (T(217) / T(90) +
                            a * (T(-23) / T(6) +
                                 a * (T(49) / T(18) +
                                      a * (T(-19) / T(18) +
                                           a * (T(7) / T(30) + a * (T(-1) / T(36) + a / T(720)))))));
            }
            if (a < T(4)) {
                const T d = T(4) - a;
                const T d2 = d * d;
                return d2 * d2 * d2 * d / T(5040);
            }
            return T(0);
        }
    }
}

/// @brief Index of the first coefficient contributing at position \f$x\f$.
///
/// The \f$n+1\f$ contributing coefficients are \f$i_0, \dots, i_0+n\f$ with
/// \f$i_0 = \lfloor x \rfloor - \lfloor n/2 \rfloor\f$ for odd \f$n\f$ and
/// \f$i_0 = \lfloor x + \frac12 \rfloor - n/2\f$ for even \f$n\f$ (integer division).
///
/// @tparam T      Floating-point scalar type.
/// @tparam Degree Spline degree.
/// @param x Evaluation point (finite, within `std::ptrdiff_t` range).
/// @return \f$i_0\f$.
template <class T, int Degree>
BSI_HOST_DEVICE inline std::ptrdiff_t first_index(T x) noexcept {
    static_assert(detail::valid_basis_params<T, Degree>,
                  "first_index: T must be floating point and 0 <= Degree <= max_degree");
    constexpr std::ptrdiff_t half = Degree / 2;
    if constexpr (Degree % 2 == 1) {
        return static_cast<std::ptrdiff_t>(std::floor(x)) - half;
    } else {
        return static_cast<std::ptrdiff_t>(std::floor(x + T(0.5))) - half;
    }
}

namespace detail {

/// @brief Cardinal B-spline values on a unit grid, from the fractional part.
///
/// With \f$M_{m+1}\f$ the cardinal B-spline of degree \f$m\f$ supported on
/// \f$[0, m+1]\f$ (so \f$\beta^m(y) = M_{m+1}(y + \frac{m+1}{2})\f$), fills
/// \f$v[j] = M_{m+1}(t + j)\f$ for \f$j = 0..m\f$ using the recurrence
/// \f$M_{m+1}(s) = \frac{s\,M_m(s) + (m+1-s)\,M_m(s-1)}{m}\f$.
/// This is the Cox-de Boor recursion specialised to a uniform grid.
///
/// @tparam T Floating-point scalar type.
/// @param t      Fractional position, \f$0 \le t < 1\f$.
/// @param m      Target degree (>= 0), `v` must hold `m + 1` values.
/// @param v      Output buffer of `m + 1` values.
template <class T>
BSI_HOST_DEVICE inline void cardinal_values(T t, int m, T* v) noexcept {
    v[0] = T(1);  // degree 0: M_1(t) = 1 on [0, 1)
    for (int d = 1; d <= m; ++d) {
        v[d] = T(0);
        // In place, descending j so that v[j-1] still holds the degree d-1 value.
        for (int j = d; j >= 1; --j) {
            const T s = t + T(j);
            v[j] = (s * v[j] + (T(d + 1) - s) * v[j - 1]) / T(d);
        }
        v[0] = t * v[0] / T(d);
    }
}

/// @brief Fractional position \f$t \in [0,1)\f$ such that \f$w_k = M_{n+1}(t + n - k)\f$.
template <class T, int Degree>
BSI_HOST_DEVICE inline T fractional_position(T x) noexcept {
    if constexpr (Degree % 2 == 1) {
        return x - std::floor(x);
    } else {
        return x - std::floor(x + T(0.5)) + T(0.5);
    }
}

}  // namespace detail

/// @brief Interpolation weights \f$w_k = \beta^n(x - (i_0 + k))\f$, \f$k = 0..n\f$.
///
/// Computed together from the fractional part of \f$x\f$ with the uniform-grid Cox-de Boor
/// recurrence (no per-weight call to `bspline`). The weights are non-negative and sum to 1.
///
/// @tparam T      Floating-point scalar type.
/// @tparam Degree Spline degree \f$n\f$.
/// @param x Evaluation point (finite).
/// @param[out] w Output array of at least `Degree + 1` values.
/// @return \f$i_0\f$ = `first_index<T, Degree>(x)`.
template <class T, int Degree>
BSI_HOST_DEVICE inline std::ptrdiff_t weights(T x, T* w) noexcept {
    static_assert(detail::valid_basis_params<T, Degree>,
                  "weights: T must be floating point and 0 <= Degree <= max_degree");
    const T t = detail::fractional_position<T, Degree>(x);
    T v[Degree + 1];
    detail::cardinal_values(t, Degree, v);
    for (int k = 0; k <= Degree; ++k) {
        w[k] = v[Degree - k];
    }
    return first_index<T, Degree>(x);
}

/// @brief Weights and their derivatives.
///
/// \f$w_k = \beta^n(u_k)\f$ and \f$dw_k = \beta^{n\prime}(u_k)\f$ with
/// \f$u_k = x - (i_0 + k)\f$, using
/// \f$\beta^{n\prime}(u) = \beta^{n-1}(u+\frac12) - \beta^{n-1}(u-\frac12)\f$ for
/// \f$n \ge 1\f$ and \f$\beta^{0\prime} = 0\f$. For \f$n = 1\f$ the derivative at a knot is
/// the right derivative. The derivatives sum to 0.
///
/// @tparam T      Floating-point scalar type.
/// @tparam Degree Spline degree \f$n\f$.
/// @param x Evaluation point (finite).
/// @param[out] w  Output weights, at least `Degree + 1` values.
/// @param[out] dw Output weight derivatives, at least `Degree + 1` values.
/// @return \f$i_0\f$ = `first_index<T, Degree>(x)`.
template <class T, int Degree>
BSI_HOST_DEVICE inline std::ptrdiff_t weights_and_derivatives(T x, T* w, T* dw) noexcept {
    static_assert(detail::valid_basis_params<T, Degree>,
                  "weights_and_derivatives: T must be floating point and 0 <= Degree <= max_degree");
    const std::ptrdiff_t i0 = weights<T, Degree>(x, w);
    if constexpr (Degree == 0) {
        dw[0] = T(0);
    } else {
        // lower[j] = M_n(t + j), j = 0..n-1 (degree n-1); out of range entries are 0.
        const T t = detail::fractional_position<T, Degree>(x);
        T lower[Degree];
        detail::cardinal_values(t, Degree - 1, lower);
        for (int k = 0; k <= Degree; ++k) {
            const int j = Degree - k;
            const T hi = (j < Degree) ? lower[j] : T(0);
            const T lo = (j >= 1) ? lower[j - 1] : T(0);
            dw[k] = hi - lo;
        }
    }
    return i0;
}

}  // namespace bsi
