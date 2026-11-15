// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Recursive 1D B-spline prefilter (samples to coefficients) with mirror boundaries.
///
/// Implements the Unser / Thevenaz-Blu-Unser algorithm. The transfer function of the
/// prefilter, \f$H(z) = 1/B^n(z)\f$ with \f$B^n(z) = \sum_k \beta^n(k) z^{-k}\f$, is
/// \f$H(z) = \lambda \prod_i \frac{-z_i}{(1 - z_i z^{-1})(1 - z_i z)}\f$, with the poles
/// \f$z_i\f$ of `bsi::pole` and gain
/// \f$\lambda = \prod_i (1 - z_i)(1 - 1/z_i)\f$. The factor \f$-z_i\f$ comes from the
/// anti-causal recursion \f$c^-_k = z_i (c^-_{k+1} - c^+_k)\f$, whose transfer function is
/// \f$-z_i/(1 - z_i z)\f$. The DC gain is \f$H(1) = 1\f$, since each pole contributes
/// \f$(1-z_i)(1-1/z_i)(-z_i)/(1-z_i)^2 = 1\f$: constants are preserved. Each pole is applied as a
/// causal first-order recursion followed by an anti-causal one. Boundaries are whole-sample
/// symmetric without repeating the edge sample (period \f$2N-2\f$, see
/// `bsi::mirror_index`). Computation is always done in `double`.

#include <cmath>
#include <cstddef>

#include "bsi/config.hpp"
#include "bsi/kernel/poles.hpp"

namespace bsi {

namespace detail {

/// Machine epsilon of `double`, as a literal (no `<limits>` in layer 0).
inline constexpr double prefilter_epsilon = 2.220446049250313e-16;

/// @brief Initial value of the causal recursion for a mirror-extended signal.
///
/// If the truncation horizon \f$\lceil \log\varepsilon / \log|z| \rceil\f$ is shorter than
/// `n`, the truncated sum \f$c_0 + \sum_{k=1}^{h-1} z^k c_k\f$ is used; otherwise the exact
/// sum over the period-\f$(2n-2)\f$ extension is evaluated.
/// @param c First sample of the line.
/// @param n Number of samples, \f$n \ge 2\f$.
/// @param step Stride between samples, in elements.
/// @param z Pole, \f$-1 < z < 0\f$.
/// @return The causal initial condition \f$c^+_0\f$.
BSI_HOST_DEVICE inline double initial_causal(const double* c, std::ptrdiff_t n, std::ptrdiff_t step,
                                             double z) noexcept {
    const std::ptrdiff_t horizon =
        static_cast<std::ptrdiff_t>(std::ceil(std::log(prefilter_epsilon) / std::log(std::abs(z))));
    if (horizon < n) {
        double zk = z;
        double sum = c[0];
        for (std::ptrdiff_t k = 1; k < horizon; ++k) {
            sum += zk * c[k * step];
            zk *= z;
        }
        return sum;
    }
    double zn = z;
    const double iz = 1.0 / z;
    double z2n = std::pow(z, static_cast<double>(n - 1));
    double sum = c[0] + z2n * c[(n - 1) * step];
    z2n = z2n * z2n * iz;
    for (std::ptrdiff_t k = 1; k <= n - 2; ++k) {
        sum += (zn + z2n) * c[k * step];
        zn *= z;
        z2n *= iz;
    }
    return sum / (1.0 - zn * zn);
}

}  // namespace detail

/// @brief In-place prefilter of one line: turns samples into B-spline coefficients.
///
/// The `n` values `c[0], c[step], ..., c[(n-1)*step]` are replaced by the coefficients
/// \f$c_k\f$ such that \f$s(m) = \sum_k c_{\mathrm{mirror}(k)} \beta^n(m-k)\f$ equals the
/// original samples at every integer \f$m \in [0, n-1]\f$. Each sample is first multiplied
/// by the gain \f$\lambda = \prod_i (1-z_i)(1-1/z_i)\f$, then for each pole \f$z_i\f$:
/// causal pass \f$c^+_k = c_k + z_i c^+_{k-1}\f$ and anti-causal pass
/// \f$c^-_k = z_i (c^-_{k+1} - c^+_k)\f$, with mirror initial conditions at both ends.
/// For degrees 0 and 1 the filter is the identity.
///
/// @tparam Degree Spline degree in [0, 7].
/// @param c Pointer to the first sample; must be non-null and valid for `n` strided
///        elements when `n >= 2`.
/// @param n Number of samples. If `n <= 1` the data is left unchanged.
/// @param step Stride between consecutive samples, in elements (non-zero).
template <int Degree>
BSI_HOST_DEVICE void prefilter_line(double* c, std::ptrdiff_t n, std::ptrdiff_t step) noexcept {
    if constexpr (pole_count<Degree> > 0) {
        if (n <= 1) return;

        double lambda = 1.0;
        for (int i = 0; i < pole_count<Degree>; ++i) {
            const double z = pole<Degree>(i);
            lambda *= (1.0 - z) * (1.0 - 1.0 / z);
        }
        for (std::ptrdiff_t k = 0; k < n; ++k) c[k * step] *= lambda;

        for (int i = 0; i < pole_count<Degree>; ++i) {
            const double z = pole<Degree>(i);
            c[0] = detail::initial_causal(c, n, step, z);
            for (std::ptrdiff_t k = 1; k < n; ++k) c[k * step] += z * c[(k - 1) * step];
            c[(n - 1) * step] = (z / (z * z - 1.0)) * (z * c[(n - 2) * step] + c[(n - 1) * step]);
            for (std::ptrdiff_t k = n - 2; k >= 0; --k) {
                c[k * step] = z * (c[(k + 1) * step] - c[k * step]);
            }
        }
    } else {
        (void)c;
        (void)n;
        (void)step;
    }
}

}  // namespace bsi
