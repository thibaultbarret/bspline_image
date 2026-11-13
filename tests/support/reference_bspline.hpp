// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @brief Reference centered B-spline \f$\beta^n(x)\f$ by Cox-de Boor recursion (test oracle).
///
/// \f$\beta^n(x) = \frac{(x+\frac{n+1}{2})\,\beta^{n-1}(x+\frac12)
///   + (\frac{n+1}{2}-x)\,\beta^{n-1}(x-\frac12)}{n}\f$, with
/// \f$\beta^0 = 1\f$ on \f$[-\frac12, \frac12)\f$ and 0 elsewhere.
/// @param n Degree (>= 0).
/// @param x Evaluation point.
/// @return \f$\beta^n(x)\f$.
inline double reference_bspline(int n, double x) {
    if (n == 0) {
        return (x >= -0.5 && x < 0.5) ? 1.0 : 0.0;
    }
    const double c = 0.5 * (n + 1);
    return ((x + c) * reference_bspline(n - 1, x + 0.5) +
            (c - x) * reference_bspline(n - 1, x - 0.5)) /
           n;
}
