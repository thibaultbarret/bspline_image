// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Poles of the B-spline interpolation filter.
///
/// The direct B-spline transform that maps samples to coefficients is the inverse of the
/// sampled basis \f$b^n[k] = \beta^n(k)\f$, whose \f$z\f$-transform is
/// \f$B^n(z) = \sum_k \beta^n(k) z^{-k}\f$. For \f$n \ge 2\f$ it factors as
/// \f$B^n(z) = \beta^n(0) \prod_{i=0}^{\lfloor n/2 \rfloor - 1}
/// (1 - z_i z^{-1})(1 - z_i z)/(-z_i)\f$ with real poles \f$-1 < z_i < 0\f$
/// (Unser, Aldroubi, Eden; Thevenaz, Blu, Unser 2000).

#include "bsi/config.hpp"

namespace bsi {

/// @brief Number of poles \f$\lfloor n/2 \rfloor\f$ of the degree-\f$n\f$ interpolation filter.
/// @tparam Degree Spline degree \f$n\f$ in [0, 7]. Degrees 0 and 1 have no poles
///         (the samples already are the coefficients).
template <int Degree>
inline constexpr int pole_count = Degree / 2;

/// @brief The \f$i\f$-th pole \f$z_i\f$ of the degree-\f$n\f$ interpolation filter.
///
/// Poles are real, in \f$(-1, 0)\f$, ordered from the one closest to \f$-1\f$ to the
/// one closest to 0. Values are those of Thevenaz, Blu, Unser (2000), to full double
/// precision.
///
/// @tparam Degree Spline degree \f$n\f$ in [0, 7].
/// @param i Pole index, \f$0 \le i <\f$ `pole_count<Degree>`. Precondition; for degrees 0
///        and 1 (no poles) the function returns 0 for any index.
/// @return The pole \f$z_i\f$.
template <int Degree>
BSI_HOST_DEVICE constexpr double pole(int i) noexcept {
    static_assert(0 <= Degree && Degree <= max_degree, "unsupported spline degree");
    if constexpr (Degree == 2) {
        (void)i;
        return -0.17157287525380990239662255158060381;
    } else if constexpr (Degree == 3) {
        (void)i;
        return -0.26794919243112270647255365849412763;
    } else if constexpr (Degree == 4) {
        return i == 0 ? -0.36134122590022017709221284132567525
                      : -0.013725429297339121360331226939128204;
    } else if constexpr (Degree == 5) {
        return i == 0 ? -0.43057534709997379185143478349352011
                      : -0.043096288203264653822712376822550182;
    } else if constexpr (Degree == 6) {
        return i == 0   ? -0.48829458930304475513011803888378906
               : i == 1 ? -0.081679271076237512597937765737059081
                        : -0.0014141518083258177510872439765585925;
    } else if constexpr (Degree == 7) {
        return i == 0   ? -0.53528043079643816554240378168164607
               : i == 1 ? -0.12255461519232669051527226435935734
                        : -0.0091486948096082769285930216516478534;
    } else {
        (void)i;
        return 0.0;
    }
}

}  // namespace bsi
