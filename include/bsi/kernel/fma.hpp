// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Explicit multiply-add shared by the layer-0 kernels.

#include <cmath>
#include <type_traits>

#include "bsi/config.hpp"

namespace bsi::detail {

/// @brief `a * b + c`, fused explicitly when the target has fast hardware FMA.
///
/// Making the fusion explicit (instead of leaving it to the compiler's floating-point
/// contraction) keeps the value accumulation of `evaluate` and `evaluate_with_gradient`
/// bit-identical: contraction and SLP vectorisation may otherwise treat the two kernels
/// differently (`-march=haswell -ffp-contract=fast`). Without fast FMA it is a plain
/// multiply-add, which no contraction can alter.
/// @tparam T Floating-point scalar type.
/// @param a First factor.
/// @param b Second factor.
/// @param c Addend.
/// @return `a * b + c`.
template <class T>
BSI_HOST_DEVICE inline T mul_add(T a, T b, T c) noexcept {
#if defined(FP_FAST_FMA)
    if constexpr (std::is_same_v<T, double>) {
        return std::fma(a, b, c);
    }
#endif
#if defined(FP_FAST_FMAF)
    if constexpr (std::is_same_v<T, float>) {
        return std::fmaf(a, b, c);
    }
#endif
    return a * b + c;
}

}  // namespace bsi::detail
