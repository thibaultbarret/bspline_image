// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Compile-time configuration macros and library-wide constants.

/// @def BSI_HOST_DEVICE
/// @brief Function qualifier making a function callable from host and device code.
///
/// Expands to nothing for plain C++ compilers, and to `__host__ __device__`
/// when compiled by CUDA (`__CUDACC__`) or HIP (`__HIPCC__`).
#if defined(__CUDACC__) || defined(__HIPCC__)
#define BSI_HOST_DEVICE __host__ __device__
#else
#define BSI_HOST_DEVICE
#endif

/// @namespace bsi
/// @brief B-spline image interpolation.
namespace bsi {

/// @brief Highest supported spline degree.
inline constexpr int max_degree = 7;

}  // namespace bsi
