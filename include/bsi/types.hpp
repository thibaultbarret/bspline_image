// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Non-owning views and small value types shared by all layers.

#include <cstddef>
#include <type_traits>

#include "bsi/config.hpp"

namespace bsi {

/// @brief Non-owning view of a row-major 2-D image.
/// @tparam P Pixel type (may be const-qualified).
template <class P>
struct ImageView {
    P* data;                ///< Pointer to pixel (0, 0).
    std::ptrdiff_t width;   ///< Number of columns.
    std::ptrdiff_t height;  ///< Number of rows.
    std::ptrdiff_t stride;  ///< Distance in elements between two consecutive rows.

    /// @brief Access a pixel.
    /// @param x Column index, in [0, width).
    /// @param y Row index, in [0, height).
    /// @return Reference to `data[y * stride + x]`.
    /// @pre `data` is valid and (x, y) lies inside the image.
    BSI_HOST_DEVICE constexpr P& at(std::ptrdiff_t x, std::ptrdiff_t y) const noexcept {
        return data[y * stride + x];
    }

    /// @brief Implicit conversion of a mutable view to a read-only view.
    /// @return An `ImageView<const U>` aliasing the same pixels.
    /// @note Only participates when `P` is not already const.
    template <class U = P, class = std::enable_if_t<!std::is_const_v<U>>>
    BSI_HOST_DEVICE constexpr operator ImageView<const U>() const noexcept {
        return ImageView<const U>{data, width, height, stride};
    }
};

/// @brief Non-owning view of a padded B-spline coefficient array.
///
/// The underlying storage is surrounded by `pad` extra cells on each side;
/// `origin` points at coefficient (0, 0).
/// @tparam T Coefficient type (may be const-qualified).
template <class T>
struct CoeffView {
    T* origin;              ///< Pointer to coefficient (0, 0) inside the padded storage.
    std::ptrdiff_t width;   ///< Number of coefficients per row (without padding).
    std::ptrdiff_t height;  ///< Number of coefficient rows (without padding).
    std::ptrdiff_t stride;  ///< Distance in elements between two consecutive storage rows.
    std::ptrdiff_t pad;     ///< Padding cells on each side.

    /// @brief Access a coefficient, padding included.
    /// @param i Column index, in [-pad, width - 1 + pad].
    /// @param j Row index, in [-pad, height - 1 + pad].
    /// @return Reference to `origin[j * stride + i]`.
    /// @pre `origin` is valid and (i, j) lies inside the padded storage.
    BSI_HOST_DEVICE constexpr T& at(std::ptrdiff_t i, std::ptrdiff_t j) const noexcept {
        return origin[j * stride + i];
    }

    /// @brief Implicit conversion of a mutable view to a read-only view.
    /// @return A `CoeffView<const T>` aliasing the same storage.
    /// @note Only participates when `T` is not already const.
    template <class U = T, class = std::enable_if_t<!std::is_const_v<U>>>
    BSI_HOST_DEVICE constexpr operator CoeffView<const U>() const noexcept {
        return CoeffView<const U>{origin, width, height, stride, pad};
    }
};

/// @brief Spatial gradient of an interpolated value.
/// @tparam T Scalar type.
template <class T>
struct Gradient {
    T dx;  ///< Partial derivative along x.
    T dy;  ///< Partial derivative along y.
};

/// @brief Interpolated value together with its spatial gradient.
/// @tparam T Scalar type.
template <class T>
struct ValueAndGradient {
    T value;  ///< Interpolated value.
    T dx;     ///< Partial derivative along x.
    T dy;     ///< Partial derivative along y.
};

/// @brief Second partial derivatives (symmetric Hessian) of the interpolated function.
/// @tparam T Scalar type.
template <class T>
struct Hessian {
    T dxx;  ///< Second partial derivative along x.
    T dxy;  ///< Mixed second partial derivative.
    T dyy;  ///< Second partial derivative along y.
};

/// @brief Interpolated value together with its spatial gradient and Hessian.
/// @tparam T Scalar type.
template <class T>
struct ValueGradientHessian {
    T value;  ///< Interpolated value.
    T dx;     ///< Partial derivative along x.
    T dy;     ///< Partial derivative along y.
    T dxx;    ///< Second partial derivative along x.
    T dxy;    ///< Mixed second partial derivative.
    T dyy;    ///< Second partial derivative along y.
};

}  // namespace bsi
