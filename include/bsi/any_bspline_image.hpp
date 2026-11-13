// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Runtime-degree facade over `BSplineImage` (layer 2, host only).

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

#include "bsi/bspline_image.hpp"
#include "bsi/config.hpp"
#include "bsi/types.hpp"

namespace bsi {

namespace detail {

/// @brief Variant holding one `BSplineImage<T, I>` for each `I` in `Is`.
template <class T, class Sequence>
struct variant_of_degrees;

template <class T, std::size_t... Is>
struct variant_of_degrees<T, std::index_sequence<Is...>> {
    using type = std::variant<BSplineImage<T, static_cast<int>(Is)>...>;
};

/// @brief Variant of all supported degrees, indexed by degree.
template <class T>
using any_impl_t =
    typename variant_of_degrees<T, std::make_index_sequence<max_degree + 1>>::type;

/// @brief Build alternative `I` of the variant in place.
template <class T, std::size_t I, class P>
any_impl_t<T> make_alternative(ImageView<const P> pixels) {
    return any_impl_t<T>(std::in_place_index<I>, pixels);
}

/// @brief Build the variant alternative whose degree is `degree`, via a generated table.
/// @throws std::invalid_argument If `degree` is outside [0, `max_degree`], or if the image
///         is invalid (see `BSplineImage`).
template <class T, class P, std::size_t... Is>
any_impl_t<T> make_impl_from(ImageView<const P> pixels, int degree, std::index_sequence<Is...>) {
    using Factory = any_impl_t<T> (*)(ImageView<const P>);
    constexpr std::array<Factory, sizeof...(Is)> table{&make_alternative<T, Is, P>...};
    if (degree < 0 || degree > max_degree) {
        throw std::invalid_argument("AnyBSplineImage: degree must be in [0, max_degree]");
    }
    return table[static_cast<std::size_t>(degree)](pixels);
}

/// @brief Build the variant alternative whose degree is `degree`.
/// @throws std::invalid_argument See `make_impl_from`.
template <class T, class P>
any_impl_t<T> make_impl(ImageView<const P> pixels, int degree) {
    return make_impl_from<T, P>(pixels, degree, std::make_index_sequence<max_degree + 1>{});
}

}  // namespace detail

/// @brief B-spline interpolated image whose degree is chosen at runtime.
///
/// Holds one of the compile-time `BSplineImage` types and forwards every call to it. The
/// forwarding methods dispatch on each call; for hot loops, dispatch once with `visit()`
/// and work on the concrete type, or use the batch methods.
/// @tparam T Floating-point type of the stored coefficients and of evaluation results.
template <std::floating_point T>
class AnyBSplineImage {
public:
    /// @brief Scalar type of coefficients and results.
    using value_type = T;

    /// @brief Build the interpolator from an image, with the given degree.
    /// @tparam P Pixel type (`uint8_t`, `uint16_t`, `float` or `double`).
    /// @param pixels Input image; it is copied, so it need not outlive the object.
    /// @param degree Spline degree, in [0, `max_degree`].
    /// @throws std::invalid_argument If `degree` is outside [0, `max_degree`], or if the
    ///         image is invalid (null data, empty, or stride smaller than width).
    template <Pixel P>
    AnyBSplineImage(ImageView<const P> pixels, int degree)
        : impl_(detail::make_impl<T>(pixels, degree)) {}

    /// @brief Spline degree chosen at construction.
    /// @return Degree in [0, `max_degree`].
    int degree() const noexcept { return static_cast<int>(impl_.index()); }

    /// @brief Image width in pixels.
    /// @return Width.
    std::ptrdiff_t width() const noexcept {
        return std::visit([](const auto& s) noexcept { return s.width(); }, impl_);
    }

    /// @brief Image height in pixels.
    /// @return Height.
    std::ptrdiff_t height() const noexcept {
        return std::visit([](const auto& s) noexcept { return s.height(); }, impl_);
    }

    /// @brief Interpolated value at (x, y).
    /// @param x Horizontal coordinate.
    /// @param y Vertical coordinate.
    /// @return Interpolated value.
    T value(T x, T y) const noexcept {
        return std::visit([&](const auto& s) noexcept { return s.value(x, y); }, impl_);
    }

    /// @brief Interpolated gradient at (x, y).
    /// @param x Horizontal coordinate.
    /// @param y Vertical coordinate.
    /// @return Gradient (dx, dy).
    Gradient<T> gradient(T x, T y) const noexcept {
        return std::visit([&](const auto& s) noexcept { return s.gradient(x, y); }, impl_);
    }

    /// @brief Interpolated value and gradient at (x, y).
    /// @param x Horizontal coordinate.
    /// @param y Vertical coordinate.
    /// @return Value and gradient.
    ValueAndGradient<T> value_and_gradient(T x, T y) const noexcept {
        return std::visit([&](const auto& s) noexcept { return s.value_and_gradient(x, y); },
                          impl_);
    }

    /// @brief Evaluate the value at many points.
    /// @param xs Horizontal coordinates.
    /// @param ys Vertical coordinates, same size as `xs`.
    /// @param out Output values, same size as `xs`.
    void values(std::span<const T> xs, std::span<const T> ys, std::span<T> out) const noexcept {
        std::visit([&](const auto& s) noexcept { s.values(xs, ys, out); }, impl_);
    }

    /// @brief Evaluate the gradient at many points.
    /// @param xs Horizontal coordinates.
    /// @param ys Vertical coordinates, same size as `xs`.
    /// @param dx Output horizontal derivatives, same size as `xs`.
    /// @param dy Output vertical derivatives, same size as `xs`.
    void gradients(std::span<const T> xs, std::span<const T> ys, std::span<T> dx,
                   std::span<T> dy) const noexcept {
        std::visit([&](const auto& s) noexcept { s.gradients(xs, ys, dx, dy); }, impl_);
    }

    /// @brief Evaluate value and gradient at many points.
    /// @param xs Horizontal coordinates.
    /// @param ys Vertical coordinates, same size as `xs`.
    /// @param v Output values, same size as `xs`.
    /// @param dx Output horizontal derivatives, same size as `xs`.
    /// @param dy Output vertical derivatives, same size as `xs`.
    void values_and_gradients(std::span<const T> xs, std::span<const T> ys, std::span<T> v,
                              std::span<T> dx, std::span<T> dy) const noexcept {
        std::visit([&](const auto& s) noexcept { s.values_and_gradients(xs, ys, v, dx, dy); },
                   impl_);
    }

    /// @brief Call `f` with the concrete `BSplineImage` type, dispatching once.
    ///
    /// Intended pattern for DIC: dispatch once, run the hot loop on the concrete type so
    /// that the compiler can inline and vectorize the evaluation.
    /// @code
    /// any.visit([&](const auto& spline) {
    ///     for (std::size_t k = 0; k < n; ++k) {
    ///         out[k] = spline.value(xs[k], ys[k]);   // no per-call dispatch
    ///     }
    /// });
    /// @endcode
    /// @tparam F Callable accepting `const BSplineImage<T, D>&` for every supported `D`, all
    ///           returning the same type.
    /// @param f Callable to invoke.
    /// @return Whatever `f` returns.
    template <class F>
    decltype(auto) visit(F&& f) const {
        return std::visit(std::forward<F>(f), impl_);
    }

private:
    detail::any_impl_t<T> impl_;
};

}  // namespace bsi
