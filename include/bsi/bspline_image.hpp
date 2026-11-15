// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// @file
/// @brief Owning B-spline image interpolator with compile-time degree (layer 1, host only).

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "bsi/config.hpp"
#include "bsi/kernel/evaluate.hpp"
#include "bsi/kernel/mirror.hpp"
#include "bsi/kernel/prefilter.hpp"
#include "bsi/types.hpp"

namespace bsi {

/// @brief Pixel types accepted as input images.
template <class P>
concept Pixel = std::same_as<P, std::uint8_t> || std::same_as<P, std::uint16_t> ||
                std::same_as<P, float> || std::same_as<P, double>;

/// @brief B-spline interpolated image of fixed degree.
///
/// Owns the prefiltered coefficients, stored as `T` with a mirror border, and evaluates the
/// interpolated value, gradient and Hessian at arbitrary real coordinates. Pixel (i, j) sits at
/// coordinate (x = i, y = j); coordinates outside the image are folded by mirror symmetry.
/// @tparam T Floating-point type of the stored coefficients and of evaluation results.
/// @tparam Degree Spline degree, in [0, `max_degree`].
template <std::floating_point T, int Degree>
    requires(Degree >= 0 && Degree <= max_degree)
class BSplineImage {
public:
    /// @brief Scalar type of coefficients and results.
    using value_type = T;
    /// @brief Spline degree.
    static constexpr int degree = Degree;

    /// @brief Build the interpolator from an image.
    ///
    /// Prefiltering is always carried out in `double`, whatever `T` is.
    /// @tparam P Pixel type (`uint8_t`, `uint16_t`, `float` or `double`).
    /// @param pixels Input image; it is copied, so it need not outlive the object.
    /// @throws std::invalid_argument If `data` is null, `width < 1`, `height < 1`
    ///         or `stride < width`.
    template <Pixel P>
    explicit BSplineImage(ImageView<const P> pixels)
        : width_(pixels.width), height_(pixels.height) {
        if (pixels.data == nullptr) {
            throw std::invalid_argument("BSplineImage: image data pointer is null");
        }
        if (pixels.width < 1 || pixels.height < 1) {
            throw std::invalid_argument("BSplineImage: width and height must be at least 1");
        }
        if (pixels.stride < pixels.width) {
            throw std::invalid_argument("BSplineImage: stride must be at least width");
        }
        build(pixels);
    }

    /// @brief Build the interpolator from a mutable image view.
    ///
    /// Same as the read-only overload; template deduction cannot apply the implicit
    /// `ImageView<P>` to `ImageView<const P>` conversion, hence this forwarding overload.
    /// @tparam P Pixel type (`uint8_t`, `uint16_t`, `float` or `double`).
    /// @param pixels Input image; it is copied.
    /// @throws std::invalid_argument As for the read-only overload.
    template <Pixel P>
    explicit BSplineImage(ImageView<P> pixels) : BSplineImage(ImageView<const P>(pixels)) {}

    /// @brief Image width in pixels.
    /// @return Number of columns.
    std::ptrdiff_t width() const noexcept { return width_; }

    /// @brief Image height in pixels.
    /// @return Number of rows.
    std::ptrdiff_t height() const noexcept { return height_; }

    /// @brief Interpolated value at (x, y).
    /// @param x Column coordinate (any finite value; mirrored outside the image).
    /// @param y Row coordinate.
    /// @return The interpolated value, or NaN if a coordinate is not finite.
    T value(T x, T y) const noexcept { return evaluate<T, Degree>(coefficients(), x, y); }

    /// @brief Interpolated gradient at (x, y).
    /// @param x Column coordinate.
    /// @param y Row coordinate.
    /// @return Partial derivatives (dx, dy); NaN if a coordinate is not finite.
    Gradient<T> gradient(T x, T y) const noexcept {
        const auto vg = evaluate_with_gradient<T, Degree>(coefficients(), x, y);
        return Gradient<T>{vg.dx, vg.dy};
    }

    /// @brief Interpolated value and gradient at (x, y).
    /// @param x Column coordinate.
    /// @param y Row coordinate.
    /// @return Value and partial derivatives; NaN if a coordinate is not finite.
    ValueAndGradient<T> value_and_gradient(T x, T y) const noexcept {
        return evaluate_with_gradient<T, Degree>(coefficients(), x, y);
    }

    /// @brief Interpolated second derivatives at (x, y).
    ///
    /// Degrees 0 to 2 give exact values for the piecewise representation (degree 0: zero;
    /// degree 1: dxx = dyy = 0 with a piecewise constant dxy; degree 2: piecewise constant) but
    /// are of limited use; degree 3 or more is recommended, and 5 for a full Newton iteration.
    /// @param x Column coordinate.
    /// @param y Row coordinate.
    /// @return Second derivatives (dxx, dxy, dyy); NaN if a coordinate is not finite.
    Hessian<T> hessian(T x, T y) const noexcept {
        const auto r = evaluate_with_hessian<T, Degree>(coefficients(), x, y);
        return Hessian<T>{r.dxx, r.dxy, r.dyy};
    }

    /// @brief Interpolated value, gradient and Hessian at (x, y).
    ///
    /// Value and gradient are identical to those of `value_and_gradient`. See `hessian` for
    /// the recommended degree.
    /// @param x Column coordinate.
    /// @param y Row coordinate.
    /// @return Value, first and second derivatives; NaN if a coordinate is not finite.
    ValueGradientHessian<T> value_gradient_hessian(T x, T y) const noexcept {
        return evaluate_with_hessian<T, Degree>(coefficients(), x, y);
    }

    /// @brief Evaluate the value at many points.
    /// @param xs Column coordinates.
    /// @param ys Row coordinates.
    /// @param out Output values.
    /// @pre `xs`, `ys` and `out` have the same size (checked by `assert` in debug builds).
    void values(std::span<const T> xs, std::span<const T> ys, std::span<T> out) const noexcept {
        assert(xs.size() == ys.size() && xs.size() == out.size());
        const auto c = coefficients();
        for (std::size_t k = 0; k < xs.size(); ++k) {
            out[k] = evaluate<T, Degree>(c, xs[k], ys[k]);
        }
    }

    /// @brief Evaluate the gradient at many points.
    /// @param xs Column coordinates.
    /// @param ys Row coordinates.
    /// @param dx Output x derivatives.
    /// @param dy Output y derivatives.
    /// @pre All spans have the same size (checked by `assert` in debug builds).
    void gradients(std::span<const T> xs, std::span<const T> ys, std::span<T> dx,
                   std::span<T> dy) const noexcept {
        assert(xs.size() == ys.size() && xs.size() == dx.size() && xs.size() == dy.size());
        const auto c = coefficients();
        for (std::size_t k = 0; k < xs.size(); ++k) {
            const auto vg = evaluate_with_gradient<T, Degree>(c, xs[k], ys[k]);
            dx[k] = vg.dx;
            dy[k] = vg.dy;
        }
    }

    /// @brief Evaluate value and gradient at many points.
    /// @param xs Column coordinates.
    /// @param ys Row coordinates.
    /// @param v Output values.
    /// @param dx Output x derivatives.
    /// @param dy Output y derivatives.
    /// @pre All spans have the same size (checked by `assert` in debug builds).
    void values_and_gradients(std::span<const T> xs, std::span<const T> ys, std::span<T> v,
                              std::span<T> dx, std::span<T> dy) const noexcept {
        assert(xs.size() == ys.size() && xs.size() == v.size() && xs.size() == dx.size() &&
               xs.size() == dy.size());
        const auto c = coefficients();
        for (std::size_t k = 0; k < xs.size(); ++k) {
            const auto vg = evaluate_with_gradient<T, Degree>(c, xs[k], ys[k]);
            v[k] = vg.value;
            dx[k] = vg.dx;
            dy[k] = vg.dy;
        }
    }

    /// @brief Evaluate the second derivatives at many points.
    /// @param xs Column coordinates.
    /// @param ys Row coordinates.
    /// @param dxx Output second derivatives along x.
    /// @param dxy Output mixed second derivatives.
    /// @param dyy Output second derivatives along y.
    /// @pre All spans have the same size (checked by `assert` in debug builds).
    void hessians(std::span<const T> xs, std::span<const T> ys, std::span<T> dxx,
                  std::span<T> dxy, std::span<T> dyy) const noexcept {
        assert(xs.size() == ys.size() && xs.size() == dxx.size() && xs.size() == dxy.size() &&
               xs.size() == dyy.size());
        const auto c = coefficients();
        for (std::size_t k = 0; k < xs.size(); ++k) {
            const auto r = evaluate_with_hessian<T, Degree>(c, xs[k], ys[k]);
            dxx[k] = r.dxx;
            dxy[k] = r.dxy;
            dyy[k] = r.dyy;
        }
    }

    /// @brief Evaluate value, gradient and Hessian at many points.
    /// @param xs Column coordinates.
    /// @param ys Row coordinates.
    /// @param v Output values.
    /// @param dx Output x derivatives.
    /// @param dy Output y derivatives.
    /// @param dxx Output second derivatives along x.
    /// @param dxy Output mixed second derivatives.
    /// @param dyy Output second derivatives along y.
    /// @pre All spans have the same size (checked by `assert` in debug builds).
    void values_gradients_hessians(std::span<const T> xs, std::span<const T> ys, std::span<T> v,
                                   std::span<T> dx, std::span<T> dy, std::span<T> dxx,
                                   std::span<T> dxy, std::span<T> dyy) const noexcept {
        assert(xs.size() == ys.size() && xs.size() == v.size() && xs.size() == dx.size() &&
               xs.size() == dy.size() && xs.size() == dxx.size() && xs.size() == dxy.size() &&
               xs.size() == dyy.size());
        const auto c = coefficients();
        for (std::size_t k = 0; k < xs.size(); ++k) {
            const auto r = evaluate_with_hessian<T, Degree>(c, xs[k], ys[k]);
            v[k] = r.value;
            dx[k] = r.dx;
            dy[k] = r.dy;
            dxx[k] = r.dxx;
            dxy[k] = r.dxy;
            dyy[k] = r.dyy;
        }
    }

    /// @brief Read-only view of the padded coefficient array.
    ///
    /// Rebuilt from the owned storage on each call, so it is valid after copy or move of
    /// the object; the view is invalidated when the object is destroyed or reassigned.
    /// @return View with `pad == required_pad<Degree>`.
    CoeffView<const T> coefficients() const noexcept {
        constexpr std::ptrdiff_t pad = required_pad<Degree>;
        const std::ptrdiff_t stride = width_ + 2 * pad;
        return CoeffView<const T>{storage_.data() + pad * stride + pad, width_, height_, stride,
                                  pad};
    }

private:
    template <class P>
    void build(ImageView<const P> pixels) {
        constexpr std::ptrdiff_t pad = required_pad<Degree>;
        const std::ptrdiff_t stride = width_ + 2 * pad;

        std::vector<double> work(static_cast<std::size_t>(width_ * height_));
        for (std::ptrdiff_t j = 0; j < height_; ++j) {
            for (std::ptrdiff_t i = 0; i < width_; ++i) {
                work[static_cast<std::size_t>(j * width_ + i)] =
                    static_cast<double>(pixels.at(i, j));
            }
        }
        for (std::ptrdiff_t j = 0; j < height_; ++j) {
            prefilter_line<Degree>(work.data() + j * width_, width_, 1);
        }
        for (std::ptrdiff_t i = 0; i < width_; ++i) {
            prefilter_line<Degree>(work.data() + i, height_, width_);
        }

        storage_.assign(static_cast<std::size_t>(stride * (height_ + 2 * pad)), T(0));
        const CoeffView<T> view{storage_.data() + pad * stride + pad, width_, height_, stride,
                                pad};
        for (std::ptrdiff_t j = 0; j < height_; ++j) {
            for (std::ptrdiff_t i = 0; i < width_; ++i) {
                view.at(i, j) = static_cast<T>(work[static_cast<std::size_t>(j * width_ + i)]);
            }
        }
        fill_mirror_padding(view);
    }

    std::vector<T> storage_;
    std::ptrdiff_t width_;
    std::ptrdiff_t height_;
};

}  // namespace bsi
