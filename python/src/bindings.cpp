// SPDX-License-Identifier: BSD-3-Clause
#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <bsi/bsi.hpp>

namespace nb = nanobind;

namespace {

template <class P>
using Image = nb::ndarray<const P, nb::ndim<2>, nb::device::cpu>;

template <class T>
using In = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using Out = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
std::span<const T> in_span(const In<T>& a) {
    return {a.data(), a.size()};
}

template <class T>
std::span<T> out_span(const Out<T>& a) {
    return {a.data(), a.size()};
}

template <class... Arrays>
void require_same_size(std::size_t n, const Arrays&... arrays) {
    if (((arrays.size() != n) || ...)) {
        throw std::invalid_argument("all arrays must have the same size");
    }
}

template <class T, class P>
bsi::AnyBSplineImage<T> make_spline(const Image<P>& image, int degree) {
    if (image.stride(1) != 1) {
        throw std::invalid_argument("image rows must be contiguous");
    }
    const bsi::ImageView<const P> view{
        image.data(), static_cast<std::ptrdiff_t>(image.shape(1)),
        static_cast<std::ptrdiff_t>(image.shape(0)),
        static_cast<std::ptrdiff_t>(image.stride(0))};
    return bsi::AnyBSplineImage<T>(view, degree);
}

template <class T, class P>
void add_ctor(nb::class_<bsi::AnyBSplineImage<T>>& cls) {
    cls.def(
        "__init__",
        [](bsi::AnyBSplineImage<T>* self, const Image<P>& image, int degree) {
            new (self) bsi::AnyBSplineImage<T>(make_spline<T, P>(image, degree));
        },
        nb::arg("image").noconvert(), nb::arg("degree"));
}

template <class T>
void bind_spline(nb::module_& m, const char* name) {
    using Spline = bsi::AnyBSplineImage<T>;
    nb::class_<Spline> cls(m, name);
    add_ctor<T, std::uint8_t>(cls);
    add_ctor<T, std::uint16_t>(cls);
    add_ctor<T, float>(cls);
    add_ctor<T, double>(cls);
    cls.def_prop_ro("degree", &Spline::degree)
        .def_prop_ro("width", [](const Spline& s) { return s.width(); })
        .def_prop_ro("height", [](const Spline& s) { return s.height(); })
        .def("value", &Spline::value, nb::arg("x"), nb::arg("y"))
        .def(
            "gradient",
            [](const Spline& s, T x, T y) {
                const auto g = s.gradient(x, y);
                return nb::make_tuple(g.dx, g.dy);
            },
            nb::arg("x"), nb::arg("y"))
        .def(
            "value_and_gradient",
            [](const Spline& s, T x, T y) {
                const auto r = s.value_and_gradient(x, y);
                return nb::make_tuple(r.value, r.dx, r.dy);
            },
            nb::arg("x"), nb::arg("y"))
        .def(
            "values",
            [](const Spline& s, const In<T>& xs, const In<T>& ys, const Out<T>& out) {
                require_same_size(xs.size(), ys, out);
                const auto x = in_span(xs);
                const auto y = in_span(ys);
                const auto o = out_span(out);
                nb::gil_scoped_release release;
                s.values(x, y, o);
            },
            nb::arg("xs"), nb::arg("ys"), nb::arg("out"))
        .def(
            "gradients",
            [](const Spline& s, const In<T>& xs, const In<T>& ys, const Out<T>& dx,
               const Out<T>& dy) {
                require_same_size(xs.size(), ys, dx, dy);
                const auto x = in_span(xs);
                const auto y = in_span(ys);
                const auto gx = out_span(dx);
                const auto gy = out_span(dy);
                nb::gil_scoped_release release;
                s.gradients(x, y, gx, gy);
            },
            nb::arg("xs"), nb::arg("ys"), nb::arg("dx"), nb::arg("dy"))
        .def(
            "values_and_gradients",
            [](const Spline& s, const In<T>& xs, const In<T>& ys, const Out<T>& v,
               const Out<T>& dx, const Out<T>& dy) {
                require_same_size(xs.size(), ys, v, dx, dy);
                const auto x = in_span(xs);
                const auto y = in_span(ys);
                const auto o = out_span(v);
                const auto gx = out_span(dx);
                const auto gy = out_span(dy);
                nb::gil_scoped_release release;
                s.values_and_gradients(x, y, o, gx, gy);
            },
            nb::arg("xs"), nb::arg("ys"), nb::arg("v"), nb::arg("dx"), nb::arg("dy"));
}

}  // namespace

NB_MODULE(_core, m) {
    bind_spline<double>(m, "Spline64");
    bind_spline<float>(m, "Spline32");
}
