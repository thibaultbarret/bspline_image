<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Digital image correlation {#dic}

## Choosing the degree

In DIC, a subset of the reference image is matched to the deformed image by minimising a
correlation criterion over sub-pixel displacements. The criterion needs grey levels and
gradients at non-integer positions, so interpolation error becomes measurement error:
a systematic **interpolation bias** that depends on the sub-pixel part of the displacement.

The bias decreases quickly as the degree grows, because high-degree splines approximate
the ideal band-limited interpolator better (Thevenaz, Blu, Unser 2000). Degree 1
(bilinear) is cheap but biased and gives piecewise-constant gradients; degree 3 is a common
compromise; degrees 5 and 7 further reduce the bias on well-resolved speckle patterns.
Odd degrees (3, 5, 7) are the usual choice. Even degrees are supported (the index
convention is \f$\lfloor x + \tfrac12 \rfloor\f$), but offer no advantage in typical DIC.

## Cost

The window grows as \f$(n+1)^2\f$ coefficients, so cost rises with the degree. See
`bench/RESULTS.md` for measured numbers: per-point time roughly doubles from degree 3 to
degree 5 and rises again towards degree 7, while degree 1 is the cheapest. Prefiltering is
paid once per image, so it is negligible compared with millions of evaluations. The
numbers in `bench/RESULTS.md` come from one noisy laptop; treat them qualitatively.

## The visit pattern

`bsi::AnyBSplineImage` selects the degree at runtime, but calling `value()` through it
dispatches on every call. For a hot loop, dispatch **once** with `visit()` and run the loop
on the concrete `bsi::BSplineImage` type, so the compiler can inline and unroll the
fixed-length evaluation:

```cpp
#include "bsi/bsi.hpp"

double ssd_subset(const bsi::AnyBSplineImage<double>& any, double cx, double cy,
                  double shift_x, double shift_y, int half) {
    return any.visit([&](const auto& spline) {          // one dispatch ...
        double ssd = 0.0;
        for (int j = -half; j <= half; ++j) {            // ... then a hot loop on the
            for (int i = -half; i <= half; ++i) {        // concrete BSplineImage<double, D>
                const double ref = spline.value(cx + i, cy + j);
                const auto def = spline.value_and_gradient(cx + i + shift_x, cy + j + shift_y);
                ssd += (def.value - ref) * (def.value - ref);
            }
        }
        return ssd;
    });
}
```

The callable must accept every supported degree and return the same type for all of them.
See `examples/dic_visit.cpp` for a complete program. The batch methods (`values`,
`gradients`, `values_and_gradients`) are the alternative when points are already in arrays;
they dispatch once per call.

For a full Newton scheme, `value_gradient_hessian` and `values_gradients_hessians` return
\f$I\f$, \f$\nabla I\f$ and \f$\nabla^2 I\f$ in one pass. Use degree 3 or more (5 is
recommended): lower degrees have a trivial or discontinuous Hessian.

All evaluation methods are `const`, so one interpolator can be read from several threads
at once; parallelise over subsets in the caller.
