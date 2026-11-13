<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Theory {#theory}

This page follows the path taken by the code: from grey levels to B-spline coefficients,
then from coefficients to values and gradients. References: M. Unser, "Splines: A Perfect
Fit for Signal and Image Processing", IEEE Signal Processing Magazine 16(6), 1999; and
P. Thevenaz, T. Blu, M. Unser, "Interpolation Revisited", IEEE Transactions on Medical
Imaging 19(7), 2000.

## The model

The interpolated signal is a linear combination of shifted B-splines:

\f[ s(x) = \sum_{k \in \mathbb{Z}} c_k \, \beta^n(x - k) \f]

\f$\beta^n\f$ is the centered B-spline of degree \f$n\f$, the \f$(n+1)\f$-fold convolution
of the indicator of \f$[-\tfrac12, \tfrac12)\f$ with itself (`bsi::bspline`). It is a
piecewise polynomial, supported on \f$[-\tfrac{n+1}{2}, \tfrac{n+1}{2}]\f$, and its
integer translates sum to 1. Only \f$n+1\f$ coefficients contribute at any \f$x\f$.

## Interpolation condition and prefilter

The spline must reproduce the samples \f$f_m\f$ (the grey levels) at the integers:

\f[ s(m) = \sum_k c_k \, \beta^n(m - k) = (b^n * c)[m] = f_m, \qquad b^n[k] = \beta^n(k). \f]

So the coefficients are obtained by filtering the samples with the inverse of the discrete
filter \f$b^n\f$, whose z-transform is \f$B^n(z) = \sum_k \beta^n(k) z^{-k}\f$:

\f[ C(z) = \frac{F(z)}{B^n(z)}. \f]

For \f$n \ge 2\f$, \f$B^n(z)\f$ factors with real roots \f$z_i \in (-1, 0)\f$, the poles
(`bsi::pole`, `bsi::pole_count` = \f$\lfloor n/2 \rfloor\f$):

\f[ \frac{1}{B^n(z)} = \lambda \prod_{i} \frac{-z_i}{(1 - z_i z^{-1})(1 - z_i z)}, \qquad
\lambda = \prod_i (1 - z_i)(1 - 1/z_i). \f]

Each factor is a stable pair of first-order recursions, one causal and one anti-causal,
so the cost is linear in the number of samples (`bsi::prefilter_line`):

\f[ c^+_k = c_k + z_i \, c^+_{k-1}, \qquad c^-_k = z_i \left( c^-_{k+1} - c^+_k \right). \f]

The samples are first multiplied by \f$\lambda\f$, then each pole is applied in turn.
Degrees 0 and 1 have no pole: the samples already are the coefficients. The image is
prefiltered along rows, then along columns (the filter is separable), once per image.

## Mirror boundary conditions

The signal of \f$N\f$ samples is extended by reflection about its first and last samples,
without repeating them: the extension is even about 0 and periodic with period
\f$2N - 2\f$ (`bsi::mirror_index`, `bsi::fold_mirror`). The prefilter initialises its
causal recursion with this extension, and the coefficient array is padded with mirrored
coefficients (`bsi::fill_mirror_padding`) so that evaluation never tests for borders.
Real coordinates are folded into \f$[0, N-1]\f$ the same way; `bsi::Folded::sign` records
whether the fold reversed the orientation, which flips the sign of derivatives.

## Separable 2-D evaluation

With pixel \f$(i, j)\f$ at \f$(x = i, y = j)\f$, the 2-D interpolant is a tensor product:

\f[ f(x, y) = \sum_{i,j} c_{i,j} \, \beta^n(x - i) \, \beta^n(y - j). \f]

The window of contributing coefficients starts at \f$i_0 = \lfloor x \rfloor - \lfloor n/2
\rfloor\f$ for odd \f$n\f$ and \f$i_0 = \lfloor x + \tfrac12 \rfloor - n/2\f$ for even
\f$n\f$ (`bsi::first_index`); the even-degree rounding is because the centered even-degree
spline has its knots at half-integers. The weights \f$w_k = \beta^n(x - i_0 - k)\f$ come
from the fractional part of \f$x\f$ by the uniform-grid Cox-de Boor recurrence
(`bsi::weights`). Evaluation then costs \f$n+1\f$ row sums along x and one weighted sum
along y (`bsi::evaluate`).

## Gradient

The derivative of a B-spline is a difference of two B-splines of one degree lower:

\f[ \beta^{n\prime}(u) = \beta^{n-1}\!\left(u + \tfrac12\right) - \beta^{n-1}\!\left(u - \tfrac12\right). \f]

Hence \f$\partial_x f\f$ uses the weight derivatives \f$dw_k = \beta^{n\prime}(x - i_0 - k)\f$
in the x direction and ordinary weights in y, and conversely for \f$\partial_y f\f$
(`bsi::weights_and_derivatives`, `bsi::evaluate_with_gradient`). The derivative is exact
for the interpolating spline, not a finite-difference estimate. For degree 1 the derivative
at a knot is the right derivative; degree 0 has zero derivative.
