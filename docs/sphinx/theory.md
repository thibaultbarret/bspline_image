<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Theory

This page follows the path taken by the code: from grey levels to B-spline coefficients,
then from coefficients to values and gradients. The method is that of {cite}`unser1999splines`
and {cite}`thevenaz2000interpolation`.

## The model

The interpolated signal is a linear combination of shifted B-splines:

$$ s(x) = \sum_{k \in \mathbb{Z}} c_k \, \beta^n(x - k) $$

$\beta^n$ is the centered B-spline of degree $n$, the $(n+1)$-fold convolution
of the indicator of $[-\tfrac12, \tfrac12)$ with itself (`bsi::bspline`). It is a
piecewise polynomial, supported on $[-\tfrac{n+1}{2}, \tfrac{n+1}{2}]$, and its
integer translates sum to 1. Only $n+1$ coefficients contribute at any $x$.

## Interpolation condition and prefilter

The spline must reproduce the samples $f_m$ (the grey levels) at the integers:

$$ s(m) = \sum_k c_k \, \beta^n(m - k) = (b^n * c)[m] = f_m, \qquad b^n[k] = \beta^n(k). $$

So the coefficients are obtained by filtering the samples with the inverse of the discrete
filter $b^n$, whose z-transform is $B^n(z) = \sum_k \beta^n(k) z^{-k}$:

$$ C(z) = \frac{F(z)}{B^n(z)}. $$

For $n \ge 2$, $B^n(z)$ factors with real roots $z_i \in (-1, 0)$, the poles
(`bsi::pole`, `bsi::pole_count` = $\lfloor n/2 \rfloor$):

$$ \frac{1}{B^n(z)} = \lambda \prod_{i} \frac{-z_i}{(1 - z_i z^{-1})(1 - z_i z)}, \qquad
\lambda = \prod_i (1 - z_i)(1 - 1/z_i). $$

Each factor is a stable pair of first-order recursions, one causal and one anti-causal,
so the cost is linear in the number of samples (`bsi::prefilter_line`):

$$ c^+_k = c_k + z_i \, c^+_{k-1}, \qquad c^-_k = z_i \left( c^-_{k+1} - c^+_k \right). $$

The samples are first multiplied by $\lambda$, then each pole is applied in turn.
Degrees 0 and 1 have no pole: the samples already are the coefficients. The image is
prefiltered along rows, then along columns (the filter is separable), once per image.

## Mirror boundary conditions

The signal of $N$ samples is extended by reflection about its first and last samples,
without repeating them: the extension is even about 0 and periodic with period
$2N - 2$ (`bsi::mirror_index`, `bsi::fold_mirror`). The prefilter initialises its
causal recursion with this extension, and the coefficient array is padded with mirrored
coefficients (`bsi::fill_mirror_padding`) so that evaluation never tests for borders.
Real coordinates are folded into $[0, N-1]$ the same way; {bsi}`bsi::Folded::sign` records
whether the fold reversed the orientation, which flips the sign of derivatives.

## Separable 2-D evaluation

With pixel $(i, j)$ at $(x = i, y = j)$, the 2-D interpolant is a tensor product:

$$ f(x, y) = \sum_{i,j} c_{i,j} \, \beta^n(x - i) \, \beta^n(y - j). $$

The window of contributing coefficients starts at $i_0 = \lfloor x \rfloor - \lfloor n/2
\rfloor$ for odd $n$ and $i_0 = \lfloor x + \tfrac12 \rfloor - n/2$ for even
$n$ (`bsi::first_index`); the even-degree rounding is because the centered even-degree
spline has its knots at half-integers. The weights $w_k = \beta^n(x - i_0 - k)$ come
from the fractional part of $x$ by the uniform-grid Cox-de Boor recurrence
(`bsi::weights`). Evaluation then costs $n+1$ row sums along x and one weighted sum
along y (`bsi::evaluate`).

## Gradient

The derivative of a B-spline is a difference of two B-splines of one degree lower:

$$ \beta^{n\prime}(u) = \beta^{n-1}\!\left(u + \tfrac12\right) - \beta^{n-1}\!\left(u - \tfrac12\right). $$

Hence $\partial_x f$ uses the weight derivatives $dw_k = \beta^{n\prime}(x - i_0 - k)$
in the x direction and ordinary weights in y, and conversely for $\partial_y f$
(`bsi::weights_and_derivatives`, `bsi::evaluate_with_gradient`). The derivative is exact
for the interpolating spline, not a finite-difference estimate. For degree 1 the derivative
at a knot is the right derivative; degree 0 has zero derivative.

## Second derivatives

Differentiating once more gives a difference of three B-splines of degree $n-2$:

$$ \beta^{n\prime\prime}(u) = \beta^{n-2}(u+1) - 2\,\beta^{n-2}(u) + \beta^{n-2}(u-1), $$

valid for $n \ge 2$ (`bsi::weights_and_second_derivatives`, `bsi::evaluate_with_hessian`).
$\partial_{xx} f$ uses the second-derivative weights along x and ordinary weights along
y, $\partial_{yy} f$ the converse, and $\partial_{xy} f$ the first-derivative weights
along both axes. Under the mirror fold, $\partial_{xx}$ and $\partial_{yy}$ are even
(unchanged), while $\partial_{xy}$ is multiplied by $s_x s_y$, the product of the fold
orientations. For degree 1, $\partial_{xx} = \partial_{yy} = 0$ but $\partial_{xy}$
is piecewise constant and generally non-zero; degree 2 is piecewise constant.

## References

```{bibliography}
:cited:
```
