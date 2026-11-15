---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---
<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Interpolation accuracy

How close is the interpolated value, and the interpolated gradient, to the underlying
continuous image? Real images have no known ground truth, so this page builds one: a
speckle that is an analytic sum of Gaussian spots. The analytic function is sampled on
the pixel grid to produce the image, and it can also be evaluated exactly at any
sub-pixel position, which is what the interpolation is compared against. Background on
the method is in {doc}`../theory`; the code below is the Python API described in the
{ref}`user guide <user-guide-dic>`.

## The test image

The speckle has 300 Gaussian spots of standard deviation 2 px on a 128 x 128 image
(`extent` = 127 px, the coordinate of the last pixel). A spot of this size covers about
a dozen pixels across, so the pattern is well resolved, as a good speckle for DIC
should be: the spectrum of a Gaussian with $\sigma = 2$ is negligible at the Nyquist
frequency, so sampling loses essentially no information. Spots are mirrored about the
image borders, so the analytic function has the same symmetry as the interpolator's
mirror boundary handling and the reference stays exact up to the edges.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from bspline_image import BSplineImage
from speckle import make_speckle, make_speckle_gradient, sample

SIZE = 128
SPECKLE_ARGS = dict(n_spots=300, radius=2.0, extent=(SIZE - 1, SIZE - 1), seed=7)

f = make_speckle(**SPECKLE_ARGS)
grad_f = make_speckle_gradient(**SPECKLE_ARGS)
image = sample(f, SIZE, SIZE)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(image, cmap="gray", origin="upper")
ax.set_xlabel("x (column, px)")
ax.set_ylabel("y (row, px)")
ax.set_title("Reference speckle, 128 x 128")
fig.tight_layout()
plt.show()
```

## Errors at sub-pixel positions

The evaluation points form a grid shifted by 0.37 px in $x$ and 0.61 px in $y$ from the
pixel centres, so none of them coincides with a sample. Points closer than 8 px to the
border are left out. For each degree $n = 0, \dots, 7$ the table gives the root-mean-square
and maximum absolute error of the value, and the RMS error of the gradient norm
$\lVert \nabla \hat f - \nabla f \rVert$ (degree 0 is piecewise constant and has no
gradient).

```{code-cell} ipython3
DEGREES = range(8)
MARGIN = 8
SHIFT_X, SHIFT_Y = 0.37, 0.61

inner = np.arange(MARGIN, SIZE - MARGIN)
ys, xs = np.meshgrid(inner + SHIFT_Y, inner + SHIFT_X, indexing="ij")
exact_value = f(xs, ys)
exact_gx, exact_gy = grad_f(xs, ys)

rms_value, max_value, rms_gradient = [], [], []
for degree in DEGREES:
    spline = BSplineImage(image, degree=degree)
    value_error = spline.values(xs, ys) - exact_value
    rms_value.append(np.sqrt(np.mean(value_error**2)))
    max_value.append(np.abs(value_error).max())
    if degree == 0:
        rms_gradient.append(np.nan)
        continue
    gx, gy = spline.gradients(xs, ys)
    rms_gradient.append(np.sqrt(np.mean((gx - exact_gx) ** 2 + (gy - exact_gy) ** 2)))

rows = ["| degree | RMS value error | max value error | RMS gradient error |",
        "|---:|---:|---:|---:|"]
for degree in DEGREES:
    gradient = "n/a" if degree == 0 else f"{rms_gradient[degree]:.2e}"
    rows.append(f"| {degree} | {rms_value[degree]:.2e} | {max_value[degree]:.2e} | {gradient} |")
display(Markdown("\n".join(rows)))
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(list(DEGREES), rms_value, "o-", label="value, RMS")
ax.semilogy(list(DEGREES), max_value, "s--", label="value, maximum")
ax.semilogy(list(DEGREES)[1:], rms_gradient[1:], "^-", label="gradient, RMS")
ax.set_xlabel("B-spline degree")
ax.set_ylabel("error (grey levels, speckle std = %.2f)" % image.std())
ax.set_title("Interpolation error versus degree")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
```

```{code-cell} ipython3
:tags: [remove-cell]

assert all(rms_value[d + 1] < rms_value[d] for d in range(3, 7)), (
    f"RMS value error is not strictly decreasing from degree 3 to 7: {rms_value}"
)
assert rms_value[5] < rms_value[1] / 10, (
    f"degree 5 error {rms_value[5]:.3e} is not below a tenth of degree 1 error {rms_value[1]:.3e}"
)
```

## Reading the results

The error falls steeply as the degree rises: linear interpolation (degree 1)
leaves a value error several orders of magnitude above that of degrees 5 to 7. Gradients are less accurate than values at every degree because
differentiation amplifies the high-frequency part of the interpolation error, and degree 1
yields a discontinuous gradient. For DIC this matters twice: the gradient enters the
correlation Jacobian and sets the convergence of the Gauss-Newton iteration, and interpolation
bias on the grey levels turns into bias on the measured displacement. Cubic splines
(degree 3) are a common compromise; the extra cost of degrees 5 and above buys accuracy when
sub-pixel precision is the goal {cite}`thevenaz2000interpolation`.
