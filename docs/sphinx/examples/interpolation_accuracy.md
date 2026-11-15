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

The speckle has Gaussian spots of standard deviation 2 px (about a dozen pixels across),
on a 160 x 160 image, with a density of 450 spots per image area. This is a well-resolved
speckle, as a good pattern for DIC should be: the spectrum of a Gaussian with
$\sigma = 2$ is negligible at the Nyquist frequency, so sampling loses essentially no
information. Spot centres are also drawn a little outside the image so that the density is
uniform up to the borders.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from bspline_image import BSplineImage
from speckle import make_speckle, make_speckle_gradient, sample

SIZE = 160
SPECKLE_ARGS = dict(n_spots=450, radius=2.0, extent=(SIZE - 1, SIZE - 1), seed=7)

f = make_speckle(**SPECKLE_ARGS)
grad_f = make_speckle_gradient(**SPECKLE_ARGS)
image = sample(f, SIZE, SIZE)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(image, cmap="gray", origin="upper")
ax.set_xlabel("x (column, px)")
ax.set_ylabel("y (row, px)")
ax.set_title("Reference speckle, 160 x 160")
fig.tight_layout()
plt.show()
```

## Errors at sub-pixel positions

The evaluation points form a grid shifted by 0.37 px in $x$ and 0.61 px in $y$ from the
pixel centres, so none of them coincides with a sample. Only points at least 24 px away
from the image border are kept: the next section shows why this margin is enough for every
degree. For each degree $n = 0, \dots, 7$ the table gives the root-mean-square (RMS) and
maximum absolute error of the value, and the RMS and maximum of the norm of the gradient
error vector, $\lVert \nabla \hat f - \nabla f \rVert$ (degree 0 is piecewise constant and
has no gradient). The last two columns divide the RMS errors by the image standard
deviation and by the RMS norm of the exact gradient, to compare values and gradients on
equal terms.

```{code-cell} ipython3
DEGREES = range(8)
MARGIN = 24
SHIFT_X, SHIFT_Y = 0.37, 0.61

# Every sub-pixel point of the grid (used by the border study) and its distance to the border.
all_idx = np.arange(0, SIZE - 1)
ys_all, xs_all = np.meshgrid(all_idx + SHIFT_Y, all_idx + SHIFT_X, indexing="ij")
border_distance = np.minimum.reduce(
    [xs_all, ys_all, SIZE - 1 - xs_all, SIZE - 1 - ys_all]
)
exact_all = f(xs_all, ys_all)
exact_gx, exact_gy = grad_f(xs_all, ys_all)
interior = border_distance >= MARGIN

value_errors = {}
rms_value, max_value, rms_gradient, max_gradient = [], [], [], []
for degree in DEGREES:
    spline = BSplineImage(image, degree=degree)
    value_errors[degree] = spline.values(xs_all, ys_all) - exact_all
    err = value_errors[degree][interior]
    rms_value.append(np.sqrt(np.mean(err**2)))
    max_value.append(np.abs(err).max())
    if degree == 0:
        rms_gradient.append(np.nan)
        max_gradient.append(np.nan)
        continue
    gx, gy = spline.gradients(xs_all, ys_all)
    gradient_error = np.hypot(gx - exact_gx, gy - exact_gy)[interior]
    rms_gradient.append(np.sqrt(np.mean(gradient_error**2)))
    max_gradient.append(gradient_error.max())

value_scale = image.std()
gradient_scale = np.sqrt(np.mean((exact_gx**2 + exact_gy**2)[interior]))

rows = [
    "| degree | RMS value error | max value error | RMS gradient error | max gradient error "
    "| RMS value / image std | RMS gradient / RMS exact gradient |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for degree in DEGREES:
    if degree == 0:
        gradient_cells = "n/a | n/a | "
        relative_gradient = "n/a"
    else:
        gradient_cells = f"{rms_gradient[degree]:.2e} | {max_gradient[degree]:.2e} | "
        relative_gradient = f"{rms_gradient[degree] / gradient_scale:.2e}"
    rows.append(
        f"| {degree} | {rms_value[degree]:.2e} | {max_value[degree]:.2e} | {gradient_cells}"
        f"{rms_value[degree] / value_scale:.2e} | {relative_gradient} |"
    )
display(Markdown("\n".join(rows)))
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(list(DEGREES), rms_value, "o-", label="value, RMS")
ax.semilogy(list(DEGREES), max_value, "s--", label="value, maximum")
ax.semilogy(list(DEGREES)[1:], rms_gradient[1:], "^-", label="gradient, RMS norm")
ax.semilogy(list(DEGREES)[1:], max_gradient[1:], "v--", label="gradient, maximum norm")
ax.set_xlabel("B-spline degree")
ax.set_ylabel("error (grey levels; speckle std = %.2f)" % image.std())
ax.set_title("Interpolation error versus degree")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
```

## Margin to keep around the region of interest

The B-spline coefficients are computed by a recursive prefilter that assumes the image
continues beyond its borders as a mirror image of itself. Real content does not: the
speckle goes on past the border with its own, different, pattern. The spline is therefore
slightly wrong near the borders, and this perturbation decays into the image, more slowly
for higher degrees, because the prefilter of a high degree has a longer memory. The figure
shows the RMS value error in bands of 1 px of distance to the nearest border.

In practice this tells how much margin to keep around the region that is actually
interpolated. When generating synthetic images, the finite-element mesh (or whatever
region of the image is used for the measurement) is always surrounded by such a margin. The
distance at which each curve reaches its plateau, given below, is the margin needed for that
degree.

```{code-cell} ipython3
BAND_COUNT = 31
PROFILE_DEGREES = (1, 3, 5, 7)
band = np.floor(border_distance).astype(int)

profiles = {
    degree: np.array(
        [np.sqrt(np.mean(value_errors[degree][band == k] ** 2)) for k in range(BAND_COUNT)]
    )
    for degree in PROFILE_DEGREES
}

PLATEAU_TOLERANCE = 1.25
plateau_distance = {}
for degree in PROFILE_DEGREES:
    plateau = rms_value[degree]
    settled = np.nonzero(profiles[degree] > PLATEAU_TOLERANCE * plateau)[0]
    plateau_distance[degree] = int(settled.max()) + 1 if settled.size else 0

fig, ax = plt.subplots(figsize=(6.5, 4))
for degree in PROFILE_DEGREES:
    (line,) = ax.semilogy(range(BAND_COUNT), profiles[degree], label=f"degree {degree}")
    ax.axhline(rms_value[degree], color=line.get_color(), ls=":", lw=1)
ax.axvline(MARGIN, color="k", ls="--", lw=1, label=f"margin used above ({MARGIN} px)")
ax.set_xlabel("distance to the nearest border (px)")
ax.set_ylabel("RMS value error (grey levels)")
ax.set_title("Boundary effect on the interpolation error")
ax.grid(True, which="both", alpha=0.3)
ax.legend(ncols=2, fontsize=8)
fig.tight_layout()
plt.show()

display(Markdown(
    "| degree | error at the border | interior plateau | distance to plateau |\n"
    "|---:|---:|---:|---:|\n"
    + "\n".join(
        f"| {d} | {profiles[d][0]:.2e} | {rms_value[d]:.2e} | {plateau_distance[d]} px |"
        for d in PROFILE_DEGREES
    )
))
```

The dotted horizontal lines are the interior plateaux, the values of the table above. The
table gives the distance beyond which the band error stays within 25 % of the plateau; it
grows with the degree (degree 1 needs no prefilter and shows no border effect), and the 24 px margin
used for the table above is beyond it for every degree.

```{code-cell} ipython3
:tags: [remove-cell]

assert all(rms_value[d + 1] < rms_value[d] for d in range(3, 7)), (
    f"RMS value error is not strictly decreasing from degree 3 to 7: {rms_value}"
)
assert rms_value[5] < rms_value[1] / 10, (
    f"degree 5 error {rms_value[5]:.3e} is not below a tenth of degree 1 error {rms_value[1]:.3e}"
)
assert rms_value[1] / rms_value[7] > 1e3, (
    f"degree 1 / degree 7 error ratio {rms_value[1] / rms_value[7]:.1f} is below 1e3"
)
assert all(
    rms_gradient[d] / gradient_scale > rms_value[d] / value_scale for d in range(1, 8)
), "relative gradient error is not larger than relative value error at every degree"
for d in (3, 5, 7):
    plateau = rms_value[d]
    assert profiles[d][0] > 2 * plateau, (
        f"degree {d}: border band error {profiles[d][0]:.3e} is not above 2x plateau {plateau:.3e}"
    )
    assert profiles[d][MARGIN] < 1.5 * plateau, (
        f"degree {d}: band at {MARGIN} px ({profiles[d][MARGIN]:.3e}) exceeds 1.5x plateau {plateau:.3e}"
    )
    assert plateau_distance[d] < MARGIN, (
        f"degree {d}: plateau distance {plateau_distance[d]} px not below the margin {MARGIN}"
    )
```

## Reading the results

The error falls steeply as the degree rises: between degree 1 and degree 7 the RMS value error
drops by more than three orders of magnitude. Relative to their own scale, gradients are
less accurate than values at every degree, because differentiation amplifies the
high-frequency part of the interpolation error, and degree 1 yields a discontinuous
gradient. For DIC this matters twice: the gradient enters the correlation Jacobian and sets
the convergence of the Gauss-Newton iteration, and interpolation bias on the grey levels
turns into bias on the measured displacement. Cubic splines (degree 3) are a common
compromise; the extra cost of degrees 5 and above buys accuracy when sub-pixel precision is
the goal {cite}`thevenaz2000interpolation`, provided the region of interest keeps the margin
shown above away from the image borders.
