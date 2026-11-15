---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---
<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Synthetic deformed image

Synthetic images with a known deformation are the standard way to validate a DIC
algorithm: the measured displacement can be compared with the one that was imposed. This
page builds such a pair. A reference speckle is sampled, a deformation is imposed, and the
deformed image is obtained by evaluating the B-spline interpolant of the reference at the
*antecedent* position of every pixel. Because the speckle is analytic (see
{doc}`interpolation_accuracy`), the deformed image can also be sampled exactly, which tells
how much the interpolation itself perturbs the synthetic image.

## Reference image

The speckle has Gaussian spots of standard deviation 2 px (about a dozen pixels across), on a
160 x 160 image, with a density of 450 spots per image area: well resolved, as DIC requires.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from bspline_image import BSplineImage
from speckle import make_speckle, sample

SIZE = 160
DEGREE = 5
MARGIN = 24

f = make_speckle(n_spots=450, radius=2.0, extent=(SIZE - 1, SIZE - 1), seed=7)
reference = sample(f, SIZE, SIZE)
```

## Imposed deformation

A pixel at position $\mathbf{x}$ in the reference image moves to
$\mathbf{x}' = \mathbf{c} + (1 + \varepsilon)\, R(\theta)\,(\mathbf{x} - \mathbf{c}) + \mathbf{t}$,
with the centre $\mathbf{c}$ of the image, a dilation $\varepsilon = 1\,\%$, a rotation
$\theta = 2^\circ$ and a translation $\mathbf{t} = (0.3, -0.2)$ px. The grey level of the
deformed image at pixel $\mathbf{x}'$ is that of the reference at $\mathbf{x}$, hence the
reference is evaluated at the antecedent
$\mathbf{x} = \mathbf{c} + R(-\theta)(\mathbf{x}' - \mathbf{c} - \mathbf{t}) / (1 + \varepsilon)$.

```{code-cell} ipython3
TRANSLATION = np.array([0.3, -0.2])
ROTATION = np.deg2rad(2.0)
DILATATION = 0.01
CENTRE = np.array([(SIZE - 1) / 2, (SIZE - 1) / 2])


def antecedent(xs, ys):
    """Reference-image position of the deformed-image positions (xs, ys)."""
    dx = xs - CENTRE[0] - TRANSLATION[0]
    dy = ys - CENTRE[1] - TRANSLATION[1]
    cos, sin = np.cos(ROTATION), np.sin(ROTATION)
    # Inverse rotation R(-theta) applied to (dx, dy), then the inverse dilation.
    rx = (cos * dx + sin * dy) / (1.0 + DILATATION)
    ry = (-sin * dx + cos * dy) / (1.0 + DILATATION)
    return CENTRE[0] + rx, CENTRE[1] + ry


rows, cols = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
source_x, source_y = antecedent(cols, rows)

spline = BSplineImage(reference, degree=DEGREE)
deformed = spline.values(source_x, source_y)
deformed_exact = f(source_x, source_y)
```

A comparable C++ program ships with the library (different displacement field and speckle):

```{literalinclude} ../../../examples/synthetic_image.cpp
:language: cpp
```

## Result

The error is measured in the useful zone, at least 24 px away from the borders, as in
practice where the region of interest (for instance a finite-element mesh) keeps a margin
inside the image. The image borders are perturbed by the prefilter's mirror assumption, see
the margin study in {doc}`interpolation_accuracy`.

```{code-cell} ipython3
inner = (slice(MARGIN, SIZE - MARGIN),) * 2
difference = deformed - deformed_exact

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, data, title in zip(
    axes[:2], (reference, deformed), ("Reference", f"Deformed (degree {DEGREE})")
):
    ax.imshow(data, cmap="gray", vmin=0, vmax=reference.max())
    ax.set_title(title)
vmax = np.abs(difference[inner]).max()
im = axes[2].imshow(difference, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[2].set_title("Interpolated - exact")
fig.colorbar(im, ax=axes[2], fraction=0.046)
for ax in axes:
    ax.set_xlabel("x (column, px)")
    ax.set_ylabel("y (row, px)")
fig.tight_layout()
plt.show()

# As in practice, the comparison is made in the useful zone, which keeps a margin
# around it (the finite-element mesh of a DIC test, say), not up to the image borders.
rms_error = np.sqrt(np.mean(difference[inner] ** 2))
relative = rms_error / reference.std()
print(f"RMS error in the interior (margin {MARGIN} px): {rms_error:.2e} ({100 * relative:.4f} % of the image std)")
```

```{code-cell} ipython3
:tags: [remove-cell]

assert relative < 0.01, (
    f"degree {DEGREE} RMS error {rms_error:.3e} is {100 * relative:.3f} % of the image std, "
    "expected below 1 %"
)
```

## What this shows

The first two panels are visually indistinguishable apart from the sub-pixel shift, rotation
and dilation, which is the point: a synthetic image should only carry the imposed
deformation. The third panel shows what is left, the interpolation error. It follows the
speckle edges, where the grey-level curvature is largest, and its RMS amplitude is a tiny
fraction of the image contrast, so it does not limit the accuracy of a DIC test with a
sub-pixel displacement. This holds for high degrees; lower degrees blur the image and would
bias the measurement, as the {doc}`accuracy study <interpolation_accuracy>` shows
{cite}`thevenaz2000interpolation`. To build such pairs for DIC benchmarks see the
{ref}`user guide <user-guide-dic>`.

## With a real speckle

The same transformation can be applied to a real speckle image, a 512 x 512 photograph of a
speckle pattern in 8-bit grey levels (0 to 255, loaded as floating point). There is no analytic
reference this time, so the interpolation error cannot be measured against an exact image.
Instead the check is a round trip: warp the image, then warp it back with the inverse
transformation, and compare with the original. The deformed image is again built by
evaluating the degree-5 interpolant of the reference at the antecedent positions; the round
trip interpolates that deformed image at the positions where each reference pixel went.

```{code-cell} ipython3
from reference_image import load_speckle_reference

real = load_speckle_reference()
REAL_SIZE = real.shape[0]
REAL_CENTRE = np.array([(REAL_SIZE - 1) / 2, (REAL_SIZE - 1) / 2])


def real_antecedent(xs, ys):
    """Reference-image position of the deformed-image positions (xs, ys)."""
    dx = xs - REAL_CENTRE[0] - TRANSLATION[0]
    dy = ys - REAL_CENTRE[1] - TRANSLATION[1]
    cos, sin = np.cos(ROTATION), np.sin(ROTATION)
    return (
        REAL_CENTRE[0] + (cos * dx + sin * dy) / (1.0 + DILATATION),
        REAL_CENTRE[1] + (-sin * dx + cos * dy) / (1.0 + DILATATION),
    )


def real_forward(xs, ys):
    """Deformed-image position of the reference positions (xs, ys): the forward transformation."""
    dx = xs - REAL_CENTRE[0]
    dy = ys - REAL_CENTRE[1]
    cos, sin = np.cos(ROTATION), np.sin(ROTATION)
    return (
        REAL_CENTRE[0] + (1.0 + DILATATION) * (cos * dx - sin * dy) + TRANSLATION[0],
        REAL_CENTRE[1] + (1.0 + DILATATION) * (sin * dx + cos * dy) + TRANSLATION[1],
    )


real_rows, real_cols = np.mgrid[0:REAL_SIZE, 0:REAL_SIZE].astype(np.float64)
real_deformed = BSplineImage(real, degree=DEGREE).values(*real_antecedent(real_cols, real_rows))
round_trip = BSplineImage(real_deformed, degree=DEGREE).values(*real_forward(real_cols, real_rows))
```

The figure shows the reference, the deformed image and their difference (top), then a 48 x 48
pixel detail of both images, drawn pixel by pixel, and the round-trip error in the useful zone (bottom). The
difference is small near the centre and grows away from it, where the rotation and the dilation
move the image more; only the round trip isolates the interpolation.

```{code-cell} ipython3
ZOOM = 48
zoom = (slice(REAL_SIZE // 2 - ZOOM // 2, REAL_SIZE // 2 + ZOOM // 2),) * 2
real_inner = (slice(MARGIN, REAL_SIZE - MARGIN),) * 2
round_trip_error = round_trip - real
round_trip_map = np.full(real.shape, np.nan)  # only the useful zone is drawn
round_trip_map[real_inner] = round_trip_error[real_inner]

fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
panels = (
    (axes[0, 0], real, "Reference"),
    (axes[0, 1], real_deformed, f"Deformed (degree {DEGREE})"),
    (axes[1, 0], real[zoom], "Reference, detail"),
    (axes[1, 1], real_deformed[zoom], "Deformed, detail"),
)
for ax, data, title in panels:
    # Pixel-by-pixel rendering for the details only; the full images are resampled smoothly.
    ax.imshow(data, cmap="gray", vmin=0, vmax=255, interpolation="nearest" if data.shape[0] == ZOOM else "auto")
    ax.set_title(title)
for ax, data, title in (
    (axes[0, 2], real_deformed - real, "Deformed - reference"),
    (axes[1, 2], round_trip_map, "Round trip - reference (useful zone)"),
):
    limit = np.nanmax(np.abs(data[real_inner]))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="grey levels")
for ax in axes.ravel():
    ax.set_xlabel("x (column, px)")
    ax.set_ylabel("y (row, px)")
fig.tight_layout()
plt.show()

round_trip_rms = np.sqrt(np.mean(round_trip_error[real_inner] ** 2))
round_trip_max = np.abs(round_trip_error[real_inner]).max()
saturated = np.mean(real == 0)
print(
    f"Round trip, interior (margin {MARGIN} px): RMS {round_trip_rms:.2f} grey levels "
    f"({100 * round_trip_rms / real.std():.2f} % of the image std), max {round_trip_max:.2f}"
)
print(f"{100 * saturated:.0f} % of the pixels are saturated at 0")
```

The two successive interpolations leave a residual of well under one percent of the image
standard deviation. The error map shows a lattice: the interpolation error vanishes where the
warped positions fall on whole pixels and peaks half-way between them, so the pattern repeats
about every 27 px, the inverse of the distance $|(1.01 \cos 2^\circ - 1,\ 1.01 \sin 2^\circ)|$,
and is tilted by the rotation. About 12 % of the pixels of this image are saturated at 0: where a smooth speckle
spot meets the clipped floor the grey level has a kink, which a smooth spline follows less
well than the rest of the image. The thresholds of the hidden check are set from the printed
measurement with a wide margin: 1.5 grey levels RMS and 8 grey levels at most.

```{code-cell} ipython3
:tags: [remove-cell]

assert 0.10 < saturated < 0.14, saturated
assert round_trip_rms < 1.5, f"round-trip RMS error {round_trip_rms:.3f} grey levels, expected below 1.5"
assert round_trip_max < 8.0, f"round-trip max error {round_trip_max:.3f} grey levels, expected below 8"
```
