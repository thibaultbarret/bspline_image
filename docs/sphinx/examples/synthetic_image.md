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

## A speckled element in tension

A real speckle image brings the texture of an actual test, and a deformation closer to the
finite-element setting of DIC. The image pair is made of a 512 x 512 uniform background
(126 grey levels, out of 255) and, at its centre, a square element of 320 x 320 pixels that
carries a real speckle (a photograph in 8-bit grey levels, loaded as floating point). The
element is a unit-size Q4 element, with natural coordinates $(s, t) \in [0, 1]^2$ mapped onto
those pixels by the bilinear shape functions of its four corners. It is loaded in uniaxial
tension along $y$ with its centre fixed: a strain $\varepsilon = 1\,\%$ and a Poisson ratio
$\nu = 0.3$ give the displacement

$$
v = \varepsilon\,(y - y_0) , \qquad u = -\nu\,\varepsilon\,(x - x_0) ,
$$

which is linear, hence reproduced exactly by the Q4 element: moving its four corners is enough.
There is no analytic reference this time, so the interpolation error cannot be measured against
an exact image. The deformed image is built as before, by interpolating the speckle at the
antecedent of each pixel, here with a degree-7 spline, for the pixels inside the deformed
element; the other pixels keep the background grey level. The generation code is in
`reference_image.py`, next to the loader.

```{code-cell} ipython3
import math

from reference_image import (
    BACKGROUND, ELEMENT_FIRST, ELEMENT_SIZE, IMAGE_SIZE, STRAIN, POISSON,
    element_nodes, element_outline, load_speckle_reference, make_element_pair, tension_displacement,
)

speckle = load_speckle_reference()
real_reference, real_deformed, inside = make_element_pair(speckle)
nodes_reference, nodes_deformed = element_nodes()
corner_u, corner_v = tension_displacement(*nodes_reference[0])
print(f"element: {ELEMENT_SIZE} x {ELEMENT_SIZE} px, strain {STRAIN}, Poisson ratio {POISSON}")
print(f"displacement of the top-left corner: ({corner_u:+.2f}, {corner_v:+.2f}) px")
```

## Result

The element corners move the most, by 1.6 px along $y$ and 0.5 px along $x$ at the top-left
corner printed above, while the centre stays in place. The figure shows the reference and the
deformed images with the outline of the element before (dashed red) and after deformation
(cyan), their difference, a 40 x 40 pixel detail of the top-left corner, drawn pixel by pixel,
and the round-trip error described below. The difference between the two images is small near
the centre and grows away from it, where the image moves more.

```{code-cell} ipython3
ZOOM_CENTRE = ELEMENT_FIRST  # the top-left corner of the element
ZOOM_HALF = 20
zoom_range = (ZOOM_CENTRE - ZOOM_HALF, ZOOM_CENTRE + ZOOM_HALF)
ROUND_TRIP_MARGIN = 12  # px, from the element boundary, as in the other examples

# Round trip: interpolate the deformed image (degree 5) at the deformed position of every
# reference pixel; inside the element it should give the reference back.
grid_rows, grid_cols = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float64)
grid_u, grid_v = tension_displacement(grid_cols, grid_rows)
round_trip = BSplineImage(real_deformed, degree=DEGREE).values(grid_cols + grid_u, grid_rows + grid_v)
useful = (slice(ELEMENT_FIRST + ROUND_TRIP_MARGIN, ELEMENT_FIRST + ELEMENT_SIZE - ROUND_TRIP_MARGIN),) * 2
round_trip_error = round_trip - real_reference
round_trip_map = np.full(real_reference.shape, np.nan)  # only the useful zone is drawn
round_trip_map[useful] = round_trip_error[useful]

fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
for ax, data, title in (
    (axes[0, 0], real_reference, "Reference"),
    (axes[0, 1], real_deformed, "Deformed"),
    (axes[1, 0], real_reference, "Reference, corner detail"),
    (axes[1, 1], real_deformed, "Deformed, corner detail"),
):
    ax.imshow(data, cmap="gray", vmin=0, vmax=255, interpolation="nearest" if ax in axes[1] else "auto")
    ax.plot(*element_outline(nodes_reference), "--", color="tab:red", linewidth=1.2)
    ax.plot(*element_outline(nodes_deformed), "-", color="tab:cyan", linewidth=1.2)
    ax.set_title(title)
for ax in axes[1, :2]:
    ax.set_xlim(*zoom_range)
    ax.set_ylim(zoom_range[1], zoom_range[0])
difference = real_deformed - real_reference
for ax, data, title in (
    (axes[0, 2], difference, "Deformed - reference"),
    (axes[1, 2], round_trip_map, "Round trip - reference (useful zone)"),
):
    limit = np.nanmax(np.abs(data[useful]))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="grey levels")
for ax in axes.ravel():
    ax.set_xlabel("x (column, px)")
    ax.set_ylabel("y (row, px)")
fig.tight_layout()
plt.show()

round_trip_rms = np.sqrt(np.mean(round_trip_error[useful] ** 2))
round_trip_max = np.abs(round_trip_error[useful]).max()
saturated = np.mean(speckle == 0)
print(
    f"Round trip, element interior (margin {ROUND_TRIP_MARGIN} px): RMS {round_trip_rms:.2f} grey levels "
    f"({100 * round_trip_rms / speckle.std():.2f} % of the speckle std), max {round_trip_max:.2f}"
)
print(f"{100 * saturated:.0f} % of the speckle pixels are saturated at 0")
```

Two checks are made, both hidden. The generated image must have the deformed element where
the Q4 mapping puts it: the pixels of the element are exactly those whose centre lies inside the
deformed corners, and every other pixel keeps the background grey level. And the round trip must
give the reference back inside the element: the two successive interpolations leave a residual
of about half a grey level RMS, well under one percent of the image standard deviation. About
12 % of the pixels of the speckle are saturated at 0: where a smooth speckle spot meets the
clipped floor the grey level has a kink, which a smooth spline follows less well than the rest of
the image. The thresholds are set from the printed measurement with a margin: 1 grey level RMS
and 6 grey levels at most.

The error map shows a lattice: the interpolation error vanishes where the warped positions fall
on whole pixels and peaks half-way between them, so the pattern repeats where the displacement
changes by one pixel, every 100 px along $y$ (the strain is 1 %) and every 333 px along $x$.

```{code-cell} ipython3
:tags: [remove-cell]

expected_columns = (math.ceil(nodes_deformed[0, 0]), math.floor(nodes_deformed[1, 0]))
expected_rows = (math.ceil(nodes_deformed[0, 1]), math.floor(nodes_deformed[2, 1]))
columns, rows_inside = np.where(inside.any(axis=0))[0], np.where(inside.any(axis=1))[0]
assert (columns[0], columns[-1]) == expected_columns, (columns[[0, -1]], expected_columns)
assert (rows_inside[0], rows_inside[-1]) == expected_rows, (rows_inside[[0, -1]], expected_rows)
assert inside.sum() == (expected_columns[1] - expected_columns[0] + 1) * (expected_rows[1] - expected_rows[0] + 1)
assert np.all(real_deformed[~inside] == BACKGROUND)
assert np.all(real_reference[:ELEMENT_FIRST] == BACKGROUND)
assert 0.10 < saturated < 0.14, saturated
assert round_trip_rms < 1.0, f"round-trip RMS error {round_trip_rms:.3f} grey levels, expected below 1"
assert round_trip_max < 6.0, f"round-trip max error {round_trip_max:.3f} grey levels, expected below 6"
```
