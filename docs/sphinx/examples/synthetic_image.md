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

The speckle has 300 Gaussian spots of standard deviation 2 px on a 128 x 128 image: spots a
few pixels wide, well resolved, as DIC requires.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from bspline_image import BSplineImage
from speckle import make_speckle, sample

SIZE = 128
DEGREE = 5
MARGIN = 8

f = make_speckle(n_spots=300, radius=2.0, extent=(SIZE - 1, SIZE - 1), seed=7)
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

The same computation in C++ (this is the shipped example, which warps with a smooth
displacement field):

```{literalinclude} ../../../examples/synthetic_image.cpp
:language: cpp
```

## Result

```{code-cell} ipython3
inner = (slice(MARGIN, SIZE - MARGIN),) * 2
difference = deformed - deformed_exact

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, data, title in zip(
    axes[:2], (reference, deformed), ("Reference", f"Deformed (degree {DEGREE})")
):
    im = ax.imshow(data, cmap="gray", vmin=0, vmax=reference.max())
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

rms_error = np.sqrt(np.mean(difference[inner] ** 2))
relative = rms_error / reference[inner].std()
print(f"RMS error in the interior: {rms_error:.2e} ({100 * relative:.4f} % of the image std)")
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
