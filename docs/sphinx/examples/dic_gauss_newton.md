---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---
<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Local DIC with Gauss-Newton

Local digital image correlation (DIC) measures the displacement of a small square subset of
pixels between a reference image $F$ and a deformed image $G$. This page estimates a pure
translation $\mathbf{u} = (u_x, u_y)$ by minimising the sum of squared differences

$$
C(\mathbf{u}) = \sum_{\mathbf{x}} \big( G_s(\mathbf{x} + \mathbf{u}) - F(\mathbf{x}) \big)^2 ,
$$

where $G_s$ is the B-spline interpolant of $G$, which provides $G$ at the sub-pixel positions
$\mathbf{x} + \mathbf{u}$ together with its derivatives. With the residual
$r = G_s(\mathbf{x} + \mathbf{u}) - F(\mathbf{x})$, the gradient and Hessian of the criterion are

$$
\nabla C = 2 \sum r\, \nabla G_s , \qquad
\nabla^2 C = 2 \sum \big( \nabla G_s \nabla G_s^{\mathsf T} + r\, \nabla^2 G_s \big) .
$$

Gauss-Newton drops the second term of the Hessian, so it only needs the gradient of $G_s$.
The next page, {doc}`dic_full_newton`, keeps it. For the DIC workflow in general see the
{ref}`user guide <user-guide-dic>`.

## Synthetic test case

$F$ is a speckle sampled on a 96 x 96 image and $G$ is the same speckle translated by a known
$\mathbf{u}^*$ and sampled exactly (see {doc}`synthetic_image`). The subset is a 31 x 31 pixel
square at the centre of the image, far from the borders, so the border effect of the prefilter
does not matter: {doc}`interpolation_accuracy` shows that a margin of about 9 px is enough at
degree 5, and in practice the region of interest always keeps a margin.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from dic_problem import reference_values, subset_x, subset_y, translated_spline

STOP_TOLERANCE = 1e-6  # px, on the norm of the update
MAX_ITERATIONS = 50
```

## Gauss-Newton solver

Each iteration evaluates $G_s$ and its gradient at the subset points shifted by the current
$\mathbf{u}$ in one batch call, then solves the $2 \times 2$ system
$J^{\mathsf T} J \, \Delta\mathbf{u} = -J^{\mathsf T} r$, where the rows of $J$ are the
gradients $\nabla G_s$ (the factor 2 cancels). The iterates, the norms of the updates and a
flag telling whether the stopping rule was met are returned.

```{code-cell} ipython3
def gauss_newton(spline, u0):
    """Minimise C by Gauss-Newton; return (iterates, update norms, converged)."""
    u = np.asarray(u0, dtype=np.float64)
    iterates, update_norms = [u], []
    for _ in range(MAX_ITERATIONS):
        values, gx, gy = spline.values_and_gradients(subset_x + u[0], subset_y + u[1])
        residual = values - reference_values
        jacobian = np.stack([gx, gy], axis=1)
        step = -np.linalg.solve(jacobian.T @ jacobian, jacobian.T @ residual)
        u = u + step
        iterates.append(u)
        update_norms.append(np.linalg.norm(step))
        if update_norms[-1] < STOP_TOLERANCE:
            break
    return np.array(iterates), np.array(update_norms), update_norms[-1] < STOP_TOLERANCE
```

Starting from $\mathbf{u} = 0$ with an imposed $\mathbf{u}^* = (0.4, 0.3)$ px and a degree-5 spline:

```{code-cell} ipython3
spline = translated_spline((0.4, 0.3), degree=5)
iterates, update_norms, converged = gauss_newton(spline, (0.0, 0.0))
print(f"converged in {len(update_norms)} iterations, u = {iterates[-1]} (imposed: [0.4 0.3])")
```

```{code-cell} ipython3
:tags: [remove-cell]

assert converged, update_norms
assert np.allclose(iterates[-1], (0.4, 0.3), atol=1e-3), iterates[-1]
```

## Sub-pixel bias

Interpolation is not exact between the pixel centres, so the minimiser of $C$ is slightly off
$\mathbf{u}^*$ and the error depends on the sub-pixel part of the displacement. This
sub-pixel bias is a classical result {cite}`thevenaz2000interpolation`. The sweep below imposes
$u^*_x$ from 0 to 1 px by steps of 0.05, with $u^*_y = 0.3$, and plots the error
$u_x - u^*_x$ of the Gauss-Newton estimate for several degrees.

With linear interpolation (degree 1) the criterion is not differentiable at integer
displacements, where the bilinear interpolant has a kink: Gauss-Newton then oscillates between
two iterates around the minimum instead of converging. Those points, which never meet the
stopping rule, are drawn as crosses and left out of the table; every other point converged.

```{code-cell} ipython3
DEGREES = (1, 3, 5, 7)
imposed_x = np.linspace(0.0, 1.0, 21)
UY_STAR = 0.3

bias, converged_points = {}, {}
for degree in DEGREES:
    errors, flags = [], []
    for ux_star in imposed_x:
        estimate, _, converged = gauss_newton(translated_spline((ux_star, UY_STAR), degree), (0.0, 0.0))
        errors.append(estimate[-1][0] - ux_star)
        flags.append(converged)
    bias[degree], converged_points[degree] = np.array(errors), np.array(flags)

fig, ax = plt.subplots(figsize=(6.5, 4))
for degree in DEGREES:
    ok = converged_points[degree]
    (line,) = ax.plot(imposed_x[ok], bias[degree][ok], marker="o", markersize=3, label=f"degree {degree}")
    ax.plot(imposed_x[~ok], bias[degree][~ok], "x", color=line.get_color(), markersize=7)
ax.axhline(0.0, color="black", linewidth=0.5)
ax.set_xlabel("imposed displacement $u^*_x$ (px)")
ax.set_ylabel("error $u_x - u^*_x$ (px)")
ax.set_yscale("symlog", linthresh=1e-6)
ax.legend()
fig.tight_layout()
plt.show()

max_bias = {d: np.abs(bias[d][converged_points[d]]).max() for d in DEGREES}
table = ["| Degree | Max. abs. bias (px) |", "|---|---|"]
table += [f"| {d} | {max_bias[d]:.2e} |" for d in DEGREES]
display(Markdown("\n".join(table)))
```

```{code-cell} ipython3
:tags: [remove-cell]

# Only degree 1 at integer displacements may fail to converge.
for degree in DEGREES:
    failed = imposed_x[~converged_points[degree]]
    assert degree == 1 or failed.size == 0, (degree, failed)
    assert np.allclose(failed, np.round(failed)), failed
maxima = [max_bias[d] for d in DEGREES]
assert maxima[2] < maxima[0], f"degree 5 bias {maxima[2]:.2e} should be below degree 1 bias {maxima[0]:.2e}"
assert all(a > b for a, b in zip(maxima, maxima[1:])), maxima
```

## What this shows

The bias oscillates with the sub-pixel part of the displacement and its amplitude falls
quickly as the degree increases: with linear interpolation (degree 1) it reaches several
thousandths of a pixel, while degrees 5 and 7 are orders of magnitude lower, as the table
shows. The vertical axis is symmetric-logarithmic (linear below $10^{-6}$ px) so that all
degrees can be read on one plot. This is the main reason to use a high-degree B-spline in a DIC
code, and the Gauss-Newton solver, which only needs the gradient from `values_and_gradients`,
is enough to reveal it.
