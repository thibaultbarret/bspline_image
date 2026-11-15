---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---
<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Local DIC with the full Newton method

This page solves the problem of {doc}`dic_gauss_newton` again, this time with the exact Hessian
of the criterion

$$
\nabla^2 C = 2 \sum \big( \nabla G_s \nabla G_s^{\mathsf T} + r\, \nabla^2 G_s \big) ,
$$

which needs the second derivatives of the interpolant: `values_gradients_hessians` returns the
value, the two first derivatives and the three second derivatives $G_{xx}$, $G_{xy}$, $G_{yy}$
in one batch call. Gauss-Newton neglects the term $r\, \nabla^2 G_s$, the residual times the
Hessian of the image; Newton keeps it. See the {ref}`user guide <user-guide-dic>` for the
overall DIC workflow.

## Setup

Same pair of images, same 31 x 31 subset far from the borders and same stopping rule as before,
degree 5 and an imposed translation $\mathbf{u}^* = (0.4, 0.3)$ px.

```{code-cell} ipython3
import numpy as np
from IPython.display import Markdown, display

from dic_problem import reference_values, subset_x, subset_y, translated_spline

STOP_TOLERANCE = 1e-6  # px, on the norm of the update
MAX_ITERATIONS = 50
U_STAR = np.array([0.4, 0.3])
START = np.zeros(2)

spline = translated_spline(U_STAR, degree=5)
```

## The two solvers

Both methods share the same loop; only the Hessian differs. For Newton it is assembled from the
second derivatives, with $G_{xy}$ used for both off-diagonal entries.

```{code-cell} ipython3
def gauss_newton_hessian(derivatives, residual):
    _, gx, gy, *_ = derivatives
    jacobian = np.stack([gx, gy], axis=1)
    return jacobian.T @ jacobian


def newton_hessian(derivatives, residual):
    _, gx, gy, gxx, gxy, gyy = derivatives
    jacobian = np.stack([gx, gy], axis=1)
    curvature = np.array([[residual @ gxx, residual @ gxy], [residual @ gxy, residual @ gyy]])
    return jacobian.T @ jacobian + curvature


def minimise(spline, u0, hessian):
    """Newton-type minimisation of C; return the iterates and the norms of the updates."""
    u = np.asarray(u0, dtype=np.float64)
    iterates, update_norms = [u], []
    for _ in range(MAX_ITERATIONS):
        derivatives = spline.values_gradients_hessians(subset_x + u[0], subset_y + u[1])
        values, gx, gy, *_ = derivatives
        residual = values - reference_values
        gradient = np.stack([gx, gy], axis=1).T @ residual  # half of the gradient of C
        step = -np.linalg.solve(hessian(derivatives, residual), gradient)  # factors 2 cancel
        u = u + step
        iterates.append(u)
        update_norms.append(np.linalg.norm(step))
        if update_norms[-1] < STOP_TOLERANCE:
            break
    return np.array(iterates), np.array(update_norms)


gn_iterates, gn_norms = minimise(spline, START, gauss_newton_hessian)
newton_iterates, newton_norms = minimise(spline, START, newton_hessian)
```

## Iterations

```{code-cell} ipython3
def cell(norms, k):
    return f"{norms[k]:.2e}" if k < len(norms) else "-"


rows = ["| Iteration | Gauss-Newton $\\|\\Delta u\\|$ (px) | Newton $\\|\\Delta u\\|$ (px) |", "|---|---|---|"]
for k in range(max(len(gn_norms), len(newton_norms))):
    rows.append(f"| {k + 1} | {cell(gn_norms, k)} | {cell(newton_norms, k)} |")
display(Markdown("\n".join(rows)))

for name, iterates, norms in (("Gauss-Newton", gn_iterates, gn_norms), ("Newton", newton_iterates, newton_norms)):
    error = np.linalg.norm(iterates[-1] - U_STAR)
    print(f"{name}: {len(norms)} iterations, u = {iterates[-1]}, |u - u*| = {error:.1e} px")
```

```{code-cell} ipython3
:tags: [remove-cell]

for iterates, norms in ((gn_iterates, gn_norms), (newton_iterates, newton_norms)):
    assert norms[-1] < STOP_TOLERANCE, norms
    assert np.linalg.norm(iterates[-1] - U_STAR) < 1e-3, iterates[-1]
```

## What this shows

Both methods reach the imposed displacement to well below $10^{-3}$ px in a handful of
iterations, with the update norm shrinking by orders of magnitude at each step. The residual
$r$ vanishes at the solution here (the images match up to the interpolation error), so the
extra term $r\, \nabla^2 G_s$ is small near the solution and Gauss-Newton is already as good as
Newton: the table shows no advantage for the full Hessian in this simple case (it does not
need fewer iterations than Gauss-Newton).
Where the exact Hessian is expected to matter is where $r$ stays large: far from the solution,
with noisy images, or with richer deformation models (affine or higher-order shape functions)
where the second derivatives of the shape functions also contribute. These cases are not
tested here.
