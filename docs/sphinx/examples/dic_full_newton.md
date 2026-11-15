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


def minimise(spline, u0, hessian, tolerance=STOP_TOLERANCE):
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
        if update_norms[-1] < tolerance:
            break
    return np.array(iterates), np.array(update_norms)

```

## Case 1: pure translation

```{code-cell} ipython3
def report(gn, newton, u_ref):
    """Show the update norms of both methods as a table and print the final estimates."""
    (gn_iterates, gn_norms), (newton_iterates, newton_norms) = gn, newton

    def cell(norms, k):
        return f"{norms[k]:.2e}" if k < len(norms) else "-"

    rows = ["| Iteration | Gauss-Newton $\\|\\Delta u\\|$ (px) | Newton $\\|\\Delta u\\|$ (px) |", "|---|---|---|"]
    for k in range(max(len(gn_norms), len(newton_norms))):
        rows.append(f"| {k + 1} | {cell(gn_norms, k)} | {cell(newton_norms, k)} |")
    display(Markdown("\n".join(rows)))
    for name, iterates, norms in (("Gauss-Newton", gn_iterates, gn_norms), ("Newton", newton_iterates, newton_norms)):
        print(f"{name}: {len(norms)} iterations, u = {iterates[-1]}, |u - ref| = {np.linalg.norm(iterates[-1] - u_ref):.1e} px")


case1 = (minimise(spline, START, gauss_newton_hessian), minimise(spline, START, newton_hessian))
report(*case1, U_STAR)
```

```{code-cell} ipython3
:tags: [remove-cell]

for iterates, norms in case1:
    assert norms[-1] < STOP_TOLERANCE, norms
    assert np.linalg.norm(iterates[-1] - U_STAR) < 1e-3, iterates[-1]
```

Both methods converge in a handful of iterations, and Newton takes one more than Gauss-Newton
here (4 against 3) because far from the solution the term $r\, \nabla^2 G_s$ is not small and
the first Newton steps are not better. Near the solution the residual $r$ vanishes, since the
deformed image is exactly a translated copy of the reference (up to the interpolation error):
the term $2 \sum r\, \nabla^2 G_s$ that Gauss-Newton drops is then negligible, so
Gauss-Newton is itself nearly quadratic and the full Hessian brings nothing.

## Case 2: a residual that does not vanish

Real images rarely match the model exactly. Here the deformed image has, in addition to the
translation, a contrast and brightness change that the translation model cannot represent:
$G = 0.8\, f(\mathbf{x} - \mathbf{u}^*) + 0.1$ (gain 0.8, offset 0.1). The residual no longer vanishes at the
minimiser, so the dropped term matters. The stopping tolerance is lowered to $10^{-10}$ px to
show the convergence rate.

```{code-cell} ipython3
TIGHT_TOLERANCE = 1e-10
GAIN, OFFSET = 0.8, 0.1
altered_spline = translated_spline(U_STAR, degree=5, gain=GAIN, offset=OFFSET)

case2 = (
    minimise(altered_spline, START, gauss_newton_hessian, TIGHT_TOLERANCE),
    minimise(altered_spline, START, newton_hessian, TIGHT_TOLERANCE),
)
minimiser = case2[1][0][-1]
report(*case2, minimiser)
print(f"minimiser shifted from the imposed translation by {np.linalg.norm(minimiser - U_STAR):.1e} px")
```

```{code-cell} ipython3
:tags: [remove-cell]

(gn2_iterates, gn2_norms), (newton2_iterates, newton2_norms) = case2
assert np.linalg.norm(gn2_iterates[-1] - newton2_iterates[-1]) < 1e-6
assert np.linalg.norm(newton2_iterates[-1] - U_STAR) < 0.05
assert gn2_norms[-1] < TIGHT_TOLERANCE and newton2_norms[-1] < TIGHT_TOLERANCE
# Quadratic convergence of Newton: each update is bounded by a modest multiple of the
# square of the previous one (checked from the second step on, above round-off).
for previous, following in zip(newton2_norms[1:-1], newton2_norms[2:]):
    assert following <= 20.0 * previous**2, newton2_norms
# Linear convergence of Gauss-Newton: successive updates keep a constant ratio, close to
# (1 - gain) / gain, the size of the dropped term relative to J^T J for a contrast change.
tail_ratios = gn2_norms[-5:][1:] / gn2_norms[-5:][:-1]
expected_ratio = (1.0 - GAIN) / GAIN
assert np.all(np.abs(tail_ratios / expected_ratio - 1.0) < 0.3), (tail_ratios, expected_ratio)
assert len(newton2_norms) <= len(gn2_norms)
```

## What this shows

Both methods converge to the same point, which is the minimiser of the criterion; it is
slightly off $\mathbf{u}^*$ because the unmodelled contrast change biases the translation. The
difference is in the convergence rate. Gauss-Newton converges linearly, and the ratio of
successive updates (about 1/4 here) reflects the size of the dropped term relative to
$J^{\mathsf T} J$: for a contrast change the residual is about $(1 - g) f$ against a gradient of
$g \nabla f$, hence a ratio close to $(1 - g)/g = 0.25$ for a gain $g = 0.8$. Newton converges
quadratically (the number of correct digits roughly doubles at each step).

Full Newton pays off when the model leaves a residual correlated with the image curvature,
as with an unmodelled contrast change or shape functions that cannot represent the true motion.
Image noise, by contrast, gives a residual that is uncorrelated with $\nabla^2 G_s$ and hardly
changes the Gauss-Newton rate, and illumination changes are usually handled by a normalised
criterion or by gain and offset parameters. The trade-off is honest: Newton needs fewer
iterations, but evaluates second derivatives at each one (about 1.5 times the cost of
`values_and_gradients` per point at degree 5, see
[`bench/RESULTS.md`](https://github.com/thibaultbarret/bspline_image/blob/main/bench/RESULTS.md)),
and at the tolerances typical of DIC (around $10^{-3}$ px) Gauss-Newton needs only a few more
iterations. `values_gradients_hessians` returns the exact second-derivative term for the whole
subset in one pass.
