---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---
<!-- SPDX-License-Identifier: BSD-3-Clause -->
# DIC parallelised over subsets

Subsets are independent of each other, so a field of subsets is the natural place to use
several threads. The evaluation methods of the interpolator are `const` and the computation
releases the Python lock, so a plain `ThreadPoolExecutor` can share **one** `BSplineImage`
between all the workers, without any copy and without any lock. This page runs the full Newton
solver of {doc}`dic_full_newton` on a grid of subsets, first sequentially and then with
several thread counts. See the {ref}`user guide <user-guide-threads>` for the advice on the
number of threads.

## Setup

The image pair is the speckled element in tension of {doc}`synthetic_image`: a 512 x 512 image
with a uniform background of 126 grey levels and, at its centre, a 320 x 320 pixel Q4 element
carrying a real speckle, stretched by 1 % along $y$ with a Poisson ratio of 0.3 and its centre
fixed. The displacement is the linear field $v = \varepsilon\,(y - y_0)$,
$u = -\nu\,\varepsilon\,(x - x_0)$. The deformed image is generated with a **degree-7**
interpolant of the speckle; the correlation then interpolates it with **degree 5**. Using a
higher degree for the synthetic image avoids measuring the displacement with the very
interpolator that produced it.

Subsets are 31 x 31 pixels, one every 32 pixels, strictly inside the element: their edges stay
at least 8 px from the element boundary, where the spline of the deformed image would see the
jump to the background. This also keeps them far from the image borders, as the margin found in
{doc}`interpolation_accuracy` requires. Each subset is solved for a translation only, and the
solution is compared with the theoretical displacement at the subset centre.

```{code-cell} ipython3
import os
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from bspline_image import BSplineImage
from reference_image import (
    ELEMENT_FIRST, ELEMENT_SIZE, POISSON, STRAIN, element_nodes, element_outline,
    load_speckle_reference, make_element_pair, tension_displacement,
)

HALF_SIZE = 15  # the subset is (2 * HALF_SIZE + 1) pixels wide
ELEMENT_MARGIN = 8  # px between the subset edges and the element boundary
STEP = 32
DIC_DEGREE = 5  # degree of the interpolant used for the correlation
STOP_TOLERANCE = 1e-8  # px, on the norm of the update
MAX_ITERATIONS = 20
REPEATS = 3  # each timing is the best of this many runs

reference, deformed, _ = make_element_pair(load_speckle_reference())
spline = BSplineImage(deformed, degree=DIC_DEGREE)  # shared by all the threads, read-only

offsets = np.arange(-HALF_SIZE, HALF_SIZE + 1)
offset_x, offset_y = (a.ravel().astype(np.float64) for a in np.meshgrid(offsets, offsets))
first = ELEMENT_FIRST + ELEMENT_MARGIN + HALF_SIZE
stop = ELEMENT_FIRST + ELEMENT_SIZE - ELEMENT_MARGIN - HALF_SIZE
axis_centres = np.arange(first, stop, STEP)
centres = [(cx, cy) for cy in axis_centres for cx in axis_centres]
centre_array = np.array(centres, dtype=np.float64)
print(f"{len(centres)} subsets of {offset_x.size} pixels")
```

The subsets, framed on the reference image with the outline of the element:

```{code-cell} ipython3
from matplotlib.patches import Rectangle

nodes_reference, nodes_deformed = element_nodes()
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.imshow(reference, cmap="gray", vmin=0, vmax=255)
for cx, cy in centres:
    ax.add_patch(Rectangle((cx - HALF_SIZE - 0.5, cy - HALF_SIZE - 0.5), 2 * HALF_SIZE + 1, 2 * HALF_SIZE + 1,
                           fill=False, edgecolor="tab:red", linewidth=0.8))
ax.plot(*element_outline(nodes_reference), color="tab:cyan", linewidth=1.2)
ax.set_xlabel("x (column, px)")
ax.set_ylabel("y (row, px)")
ax.set_title(f"{len(centres)} subsets of {2 * HALF_SIZE + 1} x {2 * HALF_SIZE + 1} pixels")
fig.tight_layout()
plt.show()
```

## The solver

`solve_subset` runs the full Newton iteration for one subset and returns the displacement and
the number of iterations. It only reads `spline` and `reference`, and builds its own arrays,
which is what makes it safe to call from several threads. The displacement reaches 1.6 px at the
corners of the element, more than the speckle grains can tolerate for a start from zero, so each
subset starts from the nearest integer displacement of the theoretical field, as a coarse
whole-pixel search would provide.

```{code-cell} ipython3
def solve_subset(center):
    """Full Newton minimisation of the SSD for the subset centred on pixel (cx, cy)."""
    cx, cy = center
    px, py = cx + offset_x, cy + offset_y
    reference_values = reference[py.astype(int), px.astype(int)]
    u = np.round(np.array(tension_displacement(cx, cy)))  # whole-pixel starting point
    for iteration in range(1, MAX_ITERATIONS + 1):
        g, gx, gy, gxx, gxy, gyy = spline.values_gradients_hessians(px + u[0], py + u[1])
        residual = g - reference_values
        gradient = np.array([residual @ gx, residual @ gy])
        hessian = np.array(
            [[gx @ gx + residual @ gxx, gx @ gy + residual @ gxy],
             [gx @ gy + residual @ gxy, gy @ gy + residual @ gyy]]
        )
        step = -np.linalg.solve(hessian, gradient)
        u = u + step
        if np.linalg.norm(step) < STOP_TOLERANCE:
            break
    return u, iteration


def solve_all(n_threads):
    """Solve every subset; with one thread the work stays in the calling thread."""
    if n_threads == 1:
        return [solve_subset(c) for c in centres]
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        return list(pool.map(solve_subset, centres))


def best_time(n_threads):
    """Return (best wall time, results) over REPEATS runs."""
    best, results = float("inf"), None
    for _ in range(REPEATS):
        start = time.perf_counter()
        results = solve_all(n_threads)
        best = min(best, time.perf_counter() - start)
    return best, results
```

## Sequential against parallel

The thread counts are 1, 2, 4, the number of logical cores reported by `os.cpu_count()`, and
a count larger than the number of subsets (the extra workers simply stay idle). The first
call is a warm-up.

```{code-cell} ipython3
solve_all(1)  # warm-up
cpu_count = os.cpu_count() or 1
thread_counts = sorted({1, 2, 4, cpu_count, len(centres) + 8})

runs = {n: best_time(n) for n in thread_counts}
t_sequential, sequential = runs[1]

rows = ["| Threads | Time (ms) | Speed-up |", "|---|---|---|"]
for n, (elapsed, _) in runs.items():
    rows.append(f"| {n} | {1e3 * elapsed:.0f} | {t_sequential / elapsed:.2f} |")
display(Markdown("\n".join(rows)))

displacements = np.array([u for u, _ in sequential])
theory = np.stack(tension_displacement(centre_array[:, 0], centre_array[:, 1]), axis=1)
errors = np.linalg.norm(displacements - theory, axis=1)
iterations = [it for _, it in sequential]
print(f"{len(centres)} subsets, {min(iterations)} to {max(iterations)} Newton iterations each, "
      f"max |u - u_theory| = {errors.max():.1e} px, mean {errors.mean():.1e} px, {cpu_count} logical cores")
```

## Measured against theoretical field

The measured displacement is compared with the theoretical field at each subset centre, and the
strains are estimated by a linear regression of the measured field,
$u \approx a_x + \varepsilon_{xx}\, x + \gamma_{xy}\, y$ and
$v \approx a_y + \gamma_{yx}\, x + \varepsilon_{yy}\, y$.

```{code-cell} ipython3
design = np.column_stack([np.ones(len(centres)), centre_array])
coef_u, coef_v = (np.linalg.lstsq(design, displacements[:, k], rcond=None)[0] for k in (0, 1))
strain_xx, strain_yy = coef_u[1], coef_v[2]
print(f"estimated strain eps_yy = {strain_yy:.6f} (imposed {STRAIN}), eps_xx = {strain_xx:.6f} "
      f"(imposed {-POISSON * STRAIN})")

grid_shape = (len(axis_centres),) * 2
last = axis_centres[-1]
extent = (first - STEP / 2, last + STEP / 2, last + STEP / 2, first - STEP / 2)
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for row, (name, k) in enumerate((("u_x", 0), ("u_y", 1))):
    limit = np.abs(theory[:, k]).max()
    for ax, data, kind in ((axes[row, 0], displacements[:, k], "measured"), (axes[row, 1], theory[:, k], "theoretical")):
        im = ax.imshow(data.reshape(grid_shape), extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"{kind} ${name}$")
        fig.colorbar(im, ax=ax, fraction=0.046, label="px")
axes[0, 2].axis("off")
im = axes[1, 2].imshow(errors.reshape(grid_shape), extent=extent, cmap="magma", vmin=0)
axes[1, 2].set_title("$|\\mathbf{u} - \\mathbf{u}_{theory}|$")
fig.colorbar(im, ax=axes[1, 2], fraction=0.046, label="px")
for ax in axes.ravel()[[0, 1, 3, 4, 5]]:
    ax.set_xlabel("subset centre x (px)")
    ax.set_ylabel("subset centre y (px)")
fig.tight_layout()
plt.show()
```

```{code-cell} ipython3
:tags: [remove-cell]

ERROR_THRESHOLD = 3e-2  # px, about twice the measured maximum (see the text)
STRAIN_TOLERANCE = 1e-4  # about ten times the measured strain error
for n, (_, results) in runs.items():
    assert len(results) == len(centres)
    assert np.array_equal(np.array([u for u, _ in results]), displacements), n
    assert [it for _, it in results] == iterations, n
assert errors.max() < ERROR_THRESHOLD, errors.max()
assert abs(strain_yy - STRAIN) < STRAIN_TOLERANCE, strain_yy
assert abs(strain_xx + POISSON * STRAIN) < STRAIN_TOLERANCE, strain_xx
assert max(iterations) < MAX_ITERATIONS, iterations
assert len(centres) > 25
assert max(thread_counts) > len(centres)
```

Every thread count gives exactly the same displacements as the sequential run: the subsets are
independent and each one performs the same arithmetic whatever thread runs it.

A translation-only subset is nevertheless unbiased at its centre for this field. The motion
inside a subset is the centre displacement plus a linear part, $\mathbf{u}(\mathbf{x}) =
\mathbf{u}_c + \nabla\mathbf{u}\,(\mathbf{x} - \mathbf{x}_c)$, which is odd about the centre:
a translation cannot represent it, but because the subset is symmetric the linear part averages
to zero, and the translation that best matches the subset is $\mathbf{u}_c$. What remains is a
second-order effect, the unmodelled stretch of the speckle inside the subset (about 0.3 px
between its centre and its edge along $y$, 1 % of 31 px). It acts as noise on each subset, so
the error shown above is of the order of $10^{-2}$ px, far above the interpolation bias, but
it has no systematic sign: the regression on the subsets recovers the strains to about
$10^{-5}$ (printed above). The hidden checks require the maximum error to stay below $3 \times 10^{-2}$ px, about
twice the measured maximum, and the strains to be within $10^{-4}$ of the imposed values, a
margin for small numerical differences between platforms that still catches a regression.

## How many threads

The speed-up depends on the machine, so nothing here asserts on it. As a rule of thumb, use about
as many threads as **physical** cores: `os.cpu_count()` counts logical cores, and hyper-threads
share the arithmetic units of their core, so they add little to this compute-bound kernel. Beyond
that, extra threads only add scheduling cost, and the largest counts of the table show little or no
gain. With few subsets the work is also unevenly shared (the last threads wait for the others), and
a thread count above the number of subsets brings nothing.

Only the interpolator calls run outside the lock. The NumPy code of `solve_subset` around them
(building the 2 x 2 Hessian, the small linear solve, the Python loop) holds the lock, and with
subsets of this size it is a large share of the time, so the Python speed-up stays well below
the number of cores (about 1.5 at 8 threads on the machine described in the
{ref}`user guide <user-guide-threads>`, against about 6 for the batch evaluation alone in C++).
The larger the share of the interpolator in the task (bigger subsets, more points per call), the
closer the scaling gets to the number of cores. With much smaller tasks, hand each worker a block of several subsets instead of one.

## The same in C++

The C++ version distributes the subsets over `std::thread` workers and shares one `const`
`BSplineImage<double, 5>` between them. It is built and run with the other examples.

```{literalinclude} ../../../examples/dic_parallel_subsets.cpp
:language: cpp
```
