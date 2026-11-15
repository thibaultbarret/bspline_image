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

A 160 x 160 speckle image and the same image translated by $\mathbf{u}^* = (0.4, 0.3)$ px,
interpolated with degree 5. Subsets are 31 x 31 pixels, one every 8 pixels, and kept at least
24 px from the border, the margin found in {doc}`interpolation_accuracy`.

```{code-cell} ipython3
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from IPython.display import Markdown, display

from bspline_image import BSplineImage
from speckle import make_speckle, sample

SIZE = 160
HALF_SIZE = 15  # the subset is (2 * HALF_SIZE + 1) pixels wide
MARGIN = 24
STEP = 8
U_STAR = np.array([0.4, 0.3])
STOP_TOLERANCE = 1e-8  # px, on the norm of the update
MAX_ITERATIONS = 20
REPEATS = 3  # each timing is the best of this many runs

speckle = make_speckle(n_spots=1250, radius=2.0, extent=(SIZE - 1, SIZE - 1), seed=11)
reference = sample(speckle, SIZE, SIZE)
deformed = sample(lambda x, y: speckle(x - U_STAR[0], y - U_STAR[1]), SIZE, SIZE)
spline = BSplineImage(deformed, degree=5)  # shared by all the threads, read-only

offsets = np.arange(-HALF_SIZE, HALF_SIZE + 1)
offset_x, offset_y = (a.ravel().astype(np.float64) for a in np.meshgrid(offsets, offsets))
first = MARGIN + HALF_SIZE
centres = [(cx, cy) for cy in range(first, SIZE - first, STEP) for cx in range(first, SIZE - first, STEP)]
print(f"{len(centres)} subsets of {offset_x.size} pixels")
```

## The solver

`solve_subset` runs the full Newton iteration for one subset and returns the displacement and
the number of iterations. It only reads `spline` and `reference`, and builds its own arrays,
which is what makes it safe to call from several threads.

```{code-cell} ipython3
def solve_subset(center):
    """Full Newton minimisation of the SSD for the subset centred on pixel (cx, cy)."""
    cx, cy = center
    px, py = cx + offset_x, cy + offset_y
    reference_values = reference[py.astype(int), px.astype(int)]
    u = np.zeros(2)
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
errors = np.linalg.norm(displacements - U_STAR, axis=1)
iterations = [it for _, it in sequential]
print(f"{len(centres)} subsets, {min(iterations)} to {max(iterations)} Newton iterations each, "
      f"max |u - u*| = {errors.max():.1e} px, {cpu_count} logical cores")
```

```{code-cell} ipython3
:tags: [remove-cell]

for n, (_, results) in runs.items():
    assert len(results) == len(centres)
    assert np.array_equal(np.array([u for u, _ in results]), displacements), n
    assert [it for _, it in results] == iterations, n
assert errors.max() < 1e-3, errors.max()
assert max(iterations) < MAX_ITERATIONS, iterations
assert len(centres) > 25
assert max(thread_counts) > len(centres)
```

Every thread count gives exactly the same displacements as the sequential run: the subsets are
independent and each one performs the same arithmetic whatever thread runs it. The error against
the imposed translation stays far below $10^{-3}$ px, the bias of the interpolation at degree 5.

## How many threads

The speed-up depends on the machine, so nothing here asserts on it. As a rule of thumb, use about
as many threads as **physical** cores: `os.cpu_count()` counts logical cores, and hyper-threads
share the arithmetic units of their core, so they add little to this compute-bound kernel. Beyond
that, extra threads only add scheduling cost, and the largest counts of the table show little or no
gain. With few
subsets the work is also unevenly shared (the last threads wait for the others), and a thread count
above the number of subsets brings nothing.

Only the interpolator calls run outside the lock. The NumPy code of `solve_subset` around them
(building the 2 x 2 Hessian, the small linear solve, the Python loop) holds the lock, and with
subsets of this size it is a large share of the time, so the Python speed-up stays well below
the number of cores (about 1.5 at 8 threads on the machine described in the
{ref}`user guide <user-guide-threads>`, against about 6 for the batch evaluation alone in C++). The larger the share of the interpolator in
the task (bigger subsets, more points per call), the closer the scaling gets to the number of
cores. With much smaller tasks, hand each worker a block of several subsets instead of one.

## The same in C++

The C++ version distributes the subsets over `std::thread` workers and shares one `const`
`BSplineImage<double, 5>` between them. It is built and run with the other examples.

```{literalinclude} ../../../examples/dic_parallel_subsets.cpp
:language: cpp
```
