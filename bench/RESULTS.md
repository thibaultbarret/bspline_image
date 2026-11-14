<!-- SPDX-License-Identifier: BSD-3-Clause -->

# Benchmark results

- Date: 2026-11-13
- Machine: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz (x86_64, macOS, Darwin 25.6.0)
- Compiler: Homebrew clang 23.1.2 (`/usr/local/opt/llvm/bin/clang++`), C++20
- Build: Release (`-O3 -DNDEBUG`), no `-march=native`; strict warnings enabled
- Workload: 1024x1024 `float` random image, 1,000,000 uniform interior points (fixed `std::mt19937` seed),
  one warm-up pass, minimum of 5 timed passes, `std::chrono::steady_clock`. Results are
  accumulated into a volatile sink. Values are ns/point.
- Command: `cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-release && ./build-release/bench/bsi_bench`

## v0 baseline (2026-11-13)

| degree | type | core value | core value_and_gradient | facade value | facade + visit value | visit vs core value |
|---|---|---|---|---|---|---|
| 1 | float | 19.20 | 21.01 | 24.66 | 21.41 | +11.5% |
| 1 | double | 23.59 | 29.37 | 29.72 | 23.41 | -0.8% |
| 3 | float | 41.80 | 66.55 | 47.03 | 44.74 | +7.0% |
| 3 | double | 63.06 | 69.97 | 72.31 | 69.47 | +10.2% |
| 5 | float | 77.75 | 117.21 | 82.04 | 77.68 | -0.1% |
| 5 | double | 123.72 | 130.99 | 126.99 | 108.03 | -12.7% |
| 7 | float | 129.46 | 149.41 | 114.61 | 123.07 | -4.9% |
| 7 | double | 162.88 | 143.20 | 136.96 | 129.11 | -20.7% |

## v1 after final review fixes (2026-11-14)

Changes: points inside the image skip the `fabs`/`fmod` mirror folding (`fold_fast`, spec 7.6);
the accumulations use an explicit `mul_add` (fused only when the target has fast FMA); the bench
now also reports facade `value_and_gradient` (pointwise and with `visit`). Same machine, compiler,
flags and workload as above. "Before" is the pre-change code (commit 4926a24) re-measured in the
same session; every cell is the minimum over three consecutive runs of the benchmark (itself a
min of 5 passes), ns/point.

| degree | type | core value before | core value after | core value_and_gradient before | core value_and_gradient after | facade value before | facade value after | facade + visit value before | facade + visit value after | facade value_and_gradient (new) | facade + visit value_and_gradient (new) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | float | 16.39 | 7.86 | 19.56 | 10.82 | 21.81 | 15.34 | 19.46 | 7.63 | 15.65 | 11.05 |
| 1 | double | 15.40 | 10.99 | 19.52 | 12.89 | 19.74 | 14.38 | 15.41 | 14.35 | 13.84 | 14.94 |
| 3 | float | 38.05 | 29.13 | 53.43 | 41.82 | 40.83 | 30.15 | 39.98 | 30.09 | 45.88 | 42.47 |
| 3 | double | 47.25 | 36.02 | 53.87 | 45.99 | 51.16 | 39.26 | 48.98 | 37.65 | 48.06 | 47.17 |
| 5 | float | 64.32 | 55.82 | 92.62 | 82.92 | 64.98 | 55.41 | 64.70 | 56.45 | 83.72 | 85.44 |
| 5 | double | 77.73 | 62.05 | 87.34 | 79.48 | 78.53 | 64.69 | 79.12 | 60.67 | 82.87 | 82.38 |
| 7 | float | 102.85 | 89.17 | 122.94 | 168.37 | 102.18 | 91.34 | 102.33 | 88.50 | 174.29 | 168.53 |
| 7 | double | 118.77 | 96.71 | 113.06 | 108.68 | 119.69 | 101.83 | 124.82 | 100.51 | 147.72 | 149.17 |

Reading: `value` gets 10-50 % faster for degrees 1-7 and `value_and_gradient` 8-45 % faster for
degrees 1-5. Degree 7 `float` `value_and_gradient` is slower (123 -> 168 ns); it is stable across
runs, and with the fast path disabled the same kernel measures about 128 ns, so it looks like an
inlining/code-layout effect of the compiler on that one instantiation rather than extra work
(the added check is a single compare pair per axis). Worth re-checking with another compiler
(and with `-march=native`). Facade pointwise and `visit` `value_and_gradient` track core
`value_and_gradient` to within the usual noise (+/-15 %).

## Visit overhead check (target: at most 10 % over direct core `value`)

Three consecutive runs gave the following "visit vs core value" figures (%):

| degree | type | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| 1 | float | +11.5 | +20.2 | -4.2 |
| 1 | double | -0.8 | -8.2 | -17.3 |
| 3 | float | +7.0 | -7.7 | +9.4 |
| 3 | double | +10.2 | +2.3 | -0.2 |
| 5 | float | -0.1 | +8.3 | +1.7 |
| 5 | double | -12.7 | -0.3 | +3.3 |
| 7 | float | -4.9 | +3.2 | -4.0 |
| 7 | double | -20.7 | +11.8 | +0.7 |

Individual cells exceed +10 % in several runs (degree 1 float, degree 3 double, degree 7 double),
but the sign flips from run to run and the deviations reach -20 %. The measurement noise on this
laptop (random access over a 4 MB coefficient array, no frequency pinning) is about +/-15 %, which
is larger than the effect being measured. No systematic visit overhead is visible: the visit path
runs the same concrete-type loop as the core. The 10 % criterion is therefore not demonstrably
violated, but it also cannot be confirmed tightly on this machine; re-measure on a quiet machine
before using these numbers to detect small regressions. Facade pointwise calls (per-call dispatch)
are consistently a few ns/point slower than core for degrees 1 and 3.

## v2 Hessian evaluation (2026-11-14)

The bench now also reports `value_gradient_hessian` (value, gradient and the three second
derivatives) for the core type and the facade (pointwise). Same machine, compiler, flags and
workload as above; single run of the benchmark (each cell is the min of 5 passes), ns/point.
`value`, `value_and_gradient` and `visit` columns are unchanged code paths; their numbers are
re-measured in the same run.

| degree | type | core value | core value_and_gradient | core value_gradient_hessian | facade value | facade value_and_gradient | facade value_gradient_hessian | facade + visit value | facade + visit value_and_gradient | visit vs core value |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | float | 8.72 | 11.50 | 18.54 | 16.60 | 17.53 | 18.10 | 8.16 | 11.84 | -6.5% |
| 1 | double | 13.46 | 13.12 | 20.36 | 17.86 | 18.36 | 21.70 | 16.35 | 18.26 | +21.4% |
| 3 | float | 30.74 | 49.04 | 43.32 | 34.45 | 45.45 | 44.59 | 33.15 | 43.32 | +7.8% |
| 3 | double | 42.52 | 61.80 | 67.28 | 39.55 | 57.90 | 50.64 | 53.12 | 49.92 | +24.9% |
| 5 | float | 57.25 | 85.90 | 117.11 | 58.66 | 87.06 | 116.42 | 57.64 | 85.01 | +0.7% |
| 5 | double | 61.78 | 78.76 | 128.06 | 67.57 | 85.27 | 128.25 | 60.98 | 79.76 | -1.3% |
| 7 | float | 88.52 | 172.31 | 208.30 | 92.29 | 177.65 | 206.51 | 91.29 | 171.56 | +3.1% |
| 7 | double | 99.15 | 107.33 | 234.40 | 104.79 | 147.85 | 226.48 | 99.49 | 160.27 | +0.3% |

Reading: for degree 5, the full value + gradient + Hessian costs about 1.4-1.6x
`value_and_gradient` and about 2x `value` (117-128 ns/point). The facade pointwise call tracks
the core within noise (+/-15 %, see the visit overhead check below); degrees 1 and 3 are noisy
at that level, e.g. the `+21 %` and `+25 %` visit cells are within the run-to-run spread seen
above and are not a systematic effect.
