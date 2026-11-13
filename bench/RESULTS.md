# Benchmark baseline

- Date: 2026-11-13
- Machine: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz (x86_64, macOS, Darwin 25.6.0)
- Compiler: Homebrew clang 23.1.2 (`/usr/local/opt/llvm/bin/clang++`), C++20
- Build: Release (`-O3 -DNDEBUG`), no `-march=native`; strict warnings enabled
- Workload: 1024x1024 `float` random image, 1,000,000 uniform interior points (fixed `std::mt19937` seed),
  one warm-up pass, minimum of 5 timed passes, `std::chrono::steady_clock`. Results are
  accumulated into a volatile sink. Values are ns/point.
- Command: `cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-release && ./build-release/bench/bsi_bench`

## Baseline (first run)

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
