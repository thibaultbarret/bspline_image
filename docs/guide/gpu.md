<!-- SPDX-License-Identifier: BSD-3-Clause -->
# Using the kernels on a GPU {#gpu}

The library has no GPU code in v1, but layer 0 (`bsi/kernel/`) is written so that the
mathematics can be reused unchanged in CUDA or HIP kernels. Layer-0 functions operate on
plain views and scalars, allocate nothing, and are qualified with `BSI_HOST_DEVICE`, which
expands to `__host__ __device__` under CUDA (`__CUDACC__`) or HIP (`__HIPCC__`) and to
nothing otherwise.

## Functions to reuse

- `bsi::evaluate` and `bsi::evaluate_with_gradient`: full 2-D value, or value and gradient,
  from a `bsi::CoeffView<const T>` and a position.
- `bsi::weights` and `bsi::weights_and_derivatives`: the 1-D weights (and derivatives) and
  the index of the first coefficient, if you write your own evaluation loop.
- `bsi::fold_mirror` and `bsi::mirror_index`: mirror folding of real and integer
  coordinates.

## Moving the coefficients

`BSplineImage::coefficients()` returns a `bsi::CoeffView<const T>` with the fields
`origin`, `width`, `height`, `stride` and `pad`. The storage is a padded row-major array of
`stride * (height + 2 * pad)` elements, and `origin` points at coefficient (0, 0) inside
it, `pad * stride + pad` elements after the start. To use it on a device:

1. Copy the whole padded storage (starting `pad * stride + pad` elements before `origin`)
   into device memory.
2. Rebuild a view on the device with the same `width`, `height`, `stride` and `pad`, and
   with `origin` pointing at the same offset inside the device buffer.
3. Call `bsi::evaluate<T, Degree>(view, x, y)` from the kernel.

The padding (`pad == bsi::required_pad<Degree>`) already holds the mirrored coefficients,
so the device code needs no border handling beyond `fold_mirror`, which `evaluate` calls.

## The prefilter stays on the host

The prefilter (`bsi::prefilter_line`) is a recursion that is sequential along each line.
In v1 it runs on the host, inside the `bsi::BSplineImage` constructor, always in `double`.
Only the evaluation, which is independent per point, is meant for the device.
