# SPDX-License-Identifier: BSD-3-Clause
"""Loader of the real speckle image used to illustrate the executed examples."""

from pathlib import Path

import numpy as np
from matplotlib.image import imread

REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "speckle_reference.png"
GREY_LEVELS = 255.0  # imread returns floats in [0, 1] for an 8-bit PNG


def load_speckle_reference() -> np.ndarray:
    """Return the reference speckle as a float64 array in grey levels 0-255, ``image[y, x]``."""
    pixels = np.asarray(imread(REFERENCE_PATH), dtype=np.float64)
    if pixels.ndim == 3:  # an RGB(A) export of a grey image: keep one channel
        pixels = pixels[..., 0]
    return np.round(pixels * GREY_LEVELS)


def plot_subset_context(image: np.ndarray, centre: tuple[int, int], half_size: int):
    """Draw ``image`` with the square subset around ``centre`` framed, and a detail of it.

    Return the matplotlib figure; the subset spans ``2 * half_size + 1`` pixels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cx, cy = centre
    low, high = -half_size - 0.5, half_size + 0.5  # pixel edges around the subset
    fig, (full, detail) = plt.subplots(1, 2, figsize=(9, 4.4), width_ratios=(1.15, 1))
    full.imshow(image, cmap="gray", vmin=0, vmax=GREY_LEVELS)
    full.add_patch(Rectangle((cx + low, cy + low), high - low, high - low, fill=False, edgecolor="tab:red", linewidth=1.5))
    full.set_title("Reference speckle and subset")
    pad = 2 * half_size  # the detail shows the subset with a margin of pixels around it
    detail.imshow(image, cmap="gray", vmin=0, vmax=GREY_LEVELS, interpolation="nearest")
    detail.add_patch(Rectangle((cx + low, cy + low), high - low, high - low, fill=False, edgecolor="tab:red", linewidth=1.5))
    detail.set_xlim(cx - pad, cx + pad)
    detail.set_ylim(cy + pad, cy - pad)
    detail.set_title(f"Detail: {2 * half_size + 1} x {2 * half_size + 1} pixel subset")
    for ax in (full, detail):
        ax.set_xlabel("x (column, px)")
        ax.set_ylabel("y (row, px)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------
# A speckled element in uniaxial tension: the image pair of the synthetic-image and the
# parallel-subsets pages.
#
# The element is a unit-size Q4 element, (s, t) in [0, 1]^2, whose four corners are mapped
# onto a ELEMENT_SIZE x ELEMENT_SIZE square of pixels at the centre of the image. Positions
# are in pixel units, with the pixel (row, column) centred on integer coordinates (x = column,
# y = row), so the element spans half a pixel beyond its first and last pixel.
# ---------------------------------------------------------------------------------------
IMAGE_SIZE = 512
ELEMENT_SIZE = 320
BACKGROUND = 126.0  # grey level of the uniform background
STRAIN = 0.01  # axial strain along y
POISSON = 0.3  # transverse strain is -POISSON * STRAIN along x
SPLINE_DEGREE = 7  # degree of the interpolant that generates the deformed image
ELEMENT_CENTRE = (IMAGE_SIZE - 1) / 2  # x0 = y0, the fixed centre of the element
ELEMENT_FIRST = IMAGE_SIZE // 2 - ELEMENT_SIZE // 2  # first pixel of the element in the image

# Natural coordinates (s, t) of the nodes, counter-clockwise from the top-left corner.
NODES_ST = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def q4_shape_functions(s, t) -> np.ndarray:
    """Bilinear shape functions of the Q4 element, shape ``(4, *np.shape(s))``."""
    s, t = np.asarray(s, dtype=np.float64), np.asarray(t, dtype=np.float64)
    return np.array([(1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t])


def q4_map(s, t, nodes: np.ndarray):
    """Position ``(x, y)`` of the natural coordinates ``(s, t)`` for the 4 node positions ``nodes``."""
    n = q4_shape_functions(s, t)
    return np.tensordot(nodes[:, 0], n, axes=1), np.tensordot(nodes[:, 1], n, axes=1)


def tension_displacement(x, y):
    """Theoretical displacement ``(u, v)`` of uniaxial tension along y, the centre being fixed."""
    return -POISSON * STRAIN * (np.asarray(x) - ELEMENT_CENTRE), STRAIN * (np.asarray(y) - ELEMENT_CENTRE)


def element_nodes() -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(4, 2)`` node positions of the element in the reference and deformed images."""
    half = ELEMENT_SIZE / 2
    reference = ELEMENT_CENTRE + half * (2 * NODES_ST - 1)
    u, v = tension_displacement(reference[:, 0], reference[:, 1])
    return reference, reference + np.stack([u, v], axis=1)


def natural_coordinates(x, y, nodes: np.ndarray):
    """Natural coordinates ``(s, t)`` of the positions ``(x, y)`` in an element with rectangular ``nodes``."""
    return (x - nodes[0, 0]) / (nodes[1, 0] - nodes[0, 0]), (y - nodes[0, 1]) / (nodes[3, 1] - nodes[0, 1])


def make_element_pair(speckle: np.ndarray):
    """Build the image pair from the full speckle image.

    Return ``(reference, deformed, inside)``: the two images in grey levels and the boolean
    mask of the pixels of the deformed image that lie inside the deformed element. The
    reference is the speckle inside the element and ``BACKGROUND`` outside. The deformed
    image is obtained by interpolating the speckle of the element, with a spline of degree
    ``SPLINE_DEGREE``, at the antecedent of every pixel of the deformed element; the other
    pixels stay at ``BACKGROUND``.
    """
    from bspline_image import BSplineImage

    last = ELEMENT_FIRST + ELEMENT_SIZE
    crop = speckle[ELEMENT_FIRST:last, ELEMENT_FIRST:last]
    reference = np.full((IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
    reference[ELEMENT_FIRST:last, ELEMENT_FIRST:last] = crop

    nodes_reference, nodes_deformed = element_nodes()
    rows, cols = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float64)
    s, t = natural_coordinates(cols, rows, nodes_deformed)
    inside = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    source_x, source_y = q4_map(s[inside], t[inside], nodes_reference)

    deformed = np.full((IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
    spline = BSplineImage(crop, degree=SPLINE_DEGREE)
    deformed[inside] = spline.values(source_x - ELEMENT_FIRST, source_y - ELEMENT_FIRST)
    return reference, deformed, inside


def element_outline(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed polygon ``(xs, ys)`` through the four nodes, for ``ax.plot``."""
    closed = np.vstack([nodes, nodes[:1]])
    return closed[:, 0], closed[:, 1]
