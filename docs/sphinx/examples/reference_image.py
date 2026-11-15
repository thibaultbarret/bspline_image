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
