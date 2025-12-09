"""
overlays.py
===========

Create 2D overlays of the skeleton with localization density and
junction centroids. These are purely for visualization and QA.

Entry points:
    make_density_overlay(...)
    make_junction_overlay(...)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve, label, center_of_mass
from skimage import io as skio_io, color as skio_color, exposure as skio_exposure

from config import PIPELINE_CONFIG as cfg
from kde_tubeness import KDEInfo
from io_utils import ensure_dir


def _map_points_to_skeleton_grid(df, info: KDEInfo):
    """
    Map X,Y points from µm space onto the upsampled skeleton grid.
    """
    X = df["X"].to_numpy(float)
    Y = df["Y"].to_numpy(float)

    H0, W0 = info.height_px, info.width_px
    H = H0 * cfg.upsample_factor
    W = W0 * cfg.upsample_factor

    ix0 = np.round((X - info.xmin_um) * cfg.px_per_um).astype(int)
    iy0 = H0 - 1 - np.round((Y - info.ymin_um) * cfg.px_per_um).astype(int)

    ix = ix0 * cfg.upsample_factor
    iy = iy0 * cfg.upsample_factor

    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

    return ix[inside], iy[inside], H, W


def make_density_overlay(df,
                         skeleton_path: str,
                         info: KDEInfo,
                         results_dir: str | Path) -> None:
    """
    Overlay localization density (2D KDE of X,Y) with the 2D skeleton.

    Writes two PNGs into `results_dir`:
        * skeleton_density_overlay_fig.png
        * skeleton_density_overlay.png
    """
    results_dir = ensure_dir(results_dir)

    skeleton = skio_io.imread(skeleton_path)
    if skeleton.ndim == 3:
        skeleton = skio_color.rgb2gray(skeleton)
    skeleton = skeleton > 0.5
    H_skel, W_skel = skeleton.shape

    ix, iy, H, W = _map_points_to_skeleton_grid(df, info)

    if (H_skel, W_skel) != (H, W):
        print("[OVERLAY] WARNING: skeleton size does not match expected upsampled grid.")

    density = np.zeros((H_skel, W_skel), dtype=np.float32)
    for xpx, ypx in zip(ix, iy):
        if 0 <= ypx < H_skel and 0 <= xpx < W_skel:
            density[ypx, xpx] += 1.0

    density_smooth = gaussian_filter(density, cfg.gauss_sigma_px_density)
    density_norm = density_smooth / (density_smooth.max() + 1e-8)

    plt.figure(figsize=(8, 8))
    plt.imshow(density_norm, cmap="magma", alpha=1.0)
    plt.imshow(skeleton, cmap="gray", alpha=0.6)
    plt.axis("off")
    plt.title("Skeleton overlaid with localization density", fontsize=14)
    fig = plt.gcf()
    fig.savefig(Path(results_dir) / "skeleton_density_overlay_fig.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    plt.imsave(Path(results_dir) / "skeleton_density_overlay.png",
               density_norm, cmap="magma")
    print("[OVERLAY] Saved basic skeleton+density overlays.")


def _junction_centroids(skeleton_bool: np.ndarray,
                        min_neighbors: int = 3):
    """
    Compute one centroid per junction cluster on a skeleton.
    """
    sk = skeleton_bool.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    nb_count = convolve(sk, kernel, mode="constant", cval=0)
    neighbors = nb_count - sk
    junc_mask = (sk == 1) & (neighbors >= min_neighbors)

    labeled, n_labels = label(junc_mask, structure=np.ones((3, 3), np.uint8))
    if n_labels == 0:
        return np.array([]), np.array([])

    cents = center_of_mass(junc_mask, labeled,
                           index=np.arange(1, n_labels + 1))
    cents = np.array(cents)
    return cents[:, 0], cents[:, 1]


def make_junction_overlay(df,
                          skeleton_path: str,
                          info: KDEInfo,
                          results_dir: str | Path) -> None:
    """
    Overlay the skeleton and localization density with junction centroids.

    Writes:
        * skeleton_density_junctions.png
        * skeleton_density_overlay2.png
    """
    results_dir = ensure_dir(results_dir)

    skeleton = skio_io.imread(skeleton_path)
    if skeleton.ndim == 3:
        skeleton = skio_color.rgb2gray(skeleton)
    skeleton = skeleton > 0.5
    H_skel, W_skel = skeleton.shape

    X = df["X"].to_numpy(float)
    Y = df["Y"].to_numpy(float)

    H0, W0 = info.height_px, info.width_px
    ix0 = np.round((X - info.xmin_um) * cfg.px_per_um).astype(int)
    iy0 = H0 - 1 - np.round((Y - info.ymin_um) * cfg.px_per_um).astype(int)

    ix = ix0 * cfg.upsample_factor
    iy = iy0 * cfg.upsample_factor

    inside = (ix >= 0) & (ix < W_skel) & (iy >= 0) & (iy < H_skel)
    ix = ix[inside]
    iy = iy[inside]

    density = np.zeros((H_skel, W_skel), dtype=np.float32)
    density[iy, ix] += 0.5

    density_smooth = gaussian_filter(density, cfg.gauss_sigma_px_density)
    density_norm = skio_exposure.rescale_intensity(
        density_smooth,
        in_range="image",
        out_range=(0, 1)
    )

    jy, jx = _junction_centroids(skeleton, min_neighbors=3)
    print(f"[OVERLAY] Found {len(jx)} junction centroids.")

    plt.figure(figsize=(9, 9))
    plt.imshow(density_norm, cmap="magma", alpha=1.0)
    plt.imshow(skeleton, cmap="gray", alpha=0.5)
    plt.scatter(jx, jy, s=40, c="lime", edgecolors="black")
    plt.axis("off")
    plt.title("Skeleton + Enhanced Density + Junction Centroids",
              fontsize=16)
    fig = plt.gcf()
    fig.savefig(Path(results_dir) / "skeleton_density_junctions.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    plt.imsave(Path(results_dir) / "skeleton_density_overlay2.png",
               density_norm, cmap="magma")
    print("[OVERLAY] Saved junction overlay images.")
