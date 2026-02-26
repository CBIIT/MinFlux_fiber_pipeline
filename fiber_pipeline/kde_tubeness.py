"""
kde_tubeness.py
===============

Stage 1: build a KDE grid from X,Y localizations, blur it, and run a
tubeness filter. The outputs are saved as TIFFs and the upsampled
tubeness image is returned for later segmentation.

This module also defines KDEInfo, which stores the mapping between
pixel coordinates and physical units so later stages can overlay the
skeleton with 3D data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile
from skimage import transform

from config import PIPELINE_CONFIG as cfg
from io_utils import base_name_no_ext, ensure_dir


@dataclass
class KDEInfo:
    """Geometry of the KDE grid used to map pixels⇔µm."""
    xmin_um: float
    ymin_um: float
    width_px: int
    height_px: int


def compute_kde_counts(X: np.ndarray,
                       Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, KDEInfo]:
    """
    Build a KDE‑style occupancy image and blur it.

    Parameters
    ----------
    X, Y : 1D arrays of float
        Localization coordinates in microns.

    Returns
    -------
    kde_blur : (H,W) float32
        Gaussian‑blurred occupancy image.
    counts : (H,W) float32
        Raw integer occupancy per pixel before blur.
    info : KDEInfo
        Mapping between pixels and physical units.
    """
    n_pts = len(X)
    if n_pts == 0:
        raise ValueError("No points for KDE.")

    minX, maxX = X.min(), X.max()
    minY, maxY = Y.min(), Y.max()

    xmin_um = float(minX - cfg.pad_um)
    xmax_um = float(maxX + cfg.pad_um)
    ymin_um = float(minY - cfg.pad_um)
    ymax_um = float(maxY + cfg.pad_um)

    width_px = int(math.floor((xmax_um - xmin_um) * cfg.px_per_um + 1.5))
    height_px = int(math.floor((ymax_um - ymin_um) * cfg.px_per_um + 1.5))
    width_px = max(width_px, 10)
    height_px = max(height_px, 10)

    counts = np.zeros((height_px, width_px), dtype=np.float32)

    for xi, yi in zip(X, Y):
        ix = int(round((xi - xmin_um) * cfg.px_per_um))
        iy = height_px - 1 - int(round((yi - ymin_um) * cfg.px_per_um))
        if 0 <= ix < width_px and 0 <= iy < height_px:
            counts[iy, ix] += 1.0

    sigma_px = cfg.gauss_sigma_um * cfg.px_per_um
    kde_blur = gaussian_filter(counts, sigma=sigma_px).astype(np.float32)

    info = KDEInfo(
        xmin_um=xmin_um,
        ymin_um=ymin_um,
        width_px=width_px,
        height_px=height_px,
    )
    return kde_blur, counts, info


def normalize_to_8bit(img32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize a float32 image to [0,1] and [0,255] (uint8).

    Returns
    -------
    norm_float : float32 image in [0,1]
    img8 : uint8 image in [0,255]
    """
    mi = float(np.nanmin(img32))
    ma = float(np.nanmax(img32))
    rng = ma - mi
    if rng <= 0:
        rng = 1e-12
    norm = (img32 - mi) / rng
    norm = np.clip(norm, 0, 1)
    img8 = (norm * 255.0 + 0.5).astype(np.uint8)
    return norm.astype(np.float32), img8


def tubeness_from_kde_gray8(kde_gray8: np.ndarray) -> np.ndarray:
    """
    Run a Sato tubeness filter on the 8‑bit KDE image.

    Parameters
    ----------
    kde_gray8 : uint8
        Normalized KDE image.

    Returns
    -------
    tubeness : float32
        Tubeness response (values >= 0).
    """
    from skimage.filters import sato

    img = kde_gray8.astype(np.float32) / 255.0
    tubeness = sato(
        img,
        sigmas=[cfg.tubeness_sigma_px],
        black_ridges=False
    ).astype(np.float32)
    tubeness[tubeness < 0] = 0
    return tubeness


def run_kde_and_tubeness(csv_path: str | Path,
                          df,
                          out_dir: str | Path) -> tuple[np.ndarray, KDEInfo]:
    """
    Execute the full KDE+tubeness stage for a single sample.

    Files written
    -------------
    * <base>_counts_32f.tif
    * <base>_kde_blur32f_sigmaXXXum_counts.tif
    * <base>_kde_gray8.tif
    * <base>_tubeness32f_sigma1um.tif
    * <base>_tubeness8bit.tif
    * <base>_tubeness8bit_scale<N>.tif

    Parameters
    ----------
    csv_path : str or Path
        Path to the localization CSV (used only for naming).
    df : pandas.DataFrame
        Localization table with columns X,Y,TraceID (and optionally Z).
        The KDE+tubeness stage only uses X and Y.
    out_dir : str or Path
        Folder where TIFFs are written.

    Returns
    -------
    tubeness_up : uint8
        Upsampled tubeness image.
    info : KDEInfo
        Pixel⇔µm mapping used for later steps.
    """
    out_dir = ensure_dir(out_dir)
    base = base_name_no_ext(csv_path)

    X = df["X"].to_numpy(float)
    Y = df["Y"].to_numpy(float)

    print(f"[KDE] {base}: computing KDE grid...")
    kde_blur, counts, info = compute_kde_counts(X, Y)

    # Save raw counts and KDE blur
    tifffile.imwrite(out_dir / f"{base}_counts_32f.tif",
                     counts.astype(np.float32),
                     dtype=np.float32)
    tifffile.imwrite(
        out_dir / f"{base}_kde_blur32f_sigma{cfg.gauss_sigma_um:.4f}um_counts.tif",
        kde_blur.astype(np.float32),
        dtype=np.float32
    )

    # Normalize & make 8‑bit
    _, kde_gray8 = normalize_to_8bit(kde_blur)
    tifffile.imwrite(out_dir / f"{base}_kde_gray8.tif",
                     kde_gray8,
                     dtype=np.uint8)

    # Tubeness
    tubeness32f = tubeness_from_kde_gray8(kde_gray8)
    tifffile.imwrite(out_dir / f"{base}_tubeness32f_sigma1um.tif",
                     tubeness32f,
                     dtype=np.float32)

    _, tubeness8 = normalize_to_8bit(tubeness32f)
    tubeness8_path = out_dir / f"{base}_tubeness8bit.tif"
    tifffile.imwrite(tubeness8_path, tubeness8, dtype=np.uint8)
    print(f"[KDE] {base}: saved tubeness 8‑bit image: {tubeness8_path}")

    # Upsample
    print(f"[KDE] {base}: upsampling tubeness by factor {cfg.upsample_factor}...")
    tubeness_up = transform.resize(
        tubeness8,
        (tubeness8.shape[0] * cfg.upsample_factor,
         tubeness8.shape[1] * cfg.upsample_factor),
        order=1,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.uint8)

    up_path = out_dir / f"{base}_tubeness8bit_scale{cfg.upsample_factor}.tif"
    tifffile.imwrite(up_path, tubeness_up, dtype=np.uint8)
    print(f"[KDE] {base}: saved upsampled tubeness image: {up_path}")

    return tubeness_up, info
