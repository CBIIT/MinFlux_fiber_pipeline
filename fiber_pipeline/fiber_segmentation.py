"""
fiber_segmentation.py
=====================

Stage 2: segment the upsampled tubeness image into a binary fiber mask
and skeleton. Also produces a colored overlay image.

Primary entry point:
    run_fiber_segmentation(tubeness_up, csv_path, out_dir, results_dir)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import color, exposure, restoration, filters, morphology, util
from skimage.morphology import disk
from skimage import io as skio_io

from config import PIPELINE_CONFIG as cfg
from io_utils import base_name_no_ext, ensure_dir


def to_gray01(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale float in [0,1]."""
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = util.img_as_float(img)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Return coordinates of skeleton pixels with exactly one 8‑connected neighbor."""
    sk = skeleton.astype(np.uint8)
    nb = ndi.convolve(sk, np.ones((3, 3), dtype=np.uint8), mode="constant")
    return np.column_stack(np.where((sk == 1) & (nb == 2)))


def bridge_small_gaps(mask: np.ndarray,
                      vesselness: np.ndarray,
                      max_gap: int = 12,
                      max_pairs: int = 200) -> np.ndarray:
    """
    Connect nearby skeleton endpoints by routing a short path
    through high‑vesselness pixels. This keeps connections plausible
    and avoids over‑bridging.
    """
    from skimage import graph

    skel = morphology.skeletonize(mask)
    endpoints = find_endpoints(skel)
    if len(endpoints) < 2:
        return mask

    used = set()
    cost = 1.0 - exposure.rescale_intensity(vesselness, out_range=(0, 1))
    count_pairs = 0

    for i, p in enumerate(endpoints):
        if i in used:
            continue
        d2 = np.sum((endpoints - p) ** 2, axis=1)
        for j in np.argsort(d2)[1:]:
            if j in used:
                continue
            d = float(np.sqrt(d2[j]))
            if d > max_gap:
                break

            start = tuple(map(int, p))
            end = tuple(map(int, endpoints[j]))
            try:
                path, _ = graph.route_through_array(
                    cost, start, end, fully_connected=True, geometric=True
                )
            except Exception:
                path = None

            if path is not None and len(path) <= 1.8 * d + 2:
                rr, cc = zip(*path)
                mask[np.array(rr), np.array(cc)] = True
                used.add(i)
                used.add(j)
                count_pairs += 1
                if count_pairs >= max_pairs:
                    return mask
                break
    return mask


def segment_fibers(img: np.ndarray):
    """
    Core segmentation routine.

    Parameters
    ----------
    img : 2D (or 3D RGB) array
        Upsampled tubeness image.

    Returns
    -------
    mask : bool array
        Segmented fiber mask.
    skeleton : bool array
        Skeletonized version of the mask.
    vessel : float array
        Vesselness response used for gap bridging.
    """
    g = to_gray01(img)

    den = restoration.denoise_tv_chambolle(g, weight=0.05)
    clahe = exposure.equalize_adapthist(den, clip_limit=0.02)

    sigmas = np.linspace(cfg.seg_sigma_min,
                         cfg.seg_sigma_max,
                         cfg.seg_n_scales)
    vessel = filters.sato(clahe, sigmas=sigmas, black_ridges=False)
    vessel = exposure.rescale_intensity(vessel, out_range=(0, 1))

    high = np.percentile(vessel, cfg.seg_hyst_high_pct)
    low = cfg.seg_hyst_low_ratio * high
    mask = filters.apply_hysteresis_threshold(vessel, low, high)

    if cfg.seg_min_obj > 0:
        mask = morphology.remove_small_objects(mask, min_size=cfg.seg_min_obj)
    if cfg.seg_close_radius > 0:
        mask = morphology.binary_closing(mask, footprint=disk(cfg.seg_close_radius))
    if cfg.seg_hole_area > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=cfg.seg_hole_area)

    if cfg.seg_bridge_gaps:
        mask = bridge_small_gaps(mask, vessel, max_gap=cfg.seg_max_gap_px)

    skel = morphology.skeletonize(mask)
    return mask, skel, vessel


def run_fiber_segmentation(tubeness_up: np.ndarray,
                           csv_path,
                           out_dir,
                           results_dir) -> str:
    """
    Run segmentation on `tubeness_up`, write mask/skeleton/overlay,
    and return the skeleton image path.

    Parameters
    ----------
    tubeness_up : uint8
        Upsampled tubeness image.
    csv_path : str or Path
        Input CSV path (for naming).
    out_dir : str or Path
        Folder for TIFF/PNG outputs (FIJI‑style).
    results_dir : str or Path
        Folder for visualization PNGs.

    Returns
    -------
    skeleton_path : str
        Path to the saved skeleton image.
    """
    out_dir = ensure_dir(out_dir)
    results_dir = ensure_dir(results_dir)
    base = base_name_no_ext(csv_path)

    print(f"[SEG] {base}: running fiber segmentation on upsampled tubeness...")

    mask, skeleton, vessel = segment_fibers(tubeness_up)

    mask_path = out_dir / f"{base}_fiber_mask_scale{cfg.upsample_factor}.png"
    skeleton_path = out_dir / f"{base}_fiber_skeleton_scale{cfg.upsample_factor}.png"
    overlay_path = out_dir / f"{base}_fiber_overlay_scale{cfg.upsample_factor}.png"

    skio_io.imsave(mask_path, util.img_as_ubyte(mask))
    skio_io.imsave(skeleton_path, util.img_as_ubyte(skeleton))

    overlay = color.label2rgb(
        mask.astype(int),
        image=to_gray01(tubeness_up),
        alpha=0.35,
        bg_label=0
    )
    skio_io.imsave(
        overlay_path,
        util.img_as_ubyte(exposure.rescale_intensity(overlay))
    )

    print(f"[SEG] {base}: mask   -> {mask_path}")
    print(f"[SEG] {base}: skel   -> {skeleton_path}")
    print(f"[SEG] {base}: overlay-> {overlay_path}")
    print(f"[SEG] {base}: segmentation complete.")

    return str(skeleton_path)
