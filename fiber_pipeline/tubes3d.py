"""
tubes3d.py
==========

Build 3D centerlines ("tubes") from TraceID‑averaged localizations and
2D skeleton segments. Also produces a 3D scatter/line plot.

Entry point:
    build_3d_tubes(df_ids, segments, merged_group_id, kde_info,
                   base_name, results_dir)

Returns:
    df_avg       - per‑TraceID averaged coordinates
    assigned_gid - 1D array label per averaged point (segment group or -1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import colorsys
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import PIPELINE_CONFIG as cfg
from kde_tubeness import KDEInfo
from io_utils import ensure_dir


def _pix_to_um(yx, info: KDEInfo):
    y, x = float(yx[0]), float(yx[1])
    x0 = x / cfg.upsample_factor
    y0 = y / cfg.upsample_factor
    X_um = info.xmin_um + (x0 / cfg.px_per_um)
    Y_um = info.ymin_um + ((info.height_px - 1 - y0) / cfg.px_per_um)
    return np.array([X_um, Y_um], float)


def _build_segment_polyline_um(seg, info: KDEInfo):
    path = [tuple(map(int, seg["anchor_u"]))] + \
           [tuple(map(int, p)) for p in seg["pixels"]] + \
           [tuple(map(int, seg["anchor_v"]))]
    clean = [path[0]]
    for p in path[1:]:
        if p != clean[-1]:
            clean.append(p)
    xy = np.array([_pix_to_um(p, info) for p in clean], float)
    return xy


def _resample_polyline(xy: np.ndarray, step: float) -> np.ndarray:
    if xy.shape[0] < 2:
        return xy.copy()
    seglen = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    Lcum = np.concatenate([[0.0], np.cumsum(seglen)])
    if Lcum[-1] <= step:
        return xy.copy()
    t = np.arange(0.0, Lcum[-1] + 1e-9, step)
    out = []
    for s_ in t:
        i = np.searchsorted(Lcum, s_, side="right") - 1
        i = min(i, len(seglen) - 1)
        ds = s_ - Lcum[i]
        if seglen[i] < 1e-12:
            out.append(xy[i])
        else:
            out.append(xy[i] + (xy[i + 1] - xy[i]) * (ds / seglen[i]))
    return np.array(out)


def _make_palette_3d(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    cols = [colorsys.hsv_to_rgb(float(h), 0.85, 1.0) for h in hues]
    return cols


def _set_3d_equal_axes(ax, X, Y, Z):
    x_range = X.max() - X.min() if len(X) else 1
    y_range = Y.max() - Y.min() if len(Y) else 1
    z_range = Z.max() - Z.min() if len(Z) else 1
    r = max(x_range, y_range, z_range) / 2.0
    x_mid = (X.max() + X.min()) / 2.0
    y_mid = (Y.max() + Y.min()) / 2.0
    z_mid = (Z.max() + Z.min()) / 2.0
    ax.set_xlim(x_mid - r, x_mid + r)
    ax.set_ylim(y_mid - r, y_mid + r)
    ax.set_zlim(z_mid - r, z_mid + r)


def build_3d_tubes(df_ids,
                   segments,
                   merged_group_id,
                   info: KDEInfo,
                   base_name: str,
                   results_dir: str | Path):
    """
    Build 3D centerlines by assigning TraceID‑averaged points to
    2D segment groups and then fitting a local Z coordinate.

    Parameters
    ----------
    df_ids : DataFrame
        Localization table with X,Y,Z,TraceID.
    segments : list of dict
        Segments from skeleton_graph.extract_and_merge_segments.
    merged_group_id : 1D array
        Group label per segment.
    info : KDEInfo
        KDE mapping (pixel⇔µm).
    base_name : str
        Sample name (for plot title).
    results_dir : str or Path
        Where to save the 3D plot.

    Returns
    -------
    df_avg : DataFrame
        One row per TraceID with mean X,Y,Z and N.
    assigned_gid : 1D np.ndarray
        Group label for each averaged point (or -1 for unassigned).
    seg_xyz_3d : dict[int, (N,3) float]
        3D centerline coordinates per segment id.
    """
    from collections import defaultdict
    import pandas as pd

    results_dir = ensure_dir(results_dir)

    df_avg = df_ids.groupby("TraceID", as_index=False).agg(
        X=("X", "mean"), Y=("Y", "mean"), Z=("Z", "mean"),
        N=("TraceID", "size")
    )

    avg_XY = df_avg[["X", "Y"]].to_numpy(float)
    avg_Z = df_avg["Z"].to_numpy(float)
    N_pts = len(df_avg)
    print(f"[TUBES] {base_name}: averaged {N_pts} TraceIDs.")

    labels = np.asarray(merged_group_id, int)
    unique_labels = np.unique(labels)
    n_groups = int(unique_labels.max()) + 1

    palette3d = _make_palette_3d(n_groups)
    lbl_to_color = {int(l): palette3d[int(l) % len(palette3d)]
                    for l in unique_labels}

    seg_xy_res: Dict[int, np.ndarray] = {}
    for sid, s in enumerate(segments):
        xy = _build_segment_polyline_um(s, info)
        seg_xy_res[sid] = _resample_polyline(xy, step=cfg.center_step)

    all_samples = []
    all_labels = []
    for sid, xy in seg_xy_res.items():
        if xy.size == 0:
            continue
        gid_ = int(labels[sid])
        all_samples.append(xy)
        all_labels.append(np.full(xy.shape[0], gid_, dtype=int))
    if not all_samples:
        raise RuntimeError("No segment samples were generated for 3D tubes.")

    SP = np.vstack(all_samples)
    LID = np.concatenate(all_labels)
    kd_sp = cKDTree(SP)

    d_min, idx_min = kd_sp.query(avg_XY, k=1)
    assigned_gid = np.where(d_min <= cfg.tube_radius, LID[idx_min], -1).astype(int)

    kd_avg = cKDTree(avg_XY)

    seg_xyz_3d: Dict[int, np.ndarray] = {}
    for sid, xy_samples in seg_xy_res.items():
        gid_ = int(labels[sid])
        if xy_samples.size == 0:
            continue
        xyz_collect = []
        for xy0 in xy_samples:
            idxs = kd_avg.query_ball_point(xy0, r=cfg.search_radius)
            if not idxs:
                continue
            idxs = np.array(idxs, int)
            dloc = np.linalg.norm(avg_XY[idxs] - xy0[None, :], axis=1)
            mask = (assigned_gid[idxs] == gid_) & (dloc <= cfg.tube_radius)
            if not np.any(mask):
                continue
            sel = idxs[mask]
            z_med = float(np.median(avg_Z[sel]))
            xy_cent = np.mean(avg_XY[sel], axis=0)
            shift = xy_cent - xy0
            dist = np.linalg.norm(shift)
            if dist > cfg.max_xy_shift:
                xy_adj = xy0 + (shift * (cfg.max_xy_shift / (dist + 1e-12)))
            else:
                xy_adj = xy_cent
            xyz_collect.append([xy_adj[0], xy_adj[1], z_med])
        seg_xyz = np.array(xyz_collect, float)
        if seg_xyz.shape[0] >= 30:
            seg_xyz_3d[sid] = seg_xyz

    print(f"[TUBES] {base_name}: built 3D centerlines for "
          f"{len(seg_xyz_3d)}/{len(segments)} segments.")

    # ---- plotting ----
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D tubes over averaged MINFLUX points\n"
                 f"(radius ≤ {cfg.tube_radius:.3f} µm; "
                 f"local XY adjust ≤ {cfg.max_xy_shift:.2f} µm)",
                 fontsize=12)

    mask_un = (assigned_gid < 0)
    if np.any(mask_un):
        ax.scatter(avg_XY[mask_un, 0], avg_XY[mask_un, 1], avg_Z[mask_un],
                   s=cfg.pt_size, c="0.6", alpha=0.35,
                   depthshade=False, marker=".")

    for gid_ in unique_labels:
        gid_ = int(gid_)
        m = (assigned_gid == gid_)
        if np.any(m):
            ax.scatter(avg_XY[m, 0], avg_XY[m, 1], avg_Z[m],
                       s=cfg.pt_size, c=[lbl_to_color[gid_]],
                       alpha=cfg.pt_alpha, depthshade=False, marker=".")

    for sid, xyz in seg_xyz_3d.items():
        gid_ = int(labels[sid])
        col = lbl_to_color[gid_]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color=col, alpha=cfg.line_alpha, linewidth=cfg.line_width)

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")

    allX = [avg_XY[:, 0]]
    allY = [avg_XY[:, 1]]
    allZ = [avg_Z]
    for xyz in seg_xyz_3d.values():
        allX.append(xyz[:, 0])
        allY.append(xyz[:, 1])
        allZ.append(xyz[:, 2])
    _set_3d_equal_axes(ax,
                       np.concatenate(allX),
                       np.concatenate(allY),
                       np.concatenate(allZ))

    plt.tight_layout()
    fig.savefig(Path(results_dir) / "3D_tubes_avg_points.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    return df_avg, assigned_gid, seg_xyz_3d
