"""
cylinder_growth.py
==================

Stage 5: grow cylindrical fibers from the 3D tubes, assign each
TraceID‑averaged point to a fiber, and compute per‑fiber summary
statistics (length and width).

Entry point:
    run_cylinder_growth(df_ids, df_avg, assigned_gid,
                        base_name, results_dir)

Returns:
    df_avg_out    - df_avg with 'fiber_grow_id'
    df_ids_out    - df_ids with 'fiber_grow_id'
    fiber_metrics - DataFrame with columns
                    ['fiber_id', 'fiber_length_um', 'fiber_width_um']
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys
from scipy.spatial import cKDTree

from config import PIPELINE_CONFIG as cfg
from io_utils import ensure_dir


def _first_pc(points: np.ndarray):
    if points.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0], float), points.mean(axis=0)
    mu = points.mean(axis=0)
    Xc = points - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    v = Vt[0]
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        v = np.array([1.0, 0.0, 0.0], float)
    else:
        v = v / nrm
    return v, mu


def _make_palette_cyl(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    return [colorsys.hsv_to_rgb(float(h), 0.85, 1.0) for h in hues]


def _set_3d_equal_cyl(ax, pts: np.ndarray):
    Xc, Yc, Zc = pts[:, 0], pts[:, 1], pts[:, 2]
    x_range = Xc.max() - Xc.min()
    y_range = Yc.max() - Yc.min()
    z_range = Zc.max() - Zc.min()
    r = max(x_range, y_range, z_range) / 2.0
    x_mid = (Xc.max() + Xc.min()) / 2.0
    y_mid = (Yc.max() + Yc.min()) / 2.0
    z_mid = (Zc.max() + Zc.min()) / 2.0
    ax.set_xlim(x_mid - r, x_mid + r)
    ax.set_ylim(y_mid - r, y_mid + r)
    ax.set_zlim(z_mid - r, z_mid + r)


def _grow_fiber_once(fid,
                     P,
                     labels,
                     kd_all,
                     fiber_rank):
    """
    Single growth iteration for a given fiber id *fid*.

    This is almost identical to your notebook logic, but using the
    configuration from config.PIPELINE_CONFIG.
    """
    idx_f = np.where(labels == fid)[0]
    if idx_f.size < cfg.min_pts_fiber:
        return False

    Pf = P[idx_f]
    v_global, mu_f = _first_pc(Pf)
    t_f = (Pf - mu_f) @ v_global
    t_min, t_max = t_f.min(), t_f.max()

    changed = False

    for is_max in (True, False):
        if is_max:
            mask_end = t_f >= (t_max - cfg.end_slice_len)
        else:
            mask_end = t_f <= (t_min + cfg.end_slice_len)
        idx_end_local = np.where(mask_end)[0]
        if idx_end_local.size < cfg.min_pts_end:
            continue

        P_end = Pf[idx_end_local]
        v_end, mu_end = _first_pc(P_end)

        dot_ve_vf = float(np.dot(v_end, v_global))
        if is_max:
            v_out = -v_end if dot_ve_vf < 0 else v_end
        else:
            v_out = -v_end if dot_ve_vf > 0 else v_end
        v_out = v_out / (np.linalg.norm(v_out) + 1e-12)

        rel_end = P_end - mu_end
        t_end = rel_end @ v_out
        radial_vec = rel_end - np.outer(t_end, v_out)
        r_vals = np.linalg.norm(radial_vec, axis=1)
        if r_vals.size == 0:
            continue

        t_tip = float(t_end.max())
        r_est = float(np.percentile(r_vals, 95) + 0.01)
        r_end = min(cfg.max_radius, r_est)
        if r_end <= 0:
            continue

        t_start = t_tip
        t_stop = t_tip + cfg.extend_step
        t_mid = 0.5 * (t_start + t_stop)
        p_mid = mu_end + v_out * t_mid
        R_sphere = float(np.sqrt((cfg.extend_step / 2.0) ** 2 + r_end ** 2))

        idx_cand = kd_all.query_ball_point(p_mid, r=R_sphere)
        if not idx_cand:
            continue

        to_add = []
        for j in idx_cand:
            lab_j = int(labels[j])
            if lab_j >= 0 and fiber_rank[lab_j] < fiber_rank[fid]:
                continue

            rel = P[j] - mu_end
            t_j = float(np.dot(rel, v_out))
            if not (t_start < t_j <= t_stop):
                continue
            r_vec = rel - t_j * v_out
            r_j = float(np.linalg.norm(r_vec))
            if r_j <= r_end:
                to_add.append(j)

        if to_add:
            labels[np.array(to_add, int)] = fid
            changed = True

    return changed


def _compute_fiber_metrics(P_valid: np.ndarray,
                           labels_valid: np.ndarray):
    """
    Compute per‑fiber length and width based on final labels.

    Length: extent of projection onto first principal component (µm).
    Width : 2 * 95th percentile of radial distance from that axis (µm).
    """
    metrics = []
    final_fibers = sorted(set(int(l) for l in labels_valid.tolist() if l >= 0))
    for fid in final_fibers:
        idx = np.where(labels_valid == fid)[0]
        if idx.size < 2:
            continue
        Pf = P_valid[idx]
        v, mu = _first_pc(Pf)
        t = (Pf - mu) @ v
        length = float(t.max() - t.min())

        rel = Pf - mu
        t_vals = rel @ v
        radial_vec = rel - np.outer(t_vals, v)
        r_vals = np.linalg.norm(radial_vec, axis=1)
        width = 2.0 * float(np.percentile(r_vals, 95))

        metrics.append((fid, length, width))

    df_metrics = pd.DataFrame(metrics,
                              columns=["fiber_id",
                                       "fiber_length_um",
                                       "fiber_width_um"])
    return df_metrics


def run_cylinder_growth(df_ids,
                        df_avg,
                        assigned_gid: np.ndarray,
                        base_name: str,
                        results_dir: str | Path):
    """
    Run the cylinder‑growth algorithm starting from `assigned_gid`
    and compute final fiber labels + metrics.

    Parameters
    ----------
    df_ids : DataFrame
        Original localization table with repeated TraceIDs.
    df_avg : DataFrame
        Per‑TraceID averages from tubes3d.build_3d_tubes.
    assigned_gid : 1D array
        Initial group id for each averaged point (or -1).
    base_name : str
        Sample name for plot titles.
    results_dir : str or Path
        Folder where plots and CSVs are written.

    Returns
    -------
    df_avg_out : DataFrame
        df_avg with column 'fiber_grow_id'.
    df_ids_out : DataFrame
        df_ids with column 'fiber_grow_id'.
    fiber_metrics : DataFrame
        Columns: ['fiber_id', 'fiber_length_um', 'fiber_width_um'].
    """
    results_dir = ensure_dir(results_dir)

    P_all = df_avg[["X", "Y", "Z"]].to_numpy(float)
    labels_all_init = np.asarray(assigned_gid, int)

    mask_valid = (labels_all_init >= 0)
    idx_valid = np.nonzero(mask_valid)[0]
    P = P_all[mask_valid]
    labels = labels_all_init[mask_valid].copy()
    Ns = P.shape[0]

    print(f"[GROW] {base_name}: using {Ns} / {len(P_all)} averaged points "
          "(assigned_gid >= 0).")

    fibers = sorted(set(labels.tolist()))
    fiber_length_init = {}
    for fid in fibers:
        idx_f = np.where(labels == fid)[0]
        if idx_f.size < cfg.min_pts_fiber:
            fiber_length_init[fid] = 0.0
            continue
        Pf = P[idx_f]
        v, mu = _first_pc(Pf)
        t = (Pf - mu) @ v
        fiber_length_init[fid] = t.max() - t.min()

    fiber_order = sorted(fibers, key=lambda fid: -fiber_length_init[fid])
    fiber_rank = {fid: rank for rank, fid in enumerate(fiber_order)}

    print("[GROW] example initial fiber lengths (µm):")
    for fid in fiber_order[:10]:
        print(f"  Fiber {fid}: {fiber_length_init[fid]:.3f} µm")

    kd_all = cKDTree(P)

    print("[GROW] starting cylinder growth iterations...")
    for it in range(1, cfg.max_global_iters + 1):
        changed_any = False
        print(f"[GROW] global iteration {it} ...")
        for fid in fiber_order:
            changed = _grow_fiber_once(fid, P, labels, kd_all, fiber_rank)
            if changed:
                changed_any = True
        if not changed_any:
            print("[GROW] no label changes in this pass; stopping.")
            break

    final_fibers = sorted(set(labels.tolist()))
    print(f"[GROW] finished. Final distinct fiber labels (non‑noise): "
          f"{len(final_fibers)}")

    # labels back to df_avg
    fiber_grow_full = -1 * np.ones(len(df_avg), int)
    fiber_grow_full[idx_valid] = labels
    df_avg_out = df_avg.copy()
    df_avg_out["fiber_grow_id"] = fiber_grow_full

    # propagate to df_ids
    df_ids_out = df_ids.merge(
        df_avg_out[["TraceID", "fiber_grow_id"]],
        on="TraceID",
        how="left"
    )

    # metrics
    fiber_metrics = _compute_fiber_metrics(P, labels)

    # ---- plotting of grown fibers ----
    P_plot = df_avg_out[["X", "Y", "Z"]].to_numpy(float)
    lab_plot = df_avg_out["fiber_grow_id"].to_numpy(int)

    uniq = sorted([l for l in np.unique(lab_plot) if l >= 0])
    palette_cyl = _make_palette_cyl(len(uniq))
    lbl_to_color_cyl = {l: palette_cyl[i] for i, l in enumerate(uniq)}

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D cylinder‑grown fibers (starting from assigned_gid)",
                 fontsize=13)

    mask_noise = (lab_plot < 0)
    if np.any(mask_noise):
        ax.scatter(P_plot[mask_noise, 0],
                   P_plot[mask_noise, 1],
                   P_plot[mask_noise, 2],
                   s=cfg.pt_size_cyl, c="0.7",
                   alpha=cfg.noise_alpha_cyl,
                   depthshade=False, marker=".")

    for l in uniq:
        m = (lab_plot == l)
        col = lbl_to_color_cyl[l]
        ax.scatter(P_plot[m, 0],
                   P_plot[m, 1],
                   P_plot[m, 2],
                   s=cfg.pt_size_cyl, c=[col],
                   alpha=cfg.pt_alpha_cyl,
                   depthshade=False, marker=".")

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")

    _set_3d_equal_cyl(ax, P_plot)
    plt.tight_layout()
    fig.savefig(Path(results_dir) / "3D_cylinder_grown_fibers.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save metrics CSV (3 columns) here
    metrics_path = Path(results_dir) / "fiber_metrics.csv"
    fiber_metrics.to_csv(metrics_path, index=False)
    print(f"[METRICS] saved fiber metrics to {metrics_path}")

    return df_avg_out, df_ids_out, fiber_metrics
