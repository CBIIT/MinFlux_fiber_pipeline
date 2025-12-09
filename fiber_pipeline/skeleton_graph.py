"""
skeleton_graph.py
=================

Build a graph of segments from a 2D binary skeleton image, merge
segments into fiber groups based on angle and proximity, and split
branches at junctions.

Entry point:
    extract_and_merge_segments(skeleton_path, results_dir)

Outputs:
    segments         - list of dicts with keys u,v,pixels,anchor_u,anchor_v
    merged_group_id  - np.ndarray, group label for each segment
    nodes            - dict of node information (junctions/endpoints)
    H, W             - skeleton image dimensions
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import colorsys
from scipy.ndimage import convolve as nd_convolve, label as cc_label, center_of_mass
from scipy.spatial import cKDTree
from skimage import io as skio_io, color as skio_color
from skimage.draw import line as skline

from config import PIPELINE_CONFIG as cfg
from io_utils import ensure_dir


def _orientation_deg_0_180(p1, p2) -> float:
    dy = float(p2[0] - p1[0])
    dx = float(p2[1] - p1[1])
    ang = np.degrees(np.arctan2(dy, dx))
    return float((ang + 180.0) % 180.0)


def _acute_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return float(d if d <= 90.0 else 180.0 - d)


def _distinct_palette(n: int, seed: int, s: float = 0.85, v: float = 1.0):
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    return [colorsys.hsv_to_rgb(h, s, v) for h in hues]


def extract_and_merge_segments(skeleton_path: str | Path,
                               results_dir: str | Path):
    """
    Main function for skeleton graph construction and merging.

    Parameters
    ----------
    skeleton_path : str or Path
        Path to the binary skeleton image.
    results_dir : str or Path
        Folder where diagnostic overlays are saved.

    Returns
    -------
    segments : list of dict
    merged_group_id : np.ndarray (len = #segments)
    nodes : dict
    H, W : int
        Dimensions of the skeleton image.
    """
    results_dir = ensure_dir(results_dir)
    skel_img = skio_io.imread(skeleton_path)
    if skel_img.ndim == 3:
        skel_img = skio_color.rgb2gray(skel_img)
    skeleton = skel_img > 0.5
    H, W = skeleton.shape

    # ---------- Build nodes and initial segments ----------
    sk = skeleton.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    nb_count = nd_convolve(sk, kernel, mode="constant", cval=0)
    neighbors = nb_count - sk
    degree = neighbors * sk

    end_mask = (sk == 1) & (degree == 1)
    junction_mask = (sk == 1) & (degree >= 3)

    jlab, nJ = cc_label(junction_mask, structure=np.ones((3, 3), np.uint8))
    ey, ex = np.nonzero(end_mask)

    nodes = {}
    node_id = -np.ones((H, W), np.int32)
    nid = 0

    for lbl in range(1, nJ + 1):
        ys, xs = np.nonzero(jlab == lbl)
        if ys.size == 0:
            continue
        cy, cx = center_of_mass(junction_mask, jlab, lbl)
        nodes[nid] = {
            "type": "J",
            "pixels": list(zip(ys, xs)),
            "centroid": (float(cy), float(cx)),
        }
        node_id[ys, xs] = nid
        nid += 1

    for (y, x) in zip(ey, ex):
        nodes[nid] = {
            "type": "E",
            "pixels": [(y, x)],
            "centroid": (float(y), float(x)),
        }
        node_id[y, x] = nid
        nid += 1

    N_nodes = nid
    if cfg.print_counts:
        print(f"[GRAPH] Junctions: {nJ} | Endpoints: {len(ey)} | Nodes total: {N_nodes}")

    OFFS = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    def nbors(y, x):
        lst = []
        for dy, dx in OFFS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skeleton[ny, nx]:
                lst.append((ny, nx))
        return lst

    def junction_exits(nid_):
        S = set(nodes[nid_]["pixels"])
        exits = {}
        for (py, px) in nodes[nid_]["pixels"]:
            for (ny, nx) in nbors(py, px):
                if (ny, nx) not in S:
                    exits.setdefault((ny, nx), (py, px))
        return [(outp, exits[outp]) for outp in exits]

    def endpoint_exits(nid_):
        (py, px) = nodes[nid_]["pixels"][0]
        return [(nbr, (py, px)) for nbr in nbors(py, px)]

    visited = set()
    segments = []

    def lin_idx(p): return p[0] * W + p[1]

    for nid0 in range(N_nodes):
        starts = (junction_exits(nid0) if nodes[nid0]["type"] == "J"
                  else endpoint_exits(nid0))
        for (out_pix, in_pix) in starts:
            sy, sx = in_pix
            cy, cx = out_pix
            e0 = tuple(sorted((lin_idx((sy, sx)), lin_idx((cy, cx)))))
            if e0 in visited:
                continue

            path = [(sy, sx), (cy, cx)]
            visited.add(e0)
            prev, cur = (sy, sx), (cy, cx)

            while True:
                nn = nbors(*cur)
                if prev in nn:
                    nn.remove(prev)

                node_nei = [(yy, xx) for (yy, xx) in nn if node_id[yy, xx] >= 0]
                int_nei = [(yy, xx) for (yy, xx) in nn if node_id[yy, xx] < 0]

                if node_nei:
                    oy, ox = node_nei[0]
                    nid1 = node_id[oy, ox]
                    if nid1 != nid0:
                        segments.append({
                            "u": nid0, "v": nid1,
                            "pixels": path,
                            "anchor_u": (sy, sx),
                            "anchor_v": (oy, ox),
                        })
                    break

                if len(int_nei) == 1:
                    nxt = int_nei[0]
                    e2 = tuple(sorted((lin_idx(cur), lin_idx(nxt))))
                    if e2 in visited:
                        break
                    visited.add(e2)
                    path.append(nxt)
                    prev, cur = cur, nxt
                else:
                    break

    # add missing direct node‑node edges
    seen_pairs = set((min(s["u"], s["v"]), max(s["u"], s["v"])) for s in segments)
    for nid0 in range(N_nodes):
        for (py, px) in nodes[nid0]["pixels"]:
            for (ny, nx) in nbors(py, px):
                nid1 = node_id[ny, nx]
                if nid1 >= 0 and nid1 != nid0:
                    key = (min(nid0, nid1), max(nid0, nid1))
                    if key not in seen_pairs:
                        segments.append({
                            "u": key[0], "v": key[1],
                            "pixels": [],
                            "anchor_u": (py, px),
                            "anchor_v": (ny, nx),
                        })
                        seen_pairs.add(key)

    uniq_seg = {}
    for s in segments:
        key = (min(s["u"], s["v"]), max(s["u"], s["v"]))
        if key not in uniq_seg or len(s["pixels"]) > len(uniq_seg[key]["pixels"]):
            uniq_seg[key] = s
    segments = list(uniq_seg.values())

    for s in segments:
        s["theta"] = _orientation_deg_0_180(
            nodes[s["u"]]["centroid"],
            nodes[s["v"]]["centroid"]
        )

    if cfg.print_counts:
        print(f"[GRAPH] Segments found: {len(segments)}")

    # ---------- First local junction‑only merges ----------
    junc_to_segments = defaultdict(list)
    for i, s in enumerate(segments):
        if nodes[s["u"]]["type"] == "J":
            junc_to_segments[s["u"]].append(i)
        if nodes[s["v"]]["type"] == "J":
            junc_to_segments[s["v"]].append(i)

    merge_neighbors = defaultdict(set)
    for jnid, ids in junc_to_segments.items():
        if len(ids) < 2:
            continue
        close = []
        involved = set()
        for a, b in combinations(ids, 2):
            d = _acute_diff(segments[a]["theta"], segments[b]["theta"])
            if d < cfg.angle_thresh_deg:
                close.append((a, b))
                involved.update([a, b])

        if len(involved) > 2:
            continue
        if len(close) == 1:
            a, b = close[0]
            merge_neighbors[a].add(b)
            merge_neighbors[b].add(a)

    class UF:
        def __init__(self, n):
            self.p = list(range(n))
            self.sz = [1] * n

        def find(self, x):
            while self.p[x] != x:
                self.p[x] = self.p[self.p[x]]
                x = self.p[x]
            return x

        def pair(self, a, b):
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            if self.sz[ra] >= 2 or self.sz[rb] >= 2:
                return
            self.p[rb] = ra
            self.sz[ra] += self.sz[rb]

    uf = UF(len(segments))
    accepted = 0
    for i in range(len(segments)):
        nbrs = merge_neighbors.get(i, set())
        if len(nbrs) == 1:
            j = next(iter(nbrs))
            if len(merge_neighbors.get(j, set())) == 1 and next(iter(merge_neighbors[j])) == i:
                if i < j:
                    uf.pair(i, j)
                    accepted += 1

    roots = np.array([uf.find(i) for i in range(len(segments))], int)
    uniq_roots = sorted(set(roots))
    root_to_gid = {r: k for k, r in enumerate(uniq_roots)}
    group_id = np.array([root_to_gid[r] for r in roots], int)
    n_groups = len(uniq_roots)
    if cfg.print_counts:
        print(f"[GRAPH] Accepted pairs: {accepted} | Groups after step1: {n_groups}")

    palette = _distinct_palette(max(n_groups, 1), cfg.palette_seed)
    rgb = np.zeros((H, W, 3), float)

    def paint_pixel(y, x, col):
        if 0 <= y < H and 0 <= x < W:
            rgb[y, x] = col

    for i, s in enumerate(segments):
        col = palette[group_id[i]]
        ay1, ax1 = map(int, s["anchor_u"])
        ay2, ax2 = map(int, s["anchor_v"])
        paint_pixel(ay1, ax1, col)
        paint_pixel(ay2, ax2, col)
        for (y, x) in s["pixels"]:
            paint_pixel(y, x, col)
        for nid_local, (ay, ax) in [(s["u"], s["anchor_u"]), (s["v"], s["anchor_v"])]:
            if nodes[nid_local]["type"] == "J":
                cy, cx = nodes[nid_local]["centroid"]
                cyi, cxi = int(round(cy)), int(round(cx))
                rr, cc = skline(int(ay), int(ax), cyi, cxi)
                for yy, xx in zip(rr, cc):
                    paint_pixel(yy, xx, col)
        if len(s["pixels"]) == 0:
            rr, cc = skline(ay1, ax1, ay2, ax2)
            for yy, xx in zip(rr, cc):
                paint_pixel(yy, xx, col)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_title("Segments merged (adjacent at junction, mutual & unique)",
                 color="w")
    plt.tight_layout()
    fig.savefig(Path(results_dir) / "segments_merged_adjacent_mutual_unique.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- Near‑neighbor merging (two passes + branch split) ----------
    # For brevity and faithful reproduction, this is almost identical
    # to your notebook code, just wrapped into this function.

    # --- helper definitions reused in the two passes ---
    def build_adjacency_and_features():
        n_seg = len(segments)
        seg_theta = np.zeros(n_seg, float)
        for i, s in enumerate(segments):
            u, v = s["u"], s["v"]
            seg_theta[i] = _orientation_deg_0_180(
                nodes[u]["centroid"], nodes[v]["centroid"]
            )

        node_to_segments = defaultdict(list)
        for i, s in enumerate(segments):
            node_to_segments[s["u"]].append(i)
            node_to_segments[s["v"]].append(i)

        adjacent_segments = defaultdict(set)
        for nid_, segs_at_node in node_to_segments.items():
            if nodes[nid_]["type"] != "J":
                continue
            if len(segs_at_node) < 2:
                continue
            for a in segs_at_node:
                for b in segs_at_node:
                    if a != b:
                        adjacent_segments[a].add(b)

        ep_coords = []
        ep_owner = []
        for sid, s in enumerate(segments):
            if nodes[s["u"]]["type"] == "E":
                y, x = map(float, s["anchor_u"])
                ep_coords.append((y, x))
                ep_owner.append(sid)
            if nodes[s["v"]]["type"] == "E":
                y, x = map(float, s["anchor_v"])
                ep_coords.append((y, x))
                ep_owner.append(sid)

        ep_coords_arr = np.asarray(ep_coords, float)
        ep_owner_arr = np.asarray(ep_owner, int)

        if len(ep_coords_arr) >= 2:
            tree_ep = cKDTree(ep_coords_arr)
            for i_idx, j_idx in tree_ep.query_pairs(r=cfg.near_radius_px,
                                                    output_type="set"):
                si, sj = ep_owner_arr[i_idx], ep_owner_arr[j_idx]
                if si != sj:
                    adjacent_segments[si].add(sj)
                    adjacent_segments[sj].add(si)

        junc_ids = [nid_ for nid_ in range(len(nodes))
                    if nodes[nid_]["type"] == "J"]
        if len(junc_ids) and len(ep_coords_arr):
            j_centers = np.array([nodes[nid]["centroid"] for nid in junc_ids], float)
            tree_j = cKDTree(j_centers)
            idx_to_nid = {k: nid for k, nid in enumerate(junc_ids)}
            for k, (y, x) in enumerate(ep_coords_arr):
                si = ep_owner_arr[k]
                hits = tree_j.query_ball_point([y, x], r=cfg.near_radius_px)
                for jidx in hits:
                    nid_ = idx_to_nid[jidx]
                    for sj in node_to_segments.get(nid_, []):
                        if sj != si:
                            adjacent_segments[si].add(sj)
                            adjacent_segments[sj].add(si)

        seg_anchors = []
        for s in segments:
            ay1, ax1 = map(float, s["anchor_u"])
            ay2, ax2 = map(float, s["anchor_v"])
            seg_anchors.append(np.array([[ay1, ax1], [ay2, ax2]], float))
        seg_anchors_arr = np.asarray(seg_anchors, float)

        def min_anchor_dist(si, sj):
            A = seg_anchors_arr[si]
            B = seg_anchors_arr[sj]
            d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
            return float(np.sqrt(d2.min()))

        def group_orientation(seg_list):
            if len(seg_list) == 1:
                return seg_theta[seg_list[0]]
            angs = np.deg2rad(seg_theta[seg_list])
            Xc = np.cos(2.0 * angs).mean()
            Yc = np.sin(2.0 * angs).mean()
            if Xc == 0 and Yc == 0:
                return float((np.rad2deg(angs.mean()) + 180.0) % 180.0)
            mean2 = np.arctan2(Yc, Xc)
            mean = 0.5 * mean2
            return float((np.rad2deg(mean) + 180.0) % 180.0)

        def group_min_anchor_distance(seg_list_g, seg_list_h):
            m = np.inf
            for si in seg_list_g:
                for sj in seg_list_h:
                    if si == sj:
                        continue
                    m = min(m, min_anchor_dist(si, sj))
            return float(m)

        return (seg_theta, node_to_segments,
                adjacent_segments, min_anchor_dist,
                group_orientation, group_min_anchor_distance,
                junc_ids)

    # --- generic mutual‑best merge step ---
    def run_mutual_best_merge(initial_group_id,
                              seg_theta,
                              adjacent_segments,
                              group_orientation,
                              group_min_anchor_distance,
                              label: str):
        group_id = initial_group_id.copy()
        it = 0
        while True:
            it += 1
            groups = defaultdict(list)
            for seg_idx, gid_ in enumerate(group_id):
                groups[gid_].append(seg_idx)
            uniq_gids = sorted(groups.keys())
            remap = {g: k for k, g in enumerate(uniq_gids)}
            G = {remap[g]: np.array(segs_, int) for g, segs_ in groups.items()}
            n_groups = len(G)

            if cfg.print_counts:
                print(f"[{label}] Iter {it}: #groups = {n_groups}")

            theta_g = np.zeros(n_groups, float)
            for g in range(n_groups):
                theta_g[g] = group_orientation(G[g])

            grp_adj = defaultdict(set)
            for g in range(n_groups):
                for si in G[g]:
                    for sj in adjacent_segments[si]:
                        gh = remap[group_id[sj]]
                        if gh != g:
                            grp_adj[g].add(gh)

            best_neighbor = {}
            for g in range(n_groups):
                if not grp_adj[g]:
                    continue
                cand = []
                for h in grp_adj[g]:
                    d_ang = _acute_diff(theta_g[g], theta_g[h])
                    if d_ang < cfg.angle_thresh_deg:
                        d_dist = group_min_anchor_distance(G[g], G[h])
                        cand.append((h, d_ang, d_dist))
                if not cand:
                    continue
                cand.sort(key=lambda t: (t[1], t[2]))
                best_neighbor[g] = cand[0]

            pair_candidates = []
            for g, (h, ang_g, dist_g) in best_neighbor.items():
                if h in best_neighbor and best_neighbor[h][0] == g:
                    if g < h:
                        pair_candidates.append((g, h))

            if cfg.print_counts:
                print(f"[{label}] Iter {it}: candidate merges = {len(pair_candidates)}")

            if not pair_candidates or it >= cfg.max_merge_iters:
                if cfg.print_counts:
                    if not pair_candidates:
                        print(f"[{label}] no more merges; stopping.")
                    else:
                        print(f"[{label}] reached max_merge_iters; stopping.")
                break

            class UFG:
                def __init__(self, n):
                    self.p = list(range(n))

                def find(self, a):
                    while self.p[a] != a:
                        self.p[a] = self.p[self.p[a]]
                        a = self.p[a]
                    return a

                def union(self, a, b):
                    ra, rb = self.find(a), self.find(b)
                    if ra != rb:
                        self.p[rb] = ra

            uf2 = UFG(n_groups)
            for g, h in pair_candidates:
                uf2.union(g, h)

            new_root = np.array([uf2.find(g) for g in range(n_groups)], int)
            uniq_roots2 = sorted(set(new_root))
            root2gid2 = {r: k for k, r in enumerate(uniq_roots2)}

            new_group_id = np.zeros_like(group_id)
            for g in range(n_groups):
                r = new_root[g]
                g_coarse = root2gid2[r]
                for si in G[g]:
                    new_group_id[si] = g_coarse
            group_id = new_group_id

        return group_id

    (seg_theta, node_to_segments,
     adjacent_segments, min_anchor_dist,
     group_orientation, group_min_anchor_distance,
     junc_ids) = build_adjacency_and_features()

    # first mutual‑best merging (near neighbors)
    group_id = np.arange(len(segments), int)
    group_id = run_mutual_best_merge(group_id,
                                     seg_theta,
                                     adjacent_segments,
                                     group_orientation,
                                     group_min_anchor_distance,
                                     "MERGE1")

    # second mutual‑best merging with the same criteria
    merged_group_id = run_mutual_best_merge(group_id,
                                            seg_theta,
                                            adjacent_segments,
                                            group_orientation,
                                            group_min_anchor_distance,
                                            "MERGE2")

    n_final_groups = int(merged_group_id.max()) + 1
    if cfg.print_counts:
        print(f"[GRAPH] After mutual‑best merging: {n_final_groups} groups")

    # ---------- Post: branch splitting at junctions ----------
    def local_orientation_at_node(seg_idx, nid_):
        s = segments[seg_idx]
        if nid_ == s["u"]:
            y0, x0 = s["anchor_u"]
            if len(s["pixels"]) >= 2:
                p0 = tuple(s["pixels"][0])
                p1 = tuple(s["pixels"][1])
                if p0 == (int(round(y0)), int(round(x0))):
                    y1, x1 = p1
                else:
                    y1, x1 = p0
            elif len(s["pixels"]) == 1:
                y1, x1 = s["pixels"][0]
            else:
                y1, x1 = s["anchor_v"]
            return _orientation_deg_0_180((float(y0), float(x0)),
                                          (float(y1), float(x1)))
        elif nid_ == s["v"]:
            y0, x0 = s["anchor_v"]
            if len(s["pixels"]) >= 1:
                y1, x1 = s["pixels"][-1]
            else:
                y1, x1 = s["anchor_u"]
            return _orientation_deg_0_180((float(y0), float(x0)),
                                          (float(y1), float(x1)))
        else:
            return seg_theta[seg_idx]

    next_label = int(merged_group_id.max()) + 1

    from itertools import combinations as comb2

    for nid_ in junc_ids:
        segs_here = node_to_segments.get(nid_, [])
        if len(segs_here) < 3:
            continue

        by_label = defaultdict(list)
        for si in segs_here:
            by_label[int(merged_group_id[si])].append(si)

        for lbl, segs_same in by_label.items():
            if len(segs_same) <= 2:
                continue

            theta_loc = {si: local_orientation_at_node(si, nid_) for si in segs_same}

            keep_pair = None
            best_angle = np.inf
            for a, b in comb2(segs_same, 2):
                da = _acute_diff(theta_loc[a], theta_loc[b])
                if da < best_angle:
                    best_angle = da
                    keep_pair = (a, b)

            keep_set = set(keep_pair) if keep_pair is not None else set()
            for si in segs_same:
                if si in keep_set:
                    continue
                merged_group_id[si] = next_label
                next_label += 1

    n_final_groups = int(merged_group_id.max()) + 1
    if cfg.print_counts:
        print(f"[GRAPH] After branch splitting: {n_final_groups} groups")

    colors3 = _distinct_palette(n_final_groups, cfg.palette_seed)
    rgb3 = np.zeros((H, W, 3), float)

    def paint3(y, x, c):
        if 0 <= y < H and 0 <= x < W:
            rgb3[y, x] = c

    for sid, s in enumerate(segments):
        col = colors3[int(merged_group_id[sid])]
        ay1, ax1 = map(int, s["anchor_u"])
        ay2, ax2 = map(int, s["anchor_v"])
        paint3(ay1, ax1, col)
        paint3(ay2, ax2, col)
        for (y, x) in s["pixels"]:
            paint3(y, x, col)
        for nid_local, (ay, ax) in [(s["u"], s["anchor_u"]), (s["v"], s["anchor_v"])]:
            if nodes[nid_local]["type"] == "J":
                cy, cx = nodes[nid_local]["centroid"]
                cyi, cxi = int(round(cy)), int(round(cx))
                rr, cc = skline(int(ay), int(ax), cyi, cxi)
                for yy, xx in zip(rr, cc):
                    paint3(yy, xx, col)
        if len(s["pixels"]) == 0:
            rr, cc = skline(ay1, ax1, ay2, ax2)
            for yy, xx in zip(rr, cc):
                paint3(yy, xx, col)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(rgb3, interpolation="nearest")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_title("Final merged segments (near neighbor + branch split)",
                 color="white", fontsize=14)
    plt.tight_layout()
    fig.savefig(Path(results_dir) / "segments_final_branch_split.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    return segments, merged_group_id, nodes, H, W
