# MINFLUX Fiber Segmentation Pipeline

Segmentation pipeline for assigning 3D MINFLUX localizations to individual **actin / filamentous fibers**.

The code:

- builds a 2D density image from MINFLUX localizations,
- segments fibers in 2D (XY) using a tubeness filter and skeletonization,
- merges skeleton segments into longer **fiber segments (tracks)**,
- lifts those segments into 3D by assigning MINFLUX localizations to fibers,
- optionally grows fibers using a 3D cylinder model, and
- writes per‑localization labels and **per‑fiber length / width statistics**.

> 💡 Throughout this README the term **“segmentation”** is used deliberately:  
> the goal is to segment localizations into fibers, not to reconstruct full physical geometry.

---

## Table of Contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Input data & folder structure](#input-data--folder-structure)
4. [Quick start](#quick-start)
5. [What the pipeline does (high‑level)](#what-the-pipeline-does-high-level)
6. [Algorithm details](#algorithm-details)
7. [Outputs](#outputs)
8. [Hyperparameters & tuning guide](#hyperparameters--tuning-guide)
9. [Fiber length & width estimation](#fiber-length--width-estimation)
10. [Tips, troubleshooting, and limitations](#tips-troubleshooting-and-limitations)

---

## Repository layout

This README assumes the code lives under:

```text
fiber_pipeline/
    __init__.py
    config.py
    io_utils.py
    kde_tubeness.py
    segment_2d.py
    skeleton_graph.py
    merge_segments.py
    tubes_3d.py
    grow_cylinders.py
    metrics.py
    run_folder.py
```

The logical roles of the modules are:

* **`config.py`**
  Central place for **all configuration and hyperparameters**:
  pixel size, KDE parameters, tubeness / segmentation parameters, merging thresholds,
  tube radii, cylinder growth settings, and input/output root folders.

* **`io_utils.py`**
  Utilities for:

  * scanning a folder for MINFLUX CSVs,
  * creating per‑sample result directories,
  * consistent base name handling,
  * safe CSV and image I/O.

* **`kde_tubeness.py`**
  2D preprocessing:

  * constructs a kernel density estimate (KDE) image in XY from raw localizations,
  * blurs the counts with a Gaussian kernel,
  * normalizes to 8‑bit,
  * applies a tubeness‑like filter (Sato) to enhance filamentous structures.

* **`segment_2d.py`**
  2D fiber **segmentation**:

  * grayscale normalization and denoising,
  * multi‑scale ridge enhancement,
  * hysteresis thresholding,
  * morphological cleanup (remove small objects, close gaps, fill holes),
  * optional geodesic gap bridging,
  * skeletonization.

* **`skeleton_graph.py`**
  Converts the binary skeleton into a **graph**:

  * identifies junction nodes and endpoints,
  * collapses multi‑pixel junctions to a single node with a centroid,
  * extracts centerline segments between nodes,
  * stores per‑segment pixel chains and anchor points at each node.

* **`merge_segments.py`**
  Rules for merging skeleton segments into longer 2D fibers:

  * orientation‑based merging at junctions,
  * near‑neighbor merging (endpoint‑to‑endpoint / endpoint‑to‑junction),
  * mutual best‑neighbor merging to prevent runaway chain merging,
  * branch splitting to enforce at most two arms per fiber at any junction.

* **`tubes_3d.py`**
  Lifts 2D segmented fibers into 3D:

  * averages localizations by `TraceID`,
  * converts 2D pixel polylines to XY in physical units (µm),
  * resamples centerlines at fixed arc‑length steps,
  * assigns averaged points to the nearest 2D fiber within a tube radius,
  * builds 3D tube centerlines using robust Z statistics and bounded XY shifts.

* **`grow_cylinders.py`**
  Optional 3D **cylinder growth** stage:

  * fits local principal components (PC1) at each fiber end,
  * extrapolates along PC1 using cylindrical neighborhoods,
  * reassigns points inside the extrapolated cylinder to the growing fiber,
  * ensures longer fibers can “steal” points from shorter fibers but not vice versa,
  * outputs refined per‑point labels `fiber_grow_id`.

* **`metrics.py`**
  Computes **per‑fiber length and width** using final fiber labels and writes a CSV with:

  * `fiber_id`
  * `fiber_length_um`
  * `fiber_width_um`

* **`run_folder.py`** (entry point)
  Batch runner that:

  * reads configuration from `config.py`,
  * scans a user‑provided input folder for MINFLUX CSVs,
  * runs the full segmentation pipeline for each valid sample,
  * saves all results in per‑sample subfolders under a parent results directory.

You should be able to operate the entire pipeline by editing **only** `config.py` and then running `run_folder.py`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CBIIT/MinFlux_fiber_pipeline.git
cd MinFlux_fiber_pipeline
```

If your code lives specifically in the `fiber_pipeline` subfolder, you should see:

```bash
ls fiber_pipeline
```

showing the Python modules (`config.py`, `run_folder.py`, etc.).

### 2. Create a Python environment (recommended)

Use Conda or `venv`. Python ≥ 3.9 is recommended.

```bash
# using conda
conda create -n minflux_fibers python=3.11 -y
conda activate minflux_fibers
```

Or with venv:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

### 3. Install dependencies

Install the core dependencies:

```bash
pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    scikit-image \
    tifffile
```

---

## Input data & folder structure

Each **sample** is assumed to have at least:

1. A **raw localization CSV with TraceID CSV** (for 3D grouping and labeling), e.g.:

   ```text
   <base_name>_RawLocs_IDs.csv
   ```

   Required columns:

   * `X`
   * `Y`
   * `Z`
   * `TraceID`

The pipeline is designed to operate on a **folder of samples**:

```text
RAW_DATA_ROOT/
    sample1_RawLocs_IDs.csv
    sample2_RawLocs_IDs.csv
    ...
```

In `config.py`, you will specify something like:

```python
RAW_DATA_ROOT = "/path/to/RAW_DATA_ROOT"
RESULTS_ROOT  = "/path/to/results"
```

The pipeline will:

* iterate over files in `RAW_DATA_ROOT`,
* skip files that do not contain the necessary columns,
* write results into:

```text
RESULTS_ROOT/
    <base_name>/
        # per-sample outputs
```

---

## Quick start

1. **Edit `config.py`**

   Set at least:

   ```python
   RAW_DATA_ROOT = "/path/to/your/raw/csv/folder"
   RESULTS_ROOT  = "/path/to/write/results"
   ```

   and verify / adjust:

   ```python
   PX_PER_UM       = 10.0
   GAUSS_SIGMA_UM  = 0.05
   PAD_UM          = 3 * GAUSS_SIGMA_UM
   TUBENESS_SIGMA_PX = 1.0
   UPSCALE_FACTOR  = 3
   # segmentation parameters…
   # merging parameters…
   # tube / cylinder parameters…
   ```

2. **Run the batch pipeline**

   From the repository root:

   ```bash
   python -m fiber_pipeline.run_folder
   ```

   or:

   ```bash
   python fiber_pipeline/run_folder.py
   ```

3. **Inspect outputs**

   For each sample `<base_name>` the pipeline will create:

   ```text
   RESULTS_ROOT/
       <base_name>/
           <base_name>_FIJI_OUT/
               # intermediate TIFs, masks, skeletons
           # PNGs with overlays and labeled skeletons
           <base_name>_TraceID_avg_with_fiber_grow_id.csv
           <base_name>_RawLocs_IDs_with_fiber_grow_id.csv
           fiber_metrics.csv
   ```

---

## What the pipeline does (high‑level)

For each sample:

1. **Build a 2D density (KDE) image in XY.**
2. **Apply a tubeness filter and segment fibers in 2D.**
3. **Skeletonize the mask and convert it into a graph of segments.**
4. **Merge segments into longer 2D fibers using geometry and adjacency.**
5. **Map those 2D fibers into 3D using MINFLUX localizations (via TraceID averaging).**
6. **Optionally grow fibers in 3D as cylinders, refining membership.**
7. **Export final per‑localization labels and per‑fiber length/width metrics.**

---

## Algorithm details

### 1. KDE image & tubeness filter

**Module:** `kde_tubeness.py`
**Config:** `PX_PER_UM`, `GAUSS_SIGMA_UM`, `PAD_UM`, `TUBENESS_SIGMA_PX`, `UPSCALE_FACTOR`

1. **Bounding box & discretization**

   Compute the XY bounding box from all localizations:

   [
   x_\text{min} = \min(X) - \text{PAD_UM}, \quad x_\text{max} = \max(X) + \text{PAD_UM}
   ]
   [
   y_\text{min} = \min(Y) - \text{PAD_UM}, \quad y_\text{max} = \max(Y) + \text{PAD_UM}
   ]

   Convert to pixel grid using `PX_PER_UM` (pixels per micron):

   ```python
   ix = round((X - xmin_um) * PX_PER_UM)
   iy = height_px - 1 - round((Y - ymin_um) * PX_PER_UM)
   ```

2. **KDE approximation**

   * Create a 2D array `counts[height_px, width_px]`.
   * Increment `counts[iy, ix]` for each localization.
   * Apply a Gaussian blur with sigma (in pixels):

     ```python
     sigma_px = GAUSS_SIGMA_UM * PX_PER_UM
     kde_blur = gaussian_filter(counts, sigma=sigma_px)
     ```

3. **Normalization & tubeness**

   * Normalize the blurred image to [0, 1] then [0, 255].
   * Apply Sato tubeness (`skimage.filters.sato`) at a chosen scale
     to highlight ridge‑like structures.

4. **Upsampling**

   * Upsample the tubeness image by `UPSCALE_FACTOR` (e.g. 3×) for better skeleton accuracy.

---

### 2. 2D fiber segmentation

**Module:** `segment_2d.py`
**Config:** `sigma_min`, `sigma_max`, `n_scales`, `hyst_high_pct`, `hyst_low_ratio`, `min_obj`, `hole_area`, `close_radius`, `bridge`, `max_gap`

Steps:

1. Convert tubeness image to float in [0, 1].

2. Apply TV‑based denoising (`denoise_tv_chambolle`) to preserve edges.

3. Apply CLAHE (`equalize_adapthist`) to boost local contrast.

4. Apply multi‑scale Sato filter on:

   ```python
   sigmas = np.linspace(sigma_min, sigma_max, n_scales)
   ```

5. Use hysteresis thresholding:

   * `high = percentile(vessel, hyst_high_pct)`
   * `low  = hyst_low_ratio * high`

   to obtain a stable binary mask.

6. Morphological cleanup:

   * `remove_small_objects` with `min_obj`,
   * `binary_closing` with a disk of radius `close_radius`,
   * `remove_small_holes` with `hole_area`.

7. **Optional bridging:**

   * Skeletonize the mask,
   * find endpoints,
   * attempt to connect nearby endpoints by routing short paths through high‑vesselness regions.

8. **Skeletonization:**

   * Skeletonize the final mask to get a 1‑pixel‑wide centerline.

---

### 3. Skeleton graph construction

**Module:** `skeleton_graph.py`

1. **Node identification**

   * Compute 8‑connected neighbor counts for each skeleton pixel.
   * Define:

     * **Endpoint (E)**: count == 1,
     * **Junction (J)**: count ≥ 3.

2. **Junction collapsing**

   * Label connected junction pixels.
   * Merge each connected junction region into a single node with:

     * a list of its pixels,
     * a centroid `(cy, cx)`.

3. **Segment extraction**

   For each node:

   * Walk out from the node via “exits” (pixels just outside node and inside the skeleton).
   * Follow the chain of skeleton pixels until encountering another node or a dead end.
   * For each successful path between nodes `(u, v)`, create a **segment** with:

     * `u`, `v`: node IDs,
     * `pixels`: interior path pixels between nodes,
     * `anchor_u`, `anchor_v`: pixels on the node boundary where the path enters/exits.

4. **Deduplication**

   * If multiple segments connect the same pair of nodes, keep the longest chain.

Result: a list of `segments` and a dictionary of `nodes`.

---

### 4. Segment merging → 2D fiber segmentation

**Module:** `merge_segments.py`
**Config:** `ANGLE_THRESH_DEG`, `NEAR_RADIUS_PX`, `MAX_ITERS`

Goal: group segments into longer, smoother fibers while avoiding unwanted branching.

1. **Segment orientation**

   * Each segment’s orientation is defined from the centroids of its endpoint nodes:

     ```python
     theta = orientation_0_180(nodes[u].centroid, nodes[v].centroid)
     ```

   * Orientation is axial (`0–180°`), ignoring direction.

2. **Adjacency**

   Build adjacency in two ways:

   * **Junction adjacency**: segments that share a junction node.
   * **Near‑neighbor adjacency**:

     * endpoint–endpoint within `NEAR_RADIUS_PX`,
     * endpoint–junction centroid within `NEAR_RADIUS_PX`.

3. **Group merging via mutual best neighbors**

   * Start with each segment in its own group.
   * At each iteration:

     * compute a **group orientation** from member segments,
     * for each group, among adjacent groups:

       * compute angular difference,
       * keep those with acute angle difference `< ANGLE_THRESH_DEG`,
       * pick best neighbor as (min angle, then min anchor distance).
     * only merge groups that **mutually** pick each other as best neighbor.
   * Repeat until no more merges or `MAX_ITERS` is reached.

4. **Branch splitting post‑processing**

   * For each junction and each fiber label:

     * gather all incident segments with the same label,
     * if there are ≥ 3, compute **local orientation** at the junction for each segment,
     * find the pair with the smallest angle difference (smoothest continuation),
     * keep that pair in the original label,
     * reassign all other segments to new labels (creating separate fibers).

Result: integer `merged_group_id` for each segment, forming a 2D segmentation of the skeleton into fibers.

---

### 5. 3D tube centerlines from MINFLUX localizations

**Module:** `tubes_3d.py`
**Config:** `TUBE_DIAM_MAX`, `TUBE_RADIUS`, `SEARCH_RADIUS`, `CENTER_STEP`, `MIN_PTS_PER_SAMPLE`, `MAX_XY_SHIFT`

1. **TraceID averaging**

   * Read the `*_RawLocs_IDs.csv` file.

   * Ensure it contains `X, Y, Z, TraceID`.

   * Group by `TraceID`:

     ```python
     df_avg = df_ids.groupby("TraceID", as_index=False).agg(
         X=("X", "mean"),
         Y=("Y", "mean"),
         Z=("Z", "mean"),
         N=("TraceID", "size"),
     )
     ```

   * `df_avg` contains one row per trace.

2. **Convert 2D centerlines to µm**

   * For each segment:

     * concatenate `(anchor_u) + pixels + (anchor_v)`,
     * remove duplicates,
     * convert each pixel `(y, x)` to `(X_um, Y_um)` using the same mapping as in KDE,
     * obtain a polyline in µm.

   * Resample these polylines at fixed arc‑length steps of `CENTER_STEP`:

     ```python
     xy_resampled = resample_polyline(xy, step=CENTER_STEP)
     ```

3. **Assign averaged points to fibers in 2D**

   * Concatenate all resampled centerline samples (with associated fiber IDs).
   * Build a KD‑tree on the XY coordinates of these samples.
   * For each averaged point `(X, Y)`:

     * find the nearest centerline sample,
     * if distance ≤ `TUBE_RADIUS`, assign that fiber ID (`assigned_gid`),
     * else assign -1 (noise/unassigned).

4. **Build 3D centerlines per segment**

   For each segment / fiber:

   * For each resampled centerline point `xy0`:

     * query nearby averaged points within `SEARCH_RADIUS` in XY,
     * keep those that:

       * are assigned to the same fiber label,
       * are within `TUBE_RADIUS` from `xy0`.
     * compute:

       * `z_med` = median of their Z values,
       * `(X_cent, Y_cent)` = mean XY;
       * optionally shift `xy0` towards `(X_cent, Y_cent)` by at most `MAX_XY_SHIFT`.
     * store `(X_adj, Y_adj, z_med)`.

   * Keep fibers with at least `MIN_PTS_PER_SAMPLE` samples.

The result is `seg_xyz_3d` mapping segment IDs to 3D centerline samples.

---

### 6. Cylinder growth from initial segmentation

**Module:** `grow_cylinders.py`
**Config:** `END_SLICE_LEN`, `EXTEND_STEP`, `MAX_RADIUS`, `MIN_PTS_FIBER`, `MIN_PTS_END`, `MAX_GLOBAL_ITERS`

Goal: refine fiber assignment by growing 3D cylinders from the initial segmentation.

1. **Prepare data**

   * Use `df_avg` from the previous step.
   * Start from `assigned_gid` labels.
   * Keep only points with `assigned_gid ≥ 0`.

2. **Initial fiber ranking**

   * For each fiber ID:

     * extract its 3D points,
     * compute principal axis (PC1) and its length (max‑min along PC1),
     * store as “fiber length”.
   * Sort fibers from longest to shortest.
   * Longer fibers are allowed to **steal** points from shorter fibers during growth.

3. **End‑slice fitting & growth**

   For each fiber:

   * compute global PC1 (`v_global`) and center (`mu_f`),
   * project points to PC1, get `t_f` (coordinates along fiber).

   For each end (min and max of `t_f`):

   * select points within `END_SLICE_LEN` from the end,
   * fit a local PC1 (`v_end`) and mean (`mu_end`),
   * orient `v_end` so it points outward from the fiber body (align with or opposite `v_global`),
   * compute radial distances of points to this local axis, estimate radius `r_end` (capped by `MAX_RADIUS`),
   * define a small extension interval `(t_start, t_stop)` of length `EXTEND_STEP` along `v_out`,
   * form a bounding sphere around the segment center,
   * query candidate points in this sphere,
   * for each candidate:

     * project onto `v_out`, check if `t` lies in `(t_start, t_stop]`,
     * compute radial distance to axis, check if ≤ `r_end`,
     * if this candidate is currently assigned to a shorter fiber (lower rank), reassign it to the current fiber.

4. **Iteration**

   * Repeat passes over all fibers for up to `MAX_GLOBAL_ITERS` or until no labels change.
   * Final labels are stored as `fiber_grow_id` in `df_avg` and merged back into `df_ids`.

This step is **optional** in principle but is currently integrated as the final refinement stage.

---

## Outputs

For each sample `<base_name>`, the pipeline writes a folder:

```text
RESULTS_ROOT/
    <base_name>/
        <base_name>_FIJI_OUT/
            <base_name>_counts_32f.tif
            <base_name>_kde_blur32f_sigma{GAUSS_SIGMA_UM:.4f}um_counts.tif
            <base_name>_kde_gray8.tif
            <base_name>_tubeness32f_sigma1um.tif
            <base_name>_tubeness8bit.tif
            <base_name>_tubeness8bit_scale{UPSCALE_FACTOR}.tif
            <base_name>_fiber_mask_scale{UPSCALE_FACTOR}.png
            <base_name>_fiber_skeleton_scale{UPSCALE_FACTOR}.png
            <base_name>_fiber_overlay_scale{UPSCALE_FACTOR}.png
        skeleton_density_overlay_fig.png
        skeleton_density_overlay.png
        skeleton_density_junctions.png
        segments_merged_adjacent_mutual_unique.png
        segments_merged_near_neighbor.png
        segments_final_branch_split.png
        3D_tubes_avg_points.png
        3D_cylinder_grown_fibers.png
        <base_name>_TraceID_avg_with_fiber_grow_id.csv
        <base_name>_RawLocs_IDs_with_fiber_grow_id.csv
        fiber_metrics.csv
```

### Key CSVs

#### `<base_name>_TraceID_avg_with_fiber_grow_id.csv`

* One row per `TraceID` (averaged localization).
* Contains at least:

  * `TraceID`
  * `X`, `Y`, `Z` (mean positions in µm)
  * `N` (number of raw localizations)
  * `fiber_grow_id` (final fiber label, or -1 for unassigned/noise).

#### `<base_name>_RawLocs_IDs_with_fiber_grow_id.csv`

* Raw localization table, with original per‑localization rows and columns:

  * `X`, `Y`, `Z`
  * `TraceID`
  * `fiber_grow_id` propagated from the averaged table.

#### `fiber_metrics.csv`

* One row per **final fiber** (unique `fiber_grow_id >= 0`).
* Columns (canonical):

  * `fiber_id`
    The integer label (same as `fiber_grow_id`).

  * `fiber_length_um`
    Estimated fiber length in micrometers (from 3D centerline).

  * `fiber_width_um`
    Estimated fiber width (effective diameter) in micrometers, derived from the radial distribution of points around the centerline or cylinder axis.

---

## Hyperparameters & tuning guide

All hyperparameters are centralized in `config.py`.
Below is a conceptual guide to what matters and how to tune it.

### 1. Geometry & KDE

* **`PX_PER_UM`**

  * Pixels per micron in the base KDE grid (e.g. 10 → 0.1 µm pixel).
  * Larger:

    * finer grid, better spatial detail,
    * more memory & compute.
  * Smaller:

    * more smoothing,
    * risk of merging nearby fibers in the density representation.

* **`GAUSS_SIGMA_UM`**

  * Gaussian sigma (µm) used to blur the point counts.
  * Larger:

    * smoother density, fills small gaps,
    * merges closely spaced structures.
  * Smaller:

    * preserves high‑frequency structure, but more noise.

* **`PAD_UM`**

  * Extra padding around the XY bounding box.
  * Usually 2–4× `GAUSS_SIGMA_UM` to avoid edge effects.

### 2. Tubeness & 2D segmentation

* **`TUBENESS_SIGMA_PX`**

  * Characteristic scale for the Sato tubeness filter.
  * Roughly match the apparent radius of fibers in pixels after KDE.

* **`UPSCALE_FACTOR`**

  * Upsampling factor for the tubeness image before segmentation (e.g. 3).
  * Larger:

    * more precise skeleton,
    * heavier processing.

* **`sigma_min`, `sigma_max`, `n_scales`**

  * Range and number of sigmas for multi‑scale ridge detection.

* **`hyst_high_pct`**

  * High threshold percentile for vesselness (e.g. 90).
  * Increase to keep only very confident ridges.

* **`hyst_low_ratio`**

  * Low threshold as a fraction of the high threshold.
  * Decrease to maintain connectivity of weaker segments.

* **`min_obj`**

  * Minimum object size (pixels) to keep after thresholding.
  * Increase to suppress tiny noise blobs.

* **`hole_area`**

  * Maximum hole area to fill.
  * Prevents small holes inside fiber blobs.

* **`close_radius`**

  * Radius for morphological closing.
  * Smooths boundaries and helps connect nearly touching components.

* **`bridge`, `max_gap`**

  * Enabling gap bridging tries to connect endpoints whose shortest path goes through
    high vesselness.
  * `max_gap` sets the maximum allowable length of the gap.

### 3. 2D segment merging

* **`ANGLE_THRESH_DEG`**

  * Max acute angle difference allowed when merging segments/groups.
  * Typical range: 20–40°.
  * Smaller:

    * more conservative, fewer merges,
    * less risk of merging diverging fibers.
  * Larger:

    * more aggressive merging, can over‑merge.

* **`NEAR_RADIUS_PX`**

  * Pixel radius for endpoint adjacency.
  * Too small:

    * segments that should connect might never be considered neighbors.
  * Too large:

    * unrelated fibers may be considered neighbors, increasing risk of wrong merges.

* **`MAX_ITERS`**

  * Safety cap on number of merge iterations.
  * Usually a small number (5–10) is sufficient.

### 4. 3D tube assignment

* **`TUBE_DIAM_MAX`** / **`TUBE_RADIUS`**

  * XY radius for initial assignment of averaged points to fibers.
  * Ideally close to physical tube radius; adjust based on MINFLUX PSF and labeling.

* **`SEARCH_RADIUS`**

  * XY neighborhood radius around each centerline sample to gather points for Z estimation.
  * Larger:

    * more robust Z estimate but more mixing if fibers are close.

* **`CENTER_STEP`**

  * Step size (µm) along centerline for resampling.
  * Smaller:

    * finer sampling, more detail,
    * more computational cost.

* **`MIN_PTS_PER_SAMPLE`**

  * Minimum points required to set a Z value for a centerline sample.

* **`MAX_XY_SHIFT`**

  * Upper bound on how far the centerline can be shifted toward the local point centroid.
  * Prevents centerlines from drifting away from the 2D skeleton.

### 5. Cylinder growth

* **`END_SLICE_LEN`**

  * Length of the end slice used to fit local PC1 at each fiber end.
  * Should be long enough to capture local direction but short enough to reflect end geometry.

* **`EXTEND_STEP`**

  * Step size in µm for each cylinder growth iteration along PC1.

* **`MAX_RADIUS`**

  * Maximum cylinder radius used during growth.
  * Typically close to or slightly larger than `TUBE_RADIUS`.

* **`MIN_PTS_FIBER`**

  * Minimum number of points for a fiber to be considered for growth.

* **`MIN_PTS_END`**

  * Minimum number of points in an end slice to fit local geometry reliably.

* **`MAX_GLOBAL_ITERS`**

  * Maximum number of global passes over fibers during growth.
  * The process usually converges earlier.

---

## Fiber length & width estimation

**Module:** `metrics.py`

This module consumes:

* final per‑point fiber labels (`fiber_grow_id`),
* per‑TraceID averaged coordinates (`df_avg`),
* optional 3D centerline samples (if available),

and outputs **`fiber_metrics.csv`** with columns:

* `fiber_id`
  Final fiber label (integer, same as `fiber_grow_id`).

* `fiber_length_um`
  Estimated fiber length in micrometers, typically computed as:

  * the arc length of a smoothed 3D centerline for that fiber, or
  * a fallback (e.g. 2D length) when 3D samples are insufficient.

* `fiber_width_um`
  Effective fiber width, approximated from radial distances of points to the fiber’s main axis or centerline. A typical rule:

  ```text
  fiber_width_um ≈ 2 × radial_scale
  ```

  where `radial_scale` might be a robust statistic such as:

  * the median radial distance, or
  * the 95th percentile of radial distances.

This width is a **statistical measure**, not a direct physical measurement of filament diameter, but is useful for comparing relative thickness across fibers and conditions.

---

## Tips, troubleshooting, and limitations

### 1. Nothing segments / everything looks empty

* Check **KDE parameters**:

  * `PX_PER_UM` too small or `GAUSS_SIGMA_UM` too large can oversmooth.
* Loosen segmentation thresholds:

  * decrease `hyst_high_pct`,
  * increase `hyst_low_ratio`,
  * reduce `min_obj`.
* Inspect intermediate images in `<base_name>_FIJI_OUT` to see where signal is lost.

### 2. Too many tiny fragments / over‑segmentation

* Increase `min_obj` (remove tiny objects).
* Increase `hole_area` (fill small gaps).
* If many long, nearly collinear segments remain separated:

  * increase `ANGLE_THRESH_DEG` slightly,
  * increase `NEAR_RADIUS_PX` to allow more adjacency candidates.

### 3. Fibers merge through crossings or branches

* Decrease `ANGLE_THRESH_DEG` so only nearly collinear segments merge.
* Decrease `NEAR_RADIUS_PX` to reduce accidental adjacency.
* Inspect:

  * `segments_merged_adjacent_mutual_unique.png`,
  * `segments_final_branch_split.png`,
    to ensure branch splitting at junctions behaves as expected.

### 4. 3D tubes look noisy or broken

* Increase `TUBE_RADIUS` to capture more points around centerlines.
* Increase `SEARCH_RADIUS` for more points per sample.
* Make sure `TraceID` grouping is correct:

  * if `TraceID` is missing or inconsistent, the 3D step will be noisy.

### 5. Cylinder growth steals points incorrectly

* Decrease `MAX_RADIUS` (tighter cylinder).
* Decrease `EXTEND_STEP` to grow more slowly.
* Increase `MIN_PTS_END` so growth only happens when there is enough evidence at the end.

### 6. Limitations

* **2D first:** the segmentation is driven primarily by XY geometry. Distinct fibers that overlap in projection but separate in Z can still be ambiguous.
* Fibers must be relatively thin and well approximated by local cylinders.
* Cylinder growth uses heuristics (e.g., longer fibers have priority) and may not always reflect the true physical connectivity in very dense networks.

---

## License & citation

Please see the repository’s `LICENSE` file for official licensing terms.

If you use this segmentation pipeline in a publication, please consider citing:

```text
@misc{minflux_fiber_pipeline,
  author       = {Adib Keikhosravi},
  title        = {MINFLUX Fiber Segmentation Pipeline},
  howpublished = {\url{https://github.com/CBIIT/MinFlux_fiber_pipeline}},
  year         = {2025}
}
```

---

