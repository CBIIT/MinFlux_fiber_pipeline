"""
run_pipeline.py
===============

Batch driver for the full MINFLUX fiber pipeline.

Usage
-----
    python run_pipeline.py /path/to/localization_folder

If no command‑line argument is given, `config.PIPELINE_CONFIG.input_dir`
is used. Results are written to:

    <results_root>/<sample_base_name>/

where `results_root` is either `config.PIPELINE_CONFIG.results_root`
(if not None) or `<input_dir>/results`.

For each CSV in the input folder that contains at least X, Y, TraceID
(and optionally Z), the following outputs are created inside the sample folder:

    * FIJI‑style TIFFs / PNGs (in <base>_FIJI_OUT/)
    * skeleton + density overlays
    * segment overlays
If the input is **3D** (X,Y,Z,TraceID), the full pipeline runs and additionally
produces:

    * 3D tubes plot
    * 3D cylinder‑grown fibers plot
    * <base>_TraceID_avg_with_fiber_grow_id.csv
    * <base>_localizations_with_fiber_grow_id.csv
    * fiber_metrics.csv  (columns: fiber_id, fiber_length_um, fiber_width_um)

If the input is **2D** (X,Y,TraceID only), Z is synthesized as 0.0 for
compatibility but the 3D‑specific stages are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

from config import PIPELINE_CONFIG as cfg
from io_utils import (
    base_name_no_ext,
    ensure_dir,
    load_localization_table,
)
from kde_tubeness import run_kde_and_tubeness
from fiber_segmentation import run_fiber_segmentation
from overlays import make_density_overlay, make_junction_overlay
from skeleton_graph import extract_and_merge_segments
# NOTE: 3D-specific stages (tubes3d + cylinder_growth) are imported lazily
# inside the 3D branch so 2D-only runs don't require their dependencies.


def run_single_sample(csv_path: Path,
                      results_root: Path) -> None:
    """
    Run the full pipeline on a single localization CSV.

    Parameters
    ----------
    csv_path : Path
        Path to one CSV file.
        Accepted schemas:
            * 3D: X,Y,Z,TraceID
            * 2D: X,Y,TraceID
    results_root : Path
        Parent directory for all sample‑specific results.
    """
    base = base_name_no_ext(csv_path)
    print(f"\n========== Processing sample: {base} ==========")

    df, ok = load_localization_table(csv_path)
    if not ok:
        print(f"[SKIP] {csv_path} does not contain usable X,Y,(Z),TraceID.")
        return

    is_3d = bool(df.attrs.get("is_3d", True))

    if not is_3d:
        print(f"[INFO] {base}: input appears 2D (no Z column). "
              "A Z=0 column will be used; 3D stages will be skipped.")

    sample_dir = ensure_dir(results_root / base)
    fiji_dir = ensure_dir(sample_dir / f"{base}_FIJI_OUT")

    # Stage 1: KDE + tubeness
    tubeness_up, kde_info = run_kde_and_tubeness(csv_path, df, fiji_dir)

    # Stage 2: segmentation -> mask + skeleton
    skeleton_path = run_fiber_segmentation(
        tubeness_up=tubeness_up,
        csv_path=csv_path,
        out_dir=fiji_dir,
        results_dir=sample_dir,
    )

    # Overlays (2D)
    make_density_overlay(df, skeleton_path, kde_info, sample_dir)
    make_junction_overlay(df, skeleton_path, kde_info, sample_dir)

    # Skeleton -> segments -> merged groups
    segments, merged_group_id, nodes, H, W = extract_and_merge_segments(
        skeleton_path,
        sample_dir,
    )

    if is_3d:
        # Lazy imports so 2D-only runs don't pull in 3D plotting modules.
        from tubes3d import build_3d_tubes
        from cylinder_growth import run_cylinder_growth

        # 3D tubes from TraceID averages
        df_avg, assigned_gid, seg_xyz_3d = build_3d_tubes(
            df_ids=df,
            segments=segments,
            merged_group_id=merged_group_id,
            info=kde_info,
            base_name=base,
            results_dir=sample_dir,
        )

        # Cylinder growth + metrics; propagate labels back to df and df_avg
        df_avg_out, df_ids_out, fiber_metrics = run_cylinder_growth(
            df_ids=df,
            df_avg=df_avg,
            assigned_gid=assigned_gid,
            base_name=base,
            results_dir=sample_dir,
        )

        # Save labelled CSVs
        df_avg_out.to_csv(
            sample_dir / f"{base}_TraceID_avg_with_fiber_grow_id.csv",
            index=False
        )
        df_ids_out.to_csv(
            sample_dir / f"{base}_localizations_with_fiber_grow_id.csv",
            index=False
        )
    else:
        print(f"[INFO] {base}: skipping 3D tube building and cylinder growth "
              "(2D input).")

    print(f"[DONE] Sample '{base}' completed. "
          f"Results in: {sample_dir}")


def main():
    # Determine input directory
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1]).resolve()
    else:
        input_dir = Path(cfg.input_dir).resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Determine results root
    if cfg.results_root is not None:
        results_root = Path(cfg.results_root).resolve()
    else:
        results_root = input_dir / "results"
    ensure_dir(results_root)

    print(f"[MAIN] Input directory : {input_dir}")
    print(f"[MAIN] Results root    : {results_root}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print("[MAIN] No CSV files found.")
        return

    for csv_path in csv_files:
        run_single_sample(csv_path, results_root)


if __name__ == "__main__":
    main()
