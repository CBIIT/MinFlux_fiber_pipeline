"""
io_utils.py
===========

Small I/O helpers shared across the pipeline:

* base_name_no_ext(path) - strip all extensions from a filename
* ensure_dir(path)       - create folder if needed
* load_localization_table(csv_path) - read X,Y,Z,TraceID from CSV
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd


def base_name_no_ext(path: str | Path) -> str:
    """
    Return just the base filename of *path* with **all** extensions removed.

    This mimics Fiji's baseNameNoExtFromPath behaviour so that names
    like "file.tif.zip" become "file".
    """
    name = os.path.basename(str(path))
    while True:
        root, ext = os.path.splitext(name)
        if not ext:
            break
        name = root
    return name


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure that directory *path* exists and return it as a Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_localization_table(csv_path: str | Path) -> Tuple[pd.DataFrame, bool]:
    """Load a localization CSV and extract X/Y/(Z)/TraceID.

    This loader supports both 3D and 2D localization tables:

    * **3D**: columns ["X", "Y", "Z", "TraceID"] are present.
    * **2D**: columns ["X", "Y", "TraceID"] are present ("Z" missing).
      In this case a "Z" column is added and filled with 0.0 so the
      downstream code can keep using the canonical ["X", "Y", "Z", "TraceID"]
      schema.

    Column matching is case-insensitive and tolerant to spaces/underscores,
    so e.g. "trace_id" or "Trace ID" are accepted.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned DataFrame with **canonical** columns ["X", "Y", "Z", "TraceID"].
        Selected columns are converted to numeric and rows with NaN are dropped.
    ok : bool
        True if X, Y, and TraceID were found and at least one valid row exists.

    Notes
    -----
    The returned DataFrame always includes a "Z" column.

    To let the rest of the pipeline know whether the input was truly 3D, this
    function sets:

        df.attrs["is_3d"] = True/False

    where False means "Z was missing and has been synthesized as 0.0".
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path, comment="#")
    except Exception:
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", comment="#")
        except Exception:
            return pd.DataFrame(), False

    # ---- robust column detection (case-insensitive; ignore spaces/underscores) ----
    def _norm_col(c: str) -> str:
        return (
            str(c)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

    norm_to_orig = {}
    for c in df.columns:
        norm_to_orig[_norm_col(c)] = c

    x_col = norm_to_orig.get("x")
    y_col = norm_to_orig.get("y")
    z_col = norm_to_orig.get("z")
    tid_col = norm_to_orig.get("traceid")

    if x_col is None or y_col is None or tid_col is None:
        return pd.DataFrame(), False

    is_3d = z_col is not None

    if is_3d:
        sel = df[[x_col, y_col, z_col, tid_col]].rename(
            columns={x_col: "X", y_col: "Y", z_col: "Z", tid_col: "TraceID"}
        )
        sel = sel.apply(pd.to_numeric, errors="coerce").dropna()
    else:
        sel = df[[x_col, y_col, tid_col]].rename(
            columns={x_col: "X", y_col: "Y", tid_col: "TraceID"}
        )
        sel = sel.apply(pd.to_numeric, errors="coerce").dropna()
        if not sel.empty:
            sel["Z"] = 0.0

    if sel.empty:
        return sel, False

    # Ensure canonical column order
    sel = sel[["X", "Y", "Z", "TraceID"]]
    sel.attrs["is_3d"] = bool(is_3d)
    return sel, True
