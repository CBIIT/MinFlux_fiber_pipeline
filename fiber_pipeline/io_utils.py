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
    """
    Load a localization CSV and try to find columns X, Y, Z, TraceID.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing at least columns X, Y, Z, TraceID.
        All four columns are converted to numeric (rows with NaN are dropped).
    ok : bool
        True if all four columns were found and valid; False otherwise.
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path, comment="#")
    except Exception:
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", comment="#")
        except Exception:
            return pd.DataFrame(), False

    required = ["X", "Y", "Z", "TraceID"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame(), False

    df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return df, False
    return df, True
