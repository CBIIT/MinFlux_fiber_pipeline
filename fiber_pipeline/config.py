"""
config.py
=========

Central location for all configuration options and hyper‑parameters
used by the MINFLUX fiber analysis pipeline.

Edit the values in `PIPELINE_CONFIG` to tune the behaviour of
the pipeline. All other modules import this object.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """
    Global configuration and hyper‑parameters for the pipeline.

    Parameters are grouped by stage (KDE, segmentation, graph, 3D tubes,
    cylinder growth, I/O). Distances are in µm unless noted.
    """

    # -------------------- I/O --------------------
    # Folder containing localization CSVs (each must have X,Y,Z,TraceID).
    input_dir: str = "/data4/Raw localizations & confocal/3D/localization_csvs"

    # Parent directory where results are stored.
    # If None, defaults to "<input_dir>/results".
    results_root: Optional[str] = None

    # -------------------- KDE / tubeness --------------------
    px_per_um: float = 10.0            # pixels per micron
    gauss_sigma_um: float = 0.05       # KDE Gaussian sigma in µm
    pad_factor: float = 3.0            # padding in units of sigma
    tubeness_sigma_px: float = 1.0     # Sato filter sigma in pixels
    upsample_factor: int = 3           # tubeness image upsampling

    # -------------------- segmentation -----------------------
    seg_sigma_min: float = 1.0
    seg_sigma_max: float = 3.5
    seg_n_scales: int = 6
    seg_hyst_high_pct: float = 90.0
    seg_hyst_low_ratio: float = 0.45
    seg_min_obj: int = 24
    seg_hole_area: int = 36
    seg_close_radius: int = 1
    seg_bridge_gaps: bool = True
    seg_max_gap_px: int = 10

    # -------------------- density overlays -------------------
    gauss_sigma_px_density: float = 2.0

    # -------------------- skeleton merging -------------------
    angle_thresh_deg: float = 30.0
    near_radius_px: float = 4.0
    max_merge_iters: int = 500
    palette_seed: int = 7
    print_counts: bool = True

    # -------------------- 3D tubes (TraceID‑averaged) --------
    tube_diam_max: float = 0.2            # max tube diameter (µm)
    search_radius: float = 0.1            # XY search radius around segment centerline
    center_step: float = 0.05             # sampling step along segments (µm)
    min_pts_per_sample: int = 20
    max_xy_shift: float = 0.20            # clamp XY shift (µm)

    line_alpha: float = 0.80
    line_width: float = 2.0
    pt_size: float = 6.0
    pt_alpha: float = 0.65

    # -------------------- cylinder growth --------------------
    end_slice_len: float = 0.25
    extend_step: float = 0.30
    max_radius: float = 0.15
    min_pts_fiber: int = 30
    min_pts_end: int = 10
    max_global_iters: int = 200

    pt_size_cyl: float = 4.0
    pt_alpha_cyl: float = 0.60
    noise_alpha_cyl: float = 0.25

    # Derived parameters (filled in __post_init__)
    pad_um: float = field(init=False)
    tube_radius: float = field(init=False)

    def __post_init__(self) -> None:
        self.pad_um = self.pad_factor * self.gauss_sigma_um
        self.tube_radius = self.tube_diam_max / 2.0


# Single global instance imported everywhere
PIPELINE_CONFIG = PipelineConfig()
