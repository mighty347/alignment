"""Brownfield Survey Ingestion Module for Indian Railways ESP Generator.

Module: modules/brownfield/survey_reader.py
Purpose: Ingests, standardizes, validates, and resamples field survey data for
         Brownfield railway projects (Doubling, 3rd/4th line, Gauge Conversion).

Engineering Standards (RDSO / Indian Railways):
    - Existing Rail Level (ERL): Top of Rail (TOR) elevation from field fly levelling / Total Station / LiDAR.
    - Existing Formation Level (EFL): Subgrade elevation. If not surveyed, derived via:
        EFL = ERL - track_buildup_m
      where track_buildup_m = 0.762m for BG Main Line (350mm ballast + 240mm PSC sleeper + 172mm 60kg rail).
    - Track Center Distance (C/C): Distance from existing track centerline to proposed track.
      IRS Schedule of Dimensions (SOD 2004 BG):
        - Minimum permissible track center: 4.265 m (Schedule I, Chapter I, Para 1)
        - Standard track center for new works: 5.300 m (IRS SOD 2004)
    - Gradient Quality Check: Flags surveyor level anomalies where |dZ / dS| exceeds
      the maximum plausible gradient threshold (default: 1 in 50 / 0.02).

Outputs:
    SurveyResult containing:
        - raw_df: Cleaned survey points.
        - interpolated_df: Uniformly sampled points at project chainage interval (e.g. 20m).
        - assets_df: Landmark features (Bridges, Level Crossings, FOB/ROB, Platforms).
        - metadata: Quality audit, SOD violations, elevation bounds.
"""

from __future__ import annotations

import io
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point

try:
    from alignment.utils.logger import get_logger
except ImportError:
    try:
        from utils.logger import get_logger
    except ImportError:
        import logging
        def get_logger(name: str):
            return logging.getLogger(name)

logger = get_logger("BrownfieldSurveyReader")

# RDSO Broad Gauge Engineering Standards (IRS SOD 2004 BG)
DEFAULT_STANDARD_TRACK_CENTER_M: float = 5.300
DEFAULT_MIN_TRACK_CENTER_M: float = 4.265
DEFAULT_TRACK_BUILDUP_M: float = 0.762
DEFAULT_BALLAST_LIFT_M: float = 0.025
DEFAULT_CHAINAGE_INTERVAL_M: float = 20.0
DEFAULT_MAX_PLAUSIBLE_GRADIENT: float = 0.02


class SurveyReaderError(Exception):
    """Custom exception raised for brownfield survey ingestion errors."""


@dataclass
class SurveyResult:
    """Container for brownfield survey ingestion outputs.

    Attributes:
        raw_df: Cleaned survey points from input file.
        interpolated_df: Uniformly resampled dataframe along continuous chainage (e.g., every 20m).
        assets_df: Key control points and structures identified along the survey.
        metadata: Audit dictionary with quality flags, elevation bounds, and SOD warnings.
    """

    raw_df: pd.DataFrame
    interpolated_df: pd.DataFrame
    assets_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def survey_df(self) -> pd.DataFrame:
        """Convenience accessor for resampled/cleaned survey levels."""
        return self.interpolated_df if not self.interpolated_df.empty else self.raw_df


class BrownfieldSurveyReader:
    """Ingests and validates field survey data for brownfield railway corridors."""

    def __init__(self, config: Optional[dict[str, Any] | str | Path] = None) -> None:
        """Initialize BrownfieldSurveyReader with parameters from config.yaml.

        Args:
            config: Master configuration dictionary or path to config_brownfield.yaml. If None,
                    defaults to loading config/config_brownfield.yaml.
        """
        self.logger = get_logger(self.__class__.__name__)

        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, (str, Path)):
            cfg_path = Path(config)
            if cfg_path.exists():
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self.config = {}
        else:
            self.config = {}

        self.logger.info("Initializing %s", self.__class__.__name__)

        # Support both dedicated config_brownfield.yaml schema and legacy embedded config
        horiz_cfg = self.config.get("horizontal", {})
        vert_cfg = self.config.get("vertical", {})
        paths_cfg = self.config.get("paths", {})
        bf_legacy = self.config.get("brownfield", {})

        self.standard_track_center_m: float = float(
            horiz_cfg.get("standard_track_center_m", bf_legacy.get("standard_track_center_m", DEFAULT_STANDARD_TRACK_CENTER_M))
        )
        self.min_track_center_m: float = float(
            horiz_cfg.get("min_track_center_m", bf_legacy.get("min_track_center_m", DEFAULT_MIN_TRACK_CENTER_M))
        )
        self.chainage_interval_m: float = float(
            horiz_cfg.get(
                "chainage_interval_m",
                bf_legacy.get(
                    "chainage_interval_m",
                    self.config.get("chainage", {}).get("interval_m", DEFAULT_CHAINAGE_INTERVAL_M),
                ),
            )
        )
        self.track_buildup_m: float = float(
            vert_cfg.get(
                "track_buildup_m",
                bf_legacy.get(
                    "track_buildup_m",
                    self.config.get("validation", {}).get("track_buildup_m", DEFAULT_TRACK_BUILDUP_M),
                ),
            )
        )
        self.default_ballast_lift_m: float = float(
            vert_cfg.get("default_ballast_lift_m", bf_legacy.get("default_ballast_lift_m", DEFAULT_BALLAST_LIFT_M))
        )
        self.max_plausible_gradient: float = float(
            vert_cfg.get("max_plausible_gradient", bf_legacy.get("max_plausible_gradient", DEFAULT_MAX_PLAUSIBLE_GRADIENT))
        )
        self.survey_data_path: str = str(
            paths_cfg.get("survey_data_path", bf_legacy.get("survey_data_path", ""))
        )
        self.survey_sheet_name: Optional[str] = paths_cfg.get(
            "survey_sheet_name", bf_legacy.get("survey_sheet_name")
        )

        self.logger.info(
            "Configured Brownfield: Std Track Center=%.2fm, Min SOD=%.3fm, Buildup=%.3fm, Interval=%.1fm, Sheet=%s",
            self.standard_track_center_m,
            self.min_track_center_m,
            self.track_buildup_m,
            self.chainage_interval_m,
            self.survey_sheet_name or "Default",
        )

    def _find_header_row_and_load_excel(self, file_path: Path, sheet_name: Any) -> pd.DataFrame:
        """Scan top rows of Excel sheet to dynamically detect true column header row across all railway zones.

        Handles:
            - Files with title banners in row 0 (e.g. Northern Railway USBRL FLS sheets).
            - Files with metadata blocks in rows 0-2.
            - Standard files with headers directly in row 0 (e.g. South Central Railway sheets).
        """
        preview = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=10)
        keywords = [
            "chainage", "ch", "rail", "erl", "rl", "formation", "fl",
            "ground", "gl", "track", "cut", "fill", "grade", "alignment", "feature"
        ]
        best_header = 0
        best_score = 0
        for r_idx in range(len(preview)):
            row_vals = [str(v).lower() for v in preview.iloc[r_idx].dropna()]
            score = sum(1 for val in row_vals if any(kw in val for kw in keywords))
            if score > best_score:
                best_score = score
                best_header = r_idx

        self.logger.info(
            "Auto-detected Excel header at row %d (score=%d) in sheet '%s'",
            best_header,
            best_score,
            sheet_name,
        )
        return pd.read_excel(file_path, sheet_name=sheet_name, header=best_header)

    def _standardize_columns(
        self,
        df: pd.DataFrame,
        alignment: Optional[Any] = None,
        max_corridor_offset_m: float = 50.0,
        allow_missing_chainage: bool = False,
    ) -> pd.DataFrame:
        """Map heterogeneous survey column names across all Indian Railway zones to standard internal names.

        Supports column conventions from SCR, NR, WR, CR, ER, SR, ECoR, NCR, etc.
        Also supports projecting spatial coordinate point clouds (Easting, Northing, Rail Level)
        onto an alignment LineString if explicit chainage is not present.
        """
        import re

        col_map: dict[str, str] = {}
        for c in df.columns:
            c_str = str(c).strip()
            # Clean string for semantic matching
            c_clean = c_str.lower().replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")
            c_clean = re.sub(r"\(.*?\)", "", c_clean).strip("_")

            # 1. Difference between rail levels (Check early to avoid colliding with rail_level)
            # Handles "diff", "dif", "diff.", "dif.", "dif_between...", etc.
            if ("diff" in c_clean or "dif_" in c_clean or c_clean.startswith("dif_") or "dif" in c_clean.split("_")) and any(k in c_clean for k in ("rail", "rl", "level")):
                col_map[c] = "diff_prl_erl_m"

            # 2. Chainage mapping
            elif any(k in c_clean for k in ("chainage", "ch_km", "chainage_km", "chainage_m", "stn_m")) or c_clean in ("ch", "km", "station", "stn"):
                if "chainage_raw" not in col_map.values():
                    col_map[c] = "chainage_raw"

            # 3. Existing Rail Level (ERL / TOR)
            elif ("exist" in c_clean or "exg" in c_clean or "ex_" in c_clean) and ("rail" in c_clean or "rl" in c_clean):
                col_map[c] = "erl_m"
            elif c_clean in ("erl", "erl_m", "tor", "top_of_rail", "elevation") and "prop" not in c_clean:
                col_map[c] = "erl_m"
            elif ("rail_level" in c_clean or c_clean == "rl") and "prop" not in c_clean and "new" not in c_clean:
                if "erl_m" not in col_map.values():
                    col_map[c] = "erl_m"

            # 4. Proposed Rail Level (PRL)
            elif ("prop" in c_clean or "new" in c_clean) and ("rail" in c_clean or "rl" in c_clean):
                col_map[c] = "prl_m"
            elif c_clean in ("prl", "prl_m"):
                col_map[c] = "prl_m"

            # 5. Existing Formation Level (EFL)
            elif ("exist" in c_clean or "exg" in c_clean or "ex_" in c_clean) and ("form" in c_clean or "fl" in c_clean):
                col_map[c] = "efl_m"
            elif c_clean in ("efl", "efl_m", "ex_formation") and "prop" not in c_clean:
                col_map[c] = "efl_m"

            # 6. Proposed Formation Level (PFL)
            elif ("prop" in c_clean or "new" in c_clean) and ("form" in c_clean or "fl" in c_clean):
                col_map[c] = "pfl_m"
            elif c_clean in ("pfl", "pfl_m"):
                col_map[c] = "pfl_m"

            # 7. Ground Level (EGL / GL / OGL / NSL)
            elif any(k in c_clean for k in ("ground", "ogl", "egl", "nsl")) or c_clean in ("gl", "gl_m"):
                col_map[c] = "egl_m"

            # 8. Track Center Distance (C/C)
            elif any(k in c_clean for k in ("track_center", "track_centre", "cc_distance", "c_c", "centre_to_centre", "center_to_center")) or c_clean == "track_center":
                col_map[c] = "track_center_m"

            # 9. Cut & Fill / Earthwork
            elif any(k in c_clean for k in ("cut", "fill", "bank")) and "form" not in c_clean:
                col_map[c] = "cut_fill_m"

            # 10. Coordinates for Plan View (Easting / Northing)
            elif c_clean in ("easting", "east", "x", "easting_m"):
                col_map[c] = "easting"
            elif c_clean in ("northing", "north", "y", "northing_m"):
                col_map[c] = "northing"

            # 11. Feature / Structure / Remarks
            elif any(k in c_clean for k in ("feature", "structure", "asset", "asset_type", "remarks", "description", "landmark")):
                col_map[c] = "feature"

            # 12. Proposed Alignment / Curvature
            elif "align" in c_clean or "curve" in c_clean:
                col_map[c] = "proposed_alignment"

            # 13. Gradients
            elif "grad" in c_clean or "grade" in c_clean:
                if "ex" in c_clean:
                    col_map[c] = "existing_gradient"
                else:
                    col_map[c] = "proposed_gradient"

            # 14. Land Boundary Available
            elif any(k in c_clean for k in ("land", "boundary", "row")):
                col_map[c] = "land_boundary_m"

        df_renamed = df.rename(columns=col_map).copy()
        # Retain original column headers so custom bands referencing exact Excel column names match cleanly
        for orig_col in df.columns:
            if orig_col not in df_renamed.columns:
                df_renamed[orig_col] = df[orig_col]

        # Handle chainage conversion and unit auto-detection
        if "chainage_raw" in df_renamed.columns:
            raw_orig_col = [c for c, mapped in col_map.items() if mapped == "chainage_raw"][0]
            c_orig_lower = str(raw_orig_col).strip().lower()
            ch_series = pd.to_numeric(df_renamed["chainage_raw"], errors="coerce").dropna()

            # Check for explicit meter indicators in header
            if "_m" in c_orig_lower or "meter" in c_orig_lower or "mtr" in c_orig_lower:
                df_renamed["chainage_m"] = pd.to_numeric(df_renamed["chainage_raw"], errors="coerce")
            elif not ch_series.empty:
                max_abs = float(ch_series.abs().max())
                diffs = ch_series.diff().abs().dropna()
                median_diff = float(diffs[diffs > 0].median()) if not diffs[diffs > 0].empty else 1.0

                # If median step is fractional (< 1.0, e.g. 0.02 km) or (header explicitly has km and values <= 500)
                if (max_abs <= 500.0 and (median_diff < 1.0 or "(km)" in c_orig_lower or "km" in c_orig_lower)) and max_abs > 0:
                    self.logger.info(
                        "Auto-detected chainage in Kilometers (max=|%.3f|, step=%.3f); converting to meters (* 1000.0)",
                        max_abs,
                        median_diff,
                    )
                    df_renamed["chainage_m"] = pd.to_numeric(df_renamed["chainage_raw"], errors="coerce") * 1000.0
                else:
                    df_renamed["chainage_m"] = pd.to_numeric(df_renamed["chainage_raw"], errors="coerce")
            else:
                df_renamed["chainage_m"] = pd.to_numeric(df_renamed["chainage_raw"], errors="coerce")

        if "chainage_m" not in df_renamed.columns:
            has_spatial_coords = "easting" in df_renamed.columns and "northing" in df_renamed.columns
            if has_spatial_coords and alignment is not None:
                self.logger.info("Chainage column missing, but Easting/Northing found. Projecting onto alignment line...")
                if isinstance(alignment, LineString):
                    line = alignment
                elif hasattr(alignment, "geometry") and isinstance(alignment.geometry, LineString):
                    line = alignment.geometry
                elif hasattr(alignment, "alignment_line") and isinstance(alignment.alignment_line, LineString):
                    line = alignment.alignment_line
                elif isinstance(alignment, (list, tuple, np.ndarray)):
                    pts = []
                    for p in alignment:
                        if hasattr(p, "easting") and hasattr(p, "northing"):
                            pts.append((float(p.easting), float(p.northing)))
                        elif isinstance(p, dict):
                            pts.append((
                                float(p.get("easting_m", p.get("easting", p.get("x", 0.0)))),
                                float(p.get("northing_m", p.get("northing", p.get("y", 0.0)))),
                            ))
                        elif isinstance(p, (list, tuple)) and len(p) >= 2:
                            pts.append((float(p[0]), float(p[1])))
                    line = LineString(pts)
                else:
                    line = None

                if line is None or line.is_empty:
                    raise SurveyReaderError("Failed to construct a valid LineString geometry from provided alignment.")

                e_vals = pd.to_numeric(df_renamed["easting"], errors="coerce")
                n_vals = pd.to_numeric(df_renamed["northing"], errors="coerce")

                ch_proj: list[float] = []
                d_proj: list[float] = []
                for e, n in zip(e_vals, n_vals):
                    if math.isnan(e) or math.isnan(n):
                        ch_proj.append(np.nan)
                        d_proj.append(np.nan)
                    else:
                        pt = Point(e, n)
                        d = float(line.distance(pt))
                        if d <= max_corridor_offset_m:
                            ch = float(line.project(pt))
                            # Snap chainage within 0.5m to merge dual-rail measurements (left/right rail) at the same cross-section
                            ch_proj.append(round(round(ch / 0.5) * 0.5, 2))
                            d_proj.append(round(d, 3))
                        else:
                            ch_proj.append(np.nan)
                            d_proj.append(round(d, 3))

                df_renamed["chainage_m"] = ch_proj
                df_renamed["corridor_offset_m"] = d_proj
                df_renamed = df_renamed.dropna(subset=["chainage_m"]).copy()

                if df_renamed.empty:
                    raise SurveyReaderError(
                        f"No survey points fell within alignment corridor offset limit (<= {max_corridor_offset_m}m)."
                    )
                self.logger.info(
                    "Projected %d spatial survey points onto alignment corridor (max offset <= %.1fm)",
                    len(df_renamed),
                    max_corridor_offset_m,
                )
            elif allow_missing_chainage:
                self.logger.debug("Chainage column missing, but allow_missing_chainage is True. Proceeding with spatial coordinates.")
            else:
                extra_hint = " (Easting/Northing found; pass alignment geometry to project spatial coordinates)" if has_spatial_coords else ""
                raise SurveyReaderError(
                    f"Survey data missing required chainage column{extra_hint}. Found columns: {list(df.columns)}"
                )

        # Ensure at least one primary elevation level exists (ERL, PRL, PFL, or EGL)
        if "erl_m" not in df_renamed.columns:
            if "prl_m" in df_renamed.columns:
                self.logger.info("ERL column not found; mapping from PRL as primary rail level reference")
                df_renamed["erl_m"] = df_renamed["prl_m"]
            elif "pfl_m" in df_renamed.columns:
                self.logger.info("ERL column not found; deriving from PFL + %.3fm track buildup", self.track_buildup_m)
                df_renamed["erl_m"] = df_renamed["pfl_m"] + self.track_buildup_m
            elif "egl_m" in df_renamed.columns:
                self.logger.info("ERL column not found; using EGL ground level as elevation reference")
                df_renamed["erl_m"] = df_renamed["egl_m"]
            else:
                raise SurveyReaderError(
                    f"Survey data missing required elevation levels (ERL, PRL, PFL, or EGL). Found columns: {list(df.columns)}"
                )

        if "prl_m" not in df_renamed.columns and "erl_m" in df_renamed.columns:
            df_renamed["prl_m"] = df_renamed["erl_m"]

        if "egl_m" not in df_renamed.columns:
            if "efl_m" in df_renamed.columns:
                df_renamed["egl_m"] = df_renamed["efl_m"]
            elif "erl_m" in df_renamed.columns:
                df_renamed["egl_m"] = df_renamed["erl_m"] - self.track_buildup_m

        return df_renamed

    def load_data(
        self,
        file_path: Optional[str | Path] = None,
        raw_df: Optional[pd.DataFrame] = None,
        sheet_name: Optional[str] = None,
        alignment: Optional[Any] = None,
        max_corridor_offset_m: float = 50.0,
    ) -> pd.DataFrame:
        """Load tabular survey data from file or direct DataFrame.

        Args:
            file_path: Optional path to CSV, XLSX, or XLS survey file.
            raw_df: Optional direct pandas DataFrame.
            sheet_name: Optional specific sheet name for Excel workbooks.
            alignment: Optional alignment LineString or coordinates for projecting spatial points.
            max_corridor_offset_m: Corridor offset tolerance in meters (default: 50.0m).

        Returns:
            Standardized DataFrame.
        """
        if raw_df is not None:
            self.logger.info("Processing survey data from provided DataFrame with %d rows", len(raw_df))
            df = raw_df.copy()
        else:
            target_path = Path(file_path) if file_path else Path(self.survey_data_path)
            if not target_path or not target_path.exists():
                raise SurveyReaderError(f"Survey file does not exist: {target_path}")

            self.logger.info("Loading brownfield survey file: %s", target_path)
            suffix = target_path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(target_path)
            elif suffix in (".xlsx", ".xls"):
                xl = pd.ExcelFile(target_path)
                available = xl.sheet_names
                if sheet_name and sheet_name in available:
                    s_name = sheet_name
                elif self.survey_sheet_name and self.survey_sheet_name in available:
                    s_name = self.survey_sheet_name
                else:
                    best = available[0]
                    for sn in available:
                        sn_low = sn.lower()
                        if any(k in sn_low for k in ("master", "l-section", "lsection", "survey", "profile", "rail")):
                            best = sn
                            break
                    s_name = best
                self.logger.info("Reading Excel sheet: '%s' (available: %s)", s_name, available)
                df = self._find_header_row_and_load_excel(target_path, s_name)
            else:
                raise SurveyReaderError(f"Unsupported survey file format '{suffix}'. Supported: .csv, .xlsx, .xls")

        return self._standardize_columns(df, alignment=alignment, max_corridor_offset_m=max_corridor_offset_m)

    def process(
        self,
        file_path: Optional[str | Path] = None,
        raw_df: Optional[pd.DataFrame] = None,
        sheet_name: Optional[str] = None,
        target_chainages: Optional[np.ndarray | list[float]] = None,
        alignment: Optional[Any] = None,
        max_corridor_offset_m: float = 50.0,
    ) -> SurveyResult:
        """Execute full brownfield survey ingestion, validation, and resampling.

        Args:
            file_path: Path to survey file.
            raw_df: Alternatively, direct DataFrame.
            sheet_name: Optional specific sheet name for Excel workbooks.
            target_chainages: Optional array of chainages to align to.
            alignment: Optional alignment LineString or coordinates for projecting spatial points.
            max_corridor_offset_m: Corridor offset tolerance in meters (default: 50.0m).

        Returns:
            SurveyResult object with processed dataframes and audit metadata.
        """
        df = self.load_data(
            file_path=file_path,
            raw_df=raw_df,
            sheet_name=sheet_name,
            alignment=alignment,
            max_corridor_offset_m=max_corridor_offset_m,
        )

        # 1. Clean and convert numeric columns
        df["chainage_m"] = pd.to_numeric(df["chainage_m"], errors="coerce")
        df["erl_m"] = pd.to_numeric(df["erl_m"], errors="coerce")

        has_coords = "easting" in df.columns and "northing" in df.columns
        if has_coords:
            df["easting"] = pd.to_numeric(df["easting"], errors="coerce")
            df["northing"] = pd.to_numeric(df["northing"], errors="coerce")

        df = df.dropna(subset=["chainage_m", "erl_m"]).copy()

        if df.empty:
            raise SurveyReaderError("No valid survey rows remaining after numeric parsing of chainage and ERL.")

        # Sort by chainage and deduplicate
        df = df.sort_values(by="chainage_m").reset_index(drop=True)
        dup_count = int(df.duplicated(subset=["chainage_m"]).sum())
        if dup_count > 0:
            self.logger.warning("Found %d duplicate chainage points in survey; averaging duplicate levels", dup_count)
            # Group by chainage and take mean for numerical, first for feature / discrete
            agg_dict: dict[str, Any] = {}
            for col in df.columns:
                if col == "chainage_m":
                    continue
                if col in ("feature", "proposed_alignment", "proposed_gradient", "existing_gradient"):
                    agg_dict[col] = "first"
                elif pd.api.types.is_numeric_dtype(df[col]):
                    agg_dict[col] = "mean"
                else:
                    agg_dict[col] = "first"
            df = df.groupby("chainage_m", as_index=False).agg(agg_dict)

        # 2. Derive EFL if absent or missing
        if "efl_m" not in df.columns:
            self.logger.info(
                "EFL column not provided in survey; deriving EFL = ERL - %.3fm (RDSO BG track buildup)",
                self.track_buildup_m,
            )
            df["efl_m"] = df["erl_m"] - self.track_buildup_m
        else:
            df["efl_m"] = pd.to_numeric(df["efl_m"], errors="coerce")
            missing_efl = df["efl_m"].isna()
            if missing_efl.any():
                df.loc[missing_efl, "efl_m"] = df.loc[missing_efl, "erl_m"] - self.track_buildup_m

        # 3. Derive Track Center if absent or missing
        if "track_center_m" not in df.columns:
            self.logger.info(
                "Track center distance column not provided; assigning standard %.2fm (IRS SOD 2004)",
                self.standard_track_center_m,
            )
            df["track_center_m"] = self.standard_track_center_m
        else:
            df["track_center_m"] = pd.to_numeric(df["track_center_m"], errors="coerce").fillna(
                self.standard_track_center_m
            )

        # 4. Feature / structure column
        if "feature" not in df.columns:
            df["feature"] = ""
        else:
            df["feature"] = df["feature"].fillna("").astype(str).str.strip()

        # 5. Quality & Anomaly Checks
        sod_violations: list[dict[str, Any]] = []
        under_sod = df[df["track_center_m"] < self.min_track_center_m]
        for _, row in under_sod.iterrows():
            sod_violations.append(
                {
                    "chainage_m": float(row["chainage_m"]),
                    "track_center_m": float(row["track_center_m"]),
                    "min_required_m": self.min_track_center_m,
                    "deficit_m": round(self.min_track_center_m - float(row["track_center_m"]), 3),
                }
            )
        if sod_violations:
            self.logger.warning(
                "CRITICAL: Found %d IRS SOD 2004 track center infringements (< %.3fm)",
                len(sod_violations),
                self.min_track_center_m,
            )

        gradient_anomalies: list[dict[str, Any]] = []
        d_chainage = np.diff(df["chainage_m"].to_numpy())
        d_erl = np.diff(df["erl_m"].to_numpy())
        valid_spans = d_chainage > 0.1
        if valid_spans.any():
            gradients = np.abs(d_erl[valid_spans] / d_chainage[valid_spans])
            spike_idx = np.where(gradients > self.max_plausible_gradient)[0]
            for idx in spike_idx:
                ch_start = float(df.iloc[idx]["chainage_m"])
                ch_end = float(df.iloc[idx + 1]["chainage_m"])
                grad_pct = round(float(gradients[idx]) * 100.0, 2)
                gradient_anomalies.append(
                    {
                        "chainage_start_m": ch_start,
                        "chainage_end_m": ch_end,
                        "apparent_gradient_pct": grad_pct,
                        "threshold_pct": round(self.max_plausible_gradient * 100.0, 2),
                    }
                )
                self.logger.warning(
                    "Survey Level Spike detected between Ch %.1fm and %.1fm: Gradient = %.2f%% (> %.2f%% threshold)",
                    ch_start,
                    ch_end,
                    grad_pct,
                    self.max_plausible_gradient * 100.0,
                )

        # 6. Uniform Continuous Resampling
        min_ch = float(df["chainage_m"].min())
        max_ch = float(df["chainage_m"].max())

        if target_chainages is not None and len(target_chainages) > 0:
            resample_ch = np.asarray(target_chainages, dtype=float)
            resample_ch = resample_ch[(resample_ch >= min_ch) & (resample_ch <= max_ch)]
        else:
            start_round = math.ceil(min_ch / self.chainage_interval_m) * self.chainage_interval_m
            end_round = math.floor(max_ch / self.chainage_interval_m) * self.chainage_interval_m
            if end_round < start_round:
                resample_ch = np.array([min_ch, max_ch], dtype=float)
            else:
                resample_ch = np.arange(
                    start_round, end_round + self.chainage_interval_m * 0.5, self.chainage_interval_m
                )

        # Monotonic linear interpolation along continuous chainage
        erl_interp = np.interp(resample_ch, df["chainage_m"], df["erl_m"])
        efl_interp = np.interp(resample_ch, df["chainage_m"], df["efl_m"])
        tc_interp = np.interp(resample_ch, df["chainage_m"], df["track_center_m"])

        interp_data: dict[str, Any] = {
            "chainage_m": np.round(resample_ch, 2),
            "erl_m": np.round(erl_interp, 3),
            "efl_m": np.round(efl_interp, 3),
            "track_center_m": np.round(tc_interp, 3),
            "feature": "",
        }

        # Interpolate additional engineering levels if present in survey
        if "prl_m" in df.columns:
            valid_prl = df.dropna(subset=["prl_m"])
            if len(valid_prl) > 1:
                interp_data["prl_m"] = np.round(np.interp(resample_ch, valid_prl["chainage_m"], valid_prl["prl_m"]), 3)
        if "pfl_m" in df.columns:
            valid_pfl = df.dropna(subset=["pfl_m"])
            if len(valid_pfl) > 1:
                interp_data["pfl_m"] = np.round(np.interp(resample_ch, valid_pfl["chainage_m"], valid_pfl["pfl_m"]), 3)
        elif "prl_m" in interp_data:
            interp_data["pfl_m"] = np.round(interp_data["prl_m"] - self.track_buildup_m, 3)

        if "egl_m" in df.columns:
            valid_egl = df.dropna(subset=["egl_m"])
            if len(valid_egl) > 1:
                interp_data["egl_m"] = np.round(np.interp(resample_ch, valid_egl["chainage_m"], valid_egl["egl_m"]), 3)

        if "diff_prl_erl_m" in df.columns:
            valid_diff = df.dropna(subset=["diff_prl_erl_m"])
            if len(valid_diff) > 1:
                interp_data["diff_prl_erl_m"] = np.round(np.interp(resample_ch, valid_diff["chainage_m"], valid_diff["diff_prl_erl_m"]), 3)
        elif "prl_m" in interp_data:
            interp_data["diff_prl_erl_m"] = np.round(interp_data["prl_m"] - interp_data["erl_m"], 3)

        if "cut_fill_m" in df.columns:
            valid_cf = df.dropna(subset=["cut_fill_m"])
            if len(valid_cf) > 1:
                interp_data["cut_fill_m"] = np.round(np.interp(resample_ch, valid_cf["chainage_m"], valid_cf["cut_fill_m"]), 3)
        elif "egl_m" in interp_data and "pfl_m" in interp_data:
            interp_data["cut_fill_m"] = np.round(interp_data["egl_m"] - interp_data["pfl_m"], 3)

        # Interpolate coordinates if present for 2D Plan View
        if has_coords:
            valid_coords = df.dropna(subset=["easting", "northing"])
            if not valid_coords.empty and len(valid_coords) > 1:
                e_interp = np.interp(resample_ch, valid_coords["chainage_m"], valid_coords["easting"])
                n_interp = np.interp(resample_ch, valid_coords["chainage_m"], valid_coords["northing"])
                interp_data["easting"] = np.round(e_interp, 4)
                interp_data["northing"] = np.round(n_interp, 4)

        # Forward-fill piecewise discrete columns (gradients, alignment)
        if "proposed_gradient" in df.columns:
            valid_pg = df.dropna(subset=["proposed_gradient"]).sort_values("chainage_m")
            if not valid_pg.empty:
                s_ch_df = pd.DataFrame({"chainage_m": resample_ch})
                merged = pd.merge_asof(s_ch_df, valid_pg[["chainage_m", "proposed_gradient"]], on="chainage_m", direction="backward")
                interp_data["proposed_gradient"] = merged["proposed_gradient"].bfill().ffill().to_numpy()

        if "existing_gradient" in df.columns:
            valid_eg = df.dropna(subset=["existing_gradient"]).sort_values("chainage_m")
            if not valid_eg.empty:
                s_ch_df = pd.DataFrame({"chainage_m": resample_ch})
                merged = pd.merge_asof(s_ch_df, valid_eg[["chainage_m", "existing_gradient"]], on="chainage_m", direction="backward")
                interp_data["existing_gradient"] = merged["existing_gradient"].bfill().ffill().to_numpy()

        if "proposed_alignment" in df.columns:
            valid_pa = df.dropna(subset=["proposed_alignment"]).sort_values("chainage_m")
            if not valid_pa.empty:
                s_ch_df = pd.DataFrame({"chainage_m": resample_ch})
                merged = pd.merge_asof(s_ch_df, valid_pa[["chainage_m", "proposed_alignment"]], on="chainage_m", direction="backward")
                interp_data["proposed_alignment"] = merged["proposed_alignment"].bfill().ffill().to_numpy()

        interp_df = pd.DataFrame(interp_data)

        # 7. Extract Asset Landmark Features
        assets_df = df[df["feature"].str.len() > 0].copy()
        if not assets_df.empty:
            self.logger.info("Identified %d landmark assets in survey data", len(assets_df))
            # Match each asset to the nearest station in interpolated_df for annotation
            for _, asset_row in assets_df.iterrows():
                ch_asset = float(asset_row["chainage_m"])
                feat_str = str(asset_row["feature"])
                idx_nearest = (interp_df["chainage_m"] - ch_asset).abs().idxmin()
                curr_feat = interp_df.at[idx_nearest, "feature"]
                interp_df.at[idx_nearest, "feature"] = (
                    f"{curr_feat}; {feat_str}" if curr_feat else feat_str
                )

        is_spatial_projection = "corridor_offset_m" in df.columns
        tie_in_anchor: Optional[dict[str, float]] = None
        yard_extent_m = 0.0
        if not df.empty:
            first_row = df.iloc[0]
            tie_in_anchor = {
                "chainage_m": round(float(first_row["chainage_m"]), 2),
                "erl_m": round(float(first_row["erl_m"]), 3),
                "pfl_m": round(float(first_row["erl_m"]) - self.track_buildup_m, 3),
            }
            yard_extent_m = round(float(df["chainage_m"].max()), 2)

        metadata: dict[str, Any] = {
            "total_survey_points": int(len(df)),
            "interpolated_stations": int(len(interp_df)),
            "chainage_start_m": min_ch,
            "chainage_end_m": max_ch,
            "total_length_m": round(max_ch - min_ch, 2),
            "erl_min_m": round(float(df["erl_m"].min()), 3),
            "erl_max_m": round(float(df["erl_m"].max()), 3),
            "track_center_min_m": round(float(df["track_center_m"].min()), 3),
            "track_center_max_m": round(float(df["track_center_m"].max()), 3),
            "sod_violations_count": len(sod_violations),
            "sod_violations": sod_violations,
            "gradient_anomalies_count": len(gradient_anomalies),
            "gradient_anomalies": gradient_anomalies,
            "assets_count": int(len(assets_df)),
            "has_coordinates": bool(has_coords and "easting" in interp_df.columns),
            "is_spatial_projection": bool(is_spatial_projection),
            "tie_in_anchor": tie_in_anchor,
            "yard_extent_m": yard_extent_m,
        }

        self.logger.info(
            "Survey ingestion complete: %d points -> %d stations (Ch %.1f to %.1f m). Min ERL=%.2fm, Max ERL=%.2fm",
            len(df),
            len(interp_df),
            min_ch,
            max_ch,
            metadata["erl_min_m"],
            metadata["erl_max_m"],
        )

        return SurveyResult(
            raw_df=df,
            interpolated_df=interp_df,
            assets_df=assets_df.reset_index(drop=True),
            metadata=metadata,
        )

    def project_spatial_survey(
        self,
        survey_source: str | Path | pd.DataFrame,
        alignment: Any,
        max_corridor_offset_m: float = 50.0,
        target_chainages: Optional[np.ndarray | list[float]] = None,
    ) -> SurveyResult:
        """Direct helper to ingest and project spatial point cloud surveys (Easting, Northing, Rail Level).

        Args:
            survey_source: File path (CSV/Excel) or pandas DataFrame.
            alignment: LineString or list of coordinates representing alignment centerline.
            max_corridor_offset_m: Maximum distance (in meters) from alignment to keep points.
            target_chainages: Optional specific chainages to resample onto.

        Returns:
            SurveyResult containing projected points, interpolated points, and tie-in anchor.
        """
        if isinstance(survey_source, pd.DataFrame):
            return self.process(
                raw_df=survey_source,
                alignment=alignment,
                max_corridor_offset_m=max_corridor_offset_m,
                target_chainages=target_chainages,
            )
        return self.process(
            file_path=survey_source,
            alignment=alignment,
            max_corridor_offset_m=max_corridor_offset_m,
            target_chainages=target_chainages,
        )

    def extract_centerline_from_survey(
        self,
        survey_source: str | Path | pd.DataFrame | io.BytesIO,
        start_chainage_m: float = 0.0,
        chainage_interval_m: float = 20.0,
        gauge_pair_max_dist_m: float = 2.5,
        cross_section_cluster_dist_m: float = 15.0,
        track_index: int = 0,
        epsg: Optional[int] = None,
        reverse_direction: bool = False,
    ) -> dict[str, Any]:
        """Extract continuous railway track alignment centerline from survey point cloud (E, N, Z).

        Engineering workflow:
            1. Pairs adjacent left/right rails (spacing <= gauge_pair_max_dist_m, e.g. BG 1.676m)
               into 3D centerline midpoints: (E_cl, N_cl, Z_cl).
            2. Clusters points at the same cross section in multi-track yards (offset <= 15m).
            3. Selects the primary track (track_index, default 0).
            4. Sequences points along the principal track corridor to eliminate back-and-forth jumps.
            5. Calculates cumulative chainages starting from start_chainage_m.
            6. Resamples uniform stations at chainage_interval_m (RDSO standard 20m).
            7. Projects UTM coordinates into WGS84 (lon, lat) for GIS map overlay.

        Args:
            survey_source: File path, pandas DataFrame, or bytes buffer.
            start_chainage_m: Starting chainage in meters (default: 0.0).
            chainage_interval_m: Resampling interval in meters (default: 20.0).
            gauge_pair_max_dist_m: Maximum transverse distance between rail pair (default: 2.5m).
            cross_section_cluster_dist_m: Max transverse offset to group parallel tracks into a cross section (default: 15.0m).
            track_index: Track line index within multi-track cross-sections (default: 0, Main Line).
            epsg: Target UTM EPSG code. If None, auto-detected or defaults to 32645 (India Zone 45N).
            reverse_direction: Whether to flip the progression direction (default: False).

        Returns:
            Dictionary containing:
                - stations: list of uniform Station objects with chainage, E, N, Z, lon, lat
                - raw_vertices: list of surveyed centerline vertices
                - raw_lines: list of tangent segments with bearings and lengths
                - total_length_m: total alignment length in meters
                - start_chainage_m: start chainage in meters
                - epsg: detected UTM EPSG
                - tie_in_anchor: starting elevation & chainage anchor
                - metadata: summary statistics
        """
        if isinstance(survey_source, pd.DataFrame):
            df = survey_source.copy()
        elif hasattr(survey_source, "read"):
            # BytesIO or file-like buffer
            import io
            if isinstance(survey_source, io.BytesIO):
                survey_source.seek(0)
            try:
                df = pd.read_csv(survey_source)
            except Exception:
                if hasattr(survey_source, "seek"):
                    survey_source.seek(0)
                df = pd.read_excel(survey_source)
        else:
            p = Path(survey_source)
            if not p.exists():
                raise SurveyReaderError(f"Survey file not found: {p}")
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            elif p.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(p)
            else:
                raise SurveyReaderError(f"Unsupported survey file format: {p.suffix}")

        df_clean = self._standardize_columns(df, allow_missing_chainage=True)
        if "easting" not in df_clean.columns or "northing" not in df_clean.columns:
            raise SurveyReaderError("Survey file missing required Easting and Northing coordinate columns.")

        if "erl_m" not in df_clean.columns:
            # Fallback to egl_m or elevation if available
            if "egl_m" in df_clean.columns:
                df_clean["erl_m"] = df_clean["egl_m"]
            elif "elevation" in df_clean.columns:
                df_clean["erl_m"] = df_clean["elevation"]
            else:
                df_clean["erl_m"] = 0.0

        # Drop NaN coordinates
        df_clean = df_clean.dropna(subset=["easting", "northing"]).reset_index(drop=True)
        if len(df_clean) < 2:
            raise SurveyReaderError("Survey contains fewer than 2 valid coordinate points.")

        east_vals = df_clean["easting"].to_numpy(dtype=float)
        north_vals = df_clean["northing"].to_numpy(dtype=float)
        rl_vals = df_clean["erl_m"].fillna(0.0).to_numpy(dtype=float)
        pt_ids = df_clean.get("point_number", pd.Series(range(1, len(df_clean) + 1))).to_numpy()

        # Step 1: Pair dual rails (spacing <= gauge_pair_max_dist_m) into 3D centerline points
        cl_pts: list[dict[str, Any]] = []
        i = 0
        n_pts = len(df_clean)
        while i < n_pts:
            if i + 1 < n_pts:
                d = math.hypot(east_vals[i + 1] - east_vals[i], north_vals[i + 1] - north_vals[i])
                if d <= gauge_pair_max_dist_m:
                    cl_pts.append({
                        "east": (east_vals[i] + east_vals[i + 1]) / 2.0,
                        "north": (north_vals[i] + north_vals[i + 1]) / 2.0,
                        "rl": (rl_vals[i] + rl_vals[i + 1]) / 2.0,
                        "pt": pt_ids[i],
                        "is_pair": True,
                    })
                    i += 2
                    continue
            cl_pts.append({
                "east": east_vals[i],
                "north": north_vals[i],
                "rl": rl_vals[i],
                "pt": pt_ids[i],
                "is_pair": False,
            })
            i += 1

        if not cl_pts:
            raise SurveyReaderError("Failed to extract any centerline points from survey.")

        # Step 2: Cluster points at the same cross section in multi-track yards
        cross_sections: list[list[dict[str, Any]]] = []
        curr_xs = [cl_pts[0]]
        for p in cl_pts[1:]:
            last = curr_xs[-1]
            d = math.hypot(p["east"] - last["east"], p["north"] - last["north"])
            if d < cross_section_cluster_dist_m:
                curr_xs.append(p)
            else:
                cross_sections.append(curr_xs)
                curr_xs = [p]
        cross_sections.append(curr_xs)

        # Step 3: Select track line from each cross-section
        track_pts: list[dict[str, Any]] = []
        for xs in cross_sections:
            selected_idx = min(track_index, len(xs) - 1)
            track_pts.append(xs[selected_idx])

        if len(track_pts) < 2:
            raise SurveyReaderError("Centerline contains fewer than 2 cross sections.")

        # Step 4: Sequence points along principal corridor axis
        p_min = min(track_pts, key=lambda p: p["east"])
        p_max = max(track_pts, key=lambda p: p["east"])
        v_line = np.array([p_max["east"] - p_min["east"], p_max["north"] - p_min["north"]])
        v_len = float(np.linalg.norm(v_line))
        if v_len > 1e-4:
            v_u = v_line / v_len
            for p in track_pts:
                vec = np.array([p["east"] - p_min["east"], p["north"] - p_min["north"]])
                p["proj"] = float(np.dot(vec, v_u))
            sorted_track = sorted(track_pts, key=lambda p: p["proj"])
        else:
            sorted_track = track_pts

        if reverse_direction:
            sorted_track = sorted_track[::-1]

        # Step 5: Compute cumulative chainages
        cum_d = [float(start_chainage_m)]
        for k in range(len(sorted_track) - 1):
            d = math.hypot(
                sorted_track[k + 1]["east"] - sorted_track[k]["east"],
                sorted_track[k + 1]["north"] - sorted_track[k]["north"],
            )
            cum_d.append(cum_d[-1] + d)

        total_len = cum_d[-1] - float(start_chainage_m)
        if total_len <= 0.0:
            raise SurveyReaderError("Total alignment length calculated is zero or negative.")

        # Step 6: Determine UTM EPSG & Transformer to WGS84
        target_epsg = epsg
        if target_epsg is None:
            # Default to UTM 45N for Eastern India / Kanthi-Egra or auto-detect
            mean_e = float(np.mean([p["east"] for p in sorted_track]))
            mean_n = float(np.mean([p["north"] for p in sorted_track]))
            target_epsg = 32645 if 200000 <= mean_e <= 800000 and 1000000 <= mean_n <= 3500000 else 32644

        transformer = Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)

        # Step 7: Build Raw Vertices & Raw Lines
        raw_vertices: list[dict[str, Any]] = []
        for idx, p in enumerate(sorted_track):
            e = float(p["east"])
            n = float(p["north"])
            z = float(p["rl"])
            ch = float(cum_d[idx])
            try:
                lon, lat = transformer.transform(e, n)
            except Exception:
                lon, lat = 0.0, 0.0
            raw_vertices.append({
                "vertex_index": idx,
                "easting_m": round(e, 4),
                "northing_m": round(n, 4),
                "elevation_m": round(z, 3),
                "chainage_m": round(ch, 3),
                "lon_deg": round(lon, 7),
                "lat_deg": round(lat, 7),
            })

        raw_lines: list[dict[str, Any]] = []
        for idx in range(len(sorted_track) - 1):
            p1 = sorted_track[idx]
            p2 = sorted_track[idx + 1]
            dx = p2["east"] - p1["east"]
            dy = p2["north"] - p1["north"]
            seg_len = math.hypot(dx, dy)
            bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
            raw_lines.append({
                "line_id": f"Line {idx + 1}",
                "segment_index": idx,
                "start_easting_m": round(p1["east"], 4),
                "start_northing_m": round(p1["north"], 4),
                "end_easting_m": round(p2["east"], 4),
                "end_northing_m": round(p2["north"], 4),
                "length_m": round(seg_len, 3),
                "bearing_deg": round(bearing, 2),
                "start_chainage_m": round(cum_d[idx], 3),
                "end_chainage_m": round(cum_d[idx + 1], 3),
            })

        # Step 8: Resample uniform stations at chainage_interval_m (RDSO standard 20m)
        interval = max(5.0, float(chainage_interval_m))
        num_samples = int(math.ceil(total_len / interval))
        sample_chs = [float(start_chainage_m) + s * interval for s in range(num_samples + 1)]
        if sample_chs[-1] < cum_d[-1]:
            sample_chs.append(cum_d[-1])

        track_easts = [p["east"] for p in sorted_track]
        track_norths = [p["north"] for p in sorted_track]
        track_rls = [p["rl"] for p in sorted_track]

        res_easts = np.interp(sample_chs, cum_d, track_easts)
        res_norths = np.interp(sample_chs, cum_d, track_norths)
        res_elevs = np.interp(sample_chs, cum_d, track_rls)

        stations: list[dict[str, Any]] = []
        for s_idx, ch in enumerate(sample_chs):
            e = float(res_easts[s_idx])
            n = float(res_norths[s_idx])
            z = float(res_elevs[s_idx])
            try:
                lon, lat = transformer.transform(e, n)
            except Exception:
                lon, lat = 0.0, 0.0

            km_val = int(ch // 1000)
            m_val = ch % 1000
            ch_str = f"{km_val}+{m_val:06.2f}" if abs(m_val) >= 100 else f"{km_val}+{m_val:05.2f}"

            stations.append({
                "chainage_m": round(ch, 3),
                "chainage_display": ch_str,
                "easting_m": round(e, 4),
                "northing_m": round(n, 4),
                "elevation_m": round(z, 3),
                "existing_rail_level_m": round(z, 3),
                "erl_m": round(z, 3),
                "existing_formation_level_m": round(z - float(self.track_buildup_m), 3),
                "efl_m": round(z - float(self.track_buildup_m), 3),
                "lon_deg": round(lon, 7),
                "lat_deg": round(lat, 7),
            })

        first_stn = stations[0]
        tie_in_anchor = {
            "chainage_m": first_stn["chainage_m"],
            "erl_m": first_stn["erl_m"],
            "pfl_m": round(first_stn["erl_m"] - self.track_buildup_m, 3),
        }

        metadata = {
            "survey_points_count": int(len(df_clean)),
            "centerline_vertices_count": len(sorted_track),
            "resampled_stations_count": len(stations),
            "cross_sections_count": len(cross_sections),
            "total_length_m": round(total_len, 3),
            "start_chainage_m": round(float(start_chainage_m), 3),
            "end_chainage_m": round(float(cum_d[-1]), 3),
            "erl_min_m": round(float(min(track_rls)), 3),
            "erl_max_m": round(float(max(track_rls)), 3),
            "epsg": target_epsg,
        }

        self.logger.info(
            "Extracted alignment centerline from survey: %d points -> %d vertices -> %d stations (Length: %.2fm). Tie-in: Ch %.1f @ %.3fm",
            len(df_clean),
            len(sorted_track),
            len(stations),
            total_len,
            tie_in_anchor["chainage_m"],
            tie_in_anchor["erl_m"],
        )

        return {
            "stations": stations,
            "raw_vertices": raw_vertices,
            "raw_lines": raw_lines,
            "total_length_m": round(total_len, 3),
            "start_chainage_m": round(float(start_chainage_m), 3),
            "epsg": target_epsg,
            "tie_in_anchor": tie_in_anchor,
            "metadata": metadata,
        }
