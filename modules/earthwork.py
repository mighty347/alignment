"""
Module: modules/earthwork.py
Purpose: Module 8 — Earthwork (Cut and Fill Calculations).

Engineering Basis:
    Calculates the vertical difference between Formation Level (FL) and 
    Existing Ground Level (EGL) at each chainage point. Computes the 
    trapezoidal cross-sectional area based on Indian Railways formation widths 
    and side slopes, and derives the earthwork volume (Average End Area method).

Input: VAlignResult (Module 7).
Output: EarthworkResult (DataFrame with Cut/Fill depths, areas, and volumes).

Standards: RDSO / Indian Railways BG
Author   : Railway ESP Generator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.vertical_alignment import VAlignResult
from utils.logger import get_logger


logger = get_logger(__name__)


class EarthworkError(Exception):
    """Custom exception for Earthwork calculation errors."""
    pass


@dataclass
class EarthworkResult:
    """Result object returned by Earthwork module.

    Attributes:
        profile_dataframe: The original DataFrame appended with earthwork columns:
            - height_m (FL - EGL)
            - ew_type ('CUT', 'FILL', 'LEVEL')
            - ew_area_sqm (Cross-sectional area)
            - ew_vol_cum (Volume for the segment since previous chainage)
        metadata: Engineering metadata dictionary with total volumes.
    """
    profile_dataframe: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class Earthwork:
    """Computes Cut/Fill depths, areas, and volumes along the alignment.

    Implements Module 8 of the ESP Generator pipeline.

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Example:
        >>> ew_result = Earthwork(config).calculate(valign_result)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

        # RDSO BG 2022 default values for BG Single Line New Works
        ew_config = config.get("earthwork", {})
        track_type = str(config.get("railway", {}).get("track_type", "single")).lower()
        track_widths = ew_config.get("formation_widths_by_track", {}).get(track_type, {})
        self.width_cut = float(ew_config.get("formation_width_cut_m", 6.85))
        self.width_fill = float(ew_config.get("formation_width_fill_m", 7.70))
        if track_type != "single" and track_widths:
            self.width_cut = float(track_widths.get("cut_m", self.width_cut))
            self.width_fill = float(track_widths.get("fill_low_m", self.width_fill))
        validation_cfg = config.get("validation", {})
        self.width_fill_high = float(
            track_widths.get(
                "fill_high_m",
                validation_cfg.get("formation_width_fill_high_m", self.width_fill),
            )
        )
        self.fill_height_threshold = float(validation_cfg.get("fill_height_threshold_m", 6.0))
        self.track_type = track_type
        self.has_track_widths = track_type == "single" or {
            "cut_m", "fill_low_m"
        }.issubset(track_widths)
        
        # Side slopes (H:1V — horizontal distance per unit vertical)
        self.slope_cut = float(ew_config.get("side_slope_cut", 1.5))    # 1.5:1 ordinary soil (RDSO)
        self.slope_fill = float(ew_config.get("side_slope_fill", 2.0))  # 2:1 embankment (RDSO)

        # Railway Boundary / Right-Of-Way Parameters
        bound_config = config.get("railway_boundary", {})
        self.bridge_top_width = float(bound_config.get("base_bridge_width_m", 7.85))
        self.cutting_bottom_width = float(bound_config.get("cutting_bottom_width_m", 9.50))
        self.bound_slope_fill = float(bound_config.get("fill_slope_ratio", 2.0))
        self.bound_slope_cut = float(bound_config.get("cut_slope_ratio", 1.0))
        self.bound_slope_berm = float(bound_config.get("berm_slope_ratio", 2.0))
        self.cut_berm_slope_ratio = float(bound_config.get("cut_berm_slope_ratio", 1.0))
        self.cut_high_extra_offset_m = float(bound_config.get("cut_high_extra_offset_m", 3.0))
        self.cutting_drain_width_m = float(bound_config.get("cutting_drain_width_m", 1.5))
        self.fill_height_threshold = float(bound_config.get("fill_height_threshold_m", 6.0))
        self.cut_height_threshold = float(bound_config.get("cut_height_threshold_m", 6.0))
        self.safety_buffer_low_m = float(bound_config.get("safety_buffer_low_m", 3.0))
        self.safety_buffer_high_m = float(bound_config.get("safety_buffer_high_m", 6.0))
        self.safety_buffer_m = float(bound_config.get("safety_buffer_m", 3.0))
        self.default_track_center_offset_m = float(bound_config.get("default_track_center_offset_m", 0.0))
        self.track_center_offset_m = self.default_track_center_offset_m
        self.default_boundary_offset_m = float(bound_config.get("default_boundary_offset_m", 15.0))
        self.min_row_interval_length_m = float(bound_config.get("min_row_interval_length_m", 100.0))
        self.approach_lead_m = float(bound_config.get("approach_lead_m", 20.0))
        self.taper_ratio = float(bound_config.get("taper_ratio", 10.0))
        self.min_taper_length_m = float(bound_config.get("min_taper_length_m", 20.0))
        self.use_row_excel = bool(bound_config.get("use_row_excel", True))

    def load_row_excel(self, excel_path: Path | str | None = None) -> pd.DataFrame | None:
        """Load Right-Of-Way (ROW) boundary schedule from Row_Excel.xlsx.

        Extracts only the 'Chainage' and 'Desired LA' columns.

        Args:
            excel_path: Optional explicit path to Row_Excel.xlsx.

        Returns:
            pd.DataFrame | None: DataFrame with columns ['chainage', 'desired_la'] sorted by chainage.
        """
        paths_cfg = self.config.get("paths", {})
        candidates = [
            excel_path,
            paths_cfg.get("row_excel_path"),
            "D:/ESP-Backup/Row_Excel.xlsx",
            "Row_Excel.xlsx",
            "data/Row_Excel.xlsx",
        ]

        target_p: Path | None = None
        for c in candidates:
            if c:
                p = Path(c)
                if p.exists():
                    target_p = p
                    break

        if target_p is None:
            self.logger.debug("No Row_Excel.xlsx found in candidate locations.")
            return None

        try:
            self.logger.info("Reading ROW boundary schedule from %s", target_p)
            df_raw = pd.read_excel(target_p) if target_p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(target_p)

            ch_col: str | None = None
            la_col: str | None = None

            for col in df_raw.columns:
                norm = " ".join(str(col).strip().split()).lower().replace("_", " ")
                if norm in ["chainage", "chainage m", "ch", "location"]:
                    ch_col = col

            # Strictly prioritize exact 'Desired LA' column
            for col in df_raw.columns:
                norm = " ".join(str(col).strip().split()).lower().replace("_", " ")
                if norm in ["desired la", "desired  la", "desired la (m)", "la desired"]:
                    la_col = col
                    break

            if la_col is None:
                for col in df_raw.columns:
                    norm = " ".join(str(col).strip().split()).lower().replace("_", " ")
                    if norm in ["desired land", "desired land with road"]:
                        la_col = col
                        break

            if ch_col is None or la_col is None:
                self.logger.warning(
                    "Row_Excel (%s) missing required columns (found: %s). Need 'Chainage' and 'Desired LA'.",
                    target_p.name, df_raw.columns.tolist(),
                )
                return None

            df_clean = df_raw[[ch_col, la_col]].dropna().copy()
            df_clean[ch_col] = pd.to_numeric(df_clean[ch_col], errors="coerce")
            df_clean[la_col] = pd.to_numeric(df_clean[la_col], errors="coerce")
            df_clean = df_clean.dropna()

            df_clean = df_clean.rename(columns={ch_col: "chainage", la_col: "desired_la"})
            df_clean = df_clean.sort_values("chainage").drop_duplicates("chainage").reset_index(drop=True)

            self.logger.info(
                "Successfully loaded %d ROW boundary stations from %s (Chainage %.1fm to %.1fm, LA range: %.1fm to %.1fm)",
                len(df_clean), target_p.name,
                float(df_clean["chainage"].min()), float(df_clean["chainage"].max()),
                float(df_clean["desired_la"].min()), float(df_clean["desired_la"].max()),
            )
            return df_clean
        except Exception as exc:
            self.logger.warning("Error reading ROW Excel %s: %s", target_p, exc)
            return None

    def compute_boundary_profile_from_row_excel(
        self,
        chainages: np.ndarray,
        row_df: pd.DataFrame | None = None,
    ) -> np.ndarray | None:
        """Compute boundary offset profile from Row_Excel Desired LA column using 1D interpolation.

        Args:
            chainages: Array of chainage values along alignment.
            row_df: Optional pre-loaded ROW DataFrame with ['chainage', 'desired_la'].

        Returns:
            np.ndarray | None: Array of boundary offsets at each chainage, or None if unavailable.
        """
        if row_df is None:
            row_df = self.load_row_excel()

        if row_df is None or row_df.empty:
            return None

        chs = np.asarray(chainages, dtype=float)
        ref_chs = row_df["chainage"].values.astype(float)
        ref_la = row_df["desired_la"].values.astype(float)

        offsets = np.interp(chs, ref_chs, ref_la)
        return offsets

    def generate_stepped_boundary_geometry(
        self,
        get_paper_pos: Any,
        start_ch: float,
        end_ch: float,
        scale_factor: float = 1.0,
        anno_scale: float = 1.0,
        row_df: pd.DataFrame | None = None,
        bridge_df: pd.DataFrame | None = None,
        profile_df: pd.DataFrame | None = None,
        sample_step_m: float = 5.0,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[dict[str, Any]]]:
        """Generate stepped 2D boundary polyline coordinates and transition callout data.

        When Desired LA changes value (e.g. 10m -> 15m), forms a clean 90-degree
        perpendicular step at the change station and captures callout metadata
        (chainage, old/new widths, normal heading) for dimensioning.

        Args:
            get_paper_pos: Callable mapping chainage -> (px, py, heading_rad).
            start_ch: Start chainage station.
            end_ch: End chainage station.
            scale_factor: CAD drawing scale factor.
            anno_scale: Annotation scale multiplier.
            row_df: Optional preloaded Row_Excel DataFrame.
            bridge_df: Optional bridge DataFrame for fallback splays.
            profile_df: Optional profile DataFrame.
            sample_step_m: Sampling distance along parallel straight/curve segments.

        Returns:
            tuple:
                - bound_left_pts: List of (x, y) coordinates for left boundary.
                - bound_right_pts: List of (x, y) coordinates for right boundary.
                - transition_callouts: List of transition metadata dicts.
        """
        import math

        if row_df is None:
            row_df = self.load_row_excel()

        total_len = max(1.0, end_ch - start_ch)
        bound_left_pts: list[tuple[float, float]] = []
        bound_right_pts: list[tuple[float, float]] = []
        transition_callouts: list[dict[str, Any]] = []

        if row_df is None or row_df.empty or not self.use_row_excel:
            # Fallback to smooth continuous baseline profile
            sample_chs = np.linspace(start_ch, end_ch, max(120, int(total_len / sample_step_m)))
            smooth_offsets = self.compute_smooth_zonal_boundary_profile(
                chainages=sample_chs,
                bridge_df=bridge_df,
                profile_df=profile_df,
            )
            for i, b_ch in enumerate(sample_chs):
                des_land = float(smooth_offsets[i])
                px_b, py_b, h_rad = get_paper_pos(b_ch)
                perp_b = h_rad + (math.pi / 2.0)
                cos_pb, sin_pb = math.cos(perp_b), math.sin(perp_b)
                offset_scaled = des_land * scale_factor

                bound_left_pts.append((px_b - offset_scaled * cos_pb, py_b - offset_scaled * sin_pb))
                bound_right_pts.append((px_b + offset_scaled * cos_pb, py_b + offset_scaled * sin_pb))

            return bound_left_pts, bound_right_pts, transition_callouts

        # Filter relevant stations from Row_Excel covering [start_ch, end_ch]
        # Include one station before and one after to maintain continuity across sheets
        r_chs = row_df["chainage"].values
        r_la = row_df["desired_la"].values

        # Build contiguous constant-width intervals
        raw_intervals: list[dict[str, Any]] = []
        n_rows = len(row_df)
        i = 0
        while i < n_rows:
            c_start = float(r_chs[i])
            w_val = float(r_la[i])
            j = i + 1
            while j < n_rows and float(r_la[j]) == w_val:
                j += 1
            c_end = float(r_chs[j - 1]) if j <= n_rows else float(r_chs[-1])
            # If next interval exists, interval extends up to the start of next
            if j < n_rows:
                c_end = float(r_chs[j])
            raw_intervals.append({
                "ch_start": c_start,
                "ch_end": c_end,
                "width": w_val,
            })
            i = j

        # Filter out short ROW width changes (do not change ROW width for segments < min_row_interval_length_m, e.g. 100m)
        min_len = self.min_row_interval_length_m
        intervals: list[dict[str, Any]] = []
        if raw_intervals:
            # Pass 1: Forward merge intervals shorter than min_len
            pass1_intervals: list[dict[str, Any]] = []
            for iv in raw_intervals:
                if not pass1_intervals:
                    pass1_intervals.append(dict(iv))
                    continue
                prev = pass1_intervals[-1]
                iv_len = iv["ch_end"] - iv["ch_start"]
                if iv_len < min_len:
                    # Merge with previous interval, adopting the wider width
                    prev["ch_end"] = iv["ch_end"]
                    prev["width"] = max(prev["width"], iv["width"])
                else:
                    if abs(iv["width"] - prev["width"]) < 1e-3:
                        prev["ch_end"] = iv["ch_end"]
                    else:
                        pass1_intervals.append(dict(iv))

            # Pass 2: Backward cleanup if the first interval is short
            pass2_intervals: list[dict[str, Any]] = []
            for iv in reversed(pass1_intervals):
                if not pass2_intervals:
                    pass2_intervals.append(dict(iv))
                    continue
                nxt = pass2_intervals[-1]
                iv_len = iv["ch_end"] - iv["ch_start"]
                if iv_len < min_len:
                    nxt["ch_start"] = iv["ch_start"]
                    nxt["width"] = max(nxt["width"], iv["width"])
                else:
                    if abs(iv["width"] - nxt["width"]) < 1e-3:
                        nxt["ch_start"] = iv["ch_start"]
                    else:
                        pass2_intervals.append(dict(iv))

            # Pass 3: Final contiguous merge
            final_intervals: list[dict[str, Any]] = []
            for iv in reversed(pass2_intervals):
                if not final_intervals:
                    final_intervals.append(dict(iv))
                elif abs(iv["width"] - final_intervals[-1]["width"]) < 1e-3:
                    final_intervals[-1]["ch_end"] = iv["ch_end"]
                else:
                    final_intervals.append(dict(iv))

            intervals = final_intervals

        # Render intervals intersecting [start_ch, end_ch]
        inter_intervals = [
            iv for iv in intervals
            if iv["ch_end"] >= start_ch and iv["ch_start"] <= end_ch
        ]

        if not inter_intervals:
            return bound_left_pts, bound_right_pts, transition_callouts

        for idx, iv in enumerate(inter_intervals):
            iv_s = max(start_ch, iv["ch_start"])
            iv_e = min(end_ch, iv["ch_end"])
            w = iv["width"]
            offset_scaled = w * scale_factor

            # Sample parallel segment along alignment
            n_steps = max(2, int(math.ceil((iv_e - iv_s) / sample_step_m)))
            seg_chs = np.linspace(iv_s, iv_e, n_steps)

            for c in seg_chs:
                px, py, h_rad = get_paper_pos(c)
                perp = h_rad + (math.pi / 2.0)
                cos_p, sin_p = math.cos(perp), math.sin(perp)

                pt_l = (px - offset_scaled * cos_p, py - offset_scaled * sin_p)
                pt_r = (px + offset_scaled * cos_p, py + offset_scaled * sin_p)

                # Avoid duplicate identical points
                if not bound_left_pts or math.hypot(pt_l[0] - bound_left_pts[-1][0], pt_l[1] - bound_left_pts[-1][1]) > 0.05:
                    bound_left_pts.append(pt_l)
                    bound_right_pts.append(pt_r)

            # Check if there is a step transition to next interval
            if idx < len(inter_intervals) - 1:
                next_iv = inter_intervals[idx + 1]
                w_next = next_iv["width"]
                if w_next != w:
                    c_step = iv_e
                    offset_next_scaled = w_next * scale_factor
                    px_s, py_s, h_s = get_paper_pos(c_step)
                    perp_s = h_s + (math.pi / 2.0)
                    cos_ps, sin_ps = math.cos(perp_s), math.sin(perp_s)

                    # Step 1: Point at old width (already at bound_left_pts[-1])
                    # Step 2: Point at new width at EXACT SAME station -> 90-degree jog!
                    pt_l_step = (px_s - offset_next_scaled * cos_ps, py_s - offset_next_scaled * sin_ps)
                    pt_r_step = (px_s + offset_next_scaled * cos_ps, py_s + offset_next_scaled * sin_ps)

                    bound_left_pts.append(pt_l_step)
                    bound_right_pts.append(pt_r_step)

                    # Record callout metadata for perpendicular dimensioning
                    transition_callouts.append({
                        "chainage": c_step,
                        "old_w": w,
                        "new_w": w_next,
                        "pos": (px_s, py_s),
                        "heading": h_s,
                        "perp": perp_s,
                        "max_w": max(w, w_next),
                        "min_w": min(w, w_next),
                    })

        return bound_left_pts, bound_right_pts, transition_callouts

    def compute_smooth_zonal_boundary_profile(
        self,
        chainages: np.ndarray,
        bridge_df: pd.DataFrame | None = None,
        profile_df: pd.DataFrame | None = None,
        row_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Compute smooth, straight-line RDSO zonal boundary profile.

        If Row_Excel.xlsx is configured or available, uses the exact 'Desired LA' column.
        Otherwise computes dynamic baseline + bridge splays.

        Args:
            chainages: Array of chainage values along alignment.
            bridge_df: Optional bridge list DataFrame.
            profile_df: Optional profile DataFrame (containing formation_level_m & elevation_m).
            row_df: Optional pre-loaded ROW DataFrame.

        Returns:
            np.ndarray: Array of smooth boundary offset values at each chainage.
        """
        chs = np.asarray(chainages, dtype=float)
        if chs.size == 0:
            return np.array([], dtype=float)

        # 1. Check Row_Excel.xlsx first if enabled
        if self.use_row_excel or row_df is not None:
            row_offsets = self.compute_boundary_profile_from_row_excel(chs, row_df=row_df)
            if row_offsets is not None:
                return row_offsets

        offsets = np.full_like(chs, fill_value=self.default_boundary_offset_m)

        if chs.size == 0:
            return offsets

        # Process Bridge Zones from bridge_df if available
        if bridge_df is not None and not bridge_df.empty:
            ch_col = None
            for col in ["chainage_m", "chainage", "Chainage", "CHAINAGE", "Location", "CH"]:
                if col in bridge_df.columns:
                    ch_col = col
                    break

            if ch_col is not None:
                for _, row in bridge_df.iterrows():
                    try:
                        br_ch = float(row[ch_col])
                        # Check unit
                        unit = str(self.config.get("vertical", {}).get("bridge_chainage_unit", "km")).lower()
                        if not ch_col.endswith("_m") and unit == "km" and br_ch < 500.0:
                            br_ch *= 1000.0
                    except (ValueError, TypeError):
                        continue

                    # Determine span/bridge length
                    br_len = 10.0  # default 10m
                    for span_col in ["span_m", "total_span_m", "span_length_m", "Span", "SPAN"]:
                        if span_col in row and pd.notna(row[span_col]):
                            try:
                                br_len = max(5.0, float(row[span_col]))
                                break
                            except (ValueError, TypeError):
                                pass

                    # Structure zone limits with approach lead
                    z_start = br_ch - (br_len / 2.0) - self.approach_lead_m
                    z_end = br_ch + (br_len / 2.0) + self.approach_lead_m

                    # Evaluate height H at bridge from profile_df
                    h_val = 2.0  # default 2m fill if profile unavailable
                    if profile_df is not None and not profile_df.empty and "chainage_m" in profile_df.columns:
                        if "formation_level_m" in profile_df.columns and "elevation_m" in profile_df.columns:
                            fl_i = float(np.interp(br_ch, profile_df["chainage_m"].values, profile_df["formation_level_m"].values))
                            egl_i = float(np.interp(br_ch, profile_df["chainage_m"].values, profile_df["elevation_m"].values))
                            h_val = fl_i - egl_i
                        elif "ew_height_m" in profile_df.columns:
                            h_val = float(np.interp(br_ch, profile_df["chainage_m"].values, profile_df["ew_height_m"].values))

                    _, _, w_bridge = self.compute_boundary_offset(h_val)
                    if np.isnan(w_bridge) or w_bridge <= self.default_boundary_offset_m:
                        continue

                    # Calculate diagonal taper length (1:10 splay)
                    delta_w = w_bridge - self.default_boundary_offset_m
                    taper_len = max(self.min_taper_length_m, delta_w * self.taper_ratio)
                    tap_in_start = z_start - taper_len
                    tap_out_end = z_end + taper_len

                    # Apply to offset profile
                    for i, c in enumerate(chs):
                        if tap_in_start <= c < z_start:
                            # Entry taper (straight diagonal)
                            frac = (c - tap_in_start) / taper_len
                            w_val = self.default_boundary_offset_m + frac * delta_w
                            offsets[i] = max(offsets[i], w_val)
                        elif z_start <= c <= z_end:
                            # Inside bridge zone (uniform maximum)
                            offsets[i] = max(offsets[i], w_bridge)
                        elif z_end < c <= tap_out_end:
                            # Exit taper (straight diagonal)
                            frac = (tap_out_end - c) / taper_len
                            w_val = self.default_boundary_offset_m + frac * delta_w
                            offsets[i] = max(offsets[i], w_val)

        return offsets

    def compute_boundary_offset(
        self,
        height_m: float,
        track_center_offset: float | None = None,
    ) -> tuple[float, float, float]:
        """Compute (slope_spread_m, toe_distance_m, desired_land_boundary_m) for a given height H.

        Engineering logic:
            - If H > 0 (FILL) and H < threshold (6.0m):
                slope = 1 in 2 -> slope_spread = 2 * H
                clear_bottom_width = (bridge_top_width / 2) + slope_spread
                toe_dist = track_center_offset + clear_bottom_width
                desired_land = toe_dist + safety_buffer_low_m (3.0m)
            - If H >= threshold (6.0m) in FILL:
                slope = 1 in 2 -> slope_spread = 2 * H
                remaining_height = H - 6.0
                berm_spread = 2 * remaining_height (1:2 slope on remaining height)
                clear_bottom_width = (bridge_top_width / 2) + slope_spread + berm_spread
                toe_dist = track_center_offset + clear_bottom_width
                desired_land = toe_dist + safety_buffer_high_m (6.0m)
            - If H <= 0 (CUT) and |H| < 6.0m:
                slope = 1 in 1 -> slope_spread = 1 * |H|
                clear_bottom_width = (cutting_bottom_width / 2) + slope_spread
                toe_dist = track_center_offset + clear_bottom_width
                desired_land = toe_dist + cutting_drain_width_m (1.5m) + safety_buffer_low_m (3.0m)
            - If H <= 0 (CUT) and |H| >= 6.0m:
                slope = 1 in 1 -> base_slope_spread = 1 * |H|
                extra_offset = 3.0m
                remaining_depth = |H| - 6.0
                berm_spread = 1 * remaining_depth (1:1 slope on remaining depth)
                clear_bottom_width = (cutting_bottom_width / 2) + base_slope_spread + extra_offset + berm_spread
                toe_dist = track_center_offset + clear_bottom_width
                desired_land = toe_dist + cutting_drain_width_m (1.5m) + safety_buffer_low_m (3.0m)

        Args:
            height_m: Formation height difference (FL - EGL in meters).
            track_center_offset: Optional custom track center offset.

        Returns:
            tuple[float, float, float]: (slope_spread_m, toe_distance_m, desired_land_m)
        """
        tc = self.track_center_offset_m if track_center_offset is None else float(track_center_offset)
        if np.isnan(height_m):
            return np.nan, np.nan, np.nan

        if height_m > 0:  # FILL
            slope_ratio = self.bound_slope_fill
            base_slope_spread = slope_ratio * height_m
            
            if height_m < self.fill_height_threshold:
                clear_bottom_width = (self.bridge_top_width / 2.0) + base_slope_spread
                toe_dist = tc + clear_bottom_width
                desired_land = toe_dist + self.safety_buffer_low_m
                return base_slope_spread, toe_dist, desired_land
            else:
                # High embankment H >= 6m: Add berm spread on remaining height & use 6m buffer
                rem_height = height_m - self.fill_height_threshold
                berm_spread = self.bound_slope_berm * rem_height
                total_spread = base_slope_spread + berm_spread
                clear_bottom_width = (self.bridge_top_width / 2.0) + total_spread
                toe_dist = tc + clear_bottom_width
                desired_land = toe_dist + self.safety_buffer_high_m
                return total_spread, toe_dist, desired_land
        else:  # CUT
            abs_h = abs(height_m)
            slope_ratio = self.bound_slope_cut
            base_slope_spread = slope_ratio * abs_h
            
            if abs_h < self.cut_height_threshold:
                clear_bottom_width = (self.cutting_bottom_width / 2.0) + base_slope_spread
                toe_dist = tc + clear_bottom_width
                desired_land = toe_dist + self.cutting_drain_width_m + self.safety_buffer_low_m
                return base_slope_spread, toe_dist, desired_land
            else:
                # Deep cutting |H| >= 6m: add 3m offset + berm spread on remaining depth (1:1) + 1.5m drain + 3m safe buffer
                rem_depth = abs_h - self.cut_height_threshold
                berm_spread = self.cut_berm_slope_ratio * rem_depth
                total_spread = base_slope_spread + self.cut_high_extra_offset_m + berm_spread
                clear_bottom_width = (self.cutting_bottom_width / 2.0) + total_spread
                toe_dist = tc + clear_bottom_width
                desired_land = toe_dist + self.cutting_drain_width_m + self.safety_buffer_low_m
                return total_spread, toe_dist, desired_land

    def calculate(self, valign_result: VAlignResult) -> EarthworkResult:
        """Calculate earthwork parameters for the entire alignment.

        Args:
            valign_result: Output from Module 7 containing EGL and FL.

        Returns:
            EarthworkResult with appended profile DataFrame and volume summaries.

        Raises:
            EarthworkError: If calculation fails.
        """
        self.logger.info("Starting Earthwork calculations...")

        try:
            df = valign_result.profile_dataframe.copy()

            if not self.has_track_widths:
                raise EarthworkError(
                    f"Formation widths are not configured for track_type={self.track_type!r}; "
                    "set earthwork.formation_widths_by_track for this track type."
                )

            if "chainage_m" not in df.columns:
                raise EarthworkError("Required column (chainage_m) not found.")
            if "elevation_m" not in df.columns or "formation_level_m" not in df.columns:
                raise EarthworkError("Required columns (elevation_m, formation_level_m) not found.")

            chainages = pd.to_numeric(df.get("chainage_m"), errors="coerce").to_numpy(dtype=float)
            if len(chainages) < 2 or not np.all(np.isfinite(chainages)):
                raise EarthworkError("Earthwork requires at least two finite chainage values.")
            if np.any(np.diff(chainages) <= 0.0):
                raise EarthworkError("Earthwork chainages must be strictly increasing.")

            # Calculate height difference (H) = FL - EGL or adopt exact cut_fill_m from Excel
            # Positive = FILL (FL > EGL), Negative = CUT (FL < EGL). Use
            # supplied cut/fill values where present, and derive missing rows
            # from the profile instead of leaving gaps in the quantities.
            derived_height = df["formation_level_m"] - df["elevation_m"]
            if "cut_fill_m" in df.columns:
                df["ew_height_m"] = pd.to_numeric(
                    df["cut_fill_m"], errors="coerce"
                ).combine_first(derived_height)
            else:
                df["ew_height_m"] = derived_height

            # Determine type
            def get_ew_type(h: float) -> str:
                if np.isnan(h): return "UNKNOWN"
                if h > 0.001: return "FILL"
                if h < -0.001: return "CUT"
                return "LEVEL"

            df["ew_type"] = df["ew_height_m"].apply(get_ew_type)

            # Calculate Area A = (W * |H|) + (S * H^2)
            # Calculate Toe Offset from centerline = (W / 2) + (|H| * S)
            # Calculate Dynamic Railway Boundary (Desired Land) Offset
            areas = []
            offsets = []
            toe_distances = []
            desired_lands = []
            for h in df["ew_height_m"].values:
                if np.isnan(h):
                    areas.append(np.nan)
                    offsets.append(np.nan)
                    toe_distances.append(np.nan)
                    desired_lands.append(np.nan)
                    continue
                
                abs_h = abs(h)
                if h > 0:
                    fill_width = (
                        self.width_fill_high
                        if abs_h > self.fill_height_threshold
                        else self.width_fill
                    )
                    a = (fill_width * abs_h) + (self.slope_fill * (abs_h ** 2))
                    offset = (fill_width / 2.0) + (abs_h * self.slope_fill)
                else:
                    a = (self.width_cut * abs_h) + (self.slope_cut * (abs_h ** 2))
                    offset = (self.width_cut / 2.0) + (abs_h * self.slope_cut)
                areas.append(a)
                offsets.append(offset)

                _, toe_d, des_land = self.compute_boundary_offset(h)
                toe_distances.append(toe_d)
                desired_lands.append(des_land)
                
            df["ew_area_sqm"] = areas
            df["ew_toe_offset_m"] = offsets
            df["toe_distance_m"] = toe_distances
            df["desired_land_boundary_m"] = desired_lands

            # Calculate Volumes using Pyramid/Pyramidal Method for zero-crossing transitions
            volumes = [0.0]  # First point has 0 volume
            chainages = df["chainage_m"].values
            heights = df["ew_height_m"].values
            
            total_cut_vol = 0.0
            total_fill_vol = 0.0
            
            for i in range(1, len(df)):
                a1 = areas[i-1]
                a2 = areas[i]
                h1 = heights[i-1]
                h2 = heights[i]
                
                if np.isnan(a1) or np.isnan(a2):
                    volumes.append(np.nan)
                    continue
                
                l = chainages[i] - chainages[i-1]
                
                # Check if transition across zero height (Cut <-> Fill)
                if not np.isnan(h1) and not np.isnan(h2) and (h1 * h2 < 0) and (abs(h1) + abs(h2)) > 1e-6:
                    x0 = (abs(h1) / (abs(h1) + abs(h2))) * l
                    v1 = (a1 / 2.0) * x0
                    v2 = (a2 / 2.0) * (l - x0)
                    
                    if h1 > 0:  # i-1 is FILL, i is CUT
                        total_fill_vol += v1
                        total_cut_vol += v2
                    else:       # i-1 is CUT, i is FILL
                        total_cut_vol += v1
                        total_fill_vol += v2
                        
                    volumes.append(v1 + v2)
                else:
                    v = ((a1 + a2) / 2.0) * l
                    if (h2 > 0) or (h1 > 0):
                        total_fill_vol += v
                    elif (h2 < 0) or (h1 < 0):
                        total_cut_vol += v
                    volumes.append(v)

            df["ew_vol_cum"] = volumes

            self.logger.info(
                "Earthwork complete. Total Cut: %.1f m³, Total Fill: %.1f m³",
                total_cut_vol, total_fill_vol
            )

            meta = {
                "total_cut_vol_m3": float(total_cut_vol),
                "total_fill_vol_m3": float(total_fill_vol),
                "net_volume_m3": float(total_fill_vol - total_cut_vol) # Positive = borrow, Negative = spoil
            }

            return EarthworkResult(profile_dataframe=df, metadata=meta)

        except Exception as e:
            self.logger.error("Earthwork calculation failed: %s", str(e))
            raise EarthworkError(f"Earthwork failed: {e}") from e

    def export_csv(
        self,
        result: EarthworkResult,
        output_path: str | Path,
    ) -> Path:
        """Export the full earthwork profile to CSV.

        Args:
            result: EarthworkResult from calculate().
            output_path: Destination CSV file path.

        Returns:
            Resolved path to the written CSV file.
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        result.profile_dataframe.to_csv(out, index=False, float_format="%.4f")
        self.logger.info("Earthwork Data exported to CSV: %s", out)
        return out
