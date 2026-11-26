"""Longitudinal Section (L-Section) Drawing Module for Indian Railways ESP Generator.

Generates official Indian Railways 9-band Longitudinal Profile DXF drawings:
  - Profile Grid: EGL (Ground), PFL (Formation), PRL (Rail Level), and Datum Line
  - Band 1: Vertical Geometry (Gradients, VIPs, VC parabolic diagrams)
  - Band 2: Horizontal Geometry (Straight tangents, curves, RHS/LHS turn arrows, TPTC/CTPT)
  - Band 3: Cut (-)/Fill (+) Height (PFL - EGL)
  - Band 4: Diff PRL & ERL
  - Band 5: Proposed Rail Level (PRL = PFL + 0.762m)
  - Band 6: Proposed Formation Level (PFL)
  - Band 7: Existing Rail Level (ERL)
  - Band 8: Existing Formation Level (EFL)
  - Band 9: Existing Ground Level (EGL)
  - Chainage Stations every 20m / 100m
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Auto-configure GDAL_DATA and PROJ_LIB for Windows conda envs
if "GDAL_DATA" not in os.environ:
    gdal_cand = Path(sys.prefix) / "Library" / "share" / "gdal"
    if gdal_cand.exists():
        os.environ["GDAL_DATA"] = str(gdal_cand)

if "PROJ_LIB" not in os.environ:
    proj_cand = Path(sys.prefix) / "Library" / "share" / "proj"
    if proj_cand.exists():
        os.environ["PROJ_LIB"] = str(proj_cand)
        os.environ["PROJ_DATA"] = str(proj_cand)

import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
import numpy as np
import pandas as pd

from utils.logger import get_logger
from modules.brownfield.brownfield_profile_drawer import wrap_band_header_text

logger = get_logger("LSectionDrawer")


class LSectionDrawerError(Exception):
    """Custom exception raised for L-Section profile drafting errors."""


@dataclass
class LSectionResult:
    """Container for L-Section profile generation outputs."""

    dxf_path: Path
    dxf_paths: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LSectionDrawer:
    """Generates official Indian Railways 9-band L-Section DXF drawings."""

    # Standard AutoCAD Color Indices (ACI)
    COL_RED = 1
    COL_YELLOW = 2
    COL_GREEN = 3
    COL_CYAN = 4
    COL_BLUE = 5
    COL_MAGENTA = 6
    COL_WHITE = 7
    COL_GREY = 8
    COL_BLACK = 250  # ACI 250 = dark grey (closest to true black; ACI 7 = white)

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

        # Directories
        paths_cfg = config.get("paths", {})
        self.outputs_dir = Path(paths_cfg.get("outputs_dir", "outputs"))
        self.ls_dir = Path(paths_cfg.get("lsection_outputs_dir", str(self.outputs_dir / "lsection")))

        # Drawing configurations
        drawing_cfg = config.get("drawing", {})
        self.h_scale: float = float(drawing_cfg.get("profile_h_scale", 5000.0))
        self.v_scale: float = float(drawing_cfg.get("profile_v_scale", 500.0))
        self.max_sheet_length_m: float = float(drawing_cfg.get("max_lsection_length_m", 5000.0))
        self.generate_complete_profile: bool = bool(drawing_cfg.get("generate_complete_profile", True))
        self.generate_sheet_drawings: bool = bool(drawing_cfg.get("generate_sheet_drawings", True))
        self.chainage_interval_m: float = float(config.get("chainage", {}).get("interval_m", 20.0))

        # Vertical constants
        self.track_buildup_m: float = float(config.get("validation", {}).get("track_buildup_m", 0.762))

        # Band layout dimensions (paper mm) - strictly from config (no hardcoded fallback)
        custom_bands = drawing_cfg.get("profile_bands") or drawing_cfg.get("bands")
        self.band_names: list[tuple[str, float]] = []
        if custom_bands and isinstance(custom_bands, list):
            for item in custom_bands:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    self.band_names.append((str(item[0]), float(item[1])))
                elif isinstance(item, dict):
                    if not item.get("enabled", True):
                        continue
                    name = str(item.get("name", item.get("title", item.get("label", ""))))
                    h = float(item.get("height_mm", item.get("height", 18.0)))
                    self.band_names.append((name, h))
        self.total_bands_height_mm = sum(h for _, h in self.band_names)

        # Sheet layout constants (A0 Sheet: 1189 x 841 mm)
        self.sheet_width_mm: float = float(drawing_cfg.get("sheet_width_mm", 1189.0))
        self.sheet_height_mm: float = float(drawing_cfg.get("sheet_height_mm", 841.0))
        self.margin_mm: float = float(drawing_cfg.get("margin_mm", 10.0))

        # Unified chainage format (synchronized with Plan View)
        self.chainage_format: str = str(
            drawing_cfg.get("chainage_format")
            or drawing_cfg.get("plan_view", {}).get("chainage_format")
            or config.get("alignment", {}).get("chainage_format")
            or config.get("chainage", {}).get("chainage_format")
            or "CH-{km}+{m:03d}"
        )

    def _setup_document(self) -> tuple[ezdxf.document.Drawing, Any]:
        """Create and initialize a standard AutoCAD DXF document with RDSO L-Section layers."""
        doc = ezdxf.new(dxfversion="R2018")
        doc.units = units.M
        doc.header["$INSUNITS"] = units.M
        msp = doc.modelspace()

        # Define line styles
        if "PRL_DASHED" not in doc.linetypes:
            doc.linetypes.add(
                name="PRL_DASHED",
                pattern="A,10.0,-3.0,10.0,-3.0",
                description="Proposed Rail Level ____  ____",
            )

        # Standard RDSO L-Section Layers
        layers_def = [
            ("BORDER", self.COL_RED, 50),
            ("TITLE_BLOCK", self.COL_RED, 35),
            ("ESP_PROF_EGL", self.COL_GREEN, 25),
            ("ESP_PROF_PFL", self.COL_RED, 35),
            ("ESP_PROF_PRL", self.COL_RED, 25),
            ("ESP_PROF_DATUM", self.COL_BLACK, 25),
            ("ESP_PROF_GRID", self.COL_GREY, 9),
            ("ESP_PROF_BANDS", self.COL_BLACK, 25),
            ("ESP_PROF_TEXT", self.COL_BLACK, 18),
            ("ESP_GEOM_TEXT", self.COL_RED, 25),
            ("ESP_HGEOM_TEXT", self.COL_CYAN, 25),
        ]

        for name, color, weight in layers_def:
            if name not in doc.layers:
                doc.layers.add(name=name, color=color, lineweight=weight)

        font_name = str(self.config.get("drawing", {}).get("font_name", "arial.ttf"))
        font_width = float(self.config.get("drawing", {}).get("font_width_factor", 1.0))

        if "Standard" in doc.styles:
            std = doc.styles.get("Standard")
            std.dxf.font = font_name
            std.dxf.width = font_width

        if "ESP_STYLE" not in doc.styles:
            st = doc.styles.add("ESP_STYLE", font=font_name)
            st.dxf.width = font_width
        else:
            st = doc.styles.get("ESP_STYLE")
            st.dxf.font = font_name
            st.dxf.width = font_width

        return doc, msp

    def _compute_smart_sheet_splits(
        self,
        total_length_m: float,
        ip_df: Optional[pd.DataFrame] = None,
        bridge_df: Optional[pd.DataFrame] = None,
    ) -> list[tuple[float, float]]:
        """Compute sheet chainage ranges ensuring NO curve or bridge is split across two sheets."""
        max_len = self.max_sheet_length_m
        if max_len <= 0.0 or total_length_m <= max_len:
            return [(0.0, total_length_m)]

        chainage_step = float(self.config.get("alignment", {}).get("chainage_interval_m", 40.0))

        critical_intervals: list[tuple[float, float, str]] = []
        if ip_df is not None and not ip_df.empty:
            for idx, r in ip_df.iterrows():
                tptc = float(r.get("tptc_chainage_m", 0.0))
                ctpt = float(r.get("ctpt_chainage_m", 0.0))
                if tptc <= 0.0 and "chainage_m" in r and "tangent_length_m" in r:
                    ch = float(r["chainage_m"])
                    tl = float(r["tangent_length_m"])
                    tptc = max(0.0, ch - tl)
                    ctpt = ch + tl
                if ctpt > tptc and ctpt > 0:
                    critical_intervals.append((tptc, ctpt, f"Curve #{r.get('ip_index', idx + 1)}"))

        if bridge_df is not None and not bridge_df.empty:
            for b_idx, b_row in bridge_df.iterrows():
                b_ch = float(b_row.get("chainage_m", b_row.get("chainage", 0.0)))
                if b_ch > 0.0:
                    critical_intervals.append((b_ch - 30.0, b_ch + 30.0, f"Bridge #{b_row.get('bridge_no', b_idx + 1)}"))

        splits: list[float] = [0.0]
        curr_start = 0.0
        while curr_start < total_length_m - 200.0:
            cand_end = curr_start + max_len
            if cand_end >= total_length_m - 200.0:
                splits.append(total_length_m)
                break

            conflict = None
            for start_i, end_i, label in critical_intervals:
                if (start_i - chainage_step) < cand_end < (end_i + chainage_step):
                    conflict = (start_i, end_i, label)
                    break

            if conflict is None:
                split_point = round(cand_end / chainage_step) * chainage_step
            else:
                c_start, c_end, label = conflict
                if (c_end + chainage_step - curr_start) <= max_len * 1.3:
                    split_point = math.ceil((c_end + chainage_step) / chainage_step) * chainage_step
                    self.logger.info("%s (%.1f-%.1f m) overlaps sheet boundary: Extended sheet to Ch %.1f m to keep feature whole.", label, c_start, c_end, split_point)
                else:
                    split_point = math.floor((c_start - chainage_step) / chainage_step) * chainage_step
                    self.logger.info("%s (%.1f-%.1f m) overlaps sheet boundary: Split before feature at Ch %.1f m to keep whole feature on next sheet.", label, c_start, c_end, split_point)

            split_point = min(max(split_point, curr_start + 500.0), total_length_m)
            splits.append(split_point)
            curr_start = split_point

        ranges: list[tuple[float, float]] = []
        for i in range(len(splits) - 1):
            ranges.append((splits[i], splits[i + 1]))
        return ranges

    def draw_lsection(
        self,
        profile_df: pd.DataFrame,
        vip_df: Optional[pd.DataFrame] = None,
        ip_df: Optional[pd.DataFrame] = None,
        topo_res: Optional[Any] = None,
        bridge_df: Optional[Any] = None,
        output_prefix: str = "lsection_plan",
    ) -> LSectionResult:
        """Generate official Indian Railways 9-band L-Section DXF drawings."""
        self.logger.info("Starting L-Section 9-Band DXF Profile Generation...")
        self.ls_dir.mkdir(parents=True, exist_ok=True)

        if profile_df is None or profile_df.empty or "chainage_m" not in profile_df.columns:
            raise LSectionDrawerError("Cannot generate L-Section: Profile DataFrame is missing or empty.")

        from modules.esp_drawer import ESPDrawer
        esp_dr = ESPDrawer(self.config)

        ch_all = profile_df["chainage_m"].values
        total_length_m = float(ch_all[-1] - ch_all[0]) if len(ch_all) > 1 else 0.0

        sheet_dxf_paths: list[Path] = []
        sheets_meta: list[dict[str, Any]] = []

        # ── 1. Complete Continuous Profile (Full Length DXF) ───────────────
        complete_dxf_path = self.ls_dir / f"{output_prefix}.dxf"
        complete_pdf_path = self.ls_dir / f"{output_prefix}.pdf"
        should_gen_complete = self.generate_complete_profile or (self.max_sheet_length_m <= 0.0)

        if should_gen_complete:
            self.logger.info(
                "Generating complete continuous L-Section profile (0.0 m to %.1f m)...",
                total_length_m
            )
            egl_all = profile_df["elevation_m"].fillna(0).values if "elevation_m" in profile_df.columns else np.zeros(len(ch_all))
            fl_all = profile_df["formation_level_m"].fillna(0).values if "formation_level_m" in profile_df.columns else egl_all.copy()

            try:
                complete_meta = esp_dr._render_sheet(
                    chainages=ch_all,
                    egl=egl_all,
                    fl=fl_all,
                    dxf_out=complete_dxf_path,
                    pdf_out=complete_pdf_path,
                    vip_df=vip_df,
                    topo_res=topo_res,
                    ip_df=ip_df,
                    start_ch=0.0,
                    profile_df=profile_df,
                    bridge_df=bridge_df,
                )
            except PermissionError:
                alt_dxf = complete_dxf_path.with_stem(complete_dxf_path.stem + "_updated")
                alt_pdf = complete_pdf_path.with_stem(complete_pdf_path.stem + "_updated")
                self.logger.warning("File '%s' is open/locked. Saving to '%s' instead.", complete_dxf_path.name, alt_dxf.name)
                complete_meta = esp_dr._render_sheet(
                    chainages=ch_all,
                    egl=egl_all,
                    fl=fl_all,
                    dxf_out=alt_dxf,
                    pdf_out=alt_pdf,
                    vip_df=vip_df,
                    topo_res=topo_res,
                    ip_df=ip_df,
                    start_ch=0.0,
                    profile_df=profile_df,
                    bridge_df=bridge_df,
                )
                complete_dxf_path = alt_dxf

            sheets_meta.append(complete_meta)
            self.logger.info("Saved Complete L-Section DXF: %s", complete_dxf_path)

        # ── 2. Multi-Sheet Chunked Drawings (Smart Curve-Aware Sheet Splits) ─────────
        should_gen_sheets = self.generate_sheet_drawings and (self.max_sheet_length_m > 0.0)
        if should_gen_sheets:
            sheet_ranges = self._compute_smart_sheet_splits(total_length_m, ip_df=ip_df, bridge_df=bridge_df)
            num_sheets = len(sheet_ranges)
            for i, (start_ch, end_ch) in enumerate(sheet_ranges):

                mask = (profile_df["chainage_m"] >= start_ch - 0.01) & (profile_df["chainage_m"] <= end_ch + 0.01)
                sheet_prof_df = profile_df[mask].copy()
                if len(sheet_prof_df) == 0:
                    continue

                ch_c = sheet_prof_df["chainage_m"].values
                egl_c = sheet_prof_df["elevation_m"].fillna(0).values if "elevation_m" in sheet_prof_df.columns else np.zeros(len(ch_c))
                fl_c = sheet_prof_df["formation_level_m"].fillna(0).values if "formation_level_m" in sheet_prof_df.columns else egl_c.copy()

                sheet_vip_df = None
                if vip_df is not None and not vip_df.empty:
                    sheet_vip_df = vip_df[
                        (vip_df["chainage_m"] >= start_ch - 500.0) & (vip_df["chainage_m"] <= end_ch + 500.0)
                    ].copy()

                sheet_ip_df = None
                if ip_df is not None and not ip_df.empty:
                    sheet_ip_df = ip_df[
                        (ip_df["chainage_m"] >= start_ch - 1000.0) & (ip_df["chainage_m"] <= end_ch + 1000.0)
                    ].copy()

                sheet_name = f"{output_prefix}_sheet{i+1}_km{start_ch/1000.0:.1f}-{end_ch/1000.0:.1f}"
                dxf_sheet_path = self.ls_dir / f"{sheet_name}.dxf"
                pdf_sheet_path = self.ls_dir / f"{sheet_name}.pdf"

                try:
                    meta = esp_dr._render_sheet(
                        chainages=ch_c,
                        egl=egl_c,
                        fl=fl_c,
                        dxf_out=dxf_sheet_path,
                        pdf_out=pdf_sheet_path,
                        vip_df=sheet_vip_df,
                        topo_res=topo_res,
                        ip_df=sheet_ip_df,
                        start_ch=start_ch,
                        profile_df=sheet_prof_df,
                        bridge_df=bridge_df,
                    )
                except PermissionError:
                    alt_dxf = dxf_sheet_path.with_stem(dxf_sheet_path.stem + "_updated")
                    alt_pdf = pdf_sheet_path.with_stem(pdf_sheet_path.stem + "_updated")
                    self.logger.warning("File '%s' is open/locked. Saving to '%s' instead.", dxf_sheet_path.name, alt_dxf.name)
                    meta = esp_dr._render_sheet(
                        chainages=ch_c,
                        egl=egl_c,
                        fl=fl_c,
                        dxf_out=alt_dxf,
                        pdf_out=alt_pdf,
                        vip_df=sheet_vip_df,
                        topo_res=topo_res,
                        ip_df=sheet_ip_df,
                        start_ch=start_ch,
                        profile_df=sheet_prof_df,
                        bridge_df=bridge_df,
                    )
                    dxf_sheet_path = alt_dxf

                sheet_dxf_paths.append(dxf_sheet_path)
                sheets_meta.append(meta)
                self.logger.info("Saved L-Section DXF sheet %d/%d: %s", i + 1, num_sheets, dxf_sheet_path)

        dxf_paths = sheet_dxf_paths if sheet_dxf_paths else [complete_dxf_path]
        primary_dxf = sheet_dxf_paths[0] if sheet_dxf_paths else complete_dxf_path

        return LSectionResult(
            dxf_path=primary_dxf,
            dxf_paths=dxf_paths,
            metadata={
                "num_sheets": len(dxf_paths),
                "total_length_m": total_length_m,
                "sheets": sheets_meta,
                "complete_dxf_path": str(complete_dxf_path) if should_gen_complete else None,
            },
        )

    def draw_profile_and_bands_onto_composite_sheet(
        self,
        msp: Any,
        chainages: np.ndarray,
        egl: np.ndarray,
        fl: np.ndarray,
        origin_x: float,
        data_origin_y: float,
        profile_origin_y: float,
        profile_top_y: float,
        draw_len_mm: float,
        label_width: float,
        datum_elev: float,
        vip_df: Optional[pd.DataFrame] = None,
        ip_df: Optional[pd.DataFrame] = None,
        start_ch: float = 0.0,
        profile_df: Optional[pd.DataFrame] = None,
        topo_res: Optional[Any] = None,
        bridge_df: Optional[Any] = None,
        align_res: Optional[Any] = None,
        template: Optional[Any] = None,
    ) -> None:
        """Render the Profile & Data Bands onto the bottom section of a shared composite DXF sheet."""
        from modules.esp_drawer import ESPDrawer
        esp_dr = ESPDrawer(self.config, template=template)

        # 1. Profile View (EGL, PFL, Datum, 10m Blue earth fill lines)
        esp_dr._draw_profile_view(
            msp=msp,
            chainages=chainages,
            egl=egl,
            fl=fl,
            origin_x=origin_x,
            origin_y=profile_origin_y,
            datum_elev=datum_elev,
            draw_length_mm=draw_len_mm,
            label_width=label_width,
            vip_df=vip_df,
            align_res=align_res,
            topo_res=topo_res,
            ip_df=ip_df,
            start_ch=start_ch,
            bridge_df=bridge_df,
            profile_df=profile_df,
        )

        # 2. 10 Data Bands (Vertical Geom, Horizontal Geom, Cut/Fill, PRL, PFL, ERL, EFL, EGL, Chainage)
        esp_dr._draw_data_rows(
            msp=msp,
            chainages=chainages,
            egl=egl,
            fl=fl,
            origin_x=origin_x,
            origin_y=data_origin_y,
            draw_length_mm=draw_len_mm,
            label_width=label_width,
            max_y=profile_top_y,
            vip_df=vip_df,
            ip_df=ip_df,
            start_ch=start_ch,
            profile_df=profile_df,
        )

    def _render_lsection_sheet(
        self,
        msp: Any,
        prof_df: pd.DataFrame,
        vip_df: pd.DataFrame,
        ip_df: pd.DataFrame,
        start_ch: float,
        end_ch: float,
        sheet_index: int,
        total_sheets: int,
    ) -> None:
        """Render a single complete L-Section sheet with 9 data bands and vertical profile."""
        total_len_m = max(1.0, end_ch - start_ch)

        # 1. Datum Calculation
        egl_vals = prof_df["elevation_m"].fillna(0).values if "elevation_m" in prof_df.columns else np.zeros(len(prof_df))
        pfl_vals = prof_df["formation_level_m"].fillna(0).values if "formation_level_m" in prof_df.columns else egl_vals.copy()

        min_elev = min(np.min(egl_vals), np.min(pfl_vals)) if len(egl_vals) > 0 else 100.0
        max_elev = max(np.max(egl_vals), np.max(pfl_vals)) if len(egl_vals) > 0 else 150.0

        datum_elev = math.floor((min_elev - 5.0) / 10.0) * 10.0
        elev_range = max(15.0, (max_elev - datum_elev) + 8.0)

        # 2. Dimensions & Scaling in Model Space Metres
        sheet_w = max(5500.0, total_len_m + 700.0)
        sheet_h = max(3800.0, sheet_w * (841.0 / 1189.0))
        margin = sheet_w * (10.0 / 1189.0)
        anno_scale = sheet_w / 1189.0

        header_col_w = 400.0 * anno_scale
        origin_x = margin + header_col_w
        origin_y = margin + 120.0 * anno_scale

        # Vertical scale factor for profile
        v_exagg = 10.0  # Vertical exaggeration factor (1:5000 H / 1:500 V = 10x)

        # 3. Outer Sheet Frame & Title Block
        self._draw_sheet_border(msp, sheet_w, sheet_h, margin)
        self._draw_title_block(msp, sheet_index, total_sheets, start_ch, end_ch, sheet_w, sheet_h, margin, anno_scale)

        # 4. Draw Data Bands Matrix (Render only active bands with available data)
        has_erl = bool("existing_rail_level_m" in prof_df.columns and prof_df["existing_rail_level_m"].notna().any() and (prof_df["existing_rail_level_m"] != 0).any())
        has_efl = bool("existing_formation_level_m" in prof_df.columns and prof_df["existing_formation_level_m"].notna().any() and (prof_df["existing_formation_level_m"] != 0).any())

        has_vgeom = (vip_df is not None and not vip_df.empty) or (len(pfl_vals) > 0 and np.any(pfl_vals != 0.0))
        has_hgeom = (ip_df is not None and not ip_df.empty)
        has_cutfill = (len(egl_vals) > 0 and len(pfl_vals) > 0 and np.any(pfl_vals != 0.0))
        has_pfl = (len(pfl_vals) > 0 and np.any(pfl_vals != 0.0))
        has_egl = (len(egl_vals) > 0 and np.any(egl_vals != 0.0))

        availability_map = {
            "VERTICAL GEOMETRY": has_vgeom,
            "HORIZONTAL GEOMETRY": has_hgeom,
            "CUT (-) / FILL (+)": has_cutfill,
            "DIFF PRL & ERL": has_erl,
            "PROP RAIL LEVEL": has_pfl,
            "PROP. FORMATION LEVEL": has_pfl,
            "EX RAIL LEVEL": has_erl,
            "EX. FORMATION LEVEL": has_efl,
            "GROUND LEVEL": has_egl,
            "CHAINAGE (m)": len(prof_df) > 0,
        }

        active_bands: list[tuple[str, float]] = []
        for name, h_mm in self.band_names:
            norm_name = name.strip().upper()
            if norm_name in availability_map:
                if availability_map[norm_name]:
                    active_bands.append((name, h_mm))
            else:
                active_bands.append((name, h_mm))

        curr_y = origin_y
        band_y_positions = []

        for name, h_mm in reversed(active_bands):
            b_height = h_mm * anno_scale
            # Band header box
            msp.add_lwpolyline(
                [(origin_x - header_col_w, curr_y), (origin_x + total_len_m, curr_y),
                 (origin_x + total_len_m, curr_y + b_height), (origin_x - header_col_w, curr_y + b_height),
                 (origin_x - header_col_w, curr_y)],
                dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.COL_BLACK, "lineweight": 25},
            )
            # Band title text (intelligently wrapped to fit header_col_w)
            usable_w = max(20.0, header_col_w - 8.0 * anno_scale)
            char_w = max(1.2, 0.78 * 2.2 * anno_scale)
            max_chars = max(18, int(usable_w / char_w))
            lines = wrap_band_header_text(name, max_chars=max_chars)
            n_lines = len(lines)
            pitch = 2.2 * anno_scale * 1.3
            total_h = (n_lines - 1) * pitch
            mid_y = curr_y + b_height / 2.0
            y_start = mid_y + (total_h / 2.0)
            for l_idx, l_text in enumerate(lines):
                line_y = y_start - l_idx * pitch
                msp.add_text(
                    l_text,
                    dxfattribs={"height": 2.2 * anno_scale, "layer": "ESP_PROF_TEXT", "style": "ESP_STYLE", "color": self.COL_BLACK},
                ).set_placement((origin_x - header_col_w / 2.0, line_y), align=TextEntityAlignment.MIDDLE_CENTER)

            band_y_positions.append((name, curr_y, b_height))
            curr_y += b_height

        datum_y = curr_y

        # 5. Draw Profile Plot Grid & Elevation Ticks
        msp.add_line((origin_x, datum_y), (origin_x + total_len_m, datum_y), dxfattribs={"layer": "ESP_PROF_DATUM", "color": self.COL_BLACK, "lineweight": 35})
        msp.add_text(
            f"DATUM LINE = {datum_elev:.2f} m",
            dxfattribs={"height": 2.8 * anno_scale, "layer": "ESP_PROF_TEXT", "style": "ESP_STYLE", "color": self.COL_RED},
        ).set_placement((origin_x - 10.0 * anno_scale, datum_y), align=TextEntityAlignment.MIDDLE_RIGHT)

        # Horizontal Elevation Grid Lines (every 5m RL)
        elev_step = 5.0
        curr_el = datum_elev + elev_step
        while curr_el <= datum_elev + elev_range:
            grid_y = datum_y + (curr_el - datum_elev) * v_exagg
            msp.add_line((origin_x, grid_y), (origin_x + total_len_m, grid_y), dxfattribs={"layer": "ESP_PROF_GRID", "color": self.COL_GREY, "lineweight": 9})
            msp.add_text(
                f"{curr_el:.1f}m",
                dxfattribs={"height": 2.0 * anno_scale, "layer": "ESP_PROF_TEXT", "style": "ESP_STYLE", "color": self.COL_BLACK},
            ).set_placement((origin_x - 10.0 * anno_scale, grid_y), align=TextEntityAlignment.MIDDLE_RIGHT)
            curr_el += elev_step

        # 6. Station Columns & Vertical Grid Lines
        stn_chs = np.arange(start_ch, end_ch + 0.1, 100.0)
        for s_ch in stn_chs:
            stn_x = origin_x + (s_ch - start_ch)
            # Vertical grid through bands and profile
            msp.add_line((stn_x, origin_y), (stn_x, datum_y + elev_range * v_exagg), dxfattribs={"layer": "ESP_PROF_GRID", "color": self.COL_GREY, "lineweight": 9})

        # 7. Draw Profile Lines (EGL: continuous ground, PFL: breaks ONLY at gradient changes)
        egl_pts = []
        for _, r in prof_df.iterrows():
            ch = float(r.get("chainage_m", 0.0))
            if ch < start_ch or ch > end_ch:
                continue
            egl = float(r.get("elevation_m", datum_elev))
            px = origin_x + (ch - start_ch)
            py_egl = datum_y + (egl - datum_elev) * v_exagg
            egl_pts.append((px, py_egl))

        # PFL: breaks ONLY at gradient changes
        from modules.esp_drawer import ESPDrawer
        esp_dr = ESPDrawer(self.config)
        sheet_prof_mask = (prof_df["chainage_m"] >= start_ch - 0.01) & (prof_df["chainage_m"] <= end_ch + 0.01)
        sheet_prof = prof_df[sheet_prof_mask] if not prof_df[sheet_prof_mask].empty else prof_df
        ch_arr = sheet_prof["chainage_m"].values
        pfl_arr = sheet_prof["formation_level_m"].values if "formation_level_m" in sheet_prof.columns else sheet_prof["elevation_m"].values
        pfl_breaks = esp_dr._get_pfl_break_points(
            chainages=ch_arr,
            fl=pfl_arr,
            vip_df=vip_df,
            start_ch=start_ch,
            draw_length_mm=total_len_m,
        )
        pfl_pts = [
            (origin_x + (b_ch - start_ch), datum_y + (b_fl - datum_elev) * v_exagg)
            for b_ch, b_fl in pfl_breaks
        ]
        prl_pts = [
            (origin_x + (b_ch - start_ch), datum_y + (b_fl + self.track_buildup_m - datum_elev) * v_exagg)
            for b_ch, b_fl in pfl_breaks
        ]

        if len(egl_pts) >= 2:
            msp.add_lwpolyline(egl_pts, dxfattribs={"layer": "ESP_PROF_EGL", "color": self.COL_GREEN, "lineweight": 25})
        if len(pfl_pts) >= 2:
            for i in range(len(pfl_pts) - 1):
                msp.add_line(pfl_pts[i], pfl_pts[i + 1], dxfattribs={"layer": "ESP_PROF_PFL", "color": self.COL_RED, "lineweight": 35})
        if len(prl_pts) >= 2:
            for i in range(len(prl_pts) - 1):
                msp.add_line(prl_pts[i], prl_pts[i + 1], dxfattribs={"layer": "ESP_PROF_PRL", "color": self.COL_RED, "lineweight": 25, "linetype": "PRL_DASHED"})

        # 8. Populate Data Bands Values (Rotated 90 deg)
        sample_step = max(20.0, float(self.config.get("drawing", {}).get("lsection_label_interval_m", 100.0)))
        sample_chs = np.arange(start_ch, end_ch + 0.1, sample_step)

        # Mapping band names to y
        band_map = {name: (y, h) for name, y, h in band_y_positions}

        for s_ch in sample_chs:
            px = origin_x + (s_ch - start_ch)
            row_match = prof_df[(prof_df["chainage_m"] >= s_ch - 0.5) & (prof_df["chainage_m"] <= s_ch + 0.5)]

            if not row_match.empty:
                r0 = row_match.iloc[0]
                egl_val = float(r0.get("elevation_m", 0.0))
                pfl_val = float(r0.get("formation_level_m", egl_val))
            else:
                egl_val = float(np.interp(s_ch, prof_df["chainage_m"].values, egl_vals))
                pfl_val = float(np.interp(s_ch, prof_df["chainage_m"].values, pfl_vals))
            prl = pfl_val + self.track_buildup_m
            cut_fill = pfl_val - egl_val

            # Chainage Band (synchronized with Plan View chainage format)
            ch_band_key = next((k for k in band_map if "chainage" in k.lower()), None)
            if ch_band_key:
                by, bh = band_map[ch_band_key]
                from utils.geo_utils import format_chainage
                ch_label = format_chainage(s_ch, self.chainage_format)
                msp.add_text(
                    ch_label,
                    dxfattribs={"height": 2.0 * anno_scale, "rotation": 90.0, "layer": "ESP_PROF_TEXT", "style": "ESP_STYLE", "color": self.COL_BLACK},
                ).set_placement((px, by + bh / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

            # Ground Level Band
            if "GROUND LEVEL" in band_map:
                by, bh = band_map["GROUND LEVEL"]
                msp.add_text(
                    f"{egl_val:.1f}",
                    dxfattribs={"height": 2.0 * anno_scale, "rotation": 90.0, "layer": "ESP_PROF_TEXT", "style": "ESP_STYLE", "color": self.COL_BLACK},
                ).set_placement((px, by + bh / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

            # Proposed Formation Level
            if "PROP. FORMATION LEVEL" in band_map:
                by, bh = band_map["PROP. FORMATION LEVEL"]
                msp.add_text(
                    f"{pfl_val:.1f}",
                    dxfattribs={"height": 2.0 * anno_scale, "rotation": 90.0, "layer": "ESP_PROF_TEXT", "style": "ESP_STYLE", "color": self.COL_RED},
                ).set_placement((px, by + bh / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

            # Proposed Rail Level
            if "PROP RAIL LEVEL" in band_map:
                by, bh = band_map["PROP RAIL LEVEL"]
                msp.add_text(
                    f"{prl:.1f}",
                    dxfattribs={"height": 2.0 * anno_scale, "rotation": 90.0, "layer": "ESP_PROF_TEXT", "style": "ESP_STYLE", "color": self.COL_RED},
                ).set_placement((px, by + bh / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

            # Cut/Fill Band (Cut is negative: PFL < EGL, Fill is positive: PFL > EGL)
            if "CUT (-) / FILL (+)" in band_map:
                by, bh = band_map["CUT (-) / FILL (+)"]
                cf_str = f"+{cut_fill:.1f}" if cut_fill >= 0 else f"{cut_fill:.1f}"
                cf_col = self.COL_BLACK if cut_fill < 0 else self.COL_RED
                msp.add_text(
                    cf_str,
                    dxfattribs={"height": 2.0 * anno_scale, "rotation": 90.0, "layer": "ESP_PROF_TEXT", "style": "ESP_STYLE", "color": cf_col},
                ).set_placement((px, by + bh / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

        # 9. Draw Band 1: Vertical Geometry (Gradients, VIPs)
        if "VERTICAL GEOMETRY" in band_map and vip_df is not None and not vip_df.empty:
            by, bh = band_map["VERTICAL GEOMETRY"]
            self._draw_vertical_geometry_band(msp, vip_df, origin_x, by, bh, start_ch, end_ch, anno_scale)

        # 10. Draw Band 2: Horizontal Geometry (Curves, TPTC/CTPT)
        if "HORIZONTAL GEOMETRY" in band_map and ip_df is not None and not ip_df.empty:
            by, bh = band_map["HORIZONTAL GEOMETRY"]
            self._draw_horizontal_geometry_band(msp, ip_df, origin_x, by, bh, start_ch, end_ch, anno_scale)

    def _draw_vertical_geometry_band(
        self,
        msp: Any,
        vip_df: pd.DataFrame,
        origin_x: float,
        band_y: float,
        band_h: float,
        start_ch: float,
        end_ch: float,
        anno_scale: float,
    ) -> None:
        """Render gradient slope lines, change points, and slope text inside Band 1."""
        if vip_df is None or vip_df.empty:
            return

        # Normalize vip_df into discrete gradient segments: (ch1, ch2, grad, slope_ratio, direction)
        segments: list[dict[str, Any]] = []

        if "startChainage" in vip_df.columns and "endChainage" in vip_df.columns:
            for _, r in vip_df.sort_values("startChainage").iterrows():
                c1 = float(r.get("startChainage", 0.0))
                c2 = float(r.get("endChainage", 0.0))
                if c2 <= c1:
                    continue
                direction = str(r.get("direction", "")).strip().upper()
                slope_ratio = float(r.get("slopeRatio") or 0.0)

                # Compute gradient decimal
                if slope_ratio > 0.0:
                    grad = (1.0 / slope_ratio) if "FALL" not in direction else (-1.0 / slope_ratio)
                    if "LEVEL" in direction:
                        grad = 0.0
                elif "gradient_pct" in r and pd.notna(r["gradient_pct"]):
                    grad = float(r["gradient_pct"]) / 100.0
                elif "startRl" in r and "endRl" in r and pd.notna(r["startRl"]) and pd.notna(r["endRl"]):
                    grad = (float(r["endRl"]) - float(r["startRl"])) / (c2 - c1)
                else:
                    grad = float(r.get("gradient", r.get("slope", 0.0)))

                segments.append({
                    "ch1": c1,
                    "ch2": c2,
                    "grad": grad,
                    "slope_ratio": slope_ratio,
                    "direction": direction,
                })
        else:
            sorted_vips = vip_df.sort_values("chainage_m").to_dict("records")
            for idx in range(len(sorted_vips) - 1):
                v1 = sorted_vips[idx]
                v2 = sorted_vips[idx + 1]
                c1 = float(v1.get("chainage_m", 0.0))
                c2 = float(v2.get("chainage_m", 0.0))
                if c2 <= c1:
                    continue

                if "gradient_to_next_pct" in v1 and pd.notna(v1["gradient_to_next_pct"]):
                    grad = float(v1["gradient_to_next_pct"]) / 100.0
                elif "gradient_pct" in v1 and pd.notna(v1["gradient_pct"]):
                    grad = float(v1["gradient_pct"]) / 100.0
                elif "elevation_m" in v1 and "elevation_m" in v2:
                    grad = (float(v2["elevation_m"]) - float(v1["elevation_m"])) / (c2 - c1)
                else:
                    grad = float(v1.get("gradient", v1.get("slope", 0.0)))

                segments.append({
                    "ch1": c1,
                    "ch2": c2,
                    "grad": grad,
                    "slope_ratio": 0.0,
                    "direction": "",
                })

        for seg in segments:
            ch1 = seg["ch1"]
            ch2 = seg["ch2"]
            grad = seg["grad"]

            if ch2 <= start_ch or ch1 >= end_ch:
                continue

            s_ch = max(start_ch, ch1)
            e_ch = min(end_ch, ch2)
            px1 = origin_x + (s_ch - start_ch)
            px2 = origin_x + (e_ch - start_ch)
            seg_len = ch2 - ch1

            # Determine slope line endpoints inside the band
            if abs(grad) < 1e-5 or "LEVEL" in seg.get("direction", ""):
                py1 = band_y + band_h / 2.0
                py2 = py1
                lbl = f"LEVEL (L={seg_len:.0f}m)"
            elif grad > 0 or "RISE" in seg.get("direction", ""):
                py1 = band_y + 3.0 * anno_scale
                py2 = band_y + band_h - 3.0 * anno_scale
                if seg["slope_ratio"] > 0:
                    one_in = int(round(seg["slope_ratio"]))
                else:
                    one_in = int(round(1.0 / grad)) if grad != 0 else 0
                lbl = f"1 IN {one_in} RISE (L={seg_len:.0f}m)"
            else:
                py1 = band_y + band_h - 3.0 * anno_scale
                py2 = band_y + 3.0 * anno_scale
                if seg["slope_ratio"] > 0:
                    one_in = int(round(seg["slope_ratio"]))
                else:
                    one_in = int(round(1.0 / abs(grad))) if grad != 0 else 0
                lbl = f"1 IN {one_in} FALL (L={seg_len:.0f}m)"

            # Draw slope tangent line
            msp.add_line(
                (px1, py1), (px2, py2),
                dxfattribs={"layer": "ESP_GEOM_TEXT", "color": self.COL_RED, "lineweight": 25},
            )
            # Draw vertical segment boundary line at segment start
            if origin_x - 0.5 <= px1 <= origin_x + (end_ch - start_ch) + 0.5:
                msp.add_line(
                    (px1, band_y), (px1, band_y + band_h),
                    dxfattribs={"layer": "ESP_GEOM_TEXT", "color": self.COL_RED, "lineweight": 18},
                )
            # Draw segment label if space permits
            if (px2 - px1) > 25.0 * anno_scale:
                msp.add_text(
                    lbl,
                    dxfattribs={
                        "height": 2.2 * anno_scale,
                        "layer": "ESP_GEOM_TEXT",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement(((px1 + px2) / 2.0, band_y + band_h / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

    def _draw_horizontal_geometry_band(
        self,
        msp: Any,
        ip_df: pd.DataFrame,
        origin_x: float,
        band_y: float,
        band_h: float,
        start_ch: float,
        end_ch: float,
        anno_scale: float,
    ) -> None:
        """Render 3-compartment curve diagrams (TRL, R, CL) inside Band 2."""
        valid_curves = [r for _, r in ip_df.iterrows() if float(r.get("radius_m", 0)) > 0]
        valid_curves.sort(key=lambda x: float(x.get("tptc_chainage_m", x.get("chainage_m", 0))))

        curr_ch = start_ch
        mid_y = band_y + band_h / 2.0
        bot_y = band_y
        top_y = band_y + band_h

        for r in valid_curves:
            ip_ch = float(r.get("chainage_m", 0.0))
            rad = float(r.get("radius_m", 0.0))
            tcl = float(r.get("curve_length_m", 0.0))
            trl = float(r.get("transition_length_m", 0.0))
            ccl = float(r.get("circular_curve_length_m", 0.0))
            if ccl <= 0.0:
                ccl = max(1.0, tcl - 2.0 * trl)

            tptc = float(r.get("tptc_chainage_m", ip_ch - tcl / 2.0))
            tpcc = float(r.get("tpcc_chainage_m", tptc + trl))
            ctpc = float(r.get("ctpc_chainage_m", tpcc + ccl))
            ctpt = float(r.get("ctpt_chainage_m", ctpc + trl))

            # 1. Straight Section
            if curr_ch < tptc and tptc > start_ch:
                s_ch = max(start_ch, curr_ch)
                e_ch = min(end_ch, tptc)
                px1 = origin_x + (s_ch - start_ch)
                px2 = origin_x + (e_ch - start_ch)

                msp.add_line((px1, mid_y), (px2, mid_y), dxfattribs={"layer": "ESP_HGEOM_TEXT", "color": self.COL_RED, "lineweight": 18})
                if (px2 - px1) > 50.0 * anno_scale:
                    msp.add_text(
                        f"STRAIGHT (L={e_ch - s_ch:.0f}m)",
                        dxfattribs={"height": 2.0 * anno_scale, "layer": "ESP_HGEOM_TEXT", "style": "ESP_STYLE", "color": self.COL_RED},
                    ).set_placement(((px1 + px2) / 2.0, mid_y + 1.5 * anno_scale), align=TextEntityAlignment.MIDDLE_CENTER)

            # 2. Curve 3-Compartment Diagram
            if ctpt > start_ch and tptc < end_ch:
                x_tptc = origin_x + (tptc - start_ch)
                x_tpcc = origin_x + (tpcc - start_ch)
                x_ctpc = origin_x + (ctpc - start_ch)
                x_ctpt = origin_x + (ctpt - start_ch)

                # Vertical divider lines
                div_xs = [x_tptc, x_ctpt]
                if trl > 0.0:
                    div_xs.extend([x_tpcc, x_ctpc])

                for vx in div_xs:
                    if origin_x - 0.5 <= vx <= origin_x + (end_ch - start_ch) + 0.5:
                        msp.add_line((vx, bot_y), (vx, top_y), dxfattribs={"layer": "ESP_HGEOM_TEXT", "color": self.COL_RED, "lineweight": 18})

                # Direction check: RHS curves arch UPWARDS (+), LHS curves dip DOWNWARDS (-)
                direction = str(r.get("direction", r.get("turn_direction", r.get("turn", "RHS")))).strip().upper()
                is_lhs = ("L" in direction) or ("LEFT" in direction)
                h_deflect = max(1.5, (top_y - 0.6 * anno_scale) - mid_y if not is_lhs else mid_y - (bot_y + 0.6 * anno_scale))

                # 2. Render 3 Distinct Compartment Arcs (Convex Arching Geometry matching Indian Railways Legend)
                # (a) Entry Transition Arc (TPTC to TPCC)
                if trl > 0.0 and x_tpcc > x_tptc:
                    n_ent = max(15, int((x_tpcc - x_tptc) / 1.0))
                    xs_ent = np.linspace(x_tptc, x_tpcc, n_ent)
                    u_ent = (xs_ent - x_tptc) / (x_tpcc - x_tptc)
                    # Convex arching ramp: takes off directly from baseline mid_y, flattens at top
                    ramp = (2.0 * u_ent - u_ent ** 2) * h_deflect
                    ys_ent = mid_y + ramp if not is_lhs else mid_y - ramp
                    pts_ent = [(float(px), float(py)) for px, py in zip(xs_ent, ys_ent) if origin_x - 0.5 <= px <= origin_x + (end_ch - start_ch) + 0.5]
                    if len(pts_ent) >= 2:
                        msp.add_lwpolyline(pts_ent, dxfattribs={"layer": "ESP_HGEOM_TEXT", "color": self.COL_RED, "lineweight": 18})

                    # Text: TRL: ... (in opposite half of compartment)
                    x_ent = (x_tptc + x_tpcc) / 2.0
                    if origin_x <= x_ent <= origin_x + (end_ch - start_ch):
                        y_trl = (mid_y - (mid_y - bot_y) * 0.40) if not is_lhs else (mid_y + (top_y - mid_y) * 0.40)
                        msp.add_text(
                            f"TRL: {trl:.0f}",
                            dxfattribs={"height": 2.0 * anno_scale, "layer": "ESP_HGEOM_TEXT", "style": "ESP_STYLE", "color": self.COL_RED},
                        ).set_placement((x_ent, y_trl), align=TextEntityAlignment.MIDDLE_CENTER)

                # (b) Circular Curve Arc (TPCC to CTPC)
                c_start_x = x_tpcc if trl > 0.0 else x_tptc
                c_end_x   = x_ctpc if trl > 0.0 else x_ctpt
                if c_end_x > c_start_x:
                    n_circ = max(20, int((c_end_x - c_start_x) / 1.0))
                    xs_circ = np.linspace(c_start_x, c_end_x, n_circ)
                    x_mid = (c_start_x + c_end_x) / 2.0
                    half_chord = (c_end_x - c_start_x) / 2.0
                    u_circ = (xs_circ - x_mid) / half_chord
                    # Full circular curve arc (rising from mid_y up to peak and returning to mid_y)
                    circ_profile = np.maximum(0.0, 1.0 - u_circ ** 2) * h_deflect
                    ys_circ = mid_y + circ_profile if not is_lhs else mid_y - circ_profile
                    pts_circ = [(float(px), float(py)) for px, py in zip(xs_circ, ys_circ) if origin_x - 0.5 <= px <= origin_x + (end_ch - start_ch) + 0.5]
                    if len(pts_circ) >= 2:
                        msp.add_lwpolyline(pts_circ, dxfattribs={"layer": "ESP_HGEOM_TEXT", "color": self.COL_RED, "lineweight": 18})

                    # Circular text: R: ... (upper) and CL: ... (lower)
                    x_circ = (c_start_x + c_end_x) / 2.0
                    if origin_x <= x_circ <= origin_x + (end_ch - start_ch):
                        msp.add_text(
                            f"R: {rad:.1f}",
                            dxfattribs={"height": 2.0 * anno_scale, "layer": "ESP_HGEOM_TEXT", "style": "ESP_STYLE", "color": self.COL_RED},
                        ).set_placement((x_circ, mid_y + (top_y - mid_y) * 0.45), align=TextEntityAlignment.MIDDLE_CENTER)

                        msp.add_text(
                            f"CL: {ccl:.3f}",
                            dxfattribs={"height": 2.0 * anno_scale, "layer": "ESP_HGEOM_TEXT", "style": "ESP_STYLE", "color": self.COL_RED},
                        ).set_placement((x_circ, mid_y - (mid_y - bot_y) * 0.45), align=TextEntityAlignment.MIDDLE_CENTER)

                # (c) Exit Transition Arc (CTPC to CTPT)
                if trl > 0.0 and x_ctpt > x_ctpc:
                    n_ext = max(15, int((x_ctpt - x_ctpc) / 1.0))
                    xs_ext = np.linspace(x_ctpc, x_ctpt, n_ext)
                    u_ext = (xs_ext - x_ctpc) / (x_ctpt - x_ctpc)
                    # Convex arching descent: flattens at top, returns smoothly to baseline mid_y
                    ramp_ext = (1.0 - u_ext ** 2) * h_deflect
                    ys_ext = mid_y + ramp_ext if not is_lhs else mid_y - ramp_ext
                    pts_ext = [(float(px), float(py)) for px, py in zip(xs_ext, ys_ext) if origin_x - 0.5 <= px <= origin_x + (end_ch - start_ch) + 0.5]
                    if len(pts_ext) >= 2:
                        msp.add_lwpolyline(pts_ext, dxfattribs={"layer": "ESP_HGEOM_TEXT", "color": self.COL_RED, "lineweight": 18})

                    # Exit text: TRL: ... (in opposite half of compartment)
                    x_ext = (x_ctpc + x_ctpt) / 2.0
                    if origin_x <= x_ext <= origin_x + (end_ch - start_ch):
                        y_trl_ext = (mid_y - (mid_y - bot_y) * 0.40) if not is_lhs else (mid_y + (top_y - mid_y) * 0.40)
                        msp.add_text(
                            f"TRL: {trl:.0f}",
                            dxfattribs={"height": 2.0 * anno_scale, "layer": "ESP_HGEOM_TEXT", "style": "ESP_STYLE", "color": self.COL_RED},
                        ).set_placement((x_ext, y_trl_ext), align=TextEntityAlignment.MIDDLE_CENTER)

                curr_ch = max(curr_ch, ctpt)

        # Final Straight Section
        if curr_ch < end_ch:
            px1 = origin_x + (curr_ch - start_ch)
            px2 = origin_x + (end_ch - start_ch)
            msp.add_line((px1, mid_y), (px2, mid_y), dxfattribs={"layer": "ESP_HGEOM_TEXT", "color": self.COL_RED, "lineweight": 18})

    def _draw_title_block(
        self,
        msp: Any,
        sheet_index: int,
        total_sheets: int,
        start_ch: float,
        end_ch: float,
        sheet_w: float,
        sheet_h: float,
        margin: float,
        anno_scale: float,
    ) -> None:
        """Render standard Indian Railways L-Section Title Block."""
        w, h = 320.0 * anno_scale, 55.0 * anno_scale
        x = sheet_w - margin - w
        y = margin

        msp.add_lwpolyline(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
            dxfattribs={"layer": "TITLE_BLOCK", "color": self.COL_RED, "lineweight": 35},
        )
        msp.add_line((x, y + h - (14.0 * anno_scale)), (x + w, y + h - (14.0 * anno_scale)), dxfattribs={"layer": "TITLE_BLOCK", "color": self.COL_RED})

        msp.add_text(
            "INDIAN RAILWAYS \u2014 LONGITUDINAL PROFILE (L-SECTION)",
            dxfattribs={"height": 3.8 * anno_scale, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_RED},
        ).set_placement((x + w / 2.0, y + h - (7.0 * anno_scale)), align=TextEntityAlignment.MIDDLE_CENTER)

        proj_name = str(self.config.get("project", {}).get("name", "RAILWAY CORRIDOR DESIGN"))
        items = [
            f"PROJECT: {proj_name}",
            f"SHEET {sheet_index} OF {total_sheets} (KM {start_ch/1000.0:.2f} TO KM {end_ch/1000.0:.2f})",
            f"SCALES: H = 1:{int(self.h_scale)}, V = 1:{int(self.v_scale)} (10x EXAGGERATION)",
            f"GAUGE: BROAD GAUGE (1676mm) | STATUS: FINAL DESIGN",
        ]
        for idx, it in enumerate(items):
            msp.add_text(
                it,
                dxfattribs={"height": 2.4 * anno_scale, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_BLACK},
            ).set_placement((x + (5.0 * anno_scale), y + h - (20.0 * anno_scale) - idx * (7.5 * anno_scale)), align=TextEntityAlignment.MIDDLE_LEFT)

    def _draw_sheet_border(self, msp: Any, sheet_w: float, sheet_h: float, margin: float) -> None:
        """Draw standard ISO outer sheet boundary frame."""
        msp.add_lwpolyline(
            [(0, 0), (sheet_w, 0), (sheet_w, sheet_h), (0, sheet_h), (0, 0)],
            dxfattribs={"layer": "BORDER", "color": self.COL_RED, "lineweight": 18},
        )
        msp.add_lwpolyline(
            [(margin, margin), (sheet_w - margin, margin), (sheet_w - margin, sheet_h - margin), (margin, sheet_h - margin), (margin, margin)],
            dxfattribs={"layer": "BORDER", "color": self.COL_RED, "lineweight": 50},
        )

    def generate_from_inputs(
        self,
        kml_path: Optional[Path | str] = None,
        dem_path: Optional[Path | str] = None,
        survey_csv_path: Optional[Path | str] = None,
        lsection_excel_path: Optional[Path | str] = None,
        bridge_list_path: Optional[Path | str] = None,
        output_prefix: str = "lsection_plan",
    ) -> dict[str, Any]:
        """Execute the complete end-to-end Indian Railways L-Section generation pipeline.

        Performs:
          1. KML alignment parsing
          2. 20m station chainage generation
          3. DEM elevation sampling & survey level overrides (from Lsection_excel and survey_csv)
          4. Horizontal curve geometry analysis
          5. Topomap & Bridge list GIS analysis
          6. Vertical alignment design (PFL, PRL, VIPs, Grade Compensation)
          7. Cut/Fill earthwork calculations
          8. Matplotlib PNG & PDF profile plotting
          9. 9-Band AutoCAD DXF generation
          10. IRS/RDSO standard validation

        Returns:
            Dictionary containing all generated result objects and file paths.
        """
        from modules.read_kml import KMLReader
        from modules.generate_chainage import ChainageGenerator
        from modules.dem_reader import DEMReader
        from modules.elevation_sampler import ElevationSampler
        from modules.horizontal_geometry import HorizontalGeometry
        from modules.vertical_alignment import VerticalAlignment
        from modules.earthwork import Earthwork
        from modules.topo_analyzer import TopoAnalyzer
        from modules.longitudinal_profile import LongitudinalProfile
        from modules.alignment_validator import AlignmentValidator

        self.logger.info("========================================================")
        self.logger.info("   INDIAN RAILWAYS L-SECTION (PROFILE) GENERATOR        ")
        self.logger.info("========================================================")

        # 1. Resolve file paths from arguments or config
        cfg_paths = self.config.get("paths", {})
        kml_p = Path(kml_path or cfg_paths.get("kml_path", r"D:\ESP\OP_2_update.kml"))
        dem_p = Path(dem_path or cfg_paths.get("dem_path", r"C:\Users\Parshant\Downloads\Kanthi_DTM_0.5M.tif"))
        survey_p = Path(survey_csv_path or cfg_paths.get("survey_csv_path", "data/survey_levels.csv"))
        lsec_p_str = lsection_excel_path or cfg_paths.get("lsection_excel_path")
        lsec_p = Path(lsec_p_str) if lsec_p_str else None
        bridge_p = Path(bridge_list_path or cfg_paths.get("bridge_list_path", "D:/ESP/Bridge_List_OP2.xlsx"))

        if not kml_p.exists():
            raise FileNotFoundError(f"Alignment KML/KMZ file not found at: {kml_p.resolve()}")

        # 2. Read Alignment KML/KMZ
        self.logger.info("Step 1: Reading alignment from %s...", kml_p.name)
        kml_result = KMLReader(self.config).read(kml_p)

        # 3. Generate Chainage Stations (20m intervals)
        self.logger.info("Step 2: Generating 20m chainage stations...")
        chainage_result = ChainageGenerator(self.config).generate(kml_result)

        # 4. Load Ground Elevations (EGL) & Formation Levels — Primary Source: Lsection Excel
        from modules.elevation_sampler import EGLResult

        if lsec_p and lsec_p.exists():
            self.logger.info(
                "Step 3: Using %s as MASTER chainage + elevation source (bypassing KML stations)...",
                lsec_p.name,
            )
            lsec_df = (
                pd.read_excel(lsec_p)
                if lsec_p.suffix.lower() in [".xlsx", ".xls"]
                else pd.read_csv(lsec_p)
            )
            # Strip column names of whitespace for robust matching
            lsec_df.columns = [str(c).strip() for c in lsec_df.columns]
            lsec_cols_lower = {c.lower(): c for c in lsec_df.columns}

            # Detect chainage column
            ch_col = next(
                (lsec_cols_lower[k] for k in [
                    "chainage", "ch", "chainage_m", "station", "chainage (m)"
                ] if k in lsec_cols_lower),
                None,
            )
            # Detect Ground Level (EGL) column
            gl_col = next(
                (lsec_cols_lower[k] for k in [
                    "ground level", "ground_level", "elevation_m", "ogl", "egl",
                    "existing_ground_level", "ground level (m)"
                ] if k in lsec_cols_lower),
                None,
            )
            # Detect Proposed Formation Level (PFL) column
            pfl_col = next(
                (lsec_cols_lower[k] for k in [
                    "proposed formation level", "proposed_formation_level",
                    "formation_level", "fl", "pfl", "formation_level_m",
                    "proposed formation level (m)"
                ] if k in lsec_cols_lower),
                None,
            )

            self.logger.info(
                "  Detected columns → Chainage: '%s' | Ground Level: '%s' | PFL: '%s'",
                ch_col, gl_col, pfl_col,
            )

            if ch_col is None:
                raise LSectionDrawerError(
                    f"Cannot find a Chainage column in {lsec_p.name}. "
                    f"Available columns: {list(lsec_df.columns)}"
                )

            # ── Filter to strict regular chainage interval (e.g. 20m) ─────────
            # Removes irregular intermediate chainages (e.g. 962, 1076, 1547)
            interval_m = float(self.config.get("chainage", {}).get("interval_m", 20.0))
            ch_numeric = pd.to_numeric(lsec_df[ch_col], errors="coerce")
            rem = ch_numeric % interval_m
            is_regular = (np.minimum(rem, interval_m - rem) < 0.5) & ch_numeric.notna()
            num_irregular = len(lsec_df) - int(is_regular.sum())
            if num_irregular > 0:
                self.logger.info(
                    "  Filtered out %d irregular/intermediate chainage stations (retaining clean %.0fm intervals)",
                    num_irregular, interval_m
                )
            lsec_filtered = lsec_df[is_regular].copy().reset_index(drop=True)
            lsec_filtered[ch_col] = np.round(pd.to_numeric(lsec_filtered[ch_col]) / interval_m) * interval_m
            lsec_filtered = lsec_filtered.drop_duplicates(subset=[ch_col], keep="first").reset_index(drop=True)

            # ── Build egl_df DIRECTLY from the filtered Excel ─────────────────
            egl_df = pd.DataFrame()
            egl_df["chainage_m"] = pd.to_numeric(lsec_filtered[ch_col], errors="coerce")

            # Carry over any spatial columns from the KML chainage if column names match
            kml_df = chainage_result.dataframe
            spatial_cols = [c for c in ["easting_m", "northing_m", "latitude", "longitude"]
                            if c in kml_df.columns]
            if spatial_cols:
                # Nearest-neighbour join: for each Excel chainage find the closest KML station
                kml_ch = kml_df["chainage_m"].values
                for col in spatial_cols:
                    kml_vals = kml_df[col].values
                    egl_df[col] = egl_df["chainage_m"].apply(
                        lambda c: float(kml_vals[np.argmin(np.abs(kml_ch - c))])
                        if not np.isnan(c) else np.nan
                    )

            egl_df["elevation_m"] = (
                pd.to_numeric(lsec_filtered[gl_col], errors="coerce")
                if gl_col is not None else np.nan
            )
            egl_df["formation_level_m"] = (
                pd.to_numeric(lsec_filtered[pfl_col], errors="coerce")
                if pfl_col is not None else np.nan
            )
            egl_df["existing_formation_level_m"] = np.nan
            egl_df["existing_rail_level_m"] = np.nan

            # Drop rows where chainage itself is NaN (e.g. header rows in Excel)
            egl_df = egl_df.dropna(subset=["chainage_m"]).reset_index(drop=True)

            valid_elev = egl_df["elevation_m"].dropna()
            min_e = float(valid_elev.min()) if not valid_elev.empty else 0.0
            max_e = float(valid_elev.max()) if not valid_elev.empty else 0.0

            self.logger.info(
                "  Loaded %d chainage stations from Excel | EGL range: %.3f m – %.3f m",
                len(egl_df), min_e, max_e,
            )
            if gl_col is None:
                self.logger.warning("  Ground Level column not found — elevation_m will be NaN.")
            if pfl_col is None:
                self.logger.warning("  Proposed Formation Level column not found — formation_level_m will be NaN.")

            egl_result = EGLResult(
                dataframe=egl_df,
                metadata={
                    "source": lsec_p.name,
                    "min_elevation": min_e,
                    "max_elevation": max_e,
                    "num_stations": len(egl_df),
                    "ch_col": ch_col,
                    "gl_col": gl_col,
                    "pfl_col": pfl_col,
                },
            )

        elif lsec_p and not lsec_p.exists():
            raise FileNotFoundError(
                f"L-Section Excel file not found: {lsec_p.resolve()}. "
                f"Provide a valid 'lsection_excel_path' in config or as argument."
            )
        else:
            raise FileNotFoundError(
                f"No valid elevation source found. "
                f"L-Section Excel path not configured and DEM raster not found at: {dem_p.resolve()}"
            )

        # Step 3b: Load Existing Formation Level & Existing Rail Level from survey_levels.csv
        # Guard: egl_result must be assigned before accessing it
        if survey_p and survey_p.exists() and 'egl_result' in dir():
            self.logger.info("Step 3b: Loading Existing Formation & Rail Levels from %s...", survey_p.name)
            s_df = pd.read_excel(survey_p) if survey_p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(survey_p)
            s_cols = {str(c).strip().lower(): c for c in s_df.columns}
            ch_col_s = next((s_cols[k] for k in ["chainage", "ch", "chainage_m", "station", "chainage (m)"] if k in s_cols), None)
            efl_col = next((s_cols[k] for k in ["existing formation level", "existing_formation_level", "efl", "existing_formation_level_m", "ex_formation_level"] if k in s_cols), None)
            erl_col = next((s_cols[k] for k in ["existing rail level", "existing_rail_level", "erl", "existing_rail_level_m", "ex_rail_level"] if k in s_cols), None)

            if ch_col_s is not None:
                for _, r in s_df.iterrows():
                    c_val = float(r[ch_col_s])
                    idx_m = egl_result.dataframe[abs(egl_result.dataframe["chainage_m"] - c_val) < 1.0].index
                    if not idx_m.empty:
                        if efl_col is not None and pd.notna(r[efl_col]):
                            egl_result.dataframe.loc[idx_m[0], "existing_formation_level_m"] = float(r[efl_col])
                        if erl_col is not None and pd.notna(r[erl_col]):
                            egl_result.dataframe.loc[idx_m[0], "existing_rail_level_m"] = float(r[erl_col])

        # 5. Horizontal Geometry Analysis
        self.logger.info("Step 4: Computing horizontal curve geometry...")
        hgeom_result = HorizontalGeometry(self.config).analyze(kml_result)

        # 6. Topomap GIS & Bridge List Analysis
        self.logger.info("Step 5: Analyzing Topomap GIS layers for proposed bridges & RUBs...")
        topo_analyzer = TopoAnalyzer(self.config)
        topo_result = topo_analyzer.analyze(kml_result)
        export_csv_enabled = bool(self.config.get("paths", {}).get("export_csv", self.config.get("export_csv", False)))
        if export_csv_enabled:
            topo_analyzer.export_csv(topo_result, str(self.outputs_dir))

        bridge_df = None
        if bridge_p.exists():
            self.logger.info("Step 6b: Loading bridge list from %s...", bridge_p.name)
            bridge_df = pd.read_excel(bridge_p, sheet_name="Bridge List", header=0)
            bridge_df.columns = [str(c).strip() for c in bridge_df.columns]
            self.logger.info("Loaded %d raw bridge entries from %s", len(bridge_df), bridge_p.name)

        # 7. Production Vertical Alignment (PFL & PRL Design)
        grad_p = Path(self.config.get("paths", {}).get("gradient_excel_path", "D:/ESP-Backup/Graidient_excel.xlsx"))
        self.logger.info("Step 6: Designing vertical profile (PFL & PRL)...")
        valign_result = VerticalAlignment(self.config).design(
            egl_result,
            hgeom_result,
            topo_result,
            survey_csv_path=survey_p if survey_p.exists() else None,
            bridge_df=bridge_df,
            gradient_excel_path=grad_p if grad_p.exists() else None,
        )

        # 8. Earthwork Calculation (Cut/Fill Heights)
        self.logger.info("Step 7: Calculating cut/fill earthwork heights...")
        earthwork_result = Earthwork(self.config).calculate(valign_result)

        # Step 8: Validation skipped for now
        self.logger.info("Step 8: Validation skipped for now.")
        val_result = None
        val_csv_path = None

        if export_csv_enabled:
            VerticalAlignment(self.config).export_csv(valign_result, str(self.outputs_dir))
            HorizontalGeometry(self.config).export_csv(hgeom_result, str(self.outputs_dir / "horizontal_geometry.csv"))

        self.logger.info("Step 9: Drawing 9-Band L-Section CAD DXF...")
        ls_result = self.draw_lsection(
            profile_df=valign_result.profile_dataframe,
            vip_df=valign_result.vip_dataframe,
            ip_df=hgeom_result.ip_dataframe,
            topo_res=topo_result,
            bridge_df=bridge_df,
            output_prefix=output_prefix,
        )

        self.logger.info("========================================================")
        self.logger.info("  L-SECTION GENERATION COMPLETE SUCCESSFULLY!           ")
        self.logger.info("========================================================")
        if len(ls_result.dxf_paths) > 1:
            self.logger.info("Generated %d L-Section Sheets (max 5 km per sheet):", len(ls_result.dxf_paths))
            for idx, dxf_p in enumerate(ls_result.dxf_paths, 1):
                self.logger.info("  - Sheet #%d DXF : %s", idx, dxf_p)
        else:
            self.logger.info("CAD DXF Drawing : %s", ls_result.dxf_path)
        self.logger.info("VIP Summary CSV : %s/vertical_ips.csv", self.outputs_dir)
        self.logger.info("Profile Data CSV : %s/chainage_formation_levels.csv", self.outputs_dir)
        self.logger.info("IRS/RDSO Validation : %s", val_csv_path)

        return {
            "lsection_result": ls_result,
            "valign_result": valign_result,
            "hgeom_result": hgeom_result,
            "earthwork_result": earthwork_result,
            "topo_result": topo_result,
            "validation_result": val_result,
        }


if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    drawer = LSectionDrawer(cfg)
    drawer.generate_from_inputs(output_prefix="OP_2_update_lsection")


