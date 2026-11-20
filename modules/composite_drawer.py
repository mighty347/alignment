"""Composite Alignment Plan & L-Section (Profile) Drawing Module for Indian Railways ESP Generator.

Unified composite engine combining:
  - Top Half: 2D Horizontal Alignment Plan (Module 9B: PlanDrawer)
      * Track centreline with true AutoCAD circular arcs and transition spirals
      * 100m perpendicular station tick marks and chainage callouts
      * Slanted leader structure callouts (bridges, RUBs, level crossings)
      * RDSO Curve Parameter Summary Table & Boxes
      * Directional junction arrows and North arrow
  - Bottom Half: Longitudinal Surface Elevation Profile (Module 9C: LSectionDrawer)
      * Existing Ground Level (EGL) & Proposed Formation Level (PFL) polylines
      * Red Datum baseline with dynamic elevation datum text (e.g. DATUM = -15m)
      * 10m interval vertical blue earth fill lines from datum to EGL
      * Vertical curve tags (VC-01, VC-02) and horizontal curve drop flags (C NO -01)
      * 10-Row Indian Railways Data Band Matrix (Top 6 Red, Bottom 4 White)

Pure Composition Architecture:
  CompositeDrawer coordinates PlanDrawer (top) + LSectionDrawer (bottom) directly.

Standards: RDSO / Indian Railways BG
Author   : Railway ESP Generator
"""

from __future__ import annotations

import math
import os
import sys
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Auto-configure GDAL_DATA and PROJ_LIB for Windows conda environments
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
from shapely.geometry import LineString

from modules.plan_drawer import PlanDrawer, PlanResult
from modules.lsection_drawer import LSectionDrawer, LSectionResult
from modules.esp_drawer import ESPDrawer
from utils.logger import get_logger

try:
    from template_models import LSectionSheetTemplate
except ImportError:
    try:
        from alignment.template_models import LSectionSheetTemplate
    except ImportError:
        LSectionSheetTemplate = None

logger = get_logger("CompositeDrawer")


class CompositeDrawerError(Exception):
    """Custom exception raised for Composite Plan & Profile drafting errors."""


@dataclass
class CompositeResult:
    """Container for Composite Alignment Plan & L-Section outputs."""

    dxf_path: Path
    dxf_paths: list[Path] = field(default_factory=list)
    pdf_path: Optional[Path] = None
    pdf_paths: list[Path] = field(default_factory=list)
    plan_result: Optional[PlanResult] = None
    lsection_result: Optional[LSectionResult] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CompositeDrawer:
    """Generates official Indian Railways unified Plan View + L-Section DXF drawings
    by directly coordinating PlanDrawer (Module 9B) and LSectionDrawer (Module 9C).
    """

    COL_RED = 1
    COL_YELLOW = 2
    COL_GREEN = 3
    COL_CYAN = 4
    COL_BLUE = 5
    COL_MAGENTA = 6
    COL_WHITE = 7
    COL_GREY = 8
    COL_BLACK = 7
    COL_PINK = 221

    def __init__(self, config: dict[str, Any], template: Optional[Any] = None) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

        # Resolve LSectionSheetTemplate
        self.template: Optional[Any] = None
        raw_tpl = template or config.get("template") or config.get("lsection_template")
        if raw_tpl is not None:
            if LSectionSheetTemplate is not None and isinstance(raw_tpl, LSectionSheetTemplate):
                self.template = raw_tpl
            elif LSectionSheetTemplate is not None and isinstance(raw_tpl, dict):
                try:
                    self.template = LSectionSheetTemplate.model_validate(raw_tpl)
                except Exception as tpl_err:
                    self.logger.warning("Failed to validate LSectionSheetTemplate: %s", tpl_err)
            else:
                self.template = raw_tpl

        paths_cfg = config.get("paths", {})
        self.outputs_dir = Path(paths_cfg.get("outputs_dir", "outputs"))
        self.comp_dir = Path(paths_cfg.get("composite_outputs_dir", str(self.outputs_dir / "composite")))
        self.comp_dir.mkdir(parents=True, exist_ok=True)

        drawing_cfg = config.get("drawing", {})
        self.generate_complete_profile: bool = bool(drawing_cfg.get("generate_complete_profile", True))
        self.generate_sheet_drawings: bool = bool(drawing_cfg.get("generate_sheet_drawings", True))
        self.export_pdf: bool = bool(drawing_cfg.get("export_pdf", True))
        self.pdf_dpi: int = int(drawing_cfg.get("pdf_dpi", 150))

        if self.template:
            self.h_scale: float = float(self.template.profile_viewport.h_scale)
            self.v_scale: float = float(self.template.profile_viewport.v_scale)
            self.plan_scale: float = float(self.template.plan_viewport.scale)
            tpl_max_len = self.template.get_available_drawing_length_m()
            self.max_sheet_length_m: float = float(drawing_cfg.get("max_lsection_length_m", tpl_max_len))
            self.config["template"] = self.template
        else:
            self.h_scale: float = float(drawing_cfg.get("profile_h_scale", 5000.0))
            self.v_scale: float = float(drawing_cfg.get("profile_v_scale", 500.0))
            self.plan_scale: float = float(drawing_cfg.get("plan_scale", 5000.0))
            self.max_sheet_length_m: float = float(drawing_cfg.get("max_lsection_length_m", 5000.0))

        # Specialized sub-drawers
        self.plan_drawer = PlanDrawer(self.config)
        self.lsection_drawer = LSectionDrawer(self.config)
        self.esp_drawer = ESPDrawer(self.config)

    def _setup_composite_document(self) -> tuple[ezdxf.document.Drawing, Any]:
        """Create and initialize a standard AutoCAD DXF document with all required layers."""
        doc = ezdxf.new(dxfversion="R2018")
        doc.units = units.MM
        doc.header["$MEASUREMENT"] = 1
        doc.header["$LTSCALE"] = 1.0
        doc.header["$CELTSCALE"] = 1.0
        doc.header["$PSLTSCALE"] = 0
        doc.header["$PLINEGEN"] = 1
        msp = doc.modelspace()

        # Line types
        if "CENTER" not in doc.linetypes:
            doc.linetypes.add("CENTER", description="Center ________  _  ________", pattern=[16.0, 10.0, -2.0, 2.0, -2.0])
        if "DASHED" not in doc.linetypes:
            doc.linetypes.add("DASHED", description="Dashed __ __ __ __", pattern=[4.0, 2.0, -2.0])
        if "RAILWAY_ROW" not in doc.linetypes:
            doc.linetypes.add(
                name="RAILWAY_ROW",
                pattern=[20.0, 14.0, -3.0, 0.0, -3.0],
                description="Railway Boundary Line ____ . ____ . ____",
            )
        if "POWER_LINE" not in doc.linetypes:
            doc.linetypes.add(
                name="POWER_LINE",
                pattern=[20.0, 12.0, -2.0, 0.0, -2.0, 0.0, -2.0],
                description="Power Line (HT/LT) ____ . . ____ . . ____",
            )
        if "DASHDOTDOT" not in doc.linetypes:
            doc.linetypes.add(
                name="DASHDOTDOT",
                pattern=[20.0, 12.0, -2.0, 0.0, -2.0, 0.0, -2.0],
                description="Dash Dot Dot Line ____ . . ____ . . ____",
            )

        # Layers
        layers = [
            # Profile & Band Layers (LSectionDrawer)
            ("PROFILE_EGL",       self.COL_BLACK,  25),
            ("PROFILE_EFL",       self.COL_BLACK,  20),
            ("PROFILE_ERL",       self.COL_BLACK,  20),
            ("PROFILE_FL",        self.COL_RED,    35),
            ("PROFILE_HFL",       self.COL_BLUE,   25),
            ("PROFILE_BED_LEVEL", self.COL_WHITE,  25),
            ("PROFILE_GRID",      self.COL_GREY,   13),
            ("PROFILE_VIP",       self.COL_RED,    18),
            ("DATA_BORDER",       self.COL_RED,    25),
            ("DATA_RED",          self.COL_RED,    13),
            ("DATA_BLACK",        self.COL_BLACK,  13),
            ("CHAINAGE",          self.COL_BLACK,  13),
            ("HGEOM",             self.COL_RED,    18),
            ("HGEOM_CURVE",       self.COL_RED,    13),
            ("VGEOM",             self.COL_RED,    18),
            ("TITLE_BLOCK",       self.COL_RED,    25),
            # Plan Layers (PlanDrawer)
            ("ALIGNMENT",           self.COL_RED,      35),  # PROPOSED MAIN LINE
            ("EXISTING_TRACK",      self.COL_BLACK,    35),  # EXISTING MAIN LINE
            ("FUTURE_TRACK",        self.COL_MAGENTA,  35),  # FUTURE LINE
            ("ALIGNMENT_GAUGE",     self.COL_GREY,     13),
            ("FORMATION_EDGE",      self.COL_GREY,     18),
            ("RAILWAY_BOUNDARY",    self.COL_WHITE,    25),  # PROP. LAND BOUNDARY (White)
            ("EXISTING_ROW",        self.COL_BLACK,    25),  # EXISTING LAND BOUNDARY
            ("LAND_ACQUISITION",    self.COL_RED,      18),  # PROP. LAND ACQUISITION
            ("CHAINAGE_TEXT",       self.COL_RED,      18),
            ("CURVE_CALLOUT",       self.COL_CYAN,     25),
            ("CURVE_TABLE",         self.COL_CYAN,     25),
            ("STRUCTURES_PLAN",     self.COL_RED,      18),
            ("VGEOM_PLAN",          self.COL_RED,      18),
            ("NORTH_ARROW",         self.COL_BLACK,    25),
            ("SHEET_BORDER",        self.COL_BLACK,    25),
            ("GRID",                self.COL_GREY,     9),
            ("TOPO_BOUNDARY",       self.COL_GREY,     18),
            ("TOPO_WATER",          self.COL_BLUE,     25),  # RIVER / WATERBODY
            ("TOPO_DRAIN",          self.COL_BLUE,     25),  # DRAIN
            ("TOPO_ROAD",           self.COL_BLACK,    25),  # PUCCA ROAD
            ("TOPO_KUTCHA_ROAD",    self.COL_MAGENTA,  18),  # KUTCHA ROAD
            ("TOPO_POWER_LINE",     self.COL_PINK,     18),  # HT / LT LINE (Dash Dot Dot in Pink)
            ("TOPO_BUILDING",       self.COL_WHITE,    25),  # HOUSE/BLD/STRUCTURE (White)
            ("TOPO_SURVEY",         self.COL_BLACK,    18),  # TBM
            ("TOPO_UTILITY",        self.COL_BLACK,    18),  # ELECTRIC POLE
            ("TOPO_DISMANTLE",      self.COL_GREEN,    25),  # DISMANTLE EXISTING WORK
            ("TOPO_RETAINING_WALL", self.COL_CYAN,     25),  # RETAINING WALL
            ("TOPO_CONTOUR",        self.COL_GREY,     13),
            ("TOPO_OVERLAY",        self.COL_GREY,     13),
            ("TOPO_TEXT",           self.COL_WHITE,    18),
            ("KML_OVERLAY",         self.COL_RED,      25),  # KML survey overlay (Dash-Dot-Dot in Red)
            ("KML_BOUNDARY",        self.COL_RED,      25),  # KML boundary line (Dash-Dot-Dot in Red)
            ("KML_TEXT",            self.COL_RED,      18),
        ]

        for name, col, lw in layers:
            if name not in doc.layers:
                if name == "RAILWAY_BOUNDARY":
                    lt = "RAILWAY_ROW"
                elif name == "TOPO_POWER_LINE":
                    lt = "POWER_LINE"
                elif name in ["KML_OVERLAY", "KML_BOUNDARY"]:
                    lt = "DASHDOTDOT"
                elif name in ["PROFILE_EFL", "PROFILE_ERL", "EXISTING_ROW", "TOPO_DISMANTLE", "TOPO_KUTCHA_ROAD"]:
                    lt = "DASHED"
                else:
                    lt = "Continuous"
                doc.layers.add(name=name, color=col, lineweight=lw, linetype=lt)

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

    def _apply_modelspace_scaling(self, msp: Any) -> None:
        """Apply global modelspace post-scaling so that 1 CAD unit = 1.0 ground metre."""
        drawing_cfg = self.config.get("drawing", {})
        auto_scale = bool(drawing_cfg.get("auto_scale_to_ground_meters", False))
        manual_factor = float(drawing_cfg.get("modelspace_scale_factor", 1.0))

        if auto_scale:
            scale_factor = self.h_scale / 1000.0
        else:
            scale_factor = manual_factor

        if abs(scale_factor - 1.0) > 1e-6:
            self.logger.info("Applying global modelspace scale factor: %.4f (1 CAD unit = 1.0 metre)...", scale_factor)
            from ezdxf.math import Matrix44
            mat = Matrix44.scale(scale_factor, scale_factor, scale_factor)
            for entity in list(msp):
                try:
                    entity.transform(mat)
                except Exception as exc:
                    self.logger.debug("Skipping scaling for entity %s: %s", entity.dxftype(), exc)

    def _export_to_pdf(self, dxf_path: Path, pdf_path: Path) -> Optional[Path]:
        """Export DXF drawing to vector PDF format using ezdxf matplotlib backend."""
        if not self.export_pdf:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"] = 42
            matplotlib.rcParams["pdf.compression"] = 6

            import matplotlib.pyplot as plt
            from ezdxf import bbox
            from ezdxf.addons.drawing import RenderContext, Frontend
            from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
            from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy

            self.logger.info("Exporting vector PDF for %s...", dxf_path.name)
            doc = ezdxf.readfile(str(dxf_path))
            msp = doc.modelspace()

            bb = bbox.extents(msp)
            min_x, min_y = float(bb.extmin.x), float(bb.extmin.y)
            max_x, max_y = float(bb.extmax.x), float(bb.extmax.y)
            w = max_x - min_x
            h = max_y - min_y

            if w <= 0.0 or h <= 0.0:
                self.logger.warning("Invalid extents for %s, skipping PDF export.", dxf_path.name)
                return None

            # Paper dimensions in inches (standard large-format drawing capped to 72 inches)
            aspect = (w / h) if h > 0 else 1.0
            if aspect >= 1.0:
                fig_w = min(72.0, max(24.0, 36.0 * aspect))
                fig_h = max(12.0, fig_w / aspect)
            else:
                fig_h = min(72.0, max(24.0, 36.0 / aspect))
                fig_w = max(12.0, fig_h * aspect)

            fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.pdf_dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect("equal", adjustable="box")

            draw_config = Configuration(
                background_policy=BackgroundPolicy.WHITE,
                color_policy=ColorPolicy.COLOR,
                min_lineweight=0.35,
            )

            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            try:
                Frontend(ctx, out, config=draw_config).draw_layout(msp, finalize=False)
            except Exception as e_draw:
                self.logger.warning("ezdxf layout draw notice for %s: %s", dxf_path.name, e_draw)

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path: Optional[Path] = None
            for try_suffix in ["", "_updated", "_new", "_v2"]:
                try_p = pdf_path if not try_suffix else pdf_path.with_stem(pdf_path.stem + try_suffix)
                try:
                    fig.savefig(str(try_p), format="pdf", dpi=self.pdf_dpi)
                    saved_path = try_p
                    self.logger.info("Successfully exported vector PDF drawing to %s", try_p.name)
                    break
                except PermissionError:
                    self.logger.warning("PDF file '%s' is locked. Trying next suffix...", try_p.name)

            if saved_path is None:
                import time
                ts_p = pdf_path.with_stem(f"{pdf_path.stem}_{int(time.time())}")
                fig.savefig(str(ts_p), format="pdf", dpi=self.pdf_dpi)
                saved_path = ts_p
                self.logger.info("Successfully exported vector PDF drawing to %s", ts_p.name)

            plt.close(fig)
            return saved_path

        except Exception as exc:
            self.logger.warning("PDF export failed for %s: %s", dxf_path.name, exc)
            return None

    def _compute_smart_sheet_splits(
        self,
        total_length_m: float,
        ip_df: Optional[pd.DataFrame] = None,
    ) -> list[tuple[float, float]]:
        """Compute sheet chainage ranges ensuring NO curve is split across two sheets."""
        max_len = self.max_sheet_length_m
        if max_len <= 0.0 or total_length_m <= max_len:
            return [(0.0, total_length_m)]

        chainage_step = float(self.config.get("alignment", {}).get("chainage_interval_m", 40.0))

        curve_intervals: list[tuple[float, float, int]] = []
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
                    curve_intervals.append((tptc, ctpt, int(r.get("ip_index", idx + 1))))

        splits: list[float] = [0.0]
        curr_start = 0.0
        while curr_start < total_length_m - 200.0:
            cand_end = curr_start + max_len
            if cand_end >= total_length_m - 200.0:
                splits.append(total_length_m)
                break

            conflict_curve = None
            for tptc, ctpt, c_idx in curve_intervals:
                if (tptc - chainage_step) < cand_end < (ctpt + chainage_step):
                    conflict_curve = (tptc, ctpt, c_idx)
                    break

            if conflict_curve is None:
                split_point = round(cand_end / chainage_step) * chainage_step
            else:
                tptc, ctpt, c_idx = conflict_curve
                # If extending to CTPT keeps sheet within 1.3x max_len (e.g. <= 6500m), include curve in current sheet
                if (ctpt + chainage_step - curr_start) <= max_len * 1.3:
                    split_point = math.ceil((ctpt + chainage_step) / chainage_step) * chainage_step
                    self.logger.info("Curve #%d (%.1f-%.1f m) overlaps sheet boundary: Extended sheet to Ch %.1f m to keep whole curve on sheet.", c_idx, tptc, ctpt, split_point)
                else:
                    split_point = math.floor((tptc - chainage_step) / chainage_step) * chainage_step
                    self.logger.info("Curve #%d (%.1f-%.1f m) overlaps sheet boundary: Split before curve at Ch %.1f m to keep whole curve on next sheet.", c_idx, tptc, ctpt, split_point)

            split_point = min(max(split_point, curr_start + 500.0), total_length_m)
            splits.append(split_point)
            curr_start = split_point

        ranges: list[tuple[float, float]] = []
        for i in range(len(splits) - 1):
            ranges.append((splits[i], splits[i + 1]))
        return ranges

    def draw_composite(
        self,
        align_res: Any,
        hgeom_res: Any,
        valign_res: Any,
        topo_res: Optional[Any] = None,
        bridge_df: Optional[Any] = None,
        output_prefix: str = "composite_plan_profile",
    ) -> CompositeResult:
        """Generate official Indian Railways Composite Plan & L-Section DXF drawings
        by coordinating PlanDrawer for top half and LSectionDrawer for bottom half.
        """
        self.logger.info("Starting Composite Plan & Profile DXF Generation...")

        try:
            # 1. Reconstruct mathematical curves using PlanDrawer
            raw_line = getattr(align_res, "alignment_line", None)
            ip_df_all = getattr(hgeom_res, "ip_dataframe", None) if hgeom_res is not None else None
            reconstructed_line = raw_line
            if raw_line is not None and ip_df_all is not None and not ip_df_all.empty:
                # reconstructed_line = self.plan_drawer.reconstruct_curved_alignment(raw_line, ip_df_all)
                align_res.alignment_line = reconstructed_line

            profile_df = None
            if valign_res is not None and hasattr(valign_res, "profile_dataframe"):
                profile_df = valign_res.profile_dataframe

            if profile_df is None or profile_df.empty or "chainage_m" not in profile_df.columns:
                raise CompositeDrawerError("Cannot generate Composite Plan: Profile DataFrame is missing or empty.")

            ch_all = profile_df["chainage_m"].values
            total_length_m = float(ch_all[-1]) if len(ch_all) > 0 else 0.0

            dxf_paths: list[Path] = []
            sheets_meta: list[dict[str, Any]] = []

            def _render_composite_chunk(start_ch: float, end_ch: float, sheet_index: int, total_sheets: int, dxf_sheet_path: Path) -> dict[str, Any]:
                mask = (profile_df["chainage_m"] >= start_ch - 0.01) & (profile_df["chainage_m"] <= end_ch + 0.01)
                sheet_prof_df = profile_df[mask].copy()
                if len(sheet_prof_df) == 0:
                    sheet_prof_df = profile_df.copy()

                ch_c = sheet_prof_df["chainage_m"].values
                egl_c = sheet_prof_df["elevation_m"].fillna(0).values if "elevation_m" in sheet_prof_df.columns else np.zeros(len(ch_c))
                fl_c = sheet_prof_df["formation_level_m"].fillna(0).values if "formation_level_m" in sheet_prof_df.columns else egl_c.copy()

                vip_c = pd.DataFrame()
                if valign_res is not None and hasattr(valign_res, "vip_dataframe") and valign_res.vip_dataframe is not None:
                    vip_c = valign_res.vip_dataframe.copy()

                ip_c = pd.DataFrame()
                if hgeom_res is not None and hasattr(hgeom_res, "ip_dataframe") and hgeom_res.ip_dataframe is not None:
                    ip_c = hgeom_res.ip_dataframe.copy()

                # Setup combined DXF document
                doc, msp = self._setup_composite_document()

                draw_length_m = max(1.0, float(ch_c[-1] - ch_c[0])) if len(ch_c) > 0 else 5000.0
                label_width = 80.0
                margin_sheet = 10.0
                sidebar_gap = 10.0
                row_height = float(self.config.get("drawing", {}).get("row_height_mm", 18.0))
                margin_bot = 12.0
                margin_mid = float(self.config.get("drawing", {}).get("margin_mid_mm", 55.0))
                margin_top = 10.0

                if self.template:
                    sheet_width = float(self.template.sheet_width_mm)
                    sheet_height = float(self.template.sheet_height_mm)
                    sidebar_width = float(self.template.sidebar_width_mm)
                    pv = self.template.profile_viewport
                    plv = self.template.plan_viewport

                    origin_x = pv.x
                    sidebar_x0 = sheet_width - sidebar_width - margin_sheet
                    sidebar_x1 = sheet_width - margin_sheet
                    draw_len_mm = max(100.0, sidebar_x0 - sidebar_gap - origin_x)

                    num_data_rows = len(self.template.get_enabled_bands()) if self.template.data_bands else 10
                    data_band_h = self.template.calculate_total_bands_height_mm() if self.template.data_bands else (num_data_rows * row_height)
                    data_origin_y = margin_bot + data_band_h
                    profile_origin_y = data_origin_y
                    profile_top_y = profile_origin_y + pv.h

                    plan_bot_y = profile_top_y + margin_mid
                    plan_top_y = sheet_height - margin_top
                    plan_area_h = plan_top_y - plan_bot_y
                    plan_origin_y = (plan_bot_y + plan_top_y) / 2.0
                else:
                    sheet_height = float(self.config.get("drawing", {}).get("sheet_height_mm", 915.0))
                    sidebar_width = float(self.config.get("drawing", {}).get("sidebar_width_mm", 153.0))
                    origin_x = margin_sheet + label_width + 10.0  # 100.0 mm

                    if self.h_scale > 0 and draw_length_m > 0:
                        draw_len_mm = (draw_length_m / self.h_scale) * 1000.0
                    else:
                        draw_len_mm = 1000.0

                    sidebar_x0 = origin_x + draw_len_mm + sidebar_gap
                    sidebar_x1 = sidebar_x0 + sidebar_width
                    sheet_width = sidebar_x1 + margin_sheet

                    num_data_rows = 10
                    data_band_h = num_data_rows * row_height        # 180.0 mm
                    profile_area_h = 120.0                          # 120.0 mm

                    data_origin_y = margin_bot + data_band_h        # 192.0 mm
                    profile_origin_y = data_origin_y                # 192.0 mm
                    profile_top_y = profile_origin_y + profile_area_h  # 310.0 mm
                    plan_bot_y = profile_top_y + margin_mid        # 325.0 mm
                    plan_top_y = sheet_height - margin_top          # 825.0 mm
                    plan_area_h = plan_top_y - plan_bot_y           # 500.0 mm
                    plan_origin_y = (plan_bot_y + plan_top_y) / 2.0  # 575.0 mm

                doc.header["$LIMMIN"] = (0.0, 0.0)
                doc.header["$LIMMAX"] = (sheet_width, sheet_height)
                doc.header["$EXTMIN"] = (0.0, 0.0, 0.0)
                doc.header["$EXTMAX"] = (sheet_width, sheet_height, 0.0)

                # Outer Sheet Borders
                msp.add_lwpolyline(
                    [(0, 0), (sheet_width, 0), (sheet_width, sheet_height), (0, sheet_height), (0, 0)],
                    dxfattribs={"layer": "SHEET_BORDER", "color": self.COL_BLACK, "lineweight": 25},
                )
                msp.add_lwpolyline(
                    [(10, 10), (sheet_width - 10, 10), (sheet_width - 10, sheet_height - 10), (10, sheet_height - 10), (10, 10)],
                    dxfattribs={"layer": "BORDER", "color": self.COL_RED, "lineweight": 50},
                )

                # Datum Calculation
                valid_egl = [e for e in egl_c if e != 0.0]
                start_egl = float(valid_egl[0]) if valid_egl else 0.0
                if self.template:
                    datum_elev = self.template.calculate_datum_elevation(start_egl)
                else:
                    datum_elev = math.floor((start_egl - 20.0) / 5.0) * 5.0

                # ── 1. Render Bottom L-Section Profile & 10 Data Bands via LSectionDrawer ──
                self.lsection_drawer.draw_profile_and_bands_onto_composite_sheet(
                    msp=msp,
                    chainages=ch_c,
                    egl=egl_c,
                    fl=fl_c,
                    origin_x=origin_x,
                    data_origin_y=data_origin_y,
                    profile_origin_y=profile_origin_y,
                    profile_top_y=profile_top_y,
                    draw_len_mm=draw_len_mm,
                    label_width=label_width,
                    datum_elev=datum_elev,
                    vip_df=vip_c,
                    ip_df=ip_c,
                    start_ch=start_ch,
                    profile_df=sheet_prof_df,
                    topo_res=topo_res,
                    bridge_df=bridge_df,
                    align_res=align_res,
                )

                # Determine alignment EPSG CRS
                align_epsg = "EPSG:32643"
                if align_res is not None and hasattr(align_res, "utm_info") and align_res.utm_info:
                    align_epsg = getattr(align_res.utm_info, "epsg_str", getattr(align_res.utm_info, "epsg", "EPSG:32643"))
                    if isinstance(align_epsg, int):
                        align_epsg = f"EPSG:{align_epsg}"
                elif align_res is not None and hasattr(align_res, "epsg_code"):
                    align_epsg = f"EPSG:{align_res.epsg_code}"

                # ── Section Heading: L - SECTION (centered horizontally in middle of clear margin gap) ──
                center_x = origin_x + draw_len_mm / 2.0
                y_lsec_title = (profile_top_y + plan_bot_y) / 2.0 + 10.0
                ul_half_w = 20.0

                msp.add_text(
                    "L - SECTION",
                    dxfattribs={
                        "height": 4.5,
                        "layer": "BORDER",
                        "style": "ESP_STYLE",
                        "color": self.COL_BLACK,
                    }
                ).set_placement((center_x, y_lsec_title), align=TextEntityAlignment.BOTTOM_CENTER)

                msp.add_line(
                    (center_x - ul_half_w, y_lsec_title - 1.2),
                    (center_x + ul_half_w, y_lsec_title - 1.2),
                    dxfattribs={
                        "layer": "BORDER",
                        "color": self.COL_BLACK,
                        "lineweight": 25,
                    }
                )

                # ── 2. Render Top 2D Plan View with Multi-Layer Geometry via PlanDrawer ──
                self.plan_drawer.draw_plan_onto_composite_sheet(
                    msp=msp,
                    reconstructed_line=reconstructed_line,
                    ip_df=ip_c,
                    start_ch=start_ch,
                    end_ch=end_ch,
                    origin_x=origin_x,
                    plan_origin_y=plan_origin_y,
                    plan_bot_y=plan_bot_y,
                    plan_top_y=plan_top_y,
                    draw_len_mm=draw_len_mm,
                    sheet_width=sheet_width,
                    sheet_height=sheet_height,
                    margin_right=sidebar_width + margin_sheet + sidebar_gap,
                    margin_top=margin_top,
                    bridge_df=bridge_df,
                    vip_df=vip_c,
                    valign_res=valign_res,
                    topo_res=topo_res,
                    epsg_code=align_epsg,
                )

                # ── 3. Render Standard 153mm Right Sidebar Panel ──
                self._draw_right_sidebar(
                    msp=msp,
                    x0=sidebar_x0,
                    x1=sidebar_x1,
                    y0=margin_sheet,
                    y1=sheet_height - margin_sheet,
                    sheet_index=sheet_index,
                    total_sheets=total_sheets,
                    start_ch=start_ch,
                    end_ch=end_ch,
                    ip_df=ip_c,
                    bridge_df=bridge_df,
                    epsg_code=align_epsg,
                    valign_res=valign_res,
                    align_res=align_res,
                    topo_res=topo_res,
                )

                # ── 4. Post-Scale Modelspace to True Ground Metres (1 CAD unit = 1.0 m) ──
                self._apply_modelspace_scaling(msp)

                saved = False
                final_path = dxf_sheet_path
                for try_suffix in ["", "_updated", "_new", "_v2", "_v3", "_v4"]:
                    try_path = dxf_sheet_path if not try_suffix else dxf_sheet_path.with_stem(dxf_sheet_path.stem + try_suffix)
                    try:
                        doc.saveas(str(try_path))
                        final_path = try_path
                        saved = True
                        break
                    except PermissionError:
                        self.logger.warning("File '%s' is open/locked. Trying next suffix...", try_path.name)
                if not saved:
                    import time
                    ts_path = dxf_sheet_path.with_stem(f"{dxf_sheet_path.stem}_{int(time.time())}")
                    doc.saveas(str(ts_path))
                    final_path = ts_path

                pdf_final_path: Optional[str] = None
                if self.export_pdf:
                    pdf_target = final_path.with_suffix(".pdf")
                    pdf_res = self._export_to_pdf(final_path, pdf_target)
                    if pdf_res and pdf_res.exists():
                        pdf_final_path = str(pdf_res)

                return {
                    "sheet_index": sheet_index,
                    "start_ch": start_ch,
                    "end_ch": end_ch,
                    "dxf_path": str(final_path),
                    "pdf_path": pdf_final_path,
                }

            sheet_dxf_paths: list[Path] = []
            sheet_pdf_paths: list[Path] = []
            # ── 1. Complete Continuous Composite Drawing (Full Length DXF) ───
            complete_dxf_path = self.comp_dir / f"{output_prefix}.dxf"
            should_gen_complete = self.generate_complete_profile or (self.max_sheet_length_m <= 0.0)

            if should_gen_complete:
                self.logger.info(
                    "Generating complete continuous Composite Plan & Profile (0.0 m to %.1f m)...",
                    total_length_m
                )
                complete_meta = _render_composite_chunk(
                    start_ch=0.0,
                    end_ch=total_length_m,
                    sheet_index=1,
                    total_sheets=1,
                    dxf_sheet_path=complete_dxf_path,
                )
                sheets_meta.append(complete_meta)
                if complete_meta.get("pdf_path"):
                    sheet_pdf_paths.append(Path(complete_meta["pdf_path"]))
                self.logger.info("Saved Complete Composite DXF: %s", complete_meta["dxf_path"])

            # ── 2. Multi-Sheet Chunked Drawings (Smart Curve-Aware Sheet Splits) ─────────
            should_gen_sheets = self.generate_sheet_drawings and (self.max_sheet_length_m > 0.0)
            if should_gen_sheets:
                ip_df = getattr(hgeom_res, "ip_dataframe", None) if hgeom_res is not None else None
                sheet_ranges = self._compute_smart_sheet_splits(total_length_m, ip_df=ip_df)
                num_sheets = len(sheet_ranges)
                for i, (start_ch, end_ch) in enumerate(sheet_ranges):
                    sheet_name = f"{output_prefix}_sheet{i+1}_km{start_ch/1000.0:.1f}-{end_ch/1000.0:.1f}"
                    dxf_sheet_path = self.comp_dir / f"{sheet_name}.dxf"

                    sheet_meta = _render_composite_chunk(
                        start_ch=start_ch,
                        end_ch=end_ch,
                        sheet_index=i + 1,
                        total_sheets=num_sheets,
                        dxf_sheet_path=dxf_sheet_path,
                    )
                    sheet_dxf_paths.append(Path(sheet_meta["dxf_path"]))
                    if sheet_meta.get("pdf_path"):
                        sheet_pdf_paths.append(Path(sheet_meta["pdf_path"]))
                    sheets_meta.append(sheet_meta)
                    self.logger.info("Saved Composite DXF sheet %d/%d: %s", i + 1, num_sheets, sheet_meta["dxf_path"])

            dxf_paths = sheet_dxf_paths if sheet_dxf_paths else [complete_dxf_path]
            primary_dxf = sheet_dxf_paths[0] if sheet_dxf_paths else complete_dxf_path
            primary_pdf = sheet_pdf_paths[0] if sheet_pdf_paths else None

            return CompositeResult(
                dxf_path=primary_dxf,
                dxf_paths=dxf_paths,
                pdf_path=primary_pdf,
                pdf_paths=sheet_pdf_paths,
                metadata={
                    "num_sheets": len(dxf_paths),
                    "total_length_m": total_length_m,
                    "sheets": sheets_meta,
                    "complete_dxf_path": str(complete_dxf_path) if should_gen_complete else None,
                    "pdf_paths": [str(p) for p in sheet_pdf_paths],
                },
            )

        except Exception as exc:
            self.logger.error("Composite Plan & Profile generation failed: %s", exc)
            raise CompositeDrawerError(f"Composite generation failed: {exc}") from exc

    def _draw_right_sidebar(
        self,
        msp: Any,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        sheet_index: int = 1,
        total_sheets: int = 1,
        start_ch: float = 0.0,
        end_ch: float = 0.0,
        ip_df: Optional[pd.DataFrame] = None,
        bridge_df: Optional[pd.DataFrame] = None,
        epsg_code: str = "EPSG:32645",
        valign_res: Optional[Any] = None,
        align_res: Optional[Any] = None,
        topo_res: Optional[Any] = None,
    ) -> None:
        """Render standard Indian Railways 153mm Right Sidebar Box, Authority of Works, and dynamic Notes."""
        color_name = str(self.config.get("drawing", {}).get("sidebar_color", "BLACK")).strip().upper()
        color_map = {
            "RED": self.COL_RED,         # 1
            "YELLOW": self.COL_YELLOW,   # 2
            "GREEN": self.COL_GREEN,     # 3
            "CYAN": self.COL_CYAN,       # 4
            "BLUE": self.COL_BLUE,       # 5
            "MAGENTA": self.COL_MAGENTA, # 6
            "BLACK": self.COL_BLACK,     # 7
            "WHITE": 7,
        }
        sb_color = color_map.get(color_name, self.COL_BLACK)
        sb_lineweight = int(self.config.get("drawing", {}).get("sidebar_lineweight", 35))

        # ── 1. Outer Sidebar Box Frame ───────────────────────────────────────
        msp.add_lwpolyline(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            dxfattribs={"layer": "BORDER", "color": sb_color, "lineweight": sb_lineweight},
        )

        notes_cfg = self.config.get("drawing", {}).get("sidebar_notes", {})
        if not notes_cfg.get("enabled", True):
            return

        # ── 2. Inner Frame & Geometry Coordinates ────────────────────────────
        pad_edge = 3.0
        bx0 = x0 + pad_edge
        bx1 = x1 - pad_edge
        tx = bx0 + 3.5  # Text left margin (~6.5 mm inside outer border)

        # ── 3. AUTHORITY OF WORKS Sub-Box ─────────────────────────────────────
        aow_top_y = y1 - pad_edge
        aow_h = 27.0
        aow_bot_y = aow_top_y - aow_h

        # Frame for Authority of Works
        msp.add_lwpolyline(
            [(bx0, aow_top_y), (bx1, aow_top_y), (bx1, aow_bot_y), (bx0, aow_bot_y), (bx0, aow_top_y)],
            dxfattribs={"layer": "BORDER", "color": sb_color, "lineweight": 25},
        )

        # Title: AUTHORITY OF WORKS
        y_aow_title = aow_top_y - 5.0
        t_aow = msp.add_text(
            "AUTHORITY OF WORKS",
            dxfattribs={
                "height": 3.2,
                "layer": "TITLE_BLOCK",
                "style": "ESP_STYLE",
                "color": self.COL_BLACK,
            },
        )
        t_aow.set_placement((tx, y_aow_title), align=TextEntityAlignment.BOTTOM_LEFT)

        # Underline under AUTHORITY OF WORKS
        aow_title_len = 48.0
        msp.add_line(
            (tx, y_aow_title - 0.8),
            (tx + aow_title_len, y_aow_title - 0.8),
            dxfattribs={"layer": "TITLE_BLOCK", "color": self.COL_BLACK, "lineweight": 18},
        )

        # Dynamic Project Metadata
        proj_cfg = self.config.get("project", {})
        pb_val = str(proj_cfg.get("pink_book_no", "") or "").strip()
        item_val = str(proj_cfg.get("item_no", "") or "").strip()
        proj_val = str(proj_cfg.get("project_id", "") or "").strip()

        row_h = 4.8
        y_r1 = y_aow_title - 1.5 - row_h
        y_r2 = y_r1 - row_h
        y_r3 = y_r2 - row_h

        msp.add_text(
            f"PINK BOOK NO.:-  {pb_val}" if pb_val else "PINK BOOK NO.:-",
            dxfattribs={"height": 2.6, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_BLACK},
        ).set_placement((tx, y_r1), align=TextEntityAlignment.BOTTOM_LEFT)

        msp.add_text(
            f"ITEM NO.         :-  {item_val}" if item_val else "ITEM NO.         :-",
            dxfattribs={"height": 2.6, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_BLACK},
        ).set_placement((tx, y_r2), align=TextEntityAlignment.BOTTOM_LEFT)

        msp.add_text(
            f"PROJECT ID      :-  {proj_val}" if proj_val else "PROJECT ID      :-",
            dxfattribs={"height": 2.6, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_BLACK},
        ).set_placement((tx, y_r3), align=TextEntityAlignment.BOTTOM_LEFT)

        # ── 4. NOTES : Header ────────────────────────────────────────────────
        y_curr = aow_bot_y - 7.0

        t_notes = msp.add_text(
            "NOTES :",
            dxfattribs={
                "height": 3.2,
                "layer": "TITLE_BLOCK",
                "style": "ESP_STYLE",
                "color": self.COL_BLACK,
            },
        )
        t_notes.set_placement((tx, y_curr), align=TextEntityAlignment.BOTTOM_LEFT)

        msp.add_line(
            (tx, y_curr - 0.8),
            (tx + 18.0, y_curr - 0.8),
            dxfattribs={"layer": "TITLE_BLOCK", "color": self.COL_BLACK, "lineweight": 18},
        )
        y_curr -= 6.0

        # ── 5. Dynamic Notes Generation ──────────────────────────────────────
        note_text_h = float(notes_cfg.get("text_height_mm", 2.6))
        line_spacing = float(notes_cfg.get("line_spacing_mm", 4.2))

        # Dynamic values resolution
        # Ruling Gradient
        ruling_disp = str(notes_cfg.get("ruling_gradient_display", "") or "").strip()
        if not ruling_disp:
            rg_val = float(self.config.get("vertical", {}).get("ruling_gradient", 0.006667))
            rg_ratio = round(1.0 / rg_val) if rg_val > 0 else 150
            ruling_disp = f"1 IN {rg_ratio}"

        # Origin station and chainage
        stn = str(notes_cfg.get("origin_station", "KANTI")).strip().upper()
        stn_ch = float(notes_cfg.get("origin_chainage_m", 0.0))

        # Speed
        spd = int(self.config.get("railway", {}).get("design_speed", 160))

        # IR Code correction date
        ir_date = str(notes_cfg.get("ir_code_correction_date", "27.04.2022")).strip()

        # VC criteria
        vc_crit = str(notes_cfg.get("vc_threshold_text", "4MM PER METER OR 0.4PERCENT")).strip()

        # Build list of notes (Use template clauses if configured, else default RDSO list)
        tpl_clauses = None
        if self.template and getattr(self.template, "sidebar", None):
            sb = self.template.sidebar
            if getattr(sb, "notes", None) and getattr(sb.notes, "clauses", None):
                tpl_clauses = sb.notes.clauses

        if tpl_clauses:
            notes_list: list[tuple[int, str]] = []
            import re
            for idx, clause in enumerate(tpl_clauses, start=1):
                clean_clause = str(clause).strip()
                clean_clause = re.sub(r"^\d+\.\s*", "", clean_clause)
                notes_list.append((idx, clean_clause))
        else:
            notes_list: list[tuple[int, str]] = [
                (1, "PROPOSED WORK SHOWN IN RED."),
                (2, "ALL DIMENSIONS SHOWN IN METER, UNLESS OTHER WISE SPECIFIED."),
                (3, f"RULING GRADIENT OF THE SECTION HAS BEEN KEPT AS {ruling_disp}."),
                (4, "ALL THE CURVE IN DIAGRAM HAS BEEN SHOWN BY INCORPORATING GRADE COMPENSATION."),
                (5, f"THIS DRAWING IS BASED ON SCALE AS PER IR ENGG. CODE 1982 CORRECTED UP TO {ir_date}."),
                (6, f"ALL CHAINAGE TAKEN FROM {stn} CONSIDERING {stn} STATION BUILDING CH:{stn_ch:.1f}."),
                (7, f"PROPOSED TRACK IS FIT FOR {spd} KMPH SPEED."),
                (8, f"TRL OF ALL THE PROPOSED CURVE HAS BEEN DESIGN FOR {spd}KMPH."),
                (9, "IT IS CERTIFIED THAT THE BED LEVEL OF PROPOSED LHS/RUB LIES ABOVE THE OBSERVED WATER LEVEL OF SITE."),
                (10, "NO PSR WILL BE IMPOSED IN THE SECTION ON ACCOUNT OF PROPOSED WORK ."),
                (11, f"VERTICAL CURVE HAS BEEN PROVIDED WHERE ALGEBRIC DIFFERENCE BETWEEN THE GRADES GREATER THAN {vc_crit}."),
            ]

        wrap_width = 68  # Characters before wrapping to fit ~138mm printable width

        for note_num, note_content in notes_list:
            prefix = f"{note_num}. " if note_num < 10 else f"{note_num}."
            full_text = f"{prefix}{note_content}"
            wrapped_lines = textwrap.wrap(full_text, width=wrap_width, subsequent_indent="   ")

            if note_num == 1:
                # Special styling: "RED." rendered in ACI 1 (Red)
                first_line = wrapped_lines[0]
                if "RED." in first_line:
                    idx = first_line.index("RED.")
                    part_before = first_line[:idx]
                    # Preceding text in standard text color (black/white)
                    msp.add_text(
                        part_before,
                        dxfattribs={
                            "height": note_text_h,
                            "layer": "TITLE_BLOCK",
                            "style": "ESP_STYLE",
                            "color": self.COL_BLACK,
                        },
                    ).set_placement((tx, y_curr), align=TextEntityAlignment.BOTTOM_LEFT)

                    offset_x = len(part_before) * (note_text_h * 0.65)
                    msp.add_text(
                        "RED.",
                        dxfattribs={
                            "height": note_text_h,
                            "layer": "TITLE_BLOCK",
                            "style": "ESP_STYLE",
                            "color": self.COL_RED,
                        },
                    ).set_placement((tx + offset_x, y_curr), align=TextEntityAlignment.BOTTOM_LEFT)
                else:
                    msp.add_text(
                        first_line,
                        dxfattribs={
                            "height": note_text_h,
                            "layer": "TITLE_BLOCK",
                            "style": "ESP_STYLE",
                            "color": self.COL_BLACK,
                        },
                    ).set_placement((tx, y_curr), align=TextEntityAlignment.BOTTOM_LEFT)
                y_curr -= line_spacing
                for sub_line in wrapped_lines[1:]:
                    msp.add_text(
                        sub_line,
                        dxfattribs={
                            "height": note_text_h,
                            "layer": "TITLE_BLOCK",
                            "style": "ESP_STYLE",
                            "color": self.COL_BLACK,
                        },
                    ).set_placement((tx, y_curr), align=TextEntityAlignment.BOTTOM_LEFT)
                    y_curr -= line_spacing
            else:
                for line in wrapped_lines:
                    msp.add_text(
                        line,
                        dxfattribs={
                            "height": note_text_h,
                            "layer": "TITLE_BLOCK",
                            "style": "ESP_STYLE",
                            "color": self.COL_BLACK,
                        },
                    ).set_placement((tx, y_curr), align=TextEntityAlignment.BOTTOM_LEFT)
                    y_curr -= line_spacing

            # Extra breathing space between notes
            y_curr -= 1.0

        # ── 6. Official Title Block Compartment (Bottom of Sidebar) ──────────
        tb_h = 56.0
        tb_bot_y = y0 + pad_edge
        tb_top_y = tb_bot_y + tb_h

        msp.add_lwpolyline(
            [(bx0, tb_bot_y), (bx1, tb_bot_y), (bx1, tb_top_y), (bx0, tb_top_y), (bx0, tb_bot_y)],
            dxfattribs={"layer": "TITLE_BLOCK", "color": self.COL_BLACK, "lineweight": 25},
        )

        banner_h = 13.0
        banner_div_y = tb_top_y - banner_h
        msp.add_line(
            (bx0, banner_div_y),
            (bx1, banner_div_y),
            dxfattribs={"layer": "TITLE_BLOCK", "color": self.COL_BLACK, "lineweight": 18},
        )

        tpl_tb = None
        if self.template and getattr(self.template, "sidebar", None):
            tpl_tb = getattr(self.template.sidebar, "title_block", None)

        zonal_name = str(
            getattr(tpl_tb, "zonal_railway", None)
            or self.config.get("project", {}).get("zone", "SOUTH CENTRAL RAILWAY")
        ).upper()
        msp.add_text(
            f"INDIAN RAILWAYS — {zonal_name}",
            dxfattribs={"height": 3.4, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_RED},
        ).set_placement(((bx0 + bx1) / 2.0, (tb_top_y + banner_div_y) / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

        proj_name = str(
            getattr(tpl_tb, "project_name", None)
            or self.config.get("project", {}).get("name", "NEW BROAD GAUGE RAILWAY LINE")
        ).upper()
        div_name = str(
            getattr(tpl_tb, "division", None)
            or self.config.get("project", {}).get("division", "HEADQUARTERS")
        ).upper()
        plan_title = str(
            getattr(tpl_tb, "drawing_title", None)
            or self.config.get("templates", {})
            .get("greenfield", {})
            .get("title_block", {})
            .get("plan_title", "WORKING PLAN AND LONGITUDINAL SECTION")
        ).upper()

        tb_items = [
            f"PROJECT  : {proj_name}",
            f"DIVISION : {div_name}",
            f"TITLE    : {plan_title}",
            f"SHEET NO : {sheet_index} OF {total_sheets} (KM {start_ch/1000.0:.2f} TO KM {end_ch/1000.0:.2f})",
            f"SCALES   : PLAN 1:{int(self.plan_scale)} | PROF H:1:{int(self.h_scale)}, V:1:{int(self.v_scale)}",
            f"NOTE     : FOR STATION YARDS & TURNOUTS REFER TO APPROVED STATION ESP",
        ]
        for idx, row_text in enumerate(tb_items):
            row_y = banner_div_y - 4.2 - (idx * 5.2)
            r_color = self.COL_RED if ("TITLE" in row_text or "NOTE" in row_text) else self.COL_BLACK
            msp.add_text(
                row_text,
                dxfattribs={"height": 2.2, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": r_color},
            ).set_placement((tx, row_y), align=TextEntityAlignment.MIDDLE_LEFT)

    def generate_from_inputs(self, output_prefix: str = "composite_plan_profile") -> dict[str, Any]:
        """Run the complete end-to-end Railway ESP pipeline to generate Composite DXF sheets."""
        from modules.read_kml import KMLReader
        from modules.generate_chainage import ChainageGenerator
        from modules.dem_reader import DEMReader
        from modules.elevation_sampler import ElevationSampler, EGLResult
        from modules.horizontal_geometry import HorizontalGeometry
        from modules.vertical_alignment import VerticalAlignment
        from modules.earthwork import Earthwork
        from modules.topo_analyzer import TopoAnalyzer
        from modules.alignment_validator import AlignmentValidator

        paths_cfg = self.config.get("paths", {})
        kml_p = Path(paths_cfg.get("kml_path", r"D:\ESP\OP_2_update.kml"))
        dem_p = Path(paths_cfg.get("dem_path", r"C:\Users\Parshant\Downloads\Kanthi_DTM_0.5M.tif"))
        survey_p_str = paths_cfg.get("survey_csv_path", paths_cfg.get("survey_levels_path"))
        survey_p = Path(survey_p_str) if survey_p_str else None
        lsec_p_str = paths_cfg.get("lsection_excel_path")
        lsec_p = Path(lsec_p_str) if lsec_p_str else None
        bridge_p_str = paths_cfg.get("bridge_list_path")
        bridge_p = Path(bridge_p_str) if bridge_p_str else None
        curve_excel_p = Path(paths_cfg.get("curve_excel_path", "")) if paths_cfg.get("curve_excel_path") else None

        self.logger.info("Step 1: Reading alignment from %s...", kml_p.name)
        kml_result = KMLReader(self.config).read(kml_p)

        self.logger.info("Step 2: Generating chainage stations along alignment...")
        chainage_result = ChainageGenerator(self.config).generate(kml_result)

        egl_df = chainage_result.dataframe.copy()
        egl_df["elevation_m"] = np.nan
        egl_df["formation_level_m"] = np.nan
        egl_df["existing_formation_level_m"] = np.nan
        egl_df["existing_rail_level_m"] = np.nan
        egl_result = EGLResult(dataframe=egl_df, metadata={"source": "auto", "min_elevation": 0.0, "max_elevation": 0.0})

        if lsec_p and lsec_p.exists():
            self.logger.info("Step 3: Loading ground elevations & formation levels from master Excel %s (Direct Excel Mode)...", lsec_p.name)
            s_df = pd.read_excel(lsec_p) if lsec_p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(lsec_p)
            s_cols = {str(c).strip().lower(): c for c in s_df.columns}
            ch_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["chainage", "ch", "station", "chainage (m)"])), None)
            gl_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["ground line", "ground level", "ground_level", "elevation_m", "ogl", "egl", "existing_ground_level"])), None)
            pfl_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["proposed formation level", "proposed_formation_level", "formation_level", "fl", "pfl"])), None)
            efl_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["existing formation level", "existing_formation_level", "efl", "ex_formation_level"])), None)
            erl_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["existing rail level", "existing_rail_level", "erl", "ex_rail_level"])), None)
            prl_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["proposed rail level", "proposed_rail_level", "prop rail level", "prl"])), None)
            cut_fill_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["bank/cutting", "bank cutting", "cut/fill", "cut_fill", "bank", "cutting"])), None)
            diff_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["dif. between", "diff between", "diff_rail", "difference"])), None)

            if ch_col is not None and len(s_df) >= 2:
                # Direct Excel Mode: Adopt exact Excel chainages as the master profile grid
                s_df = s_df.dropna(subset=[ch_col]).copy()
                s_df["_ch"] = pd.to_numeric(s_df[ch_col], errors="coerce")
                s_df = s_df.dropna(subset=["_ch"]).sort_values("_ch").reset_index(drop=True)

                master_df = pd.DataFrame()
                master_df["chainage_m"] = s_df["_ch"].astype(float)
                master_df["elevation_m"] = pd.to_numeric(s_df[gl_col], errors="coerce").astype(float) if gl_col else np.nan
                master_df["formation_level_m"] = pd.to_numeric(s_df[pfl_col], errors="coerce").astype(float) if pfl_col else np.nan
                master_df["proposed_rail_level_m"] = pd.to_numeric(s_df[prl_col], errors="coerce").astype(float) if prl_col else np.nan
                master_df["existing_formation_level_m"] = pd.to_numeric(s_df[efl_col], errors="coerce").astype(float) if efl_col else np.nan
                master_df["existing_rail_level_m"] = pd.to_numeric(s_df[erl_col], errors="coerce").astype(float) if erl_col else np.nan
                if cut_fill_col:
                    master_df["cut_fill_m"] = pd.to_numeric(s_df[cut_fill_col], errors="coerce").astype(float)
                if diff_col:
                    master_df["diff_rail_level_m"] = pd.to_numeric(s_df[diff_col], errors="coerce").astype(float)

                # Project spatial coordinates from KML chainage
                kml_df = chainage_result.dataframe
                if "easting_m" in kml_df.columns and "northing_m" in kml_df.columns:
                    kml_chs = kml_df["chainage_m"].values.astype(float)
                    master_df["easting_m"] = np.interp(master_df["chainage_m"].values, kml_chs, kml_df["easting_m"].values.astype(float))
                    master_df["northing_m"] = np.interp(master_df["chainage_m"].values, kml_chs, kml_df["northing_m"].values.astype(float))

                egl_result = EGLResult(dataframe=master_df, metadata={"source": "excel_direct", "file": lsec_p.name})
        elif dem_p and dem_p.exists():
            try:
                self.logger.info("Step 3: Sampling ground elevations from DEM %s...", dem_p.name)
                dem_result = DEMReader(self.config).read(dem_p, target_crs=kml_result.utm_info.epsg_str)
                egl_result = ElevationSampler(self.config).sample(chainage_result, dem_result)
                egl_result.dataframe["formation_level_m"] = np.nan
                egl_result.dataframe["existing_formation_level_m"] = np.nan
                egl_result.dataframe["existing_rail_level_m"] = np.nan
            except Exception as e_dem:
                self.logger.warning("Could not sample DEM (%s): %s", dem_p.name, e_dem)

        if survey_p and survey_p.exists() and (not lsec_p or not lsec_p.exists()):
            s_df = pd.read_excel(survey_p) if survey_p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(survey_p)
            s_cols = {str(c).strip().lower(): c for c in s_df.columns}
            ch_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["chainage", "ch", "station", "chainage (m)"])), None)
            efl_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["existing formation level", "existing_formation_level", "efl", "ex_formation_level"])), None)
            erl_col = next((col_name for col_key, col_name in s_cols.items() if any(sub in col_key for sub in ["existing rail level", "existing_rail_level", "erl", "ex_rail_level"])), None)
            if ch_col is not None:
                for _, r in s_df.iterrows():
                    c_val = float(r[ch_col])
                    idx_m = egl_result.dataframe[abs(egl_result.dataframe["chainage_m"] - c_val) < 1.0].index
                    if not idx_m.empty:
                        if efl_col is not None and pd.notna(r[efl_col]):
                            egl_result.dataframe.loc[idx_m[0], "existing_formation_level_m"] = float(r[efl_col])
                        if erl_col is not None and pd.notna(r[erl_col]):
                            egl_result.dataframe.loc[idx_m[0], "existing_rail_level_m"] = float(r[erl_col])

        if curve_excel_p and curve_excel_p.exists():
            self.logger.info("Step 4: Loading horizontal curve geometry from Excel schedule %s...", curve_excel_p.name)
            from modules.curve_excel_reader import CurveExcelReader
            hgeom_result = CurveExcelReader(self.config).read_curve_schedule(curve_excel_p, alignment_result=kml_result)
        else:
            self.logger.info("Step 4: Computing horizontal curve geometry from KML polyline...")
            hgeom_result = HorizontalGeometry(self.config).analyze(kml_result)

        self.logger.info("Step 5: Analyzing Topomap GIS layers for proposed bridges & RUBs...")
        topo_analyzer = TopoAnalyzer(self.config)
        topo_result = topo_analyzer.analyze(kml_result)
        export_csv_enabled = bool(self.config.get("paths", {}).get("export_csv", self.config.get("export_csv", False)))
        if export_csv_enabled:
            topo_analyzer.export_csv(topo_result, str(self.outputs_dir))

        bridge_df = None
        if bridge_p and bridge_p.exists():
            self.logger.info("Step 6b: Loading bridge list from %s...", bridge_p.name)
            bridge_df = pd.read_excel(bridge_p, sheet_name="Bridge List", header=0)
            bridge_df.columns = [str(c).strip() for c in bridge_df.columns]
            self.logger.info("Loaded %d raw bridge entries from %s", len(bridge_df), bridge_p.name)

        grad_p_str = paths_cfg.get("gradient_excel_path")
        grad_p = Path(grad_p_str) if grad_p_str else None
        self.logger.info("Step 6: Designing vertical profile (PFL & PRL)...")
        valign_result = VerticalAlignment(self.config).design(
            egl_result,
            hgeom_result,
            topo_result,
            survey_csv_path=survey_p if (survey_p is not None and survey_p.exists()) else None,
            bridge_df=bridge_df,
            gradient_excel_path=grad_p if (grad_p is not None and grad_p.exists()) else None,
        )

        self.logger.info("Step 7: Calculating cut/fill earthwork heights...")
        earthwork_result = Earthwork(self.config).calculate(valign_result)

        # Step 8: Validation skipped for now
        self.logger.info("Step 8: Validation skipped for now.")
        val_result = None
        val_csv_path = None

        if export_csv_enabled:
            VerticalAlignment(self.config).export_csv(valign_result, str(self.outputs_dir))
            HorizontalGeometry(self.config).export_csv(hgeom_result, str(self.outputs_dir / "horizontal_geometry.csv"))

        self.logger.info("Step 9: Drawing Composite Plan View & L-Section Profile CAD DXF...")
        comp_result = self.draw_composite(
            align_res=kml_result,
            hgeom_res=hgeom_result,
            valign_res=valign_result,
            topo_res=topo_result,
            bridge_df=bridge_df,
            output_prefix=output_prefix,
        )

        self.logger.info("========================================================")
        self.logger.info("  COMPOSITE PLAN & PROFILE GENERATION COMPLETE!         ")
        self.logger.info("========================================================")
        if len(comp_result.dxf_paths) > 1:
            self.logger.info("Generated %d Composite Sheets (max 5 km per sheet):", len(comp_result.dxf_paths))
            for idx, dxf_p in enumerate(comp_result.dxf_paths, 1):
                pdf_info = ""
                if idx - 1 < len(comp_result.pdf_paths):
                    pdf_info = f" | PDF: {comp_result.pdf_paths[idx - 1]}"
                self.logger.info("  - Sheet #%d DXF: %s%s", idx, dxf_p, pdf_info)
        else:
            self.logger.info("CAD DXF Drawing : %s", comp_result.dxf_path)
            if comp_result.pdf_path:
                self.logger.info("PDF Drawing     : %s", comp_result.pdf_path)

        if export_csv_enabled:
            self.logger.info("VIP Summary CSV : %s/vertical_ips.csv", self.outputs_dir)
            self.logger.info("Profile Data CSV : %s/chainage_formation_levels.csv", self.outputs_dir)
            if val_csv_path:
                self.logger.info("IRS/RDSO Validation : %s", val_csv_path)

        return {
            "composite_result": comp_result,
            "valign_result": valign_result,
            "hgeom_result": hgeom_result,
            "earthwork_result": earthwork_result,
            "topo_result": topo_result,
            "validation_result": val_result,
        }


if __name__ == "__main__":
    import yaml

    cfg_path = Path("config/config.yaml")
    if not cfg_path.exists():
        cfg_path = _ROOT / "config" / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    drawer = CompositeDrawer(cfg)
    drawer.generate_from_inputs(output_prefix="OP_2_update_composite")
