"""
Module: modules/esp_drawer.py
Purpose: Module 9 — L-Section (Longitudinal Section / Profile of Alignment) DXF Generator.

Engineering Basis:
    Generates a complete Indian Railways standard Longitudinal Section drawing (L-Section)
    from the computed pipeline results. This is the primary output document for track design,
    showing the relationship between Existing Ground Level (EGL), Proposed Formation Level (PFL),
    Proposed Rail Level (PRL), and the resulting Cut/Fill earthwork.

    Formulas per IR_Track_Design_Formula_Reference.md:
        PRL = PFL + 762mm  (350mm ballast + 240mm PSC sleeper + 172mm 60kg rail, BG Main Line)
        Cut/Fill = PFL - EGL  (negative = cut, positive = fill)
        Compensated Gradient (%) = Ruling Gradient (%) - (7 / R)  [BG]
        Vertical Curve: L = R x Q  where Q = |g1-g2|/100 (radians)

    NOTE: Existing Rail Level (ERL) and Existing Formation Level (EFL) cannot be derived
    from DEM data — they require a field levelling survey. Those rows are rendered as
    empty (border only) until survey data is available.

Drawing Components:
    1. Profile View — EGL (black), PFL (red), Y-axis elevation scale, VIP diamond markers
    2. Data Band (10 rows top to bottom):
       Row 0: VERTICAL GEOMETRY     (Red) — grade blocks, L= length, VIP diamond separators
       Row 1: HORIZONTAL GEOMETRY   (Red) — arc curve per IP, Phi markers, curve labels
       Row 2: CUT (-) FILL (+)      (Red) — PFL - EGL at each 20m chainage
       Row 3: DIFF PRL & ERL        (Red) — empty (survey data required)
       Row 4: PROP RAIL LEVEL       (Red) — PFL + 0.762m
       Row 5: PROP. FORMATION LEVEL (Red) — PFL values
       Row 6: EX RAIL LEVEL         (Black)— empty (survey data required)
       Row 7: EX. FORMATION LEVEL   (Black)— empty (survey data required)
       Row 8: GROUND LEVEL          (Black)— EGL values from DEM
       Row 9: CHAINAGE              (Black)— Km X+YYY format every 20m

Input: Results from Modules 1, 6, 7, and 8.
Output: ESPResult (path to generated DXF).

Standards: RDSO / Indian Railways BG, IRPWM Ch. V
Reference: D:\\ESP\\IR_Track_Design_Formula_Reference.md
Author   : Railway ESP Generator
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import ezdxf
import geopandas as gpd
from shapely.geometry import box
from ezdxf import units
from ezdxf.enums import TextEntityAlignment

from modules.read_kml import AlignmentResult
from modules.horizontal_geometry import HGeomResult
from modules.vertical_alignment import VAlignResult
from modules.earthwork import Earthwork, EarthworkResult
from utils.logger import get_logger


logger = get_logger(__name__)

# Engineering Constants (IR BG Main Line per IR_Track_Design_Formula_Reference.md)
# PFL to PRL buildup: 350mm ballast + 240mm PSC sleeper + 172mm 60kg rail = 762mm
TRACK_BUILDUP_M: float = 0.762


class ESPDrawerError(Exception):
    """Custom exception for DXF drawing errors."""
    pass


@dataclass
class ESPResult:
    """Result object returned by ESPDrawer.

    Attributes:
        dxf_path: Path to the primary generated AutoCAD DXF file.
        dxf_paths: List of paths to all generated AutoCAD DXF sheet files.
        pdf_paths: List of paths to all generated PDF sheet files.
        metadata: Drawing metadata dictionary.
    """
    dxf_path: Path
    dxf_paths: list[Path] = field(default_factory=list)
    pdf_paths: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dxf_paths and self.dxf_path:
            self.dxf_paths = [self.dxf_path]


class ESPDrawer:
    """Generates the Indian Railways Longitudinal Section (L-Section) DXF drawing.

    Implements Module 9 of the ESP Generator pipeline.

    Args:
        config: Configuration dictionary loaded from config.yaml.
    """

    COL_RED    = 1
    COL_YELLOW = 2
    COL_GREEN  = 3
    COL_CYAN   = 4
    COL_BLUE   = 5
    COL_WHITE  = 7
    COL_GREY   = 8   # dark grey — grid lines
    COL_BLACK  = 7   # ACI 7 maps to pure solid black (0,0,0) on white paper in ezdxf

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

        self.outputs_dir = Path(config.get("paths", {}).get("outputs_dir", "outputs"))

        # Drawing scales per RDSO: H=1:5,000  V=1:500
        drawing_cfg = config.get("drawing", {})
        self.h_scale: float = float(
            drawing_cfg.get("profile_h_scale",
            drawing_cfg.get("plan_scale",
            drawing_cfg.get("h_scale", 5000.0)))
        )
        self.v_scale: float = float(
            drawing_cfg.get("profile_v_scale",
            drawing_cfg.get("v_scale", 500.0))
        )

        # Max alignment length per L-Section sheet (default: 5000.0 m / 5 km)
        self.max_lsection_length_m: float = float(
            drawing_cfg.get("max_lsection_length_m", 5000.0)
        )
        self.generate_complete_profile: bool = bool(drawing_cfg.get("generate_complete_profile", True))
        self.generate_sheet_drawings: bool = bool(drawing_cfg.get("generate_sheet_drawings", True))

        self.row_height: float = 18.0    # mm — height of each data row

        # Resolve dynamic template if provided
        self.template = None
        raw_tpl = config.get("template") or config.get("lsection_template")
        if raw_tpl is not None:
            if hasattr(raw_tpl, "data_bands"):
                self.template = raw_tpl
            elif isinstance(raw_tpl, dict):
                try:
                    from template_models import LSectionSheetTemplate
                    self.template = LSectionSheetTemplate.model_validate(raw_tpl)
                except Exception:
                    pass

        # Track buildup constant (config or IR standard 762mm)
        vert_cfg = config.get("vertical", {})
        self.track_buildup_m: float = float(
            vert_cfg.get("track_buildup_m", TRACK_BUILDUP_M)
        )

        # Plan View Spatial Bounding Box Ledger for Collision Avoidance
        self.placed_plan_boxes: list[tuple[float, float, float, float]] = []

        # Unified Chainage Format (synchronized with Plan View)
        self.chainage_format: str = str(
            drawing_cfg.get("chainage_format")
            or drawing_cfg.get("plan_view", {}).get("chainage_format")
            or config.get("alignment", {}).get("chainage_format")
            or config.get("chainage", {}).get("chainage_format")
            or "CH-{km}+{m:03d}"
        )


    # -------------------------------------------------------------------------
    # Layer Setup
    # -------------------------------------------------------------------------
    def _setup_layers(self, doc: Any) -> None:
        """Setup standard L-Section layers with RDSO colours/lineweights."""
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

        # Register standard CAD linetypes
        if "CENTER" not in doc.linetypes:
            doc.linetypes.add("CENTER", description="Center ________  _  ________", pattern=[16.0, 10.0, -2.0, 2.0, -2.0])
        if "DASHED" not in doc.linetypes:
            doc.linetypes.add("DASHED", description="Dashed __ __ __ __", pattern=[4.0, 2.0, -2.0])
        if "DOUBLE_DASH" not in doc.linetypes:
            doc.linetypes.add("DOUBLE_DASH", description="Double Dash __ __   __ __", pattern=[6.6, 2.0, -0.6, 2.0, -2.0])
        if "TRIPLE_DASH" not in doc.linetypes:
            doc.linetypes.add("TRIPLE_DASH", description="Triple Dash __ __ __   __ __ __", pattern=[8.7, 2.0, -0.5, 2.0, -0.5, 2.0, -2.5])
        if "QUAD_DASH" not in doc.linetypes:
            doc.linetypes.add("QUAD_DASH", description="Quad Dash ________   ________", pattern=[10.8, 2.0, -0.4, 2.0, -0.4, 2.0, -0.4, 2.0, -2.5])
        if "LONG_DASH" not in doc.linetypes:
            doc.linetypes.add("LONG_DASH", description="Long Dash ________   ________", pattern=[16.0, 12.0, -4.0])
        if "RAILWAY_ROW" not in doc.linetypes:
            doc.linetypes.add(
                name="RAILWAY_ROW",
                pattern=[24.0, 15.0, -3.0, 0.0, -3.0, 0.0, -3.0],
                description="Railway ROW ____ . . ____",
            )

        layers = [
            ("PROFILE_EGL",       self.COL_BLACK,  25),  # Existing Ground Level line (0.25mm black)
            ("PROFILE_EFL",       self.COL_BLACK,  20),  # Existing Formation Level line (0.20mm black)
            ("PROFILE_ERL",       self.COL_BLACK,  20),  # Existing Rail Level line (0.20mm black)
            ("PROFILE_FL",        self.COL_RED,    35),  # Proposed Formation Level line (0.35mm red)
            ("PROFILE_HFL",       self.COL_BLUE,   25),  # High Flood Level line (0.25mm blue)
            ("PROFILE_BED_LEVEL", self.COL_WHITE,  25),  # Bridge Bed Level line (0.25mm white)
            ("PROFILE_GRID",      self.COL_GREY,   13),  # Profile grid lines (0.13mm grey)
            ("PROFILE_VIP",       self.COL_RED,    18),  # VIP diamond markers (0.18mm red)
            ("DATA_BORDER",       self.COL_RED,    25),  # Data table row & cell borders (0.25mm red)
            ("DATA_RED",          self.COL_RED,    13),  # Red data text (0.13mm red)
            ("DATA_BLACK",        self.COL_BLACK,  13),  # Black data text (0.13mm black)
            ("CHAINAGE",          self.COL_BLACK,  13),  # Chainage text (0.13mm black)
            ("HGEOM",             self.COL_RED,    18),  # Horizontal geometry row lines & text
            ("HGEOM_CURVE",       self.COL_RED,    13),  # Curve parameter box border & text
            ("VGEOM",             self.COL_RED,    18),  # Vertical geometry row lines & text
            ("TITLE_BLOCK",       self.COL_RED,    25),  # Title block frame
            # ── L-Section legacy plan layers ──────────────────────────────────
            ("RAIL-CENTERLINE",   self.COL_RED,    25),
            ("EARTHWORK-TOE",     self.COL_GREEN,  13),
            ("PLAN_CURVES",       self.COL_CYAN,   25),
            ("ROW",               self.COL_BLACK,  13),
            ("ROAD-CENTERLINE",   self.COL_YELLOW, 13),
            ("BRIDGE",            self.COL_BLACK,  13),
            ("DRAIN",             self.COL_BLUE,   13),
            ("TREE",              self.COL_GREEN,  13),
            ("BUILDINGS-PUCCA",   self.COL_BLACK,  13),
            ("LEVEL-CROSSING",    self.COL_YELLOW, 13),
            ("RAILWAY-TRACK",     self.COL_GREEN,  13),
            # ── Alignment Plan (Working Plan) layers ──────────────────────────
            ("SHEET_BORDER",      self.COL_BLACK,  25),  # Outer sheet border
            ("ALIGNMENT",         self.COL_RED,    35),  # Track centreline (0.35mm red)
            ("FORMATION_EDGE",    self.COL_GREY,   18),  # Formation edge lines (0.18mm grey)
            ("RAILWAY_BOUNDARY",  self.COL_WHITE,  25),  # Dynamic ROW boundary lines (0.25mm white)
            ("CHAINAGE_TEXT",     self.COL_RED,    18),  # Chainage tick marks & labels
            ("CURVE_TABLE",       self.COL_RED,    18),  # RDSO curve parameter boxes
            ("STRUCTURES_PLAN",   self.COL_RED,    13),  # Bridge / culvert callout text
            ("TURNOUTS_PLAN",     self.COL_RED,    13),  # SRJ turnout annotations
            ("JUNCTION_ARROWS",   self.COL_BLACK,  13),  # Directional junction arrows
            ("NORTH_ARROW",       self.COL_BLACK,  25),  # North arrow symbol
        ]

        for name, color, lw in layers:
            layer = doc.layers.add(name)
            layer.color = color
            layer.lineweight = lw
            if name in ["RAILWAY_BOUNDARY", "ROW"]:
                layer.linetype = "RAILWAY_ROW"

    # -------------------------------------------------------------------------
    # Sheet border helper (used by alignment and L‑section sheets)
    # -------------------------------------------------------------------------
    def _draw_sheet_border(
        self,
        msp: Any,
        x0: float,
        y0: float,
        width_mm: float,
        height_mm: float,
        margin: float,
    ) -> None:
        """
        Draw a rectangular border around the sheet.

        Parameters
        ----------
        msp : ezdxf.layouts.ModelSpace
            Modelspace to which the polyline is added.
        x0, y0 : float
            Lower‑left corner of the sheet.
        width_mm, height_mm : float
            Sheet dimensions in millimetres.
        margin : float
            Unused placeholder to keep signature compatible with legacy calls.
        """
        # Rectangle corners (outer border)
        x1 = x0 + width_mm
        y1 = y0 + height_mm

        # Draw closed lightweight polyline – lineweight 0.25 mm (pre-registered in _setup_layers)
        msp.add_lwpolyline(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            dxfattribs={"layer": "SHEET_BORDER", "color": self.COL_BLACK, "lineweight": 25},
        )

    def _apply_modelspace_scaling(self, msp: Any) -> None:
        """Apply global modelspace post-scaling (AutoCAD SCALE command equivalent at base point 0,0)."""
        drawing_cfg = self.config.get("drawing", {})
        auto_scale = bool(drawing_cfg.get("auto_scale_to_ground_meters", False))
        manual_factor = float(drawing_cfg.get("modelspace_scale_factor", 1.0))

        if auto_scale:
            scale_factor = self.h_scale / 1000.0
        else:
            scale_factor = manual_factor

        if abs(scale_factor - 1.0) > 1e-6:
            self.logger.info("Applying global modelspace scale factor: %.4f (AutoCAD SCALE @ 0,0)...", scale_factor)
            from ezdxf.math import Matrix44
            mat = Matrix44.scale(scale_factor, scale_factor, scale_factor)
            for entity in list(msp):
                try:
                    entity.transform(mat)
                except Exception as exc:
                    self.logger.debug("Skipping scaling for entity %s: %s", entity.dxftype(), exc)

    # -------------------------------------------------------------------------
    # Coordinate Utilities
    # -------------------------------------------------------------------------
    def _ch_to_x(self, chainage_m: float, origin_x: float, start_ch: float = 0.0) -> float:
        """Convert chainage (m) to paper X coordinate (mm) relative to sheet start chainage."""
        return origin_x + ((chainage_m - start_ch) / self.h_scale) * 1000.0

    def _elev_to_y(self, elev_m: float, datum_elev: float, origin_y: float) -> float:
        """Convert elevation (m) to paper Y coordinate (mm) above datum."""
        return origin_y + ((elev_m - datum_elev) / self.v_scale) * 1000.0

    def _col_half_width_mm(self) -> float:
        """Half a chainage cell width in mm (for centering text inside cells)."""
        interval_m = float(self.config.get("chainage", {}).get("interval_m", self.config.get("alignment", {}).get("chainage_interval", 40.0)))
        return (interval_m / self.h_scale) * 1000.0 / 2.0

    def _format_chainage(self, ch_m: float) -> str:
        """Format chainage according to unified chainage_format template (matches Plan View)."""
        from utils.geo_utils import format_chainage
        return format_chainage(ch_m, format_pattern=self.chainage_format)

    def _format_gradient(self, pct: float) -> str:
        """Format gradient % to 'RISE/FALL 1 IN X' or 'LEVEL'."""
        if abs(pct) < 1e-4:
            return "LEVEL"
        ratio = int(round(100.0 / abs(pct)))
        word = "RISE" if pct > 0 else "FALL"
        return f"{word} 1 IN {ratio}"

    def _get_pfl_break_points(
        self,
        chainages: np.ndarray,
        fl: np.ndarray,
        vip_df: Optional[pd.DataFrame] = None,
        start_ch: float = 0.0,
        draw_length_mm: float = 500.0,
    ) -> list[tuple[float, float]]:
        """Calculate vertices for the Proposed Formation Level (PFL) line.

        Breaks the line ONLY at stations where the gradient changes (VIP inflection
        points), plus the start and end of the sheet, eliminating intermediate 20m vertices.
        """
        valid_mask = ~np.isnan(chainages) & ~np.isnan(fl)
        if not np.any(valid_mask):
            return []

        ch_valid = chainages[valid_mask]
        fl_valid = fl[valid_mask]
        if len(ch_valid) <= 2:
            return [(float(c), float(f)) for c, f in zip(ch_valid, fl_valid)]

        sheet_min_ch = float(ch_valid[0])
        sheet_max_ch = float(ch_valid[-1])

        # 1. If VIP DataFrame is provided, identify true gradient inflection stations
        if vip_df is not None and not vip_df.empty and "chainage_m" in vip_df.columns:
            vips = vip_df.sort_values("chainage_m").to_dict("records")
            n_vips = len(vips)

            # Build raw gradient spans across [sheet_min_ch, sheet_max_ch]
            raw_spans: list[tuple[float, float, float]] = []
            if vips[0]["chainage_m"] > sheet_min_ch:
                c0 = float(vips[0]["chainage_m"])
                g0 = float(vips[0].get("gradient_pct", 0.0))
                raw_spans.append((sheet_min_ch, c0, g0))

            for i in range(n_vips - 1):
                c_start = float(vips[i]["chainage_m"])
                c_end = float(vips[i + 1]["chainage_m"])
                e_start = float(vips[i]["elevation_m"])
                e_end = float(vips[i + 1]["elevation_m"])
                length_m = c_end - c_start
                g_pct = ((e_end - e_start) / length_m) * 100.0 if length_m > 0.001 else 0.0
                raw_spans.append((c_start, c_end, g_pct))

            if vips[-1]["chainage_m"] < sheet_max_ch:
                cL = float(vips[-1]["chainage_m"])
                gL = float(vips[-1].get("gradient_pct", 0.0))
                raw_spans.append((cL, sheet_max_ch, gL))

            def _is_same_grad(g1: float, g2: float) -> bool:
                if abs(g1 - g2) < 0.005:
                    return True
                if abs(g1) > 1e-4 and abs(g2) > 1e-4 and (g1 * g2 > 0):
                    return int(round(100.0 / abs(g1))) == int(round(100.0 / abs(g2)))
                return False

            # Merge consecutive collinear spans
            spans: list[tuple[float, float, float]] = []
            for s_start, s_end, g_pct in raw_spans:
                if not spans:
                    spans.append((s_start, s_end, g_pct))
                else:
                    prev_start, prev_end, prev_g = spans[-1]
                    if _is_same_grad(prev_g, g_pct):
                        spans[-1] = (prev_start, s_end, prev_g)
                    else:
                        spans.append((s_start, s_end, g_pct))

            # Extract inflection chainages inside sheet
            break_chs = [sheet_min_ch]
            for s_start, s_end, _ in spans:
                if sheet_min_ch < s_start < sheet_max_ch and not any(abs(s_start - bc) < 0.1 for bc in break_chs):
                    break_chs.append(s_start)
                if sheet_min_ch < s_end < sheet_max_ch and not any(abs(s_end - bc) < 0.1 for bc in break_chs):
                    break_chs.append(s_end)
            if not any(abs(sheet_max_ch - bc) < 0.1 for bc in break_chs):
                break_chs.append(sheet_max_ch)

            break_chs.sort()
            vip_elev_map = {float(r["chainage_m"]): float(r["elevation_m"]) for r in vips if "elevation_m" in r}

            break_pts = []
            for b_ch in break_chs:
                matched_vip_elev = next((elev for v_ch, elev in vip_elev_map.items() if abs(v_ch - b_ch) < 0.1), None)
                if matched_vip_elev is not None:
                    elev = matched_vip_elev
                else:
                    elev = float(np.interp(b_ch, ch_valid, fl_valid))
                break_pts.append((b_ch, elev))

            if len(break_pts) >= 2:
                return break_pts

        # 2. Fallback: Slope change detection across (ch_valid, fl_valid)
        break_pts = [(float(ch_valid[0]), float(fl_valid[0]))]
        slopes: list[float] = []
        for i in range(len(ch_valid) - 1):
            dx = float(ch_valid[i + 1] - ch_valid[i])
            dy = float(fl_valid[i + 1] - fl_valid[i])
            slopes.append(dy / dx if abs(dx) > 1e-6 else 0.0)

        for i in range(1, len(ch_valid) - 1):
            g_in = slopes[i - 1]
            g_out = slopes[i]
            if abs(g_out - g_in) > 1e-4:  # Slope changes by > 0.01%
                break_pts.append((float(ch_valid[i]), float(fl_valid[i])))

        break_pts.append((float(ch_valid[-1]), float(fl_valid[-1])))
        return break_pts

    def _draw_km_milestone_symbol(
        self,
        msp: Any,
        x: float,
        y_line: float,
        km_val: int,
        layer: str = "CHAINAGE",
    ) -> float:
        """Draw standard Indian Railways Upward-Arrow Kilometer Milestone Post symbol.

        Returns the bottom Y coordinate of the milestone box.
        """
        w_km = 14.0       # width of milestone box in mm
        h_km = 4.0        # height of milestone box in mm
        arrow_h = 3.6     # total height of upward pointing arrow
        arrow_head_h = 1.8
        arrow_head_w = 2.4
        shaft_w = 1.0

        y_tip = y_line
        y_arrow_base = y_tip - arrow_head_h
        y_box_top = y_tip - arrow_h
        y_box_bot = y_box_top - h_km

        # Solid red upward-pointing arrow polygon (touching y_line at tip)
        arrow_poly = [
            (x, y_tip),
            (x - arrow_head_w / 2.0, y_arrow_base),
            (x - shaft_w / 2.0, y_arrow_base),
            (x - shaft_w / 2.0, y_box_top),
            (x + shaft_w / 2.0, y_box_top),
            (x + shaft_w / 2.0, y_arrow_base),
            (x + arrow_head_w / 2.0, y_arrow_base),
            (x, y_tip),
        ]
        msp.add_lwpolyline(
            arrow_poly,
            dxfattribs={"layer": layer, "color": self.COL_RED, "lineweight": 35},
        )
        try:
            hatch = msp.add_hatch(color=self.COL_RED, dxfattribs={"layer": layer})
            hatch.paths.add_polyline_path(arrow_poly, is_closed=True)
        except Exception:
            pass

        # Rectangular milestone box
        x_l = x - w_km / 2.0
        x_r = x + w_km / 2.0
        box_pts = [
            (x_l, y_box_top),
            (x_r, y_box_top),
            (x_r, y_box_bot),
            (x_l, y_box_bot),
            (x_l, y_box_top),
        ]
        msp.add_lwpolyline(
            box_pts,
            dxfattribs={"layer": layer, "color": self.COL_RED, "lineweight": 25},
        )

        # Text: KM : XX
        txt_km = msp.add_text(
            f"KM :{km_val}",
            dxfattribs={
                "height": 2.2,
                "layer": layer,
                "style": "ESP_STYLE",
                "color": self.COL_RED,
            }
        )
        txt_km.set_placement((x, (y_box_top + y_box_bot) / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)
        return y_box_bot

    # -------------------------------------------------------------------------
    # Profile View
    # -------------------------------------------------------------------------
    def _draw_profile_view(
        self,
        msp: Any,
        chainages: np.ndarray,
        egl: np.ndarray,
        fl: np.ndarray,
        origin_x: float,
        origin_y: float,
        datum_elev: float,
        draw_length_mm: float,
        label_width: float,
        vip_df: Optional[pd.DataFrame] = None,
        align_res: Optional[Any] = None,
        topo_res: Optional[Any] = None,
        ip_df: Optional[pd.DataFrame] = None,
        start_ch: float = 0.0,
        bridge_df: Optional[Any] = None,
        profile_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Draw longitudinal profile: EGL, EFL, ERL, PFL, elevation grid, VIP markers.

        Args:
            origin_y: Y coordinate of the datum line (bottom of profile area).
            start_ch: Starting chainage of the current sheet (for multi-file chunking).
        """
        if len(chainages) == 0:
            return

        # Determine max elevation for grid top (up to profile_max_height_m above datum, default 15m)
        profile_max_h = float(
            self.config.get("drawing", {}).get("profile_max_height_m", 15.0)
        )
        max_grid_elev = datum_elev + profile_max_h
        max_y = self._elev_to_y(max_grid_elev, datum_elev, origin_y)

        # ── Y-Axis Elevation Labels (Left Edge) ─────────────
        elev = datum_elev
        while elev <= max_grid_elev:
            y = self._elev_to_y(elev, datum_elev, origin_y)
            is_major = (round(elev) % 5 == 0)
            
            if is_major:
                # Small tick mark on the Y-axis
                msp.add_line(
                    (origin_x - 1.5, y),
                    (origin_x, y),
                    dxfattribs={"layer": "PROFILE_GRID", "color": self.COL_RED, "lineweight": 13}
                )
                txt = msp.add_text(
                    f"{elev:.0f}",
                    dxfattribs={
                        "height": 2.2,
                        "layer": "PROFILE_GRID",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                )
                txt.set_placement((origin_x - 2.5, y), align=TextEntityAlignment.MIDDLE_RIGHT)

            # Horizontal grey grid line (every 5m)
            if is_major and elev > datum_elev and elev < max_grid_elev:
                msp.add_line(
                    (origin_x, y),
                    (origin_x + draw_length_mm, y),
                    dxfattribs={"layer": "PROFILE_GRID", "color": self.COL_GREY, "lineweight": 5}
                )
            elev += 1.0

        # ── Vertical Grid Lines (10m Blue Lines from Datum to EGL) ────────
        ch_10m = start_ch
        end_ch = start_ch + (draw_length_mm / 1000.0) * self.h_scale
        while ch_10m <= end_ch + 0.001:
            if len(chainages) > 0 and len(egl) > 0:
                egl_val = float(np.interp(ch_10m, chainages, egl))
                # Skip invalid 0.0 values (typically caused by DEM missing data at edges)
                if egl_val != 0.0:
                    x = self._ch_to_x(ch_10m, origin_x, start_ch=start_ch)
                    y_egl = self._elev_to_y(egl_val, datum_elev, origin_y)
                    
                    # Ensure the blue line never pierces downwards through the data band
                    y_egl = max(origin_y, y_egl)
                    
                    is_100m = (round(ch_10m) % 100 == 0)
                    lw = 13 if is_100m else 0
                    msp.add_line(
                        (x, origin_y),
                        (x, y_egl),
                        dxfattribs={"layer": "PROFILE_GRID", "color": self.COL_BLUE, "lineweight": lw}
                    )
            ch_10m += 20.0

        # ── Datum line (bold red) & Profile Box ───────────────────────────
        msp.add_lwpolyline(
            [
                (origin_x, origin_y),
                (origin_x + draw_length_mm, origin_y),
                (origin_x + draw_length_mm, max_y),
                (origin_x, max_y),
                (origin_x, origin_y),
            ],
            dxfattribs={"layer": "PROFILE_GRID", "color": self.COL_RED, "lineweight": 25}
        )
        datum_lbl = msp.add_text(
            f"DATUM = {datum_elev:.0f}m",
            dxfattribs={
                "height": 3.0,
                "layer": "PROFILE_GRID",
                "style": "ESP_STYLE",
                "color": self.COL_RED,
            }
        )
        datum_lbl.set_placement(
            (origin_x - 5.0, origin_y - 1.5), align=TextEntityAlignment.TOP_RIGHT
        )


        # ── EGL polyline (black / dark blue ground line) ────────────────────
        egl_pts = [
            (self._ch_to_x(c, origin_x, start_ch=start_ch), self._elev_to_y(e, datum_elev, origin_y))
            for c, e in zip(chainages, egl)
            if not np.isnan(e)
        ]
        if egl_pts:
            msp.add_lwpolyline(
                egl_pts,
                dxfattribs={"layer": "PROFILE_EGL", "color": self.COL_BLACK, "lineweight": 25, "linetype": "DASHED"}
            )

        # ── Check availability of Existing Formation & Rail Levels for this sheet ──
        has_efl = False
        if profile_df is not None and "existing_formation_level_m" in profile_df.columns and "chainage_m" in profile_df.columns:
            m_efl = (
                (profile_df["chainage_m"] >= start_ch - 1.0) &
                (profile_df["chainage_m"] <= end_ch + 1.0) &
                profile_df["existing_formation_level_m"].notna() &
                (profile_df["existing_formation_level_m"] > 0)
            )
            has_efl = bool(np.any(m_efl))

        has_erl = False
        if profile_df is not None and "existing_rail_level_m" in profile_df.columns and "chainage_m" in profile_df.columns:
            m_erl = (
                (profile_df["chainage_m"] >= start_ch - 1.0) &
                (profile_df["chainage_m"] <= end_ch + 1.0) &
                profile_df["existing_rail_level_m"].notna() &
                (profile_df["existing_rail_level_m"] > 0)
            )
            has_erl = bool(np.any(m_erl))

        # ── Existing Formation Level (EFL) polyline (Black Dashed Line) ────
        if has_efl:
            efl_pts = []
            for _, r in profile_df.iterrows():
                c = float(r["chainage_m"])
                if start_ch - 1.0 <= c <= end_ch + 1.0:
                    val = float(r["existing_formation_level_m"])
                    if not np.isnan(val) and val > 0:
                        efl_pts.append((self._ch_to_x(c, origin_x, start_ch=start_ch), self._elev_to_y(val, datum_elev, origin_y)))
            if len(efl_pts) >= 2:
                msp.add_lwpolyline(
                    efl_pts,
                    dxfattribs={"layer": "PROFILE_EFL", "color": self.COL_BLACK, "lineweight": 20, "linetype": "DASHED"}
                )

        # ── Existing Rail Level (ERL) polyline (Black Dashed Line) ─────────
        if has_erl:
            erl_pts = []
            for _, r in profile_df.iterrows():
                c = float(r["chainage_m"])
                if start_ch - 1.0 <= c <= end_ch + 1.0:
                    val = float(r["existing_rail_level_m"])
                    if not np.isnan(val) and val > 0:
                        erl_pts.append((self._ch_to_x(c, origin_x, start_ch=start_ch), self._elev_to_y(val, datum_elev, origin_y)))
            if len(erl_pts) >= 2:
                msp.add_lwpolyline(
                    erl_pts,
                    dxfattribs={"layer": "PROFILE_ERL", "color": self.COL_BLACK, "lineweight": 20, "linetype": "DASHED"}
                )

        # ── Proposed Formation Level (PFL) line (breaks ONLY at gradient changes) ──
        pfl_break_pts = self._get_pfl_break_points(
            chainages, fl, vip_df=vip_df, start_ch=start_ch, draw_length_mm=draw_length_mm
        )
        if len(pfl_break_pts) >= 2:
            pts = [
                (self._ch_to_x(c, origin_x, start_ch=start_ch), self._elev_to_y(f, datum_elev, origin_y))
                for c, f in pfl_break_pts
            ]
            for i in range(len(pts) - 1):
                msp.add_line(
                    pts[i], pts[i + 1],
                    dxfattribs={"layer": "PROFILE_FL", "color": self.COL_RED, "lineweight": 35}
                )
        else:
            pfl_pts = [
                (self._ch_to_x(c, origin_x, start_ch=start_ch), self._elev_to_y(f, datum_elev, origin_y))
                for c, f in zip(chainages, fl)
                if not np.isnan(f)
            ]
            if len(pfl_pts) >= 2:
                msp.add_lwpolyline(
                    pfl_pts,
                    dxfattribs={"layer": "PROFILE_FL", "color": self.COL_RED, "lineweight": 35}
                )

        # ── Profile Title: <u>L - SECTION</u> ────────────────────────────
        x_mid_prof = origin_x + draw_length_mm / 2.0
        y_title_prof = max_y + 7.5
        txt_lsec = msp.add_text(
            "L - SECTION",
            dxfattribs={"height": 3.8, "layer": "PROFILE_GRID", "style": "ESP_STYLE", "color": self.COL_BLACK}
        )
        txt_lsec.set_placement((x_mid_prof, y_title_prof), align=TextEntityAlignment.MIDDLE_CENTER)
        msp.add_line(
            (x_mid_prof - 16.0, y_title_prof - 3.0),
            (x_mid_prof + 16.0, y_title_prof - 3.0),
            dxfattribs={"layer": "PROFILE_GRID", "color": self.COL_BLACK, "lineweight": 25}
        )

        # ── Left Tiered / Stepped Shelf Labels (ERL, EFL, Ground Level) ──
        ch_probe = start_ch + (18.0 / 1000.0) * self.h_scale
        y_egl_probe = None
        if len(chainages) > 0 and len(egl) > 0:
            mask_egl = ~np.isnan(egl)
            if np.any(mask_egl):
                egl_val = float(np.interp(ch_probe, chainages[mask_egl], egl[mask_egl]))
                y_egl_probe = self._elev_to_y(egl_val, datum_elev, origin_y)

        y_efl_probe = None
        if has_efl and profile_df is not None:
            m_efl = profile_df["existing_formation_level_m"].notna() & (profile_df["existing_formation_level_m"] > 0)
            if np.any(m_efl):
                efl_val = float(np.interp(ch_probe, profile_df.loc[m_efl, "chainage_m"], profile_df.loc[m_efl, "existing_formation_level_m"]))
                y_efl_probe = self._elev_to_y(efl_val, datum_elev, origin_y)

        y_erl_probe = None
        if has_erl and profile_df is not None:
            m_erl = profile_df["existing_rail_level_m"].notna() & (profile_df["existing_rail_level_m"] > 0)
            if np.any(m_erl):
                erl_val = float(np.interp(ch_probe, profile_df.loc[m_erl, "chainage_m"], profile_df.loc[m_erl, "existing_rail_level_m"]))
                y_erl_probe = self._elev_to_y(erl_val, datum_elev, origin_y)

        x_shelf_base = origin_x + 15.0

        # 1. GROUND LEVEL Shelf
        if y_egl_probe is not None:
            x_dr = x_shelf_base + 6.0
            y_sh = y_egl_probe + 8.0
            msp.add_line((x_dr, y_egl_probe), (x_dr, y_sh), dxfattribs={"layer": "PROFILE_EGL", "color": self.COL_BLACK, "lineweight": 18})
            msp.add_line((x_dr, y_sh), (x_dr + 24.0, y_sh), dxfattribs={"layer": "PROFILE_EGL", "color": self.COL_BLACK, "lineweight": 18})
            t_egl = msp.add_text("GROUND LEVEL", dxfattribs={"height": 2.0, "layer": "PROFILE_EGL", "style": "ESP_STYLE", "color": self.COL_BLACK})
            t_egl.set_placement((x_dr + 1.0, y_sh + 1.0), align=TextEntityAlignment.BOTTOM_LEFT)

        # 2. EXG. FORMATION LEVEL Shelf
        if y_efl_probe is not None and has_efl:
            x_dr = x_shelf_base + 4.0
            y_sh = y_efl_probe + 14.0
            msp.add_line((x_dr, y_efl_probe), (x_dr, y_sh), dxfattribs={"layer": "PROFILE_EFL", "color": self.COL_BLACK, "lineweight": 18})
            msp.add_line((x_dr, y_sh), (x_dr + 28.0, y_sh), dxfattribs={"layer": "PROFILE_EFL", "color": self.COL_BLACK, "lineweight": 18})
            t_efl = msp.add_text("EXG. FORMATION LEVEL", dxfattribs={"height": 2.0, "layer": "PROFILE_EFL", "style": "ESP_STYLE", "color": self.COL_BLACK})
            t_efl.set_placement((x_dr + 1.0, y_sh + 1.0), align=TextEntityAlignment.BOTTOM_LEFT)

        # 3. EXG. RAIL LEVEL Shelf
        if y_erl_probe is not None and has_erl:
            x_dr = x_shelf_base + 2.0
            y_sh = y_erl_probe + 20.0
            msp.add_line((x_dr, y_erl_probe), (x_dr, y_sh), dxfattribs={"layer": "PROFILE_ERL", "color": self.COL_BLACK, "lineweight": 18})
            msp.add_line((x_dr, y_sh), (x_dr + 24.0, y_sh), dxfattribs={"layer": "PROFILE_ERL", "color": self.COL_BLACK, "lineweight": 18})
            t_erl = msp.add_text("EXG. RAIL LEVEL", dxfattribs={"height": 2.0, "layer": "PROFILE_ERL", "style": "ESP_STYLE", "color": self.COL_BLACK})
            t_erl.set_placement((x_dr + 1.0, y_sh + 1.0), align=TextEntityAlignment.BOTTOM_LEFT)

        # 4. PROP. FORMATION LEVEL Shelf (Red)
        if len(chainages) > 0 and len(fl) > 0:
            ch_pfl_probe = start_ch + (draw_length_mm * 0.70 / 1000.0) * self.h_scale
            mask_fl = ~np.isnan(fl)
            if np.any(mask_fl):
                pfl_val = float(np.interp(ch_pfl_probe, chainages[mask_fl], fl[mask_fl]))
                y_pfl = self._elev_to_y(pfl_val, datum_elev, origin_y)
                x_pfl = self._ch_to_x(ch_pfl_probe, origin_x, start_ch=start_ch)
                y_sh_pfl = y_pfl + 12.0
                msp.add_line((x_pfl, y_pfl), (x_pfl, y_sh_pfl), dxfattribs={"layer": "PROFILE_FL", "color": self.COL_RED, "lineweight": 18})
                msp.add_line((x_pfl, y_sh_pfl), (x_pfl + 32.0, y_sh_pfl), dxfattribs={"layer": "PROFILE_FL", "color": self.COL_RED, "lineweight": 18})
                t_pfl = msp.add_text("PROP. FORMATION LEVEL", dxfattribs={"height": 2.2, "layer": "PROFILE_FL", "style": "ESP_STYLE", "color": self.COL_RED})
                t_pfl.set_placement((x_pfl + 1.0, y_sh_pfl + 1.0), align=TextEntityAlignment.BOTTOM_LEFT)


        # ── VIP Markers (Drop line from profile to datum omitted per user request) ────

        # ── Bridge Hydraulic Levels: HFL (Blue) & Bed Level (Green) ────
        self._draw_bridge_hydraulic_levels(
            msp=msp,
            bridge_df=bridge_df,
            origin_x=origin_x,
            origin_y=origin_y,
            datum_elev=datum_elev,
            draw_length_mm=draw_length_mm,
            start_ch=start_ch,
            chainages=chainages,
            fl=fl,
        )

        # ── Slanted 60° Leader Callouts (Bridges, Culverts, Curve Transitions) ────
        self._draw_slanted_leader_callouts(
            msp=msp,
            chainages=chainages,
            fl=fl,
            origin_x=origin_x,
            origin_y=origin_y,
            max_y=max_y,
            datum_elev=datum_elev,
            draw_length_mm=draw_length_mm,
            start_ch=start_ch,
            bridge_df=bridge_df,
            topo_res=topo_res,
            align_res=align_res,
            ip_df=ip_df,
        )

        # ── Profile border box (red rectangle) ───────────────────────────
        box_x0 = origin_x - label_width
        box_x1 = origin_x + draw_length_mm
        msp.add_lwpolyline(
            [
                (box_x0, origin_y),
                (box_x1, origin_y),
                (box_x1, max_y),
                (box_x0, max_y),
                (box_x0, origin_y),
            ],
            dxfattribs={"layer": "PROFILE_GRID", "color": self.COL_RED, "lineweight": 35}
        )
        # Left border label column separator
        msp.add_line(
            (origin_x, origin_y),
            (origin_x, max_y),
            dxfattribs={"layer": "PROFILE_GRID", "color": self.COL_RED, "lineweight": 25}
        )

    # -------------------------------------------------------------------------
    # Spatial Bounding Box & Collision Avoidance Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _arc_from_2_points_and_radius(
        p1: tuple[float, float],
        p2: tuple[float, float],
        radius: float,
        pm: Optional[tuple[float, float]] = None,
        center_hint: Optional[tuple[float, float]] = None,
        is_right_turn: bool = True,
    ) -> Optional[tuple[tuple[float, float], float, float, float]]:
        """Compute exact circular arc of target radius R passing strictly through p1 and p2."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        chord = math.hypot(dx, dy)
        if chord < 1e-5 or radius <= 0:
            return None

        # Radius must be at least half chord length
        r_eff = max(radius, (chord / 2.0) + 1e-4)
        h = math.sqrt(max(0.0, r_eff**2 - (chord / 2.0)**2))

        # Midpoint of chord
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0

        # Normal unit vector to chord (c_left: +90 deg, c_right: -90 deg from chord vector)
        ux, uy = dx / chord, dy / chord
        c_left = (mx - h * uy, my + h * ux)
        c_right = (mx + h * uy, my - h * ux)

        # Pick candidate center matching curve orientation
        if center_hint is not None:
            d_l = math.hypot(c_left[0] - center_hint[0], c_left[1] - center_hint[1])
            d_r = math.hypot(c_right[0] - center_hint[0], c_right[1] - center_hint[1])
            center = c_left if d_l <= d_r else c_right
        else:
            center = c_right if is_right_turn else c_left

        # Angles from center to p1 and p2
        a1 = math.degrees(math.atan2(y1 - center[1], x1 - center[0])) % 360.0
        a2 = math.degrees(math.atan2(y2 - center[1], x2 - center[0])) % 360.0

        # AutoCAD arcs sweep Counter-Clockwise from start_angle to end_angle
        if is_right_turn:
            start_angle, end_angle = a2, a1
        else:
            start_angle, end_angle = a1, a2

        return center, r_eff, start_angle, end_angle

    @staticmethod
    def _fit_arc_from_3_points(
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
    ) -> Optional[tuple[tuple[float, float], float, float, float]]:
        """Fit 2D circle center (cx, cy), radius r, start_angle_deg, end_angle_deg through 3 points p1, p2, p3."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        temp = x2 * x2 + y2 * y2
        bc = (x1 * x1 + y1 * y1 - temp) / 2.0
        cd = (temp - x3 * x3 - y3 * y3) / 2.0
        det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

        if abs(det) < 1e-6:
            return None

        cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
        cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
        r = math.hypot(x1 - cx, y1 - cy)

        a1 = math.degrees(math.atan2(y1 - cy, x1 - cx)) % 360.0
        a2 = math.degrees(math.atan2(y2 - cy, x2 - cx)) % 360.0
        a3 = math.degrees(math.atan2(y3 - cy, x3 - cx)) % 360.0

        ccw_sweep = (a3 - a1) % 360.0
        mid_sweep = (a2 - a1) % 360.0

        if mid_sweep <= ccw_sweep:
            start_angle = a1
            end_angle = a3
        else:
            start_angle = a3
            end_angle = a1

        return (cx, cy), r, start_angle, end_angle

    @staticmethod
    def _compute_text_aabb(
        text: str,
        height_mm: float,
        x: float,
        y: float,
        angle_deg: float,
        margin_mm: float = 1.5,
    ) -> tuple[float, float, float, float]:
        """Compute Axis-Aligned Bounding Box (xmin, ymin, xmax, ymax) for a rotated text label.

        Args:
            text: Label string.
            height_mm: Text height in mm.
            x: Text placement origin X coordinate (mm).
            y: Text placement origin Y coordinate (mm).
            angle_deg: Rotation angle in degrees (e.g. 60.0 or 120.0).
            margin_mm: Safety margin around the bounding box in mm.

        Returns:
            (xmin, ymin, xmax, ymax) tuple.
        """
        approx_width = max(10.0, len(text) * (height_mm * 0.65))
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        ux, uy = cos_a, sin_a
        vx, vy = -sin_a, cos_a

        p0 = (x, y)
        p1 = (x + approx_width * ux, y + approx_width * uy)
        p2 = (p1[0] + height_mm * vx, p1[1] + height_mm * vy)
        p3 = (x + height_mm * vx, y + height_mm * vy)

        xs = [p0[0], p1[0], p2[0], p3[0]]
        ys = [p0[1], p1[1], p2[1], p3[1]]

        return (
            min(xs) - margin_mm,
            min(ys) - margin_mm,
            max(xs) + margin_mm,
            max(ys) + margin_mm,
        )

    @staticmethod
    def _boxes_intersect(
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> bool:
        """Return True if two AABB bounding boxes intersect."""
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        return not (
            xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1
        )

    def _draw_bridge_hydraulic_levels(
        self,
        msp: Any,
        bridge_df: Optional[Any],
        origin_x: float,
        origin_y: float,
        datum_elev: float,
        draw_length_mm: float,
        start_ch: float = 0.0,
        chainages: Optional[np.ndarray] = None,
        fl: Optional[np.ndarray] = None,
    ) -> None:
        """Plot continuous smooth Bridge Hydraulic levels: HFL (Blue Line) and Bed Level (Green Line) on the Surface Profile."""
        from scipy.interpolate import PchipInterpolator

        # Auto-load bridge list if not provided
        if bridge_df is None or (isinstance(bridge_df, pd.DataFrame) and bridge_df.empty):
            br_path_cfg = self.config.get("paths", {}).get("bridge_list_path", "D:/ESP-Backup/Bridge_List_OP2.xlsx")
            if br_path_cfg:
                br_path = Path(br_path_cfg)
                if not br_path.is_absolute():
                    for base in [_PROJECT_ROOT, _PROJECT_ROOT.parent]:
                        if (base / br_path).exists():
                            br_path = (base / br_path).resolve()
                            break
                if br_path.exists():
                    try:
                        if br_path.suffix.lower() in [".xlsx", ".xls"]:
                            try:
                                bridge_df = pd.read_excel(br_path, sheet_name="Bridge List", header=0)
                            except Exception:
                                bridge_df = pd.read_excel(br_path, header=0)
                        else:
                            bridge_df = pd.read_csv(br_path)
                    except Exception:
                        pass

        if bridge_df is None or not hasattr(bridge_df, "iterrows") or bridge_df.empty:
            return

        end_ch = start_ch + (draw_length_mm / 1000.0) * self.h_scale

        # 1. Collect all valid surveyed HFL and Bed Level points across entire corridor
        valid_hfl_pts: list[tuple[float, float]] = []
        valid_bed_pts: list[tuple[float, float]] = []
        bridge_records: list[dict[str, Any]] = []

        for _, br_row in bridge_df.iterrows():
            row_dict = {str(k).strip().lower(): v for k, v in br_row.items() if pd.notna(v)}

            # Extract chainage
            raw_ch = None
            for key in ["approx. chainage (km)", "approx. chainage (m)", "approx. chainage", "chainage_m", "chainage_km", "chainage", "ch", "station"]:
                if key in row_dict:
                    raw_ch = row_dict[key]
                    break
            if raw_ch is None:
                continue
            try:
                ch = float(raw_ch)
                if 0.0 < ch < 75.0:
                    ch *= 1000.0
            except Exception:
                continue

            # Extract HFL
            hfl_val = None
            for key in ["hfl", "hfl (m)", "h.f.l", "h.f.l.", "high flood level", "design hfl", "hfl_m"]:
                if key in row_dict:
                    try:
                        hfl_val = float(str(row_dict[key]).replace("m", "").replace("M", "").strip())
                        break
                    except Exception:
                        pass

            # Extract Bed Level
            bed_val = None
            for key in ["bed level", "bed level (m)", "bed_level", "bed rl", "bed level (rl)", "bl", "bed_m"]:
                if key in row_dict:
                    try:
                        bed_val = float(str(row_dict[key]).replace("m", "").replace("M", "").strip())
                        break
                    except Exception:
                        pass

            if hfl_val is not None and not np.isnan(hfl_val):
                valid_hfl_pts.append((ch, hfl_val))
            if bed_val is not None and not np.isnan(bed_val):
                valid_bed_pts.append((ch, bed_val))

            if (start_ch <= ch <= end_ch) and (hfl_val is not None or bed_val is not None):
                span_str = str(row_dict.get("proposed span", row_dict.get("span", "")))
                span_w_m = 15.0
                m_span = re.search(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)', span_str, re.IGNORECASE)
                if m_span:
                    try:
                        n_spans = float(m_span.group(1))
                        s_len = float(m_span.group(2))
                        span_w_m = max(10.0, n_spans * s_len)
                    except Exception:
                        pass
                bridge_records.append({
                    "ch": ch,
                    "hfl": hfl_val,
                    "bed": bed_val,
                    "span_w_m": span_w_m,
                })

        # 2. Build Smooth Continuous Monotonic Interpolators (PCHIP without overshoot or sharp jumps)
        num_sample_pts = max(100, int((end_ch - start_ch) / 20.0))
        sample_chs = np.linspace(start_ch, end_ch, num_sample_pts)




        # ── 3. Individual Structure Station Callout Tags ──
        for br in bridge_records:
            ch = br["ch"]
            hfl_val = br["hfl"]
            bed_val = br["bed"]
            span_w_m = br["span_w_m"]

            x_center = self._ch_to_x(ch, origin_x, start_ch=start_ch)
            half_w_mm = (span_w_m / 2.0 / self.h_scale) * 1000.0
            half_w_mm = max(3.5, min(18.0, half_w_mm))
            x1_bar = max(origin_x, x_center - half_w_mm)
            x2_bar = min(origin_x + draw_length_mm, x_center + half_w_mm)

            fl_val = None
            y_pfl = origin_y + 10.0
            if chainages is not None and fl is not None and len(chainages) > 0 and len(fl) > 0:
                mask = ~np.isnan(fl)
                if np.any(mask):
                    fl_val = float(np.interp(ch, chainages[mask], fl[mask]))
                    y_pfl = self._elev_to_y(fl_val, datum_elev, origin_y)

            # Attach HFL, Bed Level & Formation Level text values cleanly ABOVE the Proposed Formation Level (PFL)
            y_text_base = y_pfl + 1.5
            if bed_val is not None and not np.isnan(bed_val):
                txt_bed = msp.add_text(
                    f"BED LEVEL: {bed_val:.2f}m",
                    dxfattribs={
                        "height": 1.6,
                        "layer": "PROFILE_BED_LEVEL",
                        "style": "ESP_STYLE",
                        "color": self.COL_WHITE,
                    }
                )
                txt_bed.set_placement((x_center, y_text_base), align=TextEntityAlignment.BOTTOM_CENTER)
                y_text_base += 2.6

            if hfl_val is not None and not np.isnan(hfl_val):
                txt_hfl = msp.add_text(
                    f"HFL: {hfl_val:.2f}m",
                    dxfattribs={
                        "height": 1.6,
                        "layer": "PROFILE_HFL",
                        "style": "ESP_STYLE",
                        "color": self.COL_BLUE,
                    }
                )
                txt_hfl.set_placement((x_center, y_text_base), align=TextEntityAlignment.BOTTOM_CENTER)
                y_text_base += 2.6

            if fl_val is not None and not np.isnan(fl_val):
                txt_pfl = msp.add_text(
                    f"FL: {fl_val:.3f}m",
                    dxfattribs={
                        "height": 1.6,
                        "layer": "PROFILE_FL",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    }
                )
                txt_pfl.set_placement((x_center, y_text_base), align=TextEntityAlignment.BOTTOM_CENTER)

    def _draw_slanted_leader_callouts(

        self,
        msp: Any,
        chainages: np.ndarray,
        fl: np.ndarray,
        origin_x: float,
        origin_y: float,
        max_y: float,
        datum_elev: float,
        draw_length_mm: float,
        start_ch: float = 0.0,
        bridge_df: Optional[Any] = None,
        topo_res: Optional[Any] = None,
        align_res: Optional[Any] = None,
        ip_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Render 60° slanted notation leader lines (Bridges, Culverts, Curve Transitions) matching Indian Railways standards."""
        callout_items: list[tuple[float, str, bool]] = []

        if bridge_df is not None and hasattr(bridge_df, "empty") and not bridge_df.empty:
            for _, br_row in bridge_df.iterrows():
                row_dict = {str(k).strip().lower(): v for k, v in br_row.items() if pd.notna(v)}

                raw_ch_val = None
                for key in [
                    "approx. chainage (km)",
                    "approx. chainage (m)",
                    "approx. chainage",
                    "chainage_m",
                    "chainage_km",
                    "chainage",
                    "ch",
                    "station",
                ]:
                    if key in row_dict:
                        raw_ch_val = row_dict[key]
                        break

                if raw_ch_val is None:
                    continue

                try:
                    ch = float(raw_ch_val)
                except (ValueError, TypeError):
                    continue

                # Auto-convert km to metres ONLY if raw value is truly in km (< 75.0)
                if 0.0 < ch < 75.0:
                    ch = ch * 1000.0

                if ch <= 0.0:
                    continue

                # Bridge number
                br_no = ""
                for key in ["bridge no.", "bridge no", "bridge_no", "br_no", "br no", "no", "s.no"]:
                    if key in row_dict:
                        br_no = str(row_dict[key]).strip()
                        break

                # Proposed Span
                span_str = ""
                for key in ["proposed span", "proposed_span", "span", "span_config", "size"]:
                    if key in row_dict:
                        span_str = str(row_dict[key]).strip()
                        break

                # Bridge type & material
                br_type = ""
                for key in [
                    "bridge type",
                    "bridge type / description",
                    "type",
                    "structure_type",
                    "bridge_type_raw",
                ]:
                    if key in row_dict:
                        br_type = str(row_dict[key]).strip()
                        break

                # Configuration / Classification (MINOR, MAJOR, RUB, ROB, LC)
                config_class = ""
                for key in ["configration", "configuration", "category", "status"]:
                    if key in row_dict:
                        config_class = str(row_dict[key]).strip()
                        break

                is_exg = "EXG" in config_class.upper() or "EXISTING" in config_class.upper()

                # Format Callout Text
                if br_no.upper().startswith("LC") or "LC" in config_class.upper():
                    txt_callout = f"PRO. {br_no} ({span_str if span_str else 'LC'}) AT CH.:{ch:.2f}M"
                elif config_class.upper() in ["RUB", "ROB"]:
                    span_part = f"{span_str} " if span_str else ""
                    type_part = f"{br_type} " if br_type else ""
                    txt_callout = f"PRO. {br_no} OF {span_part}{type_part}({config_class.upper()}) AT CH.:{ch:.2f}M".replace("  ", " ")
                else:
                    span_part = f"{span_str} " if span_str else ""
                    type_part = f"{br_type}" if br_type else "CULVERT"
                    txt_callout = f"PRO. {br_no} OF {span_part}{type_part} AT CH.:{ch:.2f}M".replace("  ", " ")

                callout_items.append((ch, txt_callout, is_exg))

        elif topo_res is not None and hasattr(topo_res, "structures") and topo_res.structures:
            for s in topo_res.structures:
                is_exg = "EXISTING" in s.structure_name.upper() or "EXG" in s.structure_name.upper()
                prefix = "EXG. " if is_exg else "PRO. "
                s_name = s.structure_name.upper()
                if not s_name.startswith("PRO.") and not s_name.startswith("EXG."):
                    s_name = f"{prefix}{s_name}"
                txt_callout = f"{s_name} AT CH.:{s.chainage_m:.2f}M"
                callout_items.append((s.chainage_m, txt_callout, is_exg))


        if ip_df is not None and not ip_df.empty:
            for idx, row in ip_df.iterrows():
                c_num = int(row.get("ip_index", idx + 1))
                tptc = float(row.get("tptc_chainage_m", 0.0))
                tpcc = float(row.get("tpcc_chainage_m", 0.0))
                ctpc = float(row.get("ctpc_chainage_m", 0.0))
                ctpt = float(row.get("ctpt_chainage_m", 0.0))

                if tptc > 0:
                    callout_items.append((tptc, f"C NO.-{c_num:02d} TPTC-1 AT CH.:{tptc:.3f}M", False))
                if tpcc > 0 and abs(tpcc - tptc) > 3.0:
                    callout_items.append((tpcc, f"C NO.-{c_num:02d} TPCC-1 AT CH.:{tpcc:.3f}M", False))
                if ctpc > 0 and abs(ctpc - tpcc) > 3.0:
                    callout_items.append((ctpc, f"C NO.-{c_num:02d} TPCC-2 AT CH.:{ctpc:.3f}M", False))
                if ctpt > 0 and abs(ctpt - ctpc) > 3.0:
                    callout_items.append((ctpt, f"C NO.-{c_num:02d} TPTC-2 AT CH.:{ctpt:.3f}M", False))

        callout_items.sort(key=lambda item: item[0])
        unique_callouts: list[tuple[float, str, bool]] = []
        for ch, label_text, is_exg in callout_items:
            is_dup = False
            for prev_ch, prev_lbl, _ in unique_callouts:
                if abs(ch - prev_ch) < 0.5 and label_text == prev_lbl:
                    is_dup = True
                    break
            if not is_dup:
                unique_callouts.append((ch, label_text, is_exg))

        last_drawn_cx = -999.0
        stagger_toggle = False

        end_ch = start_ch + (draw_length_mm / 1000.0) * self.h_scale
        for ch, label_text, is_existing in unique_callouts:
            if not (start_ch <= ch <= end_ch):
                continue
            raw_cx = self._ch_to_x(ch, origin_x, start_ch=start_ch)
            if raw_cx < origin_x - 0.5 or raw_cx > (origin_x + draw_length_mm + 0.5):
                continue

            if raw_cx - last_drawn_cx < 12.0:
                cx = max(raw_cx, last_drawn_cx + 6.0)
                stagger_toggle = not stagger_toggle
            else:
                cx = raw_cx
                stagger_toggle = False

            last_drawn_cx = cx

            pfl_y = origin_y
            if len(chainages) > 0 and len(fl) > 0:
                mask = ~np.isnan(fl)
                if np.any(mask):
                    fl_val = float(np.interp(ch, chainages[mask], fl[mask]))
                    pfl_y = self._elev_to_y(fl_val, datum_elev, origin_y)

            start_along = 8.0 if stagger_toggle else 2.0
            text_h = 2.8
            char_w = text_h * 0.72  # Average character width for AutoCAD font
            text_len = len(label_text) * char_w
            leader_len = start_along + text_len + 2.5

            angle_rad = math.radians(60)
            cos_60 = math.cos(angle_rad)
            sin_60 = math.sin(angle_rad)

            dx = leader_len * cos_60
            dy = leader_len * sin_60

            end_x = cx + dx
            end_y = max_y + dy

            # 1. Drop line from PFL with internal dogleg step if shifted (exactly matching user diagram)
            if abs(cx - raw_cx) > 0.05:
                y_jog = pfl_y + (max_y - pfl_y) * 0.50
                msp.add_line((raw_cx, pfl_y), (raw_cx, y_jog), dxfattribs={"layer": "PROFILE_VIP", "color": self.COL_RED, "lineweight": 13})
                msp.add_line((raw_cx, y_jog), (cx, y_jog), dxfattribs={"layer": "PROFILE_VIP", "color": self.COL_RED, "lineweight": 13})
                msp.add_line((cx, y_jog), (cx, max_y), dxfattribs={"layer": "PROFILE_VIP", "color": self.COL_RED, "lineweight": 13})
            else:
                msp.add_line((raw_cx, pfl_y), (raw_cx, max_y), dxfattribs={"layer": "PROFILE_VIP", "color": self.COL_RED, "lineweight": 13})

            # 2. 60° Slanted leader line starting directly from (cx, max_y)
            msp.add_line((cx, max_y), (end_x, end_y), dxfattribs={"layer": "PROFILE_VIP", "color": self.COL_RED, "lineweight": 13})

            text_x = cx + start_along * cos_60 - 0.5 * sin_60
            text_y = max_y + start_along * sin_60 + 0.5 * cos_60
            text_color = self.COL_WHITE if is_existing else self.COL_RED
            text_layer = "DATA_BLACK" if is_existing else "DATA_RED"

            txt_call = msp.add_text(
                label_text,
                dxfattribs={"height": 2.8, "rotation": 60, "layer": text_layer, "style": "ESP_STYLE", "color": text_color}
            )
            txt_call.set_placement((text_x, text_y), align=TextEntityAlignment.BOTTOM_LEFT)



    # -------------------------------------------------------------------------
    # Data Band — Table Frame
    # -------------------------------------------------------------------------
    def _draw_table_frame(
        self,
        msp: Any,
        origin_x: float,
        origin_y: float,
        draw_length_mm: float,
        num_rows: int,
        label_width: float,
        row_colors: Optional[list[int]] = None,
        row_heights: Optional[list[float]] = None,
    ) -> None:
        """Draw outer border and horizontal dividers of the full data table matching Image 2."""
        if row_heights and len(row_heights) == num_rows:
            cum_offsets = [0.0]
            for h in row_heights:
                cum_offsets.append(cum_offsets[-1] + float(h))
            y_dividers = [origin_y - off for off in cum_offsets]
        else:
            y_dividers = [origin_y - i * self.row_height for i in range(num_rows + 1)]

        label_start_x = origin_x - label_width

        # Horizontal dividers
        for i in range(num_rows + 1):
            y = y_dividers[i]
            if row_colors and i < len(row_colors):
                color = row_colors[i]
            elif row_colors and i == len(row_colors):
                color = row_colors[-1]
            else:
                color = self.COL_RED if i <= 6 else self.COL_BLACK
            msp.add_line(
                (label_start_x, y),
                (origin_x + draw_length_mm, y),
                dxfattribs={
                    "layer": "DATA_BORDER",
                    "color": color,
                    "lineweight": 9,
                    "linetype": "CONTINUOUS",
                }
            )

        # Vertical borders: left edge, label|data divider, right edge
        for r in range(num_rows):
            y_top = y_dividers[r]
            y_bot = y_dividers[r + 1]
            color = row_colors[r] if (row_colors and r < len(row_colors)) else (self.COL_RED if r < 6 else self.COL_BLACK)
            for x in (label_start_x, origin_x, origin_x + draw_length_mm):
                msp.add_line(
                    (x, y_top),
                    (x, y_bot),
                    dxfattribs={"layer": "DATA_BORDER", "color": color, "lineweight": 18}
                )

    # -------------------------------------------------------------------------
    # Data Band — Label Column
    # -------------------------------------------------------------------------
    def _draw_label_column(
        self,
        msp: Any,
        origin_x: float,
        origin_y: float,
        labels: list[tuple[str, int]],
        label_width: float,
        row_heights: Optional[list[float]] = None,
    ) -> None:
        """Draw row label text in the left label column with auto-wrapping."""
        label_cx = origin_x - label_width / 2.0
        if row_heights and len(row_heights) == len(labels):
            cum_offsets = [0.0]
            for h in row_heights:
                cum_offsets.append(cum_offsets[-1] + float(h))
            y_dividers = [origin_y - off for off in cum_offsets]
        else:
            y_dividers = [origin_y - i * self.row_height for i in range(len(labels) + 1)]

        for i, (label, color) in enumerate(labels):
            y_top = y_dividers[i]
            y_bot = y_dividers[i + 1]
            cy = (y_top + y_bot) / 2.0
            rh = (y_top - y_bot)
            
            # Wrap long strings to fit neatly in label column
            if len(label) > 28:
                words = label.split()
                lines = []
                curr = []
                curr_len = 0
                max_w_len = 28 if len(label) < 60 else 36
                for w in words:
                    if curr_len + len(w) + 1 > max_w_len and curr:
                        lines.append(" ".join(curr))
                        curr = [w]
                        curr_len = len(w)
                    else:
                        curr.append(w)
                        curr_len += len(w) + 1
                if curr:
                    lines.append(" ".join(curr))

                n_lines = len(lines)
                line_spacing = min(2.4, (rh - 2.0) / max(1, n_lines))
                start_y = cy + (n_lines - 1) * line_spacing / 2.0

                for l_idx, l_text in enumerate(lines):
                    ly = start_y - l_idx * line_spacing
                    txt_h = min(1.7, (label_width - 4.0) / (max(1, len(l_text)) * 0.75))
                    txt = msp.add_text(
                        l_text,
                        dxfattribs={
                            "height": max(1.2, txt_h),
                            "layer": "DATA_BORDER",
                            "style": "ESP_STYLE",
                            "color": color,
                        }
                    )
                    txt.set_placement((label_cx, ly), align=TextEntityAlignment.MIDDLE_CENTER)
            else:
                txt_h = min(2.0, (label_width - 4.0) / (max(1, len(label)) * 0.75))
                txt = msp.add_text(
                    label,
                    dxfattribs={
                        "height": max(1.3, txt_h),
                        "layer": "DATA_BORDER",
                        "style": "ESP_STYLE",
                        "color": color,
                    }
                )
                txt.set_placement((label_cx, cy), align=TextEntityAlignment.MIDDLE_CENTER)

    def _draw_chainage_grid(
        self,
        msp: Any,
        chainages: np.ndarray,
        origin_x: float,
        origin_y: float,
        total_height: float,
        skip_top_rows: int = 0,
        start_ch: float = 0.0,
    ) -> None:
        """Do not draw internal vertical cell lines inside data rows (matching user reference image)."""
        return

    def _draw_vc_data_columns(
        self,
        msp: Any,
        vip_df: pd.DataFrame,
        chainages: np.ndarray,
        egl: np.ndarray,
        fl: np.ndarray,
        origin_x: float,
        origin_y: float,
        total_height: float
    ) -> None:
        """Draw red drop lines and print vertically rotated text for VC starts/ends."""
        return

    def _get_label_positions(
        self,
        chainages: np.ndarray,
        origin_x: float,
        text_h: float,
        start_ch: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Calculate X positions for station row values.
        Places rotated vertical text aligned directly at each station chainage position X_i.
        """
        n = len(chainages)
        if n == 0:
            return []

        positions = []
        for i in range(n):
            x = self._ch_to_x(chainages[i], origin_x, start_ch=start_ch)
            # Safety margin at station 0 to prevent rotated text from clashing into left table frame border
            if i == 0 and abs(x - origin_x) < 2.0:
                x = origin_x + max(2.2, text_h * 0.75)
            positions.append((i, x))

        return positions

    # -------------------------------------------------------------------------
    # Data Band — Generic Row Values
    # -------------------------------------------------------------------------
    def _draw_row_values(
        self,
        msp: Any,
        chainages: np.ndarray,
        values: np.ndarray,
        origin_x: float,
        row_center_y: float,
        color: int,
        fmt: str = "{:.3f}",
        start_ch: float = 0.0,
    ) -> None:
        """Draw rotated (90°) vertical text values for one data row.

        Text is centered vertically on the row and placed in the mid-cell position.
        """
        max_text_h = float(self.config.get("drawing", {}).get("text_height_data_mm", 2.5))
        text_h = max(2.0, max_text_h)
        layer = "DATA_RED" if color == self.COL_RED else "DATA_BLACK"

        positions = self._get_label_positions(chainages, origin_x, text_h, start_ch=start_ch)

        for idx, x in positions:
            if idx >= len(values):
                continue
            v = values[idx]
            if np.isnan(v):
                continue
            txt = msp.add_text(
                fmt.format(v),
                dxfattribs={
                    "height": text_h,
                    "rotation": 90,
                    "layer": layer,
                    "style": "ESP_STYLE",
                    "color": color,
                }
            )
            txt.set_placement((x, row_center_y), align=TextEntityAlignment.MIDDLE_CENTER)

    # -------------------------------------------------------------------------
    # Row 0 — VERTICAL GEOMETRY
    # -------------------------------------------------------------------------
    def _draw_vertical_geometry_row(
        self,
        msp: Any,
        vip_df: Optional[pd.DataFrame],
        origin_x: float,
        row_top_y: float,
        draw_length_mm: float = 500.0,
        start_ch: float = 0.0,
    ) -> None:
        """Draw standard Indian Railways vertical geometry band (sloped rise/fall lines, PVI diamond tags, length & grade text)."""
        rh = self.row_height
        cy = row_top_y - rh / 2.0
        bot_y = row_top_y - rh
        max_right_x = origin_x + draw_length_mm
        end_ch = start_ch + (draw_length_mm / 1000.0) * self.h_scale

        if vip_df is None or vip_df.empty:
            # Fallback: draw flat LEVEL baseline across sheet
            msp.add_line((origin_x, cy), (max_right_x, cy), dxfattribs={"layer": "VGEOM", "color": self.COL_RED, "lineweight": 18})
            txt = msp.add_text("LEVEL", dxfattribs={"height": 2.0, "layer": "VGEOM", "style": "ESP_STYLE", "color": self.COL_RED})
            txt.set_placement(((origin_x + max_right_x) / 2.0, cy + 0.8), align=TextEntityAlignment.BOTTOM_CENTER)
            return

        vips = vip_df.sort_values("chainage_m").to_dict("records")
        n_vips = len(vips)

        # 1. Build raw gradient spans between PVI stations
        raw_spans: list[tuple[float, float, float]] = []

        # Lead span before first PVI
        if vips[0]["chainage_m"] > start_ch:
            c0 = float(vips[0]["chainage_m"])
            g0 = float(vips[0].get("gradient_pct", 0.0))
            raw_spans.append((start_ch, c0, g0))

        # Spans between consecutive VIPs
        for i in range(n_vips - 1):
            c_start = float(vips[i]["chainage_m"])
            c_end = float(vips[i + 1]["chainage_m"])
            e_start = float(vips[i]["elevation_m"])
            e_end = float(vips[i + 1]["elevation_m"])

            length_m = c_end - c_start
            if length_m > 0.001:
                g_pct = ((e_end - e_start) / length_m) * 100.0
            else:
                g_pct = 0.0
            raw_spans.append((c_start, c_end, g_pct))

        # Tail span after last PVI
        if vips[-1]["chainage_m"] < end_ch:
            cL = float(vips[-1]["chainage_m"])
            gL = float(vips[-1].get("gradient_pct", 0.0))
            raw_spans.append((cL, end_ch, gL))

        # Helper function to test if two gradients are collinear (same slope rate)
        def _is_same_gradient(g1: float, g2: float) -> bool:
            if abs(g1 - g2) < 0.005:  # slope difference < 0.005%
                return True
            if abs(g1) > 1e-4 and abs(g2) > 1e-4 and (g1 * g2 > 0):
                inv1 = int(round(100.0 / abs(g1)))
                inv2 = int(round(100.0 / abs(g2)))
                return inv1 == inv2
            return False

        # 2. Merge consecutive collinear spans with matching gradient
        spans: list[tuple[float, float, float]] = []
        for s_start, s_end, g_pct in raw_spans:
            if not spans:
                spans.append((s_start, s_end, g_pct))
            else:
                prev_start, prev_end, prev_g = spans[-1]
                if _is_same_gradient(prev_g, g_pct):
                    # Combine into single continuous span
                    spans[-1] = (prev_start, s_end, prev_g)
                else:
                    spans.append((s_start, s_end, g_pct))

        # 3. Identify true grade inflection VIP stations (omit collinear intermediate VIPs)
        inflection_chainages = set()
        for ms_start, ms_end, _ in spans:
            inflection_chainages.add(ms_start)
            inflection_chainages.add(ms_end)

        # 4. Draw PVI Rounded Dual-Column Capsule (CH & FL) at true inflection stations
        for row in vips:
            c = float(row["chainage_m"])
            if any(abs(c - ic) < 0.1 for ic in inflection_chainages):
                elev = float(row["elevation_m"])
                x = self._ch_to_x(c, origin_x, start_ch=start_ch)

                if origin_x - 2.0 <= x <= max_right_x + 2.0:
                    dx = 3.6  # half width (total width = 7.2mm)
                    r = 1.6   # corner fillet radius
                    y_top = row_top_y
                    y_bot = bot_y
                    x_l = x - dx
                    x_r = x + dx

                    # Smooth rounded capsule polygon (4 filleted corners)
                    pts = [
                        (x_l + r, y_bot),
                        (x_r - r, y_bot),
                        (x_r - r * (1.0 - math.cos(math.radians(45.0))), y_bot + r * (1.0 - math.sin(math.radians(45.0)))),
                        (x_r, y_bot + r),
                        (x_r, y_top - r),
                        (x_r - r * (1.0 - math.cos(math.radians(45.0))), y_top - r * (1.0 - math.sin(math.radians(45.0)))),
                        (x_r - r, y_top),
                        (x_l + r, y_top),
                        (x_l + r * (1.0 - math.cos(math.radians(45.0))), y_top - r * (1.0 - math.sin(math.radians(45.0)))),
                        (x_l, y_top - r),
                        (x_l, y_bot + r),
                        (x_l + r * (1.0 - math.cos(math.radians(45.0))), y_bot + r * (1.0 - math.sin(math.radians(45.0)))),
                        (x_l + r, y_bot),
                    ]
                    msp.add_lwpolyline(
                        pts,
                        dxfattribs={"layer": "VGEOM", "color": self.COL_RED, "lineweight": 22}
                    )

                    # Central vertical divider line
                    msp.add_line(
                        (x, y_bot),
                        (x, y_top),
                        dxfattribs={"layer": "VGEOM", "color": self.COL_RED, "lineweight": 18}
                    )

                    # Left column: CH:<chainage> (rotated 90°)
                    txt_ch = f"CH:{c:.2f}"
                    msp.add_text(
                        txt_ch,
                        dxfattribs={
                            "height": 1.6,
                            "rotation": 90,
                            "layer": "VGEOM",
                            "style": "ESP_STYLE",
                            "color": self.COL_RED,
                        }
                    ).set_placement((x - dx / 2.0, (y_bot + y_top) / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

                    # Right column: FL:<elevation> (rotated 90°)
                    txt_fl = f"FL:{elev:.3f}"
                    msp.add_text(
                        txt_fl,
                        dxfattribs={
                            "height": 1.6,
                            "rotation": 90,
                            "layer": "VGEOM",
                            "style": "ESP_STYLE",
                            "color": self.COL_RED,
                        }
                    ).set_placement((x + dx / 2.0, (y_bot + y_top) / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

        # 5. Render sloped/horizontal gradient lines & text labels for each merged span
        for c_start, c_end, g_pct in spans:
            if c_end <= start_ch or c_start >= end_ch:
                continue

            x_s = max(origin_x, self._ch_to_x(c_start, origin_x, start_ch=start_ch))
            x_e = min(max_right_x, self._ch_to_x(c_end, origin_x, start_ch=start_ch))

            if x_e <= x_s + 1.0:
                continue

            span_len_m = c_end - c_start
            is_level = abs(g_pct) < 0.001

            # Determine gradient line endpoints & rotation angle
            if is_level:
                p1 = (x_s, cy)
                p2 = (x_e, cy)
                rot_deg = 0.0
                grad_str = "LEVEL"
            elif g_pct > 0:
                # RISE: bottom-left corner (x_s, bot_y) to top-right corner (x_e, row_top_y)
                p1 = (x_s, bot_y)
                p2 = (x_e, row_top_y)
                rot_deg = math.degrees(math.atan2(rh, x_e - x_s))
                inv_g = int(round(100.0 / g_pct))
                grad_str = f"RISE 1 IN {inv_g}"
            else:
                # FALL: top-left corner (x_s, row_top_y) to bottom-right corner (x_e, bot_y)
                p1 = (x_s, row_top_y)
                p2 = (x_e, bot_y)
                rot_deg = -math.degrees(math.atan2(rh, x_e - x_s))
                inv_g = int(round(100.0 / abs(g_pct)))
                grad_str = f"FALL 1 IN {inv_g}"

            msp.add_line(
                p1, p2,
                dxfattribs={"layer": "VGEOM", "color": self.COL_RED, "lineweight": 18}
            )

            # Draw Parallel Text Labels (LENGTH above, GRADIENT below)
            x_mid = (x_s + x_e) / 2.0
            y_mid = (p1[1] + p2[1]) / 2.0
            length_str = f"LENGTH={span_len_m:.2f}"

            # Offset vectors perpendicular to slope direction
            rad = math.radians(rot_deg)
            nx = -math.sin(rad)
            ny = math.cos(rad)

            # Top label: LENGTH
            tx1 = x_mid + nx * 1.5
            ty1 = y_mid + ny * 1.5
            txt1 = msp.add_text(
                length_str,
                dxfattribs={
                    "height": 1.7,
                    "rotation": rot_deg,
                    "layer": "VGEOM",
                    "style": "ESP_STYLE",
                    "color": self.COL_RED,
                }
            )
            txt1.set_placement((tx1, ty1), align=TextEntityAlignment.BOTTOM_CENTER)

            # Bottom label: RISE / FALL / LEVEL
            tx2 = x_mid - nx * 2.2
            ty2 = y_mid - ny * 2.2
            txt2 = msp.add_text(
                grad_str,
                dxfattribs={
                    "height": 1.7,
                    "rotation": rot_deg,
                    "layer": "VGEOM",
                    "style": "ESP_STYLE",
                    "color": self.COL_RED,
                }
            )
            txt2.set_placement((tx2, ty2), align=TextEntityAlignment.TOP_CENTER)

    # -------------------------------------------------------------------------
    # Row 1 — HORIZONTAL GEOMETRY
    # -------------------------------------------------------------------------
    def _draw_horizontal_geometry_row(
        self,
        msp: Any,
        ip_df: Optional[pd.DataFrame],
        origin_x: float,
        row_top_y: float,
        draw_length_mm: float,
        start_ch: float = 0.0,
        max_y: Optional[float] = None,
    ) -> None:
        """Draw the Horizontal Geometry row."""
        rh = self.row_height
        cy = row_top_y - rh / 2.0
        end_ch = start_ch + (draw_length_mm / 1000.0) * self.h_scale

        if ip_df is None:
            ip_df = pd.DataFrame()

        # ── 1. Identify Curve Spans to Break Straight Baseline at Curves ─────
        curve_spans: list[tuple[float, float]] = []
        if not ip_df.empty:
            for idx, row in ip_df.iterrows():
                radius = float(row.get("radius_m", 0.0))
                ip_ch = float(row.get("chainage_m", 0.0))
                t_len = float(row.get("tangent_length_m", 0.0))
                tptc_ch = float(row.get("tptc_chainage_m", max(0.0, ip_ch - t_len)))
                ctpt_ch = float(row.get("ctpt_chainage_m", ip_ch + t_len))
                curve_len = float(row.get("curve_length_m", row.get("circular_curve_length_m", row.get("length_m", ctpt_ch - tptc_ch))))
                if radius <= 0 or curve_len < 20.0:
                    continue
                if tptc_ch < end_ch and ctpt_ch > start_ch:
                    x0 = max(origin_x, self._ch_to_x(tptc_ch, origin_x, start_ch=start_ch))
                    x1 = min(origin_x + draw_length_mm, self._ch_to_x(ctpt_ch, origin_x, start_ch=start_ch))
                    if x1 > x0:
                        curve_spans.append((x0, x1))

        curve_spans.sort(key=lambda s: s[0])

        # Draw straight baseline ONLY on non-curved segments (breaks seamlessly at curves!)
        curr_x = origin_x
        for span_x0, span_x1 in curve_spans:
            if span_x0 > curr_x + 0.1:
                msp.add_line(
                    (curr_x, cy),
                    (span_x0, cy),
                    dxfattribs={"layer": "HGEOM", "color": self.COL_RED, "lineweight": 18}
                )
            curr_x = max(curr_x, span_x1)

        if curr_x < origin_x + draw_length_mm - 0.1:
            msp.add_line(
                (curr_x, cy),
                (origin_x + draw_length_mm, cy),
                dxfattribs={"layer": "HGEOM", "color": self.COL_RED, "lineweight": 18}
            )

        if len(curve_spans) == 0:
            txt_str = msp.add_text(
                "STRAIGHT ALIGNMENT",
                dxfattribs={
                    "height": 2.2,
                    "layer": "HGEOM",
                    "style": "ESP_STYLE",
                    "color": self.COL_RED,
                }
            )
            txt_str.set_placement(
                (origin_x + draw_length_mm / 2.0, cy + 1.6),
                align=TextEntityAlignment.BOTTOM_CENTER
            )

        # ── 2. Draw 3-Compartment Curve Diagrams (TRL, R, CL) inside Horizontal Geometry Row ──
        drawn_curve_spans: list[tuple[float, float]] = []

        for idx, row in ip_df.iterrows():
            ip_ch   = float(row.get("chainage_m", 0.0))
            radius  = float(row.get("radius_m", 0.0))
            tcl     = float(row.get("curve_length_m", 0.0))
            trl     = float(row.get("transition_length_m", 0.0))
            ccl     = float(row.get("circular_curve_length_m", 0.0))
            if ccl <= 0.0:
                ccl = max(1.0, tcl - 2.0 * trl)

            if radius <= 0 or tcl < 10.0:
                continue

            tptc_ch = float(row.get("tptc_chainage_m", ip_ch - tcl / 2.0))
            tpcc_ch = float(row.get("tpcc_chainage_m", tptc_ch + trl))
            ctpc_ch = float(row.get("ctpc_chainage_m", tpcc_ch + ccl))
            ctpt_ch = float(row.get("ctpt_chainage_m", ctpc_ch + trl))

            if tptc_ch >= end_ch or ctpt_ch <= start_ch:
                continue

            max_right_x = origin_x + draw_length_mm
            x_tptc = self._ch_to_x(tptc_ch, origin_x, start_ch=start_ch)
            x_tpcc = self._ch_to_x(tpcc_ch, origin_x, start_ch=start_ch)
            x_ctpc = self._ch_to_x(ctpc_ch, origin_x, start_ch=start_ch)
            x_ctpt = self._ch_to_x(ctpt_ch, origin_x, start_ch=start_ch)

            bot_y = row_top_y - rh
            y_top = row_top_y

            # 1. Vertical divider lines at transition stations
            div_xs = [x_tptc, x_ctpt]
            if trl > 0.0:
                div_xs.extend([x_tpcc, x_ctpc])

            for vx in div_xs:
                if origin_x - 0.5 <= vx <= max_right_x + 0.5:
                    msp.add_line(
                        (vx, bot_y),
                        (vx, y_top),
                        dxfattribs={"layer": "HGEOM", "color": self.COL_RED, "lineweight": 18}
                    )

            # Direction check: RHS curves arch UPWARDS (+), LHS curves dip DOWNWARDS (-)
            direction = str(row.get("direction", row.get("turn_direction", row.get("turn", "RHS")))).strip().upper()
            is_lhs = ("L" in direction) or ("LEFT" in direction)
            h_deflect = max(1.5, (y_top - 0.6) - cy if not is_lhs else cy - (bot_y + 0.6))

            # 2. Render 3 Distinct Compartment Arcs (Convex Arching Geometry matching Indian Railways Legend)
            # (a) Entry Transition Arc (TPTC to TPCC)
            if trl > 0.0 and x_tpcc > x_tptc:
                n_ent = max(15, int((x_tpcc - x_tptc) / 1.0))
                xs_ent = np.linspace(x_tptc, x_tpcc, n_ent)
                u_ent = (xs_ent - x_tptc) / (x_tpcc - x_tptc)
                # Convex arching ramp: takes off directly from baseline cy, flattens at top
                ramp = (2.0 * u_ent - u_ent ** 2) * h_deflect
                ys_ent = cy + ramp if not is_lhs else cy - ramp
                pts_ent = [(float(px), float(py)) for px, py in zip(xs_ent, ys_ent) if origin_x - 0.5 <= px <= max_right_x + 0.5]
                if len(pts_ent) >= 2:
                    msp.add_lwpolyline(pts_ent, dxfattribs={"layer": "HGEOM", "color": self.COL_RED, "lineweight": 18})

                # Entry text: TRL: ... (in opposite half of compartment)
                x_ent = (x_tptc + x_tpcc) / 2.0
                if origin_x <= x_ent <= max_right_x:
                    y_trl = (cy - (cy - bot_y) * 0.40) if not is_lhs else (cy + (y_top - cy) * 0.40)
                    msp.add_text(
                        f"TRL: {trl:.0f}",
                        dxfattribs={
                            "height": 2.0,
                            "layer": "HGEOM",
                            "style": "ESP_STYLE",
                            "color": self.COL_RED,
                        }
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
                # Full circular curve arc (rising from cy up to peak and returning to cy)
                circ_profile = np.maximum(0.0, 1.0 - u_circ ** 2) * h_deflect
                ys_circ = cy + circ_profile if not is_lhs else cy - circ_profile
                pts_circ = [(float(px), float(py)) for px, py in zip(xs_circ, ys_circ) if origin_x - 0.5 <= px <= max_right_x + 0.5]
                if len(pts_circ) >= 2:
                    msp.add_lwpolyline(pts_circ, dxfattribs={"layer": "HGEOM", "color": self.COL_RED, "lineweight": 18})

                # Circular text: R: ... (upper) and CL: ... (lower)
                x_circ = (c_start_x + c_end_x) / 2.0
                if origin_x <= x_circ <= max_right_x:
                    msp.add_text(
                        f"R: {radius:.1f}",
                        dxfattribs={
                            "height": 2.0,
                            "layer": "HGEOM",
                            "style": "ESP_STYLE",
                            "color": self.COL_RED,
                        }
                    ).set_placement((x_circ, cy + (y_top - cy) * 0.45), align=TextEntityAlignment.MIDDLE_CENTER)

                    msp.add_text(
                        f"CL: {ccl:.3f}",
                        dxfattribs={
                            "height": 2.0,
                            "layer": "HGEOM",
                            "style": "ESP_STYLE",
                            "color": self.COL_RED,
                        }
                    ).set_placement((x_circ, cy - (cy - bot_y) * 0.45), align=TextEntityAlignment.MIDDLE_CENTER)

            # (c) Exit Transition Arc (CTPC to CTPT)
            if trl > 0.0 and x_ctpt > x_ctpc:
                n_ext = max(15, int((x_ctpt - x_ctpc) / 1.0))
                xs_ext = np.linspace(x_ctpc, x_ctpt, n_ext)
                u_ext = (xs_ext - x_ctpc) / (x_ctpt - x_ctpc)
                # Convex arching descent: flattens at top, returns smoothly to baseline cy
                ramp_ext = (1.0 - u_ext ** 2) * h_deflect
                ys_ext = cy + ramp_ext if not is_lhs else cy - ramp_ext
                pts_ext = [(float(px), float(py)) for px, py in zip(xs_ext, ys_ext) if origin_x - 0.5 <= px <= max_right_x + 0.5]
                if len(pts_ext) >= 2:
                    msp.add_lwpolyline(pts_ext, dxfattribs={"layer": "HGEOM", "color": self.COL_RED, "lineweight": 18})

                # Exit text: TRL: ... (in opposite half of compartment)
                x_ext = (x_ctpc + x_ctpt) / 2.0
                if origin_x <= x_ext <= max_right_x:
                    y_trl_ext = (cy - (cy - bot_y) * 0.40) if not is_lhs else (cy + (y_top - cy) * 0.40)
                    msp.add_text(
                        f"TRL: {trl:.0f}",
                        dxfattribs={
                            "height": 2.0,
                            "layer": "HGEOM",
                            "style": "ESP_STYLE",
                            "color": self.COL_RED,
                        }
                    ).set_placement((x_ext, y_trl_ext), align=TextEntityAlignment.MIDDLE_CENTER)

            drawn_curve_spans.append((tptc_ch, ctpt_ch))

        # ── Draw 'STRAIGHT' Labels for Tangent Sections Between/Outside Curves ──
        drawn_curve_spans.sort(key=lambda s: s[0])
        straight_segments: list[tuple[float, float]] = []

        curr_s = start_ch
        for (c_start, c_end) in drawn_curve_spans:
            if c_start > curr_s + 50.0:
                straight_segments.append((curr_s, c_start))
            curr_s = max(curr_s, c_end)
        if end_ch > curr_s + 50.0:
            straight_segments.append((curr_s, end_ch))

        for (s_start, s_end) in straight_segments:
            seg_len = s_end - s_start
            mid_ch = (s_start + s_end) / 2.0
            x_mid = min(origin_x + draw_length_mm - 15.0, max(origin_x + 15.0, self._ch_to_x(mid_ch, origin_x, start_ch=start_ch)))

            straight_str = f"Length = {seg_len:.1f}m"
            msp.add_text(
                straight_str,
                dxfattribs={
                    "height": 2.0,
                    "layer": "HGEOM",
                    "style": "ESP_STYLE",
                    "color": self.COL_RED,
                }
            ).set_placement((x_mid, cy + 1.5), align=TextEntityAlignment.MIDDLE_CENTER)

    # -------------------------------------------------------------------------
    # Data Band Orchestrator
    # -------------------------------------------------------------------------
    def _draw_data_rows(
        self,
        msp: Any,
        chainages: np.ndarray,
        egl: np.ndarray,
        fl: np.ndarray,
        origin_x: float,
        origin_y: float,
        draw_length_mm: float,
        label_width: float,
        max_y: float,
        vip_df: Optional[pd.DataFrame] = None,
        ip_df: Optional[pd.DataFrame] = None,
        start_ch: float = 0.0,
        profile_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Draw L-Section data bands dynamically based on available data.

        Omit band rows and label columns if values are not available.
        Candidate row order (top to bottom):
            - VERTICAL GEOMETRY     (Red)
            - HORIZONTAL GEOMETRY   (Red)
            - CUT (-) FILL (+)      (Red)
            - DIFF PRL & ERL        (Red — rendered only if surveyed data available)
            - PROP RAIL LEVEL       (Red)
            - PROP. FORMATION LEVEL (Red)
            - EX RAIL LEVEL         (Black — rendered only if surveyed data available)
            - EX. FORMATION LEVEL   (Black — rendered only if surveyed data available)
            - GROUND LEVEL          (Black)
            - CHAINAGE              (Black)
        """
        rh = self.row_height

        # ── Ensure strictly regular interval (e.g. 20m) for data rows and ticks ──
        interval_m = float(self.config.get("chainage", {}).get("interval_m", 20.0))
        rem = chainages % interval_m
        mask_reg = (np.minimum(rem, interval_m - rem) < 0.5)
        if not np.all(mask_reg):
            chainages = np.round(chainages[mask_reg] / interval_m) * interval_m
            egl = egl[mask_reg]
            fl = fl[mask_reg]
            _, unique_idx = np.unique(chainages, return_index=True)
            chainages = chainages[unique_idx]
            egl = egl[unique_idx]
            fl = fl[unique_idx]

        # ── Derived data ──────────────────────────────────────────────────────
        prl    = fl + self.track_buildup_m   # PRL = PFL + 762mm
        cf     = fl - egl                    # Cut(-) Fill(+): negative = cut
        
        # Existing Track & Formation Levels — populate from profile_df or fallback to PFL
        efl = np.full(len(chainages), np.nan)
        erl = np.full(len(chainages), np.nan)

        if profile_df is not None:
            if "existing_formation_level_m" in profile_df.columns and "chainage_m" in profile_df.columns:
                prof_ch = profile_df["chainage_m"].values
                prof_efl = profile_df["existing_formation_level_m"].values
                for idx_c, c_val in enumerate(chainages):
                    mask = np.isclose(prof_ch, c_val, atol=0.5)
                    if np.any(mask):
                        val = prof_efl[mask][0]
                        if not np.isnan(val) and val != 0.0:
                            efl[idx_c] = val

            if "existing_rail_level_m" in profile_df.columns and "chainage_m" in profile_df.columns:
                prof_ch = profile_df["chainage_m"].values
                prof_erl = profile_df["existing_rail_level_m"].values
                for idx_c, c_val in enumerate(chainages):
                    mask = np.isclose(prof_ch, c_val, atol=0.5)
                    if np.any(mask):
                        val = prof_erl[mask][0]
                        if not np.isnan(val) and val != 0.0:
                            erl[idx_c] = val

        # If efl is set but erl is not, calculate erl = efl + 0.738m ONLY if efl has valid non-NaN values
        erl = np.where(np.isnan(erl) & ~np.isnan(efl), efl + 0.738, erl)
        diff = np.where(np.isnan(erl), np.nan, prl - erl)

        def _has_valid_data(arr: Optional[np.ndarray]) -> bool:
            return arr is not None and len(arr) > 0 and bool(np.any(np.isfinite(arr)))

        # Define candidate bands in standard Indian Railways order
        candidate_bands: list[dict[str, Any]] = [
            {
                "id": "vgeom",
                "label": "VERTICAL GEOMETRY",
                "color": self.COL_RED,
                "is_geom": True,
                "available": (vip_df is not None and not vip_df.empty) or _has_valid_data(fl),
            },
            {
                "id": "hgeom",
                "label": "HORIZONTAL GEOMETRY",
                "color": self.COL_RED,
                "is_geom": True,
                "available": (ip_df is not None and not ip_df.empty),
            },
            {
                "id": "cut_fill",
                "label": "PRO. BANK/CUTTING (THROUGH) PROP. DN MAIN",
                "color": self.COL_RED,
                "is_geom": False,
                "available": _has_valid_data(cf),
            },
            {
                "id": "diff_prl_erl",
                "label": "DIF. BETWEEN PROP. DN MAIN (THROUGH) RAIL LEVEL & EXG. UP MAIN RAIL LEVEL",
                "color": self.COL_RED,
                "is_geom": False,
                "available": _has_valid_data(diff),
            },
            {
                "id": "prl",
                "label": "PROPOSED RAIL LEVEL",
                "color": self.COL_RED,
                "is_geom": False,
                "available": _has_valid_data(prl),
            },
            {
                "id": "pfl",
                "label": "PROPOSED FORMATION LEVEL",
                "color": self.COL_RED,
                "is_geom": False,
                "available": _has_valid_data(fl),
            },
            {
                "id": "erl",
                "label": "EXISTING RAIL LEVEL (EFL + 0.738M)",
                "color": self.COL_BLACK,
                "is_geom": False,
                "available": _has_valid_data(erl),
            },
            {
                "id": "efl",
                "label": "EXISTING FORMATION LEVEL (AS PER SITE)",
                "color": self.COL_BLACK,
                "is_geom": False,
                "available": _has_valid_data(efl),
            },
            {
                "id": "egl",
                "label": "GROUND LINE (Pro. DN line-1)",
                "color": self.COL_BLACK,
                "is_geom": False,
                "available": _has_valid_data(egl),
            },
            {
                "id": "chainage",
                "label": "CHAINAGE",
                "color": self.COL_BLACK,
                "is_geom": False,
                "available": len(chainages) > 0,
            },
        ]

        active_bands = [b for b in candidate_bands if b["available"]]

        # ── Map dynamic data bands from template if present ──
        if self.template and getattr(self.template, "data_bands", None):
            def _resolve_band_id(k_str: str, lbl_str: str) -> str:
                s = f"{k_str} {lbl_str}".lower()
                if any(x in s for x in ["diff", "dif."]):
                    return "diff_prl_erl"
                if any(x in s for x in ["cut", "bank", "fill"]):
                    return "cut_fill"
                if any(x in s for x in ["prop", "pro."]) and "rail" in s:
                    return "prl"
                if any(x in s for x in ["prop", "pro."]) and "formation" in s:
                    return "pfl"
                if any(x in s for x in ["exg", "ex.", "exist"]) and "rail" in s:
                    return "erl"
                if any(x in s for x in ["exg", "ex.", "exist"]) and "formation" in s:
                    return "efl"
                if any(x in s for x in ["ground", "egl", "ngl", "line-1", "pro. dn line"]):
                    return "egl"
                if any(x in s for x in ["chainage", "ch."]):
                    return "chainage"
                if any(x in s for x in ["grad", "vgeom", "vert"]):
                    return "vgeom"
                if any(x in s for x in ["curve", "hgeom", "horiz"]):
                    return "hgeom"
                key_defaults = {
                    "vgeom": "vgeom",
                    "gradient": "vgeom",
                    "vertical_geometry": "vgeom",
                    "gradient_and_vertical_curves": "vgeom",
                    "hgeom": "hgeom",
                    "horizontal_curve": "hgeom",
                    "horizontal_curves": "hgeom",
                    "horizontal_geometry": "hgeom",
                    "cut_fill": "cut_fill",
                    "cutting_filling": "cut_fill",
                    "diff_prl_erl": "diff_prl_erl",
                    "prl": "prl",
                    "proposed_rail_level": "prl",
                    "pfl": "pfl",
                    "proposed_formation_level": "pfl",
                    "formation_level": "pfl",
                    "erl": "erl",
                    "existing_rail_level": "erl",
                    "efl": "efl",
                    "existing_formation_level": "efl",
                    "egl": "egl",
                    "existing_ground_level": "egl",
                    "ground_level": "egl",
                    "chainage": "chainage",
                }
                return key_defaults.get(k_str, k_str)

            cand_dict = {b["id"]: b for b in candidate_bands}
            ordered_bands = []
            for t_band in self.template.data_bands:
                if not getattr(t_band, "enabled", True):
                    continue
                k = getattr(t_band, "key", "").lower().replace("-", "_").replace(" ", "_")
                lbl = getattr(t_band, "label", None) or getattr(t_band, "title", None) or k
                mapped_id = _resolve_band_id(k, lbl)
                if mapped_id in cand_dict:
                    b_info = dict(cand_dict[mapped_id])
                    if lbl:
                        b_info["label"] = lbl
                    h_mm = getattr(t_band, "height_mm", None) or getattr(t_band, "heightMm", None)
                    if h_mm and float(h_mm) > 0:
                        b_info["height_mm"] = float(h_mm)
                    if b_info["available"]:
                        ordered_bands.append(b_info)
            if ordered_bands:
                active_bands = ordered_bands

        num_rows = len(active_bands)
        if num_rows == 0:
            return

        row_heights = [float(b.get("height_mm", rh)) for b in active_bands]
        cum_offsets = [0.0]
        for h in row_heights:
            cum_offsets.append(cum_offsets[-1] + h)
        total_h = cum_offsets[-1]

        # ── Table frame ──────────────────────────────────────────────────────
        row_colors = [b["color"] for b in active_bands]
        self._draw_table_frame(msp, origin_x, origin_y, draw_length_mm, num_rows, label_width, row_colors=row_colors, row_heights=row_heights)

        # ── Label column ─────────────────────────────────────────────────────
        labels = [(b["label"], b["color"]) for b in active_bands]
        self._draw_label_column(msp, origin_x, origin_y, labels, label_width, row_heights=row_heights)

        # ── 1.0mm Station Ticks at Top & Bottom of Data Rows (skip geometry rows and chainage row) ──
        tick_h = float(self.config.get("drawing", {}).get("tick_line_length_mm", 1.0))
        for r_idx, band in enumerate(active_bands):
            if band["is_geom"] or band["id"] == "chainage":
                continue
            y_top = origin_y - cum_offsets[r_idx]
            y_bot = origin_y - cum_offsets[r_idx + 1]
            for c in chainages:
                x = self._ch_to_x(c, origin_x, start_ch=start_ch)
                msp.add_line((x, y_top), (x, y_top - tick_h), dxfattribs={"layer": "DATA_BORDER", "color": self.COL_BLACK, "lineweight": 13})
                msp.add_line((x, y_bot), (x, y_bot + tick_h), dxfattribs={"layer": "DATA_BORDER", "color": self.COL_BLACK, "lineweight": 13})

        # ── VC Annotations (Vertical Drop lines + Text) ──────────────────────
        self._draw_vc_data_columns(
            msp, vip_df, chainages, egl, fl, origin_x, origin_y, total_h
        )

        # ── Render active band contents ──────────────────────────────────────
        for r_idx, band in enumerate(active_bands):
            b_id = band["id"]
            row_top_y = origin_y - cum_offsets[r_idx]
            row_bot_y = origin_y - cum_offsets[r_idx + 1]
            row_cy = (row_top_y + row_bot_y) / 2.0

            if b_id == "vgeom":
                self._draw_vertical_geometry_row(msp, vip_df, origin_x, row_top_y, draw_length_mm=draw_length_mm, start_ch=start_ch)
            elif b_id == "hgeom":
                self._draw_horizontal_geometry_row(msp, ip_df, origin_x, row_top_y, draw_length_mm=draw_length_mm, start_ch=start_ch, max_y=max_y)
            elif b_id == "cut_fill":
                max_text_h = float(self.config.get("drawing", {}).get("text_height_data_mm", 2.5))
                text_h = max(2.0, max_text_h)
                positions = self._get_label_positions(chainages, origin_x, text_h, start_ch=start_ch)
                for idx, x in positions:
                    if idx >= len(cf):
                        continue
                    v = cf[idx]
                    if np.isnan(v):
                        continue
                    val_str = f"{v:.3f}"
                    cf_col = self.COL_BLACK if v < 0 else self.COL_RED
                    cf_layer = "DATA_BLACK" if v < 0 else "DATA_RED"
                    txt = msp.add_text(
                        val_str,
                        dxfattribs={
                            "height": text_h,
                            "rotation": 90,
                            "layer": cf_layer,
                            "style": "ESP_STYLE",
                            "color": cf_col,
                        }
                    )
                    txt.set_placement((x, row_cy), align=TextEntityAlignment.MIDDLE_CENTER)
            elif b_id == "diff_prl_erl":
                self._draw_row_values(msp, chainages, diff, origin_x, row_cy, self.COL_RED, fmt="{:.3f}", start_ch=start_ch)
            elif b_id == "prl":
                self._draw_row_values(msp, chainages, prl, origin_x, row_cy, self.COL_RED, fmt="{:.3f}", start_ch=start_ch)
            elif b_id == "pfl":
                self._draw_row_values(msp, chainages, fl, origin_x, row_cy, self.COL_RED, fmt="{:.3f}", start_ch=start_ch)
            elif b_id == "erl":
                self._draw_row_values(msp, chainages, erl, origin_x, row_cy, self.COL_BLACK, fmt="{:.3f}", start_ch=start_ch)
            elif b_id == "efl":
                self._draw_row_values(msp, chainages, efl, origin_x, row_cy, self.COL_BLACK, fmt="{:.3f}", start_ch=start_ch)
            elif b_id == "egl":
                self._draw_row_values(msp, chainages, egl, origin_x, row_cy, self.COL_BLACK, fmt="{:.3f}", start_ch=start_ch)
            elif b_id == "chainage":
                interval_m = float(self.config.get("chainage", {}).get("interval_m", 20.0))
                col_w_mm   = (interval_m / self.h_scale) * 1000.0
                max_text_h = float(self.config.get("drawing", {}).get("text_height_data_mm", 2.5))
                text_h_ch  = min(max_text_h, max(1.0, col_w_mm * 0.80))
                positions = self._get_label_positions(chainages, origin_x, text_h_ch, start_ch=start_ch)
                for idx, x in positions:
                    if idx >= len(chainages):
                        continue
                    c = chainages[idx]
                    ch_str = self._format_chainage(c)
                    txt = msp.add_text(
                        ch_str,
                        dxfattribs={
                            "height": text_h_ch,
                            "rotation": 90,
                            "layer": "CHAINAGE",
                            "style": "ESP_STYLE",
                            "color": self.COL_BLACK,
                        }
                    )
                    txt.set_placement((x, row_cy), align=TextEntityAlignment.MIDDLE_CENTER)

                    # Draw KM Post milestone symbol hanging BELOW the bottom chainage line
                    is_whole_km = (abs(c % 1000.0) < 1.0) or (abs(c % 1000.0 - 1000.0) < 1.0)
                    if is_whole_km:
                        km_val = int(round(c / 1000.0))
                        self._draw_km_milestone_symbol(
                            msp=msp,
                            x=x,
                            y_line=row_bot_y,
                            km_val=km_val,
                            layer="CHAINAGE",
                        )



    # -------------------------------------------------------------------------
    # Topomap helper (plan view — future use)
    # -------------------------------------------------------------------------
    def _draw_topomap(
        self,
        msp: Any,
        min_e: float,
        max_e: float,
        min_n: float,
        max_n: float,
        to_plan_coords: Any,
    ) -> None:
        """Read shapefiles from ROB_Topomap_SHP and plot within the bbox."""
        shp_dir = Path("D:/ESP/ROB_Topomap_SHP")
        if not shp_dir.exists():
            self.logger.warning("Topomap shapefile directory not found: %s", shp_dir)
            return

        layer_mapping = {
            "RAILWAY-TRACK.shp":   "RAILWAY-TRACK",
            "ROAD-CENTERLINE.shp": "ROAD-CENTERLINE",
            "CART-TRACK.shp":      "ROAD-CENTERLINE",
            "BRIDGE.shp":          "BRIDGE",
            "DRAIN.shp":           "DRAIN",
            "TREEE.shp":           "TREE",
            "PUCCA-BUILDING.shp":  "BUILDINGS-PUCCA",
            "ROW.shp":             "ROW",
            "LEVEL-CROSSING.shp":  "LEVEL-CROSSING",
        }

        bbox_geom = box(min_e, min_n, max_e, max_n)
        for shp_file, layer_name in layer_mapping.items():
            shp_path = shp_dir / shp_file
            if not shp_path.exists():
                continue
            try:
                gdf = gpd.read_file(shp_path, bbox=bbox_geom)
                if gdf.empty:
                    continue
                for geom in gdf.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    gt = geom.geom_type
                    if gt == "Point":
                        msp.add_circle(
                            to_plan_coords(geom.x, geom.y),
                            radius=1.0,
                            dxfattribs={"layer": layer_name}
                        )
                    elif gt in ("LineString", "LinearRing"):
                        msp.add_lwpolyline(
                            [to_plan_coords(x, y) for x, y in geom.coords],
                            dxfattribs={"layer": layer_name}
                        )
                    elif gt == "Polygon":
                        msp.add_lwpolyline(
                            [to_plan_coords(x, y) for x, y in geom.exterior.coords],
                            dxfattribs={"layer": layer_name}
                        )
                    elif gt == "MultiLineString":
                        for ln in geom.geoms:
                            msp.add_lwpolyline(
                                [to_plan_coords(x, y) for x, y in ln.coords],
                                dxfattribs={"layer": layer_name}
                            )
                    elif gt == "MultiPolygon":
                        for poly in geom.geoms:
                            msp.add_lwpolyline(
                                [to_plan_coords(x, y) for x, y in poly.exterior.coords],
                                dxfattribs={"layer": layer_name}
                            )
            except Exception as exc:
                self.logger.error("Failed to process topomap file %s: %s", shp_file, exc)

    # -------------------------------------------------------------------------
    # Sheet Rendering Helper
    # -------------------------------------------------------------------------
    def _render_sheet(
        self,
        chainages: np.ndarray,
        egl: np.ndarray,
        fl: np.ndarray,
        dxf_out: Path,
        pdf_out: Path,
        vip_df: Optional[pd.DataFrame] = None,
        align_res: Optional[Any] = None,
        topo_res: Optional[Any] = None,
        ip_df: Optional[pd.DataFrame] = None,
        start_ch: float = 0.0,
        profile_df: Optional[pd.DataFrame] = None,
        bridge_df: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Render a single DXF & PDF L-Section sheet for a chainage range."""
        doc = ezdxf.new(dxfversion="R2018")
        doc.units = units.MM
        msp = doc.modelspace()
        self._setup_layers(doc)

        start_ch_val = float(chainages[0]) if len(chainages) > 0 else start_ch
        end_ch_val = float(chainages[-1]) if len(chainages) > 0 else start_ch
        draw_length_m = end_ch_val - start_ch_val

        labels_text = [
            "VERTICAL GEOMETRY", "HORIZONTAL GEOMETRY", "CUT (-) FILL (+)",
            "DIFF PRL & ERL", "PROP RAIL LEVEL", "PROP. FORMATION LEVEL",
            "EX RAIL LEVEL", "EX. FORMATION LEVEL", "GROUND LEVEL", "CHAINAGE"
        ]
        max_label_chars = max(len(text) for text in labels_text)
        dynamic_label_width = float(max_label_chars * 2.2 + 25.0)

        sheet_width = float(self.config.get("drawing", {}).get("sheet_width_mm", 1189.0))
        margin_right = 20.0
        origin_x = dynamic_label_width + 15.0

        # Respect configured horizontal scale from config.yaml (e.g. H=1:5000)
        if self.h_scale > 0 and draw_length_m > 0:
            draw_len_mm = (draw_length_m / self.h_scale) * 1000.0
        else:
            draw_len_mm = max(500.0, sheet_width - origin_x - margin_right)
        draw_length_mm = draw_len_mm
        sheet_width = max(sheet_width, origin_x + draw_len_mm + margin_right)

        num_data_rows = 10
        data_band_h = num_data_rows * self.row_height   # 250 mm
        profile_area_h = 200.0   # target profile drawing height mm
        margin_top = 20.0
        margin_mid = 10.0
        margin_bot = 20.0

        sheet_height = margin_bot + data_band_h + margin_mid + profile_area_h + margin_top

        doc.header["$LIMMIN"] = (0.0, 0.0)
        doc.header["$LIMMAX"] = (sheet_width, sheet_height)
        doc.header["$EXTMIN"] = (0.0, 0.0, 0.0)
        doc.header["$EXTMAX"] = (sheet_width, sheet_height, 0.0)

        try:
            vports = doc.viewports.get("*Active")
            if vports:
                vport = vports[0]
                vport.dxf.center = (sheet_width / 2.0, sheet_height / 2.0)
                vport.dxf.height = sheet_height
        except Exception:
            pass

        origin_x = dynamic_label_width + 20.0
        data_origin_y = margin_bot + data_band_h
        profile_origin_y = data_origin_y

        valid_egl = [e for e in egl if e != 0.0]
        start_egl = float(valid_egl[0]) if valid_egl else 0.0
        datum_elev = math.floor((start_egl - 20.0) / 5.0) * 5.0

        profile_max_h = float(
            self.config.get("drawing", {}).get("profile_max_height_m", 15.0)
        )
        max_grid_elev = datum_elev + profile_max_h
        max_y = self._elev_to_y(max_grid_elev, datum_elev, profile_origin_y)

        # 1. Profile View
        self._draw_profile_view(
            msp,
            chainages,
            egl,
            fl,
            origin_x,
            profile_origin_y,
            datum_elev,
            draw_length_mm,
            label_width=dynamic_label_width,
            vip_df=vip_df,
            align_res=align_res,
            topo_res=topo_res,
            ip_df=ip_df,
            start_ch=start_ch,
            bridge_df=bridge_df,
        )

        # 2. Data Band
        self._draw_data_rows(
            msp,
            chainages,
            egl,
            fl,
            origin_x,
            data_origin_y,
            draw_length_mm,
            dynamic_label_width,
            max_y,
            vip_df=vip_df,
            ip_df=ip_df,
            start_ch=start_ch,
            profile_df=profile_df,
        )

        # 3. Save DXF & PDF
        self._apply_modelspace_scaling(msp)
        doc.saveas(str(dxf_out))
        self._export_to_pdf(dxf_out, pdf_out)

        return {
            "dxf_path": str(dxf_out),
            "pdf_path": str(pdf_out),
            "paper_length_mm": sheet_width,
            "paper_height_mm": sheet_height,
            "datum_elev_m": datum_elev,
            "start_ch_m": start_ch_val,
            "end_ch_m": end_ch_val,
            "draw_length_m": draw_length_m,
        }

    # -------------------------------------------------------------------------
    # Public Entry Point
    # -------------------------------------------------------------------------
    def draw(
        self,
        align_res: AlignmentResult,
        hgeom_res: HGeomResult,
        valign_res: VAlignResult,
        ew_res: EarthworkResult,
        topo_res: Optional[Any] = None,
        bridge_df: Optional[Any] = None,
    ) -> ESPResult:
        """Generate the complete L-Section (Longitudinal Section) DXF drawing(s).

        Supports multi-file chunking (max 5km per file/sheet per IR drawing standards).
        """
        self.logger.info("Starting L-Section DXF Generation...")
        try:
            df = ew_res.profile_dataframe
            if "chainage_m" not in df.columns:
                raise ESPDrawerError("profile_dataframe missing required column 'chainage_m'")

            chainages_all = df["chainage_m"].values
            if len(chainages_all) == 0:
                raise ESPDrawerError("profile_dataframe has 0 rows.")

            total_length_m = float(chainages_all[-1])
            vip_df = valign_res.vip_dataframe
            ip_df = hgeom_res.ip_dataframe if hgeom_res is not None else None

            max_len = self.max_lsection_length_m
            dxf_paths: list[Path] = []
            pdf_paths: list[Path] = []
            sheets_meta: list[dict[str, Any]] = []

            # ── 1. Complete Continuous Drawing (Full Length DXF) ───────────
            should_gen_complete = self.generate_complete_profile or (max_len <= 0.0)
            if should_gen_complete:
                dxf_out = self.outputs_dir / "engineering_scale_plan.dxf"
                dxf_out_alt = self.outputs_dir / "longitudinal_section.dxf"
                pdf_out = self.outputs_dir / "engineering_scale_plan.pdf"

                egl_all = df["elevation_m"].fillna(0).values
                fl_all = df["formation_level_m"].fillna(0).values

                meta = self._render_sheet(
                    chainages=chainages_all,
                    egl=egl_all,
                    fl=fl_all,
                    dxf_out=dxf_out,
                    pdf_out=pdf_out,
                    vip_df=vip_df,
                    align_res=align_res,
                    topo_res=topo_res,
                    ip_df=ip_df,
                    start_ch=0.0,
                    profile_df=df,
                    bridge_df=bridge_df,
                )
                import shutil
                shutil.copyfile(dxf_out, dxf_out_alt)
                if pdf_out.exists():
                    pdf_paths.append(pdf_out)
                sheets_meta.append(meta)
                self.logger.info("Saved Complete L-Section Profile: %s", dxf_out)

            sheet_dxf_paths: list[Path] = []
            # ── 2. Multi-Sheet Chunking (> max_len, e.g. > 5 km) ───────────
            should_gen_sheets = self.generate_sheet_drawings and (max_len > 0.0)
            if should_gen_sheets and total_length_m > max_len:
                num_sheets = int(math.ceil(total_length_m / max_len))
                self.logger.info(
                    "Alignment length (%.2f m) > max sheet length (%.2f m). Generating %d L-Section drawing sheets...",
                    total_length_m, max_len, num_sheets
                )

                for i in range(num_sheets):
                    start_ch = i * max_len
                    end_ch = min((i + 1) * max_len, total_length_m)

                    # Slice dataframe for this chunk
                    df_chunk = df[(df["chainage_m"] >= start_ch) & (df["chainage_m"] <= end_ch)].copy()
                    if df_chunk.empty:
                        continue

                    chainages_c = df_chunk["chainage_m"].values
                    egl_c = df_chunk["elevation_m"].fillna(0).values
                    fl_c = df_chunk["formation_level_m"].fillna(0).values

                    vip_c = None
                    if vip_df is not None and not vip_df.empty:
                        vip_c = vip_df[
                            (vip_df["chainage_m"] >= start_ch - 1000.0) &
                            (vip_df["chainage_m"] <= end_ch + 1000.0)
                        ].copy()

                    ip_c = None
                    if ip_df is not None and not ip_df.empty:
                        ip_c = ip_df[
                            (ip_df["chainage_m"] >= start_ch - 2000.0) &
                            (ip_df["chainage_m"] <= end_ch + 2000.0)
                        ].copy()

                    sheet_name = f"longitudinal_section_sheet{i+1}_km{start_ch/1000.0:.1f}-{end_ch/1000.0:.1f}"
                    dxf_sheet = self.outputs_dir / f"{sheet_name}.dxf"
                    pdf_sheet = self.outputs_dir / f"{sheet_name}.pdf"

                    sheet_meta = self._render_sheet(
                        chainages=chainages_c,
                        egl=egl_c,
                        fl=fl_c,
                        dxf_out=dxf_sheet,
                        pdf_out=pdf_sheet,
                        vip_df=vip_c,
                        align_res=align_res,
                        topo_res=topo_res,
                        ip_df=ip_c,
                        start_ch=start_ch,
                        profile_df=df_chunk,
                        bridge_df=bridge_df,
                    )
                    sheet_meta["sheet_index"] = i + 1
                    sheets_meta.append(sheet_meta)

                    sheet_dxf_paths.append(dxf_sheet)
                    if pdf_sheet.exists():
                        pdf_paths.append(pdf_sheet)

            dxf_paths = sheet_dxf_paths if sheet_dxf_paths else [self.outputs_dir / "engineering_scale_plan.dxf", self.outputs_dir / "longitudinal_section.dxf"]
            primary_dxf = sheet_dxf_paths[0] if sheet_dxf_paths else (self.outputs_dir / "engineering_scale_plan.dxf")

            return ESPResult(
                dxf_path=primary_dxf,
                dxf_paths=dxf_paths,
                pdf_paths=pdf_paths,
                metadata={
                    "num_sheets": len(dxf_paths),
                    "max_sheet_length_m": max_len,
                    "total_length_m": total_length_m,
                    "sheets": sheets_meta,
                },
            )

        except Exception as exc:
            self.logger.error("L-Section DXF/PDF generation failed: %s", exc)
            raise ESPDrawerError(f"DXF/PDF generation failed: {exc}") from exc


            self.logger.info("L-Section DXF & PDF Generation complete: %s, %s", dxf_out, pdf_out)
            return ESPResult(
                dxf_path=dxf_out,
                metadata={
                    "pdf_path":        pdf_out,
                    "paper_length_mm": sheet_width,
                    "paper_height_mm": sheet_height,
                    "datum_elev_m":    datum_elev,
                    "total_length_m":  total_length_m,
                }
            )

        except Exception as exc:
            self.logger.error("L-Section DXF/PDF generation failed: %s", exc)
            raise ESPDrawerError(f"DXF/PDF generation failed: {exc}") from exc

    # =========================================================================
    # ALIGNMENT PLAN (WORKING PLAN / PLAN VIEW) GENERATOR
    # =========================================================================
    def _render_alignment_sheet(
        self,
        chainages: np.ndarray,
        egl: np.ndarray,
        fl: np.ndarray,
        dxf_out: Path,
        pdf_out: Path,
        vip_df: Optional[pd.DataFrame],
        ip_df: Optional[pd.DataFrame],
        align_res: Optional[Any] = None,
        topo_res: Optional[Any] = None,
        bridge_df: Optional[Any] = None,
        start_ch: float = 0.0,
        profile_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        """Render one Alignment Plan DXF + PDF sheet for a chainage range.

        Identical sheet structure to _render_sheet (L-Section):
        - Same label column, same 10 data rows, same row helpers.
        - Top half is the 2-D plan view instead of the elevation profile.
        """
        doc = ezdxf.new(dxfversion="R2018")
        doc.units = units.MM
        msp = doc.modelspace()
        self._setup_layers(doc)

        start_ch_val  = float(chainages[0]) if len(chainages) > 0 else start_ch
        end_ch_val    = float(chainages[-1]) if len(chainages) > 0 else start_ch
        draw_length_m = end_ch_val - start_ch_val

        labels_text = [
            "VERTICAL GEOMETRY", "HORIZONTAL GEOMETRY", "CUT (-) FILL (+)",
            "DIFF PRL & ERL", "PROP RAIL LEVEL", "PROP. FORMATION LEVEL",
            "EX RAIL LEVEL", "EX. FORMATION LEVEL", "GROUND LEVEL", "CHAINAGE",
        ]
        max_label_chars = max(len(t) for t in labels_text)
        label_width     = float(max_label_chars * 2.2 + 25.0)

        sheet_width  = float(self.config.get("drawing", {}).get("sheet_width_mm", 1189.0))
        margin_right = 20.0
        origin_x     = label_width + 15.0

        # Respect configured horizontal scale from config.yaml (e.g. H=1:5000)
        if self.h_scale > 0 and draw_length_m > 0:
            draw_len_mm = (draw_length_m / self.h_scale) * 1000.0
        else:
            draw_len_mm = max(500.0, sheet_width - origin_x - margin_right)
        sheet_width = max(sheet_width, origin_x + draw_len_mm + margin_right)

        # ── Sheet geometry (Stacked Layout: Data Band + Profile View + Plan View) ──
        num_data_rows    = 10
        rh               = self.row_height
        data_band_h      = num_data_rows * rh               # 250 mm
        profile_area_h   = 120.0                           # 120 mm for Longitudinal Elevation Profile
                                                           # (was 160mm — reduced to give plan view more room;
                                                           #  at V=1:500, a 10m elevation range = 20mm paper,
                                                           #  so 120mm comfortably covers 60m of relief)
        plan_area_h      = float(
            self.config.get("drawing", {})
            .get("plan_view", {})
            .get("plan_area_height_mm", 350.0)
        )                                                  # 350 mm for 2D Horizontal Plan View (A0 layout)
        margin_top       = 20.0
        margin_mid       = float(self.config.get("drawing", {}).get("margin_mid_mm", 55.0))
        margin_bot       = 20.0

        sheet_height = margin_bot + data_band_h + margin_mid + profile_area_h + margin_mid + plan_area_h + margin_top

        doc.header["$LIMMIN"] = (0.0, 0.0)
        doc.header["$LIMMAX"] = (sheet_width, sheet_height)
        doc.header["$EXTMIN"] = (0.0, 0.0, 0.0)
        doc.header["$EXTMAX"] = (sheet_width, sheet_height, 0.0)
        try:
            vports = doc.viewports.get("*Active")
            if vports:
                vp = vports[0]
                vp.dxf.center = (sheet_width / 2.0, sheet_height / 2.0)
                vp.dxf.height = sheet_height
        except Exception:
            pass

        origin_x         = label_width + 20.0
        data_origin_y    = margin_bot + data_band_h                 # TOP of data band (Y=320mm)
        profile_origin_y = data_origin_y                            # Profile View origin (Y=320mm)
        profile_top_y    = profile_origin_y + profile_area_h        # TOP of profile view (Y=480mm)
        
        plan_bot_y       = profile_top_y + margin_mid               # Bottom of plan area (Y=495mm)
        plan_top_y       = plan_bot_y + plan_area_h                 # Top of plan area (Y=655mm)
        plan_origin_y    = (plan_bot_y + plan_top_y) / 2.0          # Centreline Y of plan view (Y=575mm)

        # ── Build data arrays ───────────────────────────────────────────────
        interval_m = float(self.config.get("chainage", {}).get("interval_m", 20.0))
        prl        = fl + self.track_buildup_m
        cf         = fl - egl
        erl        = prl.copy()
        efl_arr    = fl.copy()
        diff       = prl - erl

        # ── Middle Section: Surface Elevation Profile View (EGL ground line, PFL line, DATUM, Blue fill) ──
        valid_egl = [e for e in egl if e != 0.0]
        start_egl = float(valid_egl[0]) if valid_egl else 0.0
        datum_elev = math.floor((start_egl - 20.0) / 5.0) * 5.0

        self._draw_profile_view(
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
        )

        # ── Data Band (drawn second so solid white curve parameter boxes mask blue profile lines) ──
        self._draw_data_rows(
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

        # ── Section Heading: L - SECTION (centered in middle of clear margin gap) ──
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


        # ── Top Section: 2D Horizontal Alignment Plan View ─────────────────
        msp.add_lwpolyline(
            [
                (origin_x, plan_bot_y),
                (origin_x + draw_len_mm, plan_bot_y),
                (origin_x + draw_len_mm, plan_top_y),
                (origin_x, plan_top_y),
                (origin_x, plan_bot_y),
            ],
            dxfattribs={"layer": "SHEET_BORDER", "color": self.COL_BLACK, "lineweight": 25}
        )

        # Track centreline & chainage ticks (for this 5km chunk only)
        self._draw_plan_centerline_and_ticks(
            msp=msp,
            line_geom=getattr(align_res, "alignment_line", None),
            total_length_m=draw_length_m,
            origin_x=origin_x,
            origin_y=plan_origin_y,
            draw_len_mm=draw_len_mm,
            ip_df=ip_df,
            start_ch=start_ch,
        )

        # Build get_paper_pos callback for true alignment centerline orientation
        get_paper_pos_cb = None
        if align_res is not None and hasattr(align_res, "alignment_line") and align_res.alignment_line is not None:
            line_geom = align_res.alignment_line
            total_len = line_geom.length
            if total_len > 0:
                def _get_paper_pos(ch: float) -> tuple[float, float, float]:
                    dist = max(0.0, min(total_len, ch - start_ch))
                    pt_next = line_geom.interpolate(min(total_len, dist + 1.0))
                    pt_prev = line_geom.interpolate(max(0.0, dist - 1.0))
                    dx = pt_next.x - pt_prev.x
                    dy = pt_next.y - pt_prev.y
                    ang = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0.0
                    px_paper = origin_x + (ch - start_ch) * (draw_len_mm / draw_length_m)
                    py_paper = plan_origin_y
                    return px_paper, py_paper, ang
                get_paper_pos_cb = _get_paper_pos

        # Plan View Vertical Gradient Callout Arrows (oriented strictly perpendicular to alignment centerline)
        # self._draw_plan_gradient_arrows(
        #     msp=msp,
        #     vip_df=vip_df,
        #     origin_x=origin_x,
        #     plan_origin_y=plan_origin_y,
        #     draw_len_mm=draw_len_mm,
        #     total_length_m=draw_length_m,
        #     start_ch=start_ch,
        #     get_paper_pos=get_paper_pos_cb,
        #     ip_df=ip_df,
        # )

        # RDSO Curve Parameter Boxes & Transition Points (TP-1, J-1, J-2, TP-2) for this chunk
        if ip_df is not None and not ip_df.empty:
            self._draw_plan_curve_transition_points(
                msp=msp, ip_df=ip_df,
                origin_x=origin_x, plan_origin_y=plan_origin_y,
                draw_len_mm=draw_len_mm, total_length_m=draw_length_m,
                start_ch=start_ch,
            )
            # self._draw_plan_curve_parameter_boxes(
            #     msp=msp, ip_df=ip_df,
            #     origin_x=origin_x, origin_y=plan_origin_y,
            #     draw_len_mm=draw_len_mm, total_length_m=draw_length_m,
            #     start_ch=start_ch,
            # )

        # Plan View Bridge Details Callouts (matching curve transition points style)
        if bridge_df is not None and not bridge_df.empty:
            self._draw_plan_bridge_callouts(
                msp=msp, bridge_df=bridge_df,
                origin_x=origin_x, plan_origin_y=plan_origin_y,
                draw_len_mm=draw_len_mm, total_length_m=draw_length_m,
                start_ch=start_ch,
            )

        # Topomap DWG/DXF vector background overlay (roads, rivers, buildings, survey features)
        ##################################
        self._draw_plan_topo_dwg_overlay(
            msp=msp,
            align_res=align_res,
            start_ch=start_ch,
            draw_length_m=draw_length_m,
            origin_x=origin_x,
            plan_origin_y=plan_origin_y,
            draw_len_mm=draw_len_mm,
            plan_area_h=plan_area_h,
        )
        ##################################
        # Ortho-projected topo structures (bridges, culverts, infringements)
        ############################
        # self._draw_plan_topo_ortho(
        #     msp=msp,
        #     topo_res=topo_res,
        #     align_res=align_res,
        #     start_ch=start_ch,
        #     draw_length_m=draw_length_m,
        #     origin_x=origin_x,
        #     plan_origin_y=plan_origin_y,
        #     draw_len_mm=draw_len_mm,
        #     plan_area_h=plan_area_h,
        #     bridge_df=bridge_df,
        # )
        ################################
        # Structure & Turnout callouts (generic dead-ends + SRJ)
        self._draw_plan_structure_and_turnout_callouts(
            msp=msp, bridge_df=bridge_df, topo_res=topo_res, align_res=align_res,
            total_length_m=draw_length_m,
            origin_x=origin_x, origin_y=plan_origin_y, draw_len_mm=draw_len_mm,
        )

        # Junction arrows
        self._draw_plan_junction_arrows(
            msp=msp, total_length_m=draw_length_m,
            origin_x=origin_x, origin_y=plan_origin_y, draw_len_mm=draw_len_mm,
        )

        # North Arrow (top-right of plan area)
        self._draw_plan_north_arrow(
            msp, center_x=origin_x + draw_len_mm - 25.0, center_y=plan_top_y - 30.0
        )

        # Title block in label column area (plan section)
        self._draw_plan_title_block(
            msp, x=0.0, y=plan_bot_y, w=label_width + 18.0, h=plan_area_h
        )

        # ── Save DXF & PDF ──────────────────────────────────────────────────
        self._apply_modelspace_scaling(msp)
        try:
            doc.saveas(str(dxf_out))
        except PermissionError:
            alt_dxf = Path(dxf_out).with_stem(Path(dxf_out).stem + "_updated")
            self.logger.warning("File '%s' is locked in another program. Saving to '%s'.", Path(dxf_out).name, alt_dxf.name)
            doc.saveas(str(alt_dxf))
            dxf_out = alt_dxf

        self._export_to_pdf(dxf_out, pdf_out)

        return {
            "dxf_path":       str(dxf_out),
            "pdf_path":       str(pdf_out),
            "sheet_width_mm": sheet_width,
            "sheet_height_mm": sheet_height,
            "start_ch_m":     start_ch_val,
            "end_ch_m":       end_ch_val,
        }

    def draw_alignment_plan(
        self,
        align_res: AlignmentResult,
        hgeom_res: Optional[HGeomResult] = None,
        valign_res: Optional[VAlignResult] = None,
        ew_res: Optional[Any] = None,
        topo_res: Optional[Any] = None,
        bridge_df: Optional[Any] = None,
        survey_csv_path: Optional[str | Path] = None,
        output_prefix: str = "alignment_plan",
    ) -> ESPResult:
        """Generates Indian Railways Alignment Plan — max 5 km per sheet.

        Mirrors draw() (L-Section) chunking behaviour:
          - <= max_lsection_length_m  → single sheet
          - >  max_lsection_length_m  → one sheet per 5 km chunk

        Each sheet has:
          - Plan view (top half): track centreline, curve boxes, callouts
          - Data band (bottom half): same 10-row table as L-Section
        """
        self.logger.info("Starting Alignment Plan (Working Plan) DXF/PDF Generation...")

        try:
            # ── Dedicated output folder (separate from L-Section outputs/) ──
            ap_dir = Path(
                self.config.get("paths", {}).get(
                    "alignment_plan_outputs_dir",
                    str(self.outputs_dir / "alignment_plan"),
                )
            )
            ap_dir.mkdir(parents=True, exist_ok=True)

            # ── Extract upstream data ───────────────────────────────────────
            line_geom      = getattr(align_res, "alignment_line", None)
            total_length_m = line_geom.length if line_geom is not None else 1000.0
            ip_df_all      = getattr(hgeom_res,  "ip_dataframe",  None) if hgeom_res  is not None else None
            vip_df_all     = getattr(valign_res, "vip_dataframe", None) if valign_res is not None else None

            # ── Build full chainage / elevation arrays ──────────────────────
            profile_df = None
            if ew_res is not None and hasattr(ew_res, "profile_dataframe"):
                profile_df = ew_res.profile_dataframe
            elif valign_res is not None and hasattr(valign_res, "profile_dataframe"):
                profile_df = valign_res.profile_dataframe

            if profile_df is not None and not profile_df.empty and "chainage_m" in profile_df.columns:
                ch_all  = profile_df["chainage_m"].values
                egl_all = profile_df["elevation_m"].fillna(0).values if "elevation_m" in profile_df.columns else np.zeros(len(ch_all))
                fl_all  = profile_df["formation_level_m"].fillna(0).values if "formation_level_m" in profile_df.columns else egl_all.copy()
                total_length_m = float(ch_all[-1]) if len(ch_all) > 0 else total_length_m
            else:
                interval_m = float(self.config.get("chainage", {}).get("interval_m", 20.0))
                n_pts      = max(2, int(round(total_length_m / interval_m)) + 1)
                ch_all     = np.linspace(0.0, total_length_m, n_pts)

                if (vip_df_all is not None and not vip_df_all.empty
                        and "elevation_m" in vip_df_all.columns):
                    egl_all = np.interp(ch_all, vip_df_all["chainage_m"].values,
                                        vip_df_all["elevation_m"].values)
                    fl_all  = egl_all.copy()
                else:
                    egl_all = np.zeros(n_pts)
                    fl_all  = np.zeros(n_pts)

            max_len = self.max_lsection_length_m   # same 5 km limit as L-Section
            dxf_paths: list[Path] = []
            pdf_paths: list[Path] = []
            sheets_meta: list[dict] = []

            # ── 1. Complete Continuous Drawing ──────────────────────────────
            should_gen_complete = self.generate_complete_profile or (max_len <= 0.0)
            if should_gen_complete:
                dxf_out = ap_dir / f"{output_prefix}.dxf"
                pdf_out = ap_dir / f"{output_prefix}.pdf"
                meta = self._render_alignment_sheet(
                    chainages=ch_all, egl=egl_all, fl=fl_all,
                    dxf_out=dxf_out, pdf_out=pdf_out,
                    vip_df=vip_df_all, ip_df=ip_df_all,
                    align_res=align_res, topo_res=topo_res, bridge_df=bridge_df,
                    start_ch=0.0,
                    profile_df=profile_df,
                )
                self.logger.info("Saved Complete Alignment Plan DXF: %s", dxf_out)
                dxf_paths.append(dxf_out)
                if pdf_out.exists():
                    pdf_paths.append(pdf_out)
                sheets_meta.append(meta)

            # ── 2. Multi-Sheet Chunking (> max_len, e.g. 5km Part Sheets) ───
            should_gen_sheets = self.generate_sheet_drawings and (max_len > 0.0)
            if should_gen_sheets and total_length_m > max_len:
                num_sheets = int(math.ceil(total_length_m / max_len))
                self.logger.info(
                    "Alignment Plan: %.0f m total → %d sheets of %.0f m each",
                    total_length_m, num_sheets, max_len,
                )

                for i in range(num_sheets):
                    start_ch = i * max_len
                    end_ch   = min((i + 1) * max_len, total_length_m)

                    # Slice chainage arrays for this chunk
                    mask      = (ch_all >= start_ch) & (ch_all <= end_ch)
                    ch_c      = ch_all[mask]
                    egl_c     = egl_all[mask]
                    fl_c      = fl_all[mask]

                    if len(ch_c) == 0:
                        continue

                    # Slice ip_df & vip_df for this chunk (with buffer)
                    vip_c = None
                    if vip_df_all is not None and not vip_df_all.empty:
                        vip_c = vip_df_all[
                            (vip_df_all["chainage_m"] >= start_ch - 1000.0) &
                            (vip_df_all["chainage_m"] <= end_ch   + 1000.0)
                        ].copy()

                    ip_c = None
                    if ip_df_all is not None and not ip_df_all.empty:
                        ip_c = ip_df_all[
                            (ip_df_all["chainage_m"] >= start_ch - 2000.0) &
                            (ip_df_all["chainage_m"] <= end_ch   + 2000.0)
                        ].copy()

                    sheet_name = (
                        f"{output_prefix}_sheet{i+1}"
                        f"_km{start_ch/1000.0:.1f}-{end_ch/1000.0:.1f}"
                    )
                    dxf_sheet = ap_dir / f"{sheet_name}.dxf"
                    pdf_sheet = ap_dir / f"{sheet_name}.pdf"

                    prof_c = None
                    if profile_df is not None:
                        prof_c = profile_df[
                            (profile_df["chainage_m"] >= start_ch - 100.0) &
                            (profile_df["chainage_m"] <= end_ch   + 100.0)
                        ].copy()

                    sheet_meta = self._render_alignment_sheet(
                        chainages=ch_c, egl=egl_c, fl=fl_c,
                        dxf_out=dxf_sheet, pdf_out=pdf_sheet,
                        vip_df=vip_c, ip_df=ip_c,
                        align_res=align_res, topo_res=topo_res, bridge_df=bridge_df,
                        start_ch=start_ch,
                        profile_df=prof_c,
                    )
                    sheet_meta["sheet_index"] = i + 1
                    sheets_meta.append(sheet_meta)

                    dxf_paths.append(dxf_sheet)
                    if pdf_sheet.exists():
                        pdf_paths.append(pdf_sheet)

                    self.logger.info(
                        "Alignment Plan sheet %d/%d saved: %s", i + 1, num_sheets, dxf_sheet
                    )

            primary_dxf = dxf_paths[0] if dxf_paths else (ap_dir / f"{output_prefix}.dxf")

            return ESPResult(
                dxf_path=primary_dxf,
                dxf_paths=dxf_paths,
                pdf_paths=pdf_paths,
                metadata={
                    "num_sheets":       len(dxf_paths),
                    "total_length_m":   total_length_m,
                    "max_sheet_length_m": max_len,
                    "sheets":           sheets_meta,
                },
            )

        except Exception as exc:
            self.logger.error("Alignment Plan DXF/PDF generation failed: %s", exc)
            raise ESPDrawerError(f"Alignment Plan DXF/PDF generation failed: {exc}") from exc



    def _draw_plan_title_block(
        self, msp: Any, x: float, y: float, w: float, h: float
    ) -> None:
        """Draw standard Indian Railways Alignment Plan Title Block."""
        proj_name = str(self.config.get("project", {}).get("name", "RAILWAY CORRIDOR DESIGN"))
        
        # Outer title block frame
        msp.add_lwpolyline(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
            dxfattribs={"layer": "BORDER", "color": self.COL_RED, "lineweight": 35}
        )
        msp.add_line((x, y + h - 15.0), (x + w, y + h - 15.0), dxfattribs={"layer": "BORDER", "color": self.COL_RED})

        # Main Title Header
        msp.add_text(
            "INDIAN RAILWAYS — ALIGNMENT PLAN (WORKING PLAN)",
            dxfattribs={
                "height": 4.0, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_RED,
            }
        ).set_placement((x + w / 2.0, y + h - 8.0), align=TextEntityAlignment.MIDDLE_CENTER)

        # Metadata Details
        details = [
            f"PROJECT: {proj_name}",
            f"SCALE: 1:{int(self.config.get('drawing', {}).get('plan_scale', 1000))} (HORIZONTAL)",
            f"GAUGE: BROAD GAUGE (1676mm)",
            "STATUS: ISSUED FOR DESIGN & TENDER",
        ]
        for idx, text in enumerate(details):
            msp.add_text(
                text,
                dxfattribs={
                    "height": 2.5, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_BLACK,
                }
            ).set_placement((x + 5.0, y + h - 22.0 - idx * 6.0), align=TextEntityAlignment.MIDDLE_LEFT)

    def _draw_plan_centerline_and_ticks(
        self,
        msp: Any,
        line_geom: Any,
        total_length_m: float,
        origin_x: float,
        origin_y: float,
        draw_len_mm: float,
        ip_df: Optional[pd.DataFrame] = None,
        start_ch: float = 0.0,
    ) -> None:
        """Draw true 2D horizontal alignment line with curves (from KML/Shapely) and perpendicular 100m station ticks."""
        if getattr(self, "drawing_mode", "model_space_metres") == "model_space_metres":
            scale_factor = 1.0
        else:
            scale_factor = draw_len_mm / total_length_m if total_length_m > 0 else 1.0
        end_ch = start_ch + total_length_m

        has_2d = False
        sample_chs = np.linspace(start_ch, end_ch, max(100, int(total_length_m / 5.0)))
        paper_pts: list[tuple[float, float]] = []
        pts_2d = []

        if line_geom is not None and hasattr(line_geom, "interpolate"):
            try:
                # Extract 2D UTM points along alignment line
                pts_2d = [line_geom.interpolate(s) for s in sample_chs]
                p_start = pts_2d[0]
                p_end = pts_2d[-1]

                # Compute baseline angle to align start->end horizontally on sheet
                dx_base = p_end.x - p_start.x
                dy_base = p_end.y - p_start.y
                baseline_len = math.hypot(dx_base, dy_base)

                if baseline_len > 1.0:
                    theta = math.atan2(dy_base, dx_base)
                    cos_t = math.cos(theta)
                    sin_t = math.sin(theta)

                    # Compute unscaled rotated offsets (u, v) in metres
                    u_vals = []
                    v_vals = []
                    for pt in pts_2d:
                        dx = pt.x - p_start.x
                        dy = pt.y - p_start.y
                        u = dx * cos_t + dy * sin_t
                        v = -dx * sin_t + dy * cos_t
                        u_vals.append(u)
                        v_vals.append(v)

                    # Clamp vertical bulge (v) to 200mm max paper height (fits inside 500mm box)
                    max_v = max(abs(v) for v in v_vals) if v_vals else 0.0
                    v_scale_mm_per_m = draw_len_mm / total_length_m if total_length_m > 0 else 1.0
                    if max_v * v_scale_mm_per_m > 200.0 and max_v > 0:
                        v_scale_mm_per_m = 200.0 / max_v

                    # Map to paper coordinates
                    for u, v in zip(u_vals, v_vals):
                        px = origin_x + (u / baseline_len) * draw_len_mm
                        py = origin_y + v * v_scale_mm_per_m
                        paper_pts.append((px, py))

                    has_2d = True
            except Exception as e:
                self.logger.warning("Could not project 2D alignment line: %s", e)

        v_scale_mm_per_m = draw_len_mm / total_length_m if total_length_m > 0 else 1.0
        if has_2d and len(paper_pts) >= 2:
            max_v = max(abs(v) for v in v_vals) if 'v_vals' in locals() and v_vals else 0.0
            if max_v * v_scale_mm_per_m > 200.0 and max_v > 0:
                v_scale_mm_per_m = 200.0 / max_v

        def get_paper_pos(ch_m: float) -> tuple[float, float, float]:
            """Return (paper_x, paper_y, heading_rad) for any chainage."""
            if has_2d and line_geom is not None and len(pts_2d) >= 2:
                try:
                    pt = line_geom.interpolate(ch_m)
                    pt_next = line_geom.interpolate(min(end_ch, ch_m + 1.0))
                    dx_b = pts_2d[-1].x - pts_2d[0].x
                    dy_b = pts_2d[-1].y - pts_2d[0].y
                    b_len = math.hypot(dx_b, dy_b)
                    th = math.atan2(dy_b, dx_b)

                    # Point 1
                    dx1, dy1 = pt.x - pts_2d[0].x, pt.y - pts_2d[0].y
                    u1 = dx1 * math.cos(th) + dy1 * math.sin(th)
                    v1 = -dx1 * math.sin(th) + dy1 * math.cos(th)
                    px1 = origin_x + (u1 / b_len) * draw_len_mm
                    py1 = origin_y + v1 * v_scale_mm_per_m

                    # Point 2 (for tangent angle)
                    dx2, dy2 = pt_next.x - pts_2d[0].x, pt_next.y - pts_2d[0].y
                    u2 = dx2 * math.cos(th) + dy2 * math.sin(th)
                    v2 = -dx2 * math.sin(th) + dy2 * math.cos(th)
                    px2 = origin_x + (u2 / b_len) * draw_len_mm
                    py2 = origin_y + v2 * v_scale_mm_per_m

                    heading = math.atan2(py2 - py1, px2 - px1)
                    return px1, py1, heading
                except Exception:
                    pass
            x_p = origin_x + (ch_m - start_ch) * scale_factor
            return x_p, origin_y, 0.0

        self.current_get_paper_pos = get_paper_pos

        is_existing = self.config.get("drawing", {}).get("is_existing_line", False)
        track_color = self.COL_BLACK if is_existing else self.COL_RED

        if not has_2d or len(paper_pts) < 2:
            paper_pts = [
                (origin_x + (ch - start_ch) * scale_factor, origin_y)
                for ch in sample_chs
            ]
            has_2d = True

        if ip_df is None or ip_df.empty:
            raise ESPDrawerError(
                f"Plan View alignment rendering failed for sheet [{start_ch:.2f}m - {end_ch:.2f}m]: "
                "Horizontal geometry curve table (ip_df) is missing or empty."
            )

        # Render 2D track alignment strictly: Tangents & Transition curves as POLYLINES, Circular curves as ARCs
        curr_ch = start_ch
        # Filter and sort valid curves on this sheet
        valid_ips = []
        for _, ip_row in ip_df.iterrows():
            radius = float(ip_row.get("radius_m", 0.0))
            if radius <= 0:
                continue
            ip_ch = float(ip_row.get("chainage_m", 0.0))
            t_len = float(ip_row.get("tangent_length_m", 0.0))
            tptc_ch = float(ip_row.get("tptc_chainage_m", max(0.0, ip_ch - t_len)))
            tpcc_ch = float(ip_row.get("tpcc_chainage_m", tptc_ch))
            ctpc_ch = float(ip_row.get("ctpc_chainage_m", tpcc_ch))
            ctpt_ch = float(ip_row.get("ctpt_chainage_m", ip_ch + t_len))

            direction = str(ip_row.get("direction", "RHS")).upper()
            is_right_turn = "R" in direction

            if ctpt_ch > start_ch and tptc_ch < end_ch:
                valid_ips.append((tptc_ch, tpcc_ch, ctpc_ch, ctpt_ch, radius, is_right_turn))

        valid_ips.sort(key=lambda x: x[0])

        for tptc_ch, tpcc_ch, ctpc_ch, ctpt_ch, radius, is_right_turn in valid_ips:
            # 0. Straight Tangent section before transition curve (Single clean line without segmentation)
            tan_start = max(curr_ch, start_ch)
            tan_end = min(tptc_ch, end_ch)
            if tan_start < tan_end:
                p_start = get_paper_pos(tan_start)[:2]
                p_end = get_paper_pos(tan_end)[:2]
                msp.add_line(
                    p_start,
                    p_end,
                    dxfattribs={"layer": "ALIGNMENT", "color": track_color, "lineweight": 25}
                )

            # 1. Transition 1 (TP-TC to TP-CC): Multi-segment Polyline (LWPOLYLINE)
            tr1_start = max(start_ch, tptc_ch)
            tr1_end = min(end_ch, tpcc_ch)
            if tr1_start < tr1_end:
                tr1_chs = np.linspace(tr1_start, tr1_end, 15)
                tr1_pts = [get_paper_pos(c)[:2] for c in tr1_chs]
                if len(tr1_pts) >= 2:
                    msp.add_lwpolyline(
                        tr1_pts,
                        dxfattribs={"layer": "ALIGNMENT", "color": track_color, "lineweight": 25}
                    )

            # 2. Circular Curve (TP-CC to CT-PC): True AutoCAD ARC (No fallback polyline allowed)
            c_start = max(start_ch, tpcc_ch)
            c_end = min(end_ch, ctpc_ch)
            if c_start < c_end:
                c_mid = (c_start + c_end) / 2.0
                p1 = get_paper_pos(c_start)[:2]
                pm = get_paper_pos(c_mid)[:2]
                p2 = get_paper_pos(c_end)[:2]

                arc_geom = self._arc_from_2_points_and_radius(p1, p2, radius=radius * scale_factor, pm=pm, is_right_turn=is_right_turn)
                if arc_geom is None:
                    raise ESPDrawerError(
                        f"Failed to fit circular DXF ARC between chainage {c_start:.2f}m and {c_end:.2f}m: "
                        f"Points p1={p1}, p2={p2} are collinear or degenerate."
                    )

                center, r_exact, sa, ea = arc_geom
                msp.add_arc(
                    center=center,
                    radius=r_exact,
                    start_angle=sa,
                    end_angle=ea,
                    dxfattribs={"layer": "ALIGNMENT", "color": track_color, "lineweight": 25}
                )

            # 3. Transition 2 (CT-PC to CT-PT): Multi-segment Polyline (LWPOLYLINE)
            tr2_start = max(start_ch, ctpc_ch)
            tr2_end = min(end_ch, ctpt_ch)
            if tr2_start < tr2_end:
                tr2_chs = np.linspace(tr2_start, tr2_end, 15)
                tr2_pts = [get_paper_pos(c)[:2] for c in tr2_chs]
                if len(tr2_pts) >= 2:
                    msp.add_lwpolyline(
                        tr2_pts,
                        dxfattribs={"layer": "ALIGNMENT", "color": track_color, "lineweight": 25}
                    )

            curr_ch = max(curr_ch, ctpt_ch)

        # Final Straight Tangent section after last curve (Single clean line without segmentation)
        if curr_ch < end_ch:
            p_start = get_paper_pos(curr_ch)[:2]
            p_end = get_paper_pos(end_ch)[:2]
            msp.add_line(
                p_start,
                p_end,
                dxfattribs={"layer": "ALIGNMENT", "color": track_color, "lineweight": 25}
            )

        # ── Perpendicular Chainage Ticks along 2D Alignment ────────────────────
        ch_step = float(
            self.config.get("drawing", {})
            .get("plan_view", {})
            .get("chainage_tick_interval_m", 100.0)
        )
        tick_gap_mm = ch_step * scale_factor
        label_every = 1  # Show station chainage label at every 100m tick per user specification
        tick_index = 0
        ch_curr = start_ch

        # Generate 2D parallel track rails (+- 0.838m) & formation edges (+- 3.85m)
        rail_left_pts = []
        rail_right_pts = []
        form_left_pts = []
        form_right_pts = []

        gauge_half_m = (float(getattr(self, "gauge_mm", 1676.0)) / 1000.0) / 2.0  # 0.838m
        form_half_m = float(self.config.get("earthwork", {}).get("formation_width_m", 7.7)) / 2.0  # 3.85m

        bound_left_pts = []
        bound_right_pts = []

        sample_chs_fine = np.linspace(start_ch, end_ch, max(120, int(total_length_m / 4.0)))
        ew = Earthwork(self.config)
        smooth_offsets = ew.compute_smooth_zonal_boundary_profile(sample_chs_fine)

        for i, ch in enumerate(sample_chs_fine):
            px, py, heading = get_paper_pos(ch)
            perp_angle = heading + (math.pi / 2.0)
            cos_p, sin_p = math.cos(perp_angle), math.sin(perp_angle)

            r_mm = gauge_half_m * scale_factor
            f_mm = form_half_m * scale_factor
            b_mm = float(smooth_offsets[i]) * scale_factor

            rail_left_pts.append((px - r_mm * cos_p, py - r_mm * sin_p))
            rail_right_pts.append((px + r_mm * cos_p, py + r_mm * sin_p))
            form_left_pts.append((px - f_mm * cos_p, py - f_mm * sin_p))
            form_right_pts.append((px + f_mm * cos_p, py + f_mm * sin_p))

            bound_left_pts.append((px - b_mm * cos_p, py - b_mm * sin_p))
            bound_right_pts.append((px + b_mm * cos_p, py + b_mm * sin_p))

        # Check if line is Existing Rail vs Proposed Rail (from config)
        is_existing = self.config.get("drawing", {}).get("is_existing_line", False)
        track_color = self.COL_BLACK if is_existing else self.COL_RED

        # Draw Left & Right Parallel Railway Formation/Boundary lines (Pattern: ________ . . _ . . ________)
        if len(bound_left_pts) >= 2:
            msp.add_lwpolyline(
                bound_left_pts,
                dxfattribs={"layer": "RAILWAY_BOUNDARY", "color": self.COL_WHITE, "lineweight": 25, "linetype": "RAILWAY_ROW"}
            )
        if len(bound_right_pts) >= 2:
            msp.add_lwpolyline(
                bound_right_pts,
                dxfattribs={"layer": "RAILWAY_BOUNDARY", "color": self.COL_WHITE, "lineweight": 25, "linetype": "RAILWAY_ROW"}
            )

        # Start / End Junction Callout Text
        if abs(start_ch) < 0.1:
            px0, py0, h0 = get_paper_pos(0.0)
            deg0 = math.degrees(h0)
            msp.add_text(
                "DEAD END AT CH: 0.000m",
                dxfattribs={"height": 2.5, "layer": "ALIGNMENT", "style": "ESP_STYLE", "color": self.COL_RED, "rotation": deg0}
            ).set_placement((px0 - 20.0 * math.cos(h0), py0 - 20.0 * math.sin(h0)), align=TextEntityAlignment.MIDDLE_RIGHT)

            msp.add_text(
                "<<< TOWARDS START JUNCTION",
                dxfattribs={"height": 3.2, "layer": "ALIGNMENT", "style": "ESP_STYLE", "color": self.COL_BLACK, "rotation": deg0}
            ).set_placement((px0 - 50.0 * math.cos(h0), py0 - 50.0 * math.sin(h0)), align=TextEntityAlignment.MIDDLE_RIGHT)

        while ch_curr <= end_ch + 0.1:
            px, py, heading = get_paper_pos(ch_curr)
            if origin_x - 1.0 <= px <= origin_x + draw_len_mm + 1.0:
                is_label_tick = (tick_index % label_every == 0)
                tick_h = 3.0 if is_label_tick else 1.5
                tick_lw = 18 if is_label_tick else 9

                # Perpendicular angle
                perp_angle = heading + math.pi / 2.0
                dx_t = (tick_h / 2.0) * math.cos(perp_angle)
                dy_t = (tick_h / 2.0) * math.sin(perp_angle)

                msp.add_line(
                    (px - dx_t, py - dy_t),
                    (px + dx_t, py + dy_t),
                    dxfattribs={"layer": "CHAINAGE_TEXT", "color": self.COL_RED, "lineweight": tick_lw}
                )

                if is_label_tick:
                    ch_str = f"CH: {int(round(ch_curr))}"

                    offset_mm = tick_h / 2.0 + 7.0
                    lbl_x = px + offset_mm * math.cos(perp_angle)
                    lbl_y = py + offset_mm * math.sin(perp_angle)
                    rot_deg = math.degrees(perp_angle) % 360

                    if math.isnan(rot_deg) or math.isinf(rot_deg):
                        rot_deg = 90.0
                    if math.isnan(lbl_x) or math.isinf(lbl_x):
                        lbl_x = px
                    if math.isnan(lbl_y) or math.isinf(lbl_y):
                        lbl_y = py + 12.0

                    msp.add_text(
                        ch_str,
                        dxfattribs={
                            "height": 3.0, "rotation": float(rot_deg),
                            "layer": "CHAINAGE_TEXT",
                            "style": "ESP_STYLE", "color": self.COL_RED,
                        }
                    ).set_placement(
                        (float(lbl_x), float(lbl_y)),
                        align=TextEntityAlignment.MIDDLE_LEFT
                    )

            ch_curr += ch_step
            tick_index += 1

    def _draw_plan_gradient_arrows(
        self,
        msp: Any,
        vip_df: Optional[pd.DataFrame],
        origin_x: float,
        plan_origin_y: float,
        draw_len_mm: float,
        total_length_m: float,
        start_ch: float = 0.0,
        get_paper_pos: Optional[Any] = None,
        ip_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Render Plan View perpendicular vertical gradient callout arrows, PVI text labels (CH, FL, KM), and grade ratios."""
        if vip_df is None or vip_df.empty:
            return

        if get_paper_pos is None:
            get_paper_pos = getattr(self, "current_get_paper_pos", None)

        vips = vip_df.sort_values("chainage_m").to_dict("records")
        n_vips = len(vips)
        max_right_x = origin_x + draw_len_mm

        def _fmt_g(e1: float, e2: float, dist: float) -> str:
            if dist <= 0.001:
                return "LEVEL"
            g_pct = ((e2 - e1) / dist) * 100.0
            if abs(g_pct) < 0.001:
                return "LEVEL"
            inv_g = int(round(100.0 / abs(g_pct)))
            return f"1:{inv_g} R" if g_pct > 0 else f"1:{inv_g} F"

        for i in range(n_vips):
            c_pvi = float(vips[i]["chainage_m"])
            e_cur = float(vips[i]["elevation_m"])

            if get_paper_pos is not None:
                px, py, angle_rad = get_paper_pos(c_pvi)
            else:
                px = origin_x + (c_pvi - start_ch) * (draw_len_mm / total_length_m)
                py = plan_origin_y
                angle_rad = 0.0

            if not (origin_x + 5.0 <= px <= max_right_x - 5.0):
                continue

            def _calc_g_pct(e1: float, e2: float, dist: float) -> float:
                if dist <= 0.001:
                    return 0.0
                return ((e2 - e1) / dist) * 100.0

            if i > 0:
                c_prev = float(vips[i - 1]["chainage_m"])
                e_prev = float(vips[i - 1]["elevation_m"])
                g_left_pct = _calc_g_pct(e_prev, e_cur, c_pvi - c_prev)
                grad_left = _fmt_g(e_prev, e_cur, c_pvi - c_prev)
            else:
                g_left_pct = 0.0
                grad_left = "LEVEL"

            if i < n_vips - 1:
                c_next = float(vips[i + 1]["chainage_m"])
                e_next = float(vips[i + 1]["elevation_m"])
                g_right_pct = _calc_g_pct(e_cur, e_next, c_next - c_pvi)
                grad_right = _fmt_g(e_cur, e_next, c_next - c_pvi)
            else:
                g_right_pct = 0.0
                grad_right = "LEVEL"

            # Skip intermediate PVI stations where gradient does not change
            if grad_left == grad_right:
                continue

            # -----------------------------------------------------------------
            # Permanent Spatial AABB Collision-Avoidance Solver for Gradient Callouts
            # -----------------------------------------------------------------
            # Evaluates candidate stem side (-N vs +N) and stem length/text anchor
            # to guarantee zero overlap among internal callout texts and zero collision
            # against previously placed elements registered in self.placed_plan_boxes.

            tangent_deg = math.degrees(angle_rad)
            tx = math.cos(angle_rad)
            ty = math.sin(angle_rad)

            km_val = c_pvi / 1000.0
            str_km = f"KM: {km_val:.3f}"
            str_fl = f"FL: {e_cur:.3f} m"
            str_ch = f"CH: {c_pvi:.2f} m"

            # Check if PVI is within an LHS curve
            is_lhs_curve = False
            pvi_dir = str(vips[i].get("direction", "")).upper()
            if "LHS" in pvi_dir or "LEFT" in pvi_dir:
                is_lhs_curve = True
            elif ip_df is not None and not ip_df.empty:
                for _, ip_row in ip_df.iterrows():
                    ip_ch = float(ip_row.get("chainage_m", 0.0))
                    tptc_ch = float(ip_row.get("tptc_chainage_m", ip_ch))
                    ctpt_ch = float(ip_row.get("ctpt_chainage_m", ip_ch))
                    direction = str(ip_row.get("direction", "")).upper()
                    if ("LHS" in direction or "LEFT" in direction) and (tptc_ch - 50.0 <= c_pvi <= ctpt_ch + 50.0):
                        is_lhs_curve = True
                        break

            # Side directions: ALL gradient arrows ALWAYS project to the lower side (-N direction)
            # opposite to chainage ticks (which are strictly +N above track) to ensure zero overlap.
            side_mults = [-1.0]
            stem_lengths = [35.0, 45.0, 55.0]
            text_anchors = [6.0, 8.0, 10.0]

            best_config = None
            best_score = float("inf")

            for s_mult in side_mults:
                nx = -s_mult * math.sin(angle_rad)
                ny = s_mult * math.cos(angle_rad)
                perp_deg = tangent_deg + (90.0 if s_mult > 0 else -90.0)

                for stem_len in stem_lengths:
                    jx = px + stem_len * nx
                    jy = py + stem_len * ny

                    for d_anchor in text_anchors:
                        p_stem_x = px + d_anchor * nx
                        p_stem_y = py + d_anchor * ny

                        stem_rot = perp_deg % 360.0
                        if 90.0 < stem_rot <= 270.0:
                            stem_rot = (stem_rot + 180.0) % 360.0

                        s_rad = math.radians(stem_rot)
                        s_up_x = -math.sin(s_rad)
                        s_up_y = math.cos(s_rad)

                        # Bounding boxes for 3 stem text lines
                        box_km = self._compute_text_aabb(str_km, 2.0, p_stem_x + 3.5 * s_up_x, p_stem_y + 3.5 * s_up_y, stem_rot)
                        box_fl = self._compute_text_aabb(str_fl, 2.0, p_stem_x - 4.0 * s_up_x, p_stem_y - 4.0 * s_up_y, stem_rot)
                        box_ch = self._compute_text_aabb(str_ch, 2.0, p_stem_x - 9.5 * s_up_x, p_stem_y - 9.5 * s_up_y, stem_rot)

                        # Bounding boxes for left/right gradient arms
                        arm_len = 22.0
                        
                        def _get_arm_vector(g_pct: float, is_left: bool, nx_vec: float, ny_vec: float) -> tuple[float, float, float]:
                            t_dir = -1.0 if is_left else 1.0
                            if abs(g_pct) < 0.001:
                                slant_deg = 0.0
                            else:
                                g_abs = min(1.0, max(0.05, abs(g_pct)))
                                slant_deg = 15.0 + (g_abs / 1.0) * 25.0
                            
                            slant_rad = math.radians(slant_deg)
                            # Vector U = t_dir * cos(slant) * T + sin(slant) * N (ALWAYS slants away from track)
                            arm_ux = t_dir * math.cos(slant_rad) * tx + math.sin(slant_rad) * nx_vec
                            arm_uy = t_dir * math.cos(slant_rad) * ty + math.sin(slant_rad) * ny_vec
                            arm_angle_rad = math.atan2(arm_uy, arm_ux)
                            return arm_ux, arm_uy, arm_angle_rad

                        def _get_arm_aabb(grad_str: str, g_pct: float, is_left: bool) -> tuple[tuple[float, float, float, float], float, float, float, float, str]:
                            ux, uy, arm_angle_rad = _get_arm_vector(g_pct, is_left, nx, ny)
                            ex = jx + arm_len * ux
                            ey = jy + arm_len * uy

                            rot_deg = math.degrees(arm_angle_rad) % 360.0
                            if 90.0 < rot_deg <= 270.0:
                                rot_deg = (rot_deg + 180.0) % 360.0
                            rot_rad = math.radians(rot_deg)
                            t_up_x = -math.sin(rot_rad)
                            t_up_y = math.cos(rot_rad)

                            mid_x = jx + (arm_len / 2.0) * ux + 3.2 * t_up_x
                            mid_y = jy + (arm_len / 2.0) * uy + 3.2 * t_up_y

                            box_arm = self._compute_text_aabb(grad_str, 2.4, mid_x, mid_y, rot_deg)
                            return box_arm, ex, ey, ux, uy, grad_str

                        box_arm_left, ex_l, ey_l, ux_l, uy_l, _ = _get_arm_aabb(grad_right, g_right_pct, is_left=True)
                        box_arm_right, ex_r, ey_r, ux_r, uy_r, _ = _get_arm_aabb(grad_left, g_left_pct, is_left=False)

                        candidate_boxes = [box_km, box_fl, box_ch, box_arm_left, box_arm_right]

                        # Internal overlap check (stem text vs arm boxes)
                        internal_overlaps = 0
                        for b_stem in [box_km, box_fl, box_ch]:
                            for b_arm in [box_arm_left, box_arm_right]:
                                if self._boxes_intersect(b_stem, b_arm):
                                    internal_overlaps += 1

                        # External overlap check against placed_plan_boxes
                        external_overlaps = 0
                        for cb in candidate_boxes:
                            for pb in self.placed_plan_boxes:
                                if self._boxes_intersect(cb, pb):
                                    external_overlaps += 1

                        score = internal_overlaps * 100 + external_overlaps * 10 + (stem_len - 35.0)

                        if score < best_score:
                            best_score = score
                            best_config = {
                                "nx": nx, "ny": ny, "perp_deg": perp_deg,
                                "stem_len": stem_len, "jx": jx, "jy": jy,
                                "p_stem_x": p_stem_x, "p_stem_y": p_stem_y,
                                "stem_rot": stem_rot, "s_up_x": s_up_x, "s_up_y": s_up_y,
                                "str_km": str_km, "str_fl": str_fl, "str_ch": str_ch,
                                "candidate_boxes": candidate_boxes,
                            }
                            if score == 0:
                                break
                    if best_score == 0:
                        break
                if best_score == 0:
                    break

            if best_config is None:
                continue

            # Render layout using best collision-free configuration
            nx, ny = best_config["nx"], best_config["ny"]
            jx, jy = best_config["jx"], best_config["jy"]
            p_stem_x, p_stem_y = best_config["p_stem_x"], best_config["p_stem_y"]
            stem_rot = best_config["stem_rot"]
            s_up_x, s_up_y = best_config["s_up_x"], best_config["s_up_y"]

            # 1. Draw Stem Line
            msp.add_line(
                (px, py), (jx, jy),
                dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 18}
            )

            # 2. Render Stem Text Labels
            lx_km = p_stem_x + 3.5 * s_up_x
            ly_km = p_stem_y + 3.5 * s_up_y
            msp.add_text(
                best_config["str_km"],
                dxfattribs={"height": 2.0, "rotation": stem_rot, "layer": "VGEOM_PLAN", "style": "ESP_STYLE", "color": self.COL_RED}
            ).set_placement((lx_km, ly_km), align=TextEntityAlignment.BOTTOM_LEFT)

            lx_fl = p_stem_x - 4.0 * s_up_x
            ly_fl = p_stem_y - 4.0 * s_up_y
            msp.add_text(
                best_config["str_fl"],
                dxfattribs={"height": 2.0, "rotation": stem_rot, "layer": "VGEOM_PLAN", "style": "ESP_STYLE", "color": self.COL_RED}
            ).set_placement((lx_fl, ly_fl), align=TextEntityAlignment.BOTTOM_LEFT)

            lx_ch = p_stem_x - 9.5 * s_up_x
            ly_ch = p_stem_y - 9.5 * s_up_y
            msp.add_text(
                best_config["str_ch"],
                dxfattribs={"height": 2.0, "rotation": stem_rot, "layer": "VGEOM_PLAN", "style": "ESP_STYLE", "color": self.COL_RED}
            ).set_placement((lx_ch, ly_ch), align=TextEntityAlignment.BOTTOM_LEFT)

            # 3. Draw Left and Right Arms at Junction J
            arm_len = 22.0
            arrow_w = 2.5
            arrow_h = 0.8

            def _draw_arm(grad_str: str, g_pct: float, is_left: bool) -> None:
                ux, uy, arm_angle_rad = _get_arm_vector(g_pct, is_left, nx, ny)

                ex = jx + arm_len * ux
                ey = jy + arm_len * uy

                msp.add_line(
                    (jx, jy), (ex, ey),
                    dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 25}
                )

                unx, uny = -uy, ux
                p_arrow = [
                    (ex, ey),
                    (ex - arrow_w * ux + arrow_h * unx, ey - arrow_w * uy + arrow_h * uny),
                    (ex - arrow_w * ux - arrow_h * unx, ey - arrow_w * uy - arrow_h * uny),
                    (ex, ey),
                ]
                msp.add_lwpolyline(p_arrow, dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED})
                hatch = msp.add_hatch(color=self.COL_RED)
                hatch.set_solid_fill(color=self.COL_RED)
                hatch.paths.add_polyline_path(p_arrow, is_closed=True)

                rot_deg = math.degrees(arm_angle_rad) % 360.0
                if 90.0 < rot_deg <= 270.0:
                    rot_deg = (rot_deg + 180.0) % 360.0

                rot_rad = math.radians(rot_deg)
                t_up_x = -math.sin(rot_rad)
                t_up_y = math.cos(rot_rad)

                mid_x = jx + (arm_len / 2.0) * ux + 3.2 * t_up_x
                mid_y = jy + (arm_len / 2.0) * uy + 3.2 * t_up_y

                msp.add_text(
                    grad_str,
                    dxfattribs={
                        "height": 2.4,
                        "rotation": rot_deg,
                        "layer": "VGEOM_PLAN",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    }
                ).set_placement((mid_x, mid_y), align=TextEntityAlignment.BOTTOM_CENTER)

            _draw_arm(grad_right, g_right_pct, is_left=True)
            _draw_arm(grad_left, g_left_pct, is_left=False)

            # Register all placed bounding boxes in global plan ledger
            for cb in best_config["candidate_boxes"]:
                self.placed_plan_boxes.append(cb)

    def _draw_plan_curve_parameter_boxes(
        self,
        msp: Any,
        ip_df: pd.DataFrame,
        origin_x: float,
        origin_y: float,
        draw_len_mm: float,
        total_length_m: float,
        start_ch: float = 0.0,
    ) -> None:
        """Render RDSO standard red-bordered horizontal curve parameter table boxes.

        Overlap-Prevention Algorithm (Two-Pass)
        ----------------------------------------
        PASS 1 — Collect:
            For every valid IP curve, build a box descriptor dict (x_center, bh,
            box_lines, is_rhs, x_nat, leader_py) and collect into ``all_boxes``.
            x_center is clamped to stay inside the horizontal sheet boundary.

        PASS 2 — Place without overlaps:
            Boxes are split into two direction groups (RHS above track, LHS below).
            Within each group boxes are sorted left-to-right (by x_nat) and assigned
            a Y-tier using a greedy interval scan:
              • Tier 0: closest to the plan area edge (outermost).
              • Tier 1, 2, …: progressively closer to the track centreline.
            If all N_TIERS are exhausted the box is nudged rightward past the last
            occupant in Tier 0 (never dropped).
            A thin leader line from the IP track position to the box is always drawn.
        """
        scale_factor = draw_len_mm / total_length_m if total_length_m > 0 else 1.0
        end_ch       = start_ch + total_length_m

        BOX_W: float   = 80.0   # box width (mm) — fixed
        X_GAP: float   = 6.0    # minimum horizontal clearance between adjacent boxes
        Y_MARGIN: float = 8.0   # margin from plan-area edge to nearest box face
        N_TIERS: int   = 3      # max stagger rows per side

        # Actual plan-view half-height from config
        plan_area_h: float = float(
            self.config.get("drawing", {}).get("plan_view", {}).get("plan_area_height_mm", 350.0)
        )
        plan_top_y: float = origin_y + plan_area_h / 2.0 - Y_MARGIN
        plan_bot_y: float = origin_y - plan_area_h / 2.0 + Y_MARGIN

        get_paper_pos = getattr(self, "current_get_paper_pos", None)

        # ── PASS 1: collect box descriptors ──────────────────────────────────
        all_boxes: list[dict] = []

        for idx, row in ip_df.iterrows():
            radius = float(row.get("radius_m", 0.0))
            if radius <= 0:
                continue
            ip_ch = float(row.get("chainage_m", 0.0))
            if not (start_ch - 200.0 <= ip_ch <= end_ch + 200.0):
                continue

            curve_no  = int(row.get("ip_index", idx + 1))
            direction = str(row.get("direction", "RHS")).upper()
            is_rhs    = ("RHS" in direction or "RIGHT" in direction)

            # DMS conversion of deflection angle
            def_angle = float(row.get("deflection_deg", 0.0))
            d_deg = int(abs(def_angle))
            d_rem = abs(def_angle) - d_deg
            d_min = int(d_rem * 60)
            d_sec = int(round((d_rem * 60 - d_min) * 60))
            if d_sec == 60:
                d_sec, d_min = 0, d_min + 1
            if d_min == 60:
                d_min, d_deg = 0, d_deg + 1
            delta_str = f"{d_deg}\u00b0{d_min:02d}'{d_sec:02d}\""

            degree   = float(row.get("degree_of_curve",         0.0))
            ccl      = float(row.get("circular_curve_length_m", 0.0))
            tcl      = float(row.get("curve_length_m",          0.0))
            trl      = float(row.get("transition_length_m",     0.0))
            tl       = float(row.get("tangent_length_m",        0.0))
            shift_mm = float(row.get("shift_m",                 0.0)) * 1000.0
            cd       = float(row.get("actual_cd_mm",            0.0))
            cant_mm  = float(row.get("cant_mm",                 0.0))
            vmax_raw = row.get("permissible_speed_kmph", row.get("speed_kmh", 160.0))
            vmax     = float(vmax_raw) if vmax_raw is not None and not (isinstance(vmax_raw, float) and math.isnan(vmax_raw)) else 160.0

            ip_ch    = float(row.get("chainage_m", 0.0))
            tptc_ch  = float(row.get("tptc_chainage_m", ip_ch))
            tpcc_ch  = float(row.get("tpcc_chainage_m", tptc_ch))
            ctpc_ch  = float(row.get("ctpc_chainage_m", tpcc_ch))
            ctpt_ch  = float(row.get("ctpt_chainage_m", ctpc_ch))

            box_lines = [
                f"Curve No. = {curve_no}",
                f"APEX CH. = {ip_ch:.3f} m.",
                f"\u0394 = {delta_str}",
                f"D = {degree:.2f}\u00b0 R = {radius:.3f} m.",
                f"CL = {tcl:.3f} m. TL = {tl:.3f} m.",
                f"LTC = {trl:.2f} m. S = {shift_mm/1000.0:.3f} m.",
                f"TPTC-1 = {tptc_ch:.3f} m. TPCC-1 = {tpcc_ch:.3f} m.",
                f"TPCC-2 = {ctpc_ch:.3f} m. TPTC-2 = {ctpt_ch:.3f} m.",
                f"Ca = {cant_mm:.2f} mm Vmax = {vmax:.0f} kmph",
            ]

            bh = len(box_lines) * 3.6 + 4.0
            BOX_W = 65.0

            # Natural X from geometry
            if get_paper_pos is not None:
                px_ip, py_ip, _ = get_paper_pos(ip_ch)
                x_nat = px_ip
            else:
                x_nat  = origin_x + (ip_ch - start_ch) * scale_factor
                py_ip  = origin_y

            # Clamp X so box stays inside horizontal sheet boundary
            x_min_allowed = origin_x + BOX_W / 2.0 + 4.0
            x_max_allowed = origin_x + draw_len_mm - BOX_W / 2.0 - 4.0
            x_center = max(x_min_allowed, min(x_max_allowed, x_nat))

            all_boxes.append(dict(
                x_nat=x_nat, x_center=x_center, bh=bh, box_w=BOX_W,
                box_lines=box_lines, is_rhs=is_rhs,
                leader_py=py_ip,
            ))

        # Sort all boxes left-to-right for greedy tier assignment
        all_boxes.sort(key=lambda b: b["x_nat"])

        # ── PASS 2: assign Y-tiers per direction group without overlaps ───────
        def _assign_tiers(group: list[dict], is_rhs: bool) -> None:
            """Greedy interval-scan tier assignment."""
            tier_intervals: list[list[tuple[float, float]]] = [[] for _ in range(N_TIERS)]

            for b in group:
                bh      = b["bh"]
                bw      = b["box_w"]
                x_left  = b["x_center"] - bw / 2.0
                x_right = b["x_center"] + bw / 2.0

                assigned = -1
                for t in range(N_TIERS):
                    overlap = any(
                        not (x_right + X_GAP <= ol or x_left - X_GAP >= or_)
                        for ol, or_ in tier_intervals[t]
                    )
                    if not overlap:
                        assigned = t
                        break

                if assigned < 0:
                    assigned = 0
                    last_right = max(
                        (or_ for _, or_ in tier_intervals[0]),
                        default=x_right,
                    )
                    b["x_center"] = last_right + X_GAP + bw / 2.0
                    b["x_center"] = min(b["x_center"], origin_x + draw_len_mm - bw / 2.0 - 4.0)
                    x_left  = b["x_center"] - bw / 2.0
                    x_right = b["x_center"] + bw / 2.0

                tier_intervals[assigned].append((x_left, x_right))

                tier_step = bh + 4.0
                offset_up = -100.0
                if is_rhs:
                    # Shift RHS boxes upward (towards the alignment line) but keep them below it
                      # mm
                    y_bot = plan_bot_y + assigned * tier_step - offset_up
                    y_top = y_bot + bh
                    y_top = min(y_top, origin_y - 4.0)
                else:
                    y_top = plan_top_y - assigned * tier_step + offset_up
                    y_top = max(y_top, origin_y + 4.0 + bh)

                b["y_top"] = y_top

        rhs_group = [b for b in all_boxes if     b["is_rhs"]]
        lhs_group = [b for b in all_boxes if not b["is_rhs"]]
        _assign_tiers(rhs_group, is_rhs=True)
        _assign_tiers(lhs_group, is_rhs=False)

        # ── Draw every placed original Curve Box (Centered Text, No Vertical Line) ──
        for b in all_boxes:
            x_center   = b["x_center"]
            y_top      = b["y_top"]
            bh         = b["bh"]
            bw         = b["box_w"]
            box_lines  = b["box_lines"]
            is_rhs     = b["is_rhs"]
            x_nat      = b["x_nat"]
            ldr_py     = b["leader_py"]

            x_left     = x_center - bw / 2.0
            x_right    = x_center + bw / 2.0
            y_bot_box  = y_top - bh

            # 1. Solid White opaque hatch background
            hatch = msp.add_hatch(color=255)
            hatch.set_solid_fill(color=255, rgb=(255, 255, 255))
            hatch.dxf.color = 255
            hatch.paths.add_polyline_path(
                [
                    (x_left,  y_top),
                    (x_right, y_top),
                    (x_right, y_bot_box),
                    (x_left,  y_bot_box),
                ],
                is_closed=True
            )

            # 2. Red outer border rectangle
            msp.add_lwpolyline(
                [
                    (x_left,  y_top),
                    (x_right, y_top),
                    (x_right, y_bot_box),
                    (x_left,  y_bot_box),
                    (x_left,  y_top),
                ],
                dxfattribs={"layer": "CURVE_TABLE", "color": self.COL_RED, "lineweight": 25}
            )

            # 3. Crisp Red Centered Text lines (with clean 4.5mm top padding)
            for idx_l, line_text in enumerate(box_lines):
                ly = y_top - 4.5 - idx_l * 3.2
                msp.add_text(
                    line_text,
                    dxfattribs={
                        "height": 2.2,
                        "layer": "CURVE_TABLE",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    }
                ).set_placement((x_center, ly), align=TextEntityAlignment.TOP_CENTER)



    def _draw_plan_curve_transition_points(
        self,
        msp: Any,
        ip_df: pd.DataFrame,
        origin_x: float,
        plan_origin_y: float,
        draw_len_mm: float,
        total_length_m: float,
        start_ch: float = 0.0,
    ) -> None:
        """Render TPTC, TPCC, CTPC, CTPT transition point callout text & perpendicular tick lines pointing DOWNWARDS along the alignment."""
        if ip_df is None or ip_df.empty:
            return

        get_paper_pos = getattr(self, "current_get_paper_pos", None)
        if get_paper_pos is None:
            return

        end_ch = start_ch + total_length_m

        for idx, row in ip_df.iterrows():
            radius = float(row.get("radius_m", 0.0))
            if radius <= 0:
                continue

            ip_ch = float(row.get("chainage_m", 0.0))
            curve_no = int(row.get("ip_index", idx + 1))
            direction = str(row.get("direction", "RHS")).upper()
            tl = float(row.get("tangent_length_m", 0.0))
            trl = float(row.get("transition_length_m", 0.0))
            ccl = float(row.get("circular_curve_length_m", 0.0))

            # Compute transition point chainages
            ch_tptc = ip_ch - tl
            ch_tpcc = ch_tptc + trl
            ch_ctpc = ch_tpcc + ccl
            ch_ctpt = ch_ctpc + trl

            c_num_str = f"{curve_no:02d}"

            t_points = [
                (ch_tptc, f"C NO.-{c_num_str} TPTC-1 AT CH.:{ch_tptc:.3f}M"),
                (ch_tpcc, f"C NO.-{c_num_str} TPCC-1 AT CH.:{ch_tpcc:.3f}M"),
                (ch_ctpc, f"C NO.-{c_num_str} TPCC-2 AT CH.:{ch_ctpc:.3f}M"),
                (ch_ctpt, f"C NO.-{c_num_str} TPTC-2 AT CH.:{ch_ctpt:.3f}M"),
            ]

            # Compact stem offset distance (18.0mm) for clean clearance
            off_dist = 18.0

            for tp_idx, (ch_val, text_str) in enumerate(t_points):
                if not (start_ch - 10.0 <= ch_val <= end_ch + 10.0):
                    continue

                px, py, heading = get_paper_pos(ch_val)
                if not (origin_x - 5.0 <= px <= origin_x + draw_len_mm + 5.0):
                    continue

                is_rhs = ("RHS" in direction) or ("RIGHT" in direction)

                ang1 = heading - (math.pi / 2.0)
                ang2 = heading + (math.pi / 2.0)

                if is_rhs:
                    # RHS Curve: Extend DOWNWARD (below track, -45° slant)
                    perp_angle = ang1 if math.sin(ang1) <= 0.0 else ang2
                    slanted_angle = math.radians(-45.0)
                    rot_deg = -45.0
                else:
                    # LHS Curve: Extend UPWARD (above track, +45° slant)
                    perp_angle = ang1 if math.sin(ang1) >= 0.0 else ang2
                    slanted_angle = math.radians(45.0)
                    rot_deg = 45.0

                cos_p, sin_p = math.cos(perp_angle), math.sin(perp_angle)

                # Knee point (end of straight stem line, clamped strictly inside Plan View vertical bounds)
                knee_x = px + off_dist * cos_p
                knee_y = py + off_dist * sin_p
                knee_y = max(plan_origin_y - 70.0, min(plan_origin_y + 70.0, knee_y))

                # Segment 2 10° slanted underline end point
                cos_s, sin_s = math.cos(slanted_angle), math.sin(slanted_angle)
                txt_len_mm = len(text_str) * 2.0 + 4.0
                end_x = knee_x + txt_len_mm * cos_s
                end_y = knee_y + txt_len_mm * sin_s
                end_y = max(plan_origin_y - 72.0, min(plan_origin_y + 72.0, end_y))

                # 1. Target Circle Dot on Track Alignment (radius = 1.2mm)
                msp.add_circle(
                    center=(px, py),
                    radius=1.2,
                    dxfattribs={"layer": "CURVE_ANNOTATION", "color": self.COL_RED, "lineweight": 25}
                )

                # 2. Segment 1: Straight Perpendicular Stem Line from track to knee point
                msp.add_line(
                    (px, py),
                    (knee_x, knee_y),
                    dxfattribs={"layer": "CURVE_ANNOTATION", "color": self.COL_RED, "lineweight": 25}
                )

                # 3. Segment 2: 10° Slanted Underline Line extending from knee point under text details
                msp.add_line(
                    (knee_x, knee_y),
                    (end_x, end_y),
                    dxfattribs={"layer": "CURVE_ANNOTATION", "color": self.COL_RED, "lineweight": 25}
                )

                # 4. Text Sitting Directly Above Segment 2 (the 10° slanted red line)
                perp_s = slanted_angle + (math.pi / 2.0)
                tx = knee_x + 1.2 * math.cos(perp_s) + 2.0 * cos_s
                ty = knee_y + 1.2 * math.sin(perp_s) + 2.0 * sin_s

                txt_ent = msp.add_text(
                    text_str,
                    dxfattribs={
                        "height": 3.0,
                        "layer": "CURVE_ANNOTATION",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                        "rotation": rot_deg,
                    }
                )
                txt_ent.set_placement((tx, ty), align=TextEntityAlignment.BOTTOM_LEFT)

    def _draw_rdso_bridge_symbol(
        self,
        msp: Any,
        px: float,
        py: float,
        heading: float,
        span_length_m: float = 10.0,
    ) -> None:
        """Draw standard Indian Railways Bridge Abutment & Wingwall symbol across track at (px, py)."""
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        perp_angle = heading + math.pi / 2.0
        cos_p, sin_p = math.cos(perp_angle), math.sin(perp_angle)

        scale_factor = float(self.config.get("drawing", {}).get("modelspace_scale_factor", 5.0))
        if scale_factor <= 0:
            scale_factor = 5.0
        total_dist_paper_mm = span_length_m / scale_factor
        half_span = total_dist_paper_mm / 2.0
        w_half = 4.5  # Half width of abutment line across track (4.5mm)
        hook_len = 2.0  # Wingwall return L-hook length (2.0mm)

        # Abutment 1 (Entry) & Abutment 2 (Exit) center positions along track
        a1_x, a1_y = px - half_span * cos_h, py - half_span * sin_h
        a2_x, a2_y = px + half_span * cos_h, py + half_span * sin_h

        # Abutment 1 Perpendicular Line
        a1_p1 = (a1_x - w_half * cos_p, a1_y - w_half * sin_p)
        a1_p2 = (a1_x + w_half * cos_p, a1_y + w_half * sin_p)
        msp.add_line(a1_p1, a1_p2, dxfattribs={"layer": "BRIDGE_ANNOTATION", "color": self.COL_RED, "lineweight": 25})

        # Abutment 1 L-shaped Wingwalls (turning outward away from bridge center)
        msp.add_line(a1_p1, (a1_p1[0] - hook_len * cos_h, a1_p1[1] - hook_len * sin_h), dxfattribs={"layer": "BRIDGE_ANNOTATION", "color": self.COL_RED, "lineweight": 25})
        msp.add_line(a1_p2, (a1_p2[0] - hook_len * cos_h, a1_p2[1] - hook_len * sin_h), dxfattribs={"layer": "BRIDGE_ANNOTATION", "color": self.COL_RED, "lineweight": 25})

        # Abutment 2 Perpendicular Line
        a2_p1 = (a2_x - w_half * cos_p, a2_y - w_half * sin_p)
        a2_p2 = (a2_x + w_half * cos_p, a2_y + w_half * sin_p)
        msp.add_line(a2_p1, a2_p2, dxfattribs={"layer": "BRIDGE_ANNOTATION", "color": self.COL_RED, "lineweight": 25})

        # Abutment 2 L-shaped Wingwalls (turning outward away from bridge center)
        msp.add_line(a2_p1, (a2_p1[0] + hook_len * cos_h, a2_p1[1] + hook_len * sin_h), dxfattribs={"layer": "BRIDGE_ANNOTATION", "color": self.COL_RED, "lineweight": 25})
        msp.add_line(a2_p2, (a2_p2[0] + hook_len * cos_h, a2_p2[1] + hook_len * sin_h), dxfattribs={"layer": "BRIDGE_ANNOTATION", "color": self.COL_RED, "lineweight": 25})

    def _draw_plan_bridge_callouts(
        self,
        msp: Any,
        bridge_df: pd.DataFrame,
        origin_x: float,
        plan_origin_y: float,
        draw_len_mm: float,
        total_length_m: float,
        start_ch: float = 0.0,
    ) -> None:
        """Render Bridge details callout text & 2-segment dogleg leader line in Plan View."""
        if bridge_df is None or bridge_df.empty:
            return

        get_paper_pos = getattr(self, "current_get_paper_pos", None)
        if get_paper_pos is None:
            return

        # Check if line is Existing Rail / Formation line
        is_existing = self.config.get("drawing", {}).get("is_existing_line", False)
        if is_existing:
            self.logger.info("Existing formation line active — Skipping bridge rendering in Plan View per engineering directive.")
            return

        end_ch = start_ch + total_length_m

        draw_count: int = 0

        ca_cfg = self.config.get("drawing", {}).get("collision_avoidance", {})
        ca_enabled = bool(ca_cfg.get("enabled", True))
        box_margin_mm = float(ca_cfg.get("box_margin_mm", 1.5))
        shift_x_step_mm = float(ca_cfg.get("shift_x_step_mm", 12.0))
        tier_step_mm = float(ca_cfg.get("tier_step_mm", 20.0))
        max_tiers = max(5, int(ca_cfg.get("max_tiers", 5)))
        stem_base = float(ca_cfg.get("stem_base_mm", 18.0))

        slant_deg: float = float(
            self.config.get("drawing", {})
                .get("plan_view", {})
                .get("bridge_callout_slant_deg", 60.0)
        )
        slant_rad = math.radians(slant_deg)

        # Multi-directional candidate evaluation space for Plan View Bridge Callouts
        stem_tiers = [stem_base + i * tier_step_mm for i in range(max_tiers)]
        if len(stem_tiers) < 7:
            stem_tiers.extend([stem_tiers[-1] + (i + 1) * tier_step_mm for i in range(7 - len(stem_tiers))])

        max_x_steps = int(ca_cfg.get("max_x_steps", 6))
        x_shifts = [0.0]
        for step_i in range(1, max_x_steps + 1):
            x_shifts.append(-float(step_i) * shift_x_step_mm)
            x_shifts.append(+float(step_i) * shift_x_step_mm)

        for idx, row in bridge_df.iterrows():
            # Skip only if explicitly marked as existing formation line
            ex_flag = row.get("is_existing", row.get("is_existing_line", row.get("existing_formation", False)))
            if pd.notna(ex_flag) and str(ex_flag).strip().lower() in ["true", "1", "yes"]:
                self.logger.info("Bridge @ row %d is on existing formation line/level — Skipping in Plan View.", idx)
                continue

            row_dict = {str(k).strip().lower(): v for k, v in row.items() if pd.notna(v)}

            raw_ch = None
            for key in [
                "approx. chainage (km)",
                "approx. chainage (m)",
                "approx. chainage",
                "chainage_m",
                "chainage_km",
                "chainage",
                "ch",
                "station",
            ]:
                if key in row_dict:
                    raw_ch = row_dict[key]
                    break

            if raw_ch is None:
                continue

            try:
                ch_val = float(raw_ch)
            except (ValueError, TypeError):
                continue

            # Auto-convert km to metres ONLY if raw value is truly in km (< 75.0)
            if 0.0 < ch_val < 75.0:
                ch_val = ch_val * 1000.0

            if not (start_ch <= ch_val <= end_ch):
                continue

            px, py, heading = get_paper_pos(ch_val)
            if not (origin_x - 0.5 <= px <= origin_x + draw_len_mm + 0.5):
                continue

            b_no = ""
            for key in ["bridge no.", "bridge no", "bridge_no", "br_no", "br no", "no", "s.no"]:
                if key in row_dict:
                    b_no = str(row_dict[key]).strip()
                    break

            span_str = ""
            for key in ["proposed span", "proposed_span", "span", "span_config", "size"]:
                if key in row_dict:
                    span_str = str(row_dict[key]).strip()
                    break

            b_type = ""
            for key in ["bridge type", "bridge type / description", "type", "structure_type", "bridge_type_raw"]:
                if key in row_dict:
                    b_type = str(row_dict[key]).strip()
                    break

            config_class = ""
            for key in ["configration", "configuration", "category", "status"]:
                if key in row_dict:
                    config_class = str(row_dict[key]).strip()
                    break

            if b_no.upper().startswith("LC") or "LC" in config_class.upper():
                text_str = f"PRO. {b_no} ({span_str if span_str else 'LC'}) AT CH.:{ch_val:.2f}M"
            elif config_class.upper() in ["RUB", "ROB"]:
                span_part = f"{span_str} " if span_str else ""
                type_part = f"{b_type} " if b_type else ""
                text_str = f"PRO. {b_no} OF {span_part}{type_part}({config_class.upper()}) AT CH.:{ch_val:.2f}M".replace("  ", " ")
            else:
                span_part = f"{span_str} " if span_str else ""
                type_part = f"{b_type}" if b_type else "CULVERT"
                text_str = f"PRO. {b_no} OF {span_part}{type_part} AT CH.:{ch_val:.2f}M".replace("  ", " ")

            txt_len_mm = len(text_str) * 1.9 + 6.0

            # Perpendicular angles to track heading
            ang_cand_a = heading + math.pi / 2.0
            ang_cand_b = heading - math.pi / 2.0
            y_a = math.sin(ang_cand_a)
            y_b = math.sin(ang_cand_b)

            perp_up_angle   = ang_cand_a if y_a >= y_b else ang_cand_b
            perp_down_angle = ang_cand_b if y_a >= y_b else ang_cand_a

            cos_h, sin_h = math.cos(heading), math.sin(heading)

            chosen_knee_x = px
            chosen_knee_y = py
            chosen_end_x = px + txt_len_mm
            chosen_end_y = py
            chosen_tx = px
            chosen_ty = py
            chosen_rot = 0.0
            chosen_box: Optional[tuple[float, float, float, float]] = None

            found_clear = False

            if ca_enabled:
                for is_upward in [True, False]:
                    perp_angle = perp_up_angle if is_upward else perp_down_angle
                    cos_p, sin_p = math.cos(perp_angle), math.sin(perp_angle)
                    shelf_angle = (heading + slant_rad) if is_upward else (heading - slant_rad)
                    cos_sh, sin_sh = math.cos(shelf_angle), math.sin(shelf_angle)
                    rot_deg = math.degrees(shelf_angle) % 360.0
                    if 90.0 < rot_deg <= 270.0:
                        rot_deg = (rot_deg + 180.0) % 360.0

                    perp_sh_angle = shelf_angle + math.pi / 2.0

                    for off_dist in stem_tiers:
                        for s_x in x_shifts:
                            # Shift stem origin along track direction
                            shifted_px = px + s_x * cos_h
                            shifted_py = py + s_x * sin_h

                            knee_x = shifted_px + off_dist * cos_p
                            knee_y = shifted_py + off_dist * sin_p

                            end_x = knee_x + txt_len_mm * cos_sh
                            end_y = knee_y + txt_len_mm * sin_sh

                            tx = knee_x + 1.5 * math.cos(perp_sh_angle) + 2.0 * cos_sh
                            ty = knee_y + 1.5 * math.sin(perp_sh_angle) + 2.0 * sin_sh

                            cand_box = self._compute_text_aabb(
                                text=text_str,
                                height_mm=3.0,
                                x=tx,
                                y=ty,
                                angle_deg=rot_deg,
                                margin_mm=box_margin_mm,
                            )

                            has_collision = False
                            for pb in self.placed_plan_boxes:
                                if self._boxes_intersect(cand_box, pb):
                                    has_collision = True
                                    break

                            if not has_collision:
                                chosen_knee_x = knee_x
                                chosen_knee_y = knee_y
                                chosen_end_x = end_x
                                chosen_end_y = end_y
                                chosen_tx = tx
                                chosen_ty = ty
                                chosen_rot = rot_deg
                                chosen_box = cand_box
                                found_clear = True
                                break
                        if found_clear:
                            break
                    if found_clear:
                        break

            if not found_clear:
                # Emergency fallback slot elevated above standard tiers
                is_upward = (draw_count % 2 == 0)
                off_dist = stem_tiers[-1] + (draw_count + 1) * tier_step_mm
                perp_angle = perp_up_angle if is_upward else perp_down_angle
                shelf_angle = (heading + slant_rad) if is_upward else (heading - slant_rad)
                cos_p, sin_p = math.cos(perp_angle), math.sin(perp_angle)
                cos_sh, sin_sh = math.cos(shelf_angle), math.sin(shelf_angle)
                perp_sh_angle = shelf_angle + math.pi / 2.0

                chosen_knee_x = px + off_dist * cos_p
                chosen_knee_y = py + off_dist * sin_p
                chosen_end_x = chosen_knee_x + txt_len_mm * cos_sh
                chosen_end_y = chosen_knee_y + txt_len_mm * sin_sh
                chosen_tx = chosen_knee_x + 1.5 * math.cos(perp_sh_angle) + 2.0 * cos_sh
                chosen_ty = chosen_knee_y + 1.5 * math.sin(perp_sh_angle) + 2.0 * sin_sh
                chosen_rot = math.degrees(shelf_angle) % 360.0
                if 90.0 < chosen_rot <= 270.0:
                    chosen_rot = (chosen_rot + 180.0) % 360.0
                chosen_box = self._compute_text_aabb(
                    text=text_str, height_mm=3.0, x=chosen_tx, y=chosen_ty, angle_deg=chosen_rot, margin_mm=box_margin_mm
                )

            if chosen_box is not None:
                self.placed_plan_boxes.append(chosen_box)

            # ── Draw DXF Entities ─────────────────────────────────────────────
            # 1. RDSO Bridge Abutments & Wingwalls Symbol at (px, py) (only for Bridges/Culverts, not for LC)
            is_lc_entry = b_no.upper().startswith("LC") or "LC" in config_class.upper() or "LEVEL CROSSING" in config_class.upper()
            if not is_lc_entry:
                self._draw_rdso_bridge_symbol(msp, px, py, heading)

            # 2. Stem: perpendicular from track to knee point
            msp.add_line(
                (px, py),
                (chosen_knee_x, chosen_knee_y),
                dxfattribs={"layer": "BRIDGE_ANNOTATION", "color": self.COL_RED, "lineweight": 25}
            )

            # 3. Shelf: parallel/slanted to track from knee point
            msp.add_line(
                (chosen_knee_x, chosen_knee_y),
                (chosen_end_x, chosen_end_y),
                dxfattribs={"layer": "BRIDGE_ANNOTATION", "color": self.COL_RED, "lineweight": 25}
            )

            # 4. Text above shelf, rotated parallel to track
            txt_ent = msp.add_text(
                text_str,
                dxfattribs={
                    "height": 3.0,
                    "layer": "BRIDGE_ANNOTATION",
                    "style": "ESP_STYLE",
                    "color": self.COL_RED,
                    "rotation": chosen_rot,
                }
            )
            txt_ent.set_placement((chosen_tx, chosen_ty), align=TextEntityAlignment.BOTTOM_LEFT)

            draw_count += 1




    def _draw_plan_topo_ortho(
        self,
        msp: Any,
        topo_res: Optional[Any],
        align_res: Optional[Any],
        start_ch: float,
        draw_length_m: float,
        origin_x: float,
        plan_origin_y: float,
        draw_len_mm: float,
        plan_area_h: float,
        bridge_df: Optional[Any] = None,
    ) -> None:
        """Orthographically project topo structures onto the plan view area.

        Matches Indian Railways Alignment Plan drafting standards:
          - Leader stem line extending from track alignment crossing tick & target circle
          - Slanted shelf underline line (30° relative to track heading)
          - Multi-line stacked text sitting directly on/above the shelf underline
          - RED text (ACI 1) for Proposed structures, BLACK text (ACI 7) for Existing structures
          - Alternating perpendicular stem offsets for zero overlap between adjacent callouts
          - NO heavy box border around text
        """
        structures = getattr(topo_res, "structures", None) if topo_res else None
        infringements_df = getattr(topo_res, "infringements_df", None) if topo_res else None

        # Parse bridge_df into structures if topo_res structures are not provided
        if not structures and bridge_df is not None and hasattr(bridge_df, "iterrows") and not bridge_df.empty:
            from types import SimpleNamespace
            structures = []
            for row_idx, br_row in bridge_df.iterrows():
                raw_ch = br_row.get("chainage_m", br_row.get("Approx. Chainage (km)", br_row.get("chainage", None)))
                if raw_ch is None or pd.isna(raw_ch):
                    raise ValueError(f"Bridge record at row {row_idx + 1} is missing required 'chainage_m' in input data.")

                try:
                    ch = float(raw_ch)
                except (ValueError, TypeError) as err:
                    raise ValueError(f"Invalid chainage value '{raw_ch}' at row {row_idx + 1} in bridge data: {err}") from err

                if ch > 0.0 and ch < 1000.0:
                    ch = ch * 1000.0

                raw_type = br_row.get("bridge_type_raw", br_row.get("Bridge Type / Description", br_row.get("bridge_type", None)))
                if raw_type is None or pd.isna(raw_type) or str(raw_type).strip() in ("", "nan", "NAN", "None", "NONE"):
                    raise ValueError(f"Bridge record at Ch {ch:.2f}m (row {row_idx + 1}) is missing required bridge type / description in input data.")

                br_type = str(raw_type).strip().upper()
                br_no = str(br_row.get("bridge_no", br_row.get("Bridge No", ""))).strip()
                if br_no in ("nan", "None", "NAN", "NONE"):
                    br_no = ""

                cat = str(br_row.get("category", br_row.get("Category", "PROPOSED"))).strip().upper()
                span = str(br_row.get("span_config", br_row.get("Span Config", ""))).strip()
                if span in ("nan", "None", "NAN", "NONE"):
                    span = ""

                is_exg = "EXG" in cat or "EXISTING" in cat
                prefix = "EXG. " if is_exg else "PRO. "
                br_no_part = f"BRIDGE NO. {br_no}" if br_no else "BRIDGE"
                name_str = f"{prefix}{br_no_part} ({br_type})"

                stype = "MAJOR_BRIDGE" if ("MAJOR" in cat or "GIRDER" in br_type or "PSC" in br_type) else "CULVERT"

                struct = SimpleNamespace(
                    chainage_m=ch,
                    structure_type=stype,
                    structure_name=name_str,
                    span_config=span,
                    metadata={"perp_offset_m": 0.0}
                )
                structures.append(struct)

        if not structures and (infringements_df is None or infringements_df.empty):
            return

        scale_mm_per_m = draw_len_mm / draw_length_m if draw_length_m > 0 else 1.0
        get_paper_pos = getattr(self, "current_get_paper_pos", None)

        # ── Configuration parameters for Indian Railways Slanted Leader Underline Callouts ──────
        callout_cfg = self.config.get("drawing", {}).get("plan_view", {}).get("structure_callout", {})
        text_h   = float(callout_cfg.get("text_height_mm", 3.0))
        col_prop = int(callout_cfg.get("color_proposed", 1))   # Red (ACI 1)
        col_exg  = int(callout_cfg.get("color_existing", 7))   # Black (ACI 7)
        slant_deg = float(callout_cfg.get("slant_angle_deg", 60.0))  # 60° slanted angle

        # Staggered stem offset levels to prevent overlapping adjacent callouts
        candidate_stem_offsets = [
            0.0,    # Level 0: Direct from track
            -25.0,  # Level 1: Below track -25mm
            18.0,   # Level 2: Staggered +18mm
            -42.0,  # Level 3: Staggered below -42mm
            35.0,   # Level 4: Staggered +35mm
            -58.0,  # Level 5: Staggered below -58mm
        ]

        # Sort structures by chainage left-to-right
        valid_structs = []
        for struct in (structures or []):
            ch = getattr(struct, "chainage_m", 0.0)
            if start_ch - 50.0 <= ch <= start_ch + draw_length_m + 50.0:
                valid_structs.append(struct)
        valid_structs.sort(key=lambda s: getattr(s, "chainage_m", 0.0))

        # ── Draw Structures ────────────────────────────────────────
        for s_idx, struct in enumerate(valid_structs):
            ch = getattr(struct, "chainage_m", 0.0)
            stype = getattr(struct, "structure_type", "CULVERT")
            sname = getattr(struct, "structure_name", stype)
            span = str(getattr(struct, "span_config", "")).strip()
            if span in ("nan", "None", "NAN", "NONE"):
                span = ""

            is_exg = "EXG" in sname.upper() or "EXISTING" in sname.upper()
            txt_color = col_exg if is_exg else col_prop

            # Format description cleanly without duplicate "BRIDGE OF BRIDGE"
            raw_sname = str(sname).strip()
            upper_sname = raw_sname.upper()

            if "BRIDGE" in upper_sname or "RUB" in upper_sname or "CULVERT" in upper_sname:
                desc_str = raw_sname
            elif span:
                desc_str = f"BRIDGE OF {span} ({raw_sname})"
            else:
                desc_str = f"BRIDGE OF {raw_sname}"

            # Clean up any accidental duplication
            desc_str = desc_str.replace("BRIDGE OF BRIDGE", "BRIDGE OF").strip()
            desc_str = desc_str.replace("BRIDGE OF PRO. BRIDGE", "PRO. BRIDGE").replace("BRIDGE OF EXG. BRIDGE", "EXG. BRIDGE").strip()
            if desc_str.startswith("OF "):
                desc_str = "BRIDGE " + desc_str

            ch_str = f"AT CH.: {self._format_chainage(ch)}"

            # Format 2-line stacked callout: Line 1 (Desc), Line 2 (Chainage)
            text_lines = [desc_str, ch_str]

            # Obtain 2D paper position & local track heading angle
            if get_paper_pos is not None:
                px, py, heading = get_paper_pos(ch)
            else:
                px = origin_x + (ch - start_ch) * scale_mm_per_m
                py = plan_origin_y
                heading = 0.0

            # Format 2-line stacked callout: Line 1 (Desc), Line 2 (Chainage)
            text_lines = [desc_str, ch_str]

            # Obtain 2D paper position & local track heading angle
            if get_paper_pos is not None:
                px, py, heading = get_paper_pos(ch)
            else:
                px = origin_x + (ch - start_ch) * scale_mm_per_m
                py = plan_origin_y
                heading = 0.0

            cos_h, sin_h = math.cos(heading), math.sin(heading)

            ca_cfg = self.config.get("drawing", {}).get("collision_avoidance", {})
            ca_enabled = bool(ca_cfg.get("enabled", True))
            box_margin_mm = float(ca_cfg.get("box_margin_mm", 1.5))
            shift_x_step_mm = float(ca_cfg.get("shift_x_step_mm", 12.0))

            stem_tiers = [0.0, 18.0, -25.0, 35.0, -42.0, 52.0, -60.0]
            x_shifts = [0.0, -1.0 * shift_x_step_mm, +1.0 * shift_x_step_mm, -2.0 * shift_x_step_mm, +2.0 * shift_x_step_mm]

            max_line_chars = max(len(l) for l in text_lines)
            txt_len_mm = max_line_chars * 2.1 + 6.0

            chosen_stem_start_x = px
            chosen_stem_start_y = py
            chosen_end_x = px + txt_len_mm
            chosen_end_y = py
            chosen_rot = 0.0
            chosen_cos_l = math.cos(math.radians(slant_deg))
            chosen_sin_l = math.sin(math.radians(slant_deg))
            chosen_cos_pl = -chosen_sin_l
            chosen_sin_pl = chosen_cos_l
            chosen_boxes: list[tuple[float, float, float, float]] = []

            found_clear = False

            if ca_enabled:
                for stem_off in stem_tiers:
                    is_above = (stem_off >= 0.0)
                    rel_angle = math.radians(slant_deg) if is_above else math.radians(-slant_deg)
                    leader_angle = heading + rel_angle
                    cos_l, sin_l = math.cos(leader_angle), math.sin(leader_angle)
                    perp_l_angle = leader_angle + math.pi / 2.0
                    cos_pl, sin_pl = math.cos(perp_l_angle), math.sin(perp_l_angle)
                    rot_deg = math.degrees(leader_angle) % 360
                    if 90.0 < rot_deg <= 270.0:
                        rot_deg = (rot_deg + 180.0) % 360

                    perp_track = heading + math.pi / 2.0

                    for s_x in x_shifts:
                        shifted_px = px + s_x * cos_h
                        shifted_py = py + s_x * sin_h

                        stem_start_x = shifted_px + abs(stem_off) * math.cos(perp_track if is_above else perp_track + math.pi)
                        stem_start_y = shifted_py + abs(stem_off) * math.sin(perp_track if is_above else perp_track + math.pi)

                        end_x = stem_start_x + txt_len_mm * cos_l
                        end_y = stem_start_y + txt_len_mm * sin_l

                        line_spacing = text_h * 1.30
                        ordered_lines = [ch_str, desc_str]

                        cand_boxes: list[tuple[float, float, float, float]] = []
                        has_collision = False

                        for li, ltxt in enumerate(ordered_lines):
                            dist_perp = 1.2 + li * line_spacing
                            tx = stem_start_x + dist_perp * cos_pl + 2.0 * cos_l
                            ty = stem_start_y + dist_perp * sin_pl + 2.0 * sin_l

                            cbox = self._compute_text_aabb(
                                text=ltxt,
                                height_mm=text_h,
                                x=tx,
                                y=ty,
                                angle_deg=rot_deg,
                                margin_mm=box_margin_mm,
                            )
                            cand_boxes.append(cbox)

                            for pb in self.placed_plan_boxes:
                                if self._boxes_intersect(cbox, pb):
                                    has_collision = True
                                    break
                            if has_collision:
                                break

                        if not has_collision:
                            chosen_stem_start_x = stem_start_x
                            chosen_stem_start_y = stem_start_y
                            chosen_end_x = end_x
                            chosen_end_y = end_y
                            chosen_rot = rot_deg
                            chosen_cos_l = cos_l
                            chosen_sin_l = sin_l
                            chosen_cos_pl = cos_pl
                            chosen_sin_pl = sin_pl
                            chosen_boxes = cand_boxes
                            found_clear = True
                            break
                    if found_clear:
                        break

            if not found_clear:
                stem_off = candidate_stem_offsets[s_idx % len(candidate_stem_offsets)]
                is_above = (stem_off >= 0.0)
                rel_angle = math.radians(slant_deg) if is_above else math.radians(-slant_deg)
                leader_angle = heading + rel_angle
                chosen_cos_l, chosen_sin_l = math.cos(leader_angle), math.sin(leader_angle)
                perp_l_angle = leader_angle + math.pi / 2.0
                chosen_cos_pl, chosen_sin_pl = math.cos(perp_l_angle), math.sin(perp_l_angle)
                chosen_rot = math.degrees(leader_angle) % 360
                if 90.0 < chosen_rot <= 270.0:
                    chosen_rot = (chosen_rot + 180.0) % 360

                perp_track = heading + math.pi / 2.0
                chosen_stem_start_x = px + abs(stem_off) * math.cos(perp_track if is_above else perp_track + math.pi)
                chosen_stem_start_y = py + abs(stem_off) * math.sin(perp_track if is_above else perp_track + math.pi)
                chosen_end_x = chosen_stem_start_x + txt_len_mm * chosen_cos_l
                chosen_end_y = chosen_stem_start_y + txt_len_mm * chosen_sin_l

            for cb in chosen_boxes:
                self.placed_plan_boxes.append(cb)

            # 1. Alignment Crossing Target Tick & Circle Dot on Track Line
            perp_track = heading + math.pi / 2.0
            msp.add_line(
                (px - 4.0 * math.cos(perp_track), py - 4.0 * math.sin(perp_track)),
                (px + 4.0 * math.cos(perp_track), py + 4.0 * math.sin(perp_track)),
                dxfattribs={"layer": "STRUCTURES_PLAN", "color": txt_color, "lineweight": 25}
            )
            msp.add_circle(
                center=(px, py),
                radius=1.2,
                dxfattribs={"layer": "STRUCTURES_PLAN", "color": txt_color, "lineweight": 25}
            )

            # Connector line from track tick to stem_start if staggered
            if math.hypot(chosen_stem_start_x - px, chosen_stem_start_y - py) > 1.0:
                msp.add_line(
                    (px, py),
                    (chosen_stem_start_x, chosen_stem_start_y),
                    dxfattribs={"layer": "STRUCTURES_PLAN", "color": txt_color, "lineweight": 18}
                )

            # 2. Slanted Red Underline Line (acting as continuous underline under text)
            msp.add_line(
                (chosen_stem_start_x, chosen_stem_start_y),
                (chosen_end_x, chosen_end_y),
                dxfattribs={"layer": "STRUCTURES_PLAN", "color": txt_color, "lineweight": 25}
            )

            # 3. Render Stacked Text Lines Sitting Above Underline Line:
            line_spacing = text_h * 1.30
            ordered_lines = [ch_str, desc_str]  # [Bottom line, Top line]

            for li, ltxt in enumerate(ordered_lines):
                dist_perp = 1.2 + li * line_spacing
                tx = chosen_stem_start_x + dist_perp * chosen_cos_pl + 2.0 * chosen_cos_l
                ty = chosen_stem_start_y + dist_perp * chosen_sin_pl + 2.0 * chosen_sin_l

                txt_ent = msp.add_text(
                    ltxt,
                    dxfattribs={
                        "height": text_h,
                        "layer": "STRUCTURES_PLAN",
                        "style": "ESP_STYLE",
                        "color": txt_color,
                        "rotation": chosen_rot,
                    }
                )
                txt_ent.set_placement((tx, ty), align=TextEntityAlignment.BOTTOM_LEFT)


        # ── Draw Building Infringements (as orange hatched squares) ────────
        if infringements_df is not None and not infringements_df.empty:
            for _, row in infringements_df.iterrows():
                ch     = float(row.get("chainage_m", 0.0))
                offset = float(row.get("offset_m",   10.0))   # metres from centreline
                if not (start_ch - 10.0 <= ch <= start_ch + draw_length_m + 10.0):
                    continue

                px, py = _project_structure(ch, offset)
                sz = 3.0  # symbol size in mm
                hatch = msp.add_hatch(color=INFR_COLOR)
                hatch.paths.add_polyline_path(
                    [(px - sz/2, py - sz/2), (px + sz/2, py - sz/2),
                     (px + sz/2, py + sz/2), (px - sz/2, py + sz/2)],
                    is_closed=True
                )
                msp.add_lwpolyline(
                    [(px - sz/2, py - sz/2), (px + sz/2, py - sz/2),
                     (px + sz/2, py + sz/2), (px - sz/2, py + sz/2),
                     (px - sz/2, py - sz/2)],
                    dxfattribs={"layer": "STRUCTURES_PLAN", "color": INFR_COLOR, "lineweight": 9}
                )

    def _draw_plan_structure_and_turnout_callouts(
        self,
        msp: Any,
        bridge_df: Optional[Any],
        topo_res: Optional[Any],
        align_res: Optional[Any],
        total_length_m: float,
        origin_x: float,
        origin_y: float,
        draw_len_mm: float,
    ) -> None:
        """Render structure text callouts and turnout (1 IN 12 SRJ) annotations."""
        scale_factor = draw_len_mm / total_length_m if total_length_m > 0 else 1.0

        # Dead end callout at Start (CH: 0.000m)
        x_start = origin_x
        msp.add_text(
            f"DEAD END AT CH: 0.000m",
            dxfattribs={
                "height": 2.2, "rotation": 30, "layer": "STRUCTURES_PLAN",
                "style": "ESP_STYLE", "color": self.COL_RED,
            }
        ).set_placement((x_start, origin_y - 12.0), align=TextEntityAlignment.MIDDLE_LEFT)

        # Dead end callout at End
        x_end = origin_x + draw_len_mm
        msp.add_text(
            f"DEAD END AT CH: {total_length_m:.3f}m",
            dxfattribs={
                "height": 2.2, "rotation": 30, "layer": "STRUCTURES_PLAN",
                "style": "ESP_STYLE", "color": self.COL_RED,
            }
        ).set_placement((x_end - 15.0, origin_y - 12.0), align=TextEntityAlignment.MIDDLE_LEFT)

        # Render turnouts (e.g. 1 IN 12 SRJ at start and end approaches)
        if total_length_m > 200:
            srj1_ch = 120.0
            x_srj1 = origin_x + srj1_ch * scale_factor
            msp.add_text(
                f"SRJ OF 1 IN 12 AT CH: {srj1_ch:.2f}m",
                dxfattribs={
                    "height": 2.0, "rotation": 30, "layer": "TURNOUTS_PLAN",
                    "style": "ESP_STYLE", "color": self.COL_RED,
                }
            ).set_placement((x_srj1, origin_y - 8.0), align=TextEntityAlignment.MIDDLE_LEFT)

    def _draw_plan_junction_arrows(
        self,
        msp: Any,
        total_length_m: float,
        origin_x: float,
        origin_y: float,
        draw_len_mm: float,
    ) -> None:
        """Render directional junction arrows (e.g. <<< TOWARDS BHERAGHAT / TOWARDS KACHHPURA >>>)."""
        # Start junction arrow — horizontal, to the left of the alignment start
        msp.add_text(
            "<<< TOWARDS BHERAGHAT JUNCTION",
            dxfattribs={
                "height": 2.5, "rotation": 0, "layer": "JUNCTION_ARROWS",
                "style": "ESP_STYLE", "color": self.COL_BLACK,
            }
        ).set_placement((origin_x - 5.0, origin_y), align=TextEntityAlignment.MIDDLE_RIGHT)

        # End junction arrow — horizontal, to the right of the alignment end
        x_end = origin_x + draw_len_mm
        msp.add_text(
            "TOWARDS KACHHPURA JUNCTION >>>",
            dxfattribs={
                "height": 2.5, "rotation": 0, "layer": "JUNCTION_ARROWS",
                "style": "ESP_STYLE", "color": self.COL_BLACK,
            }
        ).set_placement((x_end + 5.0, origin_y), align=TextEntityAlignment.MIDDLE_LEFT)

    def _draw_plan_north_arrow(
        self, msp: Any, center_x: float, center_y: float, size: float = 20.0
    ) -> None:
        """Draw Indian Railways standard North Arrow symbol."""
        # Arrow shaft
        msp.add_line(
            (center_x, center_y - size / 2.0),
            (center_x, center_y + size / 2.0),
            dxfattribs={"layer": "NORTH_ARROW", "color": self.COL_BLACK, "lineweight": 25}
        )
        # Arrowhead pointer
        msp.add_lwpolyline(
            [
                (center_x - size * 0.15, center_y + size * 0.25),
                (center_x, center_y + size / 2.0),
                (center_x + size * 0.15, center_y + size * 0.25),
            ],
            dxfattribs={"layer": "NORTH_ARROW", "color": self.COL_BLACK, "lineweight": 25}
        )
        # 'N' Label
        msp.add_text(
            "N",
            dxfattribs={
                "height": size * 0.3, "layer": "NORTH_ARROW",
                "style": "ESP_STYLE", "color": self.COL_BLACK,
            }
        ).set_placement((center_x, center_y + size * 0.6), align=TextEntityAlignment.MIDDLE_CENTER)

    def _export_to_pdf(self, dxf_path: Path, pdf_path: Path) -> None:
        """Export DXF drawing to vector PDF format using ezdxf matplotlib addon with 1:1 true scale."""
        if not bool(self.config.get("drawing", {}).get("export_pdf", False)):
            self.logger.info("PDF export disabled in config. Skipping PDF vector rendering for %s", dxf_path.name)
            return
        try:
            import matplotlib
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42
            matplotlib.rcParams['pdf.compression'] = 6

            import matplotlib.pyplot as plt
            from ezdxf.addons.drawing import RenderContext, Frontend
            from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
            from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy

            doc = ezdxf.readfile(str(dxf_path))
            msp = doc.modelspace()

            limmax = doc.header.get("$LIMMAX", (1189.0, 841.0))
            sheet_w_mm = max(1189.0, float(limmax[0]))
            sheet_h_mm = max(841.0, float(limmax[1]))

            # Set large canvas dimensions matching reference CAD plotters (1189mm x 841mm)
            sheet_w_in = sheet_w_mm / 25.4
            sheet_h_in = sheet_h_mm / 25.4

            fig = plt.figure(figsize=(sheet_w_in, sheet_h_in), dpi=300)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.set_xlim(0, sheet_w_mm)
            ax.set_ylim(0, sheet_h_mm)
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
                self.logger.warning("ezdxf layout draw note for %s: %s", dxf_path.name, e_draw)

            # Re-enforce explicit canvas extents so PDF viewer at 25% zoom fills full screen
            ax.set_xlim(0, sheet_w_mm)
            ax.set_ylim(0, sheet_h_mm)
            fig.set_size_inches(sheet_w_in, sheet_h_in, forward=True)

            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(pdf_path), format="pdf")
            plt.close(fig)
            self.logger.info("Successfully exported large-canvas PDF drawing to %s", pdf_path)
        except Exception as exc:
            self.logger.warning("PDF export failed for %s: %s", dxf_path, exc)

    def _draw_plan_topo_dwg_overlay(
        self,
        msp: Any,
        align_res: Optional[Any],
        start_ch: float,
        draw_length_m: float,
        origin_x: float,
        plan_origin_y: float,
        draw_len_mm: float,
        plan_area_h: float,
    ) -> None:
        """Overlay true 2D vector topomap drawing features (roads, buildings, rivers, boundaries) from DWG/DXF onto the Plan View."""
        if align_res is None or not hasattr(align_res, "alignment_line") or align_res.alignment_line is None:
            return

        align_line = align_res.alignment_line

        # Locate DWG / DXF / KML topomap file
        dwg_candidates = [
            Path(r"C:\KanthiVada\Kanthi_Topo.kml"),
            Path(r"C:\KanthiVada\Kanthi\Kanthi_Topomap\Kanthi_Topomap.dxf"),
            Path(r"C:\KanthiVada\Kanthi\Kanthi_Topomap\Kanthi_Topomap.dwg"),
            Path(r"D:\ESP\ROB_Topomap_SHP"),
        ]

        topo_file = None
        for p in dwg_candidates:
            if p.exists():
                topo_file = p
                break

        if topo_file is None:
            top_dir = Path(r"C:\KanthiVada\Kanthi\Kanthi_Topomap")
            if top_dir.exists():
                for f in top_dir.glob("*"):
                    if f.suffix.lower() in [".kml", ".dxf", ".dwg", ".shp"]:
                        topo_file = f
                        break

        if topo_file is None:
            return

        self.logger.info("Overlaying Topomap features from %s onto Plan View...", topo_file.name)

        try:
            # 0. If KML file, parse Placemarks (WGS84 -> UTM)
            if topo_file.suffix.lower() == ".kml":
                import xml.etree.ElementTree as ET
                from shapely.geometry import LineString
                from pyproj import Transformer

                utm_info = getattr(align_res, "utm_info", None)
                epsg_code = getattr(utm_info, "epsg", "EPSG:32645") if utm_info else "EPSG:32645"
                transformer = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)

                tree = ET.parse(str(topo_file))
                root = tree.getroot()

                for elem in root.iter():
                    if elem.tag.endswith('Placemark'):
                        name = "TOPO"
                        for child in elem:
                            if child.tag.endswith('name') and child.text:
                                name = child.text.upper()

                        # Priority Directive: Draw Outer Boundary, River, Road, and Canal topomap features.
                        target_keywords = [
                            "BOUNDARY", "OUTER", "LIMIT", "PERIMETER", "SURVEY", "EXTENTS", "TOPO", "PL", "FRAME",
                            "RIVER", "ROAD", "CANAL", "DRAIN", "STREAM", "WATER", "NALA",
                            "HIGHWAY", "TRACK", "PATH", "STREET", "EXPRESSWAY", "DISTRIBUTARY", "NULLAH"
                        ]

                        # Extract KML LineStyle color code to classify Outer Boundary (ff989898) vs Internal Details (ff132613)
                        kml_color = ""
                        for c_elem in elem.iter():
                            if c_elem.tag.endswith('color') and c_elem.text:
                                kml_color = c_elem.text.strip().lower()
                                break

                        is_outer_boundary = (
                            "989898" in kml_color
                            or "ffff00" in kml_color
                            or "boundary" in name.lower()
                            or "outer" in name.lower()
                            or name in ["KML_1", "KML_2"]
                        )

                        only_outer_boundary = bool(
                            self.config.get("drawing", {})
                            .get("plan_view", {})
                            .get("only_outer_boundary", True)
                        )

                        # Strict Directive: Only render Outer Boundary lines. Skip internal detail clutter.
                        if only_outer_boundary and not is_outer_boundary:
                            continue

                        # Color assignment: Grey (8) for Outer Boundary, Blue (5) for Rivers, Cyan (4) for Roads/Details
                        if is_outer_boundary:
                            color = 8  # Dark Grey (ACI 8) for Outer Boundary
                        elif any(k in name for k in ["RIVER", "CANAL", "DRAIN", "STREAM", "WATER", "NALA", "NULLAH"]):
                            color = 5  # Blue (ACI 5) for Water
                        elif any(k in name for k in ["ROAD", "HIGHWAY", "TRACK", "PATH", "STREET"]):
                            color = 4  # Cyan (ACI 4) for Roads
                        else:
                            color = 4 if "132613" in kml_color else 8


                        # Minimum length threshold (5.0m)
                        min_len_m = float(
                            self.config.get("drawing", {})
                            .get("plan_view", {})
                            .get("min_topo_boundary_length_m", 5.0)
                        )

                        is_boundary_color = is_outer_boundary or kml_color == ""
                        is_internal_detail = "132613" in kml_color and not is_boundary_color

                        # Parse Polygon outer boundaries ONLY
                        has_poly = False
                        for poly_elem in elem.iter():
                            if poly_elem.tag.endswith('Polygon'):
                                outer_coords_text = None
                                for sub in poly_elem.iter():
                                    if sub.tag.endswith('outerBoundaryIs'):
                                        for c_elem in sub.iter():
                                            if c_elem.tag.endswith('coordinates') and c_elem.text:
                                                outer_coords_text = c_elem.text
                                                break
                                if not outer_coords_text:
                                    for c_elem in poly_elem.iter():
                                        if c_elem.tag.endswith('coordinates') and c_elem.text:
                                            outer_coords_text = c_elem.text
                                            break
                                if outer_coords_text:
                                    pts = []
                                    for cstr in outer_coords_text.strip().split():
                                        parts = cstr.split(',')
                                        if len(parts) >= 2:
                                            x_u, y_u = transformer.transform(float(parts[0]), float(parts[1]))
                                            pts.append((x_u, y_u))
                                    if len(pts) >= 3:
                                        from shapely.geometry import Polygon as ShapelyPoly
                                        poly_obj = ShapelyPoly(pts)
                                        if (is_boundary_color or poly_obj.exterior.length >= min_len_m) and not (is_internal_detail and poly_obj.exterior.length < min_len_m):
                                            self._project_and_draw_geom(
                                                msp, poly_obj, align_line, start_ch, draw_length_m,
                                                origin_x, plan_origin_y, draw_len_mm, plan_area_h, color=color,
                                                get_paper_pos_func=getattr(self, "current_get_paper_pos", None),
                                            )
                                            has_poly = True

                        # Parse standalone LineString boundaries
                        if not has_poly:
                            for ls in elem.iter():
                                if ls.tag.endswith('LineString'):
                                    for c_elem in ls.iter():
                                        if c_elem.tag.endswith('coordinates') and c_elem.text:
                                            pts = []
                                            for cstr in c_elem.text.strip().split():
                                                parts = cstr.split(',')
                                                if len(parts) >= 2:
                                                    x_u, y_u = transformer.transform(float(parts[0]), float(parts[1]))
                                                    pts.append((x_u, y_u))
                                            if len(pts) >= 2:
                                                ls_obj = LineString(pts)
                                                if (is_boundary_color or ls_obj.length >= min_len_m) and not (is_internal_detail and ls_obj.length < min_len_m):
                                                    self._project_and_draw_geom(
                                                        msp, ls_obj, align_line, start_ch, draw_length_m,
                                                        origin_x, plan_origin_y, draw_len_mm, plan_area_h, color=color,
                                                        get_paper_pos_func=getattr(self, "current_get_paper_pos", None),
                                                    )

            # 1. If shapefile directory or file, use GeoPandas
            elif topo_file.is_dir() or topo_file.suffix.lower() in [".shp", ".geojson"]:
                import geopandas as gpd
                only_outer_boundary = bool(
                    self.config.get("drawing", {})
                    .get("plan_view", {})
                    .get("only_outer_boundary", True)
                )
                shp_files = list(topo_file.glob("*.shp")) if topo_file.is_dir() else [topo_file]
                for shp in shp_files:
                    try:
                        gdf = gpd.read_file(shp)
                        layer_name = shp.stem.upper()
                        is_outer = any(k in layer_name for k in ["BOUNDARY", "OUTER", "LIMIT", "PERIMETER", "SURVEY", "EXTENTS", "FRAME", "TOPO", "PL"])
                        if only_outer_boundary and not is_outer:
                            continue

                        color = 8 if is_outer else (
                            5 if any(k in layer_name for k in ["RIVER", "CANAL", "DRAIN", "STREAM", "WATER", "NALA", "NULLAH", "DISTRIBUTARY"]) else 4
                        )
                        for _, row in gdf.iterrows():
                            geom = row.geometry
                            if geom is None or geom.is_empty:
                                continue
                            self._project_and_draw_geom(
                                msp, geom, align_line, start_ch, draw_length_m,
                                origin_x, plan_origin_y, draw_len_mm, plan_area_h, color=color,
                                get_paper_pos_func=getattr(self, "current_get_paper_pos", None),
                            )
                    except Exception:
                        pass

            # 2. If DXF file, use ezdxf
            elif topo_file.suffix.lower() == ".dxf":
                doc_topo = ezdxf.readfile(str(topo_file))
                msp_topo = doc_topo.modelspace()
                only_outer_boundary = bool(
                    self.config.get("drawing", {})
                    .get("plan_view", {})
                    .get("only_outer_boundary", True)
                )
                for entity in msp_topo:
                    if entity.dxftype() in ["LINE", "LWPOLYLINE", "POLYLINE"]:
                        layer_name = entity.dxf.layer.upper() if hasattr(entity, "dxf") and hasattr(entity.dxf, "layer") else ""
                        is_outer = any(k in layer_name for k in ["BOUNDARY", "OUTER", "LIMIT", "PERIMETER", "SURVEY", "EXTENTS", "FRAME", "TOPO", "PL"])
                        if only_outer_boundary and not is_outer:
                            continue

                        color = 8 if is_outer else (
                            5 if any(k in layer_name for k in ["RIVER", "CANAL", "DRAIN", "STREAM", "WATER", "NALA", "NULLAH", "DISTRIBUTARY"]) else 4
                        )
                        pts = entity.get_points(format="xy") if hasattr(entity, "get_points") else []
                        if len(pts) >= 2:
                            from shapely.geometry import LineString
                            ls = LineString(pts)
                            self._project_and_draw_geom(
                                msp, ls, align_line, start_ch, draw_length_m,
                                origin_x, plan_origin_y, draw_len_mm, plan_area_h, color=color,
                                get_paper_pos_func=getattr(self, "current_get_paper_pos", None),
                            )




            # 3. If DWG file, try GDAL/GeoPandas CAD driver
            elif topo_file.suffix.lower() == ".dwg":
                try:
                    import geopandas as gpd
                    target_keywords = [
                        "RIVER", "ROAD", "CANAL", "DRAIN", "STREAM", "WATER", "NALA",
                        "HIGHWAY", "TRACK", "PATH", "STREET", "EXPRESSWAY", "DISTRIBUTARY", "NULLAH"
                    ]
                    gdf = gpd.read_file(str(topo_file))
                    for _, row in gdf.iterrows():
                        layer_name = str(row.get("layer", row.get("LAYER", ""))).upper()
                        if layer_name and not any(k in layer_name for k in target_keywords):
                            continue
                        color = 5 if any(k in layer_name for k in ["RIVER", "CANAL", "DRAIN", "STREAM", "WATER", "NALA", "NULLAH", "DISTRIBUTARY"]) else 4
                        geom = row.geometry
                        if geom is not None and not geom.is_empty:
                            self._project_and_draw_geom(
                                msp, geom, align_line, start_ch, draw_length_m,
                                origin_x, plan_origin_y, draw_len_mm, plan_area_h, color=color,
                                get_paper_pos_func=getattr(self, "current_get_paper_pos", None),
                            )
                except Exception as e_dwg:
                    self.logger.warning("Could not read DWG via GDAL: %s", e_dwg)
        except Exception as exc:
            self.logger.warning("Topomap overlay failed: %s", exc)

    def _project_and_draw_geom(
        self,
        msp: Any,
        geom: Any,
        align_line: Any,
        start_ch: float,
        draw_length_m: float,
        origin_x: float,
        plan_origin_y: float,
        draw_len_mm: float,
        plan_area_h: float,
        color: int = 8,
        get_paper_pos_func: Optional[Any] = None,
    ) -> None:
        """Project a Shapely 2D geometry onto paper space inside the Plan View box, anchoring perpendicularly to the 2D track centerline."""
        try:
            from shapely.geometry import Point, LineString, Polygon, MultiPolygon

            end_ch = start_ch + draw_length_m
            scale_mm_per_m = draw_len_mm / draw_length_m if draw_length_m > 0 else 1.0
            max_offset_mm = (plan_area_h / 2.0) * 0.92

            if isinstance(geom, Point):
                coords = [(geom.x, geom.y)]
            elif isinstance(geom, (LineString, Polygon)):
                coords = list(geom.exterior.coords) if isinstance(geom, Polygon) else list(geom.coords)
            elif isinstance(geom, MultiPolygon):
                coords = []
                for poly in geom.geoms:
                    coords.extend(list(poly.exterior.coords))
            else:
                return

            paper_pts = []
            for x_m, y_m in coords:
                pt = Point(x_m, y_m)
                s = float(align_line.project(pt))
                if not (start_ch - 100.0 <= s <= end_ch + 100.0):
                    continue

                d_m = float(align_line.distance(pt))
                # Topomap lateral corridor boundary filter (30.0m max offset left & right of track centerline)
                max_topo_offset_m = float(
                    self.config.get("drawing", {})
                    .get("plan_view", {})
                    .get("max_topo_offset_m", 30.0)
                )
                if d_m > max_topo_offset_m:
                    continue


                nearest_pt = align_line.interpolate(s)
                next_pt = align_line.interpolate(min(align_line.length, s + 1.0))

                dx_track = next_pt.x - nearest_pt.x
                dy_track = next_pt.y - nearest_pt.y
                cross_prod = dx_track * (y_m - nearest_pt.y) - dy_track * (x_m - nearest_pt.x)
                side = 1.0 if cross_prod >= 0 else -1.0

                if get_paper_pos_func is not None:
                    align_px, align_py, heading = get_paper_pos_func(s)
                else:
                    align_px = origin_x + (s - start_ch) * scale_mm_per_m
                    align_py = plan_origin_y
                    heading = math.atan2(dy_track, dx_track) if (dx_track != 0 or dy_track != 0) else 0.0

                d_mm = min(max_offset_mm, d_m * scale_mm_per_m)
                perp_angle = heading + (math.pi / 2.0 if side >= 0 else -math.pi / 2.0)

                plan_bot_y = plan_origin_y - (plan_area_h / 2.0)
                plan_top_y = plan_origin_y + (plan_area_h / 2.0)

                raw_px = align_px + d_mm * math.cos(perp_angle)
                raw_py = align_py + d_mm * math.sin(perp_angle)

                # Clamp strictly inside red Plan View box rectangle
                px = max(origin_x + 1.0, min(origin_x + draw_len_mm - 1.0, raw_px))
                py = max(plan_bot_y + 1.5, min(plan_top_y - 1.5, raw_py))

                paper_pts.append((px, py))

            if len(paper_pts) >= 2:
                msp.add_lwpolyline(
                    paper_pts,
                    dxfattribs={"layer": "TOPO_OVERLAY", "color": color, "lineweight": 0}
                )
        except Exception:
            pass
