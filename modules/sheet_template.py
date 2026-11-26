"""
Module: modules/sheet_template.py
Purpose: Modular Drawing Sheet Template Engine for Greenfield and Brownfield
         Railway Engineering Scale Plans (ESP) and L-Section Composite Working Plans.

Engineering Standards:
    - Sheet Standard: ISO A0 (1189mm x 841mm) / Large-format Roll (1189mm x 915mm).
    - Spatial Partitioning (Standard Indian Railways Right-Sidebar Layout):
        * Section 1: Plan View Section (Top Region)
            - Greenfield: Single/Double Proposed alignment centerline, tangents, true arcs,
              clothoid transition spirals, 100m perpendicular ticks, structure callouts, topo.
            - Brownfield: Dual-track corridor (White Existing + Red Proposed with ticks),
              track center spacing callouts, curve callouts, station pennants, topo.
        * Section 2: Longitudinal Section (L-Section Profile) Section (Bottom Region)
            - Profile Grid: Datum line rounded to 5m, 5m elevation grid, EGL/PFL/PRL polylines,
              vertical curves (summit/sag parabolic curves with VIP, Lv), cut/fill hatch ords.
            - Data Bands Matrix: 10 standard bands (Greenfield) or 8-13 bands (Brownfield).
            - Horizontal Scale 1:5,000, Vertical Scale 1:500 (Vertical Exaggeration = 10x).
        * Section 3: Notes Section (Right Sidebar Panel, 153mm - 185mm)
            - General Notes (RDSO Notes 1 to 11: track buildup 0.762m, ruling gradient 1:150,
              vertical curve criteria 4mm/m, land boundary authority, colors, metric units).
        * Section 4: Title Block Section (Right Sidebar Panel bottom compartment)
            - Zonal Railway banner, Division, Project Title, CAO/C Drawing No, Sheet X of Y,
              Scales specification, Multi-Officer Approval grid, Consultant address.

Standards: RDSO / IRSOD BG 2004/2022 / Indian Railways Works Manual (IRWM)
Author: Railway ESP Generator Team
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import ezdxf
from ezdxf.enums import TextEntityAlignment

from utils.logger import get_logger
from utils.geo_utils import format_chainage

logger = get_logger("SheetTemplate")


def _require_cfg(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Return required key from a config mapping or raise ValueError."""
    if not isinstance(mapping, dict) or key not in mapping or mapping[key] is None:
        raise ValueError(f"Missing required config key '{key}' in {context}.")
    return mapping[key]


def _require_any_cfg(mappings: Sequence[dict[str, Any]], keys: Sequence[str], context: str) -> Any:
    """Return the first present non-None value across candidate mappings and keys."""
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    raise ValueError(f"Missing required config key (one of {list(keys)}) in {context}.")


_ACI_COLOR_MAP: dict[str, int] = {
    "RED": 1,
    "YELLOW": 2,
    "GREEN": 3,
    "CYAN": 4,
    "BLUE": 5,
    "MAGENTA": 6,
    "WHITE": 7,
    "BLACK": 7,
}


def _resolve_aci_color(value: Any) -> int:
    """Convert an integer or standard CAD color name to an AutoCAD Color Index."""
    if isinstance(value, int):
        return value
    name = str(value).strip().upper()
    if name in _ACI_COLOR_MAP:
        return _ACI_COLOR_MAP[name]
    return int(value)


@dataclass
class BoundingBox:
    """Represents a 2D Cartesian rectangular bounding box in sheet millimeters or model units."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """Horizontal width of bounding box."""
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        """Vertical height of bounding box."""
        return max(0.0, self.y_max - self.y_min)

    @property
    def center_x(self) -> float:
        """Horizontal midpoint coordinate."""
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        """Vertical midpoint coordinate."""
        return (self.y_min + self.y_max) / 2.0

    def contains(self, x: float, y: float) -> bool:
        """Check if a coordinate point is strictly inside the bounding box."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def scaled(self, s: float) -> BoundingBox:
        """Return a new BoundingBox scaled by factor s."""
        return BoundingBox(self.x_min * s, self.y_min * s, self.x_max * s, self.y_max * s)

    def to_polyline_points(self, s: float = 1.0) -> list[tuple[float, float]]:
        """Return 5 closed clockwise polyline vertex tuples scaled by factor s."""
        return [
            (self.x_min * s, self.y_min * s),
            (self.x_max * s, self.y_min * s),
            (self.x_max * s, self.y_max * s),
            (self.x_min * s, self.y_max * s),
            (self.x_min * s, self.y_min * s),
        ]


@dataclass
class PlanViewSection:
    """Represents the 2D Horizontal Alignment Plan View section (Top Region of sheet)."""
    bbox: BoundingBox
    header_bbox: BoundingBox
    corridor_bbox: BoundingBox
    scale: float = 5000.0
    track_mode: str = "single_proposed"  # "single_proposed" | "double_proposed" | "existing_and_proposed"
    existing_track_color: int = 7       # ACI 7 (White/Black)
    existing_track_lineweight: int = 35
    proposed_track_color: int = 1       # ACI 1 (Red)
    proposed_track_lineweight: int = 50
    chainage_tick_interval_m: float = 100.0
    major_tick_interval_m: float = 100.0
    dimension_interval_m: float = 200.0
    show_topo: bool = True
    show_north_arrow: bool = True

    @property
    def width_mm(self) -> float:
        return self.bbox.width

    @property
    def height_mm(self) -> float:
        return self.bbox.height


@dataclass
class LSectionBandDef:
    """Represents an official RDSO / Indian Railways Longitudinal Section Profile Data Band."""
    title: str
    lines: list[str]
    height_mm: float = 18.0
    header_color: int = 7
    value_color: int = 7
    field_name: str = "generic"
    format_type: str = "elevation"  # 'elevation' | 'signed' | 'chainage' | 'km_stone' | 'geometry' | 'gradient'


@dataclass
class LongitudinalSection:
    """Represents the Longitudinal Profile View & Data Bands section (Bottom Region of sheet)."""
    bbox: BoundingBox
    profile_bbox: BoundingBox
    databands_bbox: BoundingBox
    label_column_bbox: BoundingBox
    draw_len_mm: float
    h_scale: float = 5000.0
    v_scale: float = 500.0
    num_bands: int = 10
    band_row_height_mm: float = 18.0
    datum_round_step_m: float = 5.0
    datum_clearance_m: float = 20.0
    bands_config: list[LSectionBandDef] = field(default_factory=list)
    station_tick_interval_m: float = 20.0
    station_grid_interval_m: float = 100.0
    km_interval_m: float = 1000.0
    label_text_height_mm: float = 2.4
    station_text_height_mm: float = 2.0
    km_text_height_mm: float = 3.0

    @property
    def vertical_exaggeration(self) -> float:
        """Ratio of Horizontal Scale to Vertical Scale (typically 5000 / 500 = 10x)."""
        return (self.h_scale / self.v_scale) if self.v_scale > 0 else 1.0

    @property
    def width_mm(self) -> float:
        return self.bbox.width

    @property
    def height_mm(self) -> float:
        return self.bbox.height

    def calculate_datum(self, min_elevation_m: float) -> float:
        """Calculate Indian Railways standard rounded elevation datum line."""
        step = max(1.0, self.datum_round_step_m)
        clearance = self.datum_clearance_m
        return math.floor((min_elevation_m - clearance) / step) * step

    def chainage_to_sheet_x(self, chainage_m: float, start_chainage_m: float) -> float:
        """Transform along-track ground chainage into sheet X coordinate in millimeters."""
        offset_m = chainage_m - start_chainage_m
        return self.profile_bbox.x_min + (offset_m / self.h_scale) * 1000.0

    def elevation_to_sheet_y(self, elevation_m: float, datum_elev_m: float) -> float:
        """Transform ground elevation into profile sheet Y coordinate in millimeters."""
        h_diff_m = elevation_m - datum_elev_m
        return self.profile_bbox.y_min + (h_diff_m / self.v_scale) * 1000.0


@dataclass
class NotesSection:
    """Represents the General Notes block inside the Right Sidebar Panel."""
    bbox: BoundingBox
    title: str = "GENERAL NOTES:-"
    title_color: int = 1  # ACI 1 (Red)
    notes_list: list[str] = field(default_factory=list)
    text_height_mm: float = 2.4
    line_spacing_mm: float = 4.2


@dataclass
class TitleBlockSection:
    """Represents the official Indian Railways Title Block & Approval compartment."""
    bbox: BoundingBox
    railway_zone: str = "SOUTH CENTRAL RAILWAY"
    division: str = "SECUNDERABAD DIVISION"
    project_title: str = "RAILWAY CORRIDOR PROJECT"
    plan_title: str = "WORKING PLAN AND LONGITUDINAL SECTION"
    drawing_prefix: str = "CAO/C/TR/LS"
    drawing_year: str = "2026"
    scales_text: str = "PLAN 1:5000 | PROFILE H=1:5000, V=1:500 | GAUGE: BG (1676mm)"
    officers: list[str] = field(default_factory=list)
    consultant_name: str = "FLS DESIGN CONSULTANT"
    consultant_address: str = "NEW DELHI, INDIA"


@dataclass
class SidebarSection:
    """Represents the full-height Right Sidebar Panel (153mm - 185mm) on the drawing sheet."""
    bbox: BoundingBox
    width_mm: float
    color: int = 7  # ACI 7 (White/Black)
    lineweight: int = 35
    north_arrow_bbox: BoundingBox = field(init=False)
    schedules_bbox: BoundingBox = field(init=False)
    notes_section: NotesSection = field(init=False)
    title_block_section: TitleBlockSection = field(init=False)

    def __post_init__(self) -> None:
        """Partition the vertical space of the sidebar panel into functional compartments."""
        x0, y0, x1, y1 = self.bbox.as_tuple()
        total_h = self.bbox.height

        # 1. Header & North Arrow Compartment (Top 45mm)
        north_h = min(50.0, total_h * 0.06)
        self.north_arrow_bbox = BoundingBox(x0, y1 - north_h, x1, y1)

        # 2. Bottom Title Block & Signature Compartment (Bottom 230mm)
        title_block_h = min(240.0, total_h * 0.28)
        self.title_block_section = TitleBlockSection(
            bbox=BoundingBox(x0, y0, x1, y0 + title_block_h)
        )

        # 3. Notes Compartment (Directly above Title Block, ~180mm)
        notes_h = min(220.0, total_h * 0.24)
        notes_y0 = y0 + title_block_h
        notes_y1 = notes_y0 + notes_h
        self.notes_section = NotesSection(
            bbox=BoundingBox(x0, notes_y0, x1, notes_y1)
        )

        # 4. Schedules & Tables Compartment (Middle zone)
        sched_y0 = notes_y1
        sched_y1 = y1 - north_h
        self.schedules_bbox = BoundingBox(x0, sched_y0, x1, sched_y1)


class SheetTemplate:
    """Unified Drawing Sheet Template Engine for Greenfield and Brownfield Railway CAD Plans.

    Orchestrates spatial budgeting, coordinate viewports, and drafting bounding boxes
    for Plan View, Longitudinal Section, Right Sidebar Notes, and Title Block.
    """

    def __init__(
        self,
        name: str,
        template_type: str,
        sheet_width_mm: float,
        sheet_height_mm: float,
        margin_mm: float,
        sidebar_width_mm: float,
        sidebar_gap_mm: float,
        label_width_mm: float,
        margin_mid_mm: float,
        plan_view: PlanViewSection,
        longitudinal_section: LongitudinalSection,
        sidebar: SidebarSection,
        config: Optional[dict[str, Any]] = None,
        line_type: str = "2nd_line",
    ) -> None:
        self.name = name
        self.template_type = template_type  # "greenfield" | "brownfield"
        self.line_type = line_type          # "2nd_line" | "3rd_line" | "4th_line"
        self.sheet_width_mm = sheet_width_mm
        self.sheet_height_mm = sheet_height_mm
        self.margin_mm = margin_mm
        self.sidebar_width_mm = sidebar_width_mm
        self.sidebar_gap_mm = sidebar_gap_mm
        self.label_width_mm = label_width_mm
        self.margin_mid_mm = margin_mid_mm
        self.config = config or {}

        # Unified chainage format (synchronized with Plan View)
        drawing_cfg = self.config.get("drawing", {})
        plan_cfg = drawing_cfg.get("plan_view", {})
        self.chainage_format: str = str(
            drawing_cfg.get("chainage_format")
            or plan_cfg.get("chainage_format")
            or self.config.get("alignment", {}).get("chainage_format")
            or self.config.get("chainage", {}).get("chainage_format")
            or self.config.get("chainage_format")
            or "CH-{km}+{m:03d}"
        )

        # Viewport Sections
        self.plan_view = plan_view
        self.longitudinal_section = longitudinal_section
        self.sidebar = sidebar
        self.notes_section = sidebar.notes_section
        self.title_block_section = sidebar.title_block_section

        # Whole sheet border
        self.outer_bbox = BoundingBox(0.0, 0.0, sheet_width_mm, sheet_height_mm)
        self.inner_bbox = BoundingBox(margin_mm, margin_mm, sheet_width_mm - margin_mm, sheet_height_mm - margin_mm)

        # Central divider line between L-Section and Plan View
        self.divider_y = longitudinal_section.bbox.y_max + (margin_mid_mm / 2.0)

        logger.info(
            "Created %s SheetTemplate [%s | %s]: Sheet=%.0fx%.0fmm, PlanH=%.1fmm, ProfH=%.1fmm, SidebarW=%.1fmm",
            self.template_type.upper(),
            self.name,
            self.line_type,
            self.sheet_width_mm,
            self.sheet_height_mm,
            self.plan_view.height_mm,
            self.longitudinal_section.height_mm,
            self.sidebar.width_mm,
        )

    # --------------------------------------------------------------------------
    # Factory Methods
    # --------------------------------------------------------------------------

    @classmethod
    def create_greenfield(
        cls,
        config: Optional[dict[str, Any]] = None,
        corridor_length_m: float = 5000.0,
        custom_bands: Optional[list[dict[str, Any]]] = None,
    ) -> SheetTemplate:
        """Create a Greenfield New Line Railway Sheet Template per RDSO standards."""
        return cls.from_config(config=config, template_type="greenfield", corridor_length_m=corridor_length_m, custom_bands=custom_bands)

    @classmethod
    def create_brownfield(
        cls,
        config: Optional[dict[str, Any]] = None,
        corridor_length_m: float = 5000.0,
        line_type: str = "2nd_line",
        custom_bands: Optional[list[dict[str, Any]]] = None,
    ) -> SheetTemplate:
        """Create a Brownfield 2nd Line (Doubling), 3rd Line, or 4th Line Sheet Template per RDSO & IRSOD standards."""
        return cls.from_config(
            config=config,
            template_type="brownfield",
            corridor_length_m=corridor_length_m,
            line_type=line_type,
            custom_bands=custom_bands,
        )

    @classmethod
    def from_config(
        cls,
        config: Optional[dict[str, Any]] = None,
        template_type: str = "greenfield",
        corridor_length_m: float = 5000.0,
        line_type: str = "2nd_line",
        custom_bands: Optional[list[dict[str, Any]]] = None,
    ) -> SheetTemplate:
        """Instantiate a SheetTemplate strictly from the provided configuration dictionary."""
        if not isinstance(config, dict) or not config:
            raise ValueError("SheetTemplate.from_config requires a non-empty configuration dictionary.")
        cfg = config

        templates_cfg = _require_cfg(cfg, "templates", "root config")
        t_cfg = _require_cfg(templates_cfg, template_type, "templates")
        drawing_cfg = cfg.get("drawing", {})
        tpl_ctx = f"templates.{template_type}"

        # 1. Base Dimensions
        sheet_w = float(_require_any_cfg([t_cfg, drawing_cfg], ["sheet_width_mm"], tpl_ctx))
        sheet_h = float(_require_any_cfg([t_cfg, drawing_cfg], ["sheet_height_mm"], tpl_ctx))
        margin = float(_require_any_cfg([t_cfg, drawing_cfg], ["margin_mm"], tpl_ctx))
        sidebar_w = float(_require_any_cfg([t_cfg, drawing_cfg], ["sidebar_width_mm"], tpl_ctx))
        sidebar_gap = float(_require_any_cfg([t_cfg, drawing_cfg], ["sidebar_gap_mm"], tpl_ctx))
        label_w = float(_require_any_cfg([t_cfg, drawing_cfg], ["label_width_mm"], tpl_ctx))
        margin_mid = float(_require_any_cfg([t_cfg, drawing_cfg], ["margin_mid_mm"], tpl_ctx))

        # 2. Scales
        plan_scale = float(_require_any_cfg([t_cfg, drawing_cfg], ["plan_scale"], tpl_ctx))
        h_scale = float(_require_any_cfg([t_cfg, drawing_cfg], ["profile_h_scale"], tpl_ctx))
        v_scale = float(_require_any_cfg([t_cfg, drawing_cfg], ["profile_v_scale"], tpl_ctx))

        # 3. Horizontal partitioning
        x_inner_min = margin
        x_inner_max = sheet_w - margin
        y_inner_min = margin
        y_inner_max = sheet_h - margin

        x_sb_min = x_inner_max - sidebar_w
        x_sb_max = x_inner_max
        sidebar_bbox = BoundingBox(x_sb_min, y_inner_min, x_sb_max, y_inner_max)

        x_main_min = x_inner_min
        x_main_max = x_sb_min - sidebar_gap

        corridor_len = float(corridor_length_m)
        if not math.isfinite(corridor_len) or corridor_len <= 0.0:
            raise ValueError(f"corridor_length_m must be finite and positive; got {corridor_length_m!r}.")
        draw_len_mm = (corridor_len / h_scale) * 1000.0
        if (x_main_min + label_w + draw_len_mm) > x_main_max:
            needed_w = x_main_min + label_w + draw_len_mm + sidebar_gap + sidebar_w + margin
            sheet_w = max(sheet_w, needed_w)
            x_inner_max = sheet_w - margin
            x_sb_min = x_inner_max - sidebar_w
            x_sb_max = x_inner_max
            x_main_max = x_sb_min - sidebar_gap
            sidebar_bbox = BoundingBox(x_sb_min, y_inner_min, x_sb_max, y_inner_max)

        x_draw_start = x_main_min + label_w
        x_draw_end = x_draw_start + draw_len_mm

        # 4. Vertical partitioning
        t_prof = _require_cfg(t_cfg, "longitudinal_section", tpl_ctx)
        prof_ctx = f"{tpl_ctx}.longitudinal_section"
        band_h = float(_require_cfg(t_prof, "band_row_height_mm", prof_ctx))

        # Normalize line type: '2nd_line' | '3rd_line' | '4th_line'
        norm_line = str(line_type).strip().lower()
        if norm_line in ("2nd_line", "doubling", "2nd", "second", "second_line"):
            line_key = "2nd_line"
        elif norm_line in ("4th_line", "fourth_line", "4th", "quadrupling", "fourth"):
            line_key = "4th_line"
        else:
            line_key = "3rd_line"

        lines_cfg = t_cfg.get("lines", {})
        line_cfg = lines_cfg.get(line_key, {}) if isinstance(lines_cfg, dict) else {}
        line_designation = str(line_cfg.get("designation", line_key.replace("_", " ").title()))

        # Parse data bands from custom_bands or config
        raw_bands = custom_bands if (custom_bands and isinstance(custom_bands, list) and len(custom_bands) > 0) else t_prof.get("bands")
        if not raw_bands or not isinstance(raw_bands, list):
            raise ValueError(f"Missing required non-empty 'bands' list in {prof_ctx} (or custom_bands).")

        band_defs: list[LSectionBandDef] = []
        for idx, b_dict in enumerate(raw_bands):
            if not isinstance(b_dict, dict):
                raise ValueError(f"Band entry at index {idx} in {prof_ctx}.bands must be a mapping.")
            if not b_dict.get("enabled", True):
                continue
            b_ctx = f"{prof_ctx}.bands[{idx}]"
            b_title = str(_require_cfg(b_dict, "title", b_ctx)).replace("{line_designation}", line_designation)
            raw_lines = b_dict.get("lines", [b_title])
            if isinstance(raw_lines, str):
                raw_lines = [raw_lines]
            fmt_lines = [str(ln).replace("{line_designation}", line_designation) for ln in raw_lines]
            band_defs.append(
                LSectionBandDef(
                    title=b_title,
                    lines=fmt_lines,
                    height_mm=float(b_dict.get("height_mm", band_h)),
                    header_color=int(_require_cfg(b_dict, "header_color", b_ctx)),
                    value_color=int(_require_cfg(b_dict, "value_color", b_ctx)),
                    field_name=str(_require_cfg(b_dict, "field_name", b_ctx)),
                    format_type=str(b_dict.get("format_type", "elevation")),
                )
            )

        total_bands_h = sum(b.height_mm for b in band_defs)
        num_bands = len(band_defs)

        # Profile grid
        prof_grid_h = float(_require_cfg(t_prof, "profile_area_height_mm", prof_ctx))
        total_lsection_h = total_bands_h + prof_grid_h

        # L-Section Bounding Boxes
        lsection_bbox = BoundingBox(x_main_min, y_inner_min, x_main_max, y_inner_min + total_lsection_h)
        bands_bbox = BoundingBox(x_draw_start, y_inner_min, x_draw_end, y_inner_min + total_bands_h)
        label_col_bbox = BoundingBox(x_main_min, y_inner_min, x_draw_start, y_inner_min + total_bands_h)
        prof_grid_bbox = BoundingBox(x_draw_start, y_inner_min + total_bands_h, x_draw_end, y_inner_min + total_lsection_h)

        # Plan View Bounding Boxes
        y_plan_min = lsection_bbox.y_max + margin_mid
        y_plan_max = y_inner_max
        plan_bbox = BoundingBox(x_main_min, y_plan_min, x_main_max, y_plan_max)
        plan_header_h = float(t_cfg.get("plan_header_height_mm", drawing_cfg.get("plan_header_height_mm", 25.0)))
        plan_header_bbox = BoundingBox(x_main_min, y_plan_max - plan_header_h, x_main_max, y_plan_max)
        plan_corridor_bbox = BoundingBox(x_main_min, y_plan_min, x_main_max, y_plan_max - plan_header_h)

        # 5. Build Sub-Sections
        t_plan = _require_cfg(t_cfg, "plan_view", tpl_ctx)
        plan_ctx = f"{tpl_ctx}.plan_view"
        tick_interval = float(_require_cfg(t_plan, "chainage_tick_interval_m", plan_ctx))
        proposed_color = int(_require_any_cfg([t_plan], ["proposed_track_color", "track_color"], plan_ctx))
        proposed_lw = int(_require_any_cfg([t_plan], ["proposed_track_lineweight", "track_lineweight"], plan_ctx))
        if template_type == "greenfield":
            existing_color = int(t_plan.get("existing_track_color", proposed_color))
            existing_lw = int(t_plan.get("existing_track_lineweight", proposed_lw))
        else:
            existing_color = int(_require_cfg(t_plan, "existing_track_color", plan_ctx))
            existing_lw = int(_require_cfg(t_plan, "existing_track_lineweight", plan_ctx))

        plan_view_section = PlanViewSection(
            bbox=plan_bbox,
            header_bbox=plan_header_bbox,
            corridor_bbox=plan_corridor_bbox,
            scale=plan_scale,
            track_mode="single_proposed" if template_type == "greenfield" else "existing_and_proposed",
            existing_track_color=existing_color,
            existing_track_lineweight=existing_lw,
            proposed_track_color=proposed_color,
            proposed_track_lineweight=proposed_lw,
            chainage_tick_interval_m=tick_interval,
            major_tick_interval_m=float(t_plan.get("major_tick_interval_m", tick_interval)),
            dimension_interval_m=float(t_plan.get("dimension_interval_m", tick_interval)),
            show_topo=bool(_require_cfg(t_plan, "show_topo", plan_ctx)),
            show_north_arrow=bool(_require_cfg(t_plan, "show_north_arrow", plan_ctx)),
        )

        lsection_section = LongitudinalSection(
            bbox=lsection_bbox,
            profile_bbox=prof_grid_bbox,
            databands_bbox=bands_bbox,
            label_column_bbox=label_col_bbox,
            draw_len_mm=draw_len_mm,
            h_scale=h_scale,
            v_scale=v_scale,
            num_bands=num_bands,
            band_row_height_mm=band_h,
            datum_round_step_m=float(_require_cfg(t_prof, "datum_round_step_m", prof_ctx)),
            datum_clearance_m=float(_require_cfg(t_prof, "datum_clearance_m", prof_ctx)),
            bands_config=band_defs,
            station_tick_interval_m=float(_require_cfg(t_prof, "station_tick_interval_m", prof_ctx)),
            station_grid_interval_m=float(_require_cfg(t_prof, "station_grid_interval_m", prof_ctx)),
            km_interval_m=float(_require_cfg(t_prof, "km_interval_m", prof_ctx)),
            label_text_height_mm=float(_require_cfg(t_prof, "label_text_height_mm", prof_ctx)),
            station_text_height_mm=float(_require_cfg(t_prof, "station_text_height_mm", prof_ctx)),
            km_text_height_mm=float(_require_cfg(t_prof, "km_text_height_mm", prof_ctx)),
        )

        sidebar_section = SidebarSection(
            bbox=sidebar_bbox,
            width_mm=sidebar_w,
            color=_resolve_aci_color(t_cfg.get("sidebar_color", drawing_cfg.get("sidebar_color", 7))),
            lineweight=int(t_cfg.get("sidebar_lineweight", drawing_cfg.get("sidebar_lineweight", 35))),
        )

        # Populate Notes
        t_notes = _require_cfg(t_cfg, "notes", tpl_ctx)
        notes_ctx = f"{tpl_ctx}.notes"
        sidebar_section.notes_section.title = str(_require_cfg(t_notes, "title", notes_ctx))
        sidebar_section.notes_section.title_color = int(_require_cfg(t_notes, "title_color", notes_ctx))
        raw_notes = _require_cfg(t_notes, "notes_list", notes_ctx)
        if not isinstance(raw_notes, list) or not raw_notes:
            raise ValueError(f"Config key 'notes_list' in {notes_ctx} must be a non-empty list.")
        notes_to_use = [str(n) for n in raw_notes]
        if template_type == "brownfield":
            customized_notes = []
            for n_str in notes_to_use:
                if n_str.startswith("6.") and line_cfg.get("note_6"):
                    customized_notes.append(str(line_cfg["note_6"]))
                elif n_str.startswith("8.") and line_cfg.get("note_8"):
                    customized_notes.append(str(line_cfg["note_8"]))
                else:
                    customized_notes.append(n_str)
            notes_to_use = customized_notes
        sidebar_section.notes_section.notes_list = notes_to_use

        # Populate Title Block
        t_tb = _require_cfg(t_cfg, "title_block", tpl_ctx)
        tb_ctx = f"{tpl_ctx}.title_block"
        proj_cfg = cfg.get("project", {})

        sidebar_section.title_block_section.railway_zone = str(
            _require_any_cfg([t_tb, proj_cfg], ["railway_zone", "zone"], tb_ctx)
        )
        sidebar_section.title_block_section.division = str(
            _require_any_cfg([t_tb, proj_cfg], ["division"], tb_ctx)
        )
        sidebar_section.title_block_section.project_title = str(
            _require_any_cfg([line_cfg, t_tb, proj_cfg], ["project_title", "project_name", "name"], tb_ctx)
        )
        sidebar_section.title_block_section.plan_title = str(_require_cfg(t_tb, "plan_title", tb_ctx))
        sidebar_section.title_block_section.drawing_prefix = str(
            _require_any_cfg([t_tb], ["drawing_prefix", "cao_drawing_prefix"], tb_ctx)
        )
        sidebar_section.title_block_section.drawing_year = str(
            t_tb.get("drawing_year", proj_cfg.get("drawing_year", ""))
        )
        sidebar_section.title_block_section.consultant_name = str(
            _require_cfg(t_tb, "consultant_name", tb_ctx)
        )
        raw_officers = t_tb.get("officers")
        if raw_officers and isinstance(raw_officers, list):
            sidebar_section.title_block_section.officers = [str(o) for o in raw_officers]

        name = str(_require_cfg(t_cfg, "name", tpl_ctx))

        return cls(
            name=name,
            template_type=template_type,
            line_type=line_key,
            sheet_width_mm=sheet_w,
            sheet_height_mm=sheet_h,
            margin_mm=margin,
            sidebar_width_mm=sidebar_w,
            sidebar_gap_mm=sidebar_gap,
            label_width_mm=label_w,
            margin_mid_mm=margin_mid,
            plan_view=plan_view_section,
            longitudinal_section=lsection_section,
            sidebar=sidebar_section,
            config=cfg,
        )

    # --------------------------------------------------------------------------
    # CAD DXF Drafting Engine Methods
    # --------------------------------------------------------------------------

    def _ensure_font_styles(self, msp: Any, doc: Optional[ezdxf.document.Drawing] = None) -> None:
        """Ensure Standard and ESP_STYLE text styles use arial.ttf with standard width factor."""
        target_doc = doc or getattr(msp, "doc", None)
        if target_doc is None:
            return
        font_name = str(self.config.get("drawing", {}).get("font_name", "arial.ttf"))
        font_width = float(self.config.get("drawing", {}).get("font_width_factor", 1.0))
        try:
            if "Standard" in target_doc.styles:
                std = target_doc.styles.get("Standard")
                std.dxf.font = font_name
                std.dxf.width = font_width
            if "ESP_STYLE" not in target_doc.styles:
                st = target_doc.styles.add("ESP_STYLE", font=font_name)
                st.dxf.width = font_width
            else:
                st = target_doc.styles.get("ESP_STYLE")
                st.dxf.font = font_name
                st.dxf.width = font_width
        except Exception:
            pass

    def draw_sheet_frames(
        self,
        msp: Any,
        doc: Optional[ezdxf.document.Drawing] = None,
        scale_factor: float = 1.0,
        layer_border: str = "BORDER",
        layer_sheet_border: str = "SHEET_BORDER",
        layer_title_block: str = "TITLE_BLOCK",
    ) -> None:
        """Render outer border, inner margin border, section dividers, and header strips onto CAD modelspace."""
        self._ensure_font_styles(msp, doc)
        s = scale_factor
        # 1. Outer Sheet Border (ACI 7 Black/White, lw 25)
        msp.add_lwpolyline(
            self.outer_bbox.to_polyline_points(s),
            dxfattribs={"layer": layer_sheet_border, "color": 7, "lineweight": 25},
        )

        # 2. Inner Sheet Border (ACI 1 Red or ACI 7, lw 50)
        border_col = 1 if self.template_type == "greenfield" else 7
        msp.add_lwpolyline(
            self.inner_bbox.to_polyline_points(s),
            dxfattribs={"layer": layer_border, "color": border_col, "lineweight": 50},
        )

        # 3. Right Sidebar Boundary Divider Line
        sb_x0 = self.sidebar.bbox.x_min * s
        y_bot = self.inner_bbox.y_min * s
        y_top = self.inner_bbox.y_max * s
        msp.add_line(
            (sb_x0, y_bot),
            (sb_x0, y_top),
            dxfattribs={"layer": layer_sheet_border, "color": 7, "lineweight": 35},
        )

        # 4. Horizontal Divider Line between L-Section and Plan View
        div_y = self.divider_y * s
        main_x0 = self.inner_bbox.x_min * s
        main_x1 = (self.sidebar.bbox.x_min - self.sidebar_gap_mm) * s
        msp.add_line(
            (main_x0, div_y),
            (main_x1, div_y),
            dxfattribs={"layer": layer_sheet_border, "color": 7, "lineweight": 35},
        )

        # 5. Underlined Section Banner for L-Section
        banner_x = (main_x0 + main_x1) / 2.0
        msp.add_text(
            "L - SECTION",
            dxfattribs={"height": 4.5 * s, "layer": layer_title_block, "color": 1, "style": "ESP_STYLE"},
        ).set_placement((banner_x, div_y - 8.0 * s), align=TextEntityAlignment.MIDDLE_CENTER)

        # 6. Underlined Section Banner for Plan View
        plan_banner_y = self.plan_view.header_bbox.center_y * s
        msp.add_text(
            "PLAN",
            dxfattribs={"height": 5.0 * s, "layer": layer_title_block, "color": 1, "style": "ESP_STYLE"},
        ).set_placement((banner_x, plan_banner_y), align=TextEntityAlignment.MIDDLE_CENTER)

        # Underline under PLAN banner
        msp.add_line(
            (banner_x - 18.0 * s, plan_banner_y - 3.5 * s),
            (banner_x + 18.0 * s, plan_banner_y - 3.5 * s),
            dxfattribs={"layer": layer_title_block, "color": 1, "lineweight": 25},
        )

    def draw_sidebar_content(
        self,
        msp: Any,
        sheet_idx: int = 1,
        total_sheets: int = 1,
        start_km: float = 0.0,
        end_km: float = 5.0,
        theta_rad: float = 0.0,
        scale_factor: float = 1.0,
        layer_title_block: str = "TITLE_BLOCK",
        layer_north_arrow: str = "NORTH_ARROW",
        layer_sheet_border: str = "SHEET_BORDER",
    ) -> None:
        """Render True North Compass Rose, General Notes block, and Official Title Block inside the Right Sidebar."""
        self._ensure_font_styles(msp)
        s = scale_factor
        sb = self.sidebar
        x0, y0, x1, y1 = [v * s for v in sb.bbox.as_tuple()]
        w = sb.width_mm * s

        # ── 1. North Arrow Compass Rose Compartment (Top) ───────────────────────
        nb = sb.north_arrow_bbox
        # Border around North box
        msp.add_lwpolyline(
            nb.to_polyline_points(s),
            dxfattribs={"layer": layer_sheet_border, "color": 7, "lineweight": 25},
        )

        cx, cy = nb.center_x * s, nb.center_y * s
        arrow_r = min(15.0 * s, nb.height * s * 0.35)
        # Outer circle
        msp.add_circle((cx, cy), arrow_r, dxfattribs={"layer": layer_north_arrow, "color": 7, "lineweight": 18})

        # Direction vector rotated by theta
        dx = math.sin(theta_rad) * arrow_r
        dy = math.cos(theta_rad) * arrow_r
        # North arrow head
        msp.add_line((cx, cy - dy * 0.4), (cx + dx, cy + dy), dxfattribs={"layer": layer_north_arrow, "color": 7, "lineweight": 25})
        perp_x, perp_y = -dy * 0.35, dx * 0.35
        msp.add_lwpolyline(
            [(cx + dx, cy + dy), (cx + dx * 0.6 + perp_x, cy + dy * 0.6 + perp_y), (cx, cy), (cx + dx * 0.6 - perp_x, cy + dy * 0.6 - perp_y), (cx + dx, cy + dy)],
            dxfattribs={"layer": layer_north_arrow, "color": 1, "lineweight": 25},
        )
        # "N" label
        msp.add_text(
            "N",
            dxfattribs={"height": 3.8 * s, "layer": layer_north_arrow, "color": 1, "style": "ESP_STYLE"},
        ).set_placement((cx + dx * 1.3, cy + dy * 1.3), align=TextEntityAlignment.MIDDLE_CENTER)

        # ── 2. General Notes Compartment ────────────────────────────────────────
        notes = sb.notes_section
        notes_box = notes.bbox
        # Compartment boundary
        msp.add_lwpolyline(
            notes_box.to_polyline_points(s),
            dxfattribs={"layer": layer_sheet_border, "color": 7, "lineweight": 25},
        )

        # Notes Header
        msp.add_text(
            notes.title,
            dxfattribs={"height": 3.2 * s, "layer": layer_title_block, "color": notes.title_color, "style": "ESP_STYLE"},
        ).set_placement(((notes_box.x_min + 5.0) * s, (notes_box.y_max - 8.0) * s), align=TextEntityAlignment.MIDDLE_LEFT)

        # Notes list lines
        curr_y = (notes_box.y_max - 15.0) * s
        line_spacing = notes.line_spacing_mm * s
        text_h = notes.text_height_mm * s
        for note in notes.notes_list:
            if curr_y < (notes_box.y_min + 5.0) * s:
                break
            # Wrap long note text into lines of ~50 characters
            words = note.split(" ")
            wrapped_lines: list[str] = []
            curr_line = ""
            for word in words:
                if len(curr_line) + len(word) + 1 <= 52:
                    curr_line = f"{curr_line} {word}".strip()
                else:
                    wrapped_lines.append(curr_line)
                    curr_line = word
            if curr_line:
                wrapped_lines.append(curr_line)

            for wl in wrapped_lines:
                if curr_y < (notes_box.y_min + 4.0) * s:
                    break
                msp.add_text(
                    wl,
                    dxfattribs={"height": text_h, "layer": layer_title_block, "color": 7, "style": "ESP_STYLE"},
                ).set_placement(((notes_box.x_min + 5.0) * s, curr_y), align=TextEntityAlignment.MIDDLE_LEFT)
                curr_y -= line_spacing

        # ── 3. Official Title Block Compartment (Bottom) ────────────────────────
        tb = sb.title_block_section
        tb_box = tb.bbox
        # Outer boundary of title block
        msp.add_lwpolyline(
            tb_box.to_polyline_points(s),
            dxfattribs={"layer": layer_sheet_border, "color": 7, "lineweight": 35},
        )

        # Zonal Railway Banner (Top banner of Title Block)
        banner_h = 16.0 * s
        banner_y0 = tb_box.y_max * s - banner_h
        msp.add_line((tb_box.x_min * s, banner_y0), (tb_box.x_max * s, banner_y0), dxfattribs={"layer": layer_title_block, "color": 1, "lineweight": 25})
        msp.add_text(
            f"INDIAN RAILWAYS — {tb.railway_zone.upper()}",
            dxfattribs={"height": 4.0 * s, "layer": layer_title_block, "color": 1, "style": "ESP_STYLE"},
        ).set_placement((tb_box.center_x * s, (banner_y0 + tb_box.y_max * s) / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)

        # Division and Project Title
        msp.add_text(
            f"DIVISION: {tb.division.upper()}",
            dxfattribs={"height": 2.8 * s, "layer": layer_title_block, "color": 7, "style": "ESP_STYLE"},
        ).set_placement(((tb_box.x_min + 5.0) * s, banner_y0 - 6.0 * s), align=TextEntityAlignment.MIDDLE_LEFT)

        msp.add_text(
            f"PROJECT: {tb.project_title.upper()}",
            dxfattribs={"height": 3.0 * s, "layer": layer_title_block, "color": 1, "style": "ESP_STYLE"},
        ).set_placement(((tb_box.x_min + 5.0) * s, banner_y0 - 13.0 * s), align=TextEntityAlignment.MIDDLE_LEFT)

        msp.add_text(
            f"TITLE: {tb.plan_title.upper()}",
            dxfattribs={"height": 2.8 * s, "layer": layer_title_block, "color": 7, "style": "ESP_STYLE"},
        ).set_placement(((tb_box.x_min + 5.0) * s, banner_y0 - 20.0 * s), align=TextEntityAlignment.MIDDLE_LEFT)

        msp.add_text(
            f"SHEET NO: {sheet_idx} OF {total_sheets}  |  SPAN: KM {start_km:.2f} TO KM {end_km:.2f}",
            dxfattribs={"height": 2.8 * s, "layer": layer_title_block, "color": 1, "style": "ESP_STYLE"},
        ).set_placement(((tb_box.x_min + 5.0) * s, banner_y0 - 27.0 * s), align=TextEntityAlignment.MIDDLE_LEFT)

        msp.add_text(
            f"SCALES: {tb.scales_text}",
            dxfattribs={"height": 2.4 * s, "layer": layer_title_block, "color": 7, "style": "ESP_STYLE"},
        ).set_placement(((tb_box.x_min + 5.0) * s, banner_y0 - 34.0 * s), align=TextEntityAlignment.MIDDLE_LEFT)

        # Consultant & CAO Info
        msp.add_line(((tb_box.x_min) * s, banner_y0 - 39.0 * s), (tb_box.x_max * s, banner_y0 - 39.0 * s), dxfattribs={"layer": layer_title_block, "color": 7, "lineweight": 18})
        msp.add_text(
            f"DRG NO: {tb.drawing_prefix} / SHEET-{sheet_idx}",
            dxfattribs={"height": 2.8 * s, "layer": layer_title_block, "color": 1, "style": "ESP_STYLE"},
        ).set_placement(((tb_box.x_min + 5.0) * s, banner_y0 - 45.0 * s), align=TextEntityAlignment.MIDDLE_LEFT)

        msp.add_text(
            f"CONSULTANT: {tb.consultant_name.upper()}",
            dxfattribs={"height": 2.4 * s, "layer": layer_title_block, "color": 7, "style": "ESP_STYLE"},
        ).set_placement(((tb_box.x_min + 5.0) * s, banner_y0 - 52.0 * s), align=TextEntityAlignment.MIDDLE_LEFT)

    def draw_lsection_bands(
        self,
        msp: Any,
        start_chainage_m: float,
        end_chainage_m: float,
        scale_factor: float = 1.0,
        profile_df: Optional[Any] = None,
        layer_bands: str = "ESP_PROF_BANDS",
        layer_text_prop: str = "ESP_PROF_TEXT_PROP",
        layer_text_ex: str = "ESP_PROF_TEXT_EX",
        layer_grid: str = "ESP_PROF_GRID",
        chainage_format: Optional[str] = None,
    ) -> None:
        """Render complete Longitudinal Section Data Bands matrix, headers, station ticks, and km stones onto CAD modelspace.

        Args:
            msp: ezdxf modelspace object.
            start_chainage_m: Corridor start chainage in meters (e.g. 0.0 or 110000.0).
            end_chainage_m: Corridor end chainage in meters (e.g. 5000.0 or 115000.0).
            scale_factor: CAD model scaling multiplier (typically plan_scale / 1000.0).
            profile_df: Optional DataFrame containing station elevations ('chainage_m', 'prl_m', 'erl_m', etc.).
            layer_bands: DXF layer for band borders and divider lines.
            layer_text_prop: DXF layer for proposed track text annotations.
            layer_text_ex: DXF layer for existing track text annotations.
            layer_grid: DXF layer for vertical station grid lines.
            chainage_format: Optional template string for station chainages (e.g. 'CH-{km}+{m:03d}').
        """
        self._ensure_font_styles(msp)
        s = scale_factor
        active_ch_fmt = chainage_format or self.chainage_format
        lsec = self.longitudinal_section
        label_x0 = lsec.label_column_bbox.x_min * s
        label_x1 = lsec.label_column_bbox.x_max * s
        data_x0 = label_x1
        data_x1 = lsec.databands_bbox.x_max * s
        y_datum = lsec.databands_bbox.y_max * s
        y_bottom = lsec.databands_bbox.y_min * s
        corridor_len = max(1.0, float(end_chainage_m - start_chainage_m))

        # 1. Compute band vertical row bounds from Datum line downwards
        curr_y = y_datum
        band_bounds: list[tuple[LSectionBandDef, float, float]] = []
        for b_def in lsec.bands_config:
            top_y = curr_y
            bot_y = curr_y - (b_def.height_mm * s)
            curr_y = bot_y
            band_bounds.append((b_def, top_y, bot_y))

        # 2. Draw Horizontal Divider Lines
        # Top datum line connecting to profile area
        msp.add_line(
            (label_x0, y_datum),
            (data_x1, y_datum),
            dxfattribs={"layer": layer_bands, "color": 7, "lineweight": 35},
        )
        # Dividers between band rows
        for _, _, bot_y in band_bounds:
            msp.add_line(
                (label_x0, bot_y),
                (data_x1, bot_y),
                dxfattribs={"layer": layer_bands, "color": 7, "lineweight": 25},
            )

        # 3. Draw Vertical Boundary Lines
        # Left boundary of label column
        msp.add_line(
            (label_x0, y_bottom),
            (label_x0, y_datum),
            dxfattribs={"layer": layer_bands, "color": 7, "lineweight": 35},
        )
        # Divider between label column and data area
        msp.add_line(
            (label_x1, y_bottom),
            (label_x1, y_datum),
            dxfattribs={"layer": layer_bands, "color": 7, "lineweight": 35},
        )
        # Right boundary of data area
        msp.add_line(
            (data_x1, y_bottom),
            (data_x1, y_datum),
            dxfattribs={"layer": layer_bands, "color": 7, "lineweight": 35},
        )

        # 4. Render Centered Band Header Titles in Label Column
        label_mid_x = (label_x0 + label_x1) / 2.0
        base_label_h = lsec.label_text_height_mm * s

        for b_def, top_y, bot_y in band_bounds:
            row_h = top_y - bot_y
            mid_y = (top_y + bot_y) / 2.0
            lines = b_def.lines if b_def.lines else [b_def.title]
            n_lines = len(lines)

            # Fit multi-line text vertically within band height
            max_allowed_h = row_h * 0.82
            needed_h = base_label_h * (1.0 + (n_lines - 1) * 1.3)
            if needed_h > max_allowed_h and n_lines > 0:
                eff_text_h = max(1.2 * s, max_allowed_h / (1.0 + (n_lines - 1) * 1.3))
            else:
                eff_text_h = base_label_h

            pitch = eff_text_h * 1.3
            first_y = mid_y + ((n_lines - 1) * pitch / 2.0)
            txt_lay = layer_text_prop if b_def.header_color == 1 else layer_text_ex

            for i, line_text in enumerate(lines):
                y_pos = first_y - (i * pitch)
                msp.add_text(
                    line_text,
                    dxfattribs={
                        "height": eff_text_h,
                        "layer": txt_lay,
                        "color": b_def.header_color,
                        "style": "ESP_STYLE",
                    },
                ).set_placement((label_mid_x, y_pos), align=TextEntityAlignment.MIDDLE_CENTER)

        # 5. Vertical Station Grid Lines and Station Annotations across Data Area
        tick_int = lsec.station_tick_interval_m
        grid_int = lsec.station_grid_interval_m
        km_int = lsec.km_interval_m
        st_text_h = lsec.station_text_height_mm * s
        km_text_h = lsec.km_text_height_mm * s

        def _is_chainage_band_def(b_def_obj: Any) -> bool:
            b_str = (
                str(getattr(b_def_obj, "id", "") or "") + " "
                + str(getattr(b_def_obj, "field_name", "") or "") + " "
                + str(getattr(b_def_obj, "title", "") or "") + " "
                + str(getattr(b_def_obj, "source_column", "") or "")
            ).lower()
            b_words = set(b_str.replace("(", " ").replace(")", " ").replace(".", " ").replace("-", " ").replace("+", " ").split())
            return (
                any(k in b_str for k in ("chainage", "ch_m", "stn_m", "ch-", "ch+"))
                or getattr(b_def_obj, "id", "").lower() in ("ch", "chainage", "chainage_m", "chainage_raw", "km")
                or (getattr(b_def_obj, "field_name", "") or "").lower() in ("ch", "chainage", "chainage_m", "chainage_raw", "km", "station", "stn")
                or (getattr(b_def_obj, "format_type", "") or "").lower().startswith("chainage")
                or bool(b_words & {"ch", "km", "station", "stn"})
            )

        ch_band_tuple = next(
            (b for b in band_bounds if _is_chainage_band_def(b[0])),
            None,
        )
        km_band_tuple = next((b for b in band_bounds if "km" in (b[0].field_name or "").lower()), None)

        def ch_to_x(ch_m: float) -> float:
            """Convert along-track chainage in meters to model X coordinate."""
            return data_x0 + (ch_m - start_chainage_m) * ((data_x1 - data_x0) / corridor_len)

        # Offset rotated station text slightly to the right of the vertical station line
        # so numbers sit cleanly inside the station column and never intersect the grid or border lines.
        # At 1:5000 scale (s=5.0), st_text_h = 10.0 (extent: [-5, +5]).
        # An offset of 10.0 (or 2.0 * s) places the text center at x + 10.0,
        # giving a clean 5.0 unit (1.0mm paper) clearance from the vertical station line at x.
        st_x_offset = 2.0 * s

        def safe_station_text_x(x_base: float) -> float:
            """Compute shifted station text X coordinate ensuring it never crosses sheet boundaries."""
            if x_base > (data_x1 - 15.0):
                return max(data_x0 + st_x_offset, data_x1 - st_x_offset)
            return max(data_x0 + st_x_offset, x_base + st_x_offset)

        has_sub_df_stations = False
        if profile_df is not None and hasattr(profile_df, "iterrows") and "chainage_m" in profile_df.columns:
            mask = (profile_df["chainage_m"] >= (start_chainage_m - 0.01)) & (
                profile_df["chainage_m"] <= (end_chainage_m + 0.01)
            )
            has_sub_df_stations = len(profile_df[mask]) > 0

        # 5A. 100m Station Grid Lines & Chainage Labels
        st_ch = math.ceil(start_chainage_m / grid_int) * grid_int
        while st_ch <= end_chainage_m + 0.01:
            x_st = ch_to_x(st_ch)
            is_km = (int(round(st_ch)) % int(km_int) == 0)

            line_col = 7 if is_km else 8
            line_lw = 35 if is_km else 13
            # Only draw grid line if NOT directly on boundary lines (data_x0 or data_x1)
            if abs(x_st - data_x0) > 1.0 and abs(x_st - data_x1) > 1.0:
                msp.add_line(
                    (x_st, y_bottom),
                    (x_st, y_datum),
                    dxfattribs={"layer": layer_grid, "color": line_col, "lineweight": line_lw},
                )

            # If no survey stations from profile_df are present, render fallback 100m chainage labels
            if ch_band_tuple and not has_sub_df_stations:
                b_def, b_top, b_bot = ch_band_tuple
                b_mid = (b_top + b_bot) / 2.0
                fmt_type = (getattr(b_def, "format_type", "") or "").lower()
                if fmt_type in ("integer", "int", "integer_m"):
                    km_val_str = str(int(round(st_ch)))
                elif fmt_type == "chainage_2dp":
                    km_val_str = f"{st_ch / 1000.0:.2f}"
                else:
                    km_val_str = format_chainage(st_ch, format_pattern=active_ch_fmt)
                max_text_h = (row_h * 0.70) / (max(1, len(km_val_str)) * 0.75)
                eff_ch_h = min(st_text_h, max(1.0 * s, max_text_h))
                txt_lay = layer_text_prop if b_def.value_color == 1 else layer_text_ex
                msp.add_text(
                    km_val_str,
                    dxfattribs={
                        "height": eff_ch_h,
                        "layer": txt_lay,
                        "color": b_def.value_color,
                        "style": "ESP_STYLE",
                        "rotation": 90.0,
                    },
                ).set_placement((safe_station_text_x(x_st), b_mid), align=TextEntityAlignment.MIDDLE_CENTER)

            st_ch += grid_int

        # 5B. 20m Minor Station Ticks in Chainage band
        if ch_band_tuple:
            b_def, b_top, b_bot = ch_band_tuple
            tick_len = 2.0 * s
            t_ch = math.ceil(start_chainage_m / tick_int) * tick_int
            while t_ch <= end_chainage_m + 0.01:
                if int(round(t_ch)) % int(grid_int) != 0:
                    x_t = ch_to_x(t_ch)
                    msp.add_line(
                        (x_t, b_top),
                        (x_t, b_top - tick_len),
                        dxfattribs={"layer": layer_bands, "color": 7, "lineweight": 18},
                    )
                    msp.add_line(
                        (x_t, b_bot),
                        (x_t, b_bot + tick_len),
                        dxfattribs={"layer": layer_bands, "color": 7, "lineweight": 18},
                    )
                t_ch += tick_int

        # 5C. Km Stone Markings in KILOMETRE band
        if km_band_tuple:
            b_def, b_top, b_bot = km_band_tuple
            b_mid = (b_top + b_bot) / 2.0
            txt_lay = layer_text_prop if b_def.value_color == 1 else layer_text_ex
            min_km_idx = int(math.floor(start_chainage_m / km_int))
            max_km_idx = int(math.ceil(end_chainage_m / km_int))
            for k_idx in range(min_km_idx, max_km_idx + 1):
                km_st = max(start_chainage_m, k_idx * km_int)
                km_en = min(end_chainage_m, (k_idx + 1) * km_int)
                if km_en > km_st:
                    x_km_mid = (ch_to_x(km_st) + ch_to_x(km_en)) / 2.0
                    km_label = f"KM {k_idx + 1}" if k_idx >= 0 else f"KM {k_idx}"
                    msp.add_text(
                        km_label,
                        dxfattribs={
                            "height": km_text_h,
                            "layer": txt_lay,
                            "color": b_def.value_color,
                            "style": "ESP_STYLE",
                        },
                    ).set_placement((x_km_mid, b_mid), align=TextEntityAlignment.MIDDLE_CENTER)

        # 5D. Populate Elevation / Earthwork Values from profile_df (if available)
        if profile_df is not None and hasattr(profile_df, "iterrows"):
            import pandas as pd
            import re

            if "chainage_m" in profile_df.columns:
                mask = (profile_df["chainage_m"] >= (start_chainage_m - 0.01)) & (
                    profile_df["chainage_m"] <= (end_chainage_m + 0.01)
                )
                sub_df = profile_df[mask]

                def resolve_col_for_band(row_cols: list[str], b: LSectionBandDef) -> Optional[str]:
                    fname = b.field_name or ""
                    title = b.title or ""
                    # 1. Exact match
                    if fname in row_cols:
                        return fname
                    if title in row_cols:
                        return title
                    # 2. Normalized alphanumeric match
                    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
                    target_f = norm(fname)
                    target_t = norm(title)
                    for col in row_cols:
                        c_norm = norm(col)
                        if c_norm and (c_norm == target_f or c_norm == target_t):
                            return col
                    # 3. Canonical semantic aliases
                    comb = f"{fname.lower()} {title.lower()}"
                    if any(k in comb for k in ("diff", "difference")) and "rail" in comb:
                        for c in ("diff_prl_erl_m", "diff", "difference"):
                            if c in row_cols:
                                return c
                    if "cut" in comb or "fill" in comb or "bank" in comb:
                        for c in ("cut_fill_m", "cut_fill", "cut", "fill"):
                            if c in row_cols:
                                return c
                    if "track" in comb and ("center" in comb or "centre" in comb):
                        for c in ("track_center_m", "track_center", "cc_distance"):
                            if c in row_cols:
                                return c
                    if ("prop" in comb or "new" in comb) and ("rail" in comb or "prl" in comb):
                        for c in ("prl_m", "proposed_rail_level", "prop_rail_level"):
                            if c in row_cols:
                                return c
                    if ("ex" in comb or "existing" in comb or "ext" in comb) and ("rail" in comb or "erl" in comb):
                        for c in ("erl_m", "existing_rail_level", "exg_rail_level"):
                            if c in row_cols:
                                return c
                    if ("prop" in comb or "new" in comb) and ("form" in comb or "pfl" in comb):
                        for c in ("pfl_m", "proposed_formation_level", "prop_formation"):
                            if c in row_cols:
                                return c
                    if ("ex" in comb or "existing" in comb or "ext" in comb) and ("form" in comb or "efl" in comb):
                        for c in ("efl_m", "existing_formation_level"):
                            if c in row_cols:
                                return c
                    if "ground" in comb or "egl" in comb or "gl" in comb:
                        for c in ("egl_m", "ground_level", "gl"):
                            if c in row_cols:
                                return c
                    if "grad" in comb or "grade" in comb:
                        if "ex" in comb:
                            for c in ("existing_gradient", "exg_gradient"):
                                if c in row_cols:
                                    return c
                        else:
                            for c in ("proposed_gradient", "prop_gradient"):
                                if c in row_cols:
                                    return c
                    if "align" in comb or "curve" in comb:
                        if "ex" in comb:
                            for c in ("ext_alignment", "existing_alignment"):
                                if c in row_cols:
                                    return c
                        else:
                            for c in ("proposed_alignment", "prop_alignment"):
                                if c in row_cols:
                                    return c
                    return None

                row_cols = list(sub_df.columns)
                band_col_map = {idx: resolve_col_for_band(row_cols, b_def) for idx, (b_def, _, _) in enumerate(band_bounds)}

                for _, row in sub_df.iterrows():
                    ch_val = float(row["chainage_m"])
                    x_val = ch_to_x(ch_val)
                    x_text = safe_station_text_x(x_val)

                    for idx, (b_def, b_top, b_bot) in enumerate(band_bounds):
                        b_mid = (b_top + b_bot) / 2.0
                        row_h = b_top - b_bot

                        # Check if this band is the chainage band
                        is_chainage_band = (
                            (ch_band_tuple and b_def == ch_band_tuple[0])
                            or _is_chainage_band_def(b_def)
                        )
                        if is_chainage_band:
                            col = band_col_map.get(idx)
                            raw_val = row[col] if (col and col in row and not pd.isna(row[col])) else ch_val
                            try:
                                f_val = float(raw_val)
                                stn_ch = ch_val if (abs(f_val) < 500.0 and abs(ch_val) > 500.0) else f_val
                            except (ValueError, TypeError):
                                stn_ch = ch_val

                            fmt_type = (getattr(b_def, "format_type", "") or "").lower()
                            if fmt_type in ("integer", "int", "integer_m"):
                                km_val_str = str(int(round(stn_ch)))
                            elif fmt_type == "chainage_2dp":
                                km_val_str = f"{stn_ch / 1000.0:.2f}"
                            elif fmt_type in ("text", "raw", "string") and raw_val is not None and str(raw_val).strip() and str(raw_val).lower() != "nan":
                                km_val_str = str(raw_val).strip()
                            else:
                                km_val_str = format_chainage(stn_ch, format_pattern=active_ch_fmt)
                            max_text_h = (row_h * 0.70) / (max(1, len(km_val_str)) * 0.75)
                            eff_ch_h = min(st_text_h, max(1.0 * s, max_text_h))
                            txt_lay = layer_text_prop if b_def.value_color == 1 else layer_text_ex
                            msp.add_text(
                                km_val_str,
                                dxfattribs={
                                    "height": eff_ch_h,
                                    "layer": txt_lay,
                                    "color": b_def.value_color,
                                    "style": "ESP_STYLE",
                                    "rotation": 90.0,
                                },
                            ).set_placement((x_text, b_mid), align=TextEntityAlignment.MIDDLE_CENTER)
                            continue

                        col = band_col_map.get(idx)
                        if not col or col not in row or pd.isna(row[col]):
                            continue
                        raw_val = row[col]
                        val_str = str(raw_val).strip()
                        if not val_str or val_str.lower() == "nan":
                            continue

                        # Format numeric or string value
                        fmt = (b_def.format_type or "").lower()
                        try:
                            num_val = float(raw_val)
                            is_signed = (
                                fmt == "signed"
                                or "cut" in b_def.field_name.lower()
                                or "cut" in b_def.title.lower()
                                or "diff" in b_def.field_name.lower()
                                or "diff" in b_def.title.lower()
                            )
                            if fmt in ("integer", "int", "integer_m"):
                                val_text = str(int(round(num_val)))
                            elif is_signed:
                                val_text = f"{num_val:+.3f}"
                            elif "track_center" in str(col).lower() or "track_center" in b_def.field_name.lower():
                                val_text = f"{num_val:.2f}"
                            elif fmt in ("text", "raw"):
                                val_text = val_str
                            else:
                                val_text = f"{num_val:.3f}"
                        except (ValueError, TypeError):
                            val_text = val_str

                        # Auto-scale text height so string length stays comfortably within row_h
                        # (occupying at most 70% of row height, giving at least 15% clear margin on top and bottom)
                        max_text_h = (row_h * 0.70) / (max(1, len(val_text)) * 0.75)
                        eff_val_h = min(st_text_h, max(1.0 * s, max_text_h))

                        txt_lay = layer_text_prop if b_def.value_color == 1 else layer_text_ex
                        msp.add_text(
                            val_text,
                            dxfattribs={
                                "height": eff_val_h,
                                "layer": txt_lay,
                                "color": b_def.value_color,
                                "style": "ESP_STYLE",
                                "rotation": 90.0,
                            },
                        ).set_placement((x_text, b_mid), align=TextEntityAlignment.MIDDLE_CENTER)


