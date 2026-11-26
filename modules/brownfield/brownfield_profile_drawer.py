"""Brownfield Longitudinal Section (L-Section) Profile Drawer.

Module: modules/brownfield/brownfield_profile_drawer.py
Purpose: Generates official Indian Railways L-Section DXF drawings for
         Brownfield railway corridors (Doubling, 3rd/4th line, Gauge Conversion).

Engineering Standards (RDSO / Indian Railways BG):
    - Profile Grid (H=1:5000, V=1:500 standard):
        - ESP_PROF_EGL: Natural Ground Level (Green, ACI 3).
        - ESP_PROF_ERL: Existing Rail Level - Top of Rail (White, ACI 7).
        - ESP_PROF_PFL: Proposed Formation Level (Red, ACI 1, lineweight 35).
        - ESP_PROF_PRL: Proposed Rail Level (Red Dashed, ACI 1, lineweight 25).
    - 8 Standard Brownfield Doubling Data Bands (Top to Bottom):
        - Band 1: Diff.Between Rail levels (PRL - ERL; Track Lift / Lowering; Red)
        - Band 2: Height Of (+) Bank / (-) Cutting (PFL - EGL; Bank fill (+) or Cut (-); Red)
        - Band 3: Rail Level of Existing Track (M) (Surveyed Existing TOR; White)
        - Band 4: Formation Level of Existing Track (M) (Surveyed EFL or ERL - 0.762m; White)
        - Band 5: proposed Rail Level Doubling Line (PFL + 0.762m; Red)
        - Band 6: Proposed Formation Level of Doubling Line (PFL subgrade; Red)
        - Band 7: Ground Level(M) (Natural ground surface elevation EGL; White)
        - Band 8: Chainage (Km) (Full station chainage in meters, e.g. 104975.00; White)
"""

from __future__ import annotations

import math
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
import numpy as np
import pandas as pd

# Ensure esp_generator root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger

logger = get_logger("BrownfieldProfileDrawer")


def wrap_band_header_text(
    text: str,
    max_chars: int = 30,
) -> list[str]:
    """Intelligently wrap a band header title to fit within CAD header box width.

    Preserves explicit newlines, wraps long lines on word/hyphen boundaries,
    and returns a clean list of text lines so that text never overflows the cell boundary.
    """
    if not text:
        return []
    clean = str(text).strip()
    if not clean:
        return []

    lines: list[str] = []
    # If text already contains explicit newlines, process each segment
    raw_segments = clean.split("\n") if "\n" in clean else [clean]
    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) <= max_chars:
            lines.append(seg)
        else:
            wrapped = textwrap.wrap(
                seg,
                width=max_chars,
                break_long_words=False,
                break_on_hyphens=True,
            )
            if not wrapped:
                wrapped = textwrap.wrap(
                    seg,
                    width=max_chars,
                    break_long_words=True,
                    break_on_hyphens=True,
                )
            lines.extend(wrapped)
    return lines or [clean]


class BrownfieldProfileDrawerError(Exception):
    """Custom exception raised for brownfield profile drafting errors."""


@dataclass
class ProfileBandDef:
    """Definition of a single profile data band in the L-Section table.

    Attributes:
        title: Display title for the band header cell.
        height_mm: Band height in paper millimeters.
        header_color: ACI color index for header text (1=Red, 7=White).
        value_color: ACI color index for rotated station value text.
        field_name: Column name in profile_df containing values.
        format_type: String format selector ('signed_3dp', 'unsigned_3dp', 'chainage_2dp', etc.).
        lines: Explicit list of text lines for header cell multi-line rendering.
        omit_bottom_border: Whether to omit the horizontal bottom dividing line.
        line_colors: ACI colors per header line if multicolored.
        id: Unique identifier for band (e.g. 'rl_diff', 'chainage').
        enabled: Boolean flag indicating if this band should be drawn.
        source_column: Original column name in input Excel/DataFrame.
        precision: Decimal precision for numeric values.
        signed: Whether to force '+' sign on positive numbers.
        wrap_text: Whether to automatically wrap long titles to fit cell width.
    """

    title: str
    height_mm: float
    header_color: int
    value_color: int
    field_name: str
    format_type: str = "unsigned_3dp"
    lines: list[str] = field(default_factory=list)
    omit_bottom_border: bool = False
    line_colors: list[int] = field(default_factory=list)
    id: str = ""
    enabled: bool = True
    source_column: Optional[str] = None
    precision: Optional[int] = None
    signed: bool = False
    wrap_text: bool = True


@dataclass
class BrownfieldProfileResult:
    """Container for Brownfield L-Section profile drawing outputs.

    Attributes:
        dxf_path: Path to the generated AutoCAD DXF file.
        profile_df: Comprehensive profile DataFrame at all stations.
        datum_m: Computed datum line elevation.
        metadata: Audit dictionary with statistics on levels, lift, and cut/fill.
    """

    dxf_path: Path
    profile_df: pd.DataFrame
    datum_m: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BrownfieldProfileDrawer:
    """Generates official Indian Railways 8-band & 10-band L-Section DXF drawings for brownfield lines."""

    def __init__(
        self,
        config: Optional[dict[str, Any] | str | Path] = None,
        lsection_bands: Optional[list[dict[str, Any] | ProfileBandDef]] = None,
    ) -> None:
        """Initialize BrownfieldProfileDrawer.

        Args:
            config: Config dict or path to config_brownfield.yaml. Defaults to
                    config/config_brownfield.yaml if None.
            lsection_bands: Optional list of dynamic user-selected profile band
                           definitions or dictionaries from UI/config.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.custom_lsection_bands = lsection_bands

        # Load YAML if path or None provided
        if config is None or isinstance(config, (str, Path)):
            cfg_path = Path(config) if config else _ROOT / "config" / "config_brownfield.yaml"
            if cfg_path.exists():
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
                self.logger.info("Loaded brownfield configuration from: %s", cfg_path)
            else:
                self.config = {}
                self.logger.warning("Brownfield config file not found: %s; using defaults", cfg_path)
        else:
            self.config = config

        self.logger.info("Initializing %s", self.__class__.__name__)

        # Extract config sections
        vert_cfg = self.config.get("vertical", {})
        drawing_cfg = self.config.get("drawing", {})
        paths_cfg = self.config.get("paths", {})
        project_cfg = self.config.get("project", {})

        # Paths
        out_cfg = paths_cfg.get("lsection_outputs_dir", "outputs/brownfield/lsection")
        out_p = Path(out_cfg)
        self.outputs_dir = (_ROOT / out_p).resolve() if not out_p.is_absolute() else out_p

        # Vertical Parameters
        self.track_buildup_m: float = float(vert_cfg.get("track_buildup_m", 0.762))
        self.default_lift_m: float = float(vert_cfg.get("default_ballast_lift_m", 0.025))

        # Drawing Scales
        self.h_scale: float = float(drawing_cfg.get("profile_h_scale", 5000.0))
        self.v_scale: float = float(drawing_cfg.get("profile_v_scale", 500.0))
        self.chainage_interval_m: float = float(
            self.config.get("horizontal", {}).get(
                "chainage_interval_m",
                self.config.get("alignment", {}).get("chainage_interval", 25.0),
            )
        )

        # Template & Labeling
        self.template_name: str = str(drawing_cfg.get("profile_template", "brownfield_doubling"))
        self.track_label: str = str(drawing_cfg.get("profile_track_label", "Doubling Line"))
        self.header_width_mm: float = float(drawing_cfg.get("profile_header_width_mm", 65.0))
        self.default_band_height_mm: float = float(drawing_cfg.get("profile_band_height_mm", 18.4))
        self.ext_alignment_band_height_mm: float = float(drawing_cfg.get("ext_alignment_band_height_mm", 46.0))
        self.prop_alignment_band_height_mm: float = float(drawing_cfg.get("prop_alignment_band_height_mm", 44.0))
        self.prop_gradient_band_height_mm: float = float(drawing_cfg.get("prop_gradient_band_height_mm", 25.0))
        self.exg_gradient_band_height_mm: float = float(drawing_cfg.get("exg_gradient_band_height_mm", self.default_band_height_mm))
        self.chainage_band_height_mm: float = float(drawing_cfg.get("chainage_band_height_mm", 20.0))
        self.km_band_height_mm: float = float(drawing_cfg.get("km_band_height_mm", 17.0))
        self.band_text_height_mm: float = float(drawing_cfg.get("profile_band_text_height_mm", 2.0))
        self.band_text_offset_mm: float = float(drawing_cfg.get("profile_band_text_offset_mm", 0.3))
        self.header_text_height_mm: float = float(drawing_cfg.get("profile_header_text_height_mm", 2.2))
        self.wrap_band_text: bool = bool(drawing_cfg.get("wrap_band_text", True))
        self.elev_grid_step: float = float(drawing_cfg.get("profile_elevation_grid_interval_m", 5.0))
        self.include_gradient_bands: bool = bool(
            drawing_cfg.get("include_gradient_bands", False)
            or self.config.get("profile", {}).get("include_gradient_bands", False)
        )
        self.include_alignment_bands: bool = bool(
            drawing_cfg.get("include_alignment_bands", False)
            or self.config.get("profile", {}).get("include_alignment_bands", False)
        )
        self.include_km_band: bool = bool(
            drawing_cfg.get("include_km_band", False)
            or self.config.get("profile", {}).get("include_km_band", False)
            or drawing_cfg.get("include_alignment_bands", False)
            or self.config.get("profile", {}).get("include_alignment_bands", False)
            or self.template_name in ("brownfield_13band", "brownfield_doubling_13band")
        )
        self.show_gradient_band_grid_lines: bool = bool(
            drawing_cfg.get("show_gradient_band_grid_lines", False)
            or self.config.get("profile", {}).get("show_gradient_band_grid_lines", False)
        )
        self.show_alignment_band_grid_lines: bool = bool(
            drawing_cfg.get("show_alignment_band_grid_lines", False)
            or self.config.get("profile", {}).get("show_alignment_band_grid_lines", False)
        )
        self.alignment_bands_show_divider: bool = bool(
            drawing_cfg.get("alignment_bands_show_divider", False)
            or self.config.get("profile", {}).get("alignment_bands_show_divider", False)
        )
        self.band_divider_style: str = str(
            drawing_cfg.get(
                "profile_band_station_style",
                drawing_cfg.get("band_divider_style", "ticks"),
            )
        )
        self.band_tick_length_mm: float = float(
            drawing_cfg.get("profile_band_tick_length_mm", 2.0)
        )

        # Surface profile visibility controls
        self.show_egl: bool = bool(drawing_cfg.get("profile_show_egl", True))
        self.show_prl: bool = bool(drawing_cfg.get("profile_show_prl", True))
        self.show_erl: bool = bool(drawing_cfg.get("profile_show_erl", True))
        self.show_pfl: bool = bool(drawing_cfg.get("profile_show_pfl", True))
        self.show_egl_ordinates: bool = bool(drawing_cfg.get("profile_show_egl_ordinates", True))
        self.show_profile_line_labels: bool = bool(
            drawing_cfg.get("show_profile_line_labels", drawing_cfg.get("profile_show_line_labels", True))
        )
        self.profile_callouts_offset_x_mm: float = float(
            drawing_cfg.get("profile_callouts_offset_x_mm", 25.0)
        )
        self.show_vip_lines: bool = bool(drawing_cfg.get("profile_show_vip_lines", True))
        self.show_horizontal_grid: bool = bool(drawing_cfg.get("profile_show_horizontal_grid", False))
        self.show_vertical_grid: bool = bool(drawing_cfg.get("profile_show_vertical_grid", False))
        self.show_elevation_ruler: bool = bool(drawing_cfg.get("profile_show_elevation_ruler", True))
        self.show_gradient_posts: bool = bool(drawing_cfg.get("profile_show_gradient_posts", True))
        self.show_direction_arrow: bool = bool(drawing_cfg.get("profile_show_direction_arrow", True))
        self.show_curve_tp: bool = bool(drawing_cfg.get("profile_show_curve_tp", True))
        self.to_station: str = str(
            drawing_cfg.get("to_station", "")
            or self.config.get("project", {}).get("to_station", "WADI STN.")
        )
        self.from_station: str = str(
            drawing_cfg.get("from_station", "")
            or self.config.get("project", {}).get("from_station", "SECUNDERABAD")
        )
        self.gradient_post_slant_deg: float = float(drawing_cfg.get("profile_gradient_post_slant_deg", 45.0))
        self.gradient_post_leader_len_mm: float = float(drawing_cfg.get("profile_gradient_post_leader_len_mm", 45.0))
        self.gradient_post_text_height_mm: float = float(drawing_cfg.get("profile_gradient_post_text_height_mm", 2.0))
        self.show_profile_bridges: bool = bool(drawing_cfg.get("profile_show_bridges", drawing_cfg.get("show_profile_bridges", True)))
        self.show_profile_bridge_marker: bool = bool(drawing_cfg.get("profile_show_bridge_marker", False))
        self.bridge_callout_slant_deg: float = float(drawing_cfg.get("profile_bridge_callout_slant_deg", 45.0))
        self.bridge_callout_text_height_mm: float = float(drawing_cfg.get("profile_bridge_callout_text_height_mm", 2.0))
        self.bridge_callout_line_spacing_mm: float = float(drawing_cfg.get("profile_bridge_callout_line_spacing_mm", 2.8))
        self.bridge_callout_max_rise_mm: float = float(drawing_cfg.get("profile_bridge_callout_max_rise_mm", 55.0))
        self.plan_bot_y: float = float(drawing_cfg.get("plan_bot_y", 465.0))
        self.datum_round_step_m: float = float(drawing_cfg.get("datum_round_step_m", 5.0))
        self.datum_clearance_m: float = float(drawing_cfg.get("datum_clearance_m", 2.0))
        self.last_top_grid_y: Optional[float] = None
        self.last_datum_m: Optional[float] = None
        self.ordinate_deconflict_enabled: bool = bool(drawing_cfg.get("profile_ordinate_deconflict_enabled", True))
        self.ordinate_min_spacing_mm: float = float(drawing_cfg.get("profile_ordinate_min_spacing_mm", 16.0))
        self.ordinate_jog_depth_mm: float = float(drawing_cfg.get("profile_ordinate_jog_depth_mm", 20.0))

        # Layer colors
        self.col_egl: int = int(drawing_cfg.get("profile_egl_color", 7))     # White (Ground Level)
        self.col_erl: int = int(drawing_cfg.get("profile_erl_color", 7))     # White (Existing Rail)
        self.col_pfl: int = int(drawing_cfg.get("profile_pfl_color", 1))     # Red
        self.col_prl: int = int(drawing_cfg.get("profile_prl_color", 1))     # Red
        self.col_datum: int = int(drawing_cfg.get("profile_datum_color", 7)) # White
        self.col_grid: int = int(drawing_cfg.get("profile_grid_color", 7))   # White (ACI 7)
        self.col_border: int = int(drawing_cfg.get("profile_band_border_color", 7))
        self.col_text_ex: int = int(drawing_cfg.get("profile_band_text_color", 7)) # White
        self.col_text_prop: int = int(drawing_cfg.get("profile_band_prop_color", 1)) # Red
        self.font_name: str = str(drawing_cfg.get("font_name", "arial.ttf"))
        self.font_width_factor: float = float(drawing_cfg.get("font_width_factor", 0.85))

        # KM stone symbol & counterchange hatch parameters (Indian Railways standard)
        self.km_stone_hatch_pattern: str = str(
            drawing_cfg.get(
                "profile_km_stone_hatch_pattern",
                drawing_cfg.get("km_stone_hatch_pattern", "ANSI31"),
            )
        )
        self.km_stone_hatch_scale: float = float(
            drawing_cfg.get(
                "profile_km_stone_hatch_scale",
                drawing_cfg.get("km_stone_hatch_scale", 0.25),
            )
        )
        self.km_stone_hatch_angle: float = float(
            drawing_cfg.get(
                "profile_km_stone_hatch_angle",
                drawing_cfg.get("km_stone_hatch_angle", 0.0),
            )
        )
        self.km_stone_text_height_profile: float = float(
            drawing_cfg.get(
                "profile_km_stone_text_height_mm",
                drawing_cfg.get("km_stone_text_height_profile_mm", 2.2),
            )
        )

        # Unified chainage format (synchronized with Plan View)
        self.chainage_format: str = str(
            drawing_cfg.get("chainage_format")
            or drawing_cfg.get("plan_view", {}).get("chainage_format")
            or self.config.get("drawing", {}).get("chainage_format")
            or self.config.get("drawing", {}).get("plan_view", {}).get("chainage_format")
            or self.config.get("alignment", {}).get("chainage_format")
            or "CH-{km}+{m:03d}"
        )

        # Build band definitions based on selected template
        self.band_defs: list[ProfileBandDef] = self._build_band_definitions()
        self.total_band_height_mm: float = sum(b.height_mm for b in self.band_defs)

        # Legacy band_specs tuple format for backward compatibility
        self.band_specs: list[tuple[str, float, int]] = [
            (b.title, b.height_mm, b.value_color) for b in self.band_defs
        ]

    def _parse_dynamic_band_definitions(
        self, bands_config: list[dict[str, Any] | ProfileBandDef]
    ) -> list[ProfileBandDef]:
        """Parse and validate dynamic user-selected bands from UI/config.

        Args:
            bands_config: List of band configuration dicts or ProfileBandDef objects.

        Returns:
            List of enabled ProfileBandDef objects ready for CAD rendering.
        """
        color_name_map = {
            "red": 1,
            "yellow": 2,
            "green": 3,
            "cyan": 4,
            "blue": 5,
            "magenta": 6,
            "white": 7,
            "gray": 8,
            "grey": 8,
            "light_gray": 9,
        }

        def _resolve_color(col_val: Any, default: int) -> int:
            if col_val is None:
                return default
            if isinstance(col_val, int):
                return col_val
            if isinstance(col_val, str):
                s = col_val.strip().lower()
                if s.isdigit():
                    return int(s)
                return color_name_map.get(s, default)
            return default

        parsed_defs: list[ProfileBandDef] = []
        for item in bands_config:
            if isinstance(item, ProfileBandDef):
                if item.enabled:
                    parsed_defs.append(item)
                continue

            if not isinstance(item, dict):
                continue

            enabled = bool(item.get("enabled", True))
            if not enabled:
                continue

            band_id = str(item.get("id", item.get("field_name", ""))).strip()
            title = str(item.get("title", item.get("name", band_id))).strip()
            height_mm = float(item.get("height_mm", item.get("height", self.default_band_height_mm)))

            source_col = str(item.get("source_column", item.get("column", item.get("col", "")))).strip()
            field_name = str(item.get("field_name", source_col or band_id)).strip()

            h_def_col = (
                self.col_text_prop
                if any(k in band_id.lower() or k in title.lower() for k in ("prop", "diff", "cut", "bank"))
                else self.col_text_ex
            )
            header_color = _resolve_color(
                item.get("color_aci", item.get("header_color", item.get("color", h_def_col))),
                h_def_col,
            )
            value_color = _resolve_color(
                item.get("value_color", item.get("color_aci", item.get("color", header_color))),
                header_color,
            )

            signed = bool(item.get("signed", False))
            precision = item.get("precision")
            if precision is not None:
                try:
                    precision = int(precision)
                except (ValueError, TypeError):
                    precision = None

            fmt_type = str(item.get("format_type", "")).strip().lower()
            if not fmt_type:
                if signed or any(k in band_id.lower() or k in title.lower() for k in ("diff", "cut", "bank")):
                    fmt_type = "signed_3dp"
                elif any(k in band_id.lower() or k in title.lower() for k in ("chainage", "ch")):
                    fmt_type = "chainage_2dp"
                elif "gradient" in band_id.lower():
                    fmt_type = "gradient_span"
                elif "alignment" in band_id.lower():
                    fmt_type = "alignment_band"
                elif "km" in band_id.lower():
                    fmt_type = "km_stone"
                else:
                    fmt_type = "unsigned_3dp"

            wrap_text = bool(item.get("wrap_text", self.wrap_band_text))
            usable_w = max(20.0, self.header_width_mm - 8.0)
            char_w = max(1.2, 0.78 * self.header_text_height_mm)
            max_chars = max(18, int(usable_w / char_w))

            lines_raw = item.get("lines")
            if lines_raw and isinstance(lines_raw, list):
                if wrap_text:
                    lines = []
                    for lr in lines_raw:
                        lines.extend(wrap_band_header_text(str(lr), max_chars=max_chars))
                else:
                    lines = [str(l) for l in lines_raw]
            elif wrap_text:
                lines = wrap_band_header_text(title, max_chars=max_chars)
            elif "\n" in title:
                lines = [line.strip() for line in title.split("\n") if line.strip()]
            else:
                lines = [title]

            raw_line_cols = item.get("line_colors", [])
            line_colors = [_resolve_color(c, header_color) for c in raw_line_cols] if raw_line_cols else []
            omit_bottom = bool(item.get("omit_bottom_border", False))

            b_def = ProfileBandDef(
                title=title,
                height_mm=height_mm,
                header_color=header_color,
                value_color=value_color,
                field_name=field_name,
                format_type=fmt_type,
                lines=lines,
                omit_bottom_border=omit_bottom,
                line_colors=line_colors,
                id=band_id,
                enabled=enabled,
                source_column=source_col or field_name,
                precision=precision,
                signed=signed,
                wrap_text=wrap_text,
            )
            parsed_defs.append(b_def)

        return parsed_defs

    def _build_band_definitions(self) -> list[ProfileBandDef]:
        """Construct the list of band definitions according to dynamic selection or configured template.

        Template 'brownfield_doubling' exactly matches the Indian Railways doubling profile:
            Band 1: Diff.Between Rail levels (Red)
            Band 2: Height Of (+) Bank / (-) Cutting (Red)
            Band 3: Rail Level of Existing Track (M) (White)
            Band 4: Formation Level of Existing Track (M) (White)
            Band 5: proposed Rail Level Doubling Line (Red)
            Band 6: Proposed Formation Level of Doubling Line (Red)
            Band 7: Ground Level(M) (White)
            Band 8: Chainage (Km) (White)

        Returns:
            List of ProfileBandDef objects in order from top to bottom.
        """
        # 1. Dynamic User-Selected Bands (from UI payload or config)
        dynamic_bands = self.custom_lsection_bands
        if dynamic_bands is None:
            dynamic_bands = (
                self.config.get("lsection_bands")
                or self.config.get("drawing", {}).get("lsection_bands")
                or self.config.get("profile", {}).get("bands")
            )

        if dynamic_bands:
            custom_defs = self._parse_dynamic_band_definitions(dynamic_bands)
            if custom_defs:
                self.logger.info("Configured %d dynamic user-selected profile data bands", len(custom_defs))
                return custom_defs

        if self.template_name == "rdso_standard":
            # 10 Standard RDSO Bands
            return [
                ProfileBandDef("VERTICAL GEOMETRY", 24.0, self.col_text_prop, self.col_text_prop, "vert_geom", "string", ["VERTICAL GEOMETRY"]),
                ProfileBandDef("HORIZONTAL GEOMETRY", 24.0, 4, 4, "horiz_geom", "string", ["HORIZONTAL GEOMETRY"]),
                ProfileBandDef("CUT (-) / FILL (+)", 18.0, self.col_text_prop, self.col_text_prop, "cut_fill_m", "signed_3dp", ["CUT (-) / FILL (+)"]),
                ProfileBandDef("DIFF PRL & ERL", 18.0, self.col_text_prop, self.col_text_prop, "diff_prl_erl_m", "signed_3dp", ["DIFF PRL & ERL"]),
                ProfileBandDef("PROP RAIL LEVEL", 22.0, self.col_text_prop, self.col_text_prop, "prl_m", "unsigned_3dp", ["PROP RAIL LEVEL"]),
                ProfileBandDef("PROP. FORMATION LEVEL", 22.0, self.col_text_prop, self.col_text_prop, "pfl_m", "unsigned_3dp", ["PROP. FORMATION LEVEL"]),
                ProfileBandDef("EX RAIL LEVEL", 18.0, self.col_text_ex, self.col_text_ex, "erl_m", "unsigned_3dp", ["EX RAIL LEVEL"]),
                ProfileBandDef("EX. FORMATION LEVEL", 18.0, self.col_text_ex, self.col_text_ex, "efl_m", "unsigned_3dp", ["EX. FORMATION LEVEL"]),
                ProfileBandDef("GROUND LEVEL", 22.0, self.col_text_ex, self.col_text_ex, "egl_m", "unsigned_3dp", ["GROUND LEVEL"]),
                ProfileBandDef("CHAINAGE (m)", 18.0, self.col_text_ex, self.col_text_ex, "chainage_m", "chainage_2dp", ["CHAINAGE (m)"]),
            ]

        if self.template_name in ("usbr_qazigund", "usbr_qazigund_13band"):
            # Official 13-Band Layout matching USBRL Qazigund & Northern Railway Doubling standards
            # Exact top-to-bottom header text and line colors matching field reference drawing:
            b_h = self.default_band_height_mm
            return [
                ProfileBandDef(
                    title="PROPOSED\nGRADIENT",
                    height_mm=self.prop_gradient_band_height_mm,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="prop_gradient",
                    format_type="gradient_span",
                    lines=["PROPOSED", "GRADIENT"],
                    line_colors=[self.col_text_prop, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="EXISTING\nGRADIENT",
                    height_mm=self.exg_gradient_band_height_mm,
                    header_color=self.col_text_ex,
                    value_color=self.col_text_ex,
                    field_name="exg_gradient",
                    format_type="gradient_span",
                    lines=["EXISTING", "GRADIENT"],
                    line_colors=[self.col_text_ex, self.col_text_ex],
                ),
                ProfileBandDef(
                    title="PROPOSED\nALIGNMENT",
                    height_mm=self.prop_alignment_band_height_mm,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="prop_alignment",
                    format_type="alignment_band",
                    lines=["PROPOSED", "ALIGNMENT"],
                    line_colors=[self.col_text_prop, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="PROPOSED\nGRADIENT",
                    height_mm=self.prop_gradient_band_height_mm,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="prop_gradient_diagram",
                    format_type="gradient_diagram",
                    lines=["PROPOSED", "GRADIENT"],
                    line_colors=[self.col_text_prop, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="CUT & FILL",
                    height_mm=b_h,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="cut_fill_m",
                    format_type="signed_3dp",
                    lines=["CUT & FILL"],
                    line_colors=[self.col_text_ex, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="TRACK CENTER\nEXISTING UP &\nPROPOSED",
                    height_mm=b_h,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_ex,
                    field_name="track_center_m",
                    format_type="unsigned_3dp",
                    lines=["TRACK CENTER", "EXISTING UP &", "PROPOSED"],
                    line_colors=[self.col_text_prop, self.col_text_ex, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="DIFFRENCE\nEXISTING UP &\nPROPOSED\nRAIL LEVEL",
                    height_mm=b_h,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="diff_prl_erl_m",
                    format_type="unsigned_3dp",
                    lines=["DIFFRENCE", "EXISTING UP &", "PROPOSED", "RAIL LEVEL"],
                    line_colors=[self.col_text_prop, self.col_text_ex, self.col_text_prop, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="PROPOSED\nRAIL LEVEL\nIN METER",
                    height_mm=b_h,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="prl_m",
                    format_type="unsigned_3dp",
                    lines=["PROPOSED", "RAIL LEVEL", "IN METER"],
                    line_colors=[self.col_text_prop, self.col_text_prop, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="EXISTING\nUP RAIL LEVEL\nIN METER",
                    height_mm=b_h,
                    header_color=self.col_text_ex,
                    value_color=self.col_text_ex,
                    field_name="erl_m",
                    format_type="unsigned_3dp",
                    lines=["EXISTING", "UP RAIL LEVEL", "IN METER"],
                    line_colors=[self.col_text_ex, self.col_text_ex, self.col_text_ex],
                ),
                ProfileBandDef(
                    title="PROPOSED\nFORMATION\nIN METER",
                    height_mm=b_h,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="pfl_m",
                    format_type="unsigned_3dp",
                    lines=["PROPOSED", "FORMATION", "IN METER"],
                    line_colors=[self.col_text_prop, self.col_text_prop, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="GROUND LEVEL\nAT PROP.LINE\nIN METER",
                    height_mm=b_h,
                    header_color=self.col_text_ex,
                    value_color=self.col_text_ex,
                    field_name="egl_m",
                    format_type="unsigned_3dp",
                    lines=["GROUND LEVEL", "AT PROP.LINE", "IN METER"],
                    line_colors=[self.col_text_ex, self.col_text_prop, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="PROPOSED\nCHAINAGE",
                    height_mm=self.chainage_band_height_mm,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_ex,
                    field_name="chainage_m",
                    format_type="chainage_railway",
                    lines=["PROPOSED", "CHAINAGE"],
                    line_colors=[self.col_text_prop, self.col_text_prop],
                ),
                ProfileBandDef(
                    title="EXISTING\nKILOMETER",
                    height_mm=self.km_band_height_mm,
                    header_color=self.col_text_ex,
                    value_color=self.col_text_ex,
                    field_name="km_stone",
                    format_type="km_stone",
                    lines=["EXISTING", "KILOMETER"],
                    line_colors=[self.col_text_ex, self.col_text_ex],
                ),
            ]

        # Default: Official Brownfield Doubling Template
        ext_align_h = self.ext_alignment_band_height_mm
        prop_align_h = self.prop_alignment_band_height_mm
        prop_grad_h = self.prop_gradient_band_height_mm
        exg_grad_h = self.exg_gradient_band_height_mm
        b_h = self.default_band_height_mm

        alignment_bands: list[ProfileBandDef] = []
        if self.include_alignment_bands or self.template_name in ("brownfield_12band", "brownfield_doubling_12band", "brownfield_13band", "brownfield_doubling_13band"):
            alignment_bands = [
                ProfileBandDef(
                    title="EXT.Alignment",
                    height_mm=ext_align_h,
                    header_color=self.col_text_ex,
                    value_color=self.col_text_ex,
                    field_name="ext_alignment",
                    format_type="alignment_band",
                    lines=["EXT.Alignment"],
                    omit_bottom_border=not self.alignment_bands_show_divider,
                ),
                ProfileBandDef(
                    title="PROP.Alignment",
                    height_mm=prop_align_h,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="prop_alignment",
                    format_type="alignment_band",
                    lines=["PROP.Alignment"],
                    omit_bottom_border=False,
                ),
            ]

        gradient_bands: list[ProfileBandDef] = []
        if self.include_gradient_bands or self.template_name in ("brownfield_10band", "brownfield_doubling_10band", "brownfield_12band", "brownfield_doubling_12band", "brownfield_13band", "brownfield_doubling_13band"):
            gradient_bands = [
                ProfileBandDef(
                    title="Gradient of Propose Line",
                    height_mm=prop_grad_h,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="prop_gradient",
                    format_type="gradient_span",
                    lines=["Gradient of Propose Line"],
                ),
                ProfileBandDef(
                    title="Gradient of Existing Track",
                    height_mm=exg_grad_h,
                    header_color=self.col_text_ex,
                    value_color=self.col_text_ex,
                    field_name="exg_gradient",
                    format_type="gradient_span",
                    lines=["Gradient of Existing Track"],
                ),
            ]

        data_bands = [
            ProfileBandDef(
                title="Diff.Between Rail levels",
                height_mm=b_h,
                header_color=self.col_text_prop,
                value_color=self.col_text_prop,
                field_name="diff_prl_erl_m",
                format_type="signed_3dp",
                lines=["Diff.Between Rail levels"],
            ),
            ProfileBandDef(
                title="Height Of (+) Bank / (-) Cutting",
                height_mm=b_h,
                header_color=self.col_text_prop,
                value_color=self.col_text_prop,
                field_name="cut_fill_m",
                format_type="signed_3dp",
                lines=["Height Of (+) Bank / (-) Cutting"],
            ),
            ProfileBandDef(
                title="Rail Level of Existing Track (M)",
                height_mm=b_h,
                header_color=self.col_text_ex,
                value_color=self.col_text_ex,
                field_name="erl_m",
                format_type="unsigned_3dp",
                lines=["Rail Level of", "Existing Track (M)"],
            ),
            ProfileBandDef(
                title="Formation Level of Existing Track (M)",
                height_mm=b_h,
                header_color=self.col_text_ex,
                value_color=self.col_text_ex,
                field_name="efl_m",
                format_type="unsigned_3dp",
                lines=["Formation Level of", "Existing Track (M)"],
            ),
            ProfileBandDef(
                title=f"proposed Rail Level\n{self.track_label}",
                height_mm=b_h,
                header_color=self.col_text_prop,
                value_color=self.col_text_prop,
                field_name="prl_m",
                format_type="unsigned_3dp",
                lines=["proposed Rail Level", self.track_label],
            ),
            ProfileBandDef(
                title=f"Proposed Formation Level of\n{self.track_label}",
                height_mm=b_h,
                header_color=self.col_text_prop,
                value_color=self.col_text_prop,
                field_name="pfl_m",
                format_type="unsigned_3dp",
                lines=["Proposed Formation Level of", self.track_label],
            ),
            ProfileBandDef(
                title="Ground Level(M)",
                height_mm=b_h,
                header_color=self.col_text_ex,
                value_color=self.col_text_ex,
                field_name="egl_m",
                format_type="unsigned_3dp",
                lines=["Ground Level(M)"],
            ),
            ProfileBandDef(
                title="Chainage (Km)",
                height_mm=self.chainage_band_height_mm,
                header_color=self.col_text_ex,
                value_color=self.col_text_ex,
                field_name="chainage_m",
                format_type="chainage_2dp",
                lines=["Chainage (Km)"],
            ),
        ]

        km_band: list[ProfileBandDef] = []
        if self.include_km_band or self.template_name in ("brownfield_13band", "brownfield_doubling_13band"):
            km_band = [
                ProfileBandDef(
                    title="KILOMETRE",
                    height_mm=self.km_band_height_mm,
                    header_color=self.col_text_prop,
                    value_color=self.col_text_prop,
                    field_name="km_stone",
                    format_type="km_stone",
                    lines=["KILOMETRE"],
                )
            ]

        return alignment_bands + gradient_bands + data_bands + km_band

    def _setup_dxf_document(self) -> tuple[ezdxf.document.Drawing, Any]:
        """Create and initialize DXF document with RDSO L-Section layers."""
        doc = ezdxf.new(dxfversion="R2018")
        doc.units = units.M
        doc.header["$INSUNITS"] = units.M
        msp = doc.modelspace()

        # Add dashed linetype for PRL if not present
        if "PRL_DASHED" not in doc.linetypes:
            doc.linetypes.add(
                name="PRL_DASHED",
                pattern="A,10.0,-3.0,10.0,-3.0",
                description="Proposed Rail Level ____  ____",
            )

        layers = [
            ("ESP_PROF_EGL", self.col_egl, 25),
            ("ESP_PROF_ERL", self.col_erl, 25),
            ("ESP_PROF_EFL", 8, 18),
            ("ESP_PROF_PFL", self.col_pfl, 35),
            ("ESP_PROF_PRL", self.col_prl, 25),
            ("ESP_PROF_DATUM", self.col_datum, 25),
            ("ESP_PROF_GRID", self.col_grid, 9),
            ("ESP_PROF_BANDS", self.col_border, 25),
            ("ESP_PROF_TEXT_EX", self.col_text_ex, 18),
            ("ESP_PROF_TEXT_PROP", self.col_text_prop, 18),
            ("ESP_PROF_VIP", self.col_prl, 20),
            ("ESP_BRIDGES", 6, 25),
            ("ESP_BORDER", 7, 35),
        ]

        for name, color, weight in layers:
            if name not in doc.layers:
                doc.layers.add(name=name, color=color, lineweight=weight)

        if "Standard" in doc.styles:
            std = doc.styles.get("Standard")
            std.dxf.font = self.font_name
            std.dxf.width = self.font_width_factor

        if "ESP_STYLE" not in doc.styles:
            st = doc.styles.add("ESP_STYLE", font=self.font_name)
            st.dxf.width = self.font_width_factor
        else:
            st = doc.styles.get("ESP_STYLE")
            st.dxf.font = self.font_name
            st.dxf.width = self.font_width_factor

        return doc, msp

    def _prepare_profile_dataframe(
        self,
        survey_df: pd.DataFrame,
        pfl_series: Optional[Sequence[float]] = None,
        egl_series: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """Harmonize survey data and compute all band values at each station.

        Standardizes column names supporting both internal names ('chainage_m', 'erl_m')
        and official survey sheet headers ('CHAINAGE', 'GL', 'PROP.FL', 'PROP.RL',
        'EXG.FL', 'EXG.RL', 'FILL/CUT', 'PROP.RL-EXG.RL').

        Args:
            survey_df: Survey or profile DataFrame.
            pfl_series: Optional explicit Proposed Formation Level values.
            egl_series: Optional explicit Ground Level values.

        Returns:
            DataFrame with all band columns populated.
        """
        df = survey_df.copy()
        df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]

        # Map common Excel aliases to internal standard names (including CAD drawing band titles)
        alias_map = {
            "chainage": "chainage_m",
            "ch": "chainage_m",
            "chainage (km)": "chainage_m",
            "chainage (m)": "chainage_m",
            "chainage(km)": "chainage_m",
            "chainage(m)": "chainage_m",
            "gl": "egl_m",
            "ground_level": "egl_m",
            "ground level": "egl_m",
            "ground level(m)": "egl_m",
            "ground level (m)": "egl_m",
            "prop.fl": "pfl_m",
            "prop_fl": "pfl_m",
            "pfl": "pfl_m",
            "proposed formation level of doubling line": "pfl_m",
            "proposed formation level": "pfl_m",
            "prop. formation level": "pfl_m",
            "prop.rl": "prl_m",
            "prop_rl": "prl_m",
            "prl": "prl_m",
            "proposed rail level doubling line": "prl_m",
            "proposed rail level": "prl_m",
            "prop. rail level": "prl_m",
            "exg.fl": "efl_m",
            "exg_fl": "efl_m",
            "efl": "efl_m",
            "formation level of existing track (m)": "efl_m",
            "formation level of existing track": "efl_m",
            "ex formation level": "efl_m",
            "exg.rl": "erl_m",
            "exg_rl": "erl_m",
            "erl": "erl_m",
            "rail level of existing track (m)": "erl_m",
            "rail level of existing track": "erl_m",
            "ex rail level": "erl_m",
            "fill/cut": "cut_fill_m",
            "fill_cut": "cut_fill_m",
            "height of (+) bank / (-) cutting": "cut_fill_m",
            "height of (+) bank/(-) cutting": "cut_fill_m",
            "height of bank/cutting": "cut_fill_m",
            "height of bank / cutting": "cut_fill_m",
            "prop.rl-exg.rl": "diff_prl_erl_m",
            "prop_rl_exg_rl": "diff_prl_erl_m",
            "diff.between rail levels": "diff_prl_erl_m",
            "diff. between rail levels": "diff_prl_erl_m",
            "diff between rail levels": "diff_prl_erl_m",
            "diff.between rail level": "diff_prl_erl_m",
            "ground level at prop. line (m)": "egl_m",
            "ground level at prop line (m)": "egl_m",
            "ground level at prop. line": "egl_m",
            "existing up rail level in meter": "erl_m",
            "existing dn rail level in meter": "erl_m",
            "existing up rail level": "erl_m",
            "existing dn rail level": "erl_m",
            "proposed rail level in meter": "prl_m",
            "proposed formation level ": "pfl_m",
            "proposed formation level": "pfl_m",
            "proposed formation level in meter": "pfl_m",
            "difference existing up rail level and proposed rail level": "diff_prl_erl_m",
            "difference existing dn rail level and proposed rail level": "diff_prl_erl_m",
            "track center existing up and proposed": "track_center_m",
            "track center existing dn and proposed": "track_center_m",
            "track center": "track_center_m",
            "cut & fill": "cut_fill_m",
            "cut and fill": "cut_fill_m",
        }
        for col in list(df.columns):
            c_clean = str(col).strip().lower()
            if c_clean in alias_map and alias_map[c_clean] not in df.columns:
                df[alias_map[c_clean]] = df[col]

        # Map dynamic custom band source_column -> field_name from active band_defs
        for b_def in getattr(self, "band_defs", []):
            fn = b_def.field_name
            sc = getattr(b_def, "source_column", None) or fn
            if sc in df.columns and fn not in df.columns:
                df[fn] = df[sc]
            elif fn not in df.columns:
                sc_clean = sc.strip().lower()
                for c in df.columns:
                    if str(c).strip().lower() == sc_clean:
                        df[fn] = df[c]
                        break

        # Auto-resolve chainage_m from dynamic bands or df columns
        if "chainage_m" not in df.columns:
            for b_def in getattr(self, "band_defs", []):
                if (
                    b_def.format_type.startswith("chainage")
                    or b_def.id in ("ch", "chainage", "chainage_m")
                    or "chainage" in b_def.field_name.lower()
                    or "station" in b_def.field_name.lower()
                    or "ch" in b_def.field_name.lower()
                ):
                    if b_def.field_name in df.columns:
                        df["chainage_m"] = df[b_def.field_name]
                        break
                    sc = getattr(b_def, "source_column", None)
                    if sc and sc in df.columns:
                        df["chainage_m"] = df[sc]
                        break

        # Check df columns directly for chainage, km, station, distance
        if "chainage_m" not in df.columns:
            for col in df.columns:
                cl = str(col).strip().lower()
                if any(k in cl for k in ("chainage", "ch.", "ch ", "station", "stn", "dist", "km")):
                    df["chainage_m"] = df[col]
                    break

        # Fallback to the very first column in df
        if "chainage_m" not in df.columns and len(df.columns) > 0:
            df["chainage_m"] = df[df.columns[0]]

        # Ensure chainage_m is numeric; fallback to synthetic spacing if non-numeric
        ch_numeric = pd.to_numeric(df["chainage_m"], errors="coerce")
        if ch_numeric.notna().sum() > 0:
            df["chainage_m"] = ch_numeric
            df = df.dropna(subset=["chainage_m"]).sort_values(by="chainage_m").reset_index(drop=True)
        else:
            df["chainage_m"] = [float(i * 25.0) for i in range(len(df))]

        n = len(df)

        # Detect if chainages are in kilometers and convert to meters
        ch_s = pd.to_numeric(df["chainage_m"], errors="coerce").dropna()
        if not ch_s.empty:
            max_abs = float(ch_s.abs().max())
            diffs = ch_s.diff().abs().dropna()
            median_diff = float(diffs[diffs > 0].median()) if not diffs[diffs > 0].empty else 1.0
            if max_abs <= 500.0 and (median_diff < 1.0 or any("(km)" in str(c).lower() for c in survey_df.columns)) and max_abs > 0:
                self.logger.info("Auto-detected chainages in Kilometers in profile drawer; converting to meters (* 1000.0)")
                df["chainage_m"] = df["chainage_m"] * 1000.0

        # 1. Proposed Formation Level (PFL) & Proposed Rail Level (PRL)
        if pfl_series is not None and len(pfl_series) == n:
            df["pfl_m"] = pd.to_numeric(pd.Series(pfl_series), errors="coerce").interpolate().bfill().ffill()
            df["prl_m"] = df["pfl_m"] + self.track_buildup_m
        elif "pfl_m" in df.columns and df["pfl_m"].notna().any():
            df["pfl_m"] = pd.to_numeric(df["pfl_m"], errors="coerce").interpolate().bfill().ffill()
            if "prl_m" in df.columns and df["prl_m"].notna().any():
                df["prl_m"] = pd.to_numeric(df["prl_m"], errors="coerce").interpolate().bfill().ffill()
            else:
                df["prl_m"] = df["pfl_m"] + self.track_buildup_m
        elif "erl_m" in df.columns and df["erl_m"].notna().any():
            valid_erl = pd.to_numeric(df["erl_m"], errors="coerce").interpolate().bfill().ffill()
            df["prl_m"] = valid_erl + self.default_lift_m
            df["pfl_m"] = df["prl_m"] - self.track_buildup_m
        elif "egl_m" in df.columns and df["egl_m"].notna().any():
            valid_egl = pd.to_numeric(df["egl_m"], errors="coerce").interpolate().bfill().ffill()
            df["pfl_m"] = valid_egl
            df["prl_m"] = df["pfl_m"] + self.track_buildup_m
        else:
            numeric_cols = [
                c for c in df.columns
                if c != "chainage_m" and pd.to_numeric(df[c], errors="coerce").notna().any()
            ]
            if numeric_cols:
                baseline = pd.to_numeric(df[numeric_cols[0]], errors="coerce").interpolate().bfill().ffill()
                df["pfl_m"] = baseline
                df["prl_m"] = df["pfl_m"] + self.track_buildup_m
            else:
                df["pfl_m"] = 100.0
                df["prl_m"] = 100.0 + self.track_buildup_m

        # 2. Existing Ground Level (EGL)
        if egl_series is not None and len(egl_series) == n:
            df["egl_m"] = pd.to_numeric(pd.Series(egl_series), errors="coerce").interpolate().bfill().ffill()
        elif "egl_m" in df.columns and df["egl_m"].notna().any():
            df["egl_m"] = pd.to_numeric(df["egl_m"], errors="coerce").interpolate().bfill().ffill()
        else:
            df["egl_m"] = df["pfl_m"] - 0.300

        # 3. Existing Rail Level (ERL) & Existing Formation Level (EFL)
        if "erl_m" in df.columns:
            df["erl_m"] = pd.to_numeric(df["erl_m"], errors="coerce")
        else:
            df["erl_m"] = np.nan

        if "efl_m" in df.columns and df["efl_m"].notna().any():
            df["efl_m"] = pd.to_numeric(df["efl_m"], errors="coerce")
        else:
            df["efl_m"] = df["erl_m"] - self.track_buildup_m

        # 4. DIFF PRL & ERL (Track Lift / Lowering)
        if "diff_prl_erl_m" in df.columns and df["diff_prl_erl_m"].notna().any():
            df["diff_prl_erl_m"] = pd.to_numeric(df["diff_prl_erl_m"], errors="coerce")
        else:
            df["diff_prl_erl_m"] = df["prl_m"] - df["erl_m"]

        # 5. CUT (-) / FILL (+) Height (PFL - EGL)
        if "cut_fill_m" in df.columns and df["cut_fill_m"].notna().any():
            df["cut_fill_m"] = pd.to_numeric(df["cut_fill_m"], errors="coerce").fillna(df["pfl_m"] - df["egl_m"])
        else:
            df["cut_fill_m"] = df["pfl_m"] - df["egl_m"]

        # 6. Track Center Distance
        if "track_center_m" in df.columns and df["track_center_m"].notna().any():
            df["track_center_m"] = pd.to_numeric(df["track_center_m"], errors="coerce").fillna(5.300)
        else:
            df["track_center_m"] = 5.300

        for opt_c in ["proposed_gradient", "existing_gradient", "proposed_alignment", "feature"]:
            if opt_c in survey_df.columns and opt_c not in df.columns:
                df[opt_c] = survey_df[opt_c]

        return df

    @staticmethod
    def _get_line_prefix(line_label: str) -> str:
        """Format line label into standard prefix like '4th LINE' or '3rd LINE'."""
        lbl_upper = (line_label or "").upper()
        if "4TH" in lbl_upper:
            return "4th LINE"
        elif "3RD" in lbl_upper:
            return "3rd LINE"
        elif "UP" in lbl_upper:
            return "UP LINE"
        elif "DN" in lbl_upper:
            return "DN LINE"
        elif line_label:
            cleaned = line_label.replace("PRO ", "").replace("PROP ", "").strip()
            return f"{cleaned} LINE" if not cleaned.endswith("LINE") else cleaned
        return "PROP LINE"

    @staticmethod
    def deconflict_ordinate_positions(
        raw_xs: Sequence[float],
        min_gap_mm: float = 16.0,
        max_shift_mm: float = 45.0,
        required_gaps_mm: Optional[Sequence[float]] = None,
    ) -> list[float]:
        """Deconflict horizontal ordinate positions at the top boundary.

        Uses multi-pass constraint relaxation across all items to strictly
        guarantee monotonic minimum clearance between adjacent callouts while
        minimizing total horizontal shift from true station coordinates.
        """
        if len(raw_xs) <= 1:
            return [float(x) for x in raw_xs]

        n = len(raw_xs)
        indexed = sorted(enumerate(raw_xs), key=lambda x: x[1])
        orig_indices = [idx for idx, _ in indexed]
        sorted_xs = [float(val) for _, val in indexed]

        def get_gap(i: int, j: int) -> float:
            if required_gaps_mm is None:
                return min_gap_mm
            g_i = float(required_gaps_mm[orig_indices[i]])
            g_j = float(required_gaps_mm[orig_indices[j]])
            return max(g_i, g_j)

        pos = [float(x) for x in sorted_xs]
        gaps = [get_gap(i - 1, i) for i in range(1, n)]

        # Multi-pass constraint relaxation
        for _ in range(150):
            max_violation = 0.0
            # Forward pass (push right)
            for i in range(n - 1):
                req = gaps[i]
                dist = pos[i + 1] - pos[i]
                if dist < req:
                    overlap = req - dist
                    if overlap > max_violation:
                        max_violation = overlap
                    pos[i] -= overlap * 0.5
                    pos[i + 1] += overlap * 0.5

            # Backward pass (push left)
            for i in range(n - 2, -1, -1):
                req = gaps[i]
                dist = pos[i + 1] - pos[i]
                if dist < req:
                    overlap = req - dist
                    if overlap > max_violation:
                        max_violation = overlap
                    pos[i] -= overlap * 0.5
                    pos[i + 1] += overlap * 0.5

            # Soft clamping to max_shift_mm
            for i in range(n):
                shift = pos[i] - sorted_xs[i]
                if abs(shift) > max_shift_mm:
                    pos[i] = sorted_xs[i] + math.copysign(max_shift_mm, shift)

            if max_violation < 1e-4:
                break

        # Final forward pass to strictly guarantee no overlap exists
        for i in range(n - 1):
            req = gaps[i]
            if pos[i + 1] < pos[i] + req:
                pos[i + 1] = pos[i] + req

        res = [0.0] * n
        for orig_idx, val in zip(orig_indices, pos):
            res[orig_idx] = round(val, 3)
        return res

    def _draw_cranked_ordinate_line(
        self,
        msp: Any,
        v_x: float,
        target_x: float,
        y_start: float,
        top_grid_y: float,
        y_jog: float,
        layer: str,
        color: int,
        lineweight: int = 13,
    ) -> None:
        """Render a vertical ordinate line with an optional horizontal crank/dogleg step.

        If target_x equals v_x, a straight vertical line is drawn.
        If target_x differs from v_x, a 3-segment cranked line (AutoCAD Polyline) is drawn:
            1. Vertical:   (v_x, y_start) -> (v_x, y_jog)
            2. Horizontal: (v_x, y_jog) -> (target_x, y_jog)
            3. Vertical:   (target_x, y_jog) -> (target_x, top_grid_y)
        """
        if abs(target_x - v_x) < 0.5:
            msp.add_line(
                (v_x, y_start),
                (v_x, top_grid_y),
                dxfattribs={"layer": layer, "color": color, "lineweight": lineweight},
            )
        else:
            pts = [
                (v_x, y_start),
                (v_x, y_jog),
                (target_x, y_jog),
                (target_x, top_grid_y),
            ]
            msp.add_lwpolyline(
                pts,
                dxfattribs={"layer": layer, "color": color, "lineweight": lineweight},
            )

    def _draw_surface_profile_legend_callouts(
        self,
        msp: Any,
        profile_df: pd.DataFrame,
        x_origin: float,
        y_datum: float,
        scale_x: float,
        scale_y: float,
        min_ch: float,
        datum_m: float,
        top_grid_y: float,
    ) -> None:
        """Draw official RDSO stepped leader callouts for profile surface lines.

        Renders 4 stepped callout labels with downward arrowheads:
          1. PROPOSED RAIL LEVEL (Red, ACI 1, layer ESP_PROF_PRL)
          2. PROPOSED FORMATION LEVEL (Red, ACI 1, layer ESP_PROF_PFL)
          3. EXISTING RAIL LEVEL (White/Black, ACI 7, layer ESP_PROF_ERL)
          4. GROUND (Green, ACI 3, layer ESP_PROF_EGL)
        """
        if profile_df.empty or "chainage_m" not in profile_df.columns:
            return

        def _interp_val(col_name: str, ch: float) -> Optional[float]:
            if col_name not in profile_df.columns or profile_df[col_name].dropna().empty:
                return None
            valid = profile_df.dropna(subset=["chainage_m", col_name]).sort_values("chainage_m")
            if valid.empty:
                return None
            chs = valid["chainage_m"].to_numpy(dtype=float)
            vals = valid[col_name].to_numpy(dtype=float)
            if ch <= chs[0]:
                return float(vals[0])
            if ch >= chs[-1]:
                return float(vals[-1])
            return float(np.interp(ch, chs, vals))

        # Check candidate items to display based on visibility toggles
        raw_items = []
        if self.show_prl:
            raw_items.append({
                "key": "prl",
                "text": "PROPOSED RAIL LEVEL",
                "col_name": "prl_m",
                "color": self.col_prl,
                "layer": "ESP_PROF_TEXT_PROP",
                "shelf_len": 32.5,
            })
        if self.show_pfl:
            raw_items.append({
                "key": "pfl",
                "text": "PROPOSED FORMATION LEVEL",
                "col_name": "pfl_m",
                "color": self.col_pfl,
                "layer": "ESP_PROF_TEXT_PROP",
                "shelf_len": 40.0,
            })
        if self.show_erl:
            raw_items.append({
                "key": "erl",
                "text": "EXISTING RAIL LEVEL",
                "col_name": "erl_m",
                "color": self.col_erl,
                "layer": "ESP_PROF_TEXT_EX",
                "shelf_len": 32.5,
            })
        if self.show_egl:
            raw_items.append({
                "key": "egl",
                "text": "GROUND",
                "col_name": "egl_m",
                "color": 3,  # Green ACI 3 as per RDSO standard / user reference image
                "layer": "ESP_PROF_TEXT_EX",
                "shelf_len": 20.0,
            })

        if not raw_items:
            return

        chs_valid = profile_df["chainage_m"].dropna().to_numpy(dtype=float)
        span_m = float(chs_valid.max() - chs_valid.min()) if len(chs_valid) > 1 else 1000.0
        span_paper_mm = span_m * scale_x

        # Staggered horizontal stem X offsets relative to base
        # Layout: [PRL] --(+8.5)--> [PFL] --(+24.0)--> [ERL] --(+8.5)--> [GROUND]
        if 5.0 < span_paper_mm < 60.0:
            ratio = max(0.2, (span_paper_mm - 4.0) / 60.0)
            x_base = x_origin + 2.0
            stem_spacing_map = {
                "prl": 0.0,
                "pfl": 8.5 * ratio,
                "erl": 32.5 * ratio,
                "egl": 41.0 * ratio,
            }
        else:
            x_base = x_origin + self.profile_callouts_offset_x_mm
            stem_spacing_map = {
                "prl": 0.0,
                "pfl": 8.5,
                "erl": 32.5,
                "egl": 41.0,
            }

        items = []
        for it in raw_items:
            it["stem_x"] = x_base + stem_spacing_map.get(it["key"], len(items) * 12.0)
            it["ch"] = min_ch + (it["stem_x"] - x_origin) / scale_x

            # Determine elevation with robust engineering fallbacks
            z = _interp_val(it["col_name"], it["ch"])
            if z is None:
                if it["key"] == "prl":
                    z_pfl = _interp_val("pfl_m", it["ch"])
                    z_erl = _interp_val("erl_m", it["ch"])
                    z = (z_pfl + self.track_buildup_m) if z_pfl is not None else (z_erl or datum_m + 5.0)
                elif it["key"] == "pfl":
                    z_prl = _interp_val("prl_m", it["ch"])
                    z = (z_prl - self.track_buildup_m) if z_prl is not None else (datum_m + 4.0)
                elif it["key"] == "erl":
                    z_prl = _interp_val("prl_m", it["ch"])
                    z = z_prl if z_prl is not None else (datum_m + 5.0)
                elif it["key"] == "egl":
                    z_erl = _interp_val("erl_m", it["ch"])
                    z = (z_erl - 1.0) if z_erl is not None else (datum_m + 3.0)
            it["z"] = float(z)
            it["target_y"] = y_datum + (it["z"] - datum_m) * scale_y
            items.append(it)

        # Stepped shelf Y coordinates:
        # Lowest shelf is Shelf 4 (GROUND). Shelves step up by 5.5 mm each.
        max_target_y = max(it["target_y"] for it in items)
        n = len(items)
        step_y = 5.5

        # Base shelf Y for lowest item
        y_lowest = max_target_y + 14.0
        y_top_shelf = y_lowest + (n - 1) * step_y
        if y_top_shelf > top_grid_y - 8.0:
            excess = y_top_shelf - (top_grid_y - 8.0)
            y_lowest = max(max_target_y + 4.0, y_lowest - excess)

        for i, it in enumerate(items):
            it["shelf_y"] = y_lowest + (n - 1 - i) * step_y

        arrow_h = 2.4
        arrow_hw = 0.65

        for it in items:
            sx = it["stem_x"]
            sy = it["shelf_y"]
            ty = it["target_y"]
            lay = it["layer"]
            col = it["color"]
            slen = it["shelf_len"]

            # 1. Downward solid arrowhead touching profile line at (sx, ty)
            msp.add_solid(
                [(sx - arrow_hw, ty + arrow_h), (sx + arrow_hw, ty + arrow_h), (sx, ty)],
                dxfattribs={"layer": lay, "color": col},
            )
            msp.add_line((sx - arrow_hw, ty + arrow_h), (sx + arrow_hw, ty + arrow_h), dxfattribs={"layer": lay, "color": col, "lineweight": 18})
            msp.add_line((sx - arrow_hw, ty + arrow_h), (sx, ty), dxfattribs={"layer": lay, "color": col, "lineweight": 18})
            msp.add_line((sx + arrow_hw, ty + arrow_h), (sx, ty), dxfattribs={"layer": lay, "color": col, "lineweight": 18})

            # 2. Vertical stem from shelf down to arrowhead top
            msp.add_line(
                (sx, sy),
                (sx, ty + arrow_h),
                dxfattribs={"layer": lay, "color": col, "lineweight": 18},
            )

            # 3. Horizontal shelf line under the text
            msp.add_line(
                (sx, sy),
                (sx + slen, sy),
                dxfattribs={"layer": lay, "color": col, "lineweight": 18},
            )

            # 4. Text annotation placed horizontally above the shelf line
            msp.add_text(
                it["text"],
                dxfattribs={
                    "layer": lay,
                    "color": col,
                    "style": "ESP_STYLE",
                    "height": 2.2,
                },
            ).set_placement((sx + 1.2, sy + 0.8), align=TextEntityAlignment.BOTTOM_LEFT)

        self.logger.info("Rendered %d surface profile leader callouts", len(items))

    def draw_profile_bands_into_msp(
        self,
        msp: Any,
        profile_df: pd.DataFrame,
        x_origin: float = 65.0,
        y_datum: float = 0.0,
        total_length_m: Optional[float] = None,
        min_chainage: Optional[float] = None,
        prop_gradient_spans: Optional[Sequence[dict[str, Any]]] = None,
        exg_gradient_spans: Optional[Sequence[dict[str, Any]]] = None,
        bridges_df: Optional[pd.DataFrame] = None,
        curves: Optional[Sequence[dict[str, Any]]] = None,
        ext_curves: Optional[Sequence[dict[str, Any]]] = None,
        custom_band_defs: Optional[Sequence[ProfileBandDef]] = None,
    ) -> float:
        """Render the complete L-section elevation grid, polylines, and data bands into modelspace.

        Args:
            msp: ezdxf modelspace object.
            profile_df: Cleaned profile DataFrame with station elevations and chainages.
            x_origin: X offset in mm where the first station column begins (after headers).
            y_datum: Y position in mm of the profile datum line (top border of Band 1).
            total_length_m: Optional fixed span length in meters (defaults to df span).
            min_chainage: Optional fixed start chainage (defaults to df min).
            prop_gradient_spans: Optional proposed line gradient spans.
            exg_gradient_spans: Optional existing track gradient spans.
            bridges_df: Optional DataFrame of bridges and structures along corridor.
            curves: Optional Sequence of proposed horizontal curve records.
            ext_curves: Optional Sequence of existing horizontal curve records.
            custom_band_defs: Optional explicit sequence of ProfileBandDef to render.

        Returns:
            Computed datum elevation in meters.
        """
        active_band_defs: list[ProfileBandDef] = (
            list(custom_band_defs) if custom_band_defs is not None else self.band_defs
        )

        chainages = profile_df["chainage_m"].to_numpy()
        min_ch = float(chainages[0]) if min_chainage is None else float(min_chainage)
        max_ch = float(chainages[-1])
        span_m = (max_ch - min_ch) if total_length_m is None else float(total_length_m)

        # Resolve horizontal curve schedule if available
        active_curves = curves
        if active_curves is None:
            configured_curve_path = self.config.get("paths", {}).get("curve_excel_path")
            if configured_curve_path and Path(configured_curve_path).exists():
                active_curves = self._load_curve_details(configured_curve_path)

        active_ext_curves = ext_curves
        if active_ext_curves is None:
            configured_ext_path = self.config.get("paths", {}).get("ext_curve_excel_path") or self.config.get("paths", {}).get("existing_curve_excel_path")
            if not configured_ext_path:
                cand_p = _ROOT / "outputs" / "brownfield" / "extracted_existing_curves_km110_163.xlsx"
                if cand_p.exists():
                    configured_ext_path = str(cand_p)
            if configured_ext_path and Path(configured_ext_path).exists():
                active_ext_curves = self._load_existing_curve_details(configured_ext_path)

        # Auto-extract gradient spans from profile DataFrame if not explicitly passed
        if prop_gradient_spans is None or len(prop_gradient_spans) == 0:
            if "proposed_gradient" in profile_df.columns:
                prop_gradient_spans = self._extract_gradient_spans_from_dataframe(profile_df, "proposed_gradient", "prl_m")

        if exg_gradient_spans is None or len(exg_gradient_spans) == 0:
            if "existing_gradient" in profile_df.columns:
                exg_gradient_spans = self._extract_gradient_spans_from_dataframe(profile_df, "existing_gradient", "erl_m")

        # 1. Compute Datum Elevation
        min_candidates = []
        for col_c in ["egl_m", "pfl_m", "erl_m", "prl_m", "efl_m"]:
            if col_c in profile_df.columns and profile_df[col_c].notna().any():
                min_candidates.append(float(profile_df[col_c].dropna().min()))
        min_elev = min(min_candidates) if min_candidates else 100.0

        configured_datum = (
            self.config.get("vertical", {}).get("datum_elevation_m")
            or self.config.get("drawing", {}).get("datum_elevation_m")
        )
        if configured_datum is not None:
            datum_m = float(configured_datum)
        else:
            datum_m = float(math.floor((min_elev - self.datum_clearance_m) / self.datum_round_step_m) * self.datum_round_step_m)

        max_candidates = []
        for col_c in ["prl_m", "erl_m", "egl_m", "pfl_m", "efl_m"]:
            if col_c in profile_df.columns and profile_df[col_c].notna().any():
                max_candidates.append(float(profile_df[col_c].dropna().max()))
        max_elev = max(max_candidates) if max_candidates else (min_elev + 20.0)

        # Coordinate transformation multipliers
        scale_x = 1000.0 / self.h_scale  # e.g., 1000 / 5000 = 0.20
        scale_y = 1000.0 / self.v_scale  # e.g., 1000 / 500 = 2.00

        def get_model_xy(ch: float, z: float) -> tuple[float, float]:
            mx = x_origin + (ch - min_ch) * scale_x
            my = y_datum + (z - datum_m) * scale_y
            return (mx, my)

        # 2. Draw Profile Polylines in Modelspace
        egl_pts = (
            [get_model_xy(float(r["chainage_m"]), float(r["egl_m"])) for _, r in profile_df.iterrows() if pd.notna(r["egl_m"])]
            if "egl_m" in profile_df.columns
            else []
        )
        erl_pts = (
            [get_model_xy(float(r["chainage_m"]), float(r["erl_m"])) for _, r in profile_df.iterrows() if pd.notna(r["erl_m"])]
            if "erl_m" in profile_df.columns
            else []
        )
        pfl_pts = (
            [get_model_xy(float(r["chainage_m"]), float(r["pfl_m"])) for _, r in profile_df.iterrows() if pd.notna(r["pfl_m"])]
            if "pfl_m" in profile_df.columns
            else []
        )
        prl_pts = (
            [get_model_xy(float(r["chainage_m"]), float(r["prl_m"])) for _, r in profile_df.iterrows() if pd.notna(r["prl_m"])]
            if "prl_m" in profile_df.columns
            else []
        )

        # EGL (Natural Ground - White Solid)
        if self.show_egl and len(egl_pts) >= 2:
            msp.add_lwpolyline(egl_pts, dxfattribs={"layer": "ESP_PROF_EGL", "color": self.col_egl, "lineweight": 25})

        # Vertical Ordinate Lines from Datum to Ground Level (White) at every station
        if self.show_egl_ordinates and egl_pts:
            for mx, my in egl_pts:
                msp.add_line(
                    (mx, y_datum),
                    (mx, my),
                    dxfattribs={"layer": "ESP_PROF_EGL", "color": self.col_egl, "lineweight": 9},
                )

        # ERL (Existing Rail Level - White)
        if self.show_erl and len(erl_pts) >= 2:
            msp.add_lwpolyline(erl_pts, dxfattribs={"layer": "ESP_PROF_ERL", "color": self.col_erl, "lineweight": 25})

        # PFL (Proposed Formation Level - Red Solid)
        if self.show_pfl and len(pfl_pts) >= 2:
            msp.add_lwpolyline(pfl_pts, dxfattribs={"layer": "ESP_PROF_PFL", "color": self.col_pfl, "lineweight": 35})

        # PRL (Proposed Rail Level - Red Solid)
        if self.show_prl and len(prl_pts) >= 2:
            msp.add_lwpolyline(
                prl_pts,
                dxfattribs={"layer": "ESP_PROF_PRL", "color": self.col_prl, "linetype": "Continuous", "lineweight": 25},
            )

        # 3. Draw Datum Baseline and Elevation Grid
        col_step = self.chainage_interval_m * scale_x
        x_start_grid = x_origin
        x_end_grid = x_origin + span_m * scale_x + col_step

        msp.add_line(
            (x_start_grid, y_datum),
            (x_end_grid, y_datum),
            dxfattribs={"layer": "ESP_PROF_DATUM", "color": self.col_datum, "lineweight": 25},
        )

        # Horizontal Elevation Grid Lines & Left Y-Axis Elevation Ruler
        elev_step = self.elev_grid_step
        start_grid_elev = datum_m
        end_grid_elev = math.ceil(max_elev / elev_step) * elev_step + elev_step
        top_grid_y = y_datum + (end_grid_elev - datum_m) * scale_y
        self.last_top_grid_y = top_grid_y
        self.last_datum_m = datum_m

        if self.show_elevation_ruler:
            # Left Y-axis ruler vertical spine
            msp.add_line(
                (x_start_grid, y_datum),
                (x_start_grid, top_grid_y),
                dxfattribs={"layer": "ESP_PROF_GRID", "color": self.col_grid, "lineweight": 15},
            )

            # Draw 1m minor ticks and 5m major ticks with labels
            curr_elev = start_grid_elev
            while curr_elev <= end_grid_elev + 1e-4:
                gy = y_datum + (curr_elev - datum_m) * scale_y
                is_major = (round(curr_elev) % 5 == 0)

                if is_major:
                    # 5m Major Tick
                    msp.add_line(
                        (x_start_grid, gy),
                        (x_start_grid + 3.0, gy),
                        dxfattribs={"layer": "ESP_PROF_GRID", "color": self.col_grid, "lineweight": 13},
                    )
                    # Left Y-axis elevation tick label (skip if at datum level to avoid collision with DATUM +elev)
                    if abs(curr_elev - datum_m) >= 1e-3:
                        msp.add_text(
                            f"{int(round(curr_elev))}",
                            dxfattribs={
                                "layer": "ESP_PROF_TEXT_EX",
                                "color": self.col_text_ex,
                                "style": "ESP_STYLE",
                                "height": 2.2,
                            },
                        ).set_placement((x_start_grid - 1.5, gy), align=TextEntityAlignment.MIDDLE_RIGHT)

                    # Optional horizontal grid lines across the sheet
                    if self.show_horizontal_grid and curr_elev > datum_m:
                        msp.add_line(
                            (x_start_grid, gy),
                            (x_end_grid, gy),
                            dxfattribs={"layer": "ESP_PROF_GRID", "color": self.col_grid, "lineweight": 9},
                        )
                else:
                    # 1m Minor Tick
                    msp.add_line(
                        (x_start_grid, gy),
                        (x_start_grid + 1.5, gy),
                        dxfattribs={"layer": "ESP_PROF_GRID", "color": self.col_grid, "lineweight": 9},
                    )

                curr_elev += 1.0

            # Datum Elevation Annotation on Left (e.g. "DATUM +540")
            datum_str = (
                f"DATUM +{int(round(datum_m))}"
                if abs(datum_m - round(datum_m)) < 1e-3
                else f"DATUM +{datum_m:.1f}"
            )
            msp.add_text(
                datum_str,
                dxfattribs={
                    "layer": "ESP_PROF_TEXT_EX",
                    "color": self.col_text_ex,
                    "style": "ESP_STYLE",
                    "height": 2.6,
                },
            ).set_placement((x_start_grid - 2.5, y_datum), align=TextEntityAlignment.MIDDLE_RIGHT)

        # 4. Top Boundary, Direction Indicator, and Gradient Posts (PROP & EXT)
        # Top boundary line of profile grid
        msp.add_line(
            (x_start_grid, top_grid_y),
            (x_end_grid, top_grid_y),
            dxfattribs={"layer": "ESP_PROF_DATUM", "color": self.col_datum, "lineweight": 25},
        )
        # Right vertical boundary line of profile grid attaching top boundary directly to datum line
        msp.add_line(
            (x_end_grid, y_datum),
            (x_end_grid, top_grid_y),
            dxfattribs={"layer": "ESP_PROF_DATUM", "color": self.col_datum, "lineweight": 25},
        )

        # 4A. Surface Profile Leader Callouts (PRL, PFL, ERL, GROUND)
        if self.show_profile_line_labels:
            self._draw_surface_profile_legend_callouts(
                msp=msp,
                profile_df=profile_df,
                x_origin=x_start_grid,
                y_datum=y_datum,
                scale_x=scale_x,
                scale_y=scale_y,
                min_ch=min_ch,
                datum_m=datum_m,
                top_grid_y=top_grid_y,
            )

        # Direction Indicator on Top Left (e.g. "TO WARDS \n WADI STN.")
        if self.show_direction_arrow and self.to_station:
            arr_x0 = x_start_grid - 35.0
            arr_x1 = x_start_grid - 5.0
            arr_mid = (arr_x0 + arr_x1) / 2.0
            arr_y = top_grid_y

            # Horizontal line
            msp.add_line(
                (arr_x0, arr_y),
                (arr_x1, arr_y),
                dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 18},
            )
            # Arrowhead pointing left towards destination
            msp.add_line((arr_x0, arr_y), (arr_x0 + 3.0, arr_y + 1.2), dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 18})
            msp.add_line((arr_x0, arr_y), (arr_x0 + 3.0, arr_y - 1.2), dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 18})

            # Text: TO WARDS above line, <STATION> below line
            msp.add_text(
                "TO WARDS",
                dxfattribs={
                    "layer": "ESP_PROF_TEXT_EX",
                    "color": self.col_text_ex,
                    "style": "ESP_STYLE",
                    "height": 2.2,
                },
            ).set_placement((arr_mid, arr_y + 1.2), align=TextEntityAlignment.BOTTOM_CENTER)

            to_stn_text = self.to_station.upper()
            if not to_stn_text.endswith("STN.") and not to_stn_text.endswith("STATION"):
                to_stn_text += " STN."
            msp.add_text(
                to_stn_text,
                dxfattribs={
                    "layer": "ESP_PROF_TEXT_EX",
                    "color": self.col_text_ex,
                    "style": "ESP_STYLE",
                    "height": 2.2,
                },
            ).set_placement((arr_mid, arr_y - 1.2), align=TextEntityAlignment.TOP_CENTER)

        # Direction Indicator on Top Right (e.g. "TO WARDS \n SECUNDERABAD")
        if self.show_direction_arrow and getattr(self, "from_station", None):
            arr_rx0 = x_end_grid + 3.0
            arr_rx1 = x_end_grid + 30.0
            arr_rmid = (arr_rx0 + arr_rx1) / 2.0
            arr_ry = top_grid_y

            # Horizontal line
            msp.add_line(
                (arr_rx0, arr_ry),
                (arr_rx1, arr_ry),
                dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 18},
            )
            # Arrowhead pointing right towards destination/from_station
            msp.add_line((arr_rx1, arr_ry), (arr_rx1 - 3.0, arr_ry + 1.2), dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 18})
            msp.add_line((arr_rx1, arr_ry), (arr_rx1 - 3.0, arr_ry - 1.2), dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 18})

            # Text: TO WARDS above line, <STATION> below line
            msp.add_text(
                "TO WARDS",
                dxfattribs={
                    "layer": "ESP_PROF_TEXT_EX",
                    "color": self.col_text_ex,
                    "style": "ESP_STYLE",
                    "height": 2.2,
                },
            ).set_placement((arr_rmid, arr_ry + 1.2), align=TextEntityAlignment.BOTTOM_CENTER)

            from_stn_text = self.from_station.upper()
            msp.add_text(
                from_stn_text,
                dxfattribs={
                    "layer": "ESP_PROF_TEXT_EX",
                    "color": self.col_text_ex,
                    "style": "ESP_STYLE",
                    "height": 2.2,
                },
            ).set_placement((arr_rmid, arr_ry - 1.2), align=TextEntityAlignment.TOP_CENTER)

        # Helper for slanted gradient post callouts
        rad_slant = math.radians(self.gradient_post_slant_deg)
        cos_slant = math.cos(rad_slant)
        sin_slant = math.sin(rad_slant)

        def _draw_gradient_post_callout(v_x: float, text: str, col: int, lay: str) -> None:
            # Slanted leader underline (dynamic length based on text so it never overhangs)
            calc_len = 2.0 + len(text) * self.gradient_post_text_height_mm * 0.75
            leader_len = max(self.gradient_post_leader_len_mm, calc_len)
            max_allowed_rise = max(15.0, self.plan_bot_y - top_grid_y - 12.0)
            allowed_rise = min(self.bridge_callout_max_rise_mm, max_allowed_rise)
            if sin_slant > 1e-4 and (leader_len * sin_slant) > allowed_rise:
                leader_len = allowed_rise / sin_slant
            end_x = v_x + leader_len * cos_slant
            end_y = top_grid_y + leader_len * sin_slant
            msp.add_line(
                (v_x, top_grid_y),
                (end_x, end_y),
                dxfattribs={"layer": lay, "color": col, "lineweight": 18},
            )
            # Offset text slightly perpendicular above the underline
            t_x = v_x + 1.2 * cos_slant - 0.7 * sin_slant
            t_y = top_grid_y + 1.2 * sin_slant + 0.7 * cos_slant
            msp.add_text(
                text,
                dxfattribs={
                    "layer": lay,
                    "color": col,
                    "style": "ESP_STYLE",
                    "height": self.gradient_post_text_height_mm,
                    "rotation": self.gradient_post_slant_deg,
                },
            ).set_placement((t_x, t_y), align=TextEntityAlignment.BOTTOM_LEFT)

        # 4A. Load Bridge Schedule if enabled and not already provided
        active_bridges_df = bridges_df
        if self.show_profile_bridges:
            if active_bridges_df is None or (isinstance(active_bridges_df, pd.DataFrame) and active_bridges_df.empty):
                configured_br_path = self.config.get("paths", {}).get("bridge_list_path")
                if configured_br_path:
                    br_path = Path(configured_br_path)
                    if not br_path.is_absolute():
                        br_path = _ROOT / br_path
                    if br_path.exists():
                        try:
                            if br_path.suffix.lower() in [".xlsx", ".xls"]:
                                try:
                                    active_bridges_df = pd.read_excel(br_path, sheet_name="Bridge List", header=0)
                                except Exception:
                                    active_bridges_df = pd.read_excel(br_path, header=0)
                            else:
                                active_bridges_df = pd.read_csv(br_path)
                        except Exception as exc:
                            self.logger.warning("Could not auto-load bridge schedule for profile: %s", exc)

        # 4B. Collect Candidate Bridge Records for active sheet
        sheet_bridges: list[dict[str, Any]] = []
        if self.show_profile_bridges and active_bridges_df is not None and isinstance(active_bridges_df, pd.DataFrame) and not active_bridges_df.empty:
            for _, row in active_bridges_df.iterrows():
                ch_val = None
                for key in ["chainage (m)", "chainage_m", "chainage", "CHAINAGE", "approx. chainage (km)", "Approx. Chainage (km)", "ch"]:
                    for rk, rv in row.items():
                        if str(rk).strip().lower() == key.lower() and pd.notna(rv):
                            ch_val = rv
                            break
                    if ch_val is not None:
                        break
                if ch_val is None:
                    continue
                try:
                    ch = float(ch_val)
                    if 0.0 < ch < 500.0:  # in km
                        ch *= 1000.0
                except Exception:
                    continue

                if min_ch - 0.1 <= ch <= min_ch + span_m + 0.1:
                    sheet_bridges.append({
                        "ch": ch,
                        "row": row,
                        "v_x": x_origin + (ch - min_ch) * scale_x,
                    })

        # 4C. Collect VIP / Gradient Posts
        vip_items: list[dict[str, Any]] = []

        if self.show_gradient_posts and exg_gradient_spans:
            exg_vip_chs: set[float] = set()
            for sp in exg_gradient_spans:
                s_c = float(sp.get("start_ch", 0.0))
                e_c = float(sp.get("end_ch", 0.0))
                if min_ch - 0.01 <= s_c <= min_ch + span_m + 0.01:
                    exg_vip_chs.add(round(s_c, 2))
                if min_ch - 0.01 <= e_c <= min_ch + span_m + 0.01:
                    exg_vip_chs.add(round(e_c, 2))

            for v_ch in sorted(exg_vip_chs):
                v_x = x_origin + (v_ch - min_ch) * scale_x
                ch_val_str = f"{int(round(v_ch))}" if abs(v_ch - round(v_ch)) < 1e-2 else f"{v_ch:.1f}"
                vip_items.append({
                    "v_x": v_x,
                    "ch": v_ch,
                    "callout_text": f"EXT . GRADIENT POST AT CH : {ch_val_str}M",
                    "col": self.col_text_ex,
                    "lay": "ESP_PROF_TEXT_EX",
                    "line_lay": "ESP_PROF_GRID",
                    "line_col": self.col_text_ex,
                    "line_lw": 15,
                })

        if self.show_vip_lines and prop_gradient_spans:
            prop_vip_chs: set[float] = set()
            for sp in prop_gradient_spans:
                s_c = float(sp.get("start_ch", 0.0))
                e_c = float(sp.get("end_ch", 0.0))
                if min_ch - 0.01 <= s_c <= min_ch + span_m + 0.01:
                    prop_vip_chs.add(round(s_c, 2))
                if min_ch - 0.01 <= e_c <= min_ch + span_m + 0.01:
                    prop_vip_chs.add(round(e_c, 2))

            for v_ch in sorted(prop_vip_chs):
                v_x = x_origin + (v_ch - min_ch) * scale_x
                ch_val_str = f"{int(round(v_ch))}" if abs(v_ch - round(v_ch)) < 1e-2 else f"{v_ch:.1f}"
                vip_items.append({
                    "v_x": v_x,
                    "ch": v_ch,
                    "callout_text": f"PROP . GRADIENT POST AT CH : {ch_val_str}M" if self.show_gradient_posts else "",
                    "col": self.col_prl,
                    "lay": "ESP_PROF_VIP",
                    "line_lay": "ESP_PROF_VIP",
                    "line_col": self.col_prl,
                    "line_lw": 20,
                })

        # 4C-2. Collect Curve Tangent Points (TP1 and TP2) Callouts
        curve_tp_items: list[dict[str, Any]] = []
        if self.show_curve_tp and active_curves:
            seen_tps: set[tuple[str, str, str, float]] = set()
            for c in active_curves:
                line_label = c.get("line_label", "")
                prefix = self._get_line_prefix(line_label)
                c_raw = str(c.get("curve_no", "")).strip()
                c_str = ""
                if c_raw:
                    c_clean = c_raw.upper().replace("CURVE", "").replace("NO.", "").replace("NO", "").strip().lstrip("-").strip()
                    c_str = f"CURVE NO.{c_clean} " if c_clean else f"{c_raw} "
                for tp_label, tp_key in [("TP1", "tp1"), ("TP2", "tp2")]:
                    tp_val = c.get(tp_key)
                    if tp_val is None:
                        continue
                    try:
                        ch = float(tp_val)
                    except (ValueError, TypeError):
                        continue
                    key_tuple = (prefix, c_raw, tp_label, round(ch, 2))
                    if key_tuple in seen_tps:
                        continue
                    if min_ch - 0.01 <= ch <= min_ch + span_m + 0.01:
                        seen_tps.add(key_tuple)
                        v_x = x_origin + (ch - min_ch) * scale_x
                        ch_str = f"{ch:.3f}".rstrip("0").rstrip(".")
                        callout_text = f"{prefix} {c_str}{tp_label} AT CH: {ch_str}M"
                        curve_tp_items.append({
                            "kind": "CURVE_TP",
                            "v_x": v_x,
                            "ch": ch,
                            "tp_label": tp_label,
                            "curve_no": c_raw,
                            "callout_text": callout_text,
                            "col": self.col_prl,          # 1 (Red ACI)
                            "lay": "ESP_PROF_VIP",        # Red profile VIP layer
                            "line_lay": "ESP_PROF_VIP",
                            "line_col": self.col_prl,     # 1 (Red ACI)
                            "line_lw": 20,                # 0.20 mm
                            "min_gap_mm": 6.5,
                        })

        # 4D. UNIFIED Top-Boundary Ordinate Deconfliction (Gradient Posts + Bridges + Curve TPs)
        # All items that terminate at top_grid_y are resolved together so they never collide!
        unified_top_items: list[dict[str, Any]] = []
        for v in vip_items:
            unified_top_items.append({"kind": "VIP", "v_x": v["v_x"], "ch": v["ch"], "ref": v, "min_gap_mm": 6.5})
        for b in sheet_bridges:
            unified_top_items.append({"kind": "BRIDGE", "v_x": b["v_x"], "ch": b["ch"], "ref": b, "min_gap_mm": self.ordinate_min_spacing_mm})
        for tp in curve_tp_items:
            unified_top_items.append({"kind": "CURVE_TP", "v_x": tp["v_x"], "ch": tp["ch"], "ref": tp, "min_gap_mm": 6.5})

        unified_top_items.sort(key=lambda it: it["v_x"])

        if unified_top_items:
            raw_top_xs = [it["v_x"] for it in unified_top_items]
            item_min_gaps = [it.get("min_gap_mm", self.ordinate_min_spacing_mm) for it in unified_top_items]
            target_top_xs = (
                self.deconflict_ordinate_positions(
                    raw_top_xs,
                    min_gap_mm=self.ordinate_min_spacing_mm,
                    required_gaps_mm=item_min_gaps,
                )
                if self.ordinate_deconflict_enabled
                else raw_top_xs
            )
            for idx, it in enumerate(unified_top_items):
                y_jog = top_grid_y - self.ordinate_jog_depth_mm - (idx % 3) * 4.0
                it["ref"]["target_x"] = target_top_xs[idx]
                it["ref"]["y_jog"] = y_jog

        # Draw VIP / Gradient Posts using unified target_x & y_jog
        for item in vip_items:
            v_x = item["v_x"]
            target_x = item.get("target_x", v_x)
            y_jog = item.get("y_jog", top_grid_y - self.ordinate_jog_depth_mm)
            self._draw_cranked_ordinate_line(
                msp=msp,
                v_x=v_x,
                target_x=target_x,
                y_start=y_datum,
                top_grid_y=top_grid_y,
                y_jog=y_jog,
                layer=item["line_lay"],
                color=item["line_col"],
                lineweight=item["line_lw"],
            )
            if item["callout_text"]:
                _draw_gradient_post_callout(target_x, item["callout_text"], item["col"], item["lay"])

        # Draw Curve TP Callouts using unified target_x & y_jog
        for item in curve_tp_items:
            v_x = item["v_x"]
            target_x = item.get("target_x", v_x)
            y_jog = item.get("y_jog", top_grid_y - self.ordinate_jog_depth_mm)
            self._draw_cranked_ordinate_line(
                msp=msp,
                v_x=v_x,
                target_x=target_x,
                y_start=y_datum,
                top_grid_y=top_grid_y,
                y_jog=y_jog,
                layer=item["line_lay"],
                color=item["line_col"],
                lineweight=item["line_lw"],
            )
            if item["callout_text"]:
                _draw_gradient_post_callout(target_x, item["callout_text"], item["col"], item["lay"])

        # 4E. Surface Profile Bridge Callouts (4-Line 45-Degree Dual-Color Callouts)
        if self.show_profile_bridges:
            self._draw_surface_profile_bridges(
                msp=msp,
                profile_df=profile_df,
                min_ch=min_ch,
                span_m=span_m,
                scale_x=scale_x,
                scale_y=scale_y,
                x_origin=x_origin,
                y_datum=y_datum,
                datum_m=datum_m,
                top_grid_y=top_grid_y,
                bridges_df=active_bridges_df,
                resolved_bridges=sheet_bridges if sheet_bridges else None,
            )

        # 4. Draw Band Rows, Header Boxes, and Multi-Line Centered Header Titles
        band_y_curr = y_datum
        band_bounds: list[tuple[ProfileBandDef, float, float]] = []

        header_x0 = max(0.0, x_origin - self.header_width_mm)
        header_x1 = x_origin
        header_mid_x = (header_x0 + header_x1) / 2.0

        # Horizontal top boundary line of header box at y_datum connecting to datum line
        msp.add_line(
            (header_x0, y_datum),
            (header_x1, y_datum),
            dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.col_border, "lineweight": 25},
        )

        for b_def in active_band_defs:
            top_y = band_y_curr
            bot_y = band_y_curr - b_def.height_mm
            band_bounds.append((b_def, top_y, bot_y))

            # Horizontal boundary line below this band (unless omitted, e.g. between EXT & PROP alignment)
            if not b_def.omit_bottom_border:
                msp.add_line(
                    (header_x0, bot_y),
                    (x_end_grid, bot_y),
                    dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.col_border, "lineweight": 25},
                )

            # Left vertical header border
            msp.add_line((header_x0, top_y), (header_x0, bot_y), dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.col_border})
            # Header-column separator vertical line
            msp.add_line((header_x1, top_y), (header_x1, bot_y), dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.col_border, "lineweight": 25})

            # Render centered header text (single line or two lines)
            mid_y = (top_y + bot_y) / 2.0
            h_col = b_def.header_color
            layer_hdr = "ESP_PROF_TEXT_PROP" if h_col == self.col_text_prop else "ESP_PROF_TEXT_EX"

            raw_lines = b_def.lines if b_def.lines else [b_def.title]
            lines: list[str] = []
            if getattr(b_def, "wrap_text", self.wrap_band_text):
                usable_w = max(20.0, self.header_width_mm - 8.0)
                char_w = max(1.2, 0.78 * self.header_text_height_mm)
                max_chars = max(18, int(usable_w / char_w))
                for rl in raw_lines:
                    lines.extend(wrap_band_header_text(rl, max_chars=max_chars))
            else:
                lines = list(raw_lines)

            if lines:
                n_lines = len(lines)
                b_h = b_def.height_mm
                base_text_h = self.header_text_height_mm

                # Ensure wrapped lines fit vertically within band height with comfortable margins
                max_allowed_total_h = b_h * 0.82
                needed_total_h = base_text_h * (1.0 + (n_lines - 1) * 1.3)
                if needed_total_h > max_allowed_total_h:
                    effective_text_h = max(1.3, max_allowed_total_h / (1.0 + (n_lines - 1) * 1.3))
                else:
                    effective_text_h = base_text_h

                pitch = effective_text_h * 1.3
                total_h = (n_lines - 1) * pitch
                y_start_line = mid_y + (total_h / 2.0)
                for l_idx, l_text in enumerate(lines):
                    line_y = y_start_line - l_idx * pitch
                    # Check for split-color "CUT & FILL" line
                    if l_text == "CUT & FILL" and len(b_def.line_colors) >= 2:
                        cut_col = b_def.line_colors[0]
                        fill_col = b_def.line_colors[1]
                        msp.add_text(
                            "CUT ",
                            dxfattribs={
                                "layer": "ESP_PROF_TEXT_EX" if cut_col == self.col_text_ex else "ESP_PROF_TEXT_PROP",
                                "color": cut_col,
                                "style": "ESP_STYLE",
                                "height": effective_text_h,
                            },
                        ).set_placement((header_mid_x - 1.0, line_y), align=TextEntityAlignment.MIDDLE_RIGHT)
                        msp.add_text(
                            "& FILL",
                            dxfattribs={
                                "layer": "ESP_PROF_TEXT_PROP" if fill_col == self.col_text_prop else "ESP_PROF_TEXT_EX",
                                "color": fill_col,
                                "style": "ESP_STYLE",
                                "height": effective_text_h,
                            },
                        ).set_placement((header_mid_x + 1.0, line_y), align=TextEntityAlignment.MIDDLE_LEFT)
                    else:
                        line_col = (
                            b_def.line_colors[l_idx]
                            if b_def.line_colors and l_idx < len(b_def.line_colors)
                            else h_col
                        )
                        layer_line = "ESP_PROF_TEXT_PROP" if line_col == self.col_text_prop else "ESP_PROF_TEXT_EX"
                        msp.add_text(
                            l_text,
                            dxfattribs={
                                "layer": layer_line,
                                "color": line_col,
                                "style": "ESP_STYLE",
                                "height": effective_text_h,
                            },
                        ).set_placement((header_mid_x, line_y), align=TextEntityAlignment.MIDDLE_CENTER)

            band_y_curr = bot_y

        bottom_band_y = band_y_curr

        # Right vertical border enclosing all bands
        msp.add_line((x_end_grid, y_datum), (x_end_grid, bottom_band_y), dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.col_border})

        # 5. Draw Station Columns & Populate 90-Degree Rotated Values
        # Precompute vertical line spans (omitting gradient bands and/or alignment bands if configured)
        y_grid_top = y_datum + (max_elev - datum_m + elev_step) * scale_y

        # Profile grid vertical line span (above datum)
        grid_line_spans: list[tuple[float, float]] = [(y_grid_top, y_datum)]

        # Band column vertical divider line spans (inside data bands)
        raw_band_spans: list[tuple[float, float]] = []
        for b_def, top_y, bot_y in band_bounds:
            if b_def.field_name in ("prop_gradient", "exg_gradient") or b_def.format_type == "gradient_span":
                if not self.show_gradient_band_grid_lines:
                    continue
            elif b_def.field_name in ("ext_alignment", "prop_alignment") or b_def.format_type in ("empty", "alignment_band"):
                if not self.show_alignment_band_grid_lines:
                    continue
            elif b_def.field_name == "km_stone" or b_def.format_type == "km_stone":
                continue
            raw_band_spans.append((top_y, bot_y))

        # Merge contiguous vertical spans for data bands
        band_line_spans: list[tuple[float, float]] = []
        for s_top, s_bot in raw_band_spans:
            if band_line_spans and abs(band_line_spans[-1][1] - s_top) < 1e-3:
                band_line_spans[-1] = (band_line_spans[-1][0], s_bot)
            else:
                band_line_spans.append((s_top, s_bot))

        divider_ys = sorted(list({y for span in raw_band_spans for y in span}), reverse=True)
        max_data_y = max(divider_ys) if divider_ys else y_datum
        min_data_y = min(divider_ys) if divider_ys else bottom_band_y
        half_t = self.band_tick_length_mm / 2.0

        n_stns = len(profile_df)
        for idx, (_, stn) in enumerate(profile_df.iterrows()):
            ch = float(stn["chainage_m"])
            sx = x_origin + (ch - min_ch) * scale_x

            # Determine column boundary step for centering
            if idx + 1 < n_stns:
                next_ch = float(profile_df.iloc[idx + 1]["chainage_m"])
                step_w = (next_ch - ch) * scale_x
            elif idx > 0:
                prev_ch = float(profile_df.iloc[idx - 1]["chainage_m"])
                step_w = (ch - prev_ch) * scale_x
            else:
                step_w = self.chainage_interval_m * scale_x

            # Draw profile grid vertical lines above datum (only if full vertical grid is requested)
            if self.show_vertical_grid:
                for g_top, g_bot in grid_line_spans:
                    msp.add_line(
                        (sx, g_top),
                        (sx, g_bot),
                        dxfattribs={"layer": "ESP_PROF_GRID", "color": self.col_grid, "lineweight": 9},
                    )

            # Draw band vertical station divider lines inside data bands (ticks or full vertical lines)
            if self.band_divider_style == "ticks":
                if sx > x_origin + 1e-3:
                    for y_div in divider_ys:
                        if abs(y_div - max_data_y) < 1e-3:
                            y1, y2 = y_div, y_div - half_t
                        elif abs(y_div - min_data_y) < 1e-3:
                            y1, y2 = y_div, y_div + half_t
                        else:
                            y1, y2 = y_div - half_t, y_div + half_t
                        msp.add_line(
                            (sx, y1),
                            (sx, y2),
                            dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.col_border, "lineweight": 15},
                        )
            else:
                for b_top, b_bot in band_line_spans:
                    msp.add_line(
                        (sx, b_top),
                        (sx, b_bot),
                        dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.col_border, "lineweight": 15},
                    )

            # Position text directly attached to the vertical station line on the right side of the column
            text_x = sx + step_w - self.band_text_offset_mm

            # Populate each band cell
            for b_def, top_y, bot_y in band_bounds:
                if b_def.format_type in ("gradient_span", "empty", "none", "alignment_band", "km_stone"):
                    continue

                val = stn.get(b_def.field_name)
                if val is None or pd.isna(val):
                    continue

                prec = getattr(b_def, "precision", None)
                signed = getattr(b_def, "signed", False)

                b_ident = (
                    str(getattr(b_def, "id", "") or "") + " "
                    + str(getattr(b_def, "field_name", "") or "") + " "
                    + str(getattr(b_def, "title", "") or "") + " "
                    + str(getattr(b_def, "source_column", "") or "")
                ).lower()
                b_words = set(b_ident.replace("(", " ").replace(")", " ").replace(".", " ").replace("-", " ").replace("+", " ").split())
                is_ch_band = (
                    any(k in b_ident for k in ("chainage", "ch_m", "stn_m", "ch-", "ch+"))
                    or getattr(b_def, "id", "").lower() in ("ch", "chainage", "chainage_m", "chainage_raw", "km")
                    or (b_def.field_name or "").lower() in ("ch", "chainage", "chainage_m", "chainage_raw", "km", "station", "stn")
                    or (b_def.format_type or "").lower().startswith("chainage")
                    or bool(b_words & {"ch", "km", "station", "stn"})
                )

                if is_ch_band:
                    fmt_type = (getattr(b_def, "format_type", "") or "").lower()
                    if fmt_type in ("integer", "int", "integer_m"):
                        val_str = str(int(round(ch)))
                    elif fmt_type == "chainage_2dp":
                        val_str = f"{ch / 1000.0:.2f}"
                    elif fmt_type in ("text", "string") and val is not None and str(val).strip() and str(val).lower() != "nan":
                        val_str = str(val).strip()
                    else:
                        from utils.geo_utils import format_chainage
                        val_str = format_chainage(ch, self.chainage_format)
                elif b_def.format_type in ("text", "string"):
                    val_str = str(val).strip()
                else:
                    try:
                        fv = float(val)
                        if math.isnan(fv):
                            continue
                    except (ValueError, TypeError):
                        continue

                    # Format according to format selector
                    if b_def.format_type == "signed_3dp" or (signed and prec is not None) or (signed and b_def.format_type != "unsigned_3dp"):
                        p = prec if prec is not None else 3
                        val_str = f"{fv:+.{p}f}"
                    elif b_def.format_type in ("unsigned_3dp", "numeric", "number"):
                        p = prec if prec is not None else 3
                        val_str = f"{fv:.{p}f}"
                    elif prec is not None:
                        val_str = f"{fv:.{prec}f}"
                    else:
                        val_str = str(val)

                if not val_str:
                    continue

                cell_mid_y = (top_y + bot_y) / 2.0
                v_col = b_def.value_color
                if b_def.field_name == "cut_fill_m" and fv is not None:
                    v_col = self.col_text_ex if fv < 0 else self.col_text_prop
                layer_val = "ESP_PROF_TEXT_PROP" if v_col == self.col_text_prop else "ESP_PROF_TEXT_EX"

                # 90 degrees rotated text attached directly to vertical station line
                msp.add_text(
                    val_str,
                    dxfattribs={
                        "layer": layer_val,
                        "color": v_col,
                        "style": "ESP_STYLE",
                        "height": self.band_text_height_mm,
                        "rotation": 90.0,
                    },
                ).set_placement((text_x, cell_mid_y), align=TextEntityAlignment.BOTTOM_CENTER)

        # Draw final vertical column closing line at the end
        if n_stns > 0:
            last_ch = float(profile_df.iloc[-1]["chainage_m"])
            last_sx = x_origin + (last_ch - min_ch) * scale_x
            if n_stns > 1:
                prev_ch = float(profile_df.iloc[-2]["chainage_m"])
                step_w = (last_ch - prev_ch) * scale_x
            else:
                step_w = self.chainage_interval_m * scale_x
            end_col_x = last_sx + step_w
            for b_top, b_bot in band_line_spans:
                msp.add_line(
                    (end_col_x, b_top),
                    (end_col_x, b_bot),
                    dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.col_border, "lineweight": 15},
                )
            if abs(end_col_x - x_end_grid) > 1e-3:
                msp.add_line(
                    (end_col_x, y_datum),
                    (end_col_x, bottom_band_y),
                    dxfattribs={"layer": "ESP_PROF_BANDS", "color": self.col_border, "lineweight": 15},
                )

        # 6. Render Alignment & Gradient Bands & KM Stone Band
        col_step = self.chainage_interval_m * scale_x
        for b_def, top_y, bot_y in band_bounds:
            if b_def.field_name == "prop_alignment":
                self._draw_profile_alignment_band(
                    msp=msp,
                    curves=active_curves or [],
                    top_y=top_y,
                    bot_y=bot_y,
                    x_origin=x_origin,
                    min_ch=min_ch,
                    total_length_m=span_m,
                    scale_x=scale_x,
                    color=self.col_text_prop,
                    layer="ESP_PROF_TEXT_PROP",
                    line_filter="PRO",
                    col_step=col_step,
                )
            elif b_def.field_name == "ext_alignment":
                self._draw_profile_alignment_band(
                    msp=msp,
                    curves=active_ext_curves or [],
                    top_y=top_y,
                    bot_y=bot_y,
                    x_origin=x_origin,
                    min_ch=min_ch,
                    total_length_m=span_m,
                    scale_x=scale_x,
                    color=self.col_text_ex,
                    layer="ESP_PROF_TEXT_EX",
                    line_filter="EXT",
                    col_step=col_step,
                )
            elif b_def.field_name == "prop_gradient" and prop_gradient_spans:
                self._draw_gradient_band(
                    msp=msp,
                    spans=prop_gradient_spans,
                    top_y=top_y,
                    bot_y=bot_y,
                    x_origin=x_origin,
                    min_ch=min_ch,
                    total_length_m=span_m,
                    scale_x=scale_x,
                    color=self.col_text_prop,
                    layer="ESP_PROF_TEXT_PROP",
                    col_step=col_step,
                )
            elif b_def.field_name == "exg_gradient" and exg_gradient_spans:
                self._draw_gradient_band(
                    msp=msp,
                    spans=exg_gradient_spans,
                    top_y=top_y,
                    bot_y=bot_y,
                    x_origin=x_origin,
                    min_ch=min_ch,
                    total_length_m=span_m,
                    scale_x=scale_x,
                    color=self.col_text_ex,
                    layer="ESP_PROF_TEXT_EX",
                    col_step=col_step,
                )
            elif b_def.field_name == "prop_gradient_diagram" and prop_gradient_spans:
                self._draw_gradient_diagram_band(
                    msp=msp,
                    spans=prop_gradient_spans,
                    top_y=top_y,
                    bot_y=bot_y,
                    x_origin=x_origin,
                    min_ch=min_ch,
                    total_length_m=span_m,
                    scale_x=scale_x,
                    color=self.col_text_prop,
                    layer="ESP_PROF_TEXT_PROP",
                    col_step=col_step,
                )
            elif b_def.field_name == "km_stone" or b_def.format_type == "km_stone":
                self._draw_km_stone_band(
                    msp=msp,
                    top_y=top_y,
                    bot_y=bot_y,
                    x_origin=x_origin,
                    min_ch=min_ch,
                    total_length_m=span_m,
                    scale_x=scale_x,
                    color=self.col_text_prop,
                    layer="ESP_PROF_TEXT_PROP",
                    col_step=col_step,
                )

        return datum_m

    def _draw_km_stone_band(
        self,
        msp: Any,
        top_y: float,
        bot_y: float,
        x_origin: float,
        min_ch: float,
        total_length_m: float,
        scale_x: float,
        color: int,
        layer: str,
        col_step: Optional[float] = None,
    ) -> None:
        """Render official IR pentagonal KM stones and kilometer labels in Band 13 (KILOMETRE)."""
        offset_x = (self.chainage_interval_m * scale_x) if col_step is None else col_step
        max_ch = min_ch + total_length_m

        stone_entries: list[tuple[float, str]] = []
        if min_ch < 0:
            if min_ch - 1.0 <= -871.0 <= max_ch + 1.0:
                stone_entries.append((-871.0, "167/849"))
            if min_ch - 1.0 <= -720.0 <= max_ch + 1.0:
                stone_entries.append((-720.0, "168/000"))

        k_start = max(0, int(math.floor(min_ch / 1000.0)))
        k_end = int(math.ceil(max_ch / 1000.0))
        for k in range(k_start, k_end + 1):
            km_ch = k * 1000.0
            if min_ch - 1.0 <= km_ch <= max_ch + 1.0:
                stone_entries.append((km_ch, f"{k}/000"))

        for km_ch, km_label in stone_entries:
            kx = x_origin + (km_ch - min_ch) * scale_x + offset_x

            # 1. Vertical stem from top of band down to roof apex:
            stem_bot_y = top_y - 5.82
            msp.add_line(
                (kx, top_y),
                (kx, stem_bot_y),
                dxfattribs={"layer": layer, "color": color, "lineweight": 18},
            )

            # 2. Triangular roof of KM stone (45-degree slope):
            roof_bot_y = top_y - 9.76
            left_eave_x = kx - 3.94
            right_eave_x = kx + 3.94
            msp.add_line(
                (kx, stem_bot_y),
                (left_eave_x, roof_bot_y),
                dxfattribs={"layer": layer, "color": color, "lineweight": 18},
            )
            msp.add_line(
                (kx, stem_bot_y),
                (right_eave_x, roof_bot_y),
                dxfattribs={"layer": layer, "color": color, "lineweight": 18},
            )
            msp.add_line(
                (left_eave_x, roof_bot_y),
                (right_eave_x, roof_bot_y),
                dxfattribs={"layer": layer, "color": color, "lineweight": 18},
            )

            # 3. KM stone body:
            body_bot_y = top_y - 11.98
            body_left_x = kx - 1.97
            body_right_x = kx + 1.97
            msp.add_line(
                (body_left_x, roof_bot_y),
                (body_left_x, body_bot_y),
                dxfattribs={"layer": layer, "color": color, "lineweight": 18},
            )
            msp.add_line(
                (body_right_x, roof_bot_y),
                (body_right_x, body_bot_y),
                dxfattribs={"layer": layer, "color": color, "lineweight": 18},
            )
            msp.add_line(
                (body_left_x, body_bot_y),
                (body_right_x, body_bot_y),
                dxfattribs={"layer": layer, "color": color, "lineweight": 18},
            )
            # Center vertical line inside stone:
            msp.add_line(
                (kx, stem_bot_y),
                (kx, body_bot_y),
                dxfattribs={"layer": layer, "color": color, "lineweight": 18},
            )

            # 3b. Counterchange (checkerboard) diagonal hatching per Indian Railways standard:
            # - Right half of triangular roof is hatched with 45° diagonal lines (ANSI31)
            # - Left half of rectangular body is hatched with 45° diagonal lines (ANSI31)
            # - Left half of roof and right half of body remain unhatched
            try:
                # Right roof triangle hatch:
                h_roof = msp.add_hatch(color=color, dxfattribs={"layer": layer})
                if self.km_stone_hatch_pattern.upper() == "SOLID":
                    h_roof.set_solid_fill(color=color)
                else:
                    h_roof.set_pattern_fill(
                        self.km_stone_hatch_pattern,
                        color=color,
                        scale=self.km_stone_hatch_scale,
                        angle=self.km_stone_hatch_angle,
                    )
                h_roof.paths.add_polyline_path(
                    [(kx, stem_bot_y), (right_eave_x, roof_bot_y), (kx, roof_bot_y)],
                    is_closed=True,
                )

                # Left body rectangle hatch:
                h_body = msp.add_hatch(color=color, dxfattribs={"layer": layer})
                if self.km_stone_hatch_pattern.upper() == "SOLID":
                    h_body.set_solid_fill(color=color)
                else:
                    h_body.set_pattern_fill(
                        self.km_stone_hatch_pattern,
                        color=color,
                        scale=self.km_stone_hatch_scale,
                        angle=self.km_stone_hatch_angle,
                    )
                h_body.paths.add_polyline_path(
                    [
                        (kx, roof_bot_y),
                        (body_left_x, roof_bot_y),
                        (body_left_x, body_bot_y),
                        (kx, body_bot_y),
                    ],
                    is_closed=True,
                )
            except Exception as e:
                logger.warning("Failed to render KM stone counterchange hatch at %.1f: %s", km_ch, e)

            # 4. Text below stone:
            if self.template_name in ("usbr_qazigund", "usbr_qazigund_13band") or min_ch < 0:
                msp.add_text(
                    km_label,
                    dxfattribs={
                        "layer": layer,
                        "color": color,
                        "style": "ESP_STYLE",
                        "height": 1.9,
                    },
                ).set_placement((kx, bot_y + 3.0), align=TextEntityAlignment.BOTTOM_CENTER)
                msp.add_text(
                    "KM",
                    dxfattribs={
                        "layer": layer,
                        "color": color,
                        "style": "ESP_STYLE",
                        "height": 1.9,
                    },
                ).set_placement((kx, bot_y + 0.8), align=TextEntityAlignment.BOTTOM_CENTER)
            else:
                text_y = (bot_y + body_bot_y) / 2.0
                msp.add_text(
                    f"{km_ch / 1000.0:.2f} KM",
                    dxfattribs={
                        "layer": layer,
                        "color": color,
                        "style": "ESP_STYLE",
                        "height": self.km_stone_text_height_profile,
                    },
                ).set_placement((kx, text_y), align=TextEntityAlignment.MIDDLE_CENTER)

    def _draw_gradient_band(
        self,
        msp: Any,
        spans: Sequence[dict[str, Any]],
        top_y: float,
        bot_y: float,
        x_origin: float,
        min_ch: float,
        total_length_m: float,
        scale_x: float,
        color: int,
        layer: str,
        col_step: Optional[float] = None,
    ) -> None:
        """Render official IR gradient band with continuous dimension lines, arrows, and PVI diamond tags."""
        offset_x = (self.chainage_interval_m * scale_x) if col_step is None else col_step
        cy = (top_y + bot_y) / 2.0
        max_ch = min_ch + total_length_m
        tag_half_w = 1.8
        bevel_h = 2.5

        drawn_pvi_tags: list[float] = []

        def _draw_tag(pvi_x: float, elev: Optional[float]):
            if elev is None or math.isnan(elev):
                return
            if any(abs(pvi_x - k) < 2.0 for k in drawn_pvi_tags):
                return
            drawn_pvi_tags.append(pvi_x)

            # Draw vertical elongated diamond / hexagon touching top and bottom band boundaries
            p1 = (pvi_x, top_y)
            p2 = (pvi_x + tag_half_w, top_y - bevel_h)
            p3 = (pvi_x + tag_half_w, bot_y + bevel_h)
            p4 = (pvi_x, bot_y)
            p5 = (pvi_x - tag_half_w, bot_y + bevel_h)
            p6 = (pvi_x - tag_half_w, top_y - bevel_h)
            msp.add_lwpolyline(
                [p1, p2, p3, p4, p5, p6, p1],
                dxfattribs={"layer": layer, "color": color, "lineweight": 18},
            )
            # Text rotated 90 degrees inside tag
            msp.add_text(
                f"{float(elev):.3f}",
                dxfattribs={
                    "layer": layer,
                    "color": color,
                    "style": "ESP_STYLE",
                    "height": 1.9,
                    "rotation": 90.0,
                },
            ).set_placement((pvi_x, cy), align=TextEntityAlignment.MIDDLE_CENTER)

        for span in spans:
            s_ch = float(span.get("start_ch", 0.0))
            e_ch = float(span.get("end_ch", 0.0))
            if e_ch <= min_ch or s_ch >= max_ch:
                continue

            # Clip span to sheet range
            act_s_ch = max(s_ch, min_ch)
            act_e_ch = min(e_ch, max_ch)

            x1 = x_origin + (act_s_ch - min_ch) * scale_x + (offset_x if act_s_ch > min_ch else 0.0)
            x2 = x_origin + (act_e_ch - min_ch) * scale_x + offset_x

            # Draw diamond tags at PVI boundaries
            if s_ch > min_ch:
                _draw_tag(x1, span.get("start_el"))
            if e_ch < max_ch:
                _draw_tag(x2, span.get("end_el"))

            # Calculate line bounds avoiding the diamond tags
            line_start_x = x1 + (tag_half_w if s_ch > min_ch else 0.0)
            line_end_x = x2 - (tag_half_w if e_ch < max_ch else 0.0)

            if line_end_x > line_start_x + 1.0:
                # Continuous horizontal dimension line
                msp.add_line(
                    (line_start_x, cy),
                    (line_end_x, cy),
                    dxfattribs={"layer": layer, "color": color, "lineweight": 18},
                )
                span_len_mm = line_end_x - line_start_x
                arrow_len = min(2.8, span_len_mm * 0.25)
                arrow_w = 0.65

                # Left solid filled arrow pointing left towards boundary
                p_arrow_l = [
                    (line_start_x, cy),
                    (line_start_x + arrow_len, cy + arrow_w),
                    (line_start_x + arrow_len, cy - arrow_w),
                ]
                msp.add_lwpolyline(
                    p_arrow_l + [(line_start_x, cy)],
                    dxfattribs={"layer": layer, "color": color, "lineweight": 18},
                )
                msp.add_solid(
                    [p_arrow_l[0], p_arrow_l[1], p_arrow_l[2], p_arrow_l[2]],
                    dxfattribs={"layer": layer, "color": color},
                )
                try:
                    hatch_l = msp.add_hatch(color=color, dxfattribs={"layer": layer})
                    hatch_l.set_solid_fill(color=color)
                    hatch_l.paths.add_polyline_path(p_arrow_l + [(line_start_x, cy)], is_closed=True)
                except Exception:
                    pass

                # Right solid filled arrow pointing right towards boundary
                p_arrow_r = [
                    (line_end_x, cy),
                    (line_end_x - arrow_len, cy + arrow_w),
                    (line_end_x - arrow_len, cy - arrow_w),
                ]
                msp.add_lwpolyline(
                    p_arrow_r + [(line_end_x, cy)],
                    dxfattribs={"layer": layer, "color": color, "lineweight": 18},
                )
                msp.add_solid(
                    [p_arrow_r[0], p_arrow_r[1], p_arrow_r[2], p_arrow_r[2]],
                    dxfattribs={"layer": layer, "color": color},
                )
                try:
                    hatch_r = msp.add_hatch(color=color, dxfattribs={"layer": layer})
                    hatch_r.set_solid_fill(color=color)
                    hatch_r.paths.add_polyline_path(p_arrow_r + [(line_end_x, cy)], is_closed=True)
                except Exception:
                    pass

                # Format text
                dir_str = str(span.get("direction", "LEVEL")).upper()
                g_raw = str(span.get("gradient", "LEVEL")).upper()
                if "1:" in g_raw:
                    g_text = f"1 IN {g_raw.split(':')[1].strip()}"
                elif "1 IN" in g_raw:
                    g_text = g_raw
                elif g_raw in ("LEVEL", "HORIZONTAL"):
                    g_text = "LEVEL"
                else:
                    g_text = g_raw

                mid_x = (line_start_x + line_end_x) / 2.0
                if span_len_mm >= 12.0:
                    msp.add_text(
                        dir_str,
                        dxfattribs={"layer": layer, "color": color, "style": "ESP_STYLE", "height": 2.1},
                    ).set_placement((mid_x, cy + 1.2), align=TextEntityAlignment.BOTTOM_CENTER)
                    msp.add_text(
                        g_text,
                        dxfattribs={"layer": layer, "color": color, "style": "ESP_STYLE", "height": 2.1},
                    ).set_placement((mid_x, cy - 1.2), align=TextEntityAlignment.TOP_CENTER)
                elif span_len_mm >= 6.0:
                    msp.add_text(
                        f"{dir_str} {g_text}",
                        dxfattribs={"layer": layer, "color": color, "style": "ESP_STYLE", "height": 1.6},
                    ).set_placement((mid_x, cy + 1.0), align=TextEntityAlignment.BOTTOM_CENTER)

    def _extract_gradient_spans_from_dataframe(
        self,
        df: pd.DataFrame,
        grad_col: str = "proposed_gradient",
        el_col: str = "prl_m",
    ) -> list[dict[str, Any]]:
        """Extract discrete tangent gradient spans from a profile dataframe containing gradient values."""
        if grad_col not in df.columns:
            return []
        spans: list[dict[str, Any]] = []
        cur_g = None
        s_ch = None
        s_el = None
        prev_ch = None
        prev_el = None
        for _, row in df.iterrows():
            ch = row.get("chainage_m")
            g = row.get(grad_col)
            el = row.get(el_col)
            if pd.isna(ch) or pd.isna(g):
                continue
            try:
                ch = float(ch)
                g = float(g)
                el = float(el) if pd.notna(el) else None
            except (ValueError, TypeError):
                continue

            if cur_g is None:
                cur_g = g
                s_ch = ch
                s_el = el
            elif abs(g - cur_g) > 1e-4:
                dir_str = "FALL" if cur_g < 0 else ("RISING" if cur_g > 0 else "LEVEL")
                denom = abs(round(cur_g, 3))
                g_str = f"1 IN {denom:.3f}" if denom != 0 else "LEVEL"
                spans.append({
                    "start_ch": s_ch,
                    "end_ch": prev_ch,
                    "start_el": s_el,
                    "end_el": prev_el,
                    "gradient": g_str,
                    "direction": dir_str,
                })
                cur_g = g
                s_ch = prev_ch if prev_ch is not None else ch
                s_el = prev_el if prev_el is not None else el
            prev_ch = ch
            prev_el = el

        if cur_g is not None and prev_ch is not None and s_ch is not None and prev_ch > s_ch:
            dir_str = "FALL" if cur_g < 0 else ("RISING" if cur_g > 0 else "LEVEL")
            denom = abs(round(cur_g, 3))
            g_str = f"1 IN {denom:.3f}" if denom != 0 else "LEVEL"
            spans.append({
                "start_ch": s_ch,
                "end_ch": prev_ch,
                "start_el": s_el,
                "end_el": prev_el,
                "gradient": g_str,
                "direction": dir_str,
            })
        return spans

    def _draw_gradient_diagram_band(
        self,
        msp: Any,
        spans: Sequence[dict[str, Any]],
        top_y: float,
        bot_y: float,
        x_origin: float,
        min_ch: float,
        total_length_m: float,
        scale_x: float,
        color: int,
        layer: str,
        col_step: Optional[float] = None,
    ) -> None:
        """Render schematic gradient slope line diagram with length L and grade G% inside profile band."""
        offset_x = (self.chainage_interval_m * scale_x) if col_step is None else col_step
        max_ch = min_ch + total_length_m
        cy = (top_y + bot_y) / 2.0
        h_margin = (top_y - bot_y) * 0.22

        for span in spans:
            s_ch = float(span.get("start_ch", 0.0))
            e_ch = float(span.get("end_ch", 0.0))
            if e_ch <= min_ch or s_ch >= max_ch:
                continue

            act_s_ch = max(s_ch, min_ch)
            act_e_ch = min(e_ch, max_ch)
            span_len_m = e_ch - s_ch

            x1 = x_origin + (act_s_ch - min_ch) * scale_x + (offset_x if act_s_ch > min_ch else 0.0)
            x2 = x_origin + (act_e_ch - min_ch) * scale_x + offset_x

            dir_str = str(span.get("direction", "LEVEL")).upper()
            g_str = str(span.get("gradient", "LEVEL")).upper()
            denom = 400.0
            if "1 IN" in g_str:
                try:
                    denom = float(g_str.split("1 IN")[1].strip())
                except Exception:
                    denom = 400.0

            grade_pct = (100.0 / denom) if denom != 0 else 0.0
            if dir_str == "FALL":
                y1 = top_y - h_margin
                y2 = bot_y + h_margin
                g_display = f"G=-{grade_pct:.3f}%"
            elif dir_str == "RISING":
                y1 = bot_y + h_margin
                y2 = top_y - h_margin
                g_display = f"G=+{grade_pct:.3f}%"
            else:
                y1 = cy
                y2 = cy
                g_display = "LEVEL"

            # Draw sloped grade line
            msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": layer, "color": color, "lineweight": 25})

            # Draw vertical connector line at start of span
            if s_ch > min_ch:
                msp.add_line((x1, bot_y), (x1, top_y), dxfattribs={"layer": layer, "color": color, "lineweight": 13})

            mid_x = (x1 + x2) / 2.0
            mid_y_line = (y1 + y2) / 2.0
            msp.add_text(
                f"L={span_len_m:.2f}m",
                dxfattribs={"layer": layer, "color": color, "style": "ESP_STYLE", "height": 1.9},
            ).set_placement((mid_x, mid_y_line + 1.2), align=TextEntityAlignment.BOTTOM_CENTER)
            msp.add_text(
                g_display,
                dxfattribs={"layer": layer, "color": color, "style": "ESP_STYLE", "height": 1.9},
            ).set_placement((mid_x, mid_y_line - 1.2), align=TextEntityAlignment.TOP_CENTER)

    def _load_curve_details(self, excel_path: Union[str, Path]) -> list[dict[str, Any]]:
        """Load horizontal curve schedule from Excel workbook."""
        target_path = Path(excel_path)
        if not target_path.exists():
            return []

        try:
            excel_file = pd.ExcelFile(target_path)
            sheet_names = excel_file.sheet_names

            dir_map: dict[str, str] = {}
            for s in sheet_names:
                if "4th" in s.lower():
                    df_dir = pd.read_excel(target_path, sheet_name=s)
                    for _, r in df_dir.iterrows():
                        c_no = str(r.get("Curve No.", "")).strip()
                        if "LHS/RHS" in r and pd.notna(r["LHS/RHS"]):
                            dir_map[c_no] = str(r["LHS/RHS"]).strip().upper()

            all_curves: list[dict[str, Any]] = []
            sheet_configs: list[tuple[str, str]] = []
            for s in sheet_names:
                s_lower = s.lower()
                if "3rd" in s_lower:
                    sheet_configs.append((s, "PRO 3RD LINE"))
                elif "4th" in s_lower:
                    sheet_configs.append((s, "PRO 4TH LINE"))

            if not sheet_configs and sheet_names:
                sheet_configs.append((sheet_names[0], "PRO 3RD LINE"))

            for s_name, line_label in sheet_configs:
                df = pd.read_excel(target_path, sheet_name=s_name)
                cols = {str(c).strip().lower(): c for c in df.columns}
                c_no_col = next((cols[k] for k in cols if "curve no" in k or "sr" in k), None)
                tp1_col = next((cols[k] for k in cols if k == "tp1"), None)
                tp2_col = next((cols[k] for k in cols if k == "tp2"), None)
                rad_col = next((cols[k] for k in cols if "radius" in k), None)
                deg_col = next((cols[k] for k in cols if "degree" in k), None)
                cl_col = next((cols[k] for k in cols if "curve length" in k), None)
                tl_col = next((cols[k] for k in cols if "tangent" in k), None)
                delta_col = next((cols[k] for k in cols if "deflection" in k and ("∆" in k or "delta" in k or "(" in k)), None)
                if not delta_col:
                    delta_col = next((cols[k] for k in cols if "deflection" in k and "angle" in k), None)
                trl_col = next((cols[k] for k in cols if "trl" in k), None)
                se_col = next((cols[k] for k in cols if "cant actual" in k or "se" in k), None)
                speed_col = next((cols[k] for k in cols if "speed" in k and "max" in k), None)
                dir_col = next((cols[k] for k in cols if "lhs" in k or "rhs" in k or "dir" in k), None)

                if not (c_no_col and tp1_col and rad_col):
                    continue

                for _, r in df.iterrows():
                    c_no = str(r[c_no_col]).strip()
                    if not c_no or c_no.lower() in ("nan", "none", ""):
                        continue

                    direction = ""
                    if dir_col and pd.notna(r[dir_col]):
                        direction = str(r[dir_col]).strip().upper()
                    if not direction:
                        direction = dir_map.get(c_no, "RHS")

                    try:
                        radius = float(r[rad_col])
                        if radius <= 0:
                            continue
                        tp1 = float(r[tp1_col])
                        cl = float(r[cl_col]) if cl_col and pd.notna(r[cl_col]) else 0.0
                        tp2 = float(r[tp2_col]) if tp2_col and pd.notna(r[tp2_col]) else (tp1 + cl)
                        deg = float(r[deg_col]) if deg_col and pd.notna(r[deg_col]) else (1750.0 / radius)
                        tl = float(r[tl_col]) if tl_col and pd.notna(r[tl_col]) else 0.0

                        delta_val = r.get(delta_col) if delta_col else ""
                        if pd.notna(delta_val):
                            if isinstance(delta_val, (int, float)):
                                deg_val = int(delta_val)
                                rem_min = (float(delta_val) - deg_val) * 60.0
                                min_val = int(rem_min)
                                sec_val = (rem_min - min_val) * 60.0
                                delta_str = f"{deg_val}°{min_val:02d}'{sec_val:06.3f}\""
                            else:
                                delta_str = str(delta_val).strip().replace("''", '"').strip('"').replace("''", '"')
                                if delta_str and not delta_str.endswith('"'):
                                    delta_str += '"'
                        else:
                            delta_str = "0°00'00.000\""

                        trl = float(r[trl_col]) if trl_col and pd.notna(r[trl_col]) else 0.0
                        se = float(r[se_col]) if se_col and pd.notna(r[se_col]) else 0.0
                        speed = float(r[speed_col]) if speed_col and pd.notna(r[speed_col]) else 110.0
                        speed_val = int(round(speed / 5.0) * 5)

                        all_curves.append({
                            "curve_no": c_no,
                            "direction": direction,
                            "line_label": line_label,
                            "is_proposed": True,
                            "is_existing": False,
                            "delta_str": delta_str,
                            "degree": deg,
                            "radius": radius,
                            "tl": tl,
                            "cl": cl,
                            "trl": trl,
                            "se": se,
                            "speed": speed_val,
                            "tp1": tp1,
                            "tp2": tp2,
                            "apex_ch": tp1 + tl if tl > 0 else (tp1 + tp2) / 2.0,
                            "mid_ch": (tp1 + tp2) / 2.0,
                        })
                    except Exception:
                        pass

            return all_curves
        except Exception as e:
            self.logger.warning("Could not load curve details from %s: %s", target_path.name, e)
            return []

    def _load_existing_curve_details(self, excel_path: Union[str, Path]) -> list[dict[str, Any]]:
        """Load horizontal curve schedule for Existing track (DN and UP) from Excel/CSV workbook."""
        target_path = Path(excel_path)
        if not target_path.exists():
            return []

        try:
            def _clean_ch(val: Any) -> Optional[float]:
                if val is None or pd.isna(val):
                    return None
                if isinstance(val, (int, float)):
                    return float(val)
                s = str(val).strip()
                if not s:
                    return None
                parts = s.split(".")
                if len(parts) == 3:
                    try:
                        return float(parts[0]) * 1000.0 + float(f"{parts[1]}.{parts[2]}")
                    except ValueError:
                        pass
                try:
                    return float(s.replace(",", ""))
                except ValueError:
                    return None

            # Standard direction mapping for Secunderabad-Wadi corridor existing curves
            dir_lookup: dict[str, dict[str, str]] = {
                "41": {"UP": "LHS", "DN": "LHS"},
                "42": {"UP": "RHS", "DN": "RHS"},
                "43": {"UP": "RHS", "DN": "RHS"},
                "44": {"UP": "LHS", "DN": "LHS"},
                "45": {"UP": "RHS", "DN": "RHS"},
                "46": {"UP": "LHS", "DN": "LHS"},
                "47": {"UP": "RHS", "DN": "RHS"},
                "48": {"UP": "RHS", "DN": "RHS"},
                "49": {"UP": "RHS", "DN": "RHS"},
                "49-A": {"UP": "RHS", "DN": "RHS"},
                "50": {"UP": "RHS", "DN": "RHS"},
                "51": {"UP": "LHS", "DN": "LHS"},
                "52": {"UP": "LHS", "DN": "LHS"},
                "53": {"UP": "RHS", "DN": "RHS"},
                "54": {"UP": "LHS", "DN": "LHS"},
                "55": {"UP": "RHS", "DN": "RHS"},
                "56": {"UP": "RHS", "DN": "LHS"},
                "57": {"UP": "RHS", "DN": "RHS"},
                "58": {"UP": "LHS", "DN": "LHS"},
                "59": {"UP": "LHS", "DN": "RHS"},
                "60": {"UP": "LHS", "DN": "RHS"},
                "61": {"UP": "RHS", "DN": "RHS"},
                "62": {"UP": "RHS", "DN": "RHS"},
                "63": {"UP": "RHS", "DN": "RHS"},
                "64": {"UP": "LHS", "DN": "LHS"},
                "65": {"UP": "RHS", "DN": "RHS"},
                "66": {"UP": "LHS", "DN": "LHS"},
                "67": {"UP": "RHS", "DN": "RHS"},
                "68": {"UP": "LHS", "DN": "LHS"},
                "69": {"UP": "LHS", "DN": "LHS"},
                "70": {"UP": "LHS", "DN": "LHS"},
                "71": {"UP": "RHS", "DN": "RHS"},
                "72": {"UP": "RHS", "DN": "RHS"},
                "73": {"UP": "LHS", "DN": "LHS"},
                "74": {"UP": "LHS", "DN": "RHS"},
                "75": {"UP": "LHS", "DN": "LHS"},
            }

            sheet_dfs: list[tuple[str, pd.DataFrame]] = []
            if target_path.suffix.lower() == ".csv":
                sheet_dfs.append(("", pd.read_csv(target_path)))
            else:
                xl = pd.ExcelFile(target_path)
                for s_name in xl.sheet_names:
                    sheet_dfs.append((s_name, xl.parse(s_name)))

            curves: list[dict[str, Any]] = []
            for s_name, df in sheet_dfs:
                s_lower = s_name.lower()
                default_track = "UP" if "up" in s_lower else ("DN" if "dn" in s_lower else "TRACK")

                cols = {str(c).strip().lower(): c for c in df.columns}
                c_no_col = next((cols[k] for k in cols if "curve_no" in k or "curve no" in k or "sr" in k), None)
                tp1_col = next((cols[k] for k in cols if "tp1" in k), None)
                tp2_col = next((cols[k] for k in cols if "tp2" in k), None)
                rad_col = next((cols[k] for k in cols if "radius" in k), None)
                deg_col = next((cols[k] for k in cols if "degree" in k), None)
                cl_col = next((cols[k] for k in cols if "curve_length" in k or "curve length" in k), None)
                tl_col = next((cols[k] for k in cols if "tangent" in k), None)
                delta_col = next((cols[k] for k in cols if "deflection" in k and ("∆" in k or "delta" in k or "(" in k)), None)
                if not delta_col:
                    delta_col = next((cols[k] for k in cols if "deflection" in k or "delta" in k or "∆" in k), None)
                trl_col = next((cols[k] for k in cols if "transition" in k or "trl" in k), None)
                se_col = next((cols[k] for k in cols if "cant actual" in k or "cant" in k or "se" in k), None)
                speed_col = next((cols[k] for k in cols if "speed" in k and "max" in k), None)
                if not speed_col:
                    speed_col = next((cols[k] for k in cols if "speed" in k), None)
                hand_col = next((cols[k] for k in cols if "hand" in k or "dir" in k or "lhs" in k), None)
                line_col = next((cols[k] for k in cols if k == "line" or "track" in k), None)

                if not (c_no_col and rad_col and (tp1_col or tp2_col)):
                    continue

                for _, r in df.iterrows():
                    c_no = str(r[c_no_col]).strip()
                    if not c_no or c_no.lower() in ("nan", "none", ""):
                        continue

                    radius = float(r[rad_col]) if pd.notna(r[rad_col]) else 0.0
                    if radius <= 0:
                        continue

                    cl = float(r[cl_col]) if cl_col and pd.notna(r[cl_col]) else 0.0
                    tp1 = _clean_ch(r[tp1_col]) if tp1_col and pd.notna(r[tp1_col]) else None
                    tp2 = _clean_ch(r[tp2_col]) if tp2_col and pd.notna(r[tp2_col]) else None

                    if tp1 is None and tp2 is None:
                        continue
                    if tp1 is None:
                        tp1 = tp2 - cl
                    if tp2 is None:
                        tp2 = tp1 + cl
                    if tp2 < tp1:
                        tp1, tp2 = tp2, tp1

                    if line_col and pd.notna(r[line_col]):
                        trk_line = str(r[line_col]).strip().upper()
                    else:
                        trk_line = default_track
                    line_lbl = f"EXG {trk_line} LINE" if "EXG" not in trk_line else trk_line

                    if hand_col and pd.notna(r[hand_col]):
                        direction = str(r[hand_col]).strip().upper()
                    else:
                        direction = dir_lookup.get(c_no, {}).get(trk_line, "RHS")

                    tl = float(r[tl_col]) if tl_col and pd.notna(r[tl_col]) else 0.0
                    deg = float(r[deg_col]) if deg_col and pd.notna(r[deg_col]) else (1750.0 / radius)
                    delta_val = r.get(delta_col) if delta_col else ""
                    if pd.notna(delta_val):
                        if isinstance(delta_val, (int, float)):
                            deg_val = int(delta_val)
                            rem_min = (float(delta_val) - deg_val) * 60.0
                            min_val = int(rem_min)
                            sec_val = (rem_min - min_val) * 60.0
                            delta_str = f"{deg_val}°{min_val:02d}'{sec_val:06.3f}\""
                        else:
                            delta_str = str(delta_val).strip().replace("''", '"').strip('"').replace("''", '"')
                            if delta_str and not delta_str.endswith('"'):
                                delta_str += '"'
                    else:
                        delta_str = "0°00'00\""

                    trl = float(r[trl_col]) if trl_col and pd.notna(r[trl_col]) else 0.0
                    se = float(r[se_col]) if se_col and pd.notna(r[se_col]) else 0.0
                    speed = float(r[speed_col]) if speed_col and pd.notna(r[speed_col]) else 110.0

                    curves.append({
                        "curve_no": c_no,
                        "direction": direction,
                        "line_label": line_lbl,
                        "track": trk_line,
                        "is_proposed": False,
                        "is_existing": True,
                        "delta_str": delta_str,
                        "degree": deg,
                        "radius": radius,
                        "tl": tl,
                        "cl": cl,
                        "trl": trl,
                        "se": se,
                        "speed": int(round(speed / 5.0) * 5),
                        "tp1": tp1,
                        "tp2": tp2,
                        "apex_ch": tp1 + tl if tl > 0 else (tp1 + tp2) / 2.0,
                        "mid_ch": (tp1 + tp2) / 2.0,
                    })

            self.logger.info("Loaded %d existing curve detail records from %s", len(curves), target_path.name)
            return curves
        except Exception as e:
            self.logger.warning("Could not load existing curve details from %s: %s", target_path.name, e)
            return []

    def _draw_profile_alignment_band(
        self,
        msp: Any,
        curves: Sequence[dict[str, Any]],
        top_y: float,
        bot_y: float,
        x_origin: float,
        min_ch: float,
        total_length_m: float,
        scale_x: float,
        color: int,
        layer: str,
        line_filter: str = "PRO",
        col_step: Optional[float] = None,
    ) -> None:
        """Render official IR horizontal alignment band with dual track lines, curve deviations, and 10-line callout boxes."""
        offset_x = (self.chainage_interval_m * scale_x) if col_step is None else col_step
        max_ch = min_ch + total_length_m
        x_start = x_origin
        x_end = x_origin + total_length_m * scale_x + offset_x

        is_ext = (line_filter == "EXT")
        y_track_top = top_y - 6.5
        y_track_bot = top_y - 10.5

        sheet_curves = [
            c for c in curves
            if max(c.get("tp1", 0.0), min_ch) < min(c.get("tp2", 0.0), max_ch)
            or (min_ch <= c.get("mid_ch", 0.0) <= max_ch)
        ]

        def draw_single_track_line(y_base: float, trk_curves: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
            """Draw a continuous track line with circular arc for curves and straight tangents."""
            apex_points: dict[str, tuple[float, float]] = {}
            if not trk_curves:
                msp.add_line((x_start, y_base), (x_end, y_base), dxfattribs={"layer": "ESP_PROF_BANDS", "color": color, "lineweight": 25})
                return apex_points

            trk_curves.sort(key=lambda x: x.get("tp1", 0.0))
            curr_x = x_start
            for c in trk_curves:
                tp1 = float(c.get("tp1", 0.0))
                tp2 = float(c.get("tp2", 0.0))
                if tp2 <= tp1:
                    continue
                mid_ch = float(c.get("mid_ch", (tp1 + tp2) / 2.0))
                direction = str(c.get("direction", "RHS")).upper()
                is_rhs = ("R" in direction)
                dev_y = 3.0 if is_rhs else -3.0

                # Full curve coordinate limits in drawing mm
                x1_full = x_origin + (tp1 - min_ch) * scale_x
                x2_full = x_origin + (tp2 - min_ch) * scale_x

                # Check if curve intersects this sheet
                if x2_full <= x_start or x1_full >= x_end:
                    continue

                # Clamped endpoints for drawing within sheet
                x_s = max(x_start, min(x_end, x1_full))
                x_e = max(x_start, min(x_end, x2_full))

                # Tangent before curve
                if x_s > curr_x:
                    msp.add_line((curr_x, y_base), (x_s, y_base), dxfattribs={"layer": "ESP_PROF_BANDS", "color": color, "lineweight": 25})

                # Circular arc from starting chainage (tp1) to ending chainage (tp2)
                if x_e > x_s:
                    w = (x2_full - x1_full) / 2.0
                    cx = (x1_full + x2_full) / 2.0
                    h = abs(dev_y)
                    # 3-point circle radius: w^2 + (R - h)^2 = R^2 => R = (w^2 + h^2) / (2 * h)
                    radius = (w * w + h * h) / (2.0 * h)

                    if is_rhs:
                        # Upper arch: center is below y_base, apex at angle 90°
                        cy = y_base + dev_y - radius
                        dy_s = math.sqrt(max(0.0, radius * radius - (x_s - cx) ** 2))
                        a_s = math.degrees(math.atan2(dy_s, x_s - cx))
                        dy_e = math.sqrt(max(0.0, radius * radius - (x_e - cx) ** 2))
                        a_e = math.degrees(math.atan2(dy_e, x_e - cx))
                        # In AutoCAD, arcs are drawn CCW. Upper arch goes from a_e to a_s through 90°.
                        start_ang = a_e
                        end_ang = a_s
                    else:
                        # Lower arch: center is above y_base, apex at angle 270°
                        cy = y_base + dev_y + radius
                        dy_s = -math.sqrt(max(0.0, radius * radius - (x_s - cx) ** 2))
                        a_s = math.degrees(math.atan2(dy_s, x_s - cx)) % 360.0
                        dy_e = -math.sqrt(max(0.0, radius * radius - (x_e - cx) ** 2))
                        a_e = math.degrees(math.atan2(dy_e, x_e - cx)) % 360.0
                        # Lower arch goes CCW from a_s to a_e through 270°.
                        start_ang = a_s
                        end_ang = a_e

                    msp.add_arc(
                        center=(cx, cy),
                        radius=radius,
                        start_angle=start_ang,
                        end_angle=end_ang,
                        dxfattribs={"layer": "ESP_PROF_BANDS", "color": color, "lineweight": 25},
                    )

                x_mid = max(x_start, min(x_end, x_origin + (mid_ch - min_ch) * scale_x))
                c_key = f"{c.get('line_label', '')}_{c.get('curve_no', '')}"
                apex_points[c_key] = (x_mid, y_base + dev_y)
                curr_x = max(curr_x, x_e)

            if curr_x < x_end:
                msp.add_line((curr_x, y_base), (x_end, y_base), dxfattribs={"layer": "ESP_PROF_BANDS", "color": color, "lineweight": 25})
            return apex_points

        # Partition curves by track line
        if is_ext:
            curves_top = [c for c in sheet_curves if "DN" in str(c.get("line_label", "")).upper() or "DN" in str(c.get("track", "")).upper()]
            curves_bot = [c for c in sheet_curves if "UP" in str(c.get("line_label", "")).upper() or "UP" in str(c.get("track", "")).upper()]
        else:
            curves_top = [c for c in sheet_curves if "4TH" in str(c.get("line_label", "")).upper()]
            curves_bot = [c for c in sheet_curves if "3RD" in str(c.get("line_label", "")).upper() or "4TH" not in str(c.get("line_label", "")).upper()]

        apex_map_top = draw_single_track_line(y_track_top, curves_top)
        apex_map_bot = draw_single_track_line(y_track_bot, curves_bot)
        apex_map = {**apex_map_top, **apex_map_bot}

        if not sheet_curves:
            return

        # 2. Render 10-line curve parameter boxes directly below alignment
        # Box dimensions
        box_w = 34.0
        line_h = 2.1
        pad_x = 1.8
        pad_y = 1.4
        box_h = 10 * line_h + 2.0 * pad_y  # ~23.8 mm
        box_y_top = y_track_bot - 3.0       # Exactly aligns with bottom dip of LHS curves
        box_y_bot = box_y_top - box_h

        # Cluster curves by chainage (e.g. dual track curves at same station)
        clusters: list[list[dict[str, Any]]] = []
        for c in sorted(sheet_curves, key=lambda x: x.get("mid_ch", 0.0)):
            if not clusters:
                clusters.append([c])
            else:
                last_cluster = clusters[-1]
                if abs(c.get("mid_ch", 0.0) - last_cluster[0].get("mid_ch", 0.0)) < 120.0 and len(last_cluster) < 2:
                    last_cluster.append(c)
                else:
                    clusters.append([c])

        gap = 1.5
        cluster_data: list[dict[str, Any]] = []
        for cluster in clusters:
            if is_ext:
                cluster.sort(key=lambda x: 0 if "DN" in str(x.get("line_label", "")).upper() else 1)
            else:
                cluster.sort(key=lambda x: 0 if "3RD" in str(x.get("line_label", "")).upper() else 1)
            num_in_cluster = len(cluster)
            total_w = num_in_cluster * box_w + (num_in_cluster - 1) * gap
            avg_ch = sum(c.get("mid_ch", 0.0) for c in cluster) / num_in_cluster
            center_x = x_origin + (avg_ch - min_ch) * scale_x
            cluster_data.append({"cluster": cluster, "width": total_w, "center_x": center_x})

        # Global 1D collision relaxation between adjacent clusters to prevent overlap
        for _ in range(25):
            overlap_found = False
            for i in range(len(cluster_data) - 1):
                c_curr = cluster_data[i]
                c_next = cluster_data[i + 1]
                right_curr = c_curr["center_x"] + c_curr["width"] / 2.0
                left_next = c_next["center_x"] - c_next["width"] / 2.0
                if right_curr + gap > left_next:
                    overlap = (right_curr + gap) - left_next
                    shift = overlap / 2.0
                    c_curr["center_x"] -= shift
                    c_next["center_x"] += shift
                    overlap_found = True

            # Boundary enforcement
            if cluster_data:
                left_0 = cluster_data[0]["center_x"] - cluster_data[0]["width"] / 2.0
                if left_0 < x_origin + 1.0:
                    cluster_data[0]["center_x"] += (x_origin + 1.0 - left_0)
                right_last = cluster_data[-1]["center_x"] + cluster_data[-1]["width"] / 2.0
                if right_last > x_end - 1.0:
                    cluster_data[-1]["center_x"] -= (right_last - (x_end - 1.0))

            if not overlap_found:
                break

        for c_info in cluster_data:
            cluster = c_info["cluster"]
            w = c_info["width"]
            start_x = c_info["center_x"] - w / 2.0

            for idx, c in enumerate(cluster):
                bx0 = start_x + idx * (box_w + gap)
                bx1 = bx0 + box_w
                by0 = box_y_bot
                by1 = box_y_top

                # Rectangular callout boundary (no stem line)
                msp.add_lwpolyline(
                    [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)],
                    dxfattribs={"layer": "ESP_PROF_BANDS", "color": color, "lineweight": 18},
                    close=True,
                )

                # 10 lines of standard parameter text (no spaces around '=')
                r_val = float(c.get("radius", 0.0))
                r_str = f"{r_val:.2f}M" if (r_val != int(r_val)) else f"{int(r_val)}.00M"
                trl_val = float(c.get("trl", 0.0))
                trl_str = f"{int(round(trl_val))}M" if abs(trl_val - round(trl_val)) < 0.05 else f"{trl_val:.2f}M"

                c_prefix = "EXT. CURVE" if is_ext else "CURVE"
                default_lbl = "EXG UP LINE" if is_ext else "PRO 3RD LINE"
                delta_str = c.get("delta_str", '0°00\'00"')
                lines = [
                    f"{c_prefix} NO.{c.get('curve_no', '')}({c.get('direction', '')})",
                    f"{c.get('line_label', default_lbl)}",
                    f"Δ={delta_str}",
                    f"D={float(c.get('degree', 0.0)):.3f}°",
                    f"R={r_str}",
                    f"TL={float(c.get('tl', 0.0)):.3f}M",
                    f"CL={float(c.get('cl', 0.0)):.3f}M",
                    f"TRL={trl_str}",
                    f"SE={int(round(float(c.get('se', 0.0))))}MM",
                    f"SPEED={int(round(float(c.get('speed', 110.0))))}KMPH",
                ]

                y_text = by1 - pad_y - 0.2
                for line_txt in lines:
                    msp.add_text(
                        line_txt,
                        dxfattribs={
                            "height": 1.6,
                            "layer": layer,
                            "style": "ESP_STYLE",
                            "color": color,
                        },
                    ).set_placement((bx0 + pad_x, y_text), align=TextEntityAlignment.TOP_LEFT)
                    y_text -= line_h

    def _draw_surface_profile_bridges(
        self,
        msp: Any,
        profile_df: pd.DataFrame,
        min_ch: float,
        span_m: float,
        scale_x: float,
        scale_y: float,
        x_origin: float,
        y_datum: float,
        datum_m: float,
        top_grid_y: float,
        bridges_df: Optional[pd.DataFrame] = None,
        resolved_bridges: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        """Render 45-degree 4-line dual-color bridge callouts in the Surface Profile view.

        Indian Railways Standard Brownfield Profile Callout Format:
            Line 1 (White / ACI 7): CL OF EXIST. BR.NO: <NO>, AT CH: <CH>M, SPAN: <EXG_SPAN>,
            Line 2 (White / ACI 7): EXT.BL:<BL>M,EXT.FL:<FL>M,EXG.RL:<RL>M
            Line 3 (Red / ACI 1):   PROP.TO BE EXTENDED WITH SPAN:<PROP_SPAN>,
            Line 4 (Red / ACI 1):   PROP.RL:<RL>M,PROP.FL:<FL>M,PROP.BL:<BL>M

        Args:
            msp: ezdxf modelspace.
            profile_df: Profile DataFrame with chainage_m, egl_m, erl_m, prl_m, efl_m, pfl_m.
            min_ch: Minimum chainage of active sheet in meters.
            span_m: Span length of active sheet in meters.
            scale_x: Horizontal scale multiplier (e.g. 1000 / 5000 = 0.20).
            scale_y: Vertical scale multiplier (e.g. 1000 / 500 = 2.00).
            x_origin: X offset in mm for station 0 on sheet.
            y_datum: Y position in mm of profile datum line.
            datum_m: Datum elevation in meters.
            top_grid_y: Y position in mm of top grid boundary.
            bridges_df: Optional DataFrame containing bridge schedule records.
            resolved_bridges: Optional list of pre-deconflicted bridge dictionaries.

        Returns:
            Number of bridge callouts rendered.
        """
        if not self.show_profile_bridges:
            return 0

        # Auto-load bridge list from config if not explicitly provided
        if resolved_bridges is None and (bridges_df is None or (isinstance(bridges_df, pd.DataFrame) and bridges_df.empty)):
            configured_br_path = self.config.get("paths", {}).get("bridge_list_path")
            if configured_br_path:
                br_path = Path(configured_br_path)
                if not br_path.is_absolute():
                    br_path = _ROOT / br_path
                if br_path.exists():
                    try:
                        if br_path.suffix.lower() in [".xlsx", ".xls"]:
                            configured_sheet = self.config.get("paths", {}).get("bridge_sheet_name")
                            sheet_candidates = [configured_sheet, "Bridge List", "Bridge list 0-5 km", 0]
                            for sc in sheet_candidates:
                                if sc is None:
                                    continue
                                try:
                                    bridges_df = pd.read_excel(br_path, sheet_name=sc, header=0)
                                    break
                                except Exception:
                                    continue
                        else:
                            bridges_df = pd.read_csv(br_path)
                    except Exception as exc:
                        self.logger.warning("Could not auto-load bridge schedule for profile: %s", exc)

        if resolved_bridges is None and (bridges_df is None or not isinstance(bridges_df, pd.DataFrame) or bridges_df.empty):
            return 0

        end_ch = min_ch + span_m
        count = 0

        # Build interpolation arrays for profile levels in case any level is not present in row
        prof_chs = profile_df["chainage_m"].to_numpy(dtype=float) if "chainage_m" in profile_df.columns else np.array([])
        prof_egl = profile_df["egl_m"].to_numpy(dtype=float) if "egl_m" in profile_df.columns else None
        prof_erl = profile_df["erl_m"].to_numpy(dtype=float) if "erl_m" in profile_df.columns else None
        prof_prl = profile_df["prl_m"].to_numpy(dtype=float) if "prl_m" in profile_df.columns else None
        prof_efl = profile_df["efl_m"].to_numpy(dtype=float) if "efl_m" in profile_df.columns else None
        prof_pfl = profile_df["pfl_m"].to_numpy(dtype=float) if "pfl_m" in profile_df.columns else None

        rad = math.radians(self.bridge_callout_slant_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        pitch = self.bridge_callout_line_spacing_mm
        txt_h = self.bridge_callout_text_height_mm

        if resolved_bridges is not None:
            sheet_bridges = resolved_bridges
            raw_bridge_xs = [b["v_x"] for b in sheet_bridges]
            target_bridge_xs = [b.get("target_x", b["v_x"]) for b in sheet_bridges]
        else:
            # Fallback self-contained deconfliction
            sheet_bridges = []
            for _, row in bridges_df.iterrows():
                ch_val = None
                for key in ["chainage (m)", "chainage_m", "chainage", "CHAINAGE", "approx. chainage (km)", "Approx. Chainage (km)", "ch"]:
                    for rk, rv in row.items():
                        if str(rk).strip().lower() == key.lower() and pd.notna(rv):
                            ch_val = rv
                            break
                    if ch_val is not None:
                        break
                if ch_val is None:
                    continue
                try:
                    ch = float(ch_val)
                    if 0.0 < ch < 500.0:  # in km
                        ch *= 1000.0
                except Exception:
                    continue

                if min_ch - 0.1 <= ch <= end_ch + 0.1:
                    sheet_bridges.append({"ch": ch, "row": row})

            sheet_bridges.sort(key=lambda item: item["ch"])

            raw_bridge_xs = [x_origin + (b["ch"] - min_ch) * scale_x for b in sheet_bridges]
            target_bridge_xs = (
                self.deconflict_ordinate_positions(raw_bridge_xs, min_gap_mm=self.ordinate_min_spacing_mm)
                if self.ordinate_deconflict_enabled
                else list(raw_bridge_xs)
            )

        for idx, item in enumerate(sheet_bridges):
            ch = item["ch"]
            row = item["row"]
            v_x = raw_bridge_xs[idx]
            target_x = target_bridge_xs[idx]

            def _get_field(candidates: list[str]) -> Any:
                for c in candidates:
                    for rk, rv in row.items():
                        if str(rk).strip().lower() == c.lower() and pd.notna(rv):
                            return rv
                return None

            br_no_raw = _get_field(["bridge no.", "bridge_no", "br.no", "feature", "br_no"]) or f"BR @ {ch:.0f}"
            br_no_str = str(br_no_raw).strip()
            # Clean "BRNo-112" -> "112" if numeric or starts with BRNo-
            br_clean = re.sub(r"^(?:BRNo-|BR-|BR\s*)", "", br_no_str, flags=re.IGNORECASE)

            exg_span = str(_get_field(["existing span", "existing_span", "exg.span", "exg_span"]) or "").strip()
            prop_span = str(_get_field(["proposed span", "proposed_span", "prop.span", "prop_span"]) or "").strip()

            def _get_level(candidates: list[str], fallback_arr: Optional[np.ndarray], default_val: float = 0.0) -> float:
                val = _get_field(candidates)
                if val is not None:
                    try:
                        f_v = float(val)
                        if not math.isnan(f_v):
                            return f_v
                    except Exception:
                        pass
                if fallback_arr is not None and len(fallback_arr) > 0 and len(prof_chs) > 0:
                    return float(np.interp(ch, prof_chs, fallback_arr))
                return default_val

            ext_bl = _get_level(["bed level", "bed_level", "ext.bl", "bl", "gl"], prof_egl, 580.0)
            ext_fl = _get_level(["efl", "efl_m", "ext.fl", "existing formation level", "exg.fl"], prof_efl, ext_bl + 4.5)
            exg_rl = _get_level(["erl", "erl_m", "exg.rl", "existing rail level", "exg.rl"], prof_erl, ext_fl + 0.688)
            prop_rl = _get_level(["prl", "prl_m", "prop.rl", "proposed rail level", "prop.rl"], prof_prl, exg_rl)
            prop_fl = _get_level(["pfl", "pfl_m", "prop.fl", "proposed formation level", "prop.fl"], prof_pfl, prop_rl - 0.762)
            prop_bl = _get_level(["prop.bl", "prop_bl"], None, ext_bl)

            def _fmt_lvl(val: float) -> str:
                s = f"{val:.5f}".rstrip("0")
                parts = s.split(".")
                if len(parts) == 2 and len(parts[1]) < 3:
                    return f"{val:.3f}"
                return s

            # Format 4 stacked lines matching Indian Railways field drawing standard
            # Line 1 (White): CL OF EXIST . BR.NO: <no>, AT CH: <ch>M, SPAN: <exg_span>,
            if exg_span and exg_span.lower() != "nan":
                line1 = f"CL OF EXIST . BR.NO: {br_clean}, AT CH: {ch:.3f}M, SPAN: {exg_span},"
            else:
                line1 = f"CL OF EXIST . BR.NO: {br_clean}, AT CH: {ch:.3f}M,"

            # Line 2 (White): EXT.BL:<ext_bl>M,EXT.FL:<ext_fl>M,EXG.RL:<exg_rl>M
            line2 = f"EXT.BL:{_fmt_lvl(ext_bl)}M,EXT.FL:{_fmt_lvl(ext_fl)}M,EXG.RL:{_fmt_lvl(exg_rl)}M"

            # Line 3 (Red): PROP .TO BE EXTENDED WITH SPAN:<prop_span>,
            if prop_span and prop_span.lower() != "nan" and prop_span.lower() != "existing to retain":
                line3 = f"PROP .TO BE EXTENDED WITH SPAN:{prop_span},"
            else:
                line3 = "PROP .TO BE EXTENDED WITH SAME SPAN,"

            # Line 4 (Red): PROP .RL:<prop_rl>M,PROP .FL:{prop_fl:.3f}M,PROP .BL:<prop_bl>M
            line4 = f"PROP .RL:{_fmt_lvl(prop_rl)}M,PROP .FL:{_fmt_lvl(prop_fl)}M,PROP .BL:{_fmt_lvl(prop_bl)}M"

            my_rl = y_datum + (exg_rl - datum_m) * scale_y

            # Crosshair marker with center box at true ground station and elevation (v_x, my_rl)
            if self.show_profile_bridge_marker:
                c_size = 2.8
                b_size = 0.5
                msp.add_line((v_x - c_size, my_rl), (v_x + c_size, my_rl), dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 15})
                msp.add_line((v_x, my_rl - c_size), (v_x, my_rl + c_size), dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 15})
                msp.add_lwpolyline(
                    [
                        (v_x - b_size, my_rl - b_size),
                        (v_x + b_size, my_rl - b_size),
                        (v_x + b_size, my_rl + b_size),
                        (v_x - b_size, my_rl + b_size),
                    ],
                    close=True,
                    dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 15},
                )

            # Full-height vertical station ordinate line across surface profile grid to top boundary (with crank/dogleg if shifted)
            y_jog = item.get("y_jog", top_grid_y - self.ordinate_jog_depth_mm - (idx % 3) * 4.0)
            self._draw_cranked_ordinate_line(
                msp=msp,
                v_x=v_x,
                target_x=target_x,
                y_start=y_datum,
                top_grid_y=top_grid_y,
                y_jog=y_jog,
                layer="ESP_PROF_GRID",
                color=self.col_grid,
                lineweight=13,
            )

            # Continuous slanted dividing line (leader / underline) starting at (target_x, top_grid_y)
            max_char_len = max(len(line1), len(line2), len(line3), len(line4))
            line_len_mm = max(75.0, 2.0 + max_char_len * 1.15)
            max_allowed_rise = max(20.0, self.plan_bot_y - top_grid_y - 12.0)
            allowed_rise = min(self.bridge_callout_max_rise_mm, max_allowed_rise)
            if sin_a > 1e-4 and (line_len_mm * sin_a) > allowed_rise:
                line_len_mm = allowed_rise / sin_a
            x_line_end = target_x + line_len_mm * cos_a
            y_line_end = top_grid_y + line_len_mm * sin_a
            msp.add_line(
                (target_x, top_grid_y),
                (x_line_end, y_line_end),
                dxfattribs={"layer": "ESP_PROF_TEXT_EX", "color": self.col_text_ex, "lineweight": 18},
            )

            # Text Placement relative to (target_x, top_grid_y):
            s_0 = 1.2
            d_gap = 0.5

            # Line 2 (White, Existing Levels): sits directly ABOVE the slanted line
            l2_x = target_x + s_0 * cos_a - d_gap * sin_a
            l2_y = top_grid_y + s_0 * sin_a + d_gap * cos_a

            # Line 1 (White, Existing BR No & Span): sits directly ABOVE Line 2
            l1_x = l2_x - pitch * sin_a
            l1_y = l2_y + pitch * cos_a

            # Line 3 (Red, Proposed Span Extension): sits directly BELOW the slanted line
            l3_x = target_x + s_0 * cos_a + (txt_h + d_gap) * sin_a
            l3_y = top_grid_y + s_0 * sin_a - (txt_h + d_gap) * cos_a

            # Line 4 (Red, Proposed Levels): sits directly BELOW Line 3
            l4_x = l3_x + pitch * sin_a
            l4_y = l3_y - pitch * cos_a

            line_entries = [
                (line1, l1_x, l1_y, self.col_text_ex, "ESP_PROF_TEXT_EX"),
                (line2, l2_x, l2_y, self.col_text_ex, "ESP_PROF_TEXT_EX"),
                (line3, l3_x, l3_y, self.col_text_prop, "ESP_PROF_TEXT_PROP"),
                (line4, l4_x, l4_y, self.col_text_prop, "ESP_PROF_TEXT_PROP"),
            ]

            for text_str, lx, ly, col, lay in line_entries:
                msp.add_text(
                    text_str,
                    dxfattribs={
                        "layer": lay,
                        "color": col,
                        "style": "ESP_STYLE",
                        "height": txt_h,
                        "rotation": self.bridge_callout_slant_deg,
                    },
                ).set_placement((lx, ly), align=TextEntityAlignment.BOTTOM_LEFT)

            count += 1

        self.logger.info("Rendered %d surface profile 4-line bridge callouts", count)
        return count

    def draw_lsection(
        self,
        survey_df: pd.DataFrame,
        pfl_series: Optional[Sequence[float]] = None,
        egl_series: Optional[Sequence[float]] = None,
        output_name: str = "brownfield_lsection_profile.dxf",
        bridges_df: Optional[pd.DataFrame] = None,
        lsection_bands: Optional[list[dict[str, Any] | ProfileBandDef]] = None,
    ) -> BrownfieldProfileResult:
        """Generate official Indian Railways L-Section DXF drawing.

        Args:
            survey_df: Survey DataFrame with chainage_m, erl_m, efl_m.
            pfl_series: Optional proposed formation levels.
            egl_series: Optional ground levels.
            output_name: Output DXF filename.
            bridges_df: Optional DataFrame of bridges and structures along corridor.
            lsection_bands: Optional list of dynamic user-selected profile band
                           definitions or dictionaries from UI/config.

        Returns:
            BrownfieldProfileResult containing paths, profile_df, datum, and metadata.
        """
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Update dynamic bands if explicitly passed to draw_lsection
        if lsection_bands is not None:
            self.band_defs = self._parse_dynamic_band_definitions(lsection_bands)
            self.total_band_height_mm = sum(b.height_mm for b in self.band_defs)
            self.band_specs = [(b.title, b.height_mm, b.value_color) for b in self.band_defs]

        profile_df = self._prepare_profile_dataframe(survey_df, pfl_series, egl_series)

        doc, msp = self._setup_dxf_document()

        datum_m = self.draw_profile_bands_into_msp(
            msp=msp,
            profile_df=profile_df,
            x_origin=self.header_width_mm,
            y_datum=0.0,
            bridges_df=bridges_df,
        )

        out_path = self.outputs_dir / output_name
        doc.saveas(out_path)
        self.logger.info("Saved Brownfield L-Section Profile DXF: %s", out_path)

        chainages = profile_df["chainage_m"].to_numpy()
        min_ch = float(chainages[0])
        max_ch = float(chainages[-1])
        total_len_m = max_ch - min_ch

        metadata: dict[str, Any] = {
            "total_stations": int(len(profile_df)),
            "chainage_start_m": min_ch,
            "chainage_end_m": max_ch,
            "total_length_m": round(total_len_m, 2),
            "datum_m": datum_m,
            "h_scale": self.h_scale,
            "v_scale": self.v_scale,
            "template": self.template_name,
            "bands_count": len(self.band_defs),
            "band_titles": [b.title for b in self.band_defs],
        }

        if "erl_m" in profile_df.columns and profile_df["erl_m"].notna().any():
            metadata["erl_min_m"] = round(float(profile_df["erl_m"].dropna().min()), 3)
            metadata["erl_max_m"] = round(float(profile_df["erl_m"].dropna().max()), 3)
        if "prl_m" in profile_df.columns and profile_df["prl_m"].notna().any():
            metadata["prl_min_m"] = round(float(profile_df["prl_m"].dropna().min()), 3)
            metadata["prl_max_m"] = round(float(profile_df["prl_m"].dropna().max()), 3)
        if "diff_prl_erl_m" in profile_df.columns and profile_df["diff_prl_erl_m"].notna().any():
            metadata["avg_track_lift_m"] = round(float(profile_df["diff_prl_erl_m"].dropna().mean()), 3)
            metadata["max_track_lift_m"] = round(float(profile_df["diff_prl_erl_m"].dropna().max()), 3)

        return BrownfieldProfileResult(
            dxf_path=out_path,
            profile_df=profile_df,
            datum_m=datum_m,
            metadata=metadata,
        )
