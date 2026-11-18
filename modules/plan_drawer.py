"""
Module: modules/plan_drawer.py
Purpose: Module 9B — 2D Railway Alignment Plan (Working Plan) DXF/PDF Generator.

Engineering Basis & Complete Workflow:
    1. Read KML / Alignment Line: Raw polyline converted to UTM coordinates.
    2. Detect Points of Intersection (IPs): Extract deflection angles and intersection points.
    3. Calculate Curve Details: Design Radius R, Transition Spiral Ls, Tangent Length T,
       Circular Curve Length Lc, and tangency points (TP-TC, TP-CC, CT-PC, CT-PT) per RDSO norms.
    4. Reconstruct / Edit Alignment Line: Replace raw chords with true mathematical geometry
       (Straight Tangent -> Entry Clothoid Spiral -> True Circular Arc (Radius R) -> Exit Clothoid Spiral -> Tangent).
    5. Plot Updated Alignment in Plan View: Render true AutoCAD ARCs, transition polylines,
       corridor boundaries, 100m station ticks, apex callouts, and RDSO Curve Summary Table.

Single Source of Truth:
    All visual curve entities, CAD ARCs, station markers, curve callout boxes,
    and the Curve Table are strictly derived from the updated mathematical geometry.

Standards: RDSO GE:G-1, IRPWM Chapter V, Indian Railways Schedule of Dimensions (SOD).
Author   : Railway ESP Generator
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE_ROOT = _PROJECT_ROOT.parent

import math
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment, MTextEntityAlignment
from shapely.geometry import LineString, Point, Polygon, MultiLineString, MultiPolygon
from pyproj import Transformer
from modules.read_kml import AlignmentResult
from modules.horizontal_geometry import HGeomResult
from modules.earthwork import Earthwork
from utils.logger import get_logger

logger = get_logger(__name__)


class PlanDrawerError(Exception):
    """Custom exception for Alignment Plan DXF drawing errors."""
    pass


@dataclass
class TopoFeature:
    """Represents a vector topographic survey feature in UTM ground coordinates.

    Attributes:
        geometry: Shapely geometry object (LineString, Polygon, MultiPolygon, MultiLineString, Point).
        feature_type: Semantic classification ('boundary', 'water', 'road', 'building', 'contour', 'overlay').
        name: Name or description of the feature.
        layer_name: AutoCAD DXF layer name.
        color: AutoCAD Color Index (ACI).
        lineweight: AutoCAD lineweight in 1/100 mm.
        source_file: Stem name of the source shapefile/layer.
    """
    geometry: Any
    feature_type: str = "overlay"
    name: str = ""
    layer_name: str = "TOPO_OVERLAY"
    color: int = 8
    lineweight: int = 18
    source_file: str = ""
    source_type: str = "shp"


@dataclass
class PlanResult:
    """Result object returned by PlanDrawer.

    Attributes:
        dxf_path: Path to the primary generated AutoCAD DXF file.
        dxf_paths: List of paths to all generated DXF sheet files.
        pdf_paths: List of paths to all generated PDF sheet files.
        reconstructed_line: The mathematically edited and reconstructed Shapely LineString.
        metadata: Drawing metadata dictionary.
    """
    dxf_path: Path
    dxf_paths: list[Path] = field(default_factory=list)
    pdf_paths: list[Path] = field(default_factory=list)
    reconstructed_line: Optional[LineString] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dxf_paths and self.dxf_path:
            self.dxf_paths = [self.dxf_path]


def add_masked_mtext(
    msp: Any,
    text: str,
    insert: tuple[float, float],
    char_height: float,
    rotation_deg: float = 0.0,
    layer: str = "0",
    color: int = 7,
    style: str = "ESP_STYLE",
    attachment_point: MTextEntityAlignment = MTextEntityAlignment.MIDDLE_LEFT,
    width: Optional[float] = None,
    bg_scale: float = 1.05,
) -> Any:
    """Create an MText entity with native AutoCAD canvas background mask (bg_fill=3).

    Blanks out underlying CAD linework, contours, and hatching directly behind text with a tight 5% margin.
    """
    dxfattribs: dict[str, Any] = {
        "char_height": float(char_height),
        "rotation": float(rotation_deg),
        "layer": layer,
        "color": int(color),
        "attachment_point": attachment_point,
    }
    if style and hasattr(msp, "doc") and style in msp.doc.styles:
        dxfattribs["style"] = style
    if width is not None and width > 0:
        dxfattribs["width"] = float(width)

    mt = msp.add_mtext(text, dxfattribs=dxfattribs)
    mt.dxf.insert = (float(insert[0]), float(insert[1]), 0.0)
    if rotation_deg != 0.0:
        rot_rad_dir = math.radians(float(rotation_deg))
        mt.dxf.text_direction = (math.cos(rot_rad_dir), math.sin(rot_rad_dir), 0.0)
    try:
        mt.set_bg_color("canvas", scale=bg_scale)
    except Exception as exc:
        logger.debug("MTEXT canvas background fill not supported or failed: %s", exc)
    return mt


class PlanTextCollisionManager:
    """Spatial collision manager to prevent text overlapping across Plan View elements.

    Maintains a 2D spatial index of occupied bounding polygons and computes collision-free
    leader line offsets across multiple tiers, track-side flipping (LHS/RHS), and clearance checks.
    """

    def __init__(self, buffer_padding: float = 0.25) -> None:
        self.placed_polygons: list[Polygon] = []
        self.buffer_padding = buffer_padding

    def register_box(self, poly: Optional[Polygon]) -> None:
        """Register an occupied bounding polygon into the manager."""
        if poly is not None and not poly.is_empty:
            try:
                buf = poly.buffer(self.buffer_padding)
                self.placed_polygons.append(buf)
            except Exception as exc:
                logger.debug("Failed buffering polygon, using original: %s", exc)
                self.placed_polygons.append(poly)

    def is_clear(self, poly: Optional[Polygon]) -> bool:
        """Return True if candidate polygon does not intersect any placed polygon."""
        if poly is None or poly.is_empty:
            return False
        for placed in self.placed_polygons:
            try:
                if poly.intersects(placed):
                    return False
            except Exception as exc:
                logger.debug("Error during polygon intersection check: %s", exc)
                continue
        return True

    @staticmethod
    def estimate_text_box(
        x: float,
        y: float,
        text_w: float,
        text_h: float,
        rot_deg: float = 0.0,
        align: str = "BOTTOM_LEFT",
    ) -> Polygon:
        """Estimate 2D bounding polygon of text at given anchor and rotation."""
        rot_rad = math.radians(rot_deg)
        ux, uy = math.cos(rot_rad), math.sin(rot_rad)
        vx, vy = -math.sin(rot_rad), math.cos(rot_rad)

        if align == "MIDDLE_CENTER":
            ox = x - (text_w / 2.0) * ux - (text_h / 2.0) * vx
            oy = y - (text_w / 2.0) * uy - (text_h / 2.0) * vy
        elif align == "TOP_LEFT":
            ox = x - text_h * vx
            oy = y - text_h * vy
        elif align == "MIDDLE_LEFT":
            ox = x - (text_h / 2.0) * vx
            oy = y - (text_h / 2.0) * vy
        else:  # BOTTOM_LEFT
            ox = x
            oy = y

        p1 = (ox, oy)
        p2 = (ox + text_w * ux, oy + text_w * uy)
        p3 = (p2[0] + text_h * vx, p2[1] + text_h * vy)
        p4 = (p1[0] + text_h * vx, p1[1] + text_h * vy)
        return Polygon([p1, p2, p3, p4])

    def find_clear_leader_position(
        self,
        px: float,
        py: float,
        heading: float,
        text_w: float,
        text_h: float,
        formation_clearance: float,
        preferred_side: float = 1.0,
        anno_scale: float = 1.0,
        allow_flip_side: bool = True,
        candidate_tiers: Optional[list[float]] = None,
    ) -> tuple[tuple[float, float], tuple[float, float], Polygon, float, float]:
        """Find a collision-free leader line end and text position along alignment.

        Args:
            px, py: Centerline coordinate on sheet/paper.
            heading: Tangent heading angle in radians.
            text_w: Text width in drawing units.
            text_h: Text height in drawing units.
            formation_clearance: Minimum clearance distance from track centerline.
            preferred_side: +1.0 for RHS, -1.0 for LHS.
            anno_scale: Current annotation scale factor.
            allow_flip_side: If True, tests opposite side of alignment if preferred side is blocked.
            candidate_tiers: List of leader offset distances in mm (unscaled).

        Returns:
            Tuple of:
                (leader_end_pt, text_insert_pt, chosen_poly, rotation_deg, chosen_side)
        """
        if candidate_tiers is None:
            candidate_tiers = [6.0, 18.0, 30.0, 42.0, 54.0, 66.0, 78.0]

        sides_to_try = [preferred_side]
        if allow_flip_side:
            sides_to_try.append(-preferred_side)

        for side in sides_to_try:
            perp = heading + (side * math.pi / 2.0)
            perp_dx, perp_dy = math.cos(perp), math.sin(perp)
            r_deg = math.degrees(perp) % 360.0
            r_rad = math.radians(r_deg)
            t_up_x, t_up_y = -math.sin(r_rad), math.cos(r_rad)

            for offset_raw in candidate_tiers:
                lead_len = (formation_clearance + offset_raw + text_w) * anno_scale
                end_cand = (px + lead_len * perp_dx, py + lead_len * perp_dy)

                tx_cand = px + (formation_clearance + offset_raw) * perp_dx + (0.5 * anno_scale) * t_up_x
                ty_cand = py + (formation_clearance + offset_raw) * perp_dy + (0.5 * anno_scale) * t_up_y

                p1 = (tx_cand, ty_cand)
                p2 = (tx_cand + text_w * math.cos(r_rad), ty_cand + text_w * math.sin(r_rad))
                p3 = (p2[0] + text_h * t_up_x, p2[1] + text_h * t_up_y)
                p4 = (p1[0] + text_h * t_up_x, p1[1] + text_h * t_up_y)
                cand_poly = Polygon([p1, p2, p3, p4])

                if self.is_clear(cand_poly):
                    self.register_box(cand_poly)
                    return end_cand, (tx_cand, ty_cand), cand_poly, r_deg, side

        # Fallback to outermost tier on preferred side
        fallback_offset = candidate_tiers[-1] + 12.0
        perp = heading + (preferred_side * math.pi / 2.0)
        perp_dx, perp_dy = math.cos(perp), math.sin(perp)
        r_deg = math.degrees(perp) % 360.0
        r_rad = math.radians(r_deg)
        t_up_x, t_up_y = -math.sin(r_rad), math.cos(r_rad)

        lead_len = (formation_clearance + fallback_offset + text_w) * anno_scale
        end_cand = (px + lead_len * perp_dx, py + lead_len * perp_dy)
        tx_cand = px + (formation_clearance + fallback_offset) * perp_dx + (0.5 * anno_scale) * t_up_x
        ty_cand = py + (formation_clearance + fallback_offset) * perp_dy + (0.5 * anno_scale) * t_up_y

        p1 = (tx_cand, ty_cand)
        p2 = (tx_cand + text_w * math.cos(r_rad), ty_cand + text_w * math.sin(r_rad))
        p3 = (p2[0] + text_h * t_up_x, p2[1] + text_h * t_up_y)
        p4 = (p1[0] + text_h * t_up_x, p1[1] + text_h * t_up_y)
        chosen_poly = Polygon([p1, p2, p3, p4])
        self.register_box(chosen_poly)
        return end_cand, (tx_cand, ty_cand), chosen_poly, r_deg, preferred_side


class PlanDrawer:
    """Generates official Indian Railways Alignment Plan (Working Plan) DXF drawings."""

    COL_RED = 1
    COL_YELLOW = 2
    COL_GREEN = 3
    COL_CYAN = 4
    COL_BLUE = 5
    COL_MAGENTA = 6
    COL_WHITE = 7
    COL_GREY = 8
    COL_BLACK = 7
    COL_PINK = 221  # AutoCAD Color Index 221: Pink

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

        # Directories
        paths_cfg = config.get("paths", {})
        self.outputs_dir = Path(paths_cfg.get("outputs_dir", "outputs"))
        self.ap_dir = Path(paths_cfg.get("alignment_plan_outputs_dir", str(self.outputs_dir / "alignment_plan")))

        # Drawing configurations
        drawing_cfg = config.get("drawing", {})
        plan_cfg = drawing_cfg.get("plan_view", {})

        self.plan_scale: float = float(plan_cfg.get("scale", drawing_cfg.get("plan_scale", 5000.0)))
        self.max_sheet_length_m: float = float(drawing_cfg.get("max_lsection_length_m", 5000.0))
        self.generate_complete_profile: bool = bool(drawing_cfg.get("generate_complete_profile", True))
        self.generate_sheet_drawings: bool = bool(drawing_cfg.get("generate_sheet_drawings", True))
        self.tick_interval_m: float = float(plan_cfg.get("chainage_tick_interval_m", 100.0))
        self.chainage_format: str = str(
            plan_cfg.get("chainage_format")
            or drawing_cfg.get("chainage_format")
            or config.get("alignment", {}).get("chainage_format")
            or "CH-{km}+{m:03d}"
        )
        self.show_chainage_labels: bool = bool(
            plan_cfg.get("show_chainage_labels", plan_cfg.get("show_station_labels", True))
        )

        self.show_topo_layer: bool = bool(plan_cfg.get("show_topo_layer", True))
        self.show_shp_topo_layer: bool = bool(plan_cfg.get("show_shp_topo_layer", self.show_topo_layer))
        self.show_kml_topo_layer: bool = bool(plan_cfg.get("show_kml_topo_layer", self.show_topo_layer))
        kml_cfg = plan_cfg.get("kml_topo", {})
        self.kml_linetype: str = str(plan_cfg.get("kml_layer_linetype", kml_cfg.get("linetype", "DASHDOTDOT")))
        self.kml_color: int = int(plan_cfg.get("kml_layer_color", kml_cfg.get("color", self.COL_RED)))
        self.kml_lineweight: int = int(plan_cfg.get("kml_layer_lineweight", kml_cfg.get("lineweight", 25)))
        self.max_topo_offset_m: float = float(plan_cfg.get("max_topo_offset_m", 150.0))
        self.min_topo_boundary_length_m: float = float(plan_cfg.get("min_topo_boundary_length_m", 5.0))
        self.only_outer_boundary: bool = bool(plan_cfg.get("only_outer_boundary", False))
        self.epsg_code: str = str(config.get("geospatial", {}).get("target_epsg", "EPSG:32643"))
        self._cached_topo_features: Optional[list[TopoFeature]] = None
        self._cached_epsg: Optional[str] = None

        # Selected shapefile text layers to annotate
        selected_text_cfg = (
            paths_cfg.get("selected_shp_text_layers")
            or plan_cfg.get("selected_shp_text_layers", [])
        )
        if isinstance(selected_text_cfg, list):
            self.selected_shp_text_layers: set[str] = {
                Path(str(x)).stem.strip().upper() for x in selected_text_cfg if str(x).strip()
            }
        else:
            self.selected_shp_text_layers = set()

        # Track parameters
        railway_cfg = config.get("railway", {})
        self.gauge_mm: float = float(railway_cfg.get("gauge_width_mm", 1676.0))
        self.gauge_half_m: float = (self.gauge_mm / 1000.0) / 2.0  # 0.838 m

        earthwork_cfg = config.get("earthwork", {})
        self.formation_width_m: float = float(earthwork_cfg.get("formation_width_m", 7.7))
        self.formation_half_m: float = self.formation_width_m / 2.0  # 3.85 m
        self.row_offset_m: float = 15.0  # 15 m Right-of-Way offset

        # Sheet layout constants (A0 Sheet: 1189 x 841 mm)
        self.sheet_width_mm: float = float(drawing_cfg.get("sheet_width_mm", 1189.0))
        self.sheet_height_mm: float = float(drawing_cfg.get("sheet_height_mm", 841.0))
        self.margin_mm: float = float(drawing_cfg.get("margin_mm", 10.0))

        self.is_existing: bool = bool(drawing_cfg.get("is_existing_line", False))
        self.track_color: int = self.COL_BLACK if self.is_existing else self.COL_RED
        self.transition_step_m: float = float(drawing_cfg.get("transition_step_m", 5.0))
        # kml_only_mode: read from plan_view section first; fall back to drawing section
        _kml_mode_default = plan_cfg.get("kml_only_mode", drawing_cfg.get("kml_only_mode", False))
        self.kml_only_mode: bool = bool(_kml_mode_default)
        self.draw_synthetic_boundary: bool = bool(drawing_cfg.get("draw_synthetic_boundary", False))

        # In kml_only_mode we render ALL KML content — disable proximity/length filters
        # NOTE: must be set AFTER kml_only_mode is assigned above.
        self._kml_min_length_m: float = 0.0  # no short-segment filtering for KML

    def format_chainage(self, ch_m: float) -> str:
        """Format chainage according to configured Plan View chainage_format template."""
        from utils.geo_utils import format_chainage
        return format_chainage(ch_m, format_pattern=self.chainage_format)

    def reconstruct_curved_alignment(
        self, raw_line: LineString, ip_df: pd.DataFrame
    ) -> LineString:
        """Return the KML alignment as-is (true mathematical reconstruction available via kml_only_mode=false).

        When the KML track is the authoritative source this pass-through is correct.
        Set `reconstruct_curves` in config to True to use `_compute_curved_alignment()` instead.
        """
        use_reconstruction = bool(
            self.config.get("drawing", {}).get("plan_view", {}).get("reconstruct_curves", False)
        )
        if use_reconstruction and ip_df is not None and not ip_df.empty:
            return self._compute_curved_alignment(raw_line, ip_df)
        return raw_line

    def _compute_curved_alignment(
        self, raw_line: LineString, ip_df: pd.DataFrame
    ) -> LineString:
        """Rebuild the alignment geometry with true Fresnel clothoid spirals and circular arcs.

        Replaces each raw KML chord segment that lies within a curve with mathematically
        exact geometry:
          - Entry clothoid spiral  (TP-TC → TP-CC)
          - True circular arc       (TP-CC → CT-PC)
          - Exit clothoid spiral    (CT-PC → CT-PT)
        All other sections remain as straight tangent lines.

        Returns:
            A new LineString whose vertices follow the exact horizontal alignment geometry.
        """
        valid_ips = []
        for _, r in ip_df.iterrows():
            rad = float(r.get("radius_m", 0.0))
            if rad <= 0:
                continue
            ip_ch = float(r.get("chainage_m", 0.0))
            t_len = float(r.get("tangent_length_m", 0.0))
            tptc = float(r.get("tptc_chainage_m", max(0.0, ip_ch - t_len)))
            tpcc = float(r.get("tpcc_chainage_m", tptc))
            ctpc = float(r.get("ctpc_chainage_m", tpcc))
            ctpt = float(r.get("ctpt_chainage_m", ip_ch + t_len))
            direction = str(r.get("direction", "RHS"))
            ls = float(r.get("transition_length_m", 0.0))
            lc = float(r.get("circular_curve_length_m", 0.0))
            ip_e = float(r.get("easting_m", r.get("easting", 0.0)))
            ip_n = float(r.get("northing_m", r.get("northing", 0.0)))

            valid_ips.append({
                "ip_ch": ip_ch, "radius": rad, "direction": direction,
                "tptc": tptc, "tpcc": tpcc, "ctpc": ctpc, "ctpt": ctpt,
                "ls": ls, "lc": lc, "ip_e": ip_e, "ip_n": ip_n,
            })

        valid_ips.sort(key=lambda x: x["tptc"])
        if not valid_ips:
            return raw_line

        total_len = float(raw_line.length)
        reconstructed_coords: list[tuple[float, float]] = []
        curr_ch = 0.0

        for c in valid_ips:
            tptc = max(curr_ch, c["tptc"])
            tpcc = max(tptc, c["tpcc"])
            ctpc = max(tpcc, c["ctpc"])
            ctpt = min(total_len, max(ctpc, c["ctpt"]))

            # Tangency points on incoming and outgoing tangents
            p_tptc = raw_line.interpolate(tptc)
            p_ctpt = raw_line.interpolate(ctpt)

            # 1. Add true straight tangent leading up to TP-TC
            if curr_ch < tptc:
                p_start_pt = reconstructed_coords[-1] if reconstructed_coords else (raw_line.coords[0][0], raw_line.coords[0][1])
                sample_chs = np.linspace(curr_ch, tptc, max(2, int((tptc - curr_ch) / 10.0)))
                for s in sample_chs:
                    frac = (s - curr_ch) / (tptc - curr_ch) if tptc > curr_ch else 0.0
                    gx = p_start_pt[0] + frac * (p_tptc.x - p_start_pt[0])
                    gy = p_start_pt[1] + frac * (p_tptc.y - p_start_pt[1])
                    reconstructed_coords.append((gx, gy))
            
            # Intersection point (IP) coordinates
            ip_e = c["ip_e"] if c["ip_e"] != 0.0 else raw_line.interpolate(c["ip_ch"]).x
            ip_n = c["ip_n"] if c["ip_n"] != 0.0 else raw_line.interpolate(c["ip_ch"]).y

            # Incoming tangent unit vector: from TP-TC towards IP
            v_in_x, v_in_y = ip_e - p_tptc.x, ip_n - p_tptc.y
            len_in = math.hypot(v_in_x, v_in_y)
            if len_in <= 1e-4:
                raise PlanDrawerError(f"Degenerate incoming tangent vector at IP chainage {c['ip_ch']:.2f}m")
            u_in_x, u_in_y = v_in_x / len_in, v_in_y / len_in

            # Outgoing tangent unit vector: from IP towards CT-PT
            v_out_x, v_out_y = p_ctpt.x - ip_e, p_ctpt.y - ip_n
            len_out = math.hypot(v_out_x, v_out_y)
            if len_out <= 1e-4:
                raise PlanDrawerError(f"Degenerate outgoing tangent vector at IP chainage {c['ip_ch']:.2f}m")
            u_out_x, u_out_y = v_out_x / len_out, v_out_y / len_out

            # Determine turn direction via 2D cross product: u_in x u_out
            cross_z = u_in_x * u_out_y - u_in_y * u_out_x
            is_right_turn = (cross_z < 0) or ("R" in c["direction"].upper() and abs(cross_z) < 1e-5)

            # Normal vector pointing towards the INTERIOR center of curvature
            if is_right_turn:
                # Right turn -> normal is 90 deg clockwise from heading
                n_in_x, n_in_y = u_in_y, -u_in_x
                n_out_x, n_out_y = u_out_y, -u_out_x
                sign = 1.0
            else:
                # Left turn -> normal is 90 deg counter-clockwise from heading
                n_in_x, n_in_y = -u_in_y, u_in_x
                n_out_x, n_out_y = -u_out_y, u_out_x
                sign = -1.0

            rad = c["radius"]
            ls = max(1.0, c["ls"])
            lc = max(0.0, c["lc"])
            shift_m = (ls**2) / (24.0 * rad) if rad > 0 else 0.0

            # Geometric Center of Circular Arc (tangent shifted by Ls/2 along tangent and (R+s) along normal)
            p1_x = p_tptc.x + (ls / 2.0) * u_in_x + (rad + shift_m) * n_in_x
            p1_y = p_tptc.y + (ls / 2.0) * u_in_y + (rad + shift_m) * n_in_y
            p2_x = p_ctpt.x - (ls / 2.0) * u_out_x + (rad + shift_m) * n_out_x
            p2_y = p_ctpt.y - (ls / 2.0) * u_out_y + (rad + shift_m) * n_out_y

            denom = u_in_x * u_out_y - u_in_y * u_out_x
            if abs(denom) > 1e-5:
                t_param = ((p2_x - p1_x) * u_out_y - (p2_y - p1_y) * u_out_x) / denom
                center_x = p1_x + t_param * u_in_x
                center_y = p1_y + t_param * u_in_y
            else:
                center_x = (p1_x + p2_x) / 2.0
                center_y = (p1_y + p2_y) / 2.0

            # 2. Reconstruct Entry Transition Spiral (Fresnel Clothoid from TP-TC to TP-CC)
            tr1_steps = max(10, int(math.ceil(ls / self.transition_step_m)))
            for step in range(tr1_steps + 1):
                s = (step / tr1_steps) * ls
                x_local = s - (s**5) / (40.0 * (rad**2) * (ls**2)) + (s**9) / (3456.0 * (rad**4) * (ls**4)) if rad > 0 else s
                y_local = -sign * ((s**3) / (6.0 * rad * ls) - (s**7) / (336.0 * (rad**3) * (ls**3))) if rad > 0 else 0.0

                gx = p_tptc.x + x_local * u_in_x - y_local * u_in_y
                gy = p_tptc.y + x_local * u_in_y + y_local * u_in_x
                reconstructed_coords.append((gx, gy))

            p_tpcc = reconstructed_coords[-1]

            # 3. Compute Exit Transition Spiral in reverse (from CT-PT backwards to CT-PC)
            tr2_steps = max(10, int(math.ceil(ls / self.transition_step_m)))
            exit_spiral_coords: list[tuple[float, float]] = []
            for step in range(tr2_steps + 1):
                s_rev = (step / tr2_steps) * ls
                x_rev = s_rev - (s_rev**5) / (40.0 * (rad**2) * (ls**2)) + (s_rev**9) / (3456.0 * (rad**4) * (ls**4)) if rad > 0 else s_rev
                y_rev = -sign * ((s_rev**3) / (6.0 * rad * ls) - (s_rev**7) / (336.0 * (rad**3) * (ls**3))) if rad > 0 else 0.0

                gx_rev = p_ctpt.x - x_rev * u_out_x - y_rev * u_out_y
                gy_rev = p_ctpt.y - x_rev * u_out_y + y_rev * u_out_x
                exit_spiral_coords.append((gx_rev, gy_rev))

            p_ctpc = exit_spiral_coords[-1]

            # 4. Reconstruct Circular Arc (from TP-CC to CT-PC using 2-point analytical arc of exact radius R)
            arc_geom = self._arc_from_2_points_and_radius(
                p_tpcc,
                p_ctpc,
                radius=rad,
                center_hint=(center_x, center_y),
                is_right_turn=is_right_turn,
            )
            if arc_geom is not None and lc > 0:
                c_arc, r_arc, sa_deg, ea_deg = arc_geom
                a1_rad = math.atan2(p_tpcc[1] - c_arc[1], p_tpcc[0] - c_arc[0])
                a2_rad = math.atan2(p_ctpc[1] - c_arc[1], p_ctpc[0] - c_arc[0])

                if is_right_turn:
                    # Clockwise sweep from TP-CC to CT-PC
                    sweep_rad = (a1_rad - a2_rad) % (2.0 * math.pi)
                    arc_steps = max(12, int(lc / 5.0))
                    for step in range(1, arc_steps):
                        frac = step / arc_steps
                        curr_ang = a1_rad - frac * sweep_rad
                        gx = c_arc[0] + r_arc * math.cos(curr_ang)
                        gy = c_arc[1] + r_arc * math.sin(curr_ang)
                        reconstructed_coords.append((gx, gy))
                else:
                    # Counter-Clockwise sweep from TP-CC to CT-PC
                    sweep_rad = (a2_rad - a1_rad) % (2.0 * math.pi)
                    arc_steps = max(12, int(lc / 5.0))
                    for step in range(1, arc_steps):
                        frac = step / arc_steps
                        curr_ang = a1_rad + frac * sweep_rad
                        gx = c_arc[0] + r_arc * math.cos(curr_ang)
                        gy = c_arc[1] + r_arc * math.sin(curr_ang)
                        reconstructed_coords.append((gx, gy))

            # Append exit transition spiral points from CT-PC to CT-PT
            for pt in reversed(exit_spiral_coords[:-1]):
                reconstructed_coords.append(pt)

            curr_ch = ctpt

        # 5. Add final true straight tangent to end of line
        if curr_ch < total_len:
            p_last_pt = reconstructed_coords[-1] if reconstructed_coords else (raw_line.coords[0][0], raw_line.coords[0][1])
            p_end_pt = (raw_line.coords[-1][0], raw_line.coords[-1][1])
            sample_chs = np.linspace(curr_ch, total_len, max(2, int((total_len - curr_ch) / 10.0)))
            for s in sample_chs:
                frac = (s - curr_ch) / (total_len - curr_ch) if total_len > curr_ch else 0.0
                gx = p_last_pt[0] + frac * (p_end_pt[0] - p_last_pt[0])
                gy = p_last_pt[1] + frac * (p_end_pt[1] - p_last_pt[1])
                reconstructed_coords.append((gx, gy))

        # Deduplicate consecutive identical points
        dedup_coords = [reconstructed_coords[0]]
        for pt in reconstructed_coords[1:]:
            if math.hypot(pt[0] - dedup_coords[-1][0], pt[1] - dedup_coords[-1][1]) > 0.05:
                dedup_coords.append(pt)

        self.logger.info(
            "Reconstructed mathematical curved alignment line: %d precise vertices (original: %d)",
            len(dedup_coords), len(list(raw_line.coords)),
        )
        return LineString(dedup_coords)

    def _setup_document(self) -> tuple[ezdxf.document.Drawing, Any]:
        """Create and initialize a standard AutoCAD DXF document with RDSO layers in METRES."""
        doc = ezdxf.new(dxfversion="R2013")
        doc.units = units.M
        doc.header["$INSUNITS"] = units.M
        doc.header["$MEASUREMENT"] = 1
        doc.header["$LTSCALE"] = 1.0
        doc.header["$CELTSCALE"] = 1.0
        doc.header["$PSLTSCALE"] = 0
        doc.header["$PLINEGEN"] = 1
        msp = doc.modelspace()

        # Define custom line styles
        if "RAILWAY_ROW" not in doc.linetypes:
            doc.linetypes.add(
                name="RAILWAY_ROW",
                pattern=[20.0, 14.0, -3.0, 0.0, -3.0],
                description="Railway Boundary Line ____ . ____ . ____",
            )
        if "DASHED" not in doc.linetypes:
            doc.linetypes.add(
                name="DASHED",
                pattern=[10.0, 6.0, -4.0],
                description="Dashed Line __ __ __",
            )
        if "DASHDOT" not in doc.linetypes:
            doc.linetypes.add(
                name="DASHDOT",
                pattern=[14.0, 8.0, -3.0, 0.0, -3.0],
                description="Dash Dot Line __ . __ . __",
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
        if "CENTER" not in doc.linetypes:
            doc.linetypes.add(
                name="CENTER",
                pattern=[20.0, 12.0, -3.0, 2.0, -3.0],
                description="Centerline ____ _ ____ _ ____",
            )

        # Standard RDSO Plan View Layers matching Official Legend
        layers_def = [
            ("BORDER", self.COL_RED, 50),
            ("TITLE_BLOCK", self.COL_RED, 35),
            ("LEGEND", self.COL_BLACK, 25),
            ("ALIGNMENT", self.COL_RED, 35),                # PROPOSED MAIN LINE
            ("EXISTING_TRACK", self.COL_BLACK, 35),         # EXISTING MAIN LINE
            ("FUTURE_TRACK", self.COL_MAGENTA, 35),         # FUTURE LINE
            ("ALIGNMENT_GAUGE", self.COL_GREY, 13),
            ("FORMATION_EDGE", self.COL_GREY, 18),
            ("RAILWAY_BOUNDARY", self.COL_WHITE, 25),       # PROP. LAND BOUNDARY (White)
            ("EXISTING_ROW", self.COL_BLACK, 25),           # EXISTING LAND BOUNDARY
            ("LAND_ACQUISITION", self.COL_RED, 18),         # PROP. LAND ACQUISITION
            ("CHAINAGE_TEXT", self.COL_RED, 18),            # KM POST & Ticks
            ("CURVE_CALLOUT", self.COL_CYAN, 25),
            ("CURVE_TABLE", self.COL_CYAN, 25),
            ("NORTH_ARROW", self.COL_BLACK, 25),
            ("GRID", self.COL_GREY, 9),
            ("STRUCTURES_PLAN", self.COL_RED, 18),          # BRIDGE & CULVERTS
            ("VGEOM_PLAN", self.COL_RED, 18),
            ("STATION_MARKER", self.COL_GREEN, 25),         # C/L OF STATION & FLAG
            ("STATION_TEXT", self.COL_GREEN, 18),           # STATION CALLOUT TEXT
            ("TOPO_WATER", self.COL_BLUE, 25),              # RIVER & POND
            ("TOPO_DRAIN", self.COL_BLUE, 25),              # DRAIN
            ("TOPO_ROAD", self.COL_BLACK, 25),              # ROAD (PUCCA)
            ("TOPO_KUTCHA_ROAD", self.COL_MAGENTA, 18),     # KUTCHA ROAD
            ("TOPO_POWER_LINE", self.COL_PINK, 18),         # HT / LT LINE (Dash Dot Dot in Pink)
            ("TOPO_BUILDING", self.COL_WHITE, 25),          # HOUSE/BLD/STRUCTURE (White)
            ("TOPO_SURVEY", self.COL_BLACK, 18),            # TBM (Temporary Bench Mark)
            ("TOPO_UTILITY", self.COL_BLACK, 18),           # ELECTRIC POLE
            ("TOPO_DISMANTLE", self.COL_GREEN, 25),         # DISMANTLE EXISTING WORK
            ("TOPO_RETAINING_WALL", self.COL_CYAN, 25),     # RETAINING WALL
            ("TOPO_CONTOUR", self.COL_GREY, 13),
            ("TOPO_OVERLAY", self.COL_GREY, 13),
            ("TOPO_TEXT", self.COL_WHITE, 18),              # Text labels for selected shapefiles/topo
            ("KML_OVERLAY", self.kml_color, self.kml_lineweight),   # KML survey overlay (Dash-Dot-Dot in Red)
            ("KML_BOUNDARY", self.kml_color, self.kml_lineweight),  # KML boundary line (Dash-Dot-Dot in Red)
            ("KML_TEXT", self.COL_RED, 18),                 # KML placemark name labels
        ]

        for name, color, weight in layers_def:
            if name not in doc.layers:
                if name == "RAILWAY_BOUNDARY":
                    lt = "RAILWAY_ROW"
                elif name == "TOPO_POWER_LINE":
                    lt = "POWER_LINE"
                elif name == "STATION_MARKER":
                    lt = "CENTER" if "CENTER" in doc.linetypes else "DASHDOT"
                elif name in ["KML_OVERLAY", "KML_BOUNDARY"]:
                    lt = self.kml_linetype if self.kml_linetype in doc.linetypes else "DASHDOTDOT"
                elif name in ["EXISTING_ROW", "TOPO_DISMANTLE", "TOPO_KUTCHA_ROAD"]:
                    lt = "DASHED"
                else:
                    lt = "Continuous"
                doc.layers.add(name=name, color=color, lineweight=weight, linetype=lt)

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

    @staticmethod
    def _safe_scale(s: float) -> float:
        """Return the annotation scale if it is a positive number, otherwise 1.0.

        This guards against None or zero values being passed as anno_scale by the caller,
        preventing divide-by-zero or zero-height text.
        """
        return float(s) if (s is not None and float(s) > 0) else 1.0

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
        if pm is not None:
            d_l = abs(math.hypot(c_left[0] - pm[0], c_left[1] - pm[1]) - r_eff)
            d_r = abs(math.hypot(c_right[0] - pm[0], c_right[1] - pm[1]) - r_eff)
            center = c_left if d_l <= d_r else c_right
        elif center_hint is not None:
            d_l = math.hypot(c_left[0] - center_hint[0], c_left[1] - center_hint[1])
            d_r = math.hypot(c_right[0] - center_hint[0], c_right[1] - center_hint[1])
            center = c_left if d_l <= d_r else c_right
        else:
            center = c_right if is_right_turn else c_left

        # Angles from center to p1 and p2
        a1 = math.degrees(math.atan2(y1 - center[1], x1 - center[0])) % 360.0
        a2 = math.degrees(math.atan2(y2 - center[1], x2 - center[0])) % 360.0

        # If pm is provided, determine CCW sweep direction passing through pm
        if pm is not None:
            am = math.degrees(math.atan2(pm[1] - center[1], pm[0] - center[0])) % 360.0
            span = (a2 - a1) % 360.0
            mid_span = (am - a1) % 360.0
            if mid_span <= span:
                start_angle, end_angle = a1, a2
            else:
                start_angle, end_angle = a2, a1
        else:
            # AutoCAD arcs sweep Counter-Clockwise from start_angle to end_angle
            if is_right_turn:
                start_angle, end_angle = a2, a1
            else:
                start_angle, end_angle = a1, a2

        return center, r_eff, start_angle, end_angle

    @staticmethod
    def _fit_arc_from_3_points(
        p1: tuple[float, float],
        pm: tuple[float, float],
        p2: tuple[float, float],
    ) -> Optional[tuple[tuple[float, float], float, float, float]]:
        """Fit a true 2D circular arc through 3 points."""
        x1, y1 = p1
        xm, ym = pm
        x2, y2 = p2

        d = 2.0 * (x1 * (ym - y2) + xm * (y2 - y1) + x2 * (y1 - ym))
        if abs(d) < 1e-7:
            return None

        ux = (
            (x1**2 + y1**2) * (ym - y2)
            + (xm**2 + ym**2) * (y2 - y1)
            + (x2**2 + y2**2) * (y1 - ym)
        ) / d
        uy = (
            (x1**2 + y1**2) * (x2 - xm)
            + (xm**2 + ym**2) * (x1 - x2)
            + (x2**2 + y2**2) * (xm - x1)
        ) / d

        center = (ux, uy)
        radius = math.hypot(x1 - ux, y1 - uy)

        a1 = math.degrees(math.atan2(y1 - uy, x1 - ux)) % 360.0
        am = math.degrees(math.atan2(ym - uy, xm - ux)) % 360.0
        a2 = math.degrees(math.atan2(y2 - uy, x2 - ux)) % 360.0

        span = (a2 - a1) % 360.0
        mid_span = (am - a1) % 360.0

        if mid_span <= span:
            start_angle, end_angle = a1, a2
        else:
            start_angle, end_angle = a2, a1

        return center, radius, start_angle, end_angle

    # ── Topographic (Topo) Background Layer Helpers ──────────────────────────

    def _classify_topo_feature(self, name: str, kml_color: str = "") -> tuple[str, str, int, int]:
        """Classify a feature by name and color into (feature_type, layer_name, color_aci, lineweight)
        strictly based on the Official Indian Railways Plan Legend.
        """
        name_u = str(name).upper().replace("_", " ").replace("-", " ")

        # 0. Dismantle Existing Work (Highest priority check)
        if any(k in name_u for k in ["DISMANTLE", "DEMOLISH", "REMOVE", "DISMANTLING", "ABANDONED WORK", "DEMOLITION"]):
            return "dismantle", "TOPO_DISMANTLE", self.COL_GREEN, 25

        # 1. Existing Main Line / Track
        if any(k in name_u for k in ["EXISTING MAIN LINE", "EXISTING TRACK", "EXISTING RAIL", "EXISTING LINE", "EXG TRACK", "EXG LINE", "OLD TRACK", "OLD LINE", "EXG RAILWAY", "EXISTING RAILWAY"]):
            return "existing_track", "EXISTING_TRACK", self.COL_BLACK, 35

        # 2. Future Line / Track
        if any(k in name_u for k in ["FUTURE LINE", "FUTURE TRACK", "FUTURE RAILWAY", "PROPOSED FUTURE", "FUTURE ALIGNMENT"]):
            return "future_track", "FUTURE_TRACK", self.COL_MAGENTA, 35

        # 3. Kutcha Road / Cart Track / Village Path (Must be before generic 'TRACK')
        has_bld_keyword = any(b in name_u for b in ["BUILDING", "HOUSE", "HOUSEHOLD", "BLD", "STRUCTURE", "SHED", "HUT", "ROOM", "RESIDENCE"])
        if not has_bld_keyword and any(k in name_u for k in ["KUTCHA ROAD", "KACHHA ROAD", "KUTCHA", "KACHHA", "CART TRACK", "EARTHEN ROAD", "VILLAGE TRACK", "FOOTPATH", "PATHWAY", "UNPAVED"]):
            return "kutcha_road", "TOPO_KUTCHA_ROAD", self.COL_MAGENTA, 18

        # 4. River / Nala / Canal / Pond / Waterbody
        if any(k in name_u for k in ["RIVER", "CANAL", "STREAM", "WATER", "NALA", "NULLAH", "DISTRIBUTARY", "POND", "LAKE", "WATERBODY", "TANK", "RESERVOIR", "KHAL"]):
            return "water", "TOPO_WATER", self.COL_BLUE, 25

        # 5. Drain / Gutter / Ditch
        if any(k in name_u for k in ["DRAIN", "DITCH", "GUTTER", "SIDE DRAIN", "CATCH WATER DRAIN", "DRAINAGE", "CHANNEL"]):
            return "drain", "TOPO_DRAIN", self.COL_BLUE, 25

        # 6. Pucca Road / Highway / Street / Carriageway / CC Road
        words = set(name_u.split())
        if any(k in name_u for k in ["ROAD", "HIGHWAY", "STREET", "EXPRESSWAY", "LANE", "NH ", "SH ", "CARRIAGEWAY", "SHOULDER", "PAVED", "TAR ROAD", "BT ROAD", "BITUMINOUS", "PUCCA ROAD", "CC ROAD", "CONCRETE ROAD", "C.C. ROAD"]) or "CC" in words or "NH" in words or "SH" in words:
            return "road", "TOPO_ROAD", self.COL_BLACK, 25

        # 7. HT / LT Power Line / Electrical Utilities
        if (
            any(k in name_u for k in [
                "HT LINE", "LT LINE", "HT/LT", "HT LT", "POWER LINE", "ELECTRIC LINE", "TRANSMISSION", "OVERHEAD POWER",
                "11KV", "33KV", "66KV", "132KV", "220KV", "400KV", "OHE", "UTILITY HT", "UTILITY LT",
            ])
            or "HT" in words
            or "LT" in words
        ):
            return "power_line", "TOPO_POWER_LINE", self.COL_PINK, 18

        # 8. Proposed Land Acquisition
        if any(k in name_u for k in ["PROP. LAND ACQUISITION", "PROP LAND ACQUISITION", "LAND ACQUISITION", "ACQUISITION", "LAND ACQ", "PROP. ACQ", "ACQ."]):
            return "land_acquisition", "LAND_ACQUISITION", self.COL_RED, 18

        # 9. Existing Land Boundary
        if any(k in name_u for k in ["EXISTING LAND BOUNDARY", "EXISTING ROW", "EXG ROW", "EXG LAND BOUNDARY", "OLD LAND BOUNDARY", "OLD ROW", "EXISTING BOUNDARY"]):
            return "existing_boundary", "EXISTING_ROW", self.COL_BLACK, 25

        # 10. Proposed Land Boundary / Right of Way (ROW) / Station Yard / Sub-Station Boundaries
        if (
            "989898" in kml_color
            or "ffffff" in kml_color
            or "ffffffff" in kml_color
            or any(k in name_u for k in [
                "RAILWAY STATION", "STATION YARD", "TRACTION SUB STATION", "SUB STATION", "SUBSTATION", "TSS",
                "EXST.AT", "EXST. AT", "751.700",
                "PROP. LAND BOUNDARY", "PROP LAND BOUNDARY", "PROPOSED LAND BOUNDARY",
                "ROW", "RIGHT OF WAY", "R.O.W", "RAILWAY BOUNDARY", "LAND BOUNDARY",
                "BOUNDARY", "OUTER", "LIMIT", "PERIMETER", "SURVEY", "EXTENTS", "FRAME", "PL", "PLOT", "PROPERTY", "KML 1", "KML 2"
            ])
        ):
            return "boundary", "RAILWAY_BOUNDARY", self.COL_WHITE, 25

        # 12. Retaining Wall / Breast Wall
        if any(k in name_u for k in ["RETAINING WALL", "RET WALL", "RETAINING", "BREAST WALL", "TOE WALL", "WALL"]):
            return "retaining_wall", "TOPO_RETAINING_WALL", self.COL_CYAN, 25

        # 13. Buildings / Houses / Structures
        if any(k in name_u for k in ["BUILDING", "HOUSE", "HOUSEHOLD", "SHED", "HUT", "STRUCTURE", "BUILTUP", "VILLAGE", "SETTLEMENT", "ROOM", "PUCCA", "KACHHA", "KUTCHA", "BLD", "RESIDENCE"]):
            return "building", "TOPO_BUILDING", self.COL_WHITE, 25

        # 14. TBM (Temporary Bench Mark) / Survey Markers
        if any(k in name_u for k in ["TBM", "TEMPORARY BENCH MARK", "BENCH MARK", "BENCHMARK", "GTS", "BM", "SURVEY POINT", "MSL"]):
            return "tbm", "TOPO_SURVEY", self.COL_BLACK, 18

        # 15. Electric Pole / Mast / Tower
        if (
            any(k in name_u for k in [
                "ELECTRIC POLE", "POLE", "OHE MAST", "MAST", "TOWER", "LIGHT POLE", "POST", "UTILITY POLE", "UTILITY EP",
            ])
            or "EP" in words
        ):
            return "electric_pole", "TOPO_UTILITY", self.COL_BLACK, 18

        # 16. Bridges / Culverts
        if any(k in name_u for k in ["BRIDGE", "CULVERT", "ROB", "RUB", "MNB", "MJB", "MAJOR BRIDGE", "MINOR BRIDGE", "PSC SLAB", "RCC BOX", "BOX CULVERT", "PIPE CULVERT"]):
            return "bridge", "STRUCTURES_PLAN", self.COL_RED, 18

        # 17. KM Post
        if any(k in name_u for k in ["KM POST", "KM_POST", "KILOMETER POST", "KM STONE"]):
            return "km_post", "CHAINAGE_TEXT", self.COL_RED, 18

        # 18. Contours
        if any(k in name_u for k in ["CONTOUR", "ELEVATION", "TOPO CONT"]):
            return "contour", "TOPO_CONTOUR", self.COL_GREY, 13

        # 19. Proposed Main Line / Alignment Centerline / Curves
        if any(k in name_u for k in [
            "PROPOSED ALIGNMENT", "PROPOSED TRACK", "PROPOSED LINE", "PROPOSED MAIN LINE", "TRACK ALIGNMENT",
            "TRANSITION CURVE", "CIRCULAR CURVE", "SPIRAL", "CLOTHOID", "TPTC", "TPCC", "CTPC", "CTPT"
        ]) or (name_u in ["ALIGNMENT", "CENTERLINE", "CENTRELINE", "MAIN LINE", "TRACK"]):
            return "alignment", "ALIGNMENT", self.COL_RED, 35

        # Default Topo Overlay
        return "overlay", "TOPO_OVERLAY", self.COL_GREY, 13

    @staticmethod
    def _parse_coord_string(coord_text: str, transformer: Any) -> list[tuple[float, float]]:
        """Parse whitespace-separated 'lon,lat[,alt]' coordinate tokens into UTM (easting, northing).

        Validates that WGS84 lon/lat values are in the plausible geographic range before
        transforming to UTM, filtering out corrupt or header/zero entries.
        """
        pts: list[tuple[float, float]] = []
        for token in coord_text.strip().split():
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                    # Validate geographic bounds — reject clearly invalid coordinates
                    # (e.g. zero/near-zero KML header artifacts)
                    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                        continue
                    if abs(lon) < 0.01 and abs(lat) < 0.01:
                        continue  # skip (0,0) origin artifacts
                    gx, gy = transformer.transform(lon, lat)
                    # Reject invalid/degenerate UTM results (e.g. negative easting)
                    if math.isnan(gx) or math.isnan(gy) or gx <= 0.0 or gy <= 0.0:
                        continue
                    pts.append((gx, gy))
                except (ValueError, TypeError):
                    continue
        return pts

    def _find_topo_source(self) -> Optional[Path]:
        """Locate topographic shapefile directory from config and common paths."""
        paths_cfg = self.config.get("paths", {})
        candidates = [
            paths_cfg.get("topomap_shp_dir"),
            paths_cfg.get("topomap_dir"),
            _PROJECT_ROOT / "Kanthi_Topomap_SHP",
            _PROJECT_ROOT.parent / "Kanthi_Topomap_SHP",
        ]
        for c in candidates:
            if c:
                p = Path(c)
                if not p.is_absolute():
                    # Check both relative to project root and workspace root
                    for base in [_PROJECT_ROOT, _PROJECT_ROOT.parent]:
                        if (base / p).exists():
                            return (base / p).resolve()
                if p.exists():
                    return p.resolve()
        return None

    def _parse_kml_topo(self, kml_path: Path, epsg_code: str = "EPSG:32643") -> list[TopoFeature]:
        """Parse Placemarks with LineStrings and Polygons from KML/KMZ into TopoFeature objects."""
        features: list[TopoFeature] = []
        try:

            transformer = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)

            kml_content: bytes
            if kml_path.suffix.lower() == ".kmz":
                with zipfile.ZipFile(kml_path, "r") as zf:
                    kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                    if not kml_names:
                        return []
                    kml_content = zf.read(kml_names[0])
            else:
                with open(kml_path, "rb") as f:
                    kml_content = f.read()

            root = ET.fromstring(kml_content)

            def parse_container(container: Any, current_folder_name: str = "") -> None:
                for child in container:
                    tag = child.tag.split("}")[-1]
                    if tag in ["Folder", "Document"]:
                        f_name = current_folder_name
                        for f_child in child:
                            if f_child.tag.split("}")[-1] == "name" and f_child.text:
                                f_name = f_child.text.strip()
                        parse_container(child, f_name)
                    elif tag == "Placemark":
                        name = ""
                        desc = ""
                        kml_color = ""
                        for p_child in child:
                            p_tag = p_child.tag.split("}")[-1]
                            if p_tag == "name" and p_child.text:
                                name = p_child.text.strip()
                            elif p_tag == "description" and p_child.text:
                                desc = p_child.text.strip()

                        for c_elem in child.iter():
                            c_tag = c_elem.tag.split("}")[-1]
                            if c_tag == "color" and c_elem.text:
                                kml_color = c_elem.text.strip().lower()
                                break

                        full_text = f"{current_folder_name} {name} {desc}".upper()

                        has_geom = any(c.tag.split("}")[-1] in ["Polygon", "LineString", "MultiGeometry"] for c in child.iter())

                        # Skip unwanted CAD annotations (only if not a real spatial geometry)
                        if not has_geom:
                            is_unwanted = (
                                "BLOCK REFERENCE" in full_text
                                or any(kw in full_text for kw in [
                                    "CURVE NO", "CURVE_DETAILS", "CURVE DETAILS", "CURVE_CALLOUT", "CURVE CALLOUT",
                                    "TL=", "R=", "DELTA=", "RADIUS=", "DEGREE OF CURVE", "IP_NO", "APEX DISTANCE",
                                    "GRADIENT DETAILS", "GRADIENT CALLOUT", "FEL", "F.L.", "F.L",
                                    "FORMATION LEVEL", "PVI", "1 IN ", "SLOPE", "FALLING", "RISING",
                                    "HORIZONTAL F",
                                ])
                            )
                            if is_unwanted:
                                continue

                        # Classify with full context (folder name + placemark name)
                        feat_name = name.strip() if name.strip() else current_folder_name.strip()
                        classify_text = f"{current_folder_name} {name}".strip() if current_folder_name else name
                        feat_type, layer_name, color, lw = self._classify_topo_feature(classify_text, kml_color)

                        # 1. Parse Polygons (e.g. station yard boundaries, sub-station enclosures, land parcels)
                        for poly_elem in child.iter():
                            if poly_elem.tag.split("}")[-1] == "Polygon":
                                for c_node in poly_elem.iter():
                                    if c_node.tag.split("}")[-1] == "coordinates" and c_node.text:
                                        pts = self._parse_coord_string(c_node.text, transformer)
                                        if len(pts) >= 3:
                                            # Unclassified closed polygons default to railway/survey boundary
                                            p_type = "boundary" if feat_type in ["overlay", "boundary"] else feat_type
                                            p_layer = "RAILWAY_BOUNDARY" if feat_type in ["overlay", "boundary"] else layer_name
                                            p_color = self.COL_WHITE if feat_type in ["overlay", "boundary"] else color
                                            p_lw = 25 if feat_type in ["overlay", "boundary"] else lw
                                            features.append(TopoFeature(
                                                geometry=Polygon(pts),
                                                feature_type=p_type,
                                                name=feat_name,
                                                layer_name=p_layer,
                                                color=p_color,
                                                lineweight=p_lw,
                                                source_file=kml_path.name,
                                                source_type="kml",
                                            ))
                                            break

                        # 2. Parse LineStrings
                        for ls_elem in child.iter():
                            if ls_elem.tag.split("}")[-1] == "LineString":
                                for c_node in ls_elem.iter():
                                    if c_node.tag.split("}")[-1] == "coordinates" and c_node.text:
                                        pts = self._parse_coord_string(c_node.text, transformer)
                                        if len(pts) >= 2:
                                            features.append(TopoFeature(
                                                geometry=LineString(pts),
                                                feature_type=feat_type,
                                                name=feat_name,
                                                layer_name=layer_name,
                                                color=color,
                                                lineweight=lw,
                                                source_file=kml_path.name,
                                                source_type="kml",
                                            ))
                                            break

                        # 3. Parse Points
                        for pt_elem in child.iter():
                            pt_tag = pt_elem.tag.split("}")[-1]
                            if pt_tag == "Point":
                                for c_node in pt_elem.iter():
                                    if c_node.tag.split("}")[-1] == "coordinates" and c_node.text:
                                        pts = self._parse_coord_string(c_node.text, transformer)
                                        if pts:
                                            features.append(TopoFeature(
                                                geometry=Point(pts[0]),
                                                feature_type=feat_type,
                                                name=feat_name,
                                                layer_name=layer_name,
                                                color=color,
                                                lineweight=lw,
                                                source_file=kml_path.name,
                                                source_type="kml",
                                            ))

            parse_container(root)
        except Exception as exc:
            self.logger.warning("Failed to parse KML topo file %s: %s", kml_path.name, exc)

        return features

    def _parse_shp_topo(self, shp_path: Path, epsg_code: str = "EPSG:32643") -> list[TopoFeature]:
        """Parse shapefiles (.shp) using GeoPandas into TopoFeature objects."""
        features: list[TopoFeature] = []
        try:
            import geopandas as gpd
            if shp_path.is_dir():
                files = list(shp_path.rglob("*.shp"))
            else:
                files = [shp_path] if shp_path.suffix.lower() == ".shp" else []

            # Filter by selected shapefiles if configured
            selected_files = (
                self.config.get("paths", {}).get("selected_shp_files")
                or self.config.get("drawing", {}).get("plan_view", {}).get("selected_shp_files", [])
            )
            if selected_files and isinstance(selected_files, list):
                selected_stems = {Path(s).stem.upper() for s in selected_files}
                files = [f for f in files if f.stem.upper() in selected_stems or f.name.upper() in [Path(s).name.upper() for s in selected_files]]
                self.logger.info("Filtered to %d user-selected shapefile(s): %s", len(files), [f.name for f in files])
            else:
                self.logger.info("Found %d shapefile(s) in %s", len(files), shp_path)

            for f in files:
                try:
                    gdf = gpd.read_file(f)
                    if gdf.empty:
                        continue

                    # Handle CRS transformation
                    if gdf.crs:
                        if str(gdf.crs).upper() != epsg_code.upper():
                            gdf = gdf.to_crs(epsg_code)
                    else:
                        # Inspect coordinate values to infer if WGS84 (lon/lat) or UTM
                        first_geom = gdf.geometry.dropna().iloc[0] if not gdf.geometry.dropna().empty else None
                        if first_geom:
                            sample_pt = first_geom.representative_point()
                            if -180.0 <= sample_pt.x <= 180.0 and -90.0 <= sample_pt.y <= 90.0:
                                gdf = gdf.set_crs("EPSG:4326").to_crs(epsg_code)
                            else:
                                gdf = gdf.set_crs(epsg_code)

                    stem_name = f.stem.upper()
                    feat_type, layer_name, color, lw = self._classify_topo_feature(stem_name)

                    for _, row in gdf.iterrows():
                        geom = row.geometry
                        if geom is not None and not geom.is_empty:
                            # Search through common shapefile attribute columns
                            name_val = stem_name
                            for col_candidate in ["name", "NAME", "Name", "layer", "LAYER", "Layer", "type", "TYPE", "Type", "feature", "FEATURE", "DESCRIPTIO", "description", "RefName", "Entity"]:
                                if col_candidate in row and pd.notna(row[col_candidate]):
                                    candidate_str = str(row[col_candidate]).strip()
                                    if candidate_str and candidate_str != "0" and candidate_str != "None":
                                        name_val = candidate_str
                                        break

                            f_type, l_name, col, l_w = self._classify_topo_feature(name_val) if name_val != stem_name else (feat_type, layer_name, color, lw)
                            features.append(TopoFeature(
                                geometry=geom,
                                feature_type=f_type,
                                name=name_val,
                                layer_name=l_name,
                                color=col,
                                lineweight=l_w,
                                source_file=stem_name,
                                source_type="shp",
                            ))
                except Exception as e_f:
                    self.logger.debug("Could not read shapefile %s: %s", f.name, e_f)
        except Exception as exc:
            self.logger.warning("Shapefile topo parsing error: %s", exc)

        return features

    def _extract_features_from_topo_res(self, topo_res: Any) -> list[TopoFeature]:
        """Extract vector footprint geometries from TopoAnalyzer result."""
        features: list[TopoFeature] = []
        if topo_res is None:
            return features

        structures = getattr(topo_res, "structures", [])
        for st in structures:
            st_name = getattr(st, "structure_name", "STRUCTURE")
            feat_type, layer_name, color, lw = self._classify_topo_feature(st_name)
            geom = getattr(st, "geometry", None)
            if geom is not None and hasattr(geom, "is_empty") and not geom.is_empty:
                features.append(TopoFeature(
                    geometry=geom,
                    feature_type=feat_type,
                    name=st_name,
                    layer_name=layer_name,
                    color=color,
                    lineweight=lw,
                    source_file="topo_res",
                    source_type="shp",
                ))

        infringements_df = getattr(topo_res, "infringements_df", None)
        if isinstance(infringements_df, pd.DataFrame) and not infringements_df.empty:
            for _, r in infringements_df.iterrows():
                b_type = str(r.get("building_type", "BUILDING"))
                geom = r.get("geometry", None)
                if geom is not None and hasattr(geom, "is_empty") and not geom.is_empty:
                    features.append(TopoFeature(
                        geometry=geom,
                        feature_type="building",
                        name=b_type,
                        layer_name="TOPO_BUILDING",
                        color=self.COL_WHITE,
                        lineweight=25,
                        source_file="topo_res",
                        source_type="shp",
                    ))

        return features

    def extract_bridges_from_kml(
        self,
        kml_source: Path | str | None = None,
        reconstructed_line: Optional[LineString] = None,
        epsg_code: Optional[str] = None,
    ) -> pd.DataFrame:
        """Extract bridge and culvert structure details directly from KML/KMZ and calculate alignment chainages."""
        if epsg_code is None:
            epsg_code = self.epsg_code

        paths_cfg = self.config.get("paths", {})
        candidates = [
            Path(kml_source) if kml_source else None,
            Path(paths_cfg.get("kml_path", "")) if paths_cfg.get("kml_path") else None,
            Path(paths_cfg.get("topomap_kml_path", "")) if paths_cfg.get("topomap_kml_path") else None,
            _PROJECT_ROOT / "OP_2_update.kml",
        ]

        target_kml: Optional[Path] = None
        for cand in candidates:
            if cand:
                if not cand.is_absolute():
                    cand = _PROJECT_ROOT / cand
                if cand.exists():
                    target_kml = cand
                    break

        if target_kml is None:
            self.logger.warning("No KML/KMZ source found for bridge extraction.")
            return pd.DataFrame()

        self.logger.info("Extracting bridge details from KML/KMZ: %s (CRS: %s)", target_kml.name, epsg_code)
        bridge_records: list[dict[str, Any]] = []

        try:
            transformer = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)

            kml_content: bytes
            if target_kml.suffix.lower() == ".kmz":
                with zipfile.ZipFile(target_kml, "r") as zf:
                    kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                    if not kml_names:
                        return pd.DataFrame()
                    kml_content = zf.read(kml_names[0])
            else:
                with open(target_kml, "rb") as f:
                    kml_content = f.read()

            root = ET.fromstring(kml_content)

            bridge_keywords = [
                "BRIDGE", "BR ", "BR.", "BR_", "BR-", "CULVERT", "ROB", "RUB",
                "PSC SLAB", "RCC BOX", "BOX CULVERT", "PIPE CULVERT", "MNB", "MJB",
                "MAJOR BRIDGE", "MINOR BRIDGE", "L-XING", "LC"
            ]

            seq_idx = 1
            for elem in root.iter():
                if elem.tag.endswith("Placemark"):
                    name = ""
                    desc = ""
                    for child in elem:
                        tag_name = child.tag.split("}")[-1]
                        if tag_name == "name" and child.text:
                            name = child.text.strip()
                        elif tag_name == "description" and child.text:
                            desc = child.text.strip()

                    full_text = f"{name} {desc}".upper()
                    is_bridge = any(kw in full_text for kw in bridge_keywords)
                    if not is_bridge:
                        continue

                    # Extract coordinates (Point or Line/Polygon centroid)
                    coords: list[tuple[float, float]] = []
                    for c_node in elem.iter():
                        if c_node.tag.split("}")[-1] == "coordinates" and c_node.text:
                            pts = self._parse_coord_string(c_node.text, transformer)
                            coords.extend(pts)

                    if not coords:
                        continue

                    pt_x = sum(p[0] for p in coords) / len(coords)
                    pt_y = sum(p[1] for p in coords) / len(coords)

                    # Compute chainage along alignment
                    ch_m = 0.0
                    if reconstructed_line is not None and not reconstructed_line.is_empty:
                        ch_m = float(reconstructed_line.project(Point(pt_x, pt_y)))

                    # Parse bridge number
                    br_no = ""
                    m_br = re.search(r"(?:BRIDGE|BR|BR\.|BR_|BR-)\s*(?:NO\.?|NUMBER)?\s*([0-9A-Za-z\/\-]+)", full_text)
                    if m_br:
                        br_no = m_br.group(1).strip()
                    else:
                        m_no = re.search(r"(?:NO\.?)\s*([0-9A-Za-z\/\-]+)", full_text)
                        if m_no:
                            br_no = m_no.group(1).strip()
                        else:
                            br_no = str(seq_idx)

                    # Parse bridge type / span config
                    br_type = ""
                    m_span = re.search(r"(\d+\s*[xX]\s*[\d\.]+\s*M?(?:\s*[A-Za-z\s]+)?)", full_text)
                    if m_span:
                        br_type = m_span.group(1).strip()
                    else:
                        for kw in ["ROB", "RUB", "RCC BOX CULVERT", "BOX CULVERT", "PIPE CULVERT", "PSC SLAB", "MAJOR BRIDGE", "MINOR BRIDGE", "CULVERT", "BRIDGE"]:
                            if kw in full_text:
                                br_type = kw
                                break
                    if not br_type:
                        br_type = name if name else "CULVERT"

                    # Parse category (Existing vs Proposed)
                    is_exg = "EXG" in full_text or "EXISTING" in full_text or "OLD" in full_text
                    cat = "EXISTING" if is_exg else "PROPOSED"

                    bridge_records.append({
                        "s_no": seq_idx,
                        "bridge_no": br_no,
                        "chainage_m": round(ch_m, 2),
                        "bridge_type": br_type,
                        "category": cat,
                        "remarks": name,
                    })
                    seq_idx += 1

            if bridge_records:
                df = pd.DataFrame(bridge_records)
                df = df.sort_values(by="chainage_m").reset_index(drop=True)
                df["s_no"] = range(1, len(df) + 1)
                self.logger.info("Successfully extracted %d bridge records directly from KML: %s", len(df), target_kml.name)
                return df
            else:
                self.logger.warning("No bridge placemarks matching bridge keywords were found in %s", target_kml.name)
        except Exception as exc:
            self.logger.warning("Failed to extract bridges from KML %s: %s", target_kml.name, exc)

        return pd.DataFrame()

    def load_shp_topo_features(
        self,
        topo_source: Path | str | None = None,
        epsg_code: Optional[str] = None,
        topo_res: Optional[Any] = None,
    ) -> list[TopoFeature]:
        """Discover, parse, and return all topographic features strictly from Shapefiles (SHP) in UTM ground coordinates."""
        if not (self.show_topo_layer or self.show_shp_topo_layer):
            return []

        if epsg_code is None:
            epsg_code = self.epsg_code

        features: list[TopoFeature] = []

        # 1. From TopoAnalyzer result if provided
        if topo_res is not None:
            features.extend(self._extract_features_from_topo_res(topo_res))

        # 2. Resolve shapefile source path robustly
        src_path: Optional[Path] = None
        if topo_source:
            p = Path(topo_source)
            if p.is_dir() or p.suffix.lower() == ".shp":
                if p.exists():
                    src_path = p.resolve()
                elif (_PROJECT_ROOT / p).exists():
                    src_path = (_PROJECT_ROOT / p).resolve()
                elif (_PROJECT_ROOT.parent / p).exists():
                    src_path = (_PROJECT_ROOT.parent / p).resolve()

        if src_path is None:
            src_path = self._find_topo_source()

        if src_path is not None and src_path.exists():
            self.logger.info("Loading Topographic Map features strictly from Shapefile(s): %s (CRS: %s)", src_path.name, epsg_code)
            features.extend(self._parse_shp_topo(src_path, epsg_code))
        else:
            self.logger.warning("No topographic shapefile directory found at %s. Please check topomap_shp_dir in config.yaml.", src_path)

        return features

    def load_kml_topo_features(
        self,
        kml_source: Path | str | None = None,
        epsg_code: Optional[str] = None,
    ) -> list[TopoFeature]:
        """Parse and return survey boundary and topographic features from KML/KMZ."""
        if not (self.show_topo_layer or self.show_kml_topo_layer):
            return []

        if epsg_code is None:
            epsg_code = self.epsg_code

        features: list[TopoFeature] = []
        kml_cand = kml_source or self.config.get("paths", {}).get("topomap_kml_path")
        if kml_cand:
            kml_p = Path(kml_cand)
            if not kml_p.is_absolute():
                for base in [_PROJECT_ROOT, _PROJECT_ROOT.parent]:
                    if (base / kml_p).exists():
                        kml_p = (base / kml_p).resolve()
                        break
            if kml_p.exists():
                kml_feats = self._parse_kml_topo(kml_p, epsg_code)
                b_count = 0
                for kf in kml_feats:
                    # Include all KML vector features except proposed centerline alignment
                    if kf.feature_type != "alignment":
                        features.append(kf)
                        b_count += 1
                if b_count > 0:
                    self.logger.info("Loaded %d survey boundary & topo features from KML: %s", b_count, kml_p.name)
            else:
                self.logger.warning("KML topo file not found at %s", kml_p)

        if not features:
            return []

        # Merge fragmented line segments (e.g. 2-point tessellated segments from CAD KML exports)
        # into continuous polylines so that linetypes (like DASHDOTDOT) render properly in CAD
        from collections import defaultdict
        from shapely.ops import linemerge

        groups = defaultdict(list)
        merged_features: list[TopoFeature] = []

        for f in features:
            if f.geometry and f.geometry.geom_type in ("LineString", "MultiLineString"):
                key = (f.feature_type, f.layer_name, f.source_file, f.source_type, f.color, f.lineweight)
                groups[key].append(f)
            else:
                merged_features.append(f)

        for (feat_type, layer_name, src_file, src_type, col, lw), f_list in groups.items():
            geoms = [f.geometry for f in f_list]
            try:
                merged = linemerge(geoms)
                if isinstance(merged, MultiLineString):
                    lines = list(merged.geoms)
                elif isinstance(merged, LineString):
                    lines = [merged]
                else:
                    lines = geoms
            except Exception:
                lines = geoms

            for line in lines:
                try:
                    line = line.simplify(0.05, preserve_topology=True)
                except Exception:
                    pass
                merged_features.append(TopoFeature(
                    geometry=line,
                    feature_type=feat_type,
                    name=f_list[0].name,
                    layer_name=layer_name,
                    color=col,
                    lineweight=lw,
                    source_file=src_file,
                    source_type=src_type,
                ))

        if len(features) != len(merged_features):
            self.logger.info(
                "Merged %d fragmented KML line segments into %d continuous polylines for proper CAD linetype display.",
                len(features), len(merged_features),
            )

        return merged_features

    def load_topo_features(
        self,
        topo_source: Path | str | None = None,
        epsg_code: Optional[str] = None,
        topo_res: Optional[Any] = None,
        kml_source: Path | str | None = None,
    ) -> list[TopoFeature]:
        """Discover, parse, and return both Shapefile and KML topographic features in UTM ground coordinates."""
        if not self.show_topo_layer and not self.show_shp_topo_layer and not self.show_kml_topo_layer:
            return []

        if self._cached_topo_features is not None and topo_source is None and topo_res is None and kml_source is None:
            return self._cached_topo_features

        if epsg_code is None:
            epsg_code = self.epsg_code

        features: list[TopoFeature] = []
        if self.show_shp_topo_layer:
            features.extend(self.load_shp_topo_features(topo_source=topo_source, epsg_code=epsg_code, topo_res=topo_res))
        if self.show_kml_topo_layer:
            features.extend(self.load_kml_topo_features(kml_source=kml_source, epsg_code=epsg_code))

        total_feats = len(features)
        self.logger.info("Loaded %d total topographic (SHP & KML) background features.", total_feats)

        self._cached_topo_features = features
        self._cached_epsg = epsg_code
        return features

    @staticmethod
    def _clip_paper_pts_to_bounds(
        paper_pts: list[tuple[float, float]],
        clip_bounds: Optional[tuple[float, float, float, float]],
    ) -> list[list[tuple[float, float]]]:
        """Hard-clip polyline paper coordinates to sheet bounds using Shapely intersection."""
        if clip_bounds is None or len(paper_pts) < 2:
            return [paper_pts]

        min_x, max_x, min_y, max_y = clip_bounds
        xs = [pt[0] for pt in paper_pts]
        ys = [pt[1] for pt in paper_pts]

        # Fast reject if completely outside bounding box
        if max(xs) < min_x - 50.0 or min(xs) > max_x + 50.0 or max(ys) < min_y - 50.0 or min(ys) > max_y + 50.0:
            return []

        # Check if any vertex lies outside sheet bounds
        needs_clip = any(
            pt[0] < min_x or pt[0] > max_x or pt[1] < min_y or pt[1] > max_y
            for pt in paper_pts
        )
        if not needs_clip:
            try:
                simp = LineString(paper_pts).simplify(0.02, preserve_topology=True)
                return [list(simp.coords)]
            except Exception:
                return [paper_pts]

        try:
            clip_box = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
            paper_line = LineString(paper_pts)
            clipped = paper_line.intersection(clip_box)
            if clipped.is_empty:
                return []

            raw_segs: list[list[tuple[float, float]]] = []
            if clipped.geom_type == "LineString":
                raw_segs = [list(clipped.coords)]
            elif clipped.geom_type == "MultiLineString":
                raw_segs = [list(seg.coords) for seg in clipped.geoms]

            cleaned: list[list[tuple[float, float]]] = []
            for seg in raw_segs:
                if len(seg) >= 2:
                    ls = LineString(seg)
                    if ls.length > 0.5:
                        simp = ls.simplify(0.02, preserve_topology=True)
                        cleaned.append(list(simp.coords))
            return cleaned if cleaned else raw_segs
        except Exception:
            return [paper_pts]

    def _draw_shp_topo_layer(
        self,
        msp: Any,
        features: list[TopoFeature],
        line_geom: LineString,
        start_ch: float,
        end_ch: float,
        coord_transform: Callable[[float, float], tuple[float, float]],
        clip_bounds: Optional[tuple[float, float, float, float]] = None,
        full_line_geom: Optional[LineString] = None,
        anno_scale: float = 5.0,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Render Shapefile (SHP) topographic survey features onto Plan View (TOPO_* layers)."""
        if not features or not self.show_shp_topo_layer or line_geom is None or line_geom.is_empty:
            return

        apply_distance_filter = not self.kml_only_mode
        apply_length_filter = not self.kml_only_mode

        # Build a sheet corridor ROI polygon to strictly clip all features to this sheet's 5km chunk
        sheet_roi: Optional[Any] = None
        total_len = float(line_geom.length) if line_geom is not None else 0.0
        if end_ch - start_ch < total_len - 50.0:
            s_0 = max(0.0, start_ch - 80.0)
            s_1 = min(total_len, end_ch + 80.0)
            n_samples = max(20, int((s_1 - s_0) / 10.0))
            chs = np.linspace(s_0, s_1, n_samples)
            roi_pts = [line_geom.interpolate(s) for s in chs]
            if len(roi_pts) >= 2:
                corridor_w = max(500.0, self.max_topo_offset_m * 1.5)
                sheet_roi = LineString(roi_pts).buffer(corridor_w)

        ref_line: Optional[LineString] = None
        if apply_distance_filter:
            ref_line = full_line_geom if (full_line_geom is not None and not full_line_geom.is_empty) else line_geom

        drawn_count = 0
        skipped_dist = 0
        skipped_len = 0

        for feat in features:
            if self.only_outer_boundary and feat.feature_type != "boundary":
                continue

            name_u = str(feat.name).upper()
            layer_u = str(feat.layer_name).upper()

            if "BLOCK REFERENCE" in f"{name_u} {layer_u}":
                continue

            geom = feat.geometry
            if geom is None or geom.is_empty:
                continue

            # Prevent drawing duplicate survey raw centerline linestrings over proposed track centerline
            if ref_line is not None and feat.feature_type in ["alignment", "existing_track"] and isinstance(geom, (LineString, MultiLineString)):
                try:
                    if float(ref_line.distance(geom)) < 1.0 and float(geom.length) > 10.0:
                        continue
                except Exception:
                    pass

            # Strict sheet corridor clipping
            if sheet_roi is not None:
                if geom.disjoint(sheet_roi):
                    continue
                try:
                    clipped_geom = geom.intersection(sheet_roi)
                    if clipped_geom.is_empty:
                        continue
                    geom = clipped_geom
                except Exception:
                    pass

            # Distance filter — exempt survey boundaries & yard limits
            if apply_distance_filter and ref_line is not None and feat.feature_type not in ["boundary", "existing_boundary", "land_acquisition"]:
                try:
                    dist = float(ref_line.distance(geom))
                    if dist > self.max_topo_offset_m:
                        skipped_dist += 1
                        continue
                except Exception:
                    pass

            coord_seqs: list[list[tuple[float, float]]] = []
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                if not apply_length_filter or geom.length >= self.min_topo_boundary_length_m:
                    coord_seqs.append(coords)
                else:
                    skipped_len += 1
            elif isinstance(geom, Polygon):
                if not apply_length_filter or geom.exterior.length >= self.min_topo_boundary_length_m:
                    coord_seqs.append(list(geom.exterior.coords))
                    for interior in geom.interiors:
                        coord_seqs.append(list(interior.coords))
                else:
                    skipped_len += 1
            elif isinstance(geom, MultiLineString):
                for ls in geom.geoms:
                    if not apply_length_filter or ls.length >= self.min_topo_boundary_length_m:
                        coord_seqs.append(list(ls.coords))
                    else:
                        skipped_len += 1
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    if not apply_length_filter or poly.exterior.length >= self.min_topo_boundary_length_m:
                        coord_seqs.append(list(poly.exterior.coords))
                    else:
                        skipped_len += 1
            elif isinstance(geom, Point):
                coord_seqs.append([(geom.x, geom.y)])

            for raw_coords in coord_seqs:
                if len(raw_coords) == 1:
                    x_utm, y_utm = raw_coords[0]
                    px, py = coord_transform(x_utm, y_utm)
                    if clip_bounds is not None:
                        min_x, max_x, min_y, max_y = clip_bounds
                        if not (min_x <= px <= max_x and min_y <= py <= max_y):
                            continue
                    try:
                        scale_a = anno_scale if (anno_scale is not None and anno_scale > 0) else 1.0
                        label = feat.name.strip() if feat.name else ""

                        if feat.feature_type == "tbm" or "TBM" in label.upper():
                            r_tbm = 2.0 * scale_a
                            msp.add_circle((px, py), radius=r_tbm, dxfattribs={"layer": "TOPO_SURVEY", "color": self.COL_BLACK, "lineweight": 25})
                            msp.add_line((px - r_tbm, py), (px + r_tbm, py), dxfattribs={"layer": "TOPO_SURVEY", "color": self.COL_BLACK, "lineweight": 18})
                            msp.add_line((px, py - r_tbm), (px, py + r_tbm), dxfattribs={"layer": "TOPO_SURVEY", "color": self.COL_BLACK, "lineweight": 18})
                            hatch = msp.add_hatch(color=self.COL_BLACK, dxfattribs={"layer": "TOPO_SURVEY"})
                            hatch.paths.add_polyline_path([(px, py), (px, py + r_tbm), (px - r_tbm, py), (px, py)], is_closed=True)
                            hatch.paths.add_polyline_path([(px, py), (px, py - r_tbm), (px + r_tbm, py), (px, py)], is_closed=True)
                            drawn_count += 1
                        elif feat.feature_type == "electric_pole" or "POLE" in label.upper() or "MAST" in label.upper():
                            ph = 2.2 * scale_a
                            p_pole = [(px - 0.7 * scale_a, py + ph), (px + 0.7 * scale_a, py + ph), (px + 1.2 * scale_a, py - ph), (px - 1.2 * scale_a, py - ph), (px - 0.7 * scale_a, py + ph)]
                            msp.add_lwpolyline(p_pole, dxfattribs={"layer": "TOPO_UTILITY", "color": self.COL_BLACK, "lineweight": 25})
                            drawn_count += 1

                        if self._should_display_feature_text(feat):
                            txt_label = str(feat.name).strip()
                            if txt_label and txt_label != "0" and txt_label.lower() != "none":
                                txt_h = max(1.5, 2.0 * scale_a)
                                text_col = self.COL_WHITE if feat.color == self.COL_BLACK else feat.color
                                style_name = "ESP_STYLE" if hasattr(msp, "doc") and "ESP_STYLE" in msp.doc.styles else "Standard"
                                txt_ent = msp.add_text(
                                    text=txt_label,
                                    dxfattribs={"layer": "TOPO_TEXT", "color": text_col, "height": txt_h, "style": style_name},
                                )
                                txt_ent.set_placement((px, py + 3.0 * scale_a), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
                    except Exception:
                        pass
                    continue

                paper_pts = [coord_transform(x_utm, y_utm) for x_utm, y_utm in raw_coords]
                if len(paper_pts) < 2:
                    continue

                clipped_segs = self._clip_paper_pts_to_bounds(paper_pts, clip_bounds)

                try:
                    cad_l = feat.layer_name
                    col = feat.color
                    lw = feat.lineweight

                    if feat.feature_type == "alignment":
                        continue

                    # Determine specific DXF attributes per feature type (TOPO_* layers)
                    if feat.feature_type == "boundary" or cad_l == "RAILWAY_BOUNDARY":
                        cad_l = "RAILWAY_BOUNDARY"
                        col = self.COL_WHITE
                        lw = 25
                        dxf_attrs = {
                            "layer": cad_l, "color": col, "lineweight": lw,
                            "linetype": "RAILWAY_ROW", "ltscale": 1.0, "flags": 128,
                        }
                    elif feat.feature_type == "existing_boundary":
                        dxf_attrs = {"layer": "EXISTING_ROW", "color": self.COL_BLACK, "lineweight": 25, "linetype": "DASHED"}
                    elif feat.feature_type == "dismantle":
                        dxf_attrs = {"layer": "TOPO_DISMANTLE", "color": self.COL_GREEN, "lineweight": 25, "linetype": "DASHED"}
                    elif feat.feature_type == "kutcha_road":
                        dxf_attrs = {"layer": "TOPO_KUTCHA_ROAD", "color": self.COL_MAGENTA, "lineweight": 18, "linetype": "DASHED"}
                    elif feat.feature_type == "land_acquisition":
                        dxf_attrs = {"layer": "LAND_ACQUISITION", "color": self.COL_RED, "lineweight": 18}
                    elif feat.feature_type == "retaining_wall":
                        dxf_attrs = {"layer": "TOPO_RETAINING_WALL", "color": self.COL_CYAN, "lineweight": 25}
                    elif feat.feature_type == "existing_track":
                        dxf_attrs = {"layer": "EXISTING_TRACK", "color": self.COL_BLACK, "lineweight": 35}
                    elif feat.feature_type == "future_track":
                        dxf_attrs = {"layer": "FUTURE_TRACK", "color": self.COL_MAGENTA, "lineweight": 35}
                    elif feat.feature_type == "water":
                        dxf_attrs = {"layer": "TOPO_WATER", "color": self.COL_BLUE, "lineweight": 25}
                    elif feat.feature_type == "drain":
                        dxf_attrs = {"layer": "TOPO_DRAIN", "color": self.COL_BLUE, "lineweight": 25}
                    elif feat.feature_type == "road":
                        dxf_attrs = {"layer": "TOPO_ROAD", "color": self.COL_BLACK, "lineweight": 25}
                    elif feat.feature_type == "power_line":
                        dxf_attrs = {
                            "layer": "TOPO_POWER_LINE", "color": self.COL_PINK,
                            "lineweight": 18, "linetype": "POWER_LINE", "ltscale": 1.0, "flags": 128,
                        }
                    elif feat.feature_type == "building":
                        dxf_attrs = {"layer": "TOPO_BUILDING", "color": self.COL_WHITE, "lineweight": 25}
                    else:
                        dxf_attrs = {"layer": cad_l, "color": col, "lineweight": lw}

                    for seg_pts in clipped_segs:
                        if len(seg_pts) >= 2:
                            msp.add_lwpolyline(seg_pts, dxfattribs=dxf_attrs)
                    drawn_count += 1
                except Exception as e_draw:
                    self.logger.debug("Skipping shapefile polyline: %s", e_draw)

            if self._should_display_feature_text(feat):
                scale_a = anno_scale if (anno_scale is not None and anno_scale > 0) else 1.0
                self._draw_feature_text_label(
                    msp=msp,
                    feat=feat,
                    coord_transform=coord_transform,
                    clip_bounds=clip_bounds,
                    scale_a=scale_a,
                    collision_mgr=collision_mgr,
                    line_geom=line_geom,
                )

        self.logger.info(
            "Rendered %d Shapefile Topo entities onto Plan View (Ch %.1fm - %.1fm). Skipped: dist_filter=%d, len_filter=%d.",
            drawn_count, start_ch, end_ch, skipped_dist, skipped_len,
        )

    def _draw_kml_topo_layer(
        self,
        msp: Any,
        features: list[TopoFeature],
        line_geom: LineString,
        start_ch: float,
        end_ch: float,
        coord_transform: Callable[[float, float], tuple[float, float]],
        clip_bounds: Optional[tuple[float, float, float, float]] = None,
        full_line_geom: Optional[LineString] = None,
        anno_scale: float = 5.0,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Render KML/KMZ survey & background features onto Plan View (KML_* separate layers)."""
        if not features or not self.show_kml_topo_layer or line_geom is None or line_geom.is_empty:
            return

        # Build sheet corridor ROI polygon (300m lateral corridor for KML features)
        sheet_roi: Optional[Any] = None
        total_len = float(line_geom.length) if line_geom is not None else 0.0
        if end_ch - start_ch < total_len - 50.0:
            s_0 = max(0.0, start_ch - 80.0)
            s_1 = min(total_len, end_ch + 80.0)
            n_samples = max(20, int((s_1 - s_0) / 10.0))
            chs = np.linspace(s_0, s_1, n_samples)
            roi_pts = [line_geom.interpolate(s) for s in chs]
            if len(roi_pts) >= 2:
                corridor_w = 300.0 if self.kml_only_mode else max(500.0, self.max_topo_offset_m * 1.5)
                sheet_roi = LineString(roi_pts).buffer(corridor_w)

        ref_line = full_line_geom if (full_line_geom is not None and not full_line_geom.is_empty) else line_geom
        drawn_count = 0

        for feat in features:
            if self.only_outer_boundary and feat.feature_type != "boundary":
                continue

            geom = feat.geometry
            if geom is None or geom.is_empty:
                continue

            # Prevent drawing duplicate KML raw centerline linestrings over proposed CAD alignment
            if feat.feature_type == "alignment":
                continue
            if ref_line is not None and isinstance(geom, (LineString, MultiLineString)):
                try:
                    if float(ref_line.distance(geom)) < 1.0 and float(geom.length) > 10.0:
                        continue
                except Exception:
                    pass

            # Corridor clip
            if sheet_roi is not None:
                if geom.disjoint(sheet_roi):
                    continue
                try:
                    clipped_geom = geom.intersection(sheet_roi)
                    if clipped_geom.is_empty:
                        continue
                    geom = clipped_geom
                except Exception:
                    pass

            coord_seqs: list[list[tuple[float, float]]] = []
            if isinstance(geom, LineString):
                coord_seqs.append(list(geom.coords))
            elif isinstance(geom, Polygon):
                coord_seqs.append(list(geom.exterior.coords))
                for interior in geom.interiors:
                    coord_seqs.append(list(interior.coords))
            elif isinstance(geom, MultiLineString):
                for ls in geom.geoms:
                    coord_seqs.append(list(ls.coords))
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    coord_seqs.append(list(poly.exterior.coords))
            elif isinstance(geom, Point):
                coord_seqs.append([(geom.x, geom.y)])

            # Render onto dedicated KML_* layers
            for raw_coords in coord_seqs:
                if len(raw_coords) == 1:
                    x_utm, y_utm = raw_coords[0]
                    px, py = coord_transform(x_utm, y_utm)
                    if clip_bounds is not None:
                        min_x, max_x, min_y, max_y = clip_bounds
                        if not (min_x <= px <= max_x and min_y <= py <= max_y):
                            continue
                    try:
                        scale_a = anno_scale if (anno_scale is not None and anno_scale > 0) else 1.0
                        label = feat.name.strip() if feat.name else ""
                        if feat.feature_type == "tbm" or "TBM" in label.upper():
                            r_tbm = 2.0 * scale_a
                            msp.add_circle((px, py), radius=r_tbm, dxfattribs={"layer": "KML_SURVEY", "color": self.COL_BLACK, "lineweight": 25})
                            msp.add_line((px - r_tbm, py), (px + r_tbm, py), dxfattribs={"layer": "KML_SURVEY", "color": self.COL_BLACK, "lineweight": 18})
                            msp.add_line((px, py - r_tbm), (px, py + r_tbm), dxfattribs={"layer": "KML_SURVEY", "color": self.COL_BLACK, "lineweight": 18})
                            drawn_count += 1
                        elif feat.feature_type == "electric_pole" or "POLE" in label.upper() or "MAST" in label.upper():
                            ph = 2.2 * scale_a
                            p_pole = [(px - 0.7 * scale_a, py + ph), (px + 0.7 * scale_a, py + ph), (px + 1.2 * scale_a, py - ph), (px - 1.2 * scale_a, py - ph), (px - 0.7 * scale_a, py + ph)]
                            msp.add_lwpolyline(p_pole, dxfattribs={"layer": "KML_UTILITY", "color": self.COL_BLACK, "lineweight": 25})
                            drawn_count += 1
                        elif label and label != "0" and label.lower() != "none" and not label.startswith("Polyline"):
                            txt_h = max(1.5, 2.0 * scale_a)
                            style_name = "ESP_STYLE" if hasattr(msp, "doc") and "ESP_STYLE" in msp.doc.styles else "Standard"
                            txt_ent = msp.add_text(
                                text=label,
                                dxfattribs={"layer": "KML_TEXT", "color": self.COL_WHITE, "height": txt_h, "style": style_name},
                            )
                            txt_ent.set_placement((px, py + 2.5 * scale_a), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
                    except Exception:
                        pass
                    continue

                paper_pts = [coord_transform(x_utm, y_utm) for x_utm, y_utm in raw_coords]
                if len(paper_pts) < 2:
                    continue

                clipped_segs = self._clip_paper_pts_to_bounds(paper_pts, clip_bounds)

                try:
                    # Assign dedicated KML layers for clear separation from Shapefiles
                    if feat.feature_type in ["boundary", "existing_boundary"]:
                        dxf_attrs = {
                            "layer": "KML_BOUNDARY",
                            "color": self.kml_color,
                            "lineweight": self.kml_lineweight,
                            "linetype": self.kml_linetype,
                            "ltscale": 1.0,
                            "flags": 128,
                        }
                    elif feat.feature_type in ["road", "kutcha_road"]:
                        dxf_attrs = {"layer": "KML_ROAD", "color": self.COL_BLACK, "lineweight": 25}
                    elif feat.feature_type in ["water", "drain"]:
                        dxf_attrs = {"layer": "KML_WATER", "color": self.COL_BLUE, "lineweight": 25}
                    elif feat.feature_type == "building":
                        dxf_attrs = {"layer": "KML_BUILDING", "color": self.COL_WHITE, "lineweight": 25}
                    elif feat.feature_type == "power_line":
                        dxf_attrs = {
                            "layer": "KML_POWER_LINE", "color": self.COL_PINK,
                            "lineweight": 18, "linetype": "POWER_LINE", "ltscale": 1.0, "flags": 128,
                        }
                    elif feat.feature_type in ["existing_track", "future_track"]:
                        dxf_attrs = {"layer": "KML_TRACK", "color": self.COL_BLACK, "lineweight": 35}
                    else:
                        dxf_attrs = {
                            "layer": "KML_OVERLAY",
                            "color": self.kml_color,
                            "lineweight": self.kml_lineweight,
                            "linetype": self.kml_linetype,
                            "ltscale": 1.0,
                            "flags": 128,
                        }

                    for seg_pts in clipped_segs:
                        if len(seg_pts) >= 2:
                            msp.add_lwpolyline(seg_pts, dxfattribs=dxf_attrs)
                    drawn_count += 1
                except Exception as e_draw:
                    self.logger.debug("Skipping KML polyline: %s", e_draw)

        # Render boundary distance change transition callouts (e.g. Ch. 11830 M / 15 M, 50 M)
        bound_cfg = self.config.get("railway_boundary", {})
        show_labels = bool(bound_cfg.get("show_offset_change_labels", True))
        if show_labels and ref_line is not None and not ref_line.is_empty:
            boundary_geoms: list[LineString] = []
            for feat in features:
                f_name_u = str(feat.name).upper()
                f_layer_u = str(feat.layer_name).upper()
                is_b = (
                    feat.feature_type in ["boundary", "existing_boundary"]
                    or "BOUNDARY" in f_layer_u
                    or "BOUNDARY" in f_name_u
                    or "ROW" in f_name_u
                    or "ROW" in f_layer_u
                    or "LAND" in f_name_u
                    or "LAND" in f_layer_u
                )
                if is_b:
                    geom = feat.geometry
                    if isinstance(geom, LineString):
                        boundary_geoms.append(geom)
                    elif isinstance(geom, MultiLineString):
                        boundary_geoms.extend(geom.geoms)
                    elif isinstance(geom, Polygon):
                        boundary_geoms.append(LineString(geom.exterior.coords))
                    elif isinstance(geom, MultiPolygon):
                        for p in geom.geoms:
                            boundary_geoms.append(LineString(p.exterior.coords))

            # Fallback: if no features explicitly tagged as boundary, check polylines within corridor
            if not boundary_geoms:
                for feat in features:
                    if feat.feature_type == "alignment":
                        continue
                    geom = feat.geometry
                    if geom is not None and isinstance(geom, (LineString, MultiLineString)) and geom.length > 20.0:
                        d = float(ref_line.distance(geom))
                        if 5.0 <= d <= 150.0:
                            if isinstance(geom, LineString):
                                boundary_geoms.append(geom)
                            elif isinstance(geom, MultiLineString):
                                boundary_geoms.extend(geom.geoms)

            if boundary_geoms:
                min_step = float(bound_cfg.get("min_offset_change_threshold_m", 1.0))
                label_color = int(bound_cfg.get("color", self.kml_color))
                layer_text = str(bound_cfg.get("layer_text", "KML_BOUNDARY"))
                scale_a = anno_scale if (anno_scale is not None and anno_scale > 0) else 1.0
                text_h = float(bound_cfg.get("text_height_mm", 2.5)) * scale_a

                transitions = self._detect_kml_boundary_transitions(
                    boundary_lines=boundary_geoms,
                    centerline=ref_line,
                    start_ch=start_ch,
                    end_ch=end_ch,
                    min_step_m=min_step,
                )
                for tr in transitions:
                    self._draw_boundary_transition_callout(
                        msp=msp,
                        transition=tr,
                        coord_transform_fn=coord_transform,
                        layer_text=layer_text,
                        layer_line="KML_BOUNDARY",
                        color=label_color,
                        text_h=text_h,
                        anno_scale=scale_a,
                        collision_mgr=collision_mgr,
                    )

        self.logger.info(
            "Rendered %d KML entities onto Plan View (Ch %.1fm - %.1fm) on separate KML_* layers.",
            drawn_count, start_ch, end_ch,
        )

    def _draw_topo_layer(
        self,
        msp: Any,
        features: list[TopoFeature],
        line_geom: LineString,
        start_ch: float,
        end_ch: float,
        coord_transform: Callable[[float, float], tuple[float, float]],
        clip_bounds: Optional[tuple[float, float, float, float]] = None,
        full_line_geom: Optional[LineString] = None,
        anno_scale: float = 5.0,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Render both Shapefile and KML topographic features onto Plan View using dedicated layer functions.

        Shapefile features are rendered via _draw_shp_topo_layer (TOPO_* layers).
        KML features are rendered via _draw_kml_topo_layer (KML_* separate layers).
        """
        if not features or not self.show_topo_layer or line_geom is None or line_geom.is_empty:
            return

        shp_feats: list[TopoFeature] = []
        kml_feats: list[TopoFeature] = []

        for f in features:
            s_type = getattr(f, "source_type", "").lower()
            if s_type == "kml":
                kml_feats.append(f)
            elif s_type == "shp":
                shp_feats.append(f)
            else:
                # Untagged: check filename or default to shp
                sf = getattr(f, "source_file", "").lower()
                if sf.endswith(".kml") or sf.endswith(".kmz"):
                    kml_feats.append(f)
                else:
                    shp_feats.append(f)

        if shp_feats and self.show_shp_topo_layer:
            self._draw_shp_topo_layer(
                msp=msp,
                features=shp_feats,
                line_geom=line_geom,
                start_ch=start_ch,
                end_ch=end_ch,
                coord_transform=coord_transform,
                clip_bounds=clip_bounds,
                full_line_geom=full_line_geom,
                anno_scale=anno_scale,
                collision_mgr=collision_mgr,
            )

        if kml_feats and self.show_kml_topo_layer:
            self._draw_kml_topo_layer(
                msp=msp,
                features=kml_feats,
                line_geom=line_geom,
                start_ch=start_ch,
                end_ch=end_ch,
                coord_transform=coord_transform,
                clip_bounds=clip_bounds,
                full_line_geom=full_line_geom,
                anno_scale=anno_scale,
                collision_mgr=collision_mgr,
            )

    def _should_display_feature_text(self, feat: TopoFeature) -> bool:
        """Check if the given feature matches the user-selected text layers."""
        if not self.selected_shp_text_layers:
            return False

        source_stem = getattr(feat, "source_file", "").upper().strip()
        feat_name_u = str(feat.name).strip().upper()
        feat_layer_u = str(feat.layer_name).strip().upper()

        if source_stem and source_stem in self.selected_shp_text_layers:
            return True
        if feat_name_u and feat_name_u in self.selected_shp_text_layers:
            return True
        if feat_layer_u and feat_layer_u in self.selected_shp_text_layers:
            return True

        for target in self.selected_shp_text_layers:
            if not target:
                continue
            if source_stem and (target == source_stem or target in source_stem or source_stem in target):
                return True
            if len(target) >= 3 and (target in feat_name_u or feat_name_u in target):
                return True

        return False

    def _draw_feature_text_label(
        self,
        msp: Any,
        feat: TopoFeature,
        coord_transform: Callable[[float, float], tuple[float, float]],
        clip_bounds: Optional[tuple[float, float, float, float]] = None,
        scale_a: float = 1.0,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
        line_geom: Optional[LineString] = None,
    ) -> None:
        """Render text annotation for a selected topographic feature at its centroid/representative point with collision avoidance."""
        txt_label = str(feat.name).strip()
        if not txt_label or txt_label == "0" or txt_label.lower() == "none" or feat.geometry is None or feat.geometry.is_empty:
            return

        try:
            rep_pt = feat.geometry.representative_point()

            # 1. Suppress topo labels within the track formation clearance zone
            if line_geom is not None and not line_geom.is_empty:
                clearance_dist = max(5.0, (self.formation_width_m / 2.0) + 2.0)
                if feat.geometry.distance(line_geom) < clearance_dist:
                    return

            tx, ty = coord_transform(rep_pt.x, rep_pt.y)

            if clip_bounds is not None:
                min_x, max_x, min_y, max_y = clip_bounds
                if not (min_x <= tx <= max_x and min_y <= ty <= max_y):
                    return

            txt_h = max(1.5, 2.0 * scale_a)
            text_w = self._calc_text_width(txt_label, text_height=txt_h)

            # 2. Collision check against high-priority railway infrastructure callouts
            cand_box = PlanTextCollisionManager.estimate_text_box(
                tx, ty, text_w=text_w, text_h=txt_h, rot_deg=0.0, align="MIDDLE_CENTER"
            )
            if collision_mgr is not None and not collision_mgr.is_clear(cand_box):
                return

            if collision_mgr is not None:
                collision_mgr.register_box(cand_box)

            text_color = self.COL_WHITE if feat.color == self.COL_BLACK else feat.color
            style_name = "ESP_STYLE" if hasattr(msp, "doc") and "ESP_STYLE" in msp.doc.styles else "Standard"

            add_masked_mtext(
                msp=msp,
                text=txt_label,
                insert=(tx, ty),
                char_height=txt_h,
                rotation_deg=0.0,
                layer="TOPO_TEXT",
                color=text_color,
                style=style_name,
                attachment_point=MTextEntityAlignment.MIDDLE_CENTER,
                bg_scale=1.05,
            )
        except Exception as exc:
            self.logger.debug("Could not render topo text label %s: %s", txt_label, exc)

    def _render_single_plan_doc(
        self,
        reconstructed_line: LineString,
        ip_df: pd.DataFrame,
        vip_df: Optional[pd.DataFrame],
        bridge_df: Optional[pd.DataFrame],
        start_ch: float,
        end_ch: float,
        sheet_index: int,
        total_sheets: int,
        dxf_sheet_path: Path,
        topo_features: Optional[list[TopoFeature]] = None,
    ) -> dict[str, Any]:
        sheet_ip_df = pd.DataFrame()
        if not ip_df.empty:
            sheet_ip_df = ip_df[
                (ip_df["chainage_m"] >= start_ch - 1500.0)
                & (ip_df["chainage_m"] <= end_ch + 1500.0)
            ].copy()

        doc, msp = self._setup_document()

        # Dynamic sheet framing based on track bounding box in True Metres
        sample_chs = np.linspace(start_ch, end_ch, max(100, int((end_ch - start_ch) / 5.0)))
        pts_2d = [reconstructed_line.interpolate(s) for s in sample_chs]
        p_start, p_end = pts_2d[0], pts_2d[-1]
        dx_b, dy_b = p_end.x - p_start.x, p_end.y - p_start.y
        baseline_len = math.hypot(dx_b, dy_b)
        theta = math.atan2(dy_b, dx_b) if baseline_len > 1.0 else 0.0
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        u_vals = [(pt.x - p_start.x) * cos_t + (pt.y - p_start.y) * sin_t for pt in pts_2d]
        v_vals = [-(pt.x - p_start.x) * sin_t + (pt.y - p_start.y) * cos_t for pt in pts_2d]
        u_min, u_max = min(u_vals), max(u_vals)
        v_min, v_max = min(v_vals), max(v_vals)

        # Enclosing Sheet Dimensions in Metres (ISO A0 Aspect Ratio)
        sheet_w = max(5500.0, (u_max - u_min) + 600.0)
        sheet_h = max(sheet_w * (841.0 / 1189.0), (v_min + v_max) + 1200.0)
        # Re-balance width if height was expanded
        sheet_w = max(sheet_w, sheet_h * (1189.0 / 841.0))
        margin = sheet_w * (10.0 / 1189.0)
        anno_scale = sheet_w / 1189.0

        origin_x = 300.0 - u_min
        origin_y = (sheet_h / 2.0) - ((v_min + v_max) / 2.0)

        # 1. Draw Title Block & Outer Sheet Border
        self._draw_sheet_border(msp, sheet_w, sheet_h, margin)
        self._draw_title_block(msp, sheet_index=sheet_index, total_sheets=total_sheets, start_ch=start_ch, end_ch=end_ch, sheet_w=sheet_w, sheet_h=sheet_h, margin=margin, anno_scale=anno_scale)

        # 1b. Draw Complete KML / Topographic Survey Background Layer in METRES
        if topo_features:
            def transform_doc(x_m: float, y_m: float) -> tuple[float, float]:
                u = (x_m - p_start.x) * cos_t + (y_m - p_start.y) * sin_t
                v = -(x_m - p_start.x) * sin_t + (y_m - p_start.y) * cos_t
                return origin_x + u, origin_y + v

            clip_bounds_doc = (margin + 2.0, sheet_w - margin - 2.0, margin + 2.0, sheet_h - margin - 2.0)
            # self._draw_topo_layer(
            #     msp=msp,
            #     features=topo_features,
            #     line_geom=reconstructed_line,
            #     start_ch=start_ch,
            #     end_ch=end_ch,
            #     coord_transform=transform_doc,
            #     clip_bounds=clip_bounds_doc,
            #     full_line_geom=reconstructed_line,
            #     anno_scale=anno_scale,
            # )

        # Define track/station to paper position mapping for callout annotations
        def get_paper_pos(ch_m: float) -> tuple[float, float, float]:
            s = min(float(reconstructed_line.length), max(0.0, ch_m))
            pt = reconstructed_line.interpolate(s)
            u = (pt.x - p_start.x) * cos_t + (pt.y - p_start.y) * sin_t
            v = -(pt.x - p_start.x) * sin_t + (pt.y - p_start.y) * cos_t
            px = origin_x + u
            py = origin_y + v
            ds = 0.5
            s_bwd = max(0.0, s - ds)
            s_fwd = min(float(reconstructed_line.length), s + ds)
            pt_b = reconstructed_line.interpolate(s_bwd)
            pt_f = reconstructed_line.interpolate(s_fwd)
            dx = pt_f.x - pt_b.x
            dy = pt_f.y - pt_b.y
            u_fwd = dx * cos_t + dy * sin_t
            v_fwd = -dx * sin_t + dy * cos_t
            heading = math.atan2(v_fwd, u_fwd) if (abs(u_fwd) > 1e-6 or abs(v_fwd) > 1e-6) else 0.0
            return px, py, heading

        # Unified Spatial Text Collision Manager for this sheet
        collision_mgr = PlanTextCollisionManager(buffer_padding=0.25 * anno_scale)

        # 2. Draw Reconstructed Alignment Track (True AutoCAD ARCs, Transition Spirals, Tangents, Station Ticks)
        get_paper_pos = self._draw_track_geometry(
            msp=msp,
            line_geom=reconstructed_line,
            ip_df=sheet_ip_df,
            start_ch=start_ch,
            end_ch=end_ch,
            origin_x=origin_x,
            origin_y=origin_y,
            anno_scale=anno_scale,
            collision_mgr=collision_mgr,
        )

        # 2b. Draw Railway Boundary (Right-Of-Way)
        if self.config.get("railway_boundary", {}).get("enabled", True):
            self._draw_railway_boundary(
                msp=msp,
                get_pos_fn=get_paper_pos,
                start_ch=start_ch,
                end_ch=end_ch,
                scale_factor=1.0,
                anno_scale=anno_scale,
                bridge_df=bridge_df,
                valign_res=vip_df,
                collision_mgr=collision_mgr,
            )

        # 3. Draw Track Callouts
        # Curve Callouts (Apex detail boxes & Tangent Point Flags)
        if not sheet_ip_df.empty:
            self._draw_curve_callouts(
                msp=msp,
                ip_df=sheet_ip_df,
                get_paper_pos=get_paper_pos,
                start_ch=start_ch,
                end_ch=end_ch,
                anno_scale=anno_scale,
                collision_mgr=collision_mgr,
            )
            self._draw_curve_tangent_points(
                msp=msp,
                ip_df=sheet_ip_df,
                get_paper_pos=get_paper_pos,
                start_ch=start_ch,
                end_ch=end_ch,
                anno_scale=anno_scale,
                collision_mgr=collision_mgr,
            )

        # Bridge / Structure Callouts (rendered only when bridge_df schedule is provided)
        if bridge_df is not None and not bridge_df.empty:
            self._draw_bridge_callouts(
                msp=msp,
                bridge_df=bridge_df,
                get_paper_pos=get_paper_pos,
                start_ch=start_ch,
                end_ch=end_ch,
                anno_scale=anno_scale,
                collision_mgr=collision_mgr,
            )

        # Gradient Callouts (PVI flags & slope indicator arms along alignment)
        if vip_df is not None and not vip_df.empty:
            self._draw_gradient_callouts(
                msp=msp,
                vip_df=vip_df,
                get_paper_pos=get_paper_pos,
                start_ch=start_ch,
                end_ch=end_ch,
                anno_scale=anno_scale,
                collision_mgr=collision_mgr,
            )

        # Station Centerline Flags (rendered when station schedule is available)
        self._draw_station_flags(
            msp=msp,
            station_df=None,
            get_paper_pos=get_paper_pos,
            start_ch=start_ch,
            end_ch=end_ch,
            anno_scale=anno_scale,
            collision_mgr=collision_mgr,
            scale_factor=1.0,
        )

        # 4. Draw Curve Table (Bottom of Plan View)
        curve_table_w = 0.0
        if not sheet_ip_df.empty:
            curve_table_w = sum([14.0, 14.0, 22.0, 22.0, 22.0, 22.0, 20.0, 16.0, 18.0, 18.0, 16.0, 20.0, 20.0, 20.0, 18.0, 20.0]) * anno_scale
            self._draw_curve_table(
                msp=msp,
                ip_df=sheet_ip_df,
                x=margin + (15.0 * anno_scale),
                y=margin + (8.0 * anno_scale),
                anno_scale=anno_scale,
            )

        # 4b. Draw Gradient Table
        grad_table_w = 0.0
        if vip_df is not None and not vip_df.empty:
            grad_table_w = sum([14.0, 22.0, 22.0, 18.0, 22.0, 18.0, 18.0, 20.0]) * anno_scale
            x_grad = (margin + 15.0 * anno_scale + curve_table_w + 10.0 * anno_scale) if (not sheet_ip_df.empty) else (margin + 15.0 * anno_scale)
            self._draw_gradient_table(
                msp=msp,
                vip_df=vip_df,
                x=x_grad,
                y=margin + (8.0 * anno_scale),
                anno_scale=anno_scale,
                start_ch=start_ch,
                end_ch=end_ch,
            )

        # 4c. Draw Bridge / Structure Table
        if bridge_df is not None and not bridge_df.empty:
            x_br = margin + 15.0 * anno_scale
            if not sheet_ip_df.empty:
                x_br += curve_table_w + 10.0 * anno_scale
            if vip_df is not None and not vip_df.empty:
                x_br += grad_table_w + 10.0 * anno_scale
            self._draw_bridge_table(
                msp=msp,
                bridge_df=bridge_df,
                x=x_br,
                y=margin + (8.0 * anno_scale),
                anno_scale=anno_scale,
                start_ch=start_ch,
                end_ch=end_ch,
            )

        # 5. Draw North Arrow & Scale (always rendered)
        self._draw_north_arrow_and_scale(
            msp=msp,
            x=origin_x + 50.0 * anno_scale,
            y=sheet_h - margin - (70.0 * anno_scale),
            anno_scale=anno_scale,
        )

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
                self.logger.warning("File '%s' is locked in AutoCAD. Trying next suffix...", try_path.name)
        if not saved:
            import time
            ts_path = dxf_sheet_path.with_stem(f"{dxf_sheet_path.stem}_{int(time.time())}")
            doc.saveas(str(ts_path))
            final_path = ts_path

        return {
            "sheet_index": sheet_index,
            "start_ch": start_ch,
            "end_ch": end_ch,
            "dxf_path": str(final_path),
        }

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

    def _resolve_vip_df(self, vip_df: Optional[pd.DataFrame], valign_res: Optional[Any] = None) -> Optional[pd.DataFrame]:
        """Resolve the VIP (Vertical Intersection Point) DataFrame from all available sources.

        Resolution priority:
          1. The ``vip_df`` argument if already populated with real elevation data.
          2. ``valign_res.vip_dataframe`` from a VerticalAlignment result object.
          3. ``outputs/vertical_ips.csv`` auto-saved by the vertical alignment module.
          4. The gradient Excel file configured under ``paths.gradient_excel_path``.

        Returns the resolved DataFrame, or None if no source is found.
        """
        # 1. Use caller-supplied df if it has real (non-placeholder) data
        if vip_df is not None and not vip_df.empty:
            if not (("elevation_m" in vip_df.columns) and (vip_df["elevation_m"] == 10.0).all()):
                return vip_df

        # 2. Try the vertical alignment result object
        if valign_res is not None:
            candidate = getattr(valign_res, "vip_dataframe", None)
            if candidate is not None and not candidate.empty:
                return candidate

        # 3. Try the auto-saved vertical_ips.csv
        v_path = self.outputs_dir / "vertical_ips.csv"
        if not v_path.is_absolute():
            v_path = _PROJECT_ROOT / v_path
        if v_path.exists():
            try:
                df = pd.read_csv(v_path)
                df.columns = [str(c).strip() for c in df.columns]
                self.logger.info("Auto-loaded %d vertical IP records from %s", len(df), v_path.name)
                return df
            except Exception as exc:
                self.logger.warning("Could not auto-load vertical_ips.csv: %s", exc)

        # 4. Try the gradient Excel configured in paths
        grad_excel_cfg = self.config.get("paths", {}).get("gradient_excel_path")
        if grad_excel_cfg:
            ge_path = Path(grad_excel_cfg)
            if not ge_path.is_absolute():
                ge_path = _PROJECT_ROOT / ge_path
            if ge_path.exists():
                try:
                    from modules.vertical_alignment import VerticalAlignment
                    va = VerticalAlignment(self.config)
                    df = va.read_gradient_excel(ge_path)
                    self.logger.info("Auto-loaded %d vertical IP records from %s", len(df), ge_path.name)
                    return df
                except Exception as exc:
                    self.logger.warning("Could not auto-load gradient excel from %s: %s", ge_path.name, exc)

        return vip_df  # return whatever was passed (could be None or empty)

    def draw_alignment_plan(
        self,
        align_res: AlignmentResult,
        hgeom_res: Optional[HGeomResult] = None,
        valign_res: Optional[Any] = None,
        bridge_df: Optional[pd.DataFrame] = None,
        vip_df: Optional[pd.DataFrame] = None,
        topo_res: Optional[Any] = None,
        topo_path: Optional[str | Path] = None,
        output_prefix: str = "alignment_plan",
    ) -> PlanResult:
        """Generate official Indian Railways Alignment Plan DXF drawings."""
        self.logger.info("Starting Alignment Plan DXF Generation...")
        self.ap_dir.mkdir(parents=True, exist_ok=True)

        # Determine target EPSG CRS
        epsg_str = self.epsg_code
        if hasattr(align_res, "utm_info") and align_res.utm_info:
            epsg_str = getattr(align_res.utm_info, "epsg_str", getattr(align_res.utm_info, "epsg", self.epsg_code))
            if isinstance(epsg_str, int):
                epsg_str = f"EPSG:{epsg_str}"
        elif hasattr(align_res, "epsg_code"):
            epsg_str = f"EPSG:{align_res.epsg_code}"

        # Load Topographic survey background features strictly from Shapefiles
        topo_src_to_use = topo_path or self._find_topo_source()
        topo_features = self.load_topo_features(topo_source=topo_src_to_use, epsg_code=epsg_str, topo_res=topo_res)


        vip_df = self._resolve_vip_df(vip_df, valign_res)

        if bridge_df is None or (isinstance(bridge_df, pd.DataFrame) and bridge_df.empty):
            bridge_df = pd.DataFrame()

        raw_line = getattr(align_res, "alignment_line", None)
        if raw_line is None:
            raise PlanDrawerError("Cannot generate Alignment Plan: Alignment line geometry is missing.")

        ip_df = getattr(hgeom_res, "ip_dataframe", None) if hgeom_res is not None else None
        if ip_df is None or ip_df.empty:
            self.logger.warning("No horizontal curves found in ip_dataframe; drawing straight alignment.")
            ip_df = pd.DataFrame()

        # Step 4: Reconstruct and edit the alignment line with true mathematical curves
        reconstructed_line = self.reconstruct_curved_alignment(raw_line, ip_df)
        total_length_m = float(reconstructed_line.length)

        # Do not extract bridge records from KML (only use official bridge list schedule if provided)
        if bridge_df is None or (isinstance(bridge_df, pd.DataFrame) and bridge_df.empty):
            bridge_df = pd.DataFrame()

        sheet_dxf_paths: list[Path] = []
        sheets_meta: list[dict[str, Any]] = []

        # ── 1. Complete Continuous Plan Drawing (Full Length DXF) ──────────
        complete_dxf_path = self.ap_dir / f"{output_prefix}.dxf"
        should_gen_complete = self.generate_complete_profile or (self.max_sheet_length_m <= 0.0)

        if should_gen_complete:
            self.logger.info(
                "Generating complete continuous Alignment Plan (0.0 m to %.1f m)...",
                total_length_m
            )
            complete_meta = self._render_single_plan_doc(
                reconstructed_line=reconstructed_line,
                ip_df=ip_df,
                vip_df=vip_df,
                bridge_df=bridge_df,
                start_ch=0.0,
                end_ch=total_length_m,
                sheet_index=1,
                total_sheets=1,
                dxf_sheet_path=complete_dxf_path,
                topo_features=topo_features,
            )
            sheets_meta.append(complete_meta)
            self.logger.info("Saved Complete Alignment Plan DXF: %s", complete_meta["dxf_path"])

        # ── 2. Multi-Sheet Chunked Drawings (Smart Curve-Aware Sheet Splits) ─────────
        should_gen_sheets = self.generate_sheet_drawings and (self.max_sheet_length_m > 0.0)
        if should_gen_sheets:
            sheet_ranges = self._compute_smart_sheet_splits(total_length_m, ip_df=ip_df)
            num_sheets = len(sheet_ranges)
            for i, (start_ch, end_ch) in enumerate(sheet_ranges):

                sheet_name = f"{output_prefix}_sheet{i+1}_km{start_ch/1000.0:.1f}-{end_ch/1000.0:.1f}"
                dxf_sheet_path = self.ap_dir / f"{sheet_name}.dxf"

                sheet_meta = self._render_single_plan_doc(
                    reconstructed_line=reconstructed_line,
                    ip_df=ip_df,
                    vip_df=vip_df,
                    bridge_df=bridge_df,
                    start_ch=start_ch,
                    end_ch=end_ch,
                    sheet_index=i + 1,
                    total_sheets=num_sheets,
                    dxf_sheet_path=dxf_sheet_path,
                    topo_features=topo_features,
                )
                sheet_dxf_paths.append(Path(sheet_meta["dxf_path"]))
                sheets_meta.append(sheet_meta)
                self.logger.info("Saved Alignment Plan DXF sheet %d/%d: %s", i + 1, num_sheets, sheet_meta["dxf_path"])

        dxf_paths = sheet_dxf_paths if sheet_dxf_paths else [complete_dxf_path]
        primary_dxf = sheet_dxf_paths[0] if sheet_dxf_paths else complete_dxf_path

        return PlanResult(
            dxf_path=primary_dxf,
            dxf_paths=dxf_paths,
            reconstructed_line=reconstructed_line,
            metadata={
                "num_sheets": len(dxf_paths),
                "total_length_m": total_length_m,
                "sheets": sheets_meta,
                "complete_dxf_path": str(complete_dxf_path) if should_gen_complete else None,
            },
        )


    def draw_plan_onto_composite_sheet(
        self,
        msp: Any,
        reconstructed_line: Optional[LineString],
        ip_df: Optional[pd.DataFrame],
        start_ch: float,
        end_ch: float,
        origin_x: float,
        plan_origin_y: float,
        plan_bot_y: float,
        plan_top_y: float,
        draw_len_mm: float,
        sheet_width: float,
        sheet_height: float,
        margin_right: float = 20.0,
        margin_top: float = 20.0,
        bridge_df: Optional[pd.DataFrame] = None,
        vip_df: Optional[pd.DataFrame] = None,
        valign_res: Optional[Any] = None,
        topo_res: Optional[Any] = None,
        topo_path: Optional[str | Path] = None,
        epsg_code: Optional[str] = None,
        align_res: Optional[Any] = None,
    ) -> None:
        """Render the 2D Plan View (track centerline, 100m ticks, curve callouts, table, north arrow, topo layer)
        onto the top section of a shared composite DXF sheet.
        """
        # Outer Plan Box
        msp.add_lwpolyline(
            [
                (origin_x, plan_bot_y),
                (origin_x + draw_len_mm, plan_bot_y),
                (origin_x + draw_len_mm, plan_top_y),
                (origin_x, plan_top_y),
                (origin_x, plan_bot_y),
            ],
            dxfattribs={"layer": "SHEET_BORDER", "color": self.COL_BLACK, "lineweight": 25},
        )

        draw_length_m = max(1.0, end_ch - start_ch)
        scale_factor = (draw_len_mm / draw_length_m) if draw_length_m > 0 else 1.0

        # Extract bridge records directly from KML/KMZ
        # if bridge_df is None or (isinstance(bridge_df, pd.DataFrame) and bridge_df.empty):
        #     bridge_df = self.extract_bridges_from_kml(
        #         kml_source=topo_path,
        #         reconstructed_line=reconstructed_line,
        #         epsg_code=epsg_code or self.epsg_code,
        #     )
        
        if bridge_df is None or (isinstance(bridge_df, pd.DataFrame) and bridge_df.empty):
            br_path_cfg = self.config.get("paths", {}).get("bridge_list_path", "D:/ESP/Bridge_List_OP2.xlsx")
            if br_path_cfg:
                br_path = Path(br_path_cfg)
                if not br_path.is_absolute():
                    br_path = _PROJECT_ROOT / br_path
                if br_path.exists():
                    try:
                        if br_path.suffix.lower() in [".xlsx", ".xls"]:
                            try:
                                bridge_df = pd.read_excel(br_path, sheet_name="Bridge List", header=0)
                            except Exception:
                                bridge_df = pd.read_excel(br_path, header=0)
                        else:
                            bridge_df = pd.read_csv(br_path)
                        if bridge_df is not None:
                            bridge_df.columns = [str(c).strip() for c in bridge_df.columns]
                    except Exception as e_br:
                        self.logger.warning("Could not auto-load bridge list in composite sheet: %s", e_br)

        if reconstructed_line is None or reconstructed_line.is_empty:
            return

        plan_area_h = plan_top_y - plan_bot_y
        num_pts = max(100, int(draw_length_m / 10.0))
        sample_chs = np.linspace(start_ch, end_ch, num_pts)
        pts_2d = [reconstructed_line.interpolate(s) for s in sample_chs]
        p_start, p_end = pts_2d[0], pts_2d[-1]
        dx_b, dy_b = p_end.x - p_start.x, p_end.y - p_start.y
        b_len = math.hypot(dx_b, dy_b)
        theta = math.atan2(dy_b, dx_b) if b_len > 1.0 else 0.0
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        u_vals = [(pt.x - p_start.x) * cos_t + (pt.y - p_start.y) * sin_t for pt in pts_2d]
        v_vals = [-(pt.x - p_start.x) * sin_t + (pt.y - p_start.y) * cos_t for pt in pts_2d]
        u_min, u_max = min(u_vals), max(u_vals)
        v_min, v_max = min(v_vals), max(v_vals)
        v_mid = (v_min + v_max) / 2.0

        # Infer EPSG CRS from alignment result if not explicitly passed
        if epsg_code is None and align_res is not None:
            if hasattr(align_res, "utm_info") and align_res.utm_info:
                epsg_code = getattr(align_res.utm_info, "epsg_str", getattr(align_res.utm_info, "epsg", self.epsg_code))
                if isinstance(epsg_code, int):
                    epsg_code = f"EPSG:{epsg_code}"
            elif hasattr(align_res, "epsg_code"):
                epsg_code = f"EPSG:{align_res.epsg_code}"

        # Unified Spatial Text Collision Manager for Composite Sheet Plan View
        collision_mgr = PlanTextCollisionManager(buffer_padding=0.25)

        # ── 0. Render Topo Background Layer strictly from Shapefiles ──
        topo_src_to_use = topo_path or self._find_topo_source()
        topo_feats = self.load_topo_features(topo_source=topo_src_to_use, epsg_code=epsg_code or self.epsg_code, topo_res=topo_res)
        def transform_comp(x_m: float, y_m: float) -> tuple[float, float]:
            u = (x_m - p_start.x) * cos_t + (y_m - p_start.y) * sin_t
            v = -(x_m - p_start.x) * sin_t + (y_m - p_start.y) * cos_t
            px = origin_x + (u - u_min) * scale_factor
            py = plan_origin_y + (v - v_mid) * scale_factor
            return px, py

        if topo_feats:
            clip_bounds = (origin_x + 1.0, origin_x + draw_len_mm - 1.0, plan_bot_y + 1.5, plan_top_y - 1.5)
            self._draw_topo_layer(
                msp=msp,
                features=topo_feats,
                line_geom=reconstructed_line,
                start_ch=start_ch,
                end_ch=end_ch,
                coord_transform=transform_comp,
                clip_bounds=clip_bounds,
                full_line_geom=reconstructed_line,
                anno_scale=1.0,
                collision_mgr=collision_mgr,
            )

        def _get_pos(ch: float) -> tuple[float, float, float]:
            s = min(float(reconstructed_line.length), max(0.0, ch))
            pt = reconstructed_line.interpolate(s)
            u = (pt.x - p_start.x) * cos_t + (pt.y - p_start.y) * sin_t
            v = -(pt.x - p_start.x) * sin_t + (pt.y - p_start.y) * cos_t
            px = origin_x + (u - u_min) * scale_factor
            py = plan_origin_y + (v - v_mid) * scale_factor

            # Smooth Central Difference for exact tangent heading
            ds = 0.5
            s_bwd = max(0.0, s - ds)
            s_fwd = min(float(reconstructed_line.length), s + ds)
            pt_bwd = reconstructed_line.interpolate(s_bwd)
            pt_fwd = reconstructed_line.interpolate(s_fwd)
            dx_t = pt_fwd.x - pt_bwd.x
            dy_t = pt_fwd.y - pt_bwd.y
            u_fwd = dx_t * cos_t + dy_t * sin_t
            v_fwd = -dx_t * sin_t + dy_t * cos_t
            heading_rad = math.atan2(v_fwd, u_fwd) if (abs(u_fwd) > 1e-6 or abs(v_fwd) > 1e-6) else 0.0
            return px, py, heading_rad

        # ── 1. Render Piecewise Track Geometry (Centerline, Rails, Formation, Boundary ROW) ──
        valid_curves = []
        if ip_df is not None and not ip_df.empty:
            for _, r in ip_df.iterrows():
                rad = float(r.get("radius_m", 0.0))
                if rad <= 0:
                    continue
                ip_ch = float(r.get("chainage_m", 0.0))
                t_len = float(r.get("tangent_length_m", 0.0))
                tptc = float(r.get("tptc_chainage_m", max(0.0, ip_ch - t_len)))
                tpcc = float(r.get("tpcc_chainage_m", tptc))
                ctpc = float(r.get("ctpc_chainage_m", tpcc))
                ctpt = float(r.get("ctpt_chainage_m", ip_ch + t_len))
                direction = str(r.get("direction", "RHS")).upper()
                is_right_turn = "R" in direction
                if ctpt > start_ch and tptc < end_ch:
                    valid_curves.append((tptc, tpcc, ctpc, ctpt, rad, is_right_turn))
        valid_curves.sort(key=lambda x: x[0])

        # ── 1. Render Piecewise Track Geometry (Tangents, Clothoid Spirals, Native CAD ARCs) ──
        curr_ch = start_ch
        for tptc, tpcc, ctpc, ctpt, rad, is_right_turn in valid_curves:
            # 1. Straight Tangent Section (Single clean line without segmentation)
            t_s = max(curr_ch, start_ch)
            t_e = min(tptc, end_ch)
            if t_s < t_e:
                p_tan_s = _get_pos(t_s)[:2]
                p_tan_e = _get_pos(t_e)[:2]
                msp.add_line(
                    p_tan_s,
                    p_tan_e,
                    dxfattribs={"layer": "ALIGNMENT", "color": self.COL_RED, "lineweight": 35},
                )

            # 2. Entry Transition Spiral 1 (TP-TC to TP-CC: Clothoid Spiral)
            tr1_s = max(start_ch, tptc)
            tr1_e = min(end_ch, tpcc)
            if tr1_s < tr1_e:
                chs = np.linspace(tr1_s, tr1_e, max(10, int((tr1_e - tr1_s) / 5.0)))
                pts_tr1 = [_get_pos(c)[:2] for c in chs]
                if len(pts_tr1) >= 2:
                    msp.add_lwpolyline(pts_tr1, dxfattribs={"layer": "ALIGNMENT", "color": self.COL_RED, "lineweight": 35})

            # 3. Circular Curve (TP-CC to CT-PC: Constant Curvature -> True AutoCAD ARC)
            c_s = max(start_ch, tpcc)
            c_e = min(end_ch, ctpc)
            if c_s < c_e:
                c_mid = (c_s + c_e) / 2.0
                p1 = _get_pos(c_s)[:2]
                pm = _get_pos(c_mid)[:2]
                p2 = _get_pos(c_e)[:2]

                arc_res = self._fit_arc_from_3_points(p1, pm, p2)
                if arc_res is None:
                    arc_res = self._arc_from_2_points_and_radius(p1, p2, radius=rad * scale_factor, pm=pm, is_right_turn=is_right_turn)
                if arc_res is not None:
                    center, r_exact, sa, ea = arc_res
                    msp.add_arc(
                        center=center,
                        radius=r_exact,
                        start_angle=sa,
                        end_angle=ea,
                        dxfattribs={"layer": "ALIGNMENT", "color": self.COL_RED, "lineweight": 35},
                    )

            # 4. Exit Transition Spiral 2 (CT-PC to CT-PT: Clothoid Spiral)
            tr2_s = max(start_ch, ctpc)
            tr2_e = min(end_ch, ctpt)
            if tr2_s < tr2_e:
                chs = np.linspace(tr2_s, tr2_e, max(10, int((tr2_e - tr2_s) / 5.0)))
                pts_tr2 = [_get_pos(c)[:2] for c in chs]
                if len(pts_tr2) >= 2:
                    msp.add_lwpolyline(pts_tr2, dxfattribs={"layer": "ALIGNMENT", "color": self.COL_RED, "lineweight": 35})

            curr_ch = max(curr_ch, ctpt)

        # Remaining straight section after last curve (Single clean line without segmentation)
        if curr_ch < end_ch:
            t_s = max(curr_ch, start_ch)
            t_e = end_ch
            p_tan_s = _get_pos(t_s)[:2]
            p_tan_e = _get_pos(t_e)[:2]
            msp.add_line(
                p_tan_s,
                p_tan_e,
                dxfattribs={"layer": "ALIGNMENT", "color": self.COL_RED, "lineweight": 35},
            )

        # Station Ticks across alignment (every 100m and 1km)
        tick_interval = self.tick_interval_m if self.tick_interval_m > 0 else 100.0
        first_tick = math.ceil(start_ch / tick_interval) * tick_interval
        curr_tick = first_tick
        while curr_tick <= end_ch:
            px, py, heading_rad = _get_pos(curr_tick)
            perp = heading_rad + (math.pi / 2.0)
            nx, ny = math.cos(perp), math.sin(perp)
            is_km = (int(round(curr_tick)) % 1000 == 0)
            tick_len = (6.0 if is_km else 3.5)
            p_tick_1 = (px - (tick_len / 2.0) * nx, py - (tick_len / 2.0) * ny)
            p_tick_2 = (px + (tick_len / 2.0) * nx, py + (tick_len / 2.0) * ny)
            msp.add_line(
                p_tick_1,
                p_tick_2,
                dxfattribs={"layer": "CHAINAGE_TEXT", "color": self.COL_RED, "lineweight": 18 if is_km else 13},
            )

            # Station chainage label
            if not self.show_chainage_labels:
                curr_tick += tick_interval
                continue

            ch_label = self.format_chainage(curr_tick)

            lbl_x = px + ((tick_len / 2.0) + 1.5) * nx
            lbl_y = py + ((tick_len / 2.0) + 1.5) * ny

            rot_deg = math.degrees(perp) % 360.0
            txt_h = 3.0
            txt_w = self._calc_text_width(ch_label, text_height=txt_h)

            tick_poly = PlanTextCollisionManager.estimate_text_box(
                lbl_x, lbl_y, text_w=txt_w, text_h=txt_h, rot_deg=rot_deg, align="MIDDLE_LEFT"
            )
            collision_mgr.register_box(tick_poly)

            add_masked_mtext(
                msp=msp,
                text=ch_label,
                insert=(lbl_x, lbl_y),
                char_height=txt_h,
                rotation_deg=float(rot_deg),
                layer="CHAINAGE_TEXT",
                color=self.COL_RED,
                style="ESP_STYLE",
                attachment_point=MTextEntityAlignment.MIDDLE_LEFT,
                bg_scale=1.05,
            )

            curr_tick += tick_interval

        vip_df = self._resolve_vip_df(vip_df, valign_res)

        # Draw Curve Callouts & Tangent Point Flags
        if ip_df is not None and not ip_df.empty:
            self._draw_curve_callouts(
                msp=msp,
                ip_df=ip_df,
                get_paper_pos=_get_pos,
                start_ch=start_ch,
                end_ch=end_ch,
                anno_scale=1.0,
                collision_mgr=collision_mgr,
            )
            self._draw_curve_tangent_points(
                msp=msp,
                ip_df=ip_df,
                get_paper_pos=_get_pos,
                start_ch=start_ch,
                end_ch=end_ch,
                anno_scale=1.0,
                collision_mgr=collision_mgr,
            )

        # Draw Bridge Callouts (rendered only when bridge_df schedule is provided)
        if bridge_df is not None and not bridge_df.empty:
            self._draw_bridge_callouts(
                msp=msp,
                bridge_df=bridge_df,
                get_paper_pos=_get_pos,
                start_ch=start_ch,
                end_ch=end_ch,
                anno_scale=1.0,
                collision_mgr=collision_mgr,
            )

        # Draw Gradient Callouts
        if vip_df is not None and not vip_df.empty:
            self._draw_gradient_callouts(
                msp=msp,
                vip_df=vip_df,
                get_paper_pos=_get_pos,
                start_ch=start_ch,
                end_ch=end_ch,
                anno_scale=1.0,
                collision_mgr=collision_mgr,
            )

        # Draw Station Centerline Flags (rendered when station schedule is available)
        self._draw_station_flags(
            msp=msp,
            station_df=None,
            get_paper_pos=_get_pos,
            start_ch=start_ch,
            end_ch=end_ch,
            anno_scale=1.0,
            collision_mgr=collision_mgr,
            scale_factor=scale_factor,
        )



        # Curve Summary Table (Bottom of Plan View Panel)
        curve_table_w = 0.0
        if ip_df is not None and not ip_df.empty:
            curve_table_w = sum([14.0, 14.0, 22.0, 22.0, 22.0, 22.0, 20.0, 16.0, 18.0, 18.0, 16.0, 20.0, 20.0, 20.0, 18.0, 20.0])
            self._draw_curve_table(
                msp=msp,
                ip_df=ip_df,
                x=origin_x + 15.0,
                y=plan_bot_y + 8.0,
                anno_scale=1.0,
            )

        # Gradient Summary Table (Bottom of Plan View Panel)
        grad_table_w = 0.0
        if vip_df is not None and not vip_df.empty:
            grad_table_w = sum([14.0, 22.0, 22.0, 18.0, 22.0, 18.0, 18.0, 20.0])
            x_grad = (origin_x + 15.0 + curve_table_w + 10.0) if (ip_df is not None and not ip_df.empty) else (origin_x + 15.0)
            self._draw_gradient_table(
                msp=msp,
                vip_df=vip_df,
                x=x_grad,
                y=plan_bot_y + 8.0,
                anno_scale=1.0,
                start_ch=start_ch,
                end_ch=end_ch,
            )

        # Bridge & Structure Summary Table (Bottom of Plan View Panel)
        if bridge_df is not None and not bridge_df.empty:
            x_br = origin_x + 15.0
            if ip_df is not None and not ip_df.empty:
                x_br += curve_table_w + 10.0
            if vip_df is not None and not vip_df.empty:
                x_br += grad_table_w + 10.0
            self._draw_bridge_table(
                msp=msp,
                bridge_df=bridge_df,
                x=x_br,
                y=plan_bot_y + 8.0,
                anno_scale=1.0,
                start_ch=start_ch,
                end_ch=end_ch,
            )

        # Plan View Header (Station direction arrows, Underlined PLAN title, and North Arrow)
        from_st = self.config.get("project", {}).get("from_station", "KANTI")
        to_st = self.config.get("project", {}).get("to_station", "EGRA")
        self._draw_plan_view_header(
            msp=msp,
            origin_x=origin_x,
            plan_top_y=plan_top_y,
            draw_len_mm=draw_len_mm,
            from_station=from_st,
            to_station=to_st,
            theta=theta,
        )

    def _extract_kml_boundary_lines(self, kml_path: Path, epsg_code: str = "EPSG:32643") -> list[LineString]:
        """Extract boundary / ROW polylines directly from KML/KMZ survey."""
        features = self._parse_kml_topo(kml_path, epsg_code)
        boundary_lines: list[LineString] = []
        for feat in features:
            f_name_u = str(feat.name).upper()
            f_layer_u = str(feat.layer_name).upper()
            is_boundary = (
                feat.feature_type in ["boundary", "existing_boundary"]
                or "BOUNDARY" in f_layer_u
                or "BOUNDARY" in f_name_u
                or "ROW" in f_name_u
                or "ROW" in f_layer_u
                or "LAND" in f_name_u
                or "LAND" in f_layer_u
            )
            if is_boundary:
                geom = feat.geometry
                if isinstance(geom, LineString):
                    boundary_lines.append(geom)
                elif isinstance(geom, MultiLineString):
                    boundary_lines.extend(geom.geoms)
                elif isinstance(geom, Polygon):
                    boundary_lines.append(LineString(geom.exterior.coords))
                elif isinstance(geom, MultiPolygon):
                    for p in geom.geoms:
                        boundary_lines.append(LineString(p.exterior.coords))
        return boundary_lines

    def _detect_kml_boundary_transitions(
        self,
        boundary_lines: list[LineString],
        centerline: LineString,
        start_ch: float,
        end_ch: float,
        min_step_m: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Detect stations where KML boundary changes perpendicular offset from centerline.

        Identifies perpendicular steps / jogs in boundary polylines where offset jumps
        (e.g. 15m to 50m) and captures:
            - chainage: station along alignment
            - d_before: distance from track centre before the chainage
            - d_after: distance from track centre after the chainage
            - pt1, pt2: start/end points of the transition jog
            - heading: tangent heading angle of alignment
            - side: "LHS" or "RHS"
        """
        transitions: list[dict[str, Any]] = []
        if not boundary_lines or centerline is None or centerline.is_empty:
            return transitions

        cl_len = float(centerline.length)
        bound_cfg = self.config.get("railway_boundary", {})
        eff_min_step = max(0.2, min_step_m - 0.1)
        max_ds = float(bound_cfg.get("max_step_span_m", 15.0))

        for b_line in boundary_lines:
            if b_line.is_empty or float(centerline.distance(b_line)) > 250.0:
                continue

            coords = list(b_line.coords)
            n_pts = len(coords)
            if n_pts < 2:
                continue

            # Compute along-track chainage & perpendicular distance for each vertex
            proj_data: list[tuple[float, float]] = []
            for pt in coords:
                p_sh = Point(pt[0], pt[1])
                s = float(centerline.project(p_sh))
                d = float(centerline.distance(p_sh))
                proj_data.append((s, d))

            for i in range(n_pts - 1):
                s1, d1 = proj_data[i]
                s2, d2 = proj_data[i + 1]
                ds = abs(s2 - s1)
                dd = abs(d2 - d1)

                # Step condition: perpendicular jog or rapid lateral change (with 0.1m floating point tolerance)
                if ds <= max_ds and dd >= eff_min_step:
                    ch = (s1 + s2) / 2.0
                    if not (start_ch - 10.0 <= ch <= end_ch + 10.0):
                        continue

                    # Determine before / after ordering based on chainage direction
                    if i > 0:
                        s_prev = proj_data[i - 1][0]
                        if s_prev < s1:
                            d_before, d_after = d1, d2
                        else:
                            d_before, d_after = d2, d1
                    elif i + 2 < n_pts:
                        s_next = proj_data[i + 2][0]
                        if s_next > s2:
                            d_before, d_after = d1, d2
                        else:
                            d_before, d_after = d2, d1
                    else:
                        if s1 <= s2:
                            d_before, d_after = d1, d2
                        else:
                            d_before, d_after = d2, d1

                    # Tangent heading of centerline at ch
                    s_mid = max(0.0, min(cl_len, ch))
                    p_fwd = centerline.interpolate(min(cl_len, s_mid + 0.5))
                    p_bwd = centerline.interpolate(max(0.0, s_mid - 0.5))
                    tx = p_fwd.x - p_bwd.x
                    ty = p_fwd.y - p_bwd.y
                    heading = math.atan2(ty, tx)

                    # Determine side (LHS vs RHS)
                    p_c = centerline.interpolate(s_mid)
                    mx = (coords[i][0] + coords[i + 1][0]) / 2.0
                    my = (coords[i][1] + coords[i + 1][1]) / 2.0
                    vx = mx - p_c.x
                    vy = my - p_c.y
                    cross = tx * vy - ty * vx
                    side = "LHS" if cross >= 0.0 else "RHS"

                    transitions.append({
                        "chainage": ch,
                        "d_before": d_before,
                        "d_after": d_after,
                        "pt1": coords[i],
                        "pt2": coords[i + 1],
                        "heading": heading,
                        "side": side,
                        "track_pt": (p_c.x, p_c.y),
                    })

        # Filter by requested side (e.g. single side "RHS", "LHS", "SINGLE", or "BOTH")
        target_side = str(bound_cfg.get("callout_side", "BOTH")).strip().upper()
        if target_side in ["RHS", "LHS"]:
            transitions = [tr for tr in transitions if tr.get("side", "").upper() == target_side]
        elif target_side == "SINGLE":
            # At each station where both LHS and RHS have transitions, keep only one (prefer RHS)
            # Retain transitions that only occur on LHS or only on RHS
            single_side_tr: list[dict[str, Any]] = []
            for tr in sorted(transitions, key=lambda x: (x["chainage"], 0 if x.get("side") == "RHS" else 1)):
                if not single_side_tr or abs(tr["chainage"] - single_side_tr[-1]["chainage"]) >= 5.0:
                    single_side_tr.append(tr)
            transitions = single_side_tr

        # Deduplicate transitions close to each other
        unique_tr: list[dict[str, Any]] = []
        for tr in sorted(transitions, key=lambda x: x["chainage"]):
            if not unique_tr:
                unique_tr.append(tr)
            else:
                prev = unique_tr[-1]
                if abs(tr["chainage"] - prev["chainage"]) < 2.0 and abs(tr["d_before"] - prev["d_before"]) < 0.5 and abs(tr["d_after"] - prev["d_after"]) < 0.5 and tr["side"] == prev["side"]:
                    continue
                unique_tr.append(tr)

        return unique_tr

    def _draw_boundary_transition_callout(
        self,
        msp: Any,
        transition: dict[str, Any],
        coord_transform_fn: Callable[[float, float], tuple[float, float]],
        layer_text: str = "KML_BOUNDARY",
        layer_line: str = "KML_BOUNDARY",
        color: int = 1,
        text_h: float = 2.5,
        anno_scale: float = 1.0,
        row_offset: Optional[float] = None,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Render stacked transition label: Ch. <chainage> M / <d_before> M, <d_after> M with divider line.

        Places callout strictly OUTWARD from the railway boundary (away from the track centerline)
        with an outward clearance offset so the text bounding box never overlaps the boundary line.
        Matches Indian Railways alignment plan standard for ROW width changes.
        """
        ch = float(transition.get("chainage", 0.0))
        d_before = float(transition.get("d_before", 0.0))
        d_after = float(transition.get("d_after", 0.0))
        pt1 = transition.get("pt1")
        pt2 = transition.get("pt2")
        heading = float(transition.get("heading", 0.0))
        side = str(transition.get("side", "LHS"))

        if pt1 is None or pt2 is None:
            return

        p1_paper = coord_transform_fn(pt1[0], pt1[1])
        p2_paper = coord_transform_fn(pt2[0], pt2[1])

        # Identify outer (furthest from track) and inner points of the jog
        if d_after >= d_before:
            p_outer = p2_paper
            p_inner = p1_paper
        else:
            p_outer = p1_paper
            p_inner = p2_paper

        # Draw perpendicular step line between p_inner and p_outer if not already drawn
        step_len_paper = math.hypot(p_outer[0] - p_inner[0], p_outer[1] - p_inner[1])
        if step_len_paper > 0.5:
            msp.add_line(
                p_inner,
                p_outer,
                dxfattribs={
                    "layer": layer_line,
                    "color": color,
                    "lineweight": 25,
                },
            )

        # Compute true outward unit vector (away from track centerline) and track position
        if "track_pt" in transition:
            trk_pt = transition["track_pt"]
            trk_paper = coord_transform_fn(trk_pt[0], trk_pt[1])
            dist_track = math.hypot(p_outer[0] - trk_paper[0], p_outer[1] - trk_paper[1])
            if dist_track > 0.1:
                u_out = (
                    (p_outer[0] - trk_paper[0]) / dist_track,
                    (p_outer[1] - trk_paper[1]) / dist_track,
                )
            else:
                perp_angle = heading + (math.pi / 2.0 if side == "LHS" else -math.pi / 2.0)
                u_out = (math.cos(perp_angle), math.sin(perp_angle))
                dist_track = max(d_before, d_after, 15.0) * anno_scale
        elif step_len_paper > 0.1:
            u_out = (
                (p_outer[0] - p_inner[0]) / step_len_paper,
                (p_outer[1] - p_inner[1]) / step_len_paper,
            )
            dist_track = max(d_before, d_after, 15.0) * anno_scale
            trk_paper = (p_outer[0] - dist_track * u_out[0], p_outer[1] - dist_track * u_out[1])
        else:
            perp_angle = heading + (math.pi / 2.0 if side == "LHS" else -math.pi / 2.0)
            u_out = (math.cos(perp_angle), math.sin(perp_angle))
            dist_track = max(d_before, d_after, 15.0) * anno_scale
            trk_paper = (p_outer[0] - dist_track * u_out[0], p_outer[1] - dist_track * u_out[1])

        def _fmt_dim(val: float) -> str:
            if abs(val - round(val)) < 0.05:
                return f"{int(round(val))} M"
            return f"{val:.1f} M"

        upper_text = f"Ch. {int(round(ch))} M"
        lower_text = f"{_fmt_dim(d_before)}, {_fmt_dim(d_after)}"

        scale_a = anno_scale if (anno_scale is not None and anno_scale > 0) else (text_h / 2.5)
        bound_cfg = self.config.get("railway_boundary", {}) if hasattr(self, "config") else {}

        # Resolve outward clearance offset from ROW line at runtime
        # Guarantees the text bounding box never touches or overlaps the railway boundary line
        if row_offset is not None and row_offset >= 0:
            outward_offset = float(row_offset)
        elif "callout_offset" in transition:
            outward_offset = float(transition["callout_offset"])
        elif "callout_offset_m" in transition:
            outward_offset = float(transition["callout_offset_m"]) * scale_a
        else:
            cfg_val = bound_cfg.get("callout_offset_m", bound_cfg.get("callout_outward_offset_m", None))
            if cfg_val is not None:
                outward_offset = float(cfg_val) * scale_a
            else:
                # Default clean clearance: 2.5 * text_h (~6.25mm in paper) away from ROW boundary
                outward_offset = 2.5 * text_h

        text_w = max(
            self._calc_text_width(upper_text, text_height=text_h),
            self._calc_text_width(lower_text, text_height=text_h),
        )
        line_len = text_w + 3.0 * scale_a
        margin = 1.5 * scale_a
        L_div = line_len + margin
        gap = 0.45 * scale_a
        box_w = text_w + 3.0 * scale_a
        box_h = 2.0 * text_h + 2.0 * gap

        def _calc_placement(p_out_anchor: tuple[float, float], u_vec: tuple[float, float]):
            s_end = (p_out_anchor[0] + outward_offset * u_vec[0], p_out_anchor[1] + outward_offset * u_vec[1])
            d_end = (s_end[0] + L_div * u_vec[0], s_end[1] + L_div * u_vec[1])

            th = math.atan2(u_vec[1], u_vec[0])
            th_deg = math.degrees(th) % 360.0
            if 90.0 < th_deg <= 270.0:
                r_deg = (th_deg + 180.0) % 360.0
                is_fl = True
            else:
                r_deg = th_deg
                is_fl = False

            r_rad = math.radians(r_deg)
            uv = (math.cos(r_rad), math.sin(r_rad))
            vv = (-math.sin(r_rad), math.cos(r_rad))

            if not is_fl:
                pb = (s_end[0] + margin * u_vec[0], s_end[1] + margin * u_vec[1])
            else:
                pb = (d_end[0] - margin * u_vec[0], d_end[1] - margin * u_vec[1])

            pu = (pb[0] + gap * vv[0], pb[1] + gap * vv[1])
            pl = (pb[0] - gap * vv[0], pb[1] - gap * vv[1])

            mid_p = (s_end[0] + (L_div / 2.0) * u_vec[0], s_end[1] + (L_div / 2.0) * u_vec[1])
            b_poly = PlanTextCollisionManager.estimate_text_box(
                mid_p[0], mid_p[1], text_w=box_w, text_h=box_h, rot_deg=r_deg, align="MIDDLE_CENTER"
            )
            return s_end, d_end, r_deg, pu, pl, b_poly

        # 1. Candidate placement on current side of alignment line
        stem_end, div_end, rot_deg, p_up, p_low, cand_poly = _calc_placement(p_outer, u_out)

        # 2. Check overlap against chainage text (or any entity in collision_mgr)
        has_overlap = False
        if collision_mgr is not None and not collision_mgr.is_clear(cand_poly):
            has_overlap = True
        elif side == "LHS" and (abs(ch % 100.0) < 15.0 or abs(ch % 100.0 - 100.0) < 15.0):
            # Proximity to 100m station chainage tick/label on LHS
            has_overlap = True

        # 3. If overlapping, move boundary transition callout to the opposite side of the alignment line
        if has_overlap:
            u_out_opp = (-u_out[0], -u_out[1])
            p_outer_opp = (
                trk_paper[0] + dist_track * u_out_opp[0],
                trk_paper[1] + dist_track * u_out_opp[1],
            )
            stem_end_opp, div_end_opp, rot_deg_opp, p_up_opp, p_low_opp, cand_poly_opp = _calc_placement(p_outer_opp, u_out_opp)

            if collision_mgr is None or collision_mgr.is_clear(cand_poly_opp):
                p_outer = p_outer_opp
                u_out = u_out_opp
                stem_end, div_end = stem_end_opp, div_end_opp
                rot_deg, p_up, p_low, cand_poly = rot_deg_opp, p_up_opp, p_low_opp, cand_poly_opp
                side = "RHS" if side == "LHS" else "LHS"

        div_start = p_outer

        style_name = "ESP_STYLE" if hasattr(msp, "doc") and "ESP_STYLE" in msp.doc.styles else "Standard"

        # Continuous leader & divider line from boundary edge to end of callout
        msp.add_line(
            div_start,
            div_end,
            dxfattribs={"layer": layer_text, "color": color, "lineweight": 18},
        )

        if collision_mgr is not None:
            # Register callout bounding box in collision_mgr so other annotations don't overlap it
            collision_mgr.register_box(cand_poly)

        # Upper text (Ch. ... M)
        add_masked_mtext(
            msp=msp,
            text=upper_text,
            insert=p_up,
            char_height=text_h,
            rotation_deg=rot_deg,
            layer=layer_text,
            color=color,
            style=style_name,
            attachment_point=MTextEntityAlignment.BOTTOM_LEFT,
            bg_scale=1.05,
        )

        # Lower text (15 M, 50 M)
        add_masked_mtext(
            msp=msp,
            text=lower_text,
            insert=p_low,
            char_height=text_h,
            rotation_deg=rot_deg,
            layer=layer_text,
            color=color,
            style=style_name,
            attachment_point=MTextEntityAlignment.TOP_LEFT,
            bg_scale=1.05,
        )

    def _draw_boundary_transition_callout_from_ew(
        self,
        msp: Any,
        transition: dict[str, Any],
        scale_factor: float = 1.0,
        anno_scale: float = 1.0,
        layer_text: str = "RAILWAY_BOUNDARY_TEXT",
        color: int = 1,
        text_h: float = 2.5,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Render stacked transition callout from Earthwork stepped boundary data."""
        ch = float(transition.get("chainage", 0.0))
        old_w = float(transition.get("old_w", 0.0))
        new_w = float(transition.get("new_w", 0.0))
        px_s, py_s = transition.get("pos", (0.0, 0.0))
        h_s = float(transition.get("heading", 0.0))
        perp_s = float(transition.get("perp", h_s + math.pi / 2.0))

        cos_p, sin_p = math.cos(perp_s), math.sin(perp_s)

        # Render for requested boundary side (default RHS for single side)
        bound_cfg = self.config.get("railway_boundary", {})
        target_side = str(bound_cfg.get("callout_side", "BOTH")).strip().upper()
        if target_side == "LHS":
            sides_to_render = [(-1.0, "LHS")]
        elif target_side in ["RHS", "SINGLE"]:
            sides_to_render = [(1.0, "RHS")]
        else:
            sides_to_render = [(-1.0, "LHS"), (1.0, "RHS")]

        for sign, side in sides_to_render:
            p1_paper = (px_s + sign * old_w * scale_factor * cos_p, py_s + sign * old_w * scale_factor * sin_p)
            p2_paper = (px_s + sign * new_w * scale_factor * cos_p, py_s + sign * new_w * scale_factor * sin_p)

            # Perpendicular step line
            msp.add_line(
                p1_paper,
                p2_paper,
                dxfattribs={"layer": "RAILWAY_BOUNDARY", "color": self.COL_WHITE, "lineweight": 25},
            )

            tr_dict = {
                "chainage": ch,
                "d_before": old_w,
                "d_after": new_w,
                "pt1": p1_paper,
                "pt2": p2_paper,
                "heading": h_s,
                "side": side,
                "track_pt": (px_s, py_s),
            }
            # Paper coordinates are already transformed
            self._draw_boundary_transition_callout(
                msp=msp,
                transition=tr_dict,
                coord_transform_fn=lambda x, y: (x, y),
                layer_text=layer_text,
                color=color,
                text_h=text_h,
                anno_scale=anno_scale,
                collision_mgr=collision_mgr,
            )

    def _draw_railway_boundary(
        self,
        msp: Any,
        get_pos_fn: Any,
        start_ch: float,
        end_ch: float,
        scale_factor: float = 1.0,
        anno_scale: float = 1.0,
        bridge_df: Optional[pd.DataFrame] = None,
        valign_res: Any = None,
        coord_transform_fn: Optional[Callable[[float, float], tuple[float, float]]] = None,
        kml_source: Optional[Path | str] = None,
        line_geom: Optional[LineString] = None,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Render Railway Boundary (ROW) from KML/KMZ survey or dynamic RDSO schedule."""
        bound_cfg = self.config.get("railway_boundary", {})
        use_kml = bool(bound_cfg.get("use_kml_boundary", True))
        show_labels = bool(bound_cfg.get("show_offset_change_labels", True))
        min_step = float(bound_cfg.get("min_offset_change_threshold_m", 1.0))
        label_color = int(bound_cfg.get("color", self.COL_RED))
        layer_text = str(bound_cfg.get("layer_text", "RAILWAY_BOUNDARY_TEXT"))
        text_h = float(bound_cfg.get("text_height_mm", 2.5)) * anno_scale

        drawn_from_kml = False
        kml_boundary_lines: list[LineString] = []
        if use_kml and coord_transform_fn is not None:
            # Locate KML/KMZ source
            kml_candidates = [
                kml_source,
                self.config.get("paths", {}).get("kml_path"),
                self.config.get("paths", {}).get("topomap_kml_path"),
                _PROJECT_ROOT / "kanti_egra_updated.kmz",
                _PROJECT_ROOT.parent / "kanti_egra_updated.kmz",
            ]
            kml_p: Optional[Path] = None
            for c in kml_candidates:
                if c:
                    p = Path(c)
                    if not p.is_absolute():
                        for base in [_PROJECT_ROOT, _PROJECT_ROOT.parent]:
                            if (base / p).exists():
                                kml_p = (base / p).resolve()
                                break
                    if p.exists():
                        kml_p = p.resolve()
                        break

            if kml_p is not None and kml_p.exists():
                kml_boundary_lines = self._extract_kml_boundary_lines(kml_p, self.epsg_code)
                if kml_boundary_lines:
                    # Clip lines to current sheet chainage corridor if line_geom is provided
                    sheet_roi = None
                    if line_geom is not None and not line_geom.is_empty:
                        total_len = float(line_geom.length)
                        if end_ch - start_ch < total_len - 50.0:
                            s_0 = max(0.0, start_ch - 80.0)
                            s_1 = min(total_len, end_ch + 80.0)
                            chs = np.linspace(s_0, s_1, max(20, int((s_1 - s_0) / 10.0)))
                            roi_pts = [line_geom.interpolate(s) for s in chs]
                            if len(roi_pts) >= 2:
                                sheet_roi = LineString(roi_pts).buffer(500.0)

                    for b_line in kml_boundary_lines:
                        geom_to_draw = b_line
                        if sheet_roi is not None:
                            if b_line.disjoint(sheet_roi):
                                continue
                            try:
                                clipped = b_line.intersection(sheet_roi)
                                if clipped.is_empty:
                                    continue
                                geom_to_draw = clipped
                            except Exception:
                                pass

                        segs = [geom_to_draw] if isinstance(geom_to_draw, LineString) else (geom_to_draw.geoms if isinstance(geom_to_draw, MultiLineString) else [])
                        for seg in segs:
                            pts = [coord_transform_fn(x, y) for x, y in seg.coords]
                            if len(pts) >= 2:
                                msp.add_lwpolyline(
                                    pts,
                                    dxfattribs={
                                        "layer": "RAILWAY_BOUNDARY",
                                        "color": self.COL_WHITE,
                                        "lineweight": 25,
                                        "linetype": "RAILWAY_ROW",
                                        "ltscale": 1.0,
                                        "flags": 128,
                                    },
                                )
                                drawn_from_kml = True

                    if drawn_from_kml:
                        self.logger.info("Rendered %d boundary polylines directly from KMZ: %s", len(kml_boundary_lines), kml_p.name)
                        if show_labels and line_geom is not None and not line_geom.is_empty:
                            transitions = self._detect_kml_boundary_transitions(
                                boundary_lines=kml_boundary_lines,
                                centerline=line_geom,
                                start_ch=start_ch,
                                end_ch=end_ch,
                                min_step_m=min_step,
                            )
                            for tr in transitions:
                                self._draw_boundary_transition_callout(
                                    msp=msp,
                                    transition=tr,
                                    coord_transform_fn=coord_transform_fn,
                                    layer_text=layer_text,
                                    color=label_color,
                                    text_h=text_h,
                                    anno_scale=anno_scale,
                                    collision_mgr=collision_mgr,
                                )

        if not drawn_from_kml:
            ew = Earthwork(self.config)
            p_df = getattr(valign_res, "profile_dataframe", None) if valign_res is not None else None
            bound_left_pts, bound_right_pts, transitions = ew.generate_stepped_boundary_geometry(
                get_paper_pos=get_pos_fn,
                start_ch=start_ch,
                end_ch=end_ch,
                scale_factor=scale_factor,
                anno_scale=anno_scale,
                bridge_df=bridge_df,
                profile_df=p_df,
                sample_step_m=5.0,
            )
            if len(bound_left_pts) >= 2:
                msp.add_lwpolyline(
                    bound_left_pts,
                    dxfattribs={"layer": "RAILWAY_BOUNDARY", "color": self.COL_WHITE, "lineweight": 25, "linetype": "RAILWAY_ROW", "flags": 128},
                )
            if len(bound_right_pts) >= 2:
                msp.add_lwpolyline(
                    bound_right_pts,
                    dxfattribs={"layer": "RAILWAY_BOUNDARY", "color": self.COL_WHITE, "lineweight": 25, "linetype": "RAILWAY_ROW", "flags": 128},
                )
            if show_labels and transitions:
                for tr in transitions:
                    self._draw_boundary_transition_callout_from_ew(
                        msp=msp,
                        transition=tr,
                        scale_factor=scale_factor,
                        anno_scale=anno_scale,
                        layer_text=layer_text,
                        color=label_color,
                        text_h=text_h,
                        collision_mgr=collision_mgr,
                    )

    def _draw_track_geometry(
        self,
        msp: Any,
        line_geom: LineString,
        ip_df: pd.DataFrame,
        start_ch: float,
        end_ch: float,
        origin_x: float,
        origin_y: float,
        anno_scale: float,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> Any:
        """Render 2D alignment in True Model Space METRES (1 unit = 1.0 metre)."""
        total_len_m = max(1.0, end_ch - start_ch)
        scale_factor = 1.0

        sample_chs = np.linspace(start_ch, end_ch, max(100, int(total_len_m / 5.0)))
        pts_2d = [line_geom.interpolate(s) for s in sample_chs]
        p_start, p_end = pts_2d[0], pts_2d[-1]
            
        dx_b = p_end.x - p_start.x
        dy_b = p_end.y - p_start.y
        baseline_len = math.hypot(dx_b, dy_b)
        theta = math.atan2(dy_b, dx_b) if baseline_len > 1.0 else 0.0

        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def get_paper_pos(ch_m: float) -> tuple[float, float, float]:
            """Return (pos_x, pos_y, heading_rad) in true 1:1 metres using smooth central difference."""
            s = min(float(line_geom.length), max(0.0, ch_m))
            pt = line_geom.interpolate(s)
            u = (pt.x - p_start.x) * cos_t + (pt.y - p_start.y) * sin_t
            v = -(pt.x - p_start.x) * sin_t + (pt.y - p_start.y) * cos_t
            px = origin_x + u
            py = origin_y + v

            # Smooth Central Difference for exact tangent heading
            ds = 0.5
            s_bwd = max(0.0, s - ds)
            s_fwd = min(float(line_geom.length), s + ds)
            pt_b = line_geom.interpolate(s_bwd)
            pt_f = line_geom.interpolate(s_fwd)
            dx = pt_f.x - pt_b.x
            dy = pt_f.y - pt_b.y
            u_fwd = dx * cos_t + dy * sin_t
            v_fwd = -dx * sin_t + dy * cos_t
            heading = math.atan2(v_fwd, u_fwd) if (abs(u_fwd) > 1e-6 or abs(v_fwd) > 1e-6) else 0.0
            return px, py, heading

        valid_curves = []
        if not ip_df.empty:
            for _, r in ip_df.iterrows():
                rad = float(r.get("radius_m", 0.0))
                if rad <= 0:
                    continue
                ip_ch = float(r.get("chainage_m", 0.0))
                t_len = float(r.get("tangent_length_m", 0.0))
                tptc = float(r.get("tptc_chainage_m", max(0.0, ip_ch - t_len)))
                tpcc = float(r.get("tpcc_chainage_m", tptc))
                ctpc = float(r.get("ctpc_chainage_m", tpcc))
                ctpt = float(r.get("ctpt_chainage_m", ip_ch + t_len))

                direction = str(r.get("direction", "RHS")).upper()
                is_right_turn = "R" in direction

                if ctpt > start_ch and tptc < end_ch:
                    valid_curves.append((tptc, tpcc, ctpc, ctpt, rad, is_right_turn))

        valid_curves.sort(key=lambda x: x[0])

        # Render Piecewise Geometry: Tangents (sampled polylines), Transitions (clothoid spirals), True Arcs
        curr_ch = start_ch
        for tptc, tpcc, ctpc, ctpt, rad, is_right_turn in valid_curves:
            # 1. Straight Tangent Section (Single clean line without segmentation when deflection angle is constant)
            t_s = max(curr_ch, start_ch)
            t_e = min(tptc, end_ch)
            if t_s < t_e:
                p_tan_s = get_paper_pos(t_s)[:2]
                p_tan_e = get_paper_pos(t_e)[:2]
                msp.add_line(
                    p_tan_s,
                    p_tan_e,
                    dxfattribs={"layer": "ALIGNMENT", "color": self.track_color, "lineweight": 35},
                )

            # 2. Transition Spiral 1 (TP-TC to TP-CC: Clothoid Spiral along KML alignment)
            tr1_s = max(start_ch, tptc)
            tr1_e = min(end_ch, tpcc)
            if tr1_s < tr1_e:
                chs = np.linspace(tr1_s, tr1_e, max(5, int((tr1_e - tr1_s) / 5.0)))
                pts_tr1 = [get_paper_pos(c)[:2] for c in chs]
                if len(pts_tr1) >= 2:
                    msp.add_lwpolyline(pts_tr1, dxfattribs={"layer": "ALIGNMENT", "color": self.track_color, "lineweight": 35})

            # 3. Circular Curve (TP-CC to CT-PC: Constant Curvature -> True AutoCAD ARC strictly on KML centerline)
            c_s = max(start_ch, tpcc)
            c_e = min(end_ch, ctpc)
            if c_s < c_e:
                c_mid = (c_s + c_e) / 2.0
                p1 = get_paper_pos(c_s)[:2]
                pm = get_paper_pos(c_mid)[:2]
                p2 = get_paper_pos(c_e)[:2]

                # Use exact nominal design radius R so the AutoCAD ARC entity properties match R in the callout box (e.g. 700.0 m)
                arc_res = self._arc_from_2_points_and_radius(p1, p2, radius=rad, pm=pm, is_right_turn=is_right_turn)
                if arc_res is None:
                    arc_res = self._fit_arc_from_3_points(p1, pm, p2)

                if arc_res is not None:
                    center, r_exact, sa, ea = arc_res
                    msp.add_arc(
                        center=center,
                        radius=r_exact,
                        start_angle=sa,
                        end_angle=ea,
                        dxfattribs={"layer": "ALIGNMENT", "color": self.track_color, "lineweight": 35},
                    )
                else:
                    chs_arc = np.linspace(c_s, c_e, max(10, int((c_e - c_s) / 5.0)))
                    pts_arc = [get_paper_pos(c)[:2] for c in chs_arc]
                    if len(pts_arc) >= 2:
                        msp.add_lwpolyline(pts_arc, dxfattribs={"layer": "ALIGNMENT", "color": self.track_color, "lineweight": 35})

            # 4. Transition Spiral 2 (CT-PC to CT-PT: Clothoid Spiral along KML alignment)
            tr2_s = max(start_ch, ctpc)
            tr2_e = min(end_ch, ctpt)
            if tr2_s < tr2_e:
                chs = np.linspace(tr2_s, tr2_e, max(5, int((tr2_e - tr2_s) / 5.0)))
                pts_tr2 = [get_paper_pos(c)[:2] for c in chs]
                if len(pts_tr2) >= 2:
                    msp.add_lwpolyline(pts_tr2, dxfattribs={"layer": "ALIGNMENT", "color": self.track_color, "lineweight": 35})

            curr_ch = max(curr_ch, ctpt)

        # Final Straight Tangent Section (Single clean line without segmentation)
        if curr_ch < end_ch:
            t_s = max(curr_ch, start_ch)
            t_e = end_ch
            p_tan_s = get_paper_pos(t_s)[:2]
            p_tan_e = get_paper_pos(t_e)[:2]
            msp.add_line(
                p_tan_s,
                p_tan_e,
                dxfattribs={"layer": "ALIGNMENT", "color": self.track_color, "lineweight": 35},
            )

        # Render Station Ticks (every 100m and 1km perpendicular to alignment)
        tick_interval = self.tick_interval_m if self.tick_interval_m > 0 else 100.0
        first_tick = math.ceil(start_ch / tick_interval) * tick_interval
        curr_tick = first_tick
        while curr_tick <= end_ch:
            px, py, heading = get_paper_pos(curr_tick)
            perp = heading + (math.pi / 2.0)
            nx, ny = math.cos(perp), math.sin(perp)

            is_km = (int(round(curr_tick)) % 1000 == 0)
            tick_len = (6.0 if is_km else 3.5) * anno_scale

            # Tick mark perpendicular to alignment
            p_tick_1 = (px - (tick_len / 2.0) * nx, py - (tick_len / 2.0) * ny)
            p_tick_2 = (px + (tick_len / 2.0) * nx, py + (tick_len / 2.0) * ny)
            msp.add_line(
                p_tick_1,
                p_tick_2,
                dxfattribs={"layer": "CHAINAGE_TEXT", "color": self.COL_RED, "lineweight": 18 if is_km else 13},
            )

            # Station chainage label
            if not self.show_chainage_labels:
                curr_tick += tick_interval
                continue

            ch_label = self.format_chainage(curr_tick)

            lbl_x = px + ((tick_len / 2.0) + 1.5 * anno_scale) * nx
            lbl_y = py + ((tick_len / 2.0) + 1.5 * anno_scale) * ny

            rot_deg = math.degrees(perp) % 360.0
            txt_h = 3.0 * anno_scale
            txt_w = self._calc_text_width(ch_label, text_height=txt_h)

            if collision_mgr is not None:
                tick_poly = PlanTextCollisionManager.estimate_text_box(
                    lbl_x, lbl_y, text_w=txt_w, text_h=txt_h, rot_deg=rot_deg, align="MIDDLE_LEFT"
                )
                collision_mgr.register_box(tick_poly)

            add_masked_mtext(
                msp=msp,
                text=ch_label,
                insert=(lbl_x, lbl_y),
                char_height=txt_h,
                rotation_deg=float(rot_deg),
                layer="CHAINAGE_TEXT",
                color=self.COL_RED,
                style="ESP_STYLE",
                attachment_point=MTextEntityAlignment.MIDDLE_LEFT,
                bg_scale=1.05,
            )

            curr_tick += tick_interval

        return get_paper_pos

    def _draw_km_post(
        self,
        msp: Any,
        px: float,
        py: float,
        heading_rad: float,
        km_val: float,
        anno_scale: float = 1.0,
        exact_ch: float = 0.0,
    ) -> None:
        """Draw standard Indian Railways Kilometer Post milestone symbol and chainage label."""
        scale = anno_scale if (anno_scale is not None and anno_scale > 0) else 1.0
        perp = heading_rad + (math.pi / 2.0)
        nx, ny = math.cos(perp), math.sin(perp)
        tx, ty = math.cos(heading_rad), math.sin(heading_rad)

        # Offset from track centerline (6.0m perpendicular offset)
        offset_d = 6.0 * scale
        base_x = px + offset_d * nx
        base_y = py + offset_d * ny

        # Format label (KM 0+000, KM 1+000, ...)
        km_int = int(round(km_val))
        label = f"KM {km_int}+000"

        # Milestone box dimensions (8.0m x 3.2m in model units)
        w_box = 8.0 * scale
        h_box = 3.2 * scale

        # Milestone rectangular box aligned with track heading
        p1 = (base_x - (w_box / 2.0) * tx, base_y - (w_box / 2.0) * ty)
        p2 = (base_x + (w_box / 2.0) * tx, base_y + (w_box / 2.0) * ty)
        p3 = (p2[0] + h_box * nx, p2[1] + h_box * ny)
        p4 = (p1[0] + h_box * nx, p1[1] + h_box * ny)

        # Draw milestone boundary box
        msp.add_lwpolyline(
            [p1, p2, p3, p4, p1],
            dxfattribs={"layer": "CHAINAGE_TEXT", "color": self.COL_RED, "lineweight": 25},
        )

        # Leader stem connecting track tick to milestone
        p_tick_edge = (px + (3.0 * scale) * nx, py + (3.0 * scale) * ny)
        p_box_mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        msp.add_line(
            p_tick_edge,
            p_box_mid,
            dxfattribs={"layer": "CHAINAGE_TEXT", "color": self.COL_RED, "lineweight": 18},
        )

        # Text rotation aligned to track (readable left-to-right)
        rot_deg = math.degrees(heading_rad) % 360.0
        if 90.0 < rot_deg <= 270.0:
            rot_deg = (rot_deg + 180.0) % 360.0

        center_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0
        center_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0

        msp.add_text(
            label,
            dxfattribs={
                "layer": "CHAINAGE_TEXT",
                "style": "ESP_STYLE",
                "color": self.COL_RED,
                "height": 1.8 * scale,
                "rotation": float(rot_deg),
            },
        ).set_placement(
            (center_x, center_y),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

    def _draw_curve_callouts(
        self,
        msp: Any,
        ip_df: pd.DataFrame,
        get_paper_pos: Any,
        start_ch: float,
        end_ch: float,
        anno_scale: float,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Place clean red Curve Details Callout Boxes near apex intersection points matching standard format."""
        for _, r in ip_df.iterrows():
            ip_ch = float(r.get("chainage_m", 0.0))
            if not (start_ch <= ip_ch <= end_ch):
                continue

            ip_idx = int(r.get("ip_index", 1))
            rad = float(r.get("radius_m", 0.0))
            if rad <= 0:
                continue

            deg_c = float(r.get("degree_of_curve", 1750.0 / rad if rad > 0 else 0.0))
            delta = float(r.get("deflection_deg", 0.0))
            direction = str(r.get("direction", "RHS")).upper()
            ls = float(r.get("transition_length_m", 0.0))
            lc = float(r.get("circular_curve_length_m", 0.0))
            cl = float(r.get("curve_length_m", lc + 2.0 * ls))
            t_len = float(r.get("tangent_length_m", 0.0))
            tptc = float(r.get("tptc_chainage_m", max(0.0, ip_ch - t_len)))
            tpcc = float(r.get("tpcc_chainage_m", tptc))
            ctpc = float(r.get("ctpc_chainage_m", tpcc))
            ctpt = float(r.get("ctpt_chainage_m", ip_ch + t_len))
            shift_m = float(r.get("shift_m", (ls**2) / (24.0 * rad) if rad > 0 else 0.0))
            cant_mm = float(r.get("cant_mm", r.get("cant", 0.0)))
            v_max = int(round(float(r.get("permissible_speed_kmph", self.config.get("railway", {}).get("design_speed", 100.0)))))

            # Format deflection angle into Degrees, Minutes, Seconds
            abs_delta = abs(delta)
            d_deg = int(abs_delta)
            d_min = int((abs_delta - d_deg) * 60)
            d_sec = int(round(((abs_delta - d_deg) * 60 - d_min) * 60))
            if d_sec >= 60:
                d_min += 1
                d_sec -= 60
            if d_min >= 60:
                d_deg += 1
                d_min -= 60
            delta_str = f"{d_deg}\u00b0{d_min:02d}'{d_sec:02d}\""

            rad_str = f"{int(round(rad))}" if abs(rad - round(rad)) < 0.01 else f"{rad:.2f}"
            ls_str = f"{int(round(ls))}" if abs(ls - round(ls)) < 0.01 else f"{ls:.2f}"

            c_no_raw = r.get("curve_no") or r.get("Curve No.") or r.get("Sr. NO.") or r.get("ip_index") or ip_idx
            c_no_clean = str(c_no_raw).strip()
            if not c_no_clean or c_no_clean.lower() in ("nan", "none"):
                c_no_clean = str(ip_idx)
            c_no_clean = re.sub(r"(?i)\bcurve\b", "", c_no_clean).strip().lstrip(".").strip()

            # Lines matching standard RDSO curve details layout
            lines = [
                f"Curve No. = {c_no_clean}",
                f"APEX CH. = {ip_ch:.3f} m.",
                f"\u0394 = {delta_str}",
                f"D = {deg_c:.2f}\u00b0 R = {rad_str} m.",
                f"CL = {cl:.3f} m. TL = {t_len:.3f} m.",
                f"LTC = {ls_str} m. S = {shift_m:.3f} m.",
                f"Ca = {cant_mm:.2f} mm Vmax = {v_max} kmph",
            ]

            px, py, heading = get_paper_pos(ip_ch)
            side_sign = 1.0 if "R" in direction else -1.0
            perp = heading + (side_sign * math.pi / 2.0)

            # Text baseline runs along track tangent, right-reading
            rot_deg = math.degrees(heading) % 360.0
            if 90.0 < rot_deg <= 270.0:
                rot_deg = (rot_deg + 180.0) % 360.0
            rot_rad = math.radians(rot_deg)

            u_x, u_y = math.cos(rot_rad), math.sin(rot_rad)
            # Box normal vector pointing away from track
            v_x, v_y = -math.sin(rot_rad), math.cos(rot_rad)

            out_x, out_y = math.cos(perp), math.sin(perp)
            # Determine whether outward leader points along +v or -v so text remains strictly right-reading
            sign_v = 1.0 if (v_x * out_x + v_y * out_y) >= 0.0 else -1.0

            # Box geometry dimensions
            line_h = 3.6 * anno_scale
            bw = 54.0 * anno_scale
            bh = (len(lines) * line_h + 4.0 * anno_scale)
            half_w = bw / 2.0
            half_h = bh / 2.0

            # Leader line extending perpendicularly outward from apex
            leader_len = 45.0 * anno_scale
            lead_end_x = px + leader_len * out_x
            lead_end_y = py + leader_len * out_y

            # Box center pushed outward from leader end
            bc_x = lead_end_x + half_h * sign_v * v_x
            bc_y = lead_end_y + half_h * sign_v * v_y

            def _to_cad(lx: float, ly: float) -> tuple[float, float]:
                """Transform local box (lx, ly) relative to center into global CAD space."""
                return (bc_x + lx * u_x + ly * v_x, bc_y + lx * u_y + ly * v_y)

            # Draw Outer Box Rectangle (without stem line from track)
            c1 = _to_cad(-half_w, -half_h)
            c2 = _to_cad(half_w, -half_h)
            c3 = _to_cad(half_w, half_h)
            c4 = _to_cad(-half_w, half_h)
            msp.add_lwpolyline([c1, c2, c3, c4, c1], dxfattribs={"layer": "CURVE_CALLOUT", "color": self.COL_RED, "lineweight": 25})

            if collision_mgr is not None:
                box_poly = Polygon([c1, c2, c3, c4])
                collision_mgr.register_box(box_poly)

            # Draw Text Lines inside box
            for i, line_text in enumerate(lines):
                t_lx = -half_w + 3.0 * anno_scale
                t_ly = half_h - 3.2 * anno_scale - i * line_h
                t_pt = _to_cad(t_lx, t_ly)

                add_masked_mtext(
                    msp=msp,
                    text=line_text,
                    insert=t_pt,
                    char_height=2.0 * anno_scale,
                    rotation_deg=float(rot_deg),
                    layer="CURVE_CALLOUT",
                    color=self.COL_RED,
                    style="ESP_STYLE",
                    attachment_point=MTextEntityAlignment.BOTTOM_LEFT,
                    bg_scale=1.05,
                )

    def _draw_curve_tangent_points(
        self,
        msp: Any,
        ip_df: pd.DataFrame,
        get_paper_pos: Any,
        start_ch: float,
        end_ch: float,
        anno_scale: float,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Render RDSO standard tangent point radial extension lines and text labels
        (e.g. 'TPTC-1: 991.21', 'TPCC-2: 1261.34') along the track alignment with clean lateral clearance.
        """
        if ip_df is None or ip_df.empty:
            return

        for _, r in ip_df.iterrows():
            ip_idx = int(r.get("ip_index", 1))
            rad = float(r.get("radius_m", 0.0))
            if rad <= 0:
                continue

            ip_ch = float(r.get("chainage_m", 0.0))
            t_len = float(r.get("tangent_length_m", 0.0))
            tptc = float(r.get("tptc_chainage_m", max(0.0, ip_ch - t_len)))
            tpcc = float(r.get("tpcc_chainage_m", tptc))
            ctpc = float(r.get("ctpc_chainage_m", tpcc))
            ctpt = float(r.get("ctpt_chainage_m", ip_ch + t_len))

            direction = str(r.get("direction", "RHS")).upper()
            c_side = 1.0 if "R" in direction else -1.0

            # Tangent points along the curve
            tangent_pts = [
                ("TPTC-1", tptc),
                ("TPCC-1", tpcc),
                ("TPCC-2", ctpc),
                ("TPTC-2", ctpt),
            ]

            for tp_name, tp_ch in tangent_pts:
                if tp_ch <= 0:
                    continue
                if not (start_ch <= tp_ch <= end_ch):
                    continue

                px, py, heading = get_paper_pos(tp_ch)
                perp = heading + (c_side * math.pi / 2.0)
                perp_dx, perp_dy = math.cos(perp), math.sin(perp)

                # 1. Dumbbell Symbol at track centerline (2 filled solid red discs on either side of track)
                d_offset = 1.3 * anno_scale
                r_disc = 0.65 * anno_scale
                c_top = (px + d_offset * perp_dx, py + d_offset * perp_dy)
                c_bot = (px - d_offset * perp_dx, py - d_offset * perp_dy)

                # Connecting stem line between both dumbbell discs crossing track
                msp.add_line(
                    c_bot, c_top,
                    dxfattribs={"layer": "CURVE_CALLOUT", "color": self.COL_RED, "lineweight": 18},
                )

                # Two solid filled discs in red
                for c_node in [c_top, c_bot]:
                    msp.add_circle(
                        c_node, r_disc,
                        dxfattribs={"layer": "CURVE_CALLOUT", "color": self.COL_RED, "lineweight": 18},
                    )
                    hatch = msp.add_hatch(dxfattribs={"layer": "CURVE_CALLOUT", "color": self.COL_RED})
                    hatch.set_solid_fill(color=self.COL_RED)
                    hatch.dxf.color = self.COL_RED
                    hatch.paths.add_edge_path().add_ellipse(c_node, (r_disc, 0.0), 1.0)

                # 2. Collision-free offset tier & side selection
                lbl_text = f"{tp_name} AT CH: {tp_ch:.2f}M"
                text_h = 2.2 * anno_scale
                gap = 0.5 * anno_scale
                text_w = self._calc_text_width(lbl_text, text_height=text_h)

                best_c_node = c_top
                best_start_offset = 2.5 * anno_scale
                best_end_x = c_top[0] + (best_start_offset + text_w) * perp_dx
                best_end_y = c_top[1] + (best_start_offset + text_w) * perp_dy
                best_t_pos = (c_top[0] + best_start_offset * perp_dx, c_top[1] + best_start_offset * perp_dy)
                best_rot_deg = math.degrees(math.atan2(perp_dy, perp_dx)) % 360.0
                found_clear = False

                candidate_tiers = [2.5, 14.0, 26.0, 38.0, 50.0]
                sides_to_try = [c_side, -c_side]

                for s_try in sides_to_try:
                    p_try = heading + (s_try * math.pi / 2.0)
                    p_dx, p_dy = math.cos(p_try), math.sin(p_try)
                    rot_try = math.degrees(math.atan2(p_dy, p_dx)) % 360.0
                    r_rad_try = math.radians(rot_try)
                    ey_x, ey_y = -math.sin(r_rad_try), math.cos(r_rad_try)
                    base_node = c_top if s_try == c_side else c_bot

                    for off_raw in candidate_tiers:
                        s_off = off_raw * anno_scale
                        t_x = base_node[0] + s_off * p_dx + gap * ey_x
                        t_y = base_node[1] + s_off * p_dy + gap * ey_y
                        cand_poly = PlanTextCollisionManager.estimate_text_box(
                            t_x, t_y, text_w=text_w, text_h=text_h, rot_deg=rot_try, align="BOTTOM_LEFT"
                        )
                        if collision_mgr is None or collision_mgr.is_clear(cand_poly):
                            best_c_node = base_node
                            best_start_offset = s_off
                            best_end_x = base_node[0] + (s_off + text_w) * p_dx
                            best_end_y = base_node[1] + (s_off + text_w) * p_dy
                            best_t_pos = (t_x, t_y)
                            best_rot_deg = rot_try
                            if collision_mgr is not None:
                                collision_mgr.register_box(cand_poly)
                            found_clear = True
                            break
                    if found_clear:
                        break

                if not found_clear and collision_mgr is not None:
                    cand_poly = PlanTextCollisionManager.estimate_text_box(
                        best_t_pos[0], best_t_pos[1], text_w=text_w, text_h=text_h, rot_deg=best_rot_deg, align="BOTTOM_LEFT"
                    )
                    collision_mgr.register_box(cand_poly)

                # Radial Leader Line extending outward exactly equal to text length
                msp.add_line(
                    best_c_node, (best_end_x, best_end_y),
                    dxfattribs={"layer": "CURVE_CALLOUT", "color": self.COL_RED, "lineweight": 18},
                )

                add_masked_mtext(
                    msp=msp,
                    text=lbl_text,
                    insert=best_t_pos,
                    char_height=text_h,
                    rotation_deg=float(best_rot_deg),
                    layer="CURVE_CALLOUT",
                    color=self.COL_RED,
                    style="ESP_STYLE",
                    attachment_point=MTextEntityAlignment.BOTTOM_LEFT,
                    bg_scale=1.05,
                )

    def _draw_rdso_bridge_symbol(
        self,
        msp: Any,
        px: float,
        py: float,
        heading: float,
        anno_scale: float = 1.0,
        span_length_m: float = 10.0,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Draw standard Indian Railways Bridge Abutments & Flared Wingwalls symbol across track at (px, py).

        Renders:
          1. Upstream Abutment with 35° flared wing walls pointing backward away from bridge.
          2. Downstream Abutment with 35° flared wing walls pointing forward away from bridge.
          3. Total distance between the two bridge abutment lines is fixed to 10.0m.
          4. Returns (p_top_stem, p_bot_stem) connection points for callout leader stem.
        """
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        perp_angle = heading + math.pi / 2.0
        cos_p, sin_p = math.cos(perp_angle), math.sin(perp_angle)

        # Modelspace post-scale factor (1:5000 scale -> 5.0 CAD units per mm)
        drawing_cfg = self.config.get("drawing", {})
        scale_factor = float(drawing_cfg.get("modelspace_scale_factor", 5.0))
        if scale_factor <= 0:
            scale_factor = 5.0

        # Total distance between bridge abutments in ground metres = 10.0m
        total_dist_ground_m = float(span_length_m) if span_length_m else 10.0
        total_dist_paper_mm = total_dist_ground_m / scale_factor  # 10.0m / 5.0 = 2.0mm paper (scales to 10.0m in CAD)

        w_half = max(4.0, (self.formation_width_m / 2.0) + 0.5) * anno_scale
        half_span = (total_dist_paper_mm / 2.0) * anno_scale  # 1.0mm paper -> 5.0m half-span (10m total)
        wing_len = max(2.5, w_half * 0.5) * anno_scale

        splay_rad = math.radians(35.0)
        cos_sp, sin_sp = math.cos(splay_rad), math.sin(splay_rad)

        # Abutment 1 (Entry / Upstream) along track
        a1_x = px - half_span * cos_h
        a1_y = py - half_span * sin_h
        a1_p1 = (a1_x - w_half * cos_p, a1_y - w_half * sin_p)  # Bottom
        a1_p2 = (a1_x + w_half * cos_p, a1_y + w_half * sin_p)  # Top

        # Wingwall 1 Left (flaring backward & outward bottom)
        w1_p1 = (
            a1_p1[0] - wing_len * (cos_sp * cos_h + sin_sp * cos_p),
            a1_p1[1] - wing_len * (cos_sp * sin_h + sin_sp * sin_p),
        )
        # Wingwall 1 Right (flaring backward & outward top)
        w1_p2 = (
            a1_p2[0] - wing_len * (cos_sp * cos_h - sin_sp * cos_p),
            a1_p2[1] - wing_len * (cos_sp * sin_h - sin_sp * sin_p),
        )

        # Abutment 2 (Exit / Downstream) along track
        a2_x = px + half_span * cos_h
        a2_y = py + half_span * sin_h
        a2_p1 = (a2_x - w_half * cos_p, a2_y - w_half * sin_p)  # Bottom
        a2_p2 = (a2_x + w_half * cos_p, a2_y + w_half * sin_p)  # Top

        # Wingwall 2 Left (flaring forward & outward bottom)
        w2_p1 = (
            a2_p1[0] + wing_len * (cos_sp * cos_h - sin_sp * cos_p),
            a2_p1[1] + wing_len * (cos_sp * sin_h - sin_sp * sin_p),
        )
        # Wingwall 2 Right (flaring forward & outward top)
        w2_p2 = (
            a2_p2[0] + wing_len * (cos_sp * cos_h + sin_sp * cos_p),
            a2_p2[1] + wing_len * (cos_sp * sin_h + sin_sp * sin_p),
        )

        dxf_attrs = {"layer": "STRUCTURES_PLAN", "color": self.COL_RED, "lineweight": 25}

        # Abutment 1 + Wingwalls
        msp.add_line(a1_p1, a1_p2, dxfattribs=dxf_attrs)
        msp.add_line(a1_p1, w1_p1, dxfattribs=dxf_attrs)
        msp.add_line(a1_p2, w1_p2, dxfattribs=dxf_attrs)

        # Abutment 2 + Wingwalls
        msp.add_line(a2_p1, a2_p2, dxfattribs=dxf_attrs)
        msp.add_line(a2_p1, w2_p1, dxfattribs=dxf_attrs)
        msp.add_line(a2_p2, w2_p2, dxfattribs=dxf_attrs)

        # Top and bottom center connection points
        p_top_stem = (px + w_half * cos_p, py + w_half * sin_p)
        p_bot_stem = (px - w_half * cos_p, py - w_half * sin_p)
        return p_top_stem, p_bot_stem

    def _calc_text_width(self, text: str, text_height: float = 2.2, width_factor: float = 1.0) -> float:
        """Calculate exact tight CAD vector text length considering AutoCAD simplex.shx character widths."""
        w_sum = 0.0
        for c in text:
            if c in "MW@#%":
                w_sum += 0.82
            elif c in "QOGCDU":
                w_sum += 0.68
            elif c in "ABEKPRVXYZ":
                w_sum += 0.62
            elif c in "023456789":
                w_sum += 0.58
            elif c in "FLTJ":
                w_sum += 0.52
            elif c == "1":
                w_sum += 0.35
            elif c in "()[]-_/":
                w_sum += 0.38
            elif c == " ":
                w_sum += 0.28
            elif c in "I.,:;'|!":
                w_sum += 0.22
            else:
                w_sum += 0.60
        return w_sum * text_height * width_factor

    def _draw_bridge_callouts(
        self,
        msp: Any,
        bridge_df: Optional[pd.DataFrame],
        get_paper_pos: Any,
        start_ch: float,
        end_ch: float,
        anno_scale: float,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Render standard Indian Railways bridge and culvert structure callouts at 60° angle in Plan View."""
        if bridge_df is None or not hasattr(bridge_df, "empty") or bridge_df.empty:
            return

        placed_boxes: list[Polygon] = []
        drawn_count = 0
        for _, br_row in bridge_df.iterrows():
            # Build case-insensitive, stripped column dictionary
            row_dict = {str(k).strip().lower(): v for k, v in br_row.items() if pd.notna(v)}

            # Robust chainage column extraction
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

            if not (start_ch <= ch <= end_ch):
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

            # Configuration / Classification (MINOR, MAJOR, RUB, ROB, LC, EXISTING)
            config_class = ""
            for key in ["configration", "configuration", "category", "status"]:
                if key in row_dict:
                    config_class = str(row_dict[key]).strip()
                    break

            is_exg = "EX" in config_class.upper() or "EXIST" in config_class.upper()
            prefix = "EXG." if is_exg else "PRO."

            # Format bridge label (e.g. "1" -> "BRIDGE NO. 1", "BR-1" -> "BR-1", "LC-1" -> "LC-1")
            br_label = br_no
            if br_no and not any(br_no.upper().startswith(p) for p in ["BRIDGE", "BR", "LC"]):
                br_label = f"BRIDGE NO. {br_no}"

            # Format Callout Text
            if br_label.upper().startswith("LC") or "LC" in config_class.upper():
                txt_callout = f"{prefix} {br_label} ({span_str if span_str else 'LC'}) AT CH.:{ch:.2f}M"
            elif config_class.upper() in ["RUB", "ROB"]:
                span_part = f"{span_str} " if span_str else ""
                type_part = f"{br_type} " if br_type else ""
                txt_callout = f"{prefix} {br_label} OF {span_part}{type_part}({config_class.upper()}) AT CH.:{ch:.2f}M".replace("  ", " ")
            else:
                span_part = f"{span_str} " if span_str else ""
                type_part = f"{br_type}" if br_type else "CULVERT"
                txt_callout = f"{prefix} {br_label} OF {span_part}{type_part} AT CH.:{ch:.2f}M".replace("  ", " ")

            px, py, heading = get_paper_pos(ch)
            perp = heading + (math.pi / 2.0)
            w_half = max(4.0, (self.formation_width_m / 2.0) + 0.5)

            # Parse span length in meters from span string if available
            parsed_span_m = None
            if span_str:
                m_span = re.findall(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)", span_str, re.IGNORECASE)
                if m_span:
                    try:
                        parsed_span_m = float(m_span[0][0]) * float(m_span[0][1])
                    except (ValueError, TypeError):
                        pass
                else:
                    m_num = re.findall(r"(\d+(?:\.\d+)?)", span_str)
                    if m_num:
                        try:
                            parsed_span_m = float(m_num[0])
                        except (ValueError, TypeError):
                            pass

            is_lc = (
                br_label.upper().startswith("LC")
                or "LC" in config_class.upper()
                or "LEVEL CROSSING" in config_class.upper()
                or "LEVEL CROSSING" in br_type.upper()
                or "LC" in br_type.upper()
            )

            # 1. Draw RDSO Bridge Abutments and Flared Wingwalls Sign on alignment line (only for Bridges/Culverts, not for LC)
            if not is_lc:
                self._draw_rdso_bridge_symbol(
                    msp=msp,
                    px=px,
                    py=py,
                    heading=heading,
                    anno_scale=anno_scale,
                    span_length_m=10.0,
                )

            # 2. Dynamic Collision Avoidance Solver (Multi-Tier & Both Sides):
            sh_len = self._calc_text_width(txt_callout, text_height=2.2 * anno_scale)
            candidate_offsets = [6.0, 18.0, 30.0, 42.0, 54.0, 66.0, 78.0]

            if collision_mgr is not None:
                best_p_end, best_t_pos, chosen_poly, r_deg, chosen_side = collision_mgr.find_clear_leader_position(
                    px=px,
                    py=py,
                    heading=heading,
                    text_w=sh_len,
                    text_h=2.3 * anno_scale,
                    formation_clearance=w_half,
                    preferred_side=1.0,
                    anno_scale=anno_scale,
                    allow_flip_side=True,
                    candidate_tiers=candidate_offsets,
                )
            else:
                perp_dx, perp_dy = math.cos(perp), math.sin(perp)
                r_deg = math.degrees(perp) % 360.0
                r_rad = math.radians(r_deg)
                t_up_x, t_up_y = -math.sin(r_rad), math.cos(r_rad)

                best_p_end = None
                best_t_pos = None
                chosen_poly = None
                found_clear_pos = False

                for offset_raw in candidate_offsets:
                    lead_len = (w_half + offset_raw + sh_len) * anno_scale
                    end_cand = (px + lead_len * perp_dx, py + lead_len * perp_dy)

                    tx_cand = px + (w_half + offset_raw) * perp_dx + (0.5 * anno_scale) * t_up_x
                    ty_cand = py + (w_half + offset_raw) * perp_dy + (0.5 * anno_scale) * t_up_y

                    th = 2.3 * anno_scale
                    p1 = (tx_cand, ty_cand)
                    p2 = (tx_cand + sh_len * math.cos(r_rad), ty_cand + sh_len * math.sin(r_rad))
                    p3 = (p2[0] + th * t_up_x, p2[1] + th * t_up_y)
                    p4 = (p1[0] + th * t_up_x, p1[1] + th * t_up_y)
                    cand_poly = Polygon([p1, p2, p3, p4])

                    has_overlap = any(cand_poly.intersects(placed) for placed in placed_boxes)
                    if not has_overlap:
                        best_p_end = end_cand
                        best_t_pos = (tx_cand, ty_cand)
                        chosen_poly = cand_poly
                        found_clear_pos = True
                        break

                if not found_clear_pos:
                    tier = drawn_count % 5
                    fallback_offset = 6.0 + tier * 14.0
                    lead_len = (w_half + fallback_offset + sh_len) * anno_scale
                    best_p_end = (px + lead_len * perp_dx, py + lead_len * perp_dy)
                    best_t_pos = (
                        px + (w_half + fallback_offset) * perp_dx + (0.5 * anno_scale) * t_up_x,
                        py + (w_half + fallback_offset) * perp_dy + (0.5 * anno_scale) * t_up_y,
                    )
                    th = 2.3 * anno_scale
                    p1 = best_t_pos
                    p2 = (p1[0] + sh_len * math.cos(r_rad), p1[1] + sh_len * math.sin(r_rad))
                    p3 = (p2[0] + th * t_up_x, p2[1] + th * t_up_y)
                    p4 = (p1[0] + th * t_up_x, p1[1] + th * t_up_y)
                    chosen_poly = Polygon([p1, p2, p3, p4])

                if chosen_poly is not None:
                    placed_boxes.append(chosen_poly.buffer(0.25 * anno_scale))

            # 3. Draw Continuous Perpendicular Leader Line from track centerline (px, py) to end
            msp.add_line(
                (px, py),
                best_p_end,
                dxfattribs={"layer": "STRUCTURES_PLAN", "color": self.COL_RED, "lineweight": 18},
            )

            # 4. Draw Bridge details text with native AutoCAD canvas background mask
            add_masked_mtext(
                msp=msp,
                text=txt_callout,
                insert=best_t_pos,
                char_height=2.2 * anno_scale,
                rotation_deg=float(r_deg),
                layer="STRUCTURES_PLAN",
                color=self.COL_RED,
                style="ESP_STYLE",
                attachment_point=MTextEntityAlignment.BOTTOM_LEFT,
                bg_scale=1.05,
            )

            drawn_count += 1

    def _draw_gradient_callouts(
        self,
        msp: Any,
        vip_df: Optional[pd.DataFrame],
        get_paper_pos: Any,
        start_ch: float,
        end_ch: float,
        anno_scale: float,
        side_mult: float = -1.0,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
    ) -> None:
        """Render standard Indian Railways gradient change callouts (PVI/VIP flags & slope arms) in Plan View.

        Engineering Format (per RDSO standards):
          - Vertical Stem Line at PVI chainage perpendicular to track.
          - Vertical text along the stem:
              Line 1: 'CH: <chainage>m (F/CSB)'
              Line 2: 'F.L = <formation_level>m.'
          - Slope indicator arms at top of stem:
              Left Arm (incoming grade): Slopes down to left (if rising), up to left (if falling), or flat (if level).
              Right Arm (outgoing grade): Slopes down to right (if falling), up to right (if rising), or flat (if level).
              Gradient Text (e.g. '1 IN 2000', '1 IN 815', 'LEVEL') aligned parallel above each arm.
        """
        if vip_df is None or not hasattr(vip_df, "empty") or vip_df.empty:
            return

        vips = []
        for _, r in vip_df.iterrows():
            ch = float(r.get("chainage_m", r.get("chainage", 0.0)))
            fl = float(r.get("elevation_m", r.get("formation_level_m", r.get("formation_level", 0.0))))
            g_ratio_str = str(r.get("gradient_ratio_str", "")).strip().upper()
            vips.append({
                "chainage_m": ch,
                "elevation_m": fl,
                "gradient_ratio_str": g_ratio_str,
            })

        vips.sort(key=lambda x: x["chainage_m"])
        n_vips = len(vips)
        if n_vips == 0:
            return

        def _fmt_grade(e1: float, e2: float, dist: float, default_str: str = "") -> tuple[float, str]:
            if dist <= 0.001:
                s = str(default_str).strip().upper()
                if "LEVEL" in s or not s:
                    return 0.0, "LEVEL"
                if s.startswith("RISE "):
                    return 0.0, f"{s[5:].strip()} R"
                if s.startswith("FALL "):
                    return 0.0, f"{s[5:].strip()} F"
                return 0.0, s
            g_pct = ((e2 - e1) / dist) * 100.0
            if abs(g_pct) < 1e-4:
                return 0.0, "LEVEL"
            ratio = int(round(100.0 / abs(g_pct)))
            direction = "R" if g_pct > 0 else "F"
            return g_pct, f"1 IN {ratio} {direction}"

        for idx, vip in enumerate(vips):
            ch_pvi = vip["chainage_m"]
            fl_pvi = vip["elevation_m"]

            if ch_pvi < start_ch or ch_pvi > end_ch:
                continue

            # Calculate incoming gradient (Left)
            if idx > 0:
                c_prev = vips[idx - 1]["chainage_m"]
                e_prev = vips[idx - 1]["elevation_m"]
                g_left_pct, grad_left_str = _fmt_grade(e_prev, fl_pvi, ch_pvi - c_prev, vips[idx - 1]["gradient_ratio_str"])
            else:
                g_left_pct = 0.0
                grad_left_str = "LEVEL"

            # Calculate outgoing gradient (Right)
            if idx < n_vips - 1:
                c_next = vips[idx + 1]["chainage_m"]
                e_next = vips[idx + 1]["elevation_m"]
                g_right_pct, grad_right_str = _fmt_grade(fl_pvi, e_next, c_next - ch_pvi, vip["gradient_ratio_str"])
            else:
                g_right_pct = 0.0
                grad_right_str = "LEVEL"

            # Skip if both sides are identical level/grade (no grade change)
            if idx > 0 and idx < n_vips - 1 and abs(g_left_pct - g_right_pct) < 1e-4:
                continue

            px, py, heading = get_paper_pos(ch_pvi)
            perp = heading + (side_mult * math.pi / 2.0)
            nx, ny = math.cos(perp), math.sin(perp)
            tx, ty = math.cos(heading), math.sin(heading)

            # Offset the entire callout structure from the track alignment line
            offset_dist = 32.0 * anno_scale
            apex_x = px + offset_dist * nx
            apex_y = py + offset_dist * ny

            stem_len = 32.0 * anno_scale
            end_stem_x = apex_x + stem_len * nx
            end_stem_y = apex_y + stem_len * ny

            # Create a unified AutoCAD Block definition for this gradient callout
            doc = msp.doc
            base_name = f"GRADIENT_CALLOUT_CH_{int(round(ch_pvi))}"
            block_name = base_name
            counter = 1
            while block_name in doc.blocks:
                block_name = f"{base_name}_{counter}"
                counter += 1

            blk = doc.blocks.new(name=block_name, base_point=(apex_x, apex_y))

            # 1. Draw Vertical Stem Line extending downward from apex
            blk.add_line(
                (apex_x, apex_y),
                (end_stem_x, end_stem_y),
                dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 18},
            )

            # 2. Draw Vertical Text alongside Stem Line (strictly centered on stem)
            stem_deg = (math.degrees(perp) + 180.0) % 360.0
            stem_rad = math.radians(stem_deg)

            s_up_x = -math.sin(stem_rad)
            s_up_y = math.cos(stem_rad)

            txt_ch = f"CH: {ch_pvi:.1f}m"
            txt_fl = f"F.L = {fl_pvi:.3f}m."

            # Midpoint of the stem line
            mx = (apex_x + end_stem_x) / 2.0
            my = (apex_y + end_stem_y) / 2.0

            # Right side of stem (CH)
            pos_ch_x = mx - (2.2 * anno_scale) * s_up_x
            pos_ch_y = my - (2.2 * anno_scale) * s_up_y
            blk.add_text(
                txt_ch,
                dxfattribs={
                    "height": 2.2 * anno_scale,
                    "rotation": float(stem_deg),
                    "layer": "VGEOM_PLAN",
                    "style": "ESP_STYLE",
                    "color": self.COL_RED,
                },
            ).set_placement((pos_ch_x, pos_ch_y), align=TextEntityAlignment.MIDDLE_CENTER)

            # Left side of stem (FL)
            pos_fl_x = mx + (2.2 * anno_scale) * s_up_x
            pos_fl_y = my + (2.2 * anno_scale) * s_up_y
            blk.add_text(
                txt_fl,
                dxfattribs={
                    "height": 2.2 * anno_scale,
                    "rotation": float(stem_deg),
                    "layer": "VGEOM_PLAN",
                    "style": "ESP_STYLE",
                    "color": self.COL_RED,
                },
            ).set_placement((pos_fl_x, pos_fl_y), align=TextEntityAlignment.MIDDLE_CENTER)

            # 3. Draw Left and Right Gradient Indicator Arms branching from apex (apex_x, apex_y)
            arm_len = 24.0 * anno_scale
            slant_rad = math.radians(14.0)
            tan_slant = math.tan(slant_rad)
            arrow_len = 3.5 * anno_scale
            arrow_ang = math.radians(20.0)

            # Left Arm (Incoming Grade from left to apex)
            if g_left_pct > 0.001:
                u_lx, u_ly = -tx + tan_slant * nx, -ty + tan_slant * ny
            elif g_left_pct < -0.001:
                u_lx, u_ly = -tx - tan_slant * nx, -ty - tan_slant * ny
            else:
                u_lx, u_ly = -tx, -ty

            mag_l = math.hypot(u_lx, u_ly)
            u_lx, u_ly = u_lx / mag_l, u_ly / mag_l
            end_lx, end_ly = apex_x + arm_len * u_lx, apex_y + arm_len * u_ly

            # Draw Left Arm line
            blk.add_line(
                (apex_x, apex_y),
                (end_lx, end_ly),
                dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 18},
            )

            # Left Solid Filled Arrowhead at outer end
            ang_arm_l = math.atan2(u_ly, u_lx)
            b1_lx = end_lx - arrow_len * math.cos(ang_arm_l - arrow_ang)
            b1_ly = end_ly - arrow_len * math.sin(ang_arm_l - arrow_ang)
            b2_lx = end_lx - arrow_len * math.cos(ang_arm_l + arrow_ang)
            b2_ly = end_ly - arrow_len * math.sin(ang_arm_l + arrow_ang)
            p_arrow_l = [(end_lx, end_ly), (b1_lx, b1_ly), (b2_lx, b2_ly), (end_lx, end_ly)]
            blk.add_lwpolyline(p_arrow_l, dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 18})
            hatch_l = blk.add_hatch(color=self.COL_RED, dxfattribs={"layer": "VGEOM_PLAN"})
            hatch_l.set_solid_fill(color=self.COL_RED)
            hatch_l.paths.add_polyline_path(p_arrow_l, is_closed=True)

            # Text above Left Arm
            ang_l_deg = math.degrees(math.atan2(u_ly, u_lx)) % 360.0
            if 90.0 < ang_l_deg <= 270.0:
                ang_l_deg = (ang_l_deg + 180.0) % 360.0
            ang_l_rad = math.radians(ang_l_deg)
            t_up_lx, t_up_ly = -math.sin(ang_l_rad), math.cos(ang_l_rad)
            mid_lx = (apex_x + end_lx) / 2.0 + (0.6 * anno_scale) * t_up_lx
            mid_ly = (apex_y + end_ly) / 2.0 + (0.6 * anno_scale) * t_up_ly

            blk.add_text(
                grad_left_str,
                dxfattribs={
                    "height": 2.2 * anno_scale,
                    "rotation": float(ang_l_deg),
                    "layer": "VGEOM_PLAN",
                    "style": "ESP_STYLE",
                    "color": self.COL_RED,
                },
            ).set_placement((mid_lx, mid_ly), align=TextEntityAlignment.BOTTOM_CENTER)

            # Right Arm (Outgoing Grade from apex to right)
            if g_right_pct < -0.001:
                u_rx, u_ry = tx + tan_slant * nx, ty + tan_slant * ny
            elif g_right_pct > 0.001:
                u_rx, u_ry = tx - tan_slant * nx, ty - tan_slant * ny
            else:
                u_rx, u_ry = tx, ty

            mag_r = math.hypot(u_rx, u_ry)
            u_rx, u_ry = u_rx / mag_r, u_ry / mag_r
            end_rx, end_ry = apex_x + arm_len * u_rx, apex_y + arm_len * u_ry

            # Draw Right Arm line
            blk.add_line(
                (apex_x, apex_y),
                (end_rx, end_ry),
                dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 18},
            )

            # Right Solid Filled Arrowhead at outer end
            ang_arm_r = math.atan2(u_ry, u_rx)
            b1_rx = end_rx - arrow_len * math.cos(ang_arm_r - arrow_ang)
            b1_ry = end_ry - arrow_len * math.sin(ang_arm_r - arrow_ang)
            b2_rx = end_rx - arrow_len * math.cos(ang_arm_r + arrow_ang)
            b2_ry = end_ry - arrow_len * math.sin(ang_arm_r + arrow_ang)
            p_arrow_r = [(end_rx, end_ry), (b1_rx, b1_ry), (b2_rx, b2_ry), (end_rx, end_ry)]
            blk.add_lwpolyline(p_arrow_r, dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 18})
            hatch_r = blk.add_hatch(color=self.COL_RED, dxfattribs={"layer": "VGEOM_PLAN"})
            hatch_r.set_solid_fill(color=self.COL_RED)
            hatch_r.paths.add_polyline_path(p_arrow_r, is_closed=True)

            # Text above Right Arm
            ang_r_deg = math.degrees(math.atan2(u_ry, u_rx)) % 360.0
            if 90.0 < ang_r_deg <= 270.0:
                ang_r_deg = (ang_r_deg + 180.0) % 360.0
            ang_r_rad = math.radians(ang_r_deg)
            t_up_rx, t_up_ry = -math.sin(ang_r_rad), math.cos(ang_r_rad)
            mid_rx = (apex_x + end_rx) / 2.0 + (0.6 * anno_scale) * t_up_rx
            mid_ry = (apex_y + end_ry) / 2.0 + (0.6 * anno_scale) * t_up_ry

            blk.add_text(
                grad_right_str,
                dxfattribs={
                    "height": 2.2 * anno_scale,
                    "rotation": float(ang_r_deg),
                    "layer": "VGEOM_PLAN",
                    "style": "ESP_STYLE",
                    "color": self.COL_RED,
                },
            ).set_placement((mid_rx, mid_ry), align=TextEntityAlignment.BOTTOM_CENTER)

            # Insert unified block reference into modelspace
            msp.add_blockref(block_name, (apex_x, apex_y), dxfattribs={"layer": "VGEOM_PLAN"})

            if collision_mgr is not None:
                # Register stem and arms corridor in collision manager
                all_pts = [(apex_x, apex_y), (end_stem_x, end_stem_y), (end_lx, end_ly), (end_rx, end_ry)]
                min_x = min(p[0] for p in all_pts) - 1.0 * anno_scale
                max_x = max(p[0] for p in all_pts) + 1.0 * anno_scale
                min_y = min(p[1] for p in all_pts) - 1.0 * anno_scale
                max_y = max(p[1] for p in all_pts) + 1.0 * anno_scale
                grad_poly = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
                collision_mgr.register_box(grad_poly)

    def _load_station_schedule(self, station_path: Optional[str | Path] = None) -> pd.DataFrame:
        """Load and parse the Railway Station Schedule Excel (e.g. Station_Excel.xlsx).

        Returns:
            DataFrame with columns: ['name', 'chainage_m', 'location_km', 'station_class']
        """
        path_str = station_path or self.config.get("paths", {}).get("station_excel_path")
        if not path_str:
            return pd.DataFrame()
        p = Path(path_str)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.exists():
            self.logger.debug("Station Excel path does not exist: %s", p)
            return pd.DataFrame()

        try:
            if p.suffix.lower() in [".xlsx", ".xls"]:
                try:
                    df = pd.read_excel(p, sheet_name="Station List")
                except Exception:
                    df = pd.read_excel(p)
            else:
                df = pd.read_csv(p)

            col_names = [str(c).strip().upper() for c in df.columns]
            if not any("NAME OF STATION" in c for c in col_names):
                for r_idx in range(min(12, len(df))):
                    row_vals = [str(v).strip().upper() for v in df.iloc[r_idx]]
                    if any("NAME OF STATION" in v for v in row_vals):
                        df.columns = [str(v).strip() for v in df.iloc[r_idx]]
                        df = df.iloc[r_idx + 1:].reset_index(drop=True)
                        break

            col_map = {str(c).strip().upper(): c for c in df.columns}
            name_col = next((col_map[c] for c in col_map if "NAME OF STATION" in c or "STATION NAME" in c or c == "STATION"), None)
            loc_col = next((col_map[c] for c in col_map if "LOCATION" in c or "KM" in c or "CHAINAGE" in c), None)
            cls_col = next((col_map[c] for c in col_map if "CLASS" in c), None)

            if not name_col or not loc_col:
                self.logger.warning("Could not identify station name or location columns in %s", p)
                return pd.DataFrame()

            records = []
            for _, row in df.iterrows():
                raw_name = str(row[name_col]).strip()
                if not raw_name or raw_name.lower() in ["nan", "none", ""]:
                    continue
                # Clean name: remove trailing 'station', extra whitespace
                clean_name = re.sub(r'(?i)\bstation\b', '', raw_name).strip()
                clean_name = re.sub(r'\s+', ' ', clean_name).upper()

                try:
                    loc_val = float(row[loc_col])
                except (ValueError, TypeError):
                    continue

                if loc_val > 500.0:
                    ch_m = loc_val
                    loc_km = loc_val / 1000.0
                else:
                    loc_km = loc_val
                    ch_m = loc_val * 1000.0

                st_class = str(row[cls_col]).strip() if cls_col and pd.notna(row.get(cls_col)) else ""

                records.append({
                    "name": clean_name,
                    "location_km": loc_km,
                    "chainage_m": ch_m,
                    "station_class": st_class,
                })

            st_df = pd.DataFrame(records)
            if not st_df.empty:
                st_df.sort_values("chainage_m", inplace=True)
                self.logger.info("Loaded %d stations from %s", len(st_df), p)
            return st_df
        except Exception as exc:
            self.logger.warning("Failed to load station schedule from %s: %s", p, exc)
            return pd.DataFrame()

    def _draw_station_flags(
        self,
        msp: Any,
        station_df: Optional[pd.DataFrame],
        get_paper_pos: Any,
        start_ch: float,
        end_ch: float,
        anno_scale: float = 1.0,
        collision_mgr: Optional[PlanTextCollisionManager] = None,
        scale_factor: float = 1.0,
    ) -> None:
        """Render RDSO standard Station Centerline Flags (C/L of Station callout & waving pennant) in Plan View.

        Demarcates Centerline of Station perpendicular to track:
          - Centerline (green dash-dot stem across track corridor to flagpole tip)
          - Rotated rectangular label box with background mask:
              Line 1: 'C/L OF <STATION NAME>'
              Line 2: 'RAILWAY STATION,'
              Line 3: 'AT Km: <LOCATION IN KM>'
          - Waving swallowtail railway pennant flag symbol at pole tip.
        """
        cfg = self.config.get("station_marker", {})
        if not cfg.get("enabled", True):
            return

        if station_df is None or station_df.empty:
            station_df = self._load_station_schedule()
        if station_df is None or station_df.empty:
            return

        doc = getattr(msp, "doc", None)
        if doc is not None:
            if "CENTER" not in doc.linetypes:
                try:
                    doc.linetypes.add(
                        name="CENTER",
                        pattern=[20.0, 12.0, -3.0, 2.0, -3.0],
                        description="Centerline ____ _ ____ _ ____",
                    )
                except Exception:
                    pass
            for l_name, l_col, l_wt, l_lt in [
                ("STATION_MARKER", self.COL_GREEN, 25, "CENTER"),
                ("STATION_TEXT", self.COL_GREEN, 18, "Continuous"),
            ]:
                if l_name not in doc.layers:
                    try:
                        doc.layers.add(
                            name=l_name,
                            color=l_col,
                            lineweight=l_wt,
                            linetype=l_lt if l_lt in doc.linetypes else "Continuous",
                        )
                    except Exception:
                        pass

        marker_layer = cfg.get("layer", "STATION_MARKER")
        marker_color = int(cfg.get("color", self.COL_GREEN))
        lineweight = int(cfg.get("lineweight", 25))
        linetype = str(cfg.get("linetype", "CENTER"))
        text_layer = cfg.get("text_layer", "STATION_TEXT")
        text_color = int(cfg.get("text_color", self.COL_GREEN))

        base_text_h = float(cfg.get("text_height_m", 2.5)) * anno_scale
        pole_len = float(cfg.get("pole_length_m", 75.0)) * anno_scale
        stem_back = float(cfg.get("stem_back_extend_m", 25.0)) * anno_scale
        box_offset = float(cfg.get("box_offset_m", 35.0)) * anno_scale
        box_padding = float(cfg.get("box_padding_m", 2.0)) * anno_scale
        flag_w = float(cfg.get("flag_width_m", 25.0)) * anno_scale
        flag_h = float(cfg.get("flag_height_m", 30.0)) * anno_scale
        placement_side = str(cfg.get("placement_side", "RHS")).upper()
        side_sign = 1.0 if "R" in placement_side else -1.0

        for _, row in station_df.iterrows():
            ch_m = float(row.get("chainage_m", 0.0))
            if ch_m < start_ch or ch_m > end_ch:
                continue

            st_name = str(row.get("name", "")).strip().upper()
            loc_km = float(row.get("location_km", ch_m / 1000.0))

            px, py, heading_rad = get_paper_pos(ch_m)

            # Detailed Class B Yard Rendering (2 Loop Lines, 4 Turnouts, 2 Platforms, FOB, Building, CSR)
            # Scaled uniformly with plan view scale factor
            if cfg.get("detailed_yard", True):
                try:
                    from esp_yard_engine import (
                        AlignmentProjector,
                        YardConfig,
                        generate_projected_tracks,
                        generate_projected_civil_elements,
                        YardDXFRenderer,
                    )
                    yard_proj = AlignmentProjector(
                        get_paper_pos,
                        start_chainage_m=start_ch,
                        scale_factor=scale_factor,
                    )
                    yard_cfg = YardConfig.create_standard_class_b_yard(
                        station_name=st_name,
                        centerline_chainage_m=ch_m,
                    )
                    yard_trk = generate_projected_tracks(yard_cfg, yard_proj)
                    yard_civ = generate_projected_civil_elements(yard_cfg, yard_proj)
                    y_renderer = YardDXFRenderer()
                    if doc is not None:
                        y_renderer.setup_layers(doc)
                    show_station_only = bool(cfg.get("show_station_text_only", False))
                    if show_station_only:
                        show_yard_text = False
                    else:
                        show_yard_text = bool(cfg.get("show_yard_text", cfg.get("show_text", True)))
                    crossover_as_spline = bool(cfg.get("crossover_as_spline", True))
                    y_renderer.render_yard(
                        msp=msp,
                        yard_tracks=yard_trk,
                        civil_layout=yard_civ,
                        anno_scale=anno_scale,
                        render_flagpole=False,  # PlanDrawer handles flagpole below
                        render_text=show_yard_text,
                        crossover_as_spline=crossover_as_spline,
                        show_station_text_only=show_station_only,
                    )
                    self.logger.info(
                        "Rendered detailed Class B station yard for '%s' at Ch %.2f m (scale_factor=%.4f, text=%s, spline=%s)",
                        st_name,
                        ch_m,
                        scale_factor,
                        show_yard_text,
                        crossover_as_spline,
                    )
                except Exception as e_yard:
                    self.logger.warning("Could not render detailed yard for %s: %s", st_name, e_yard)

            # Normal vector (perpendicular to track, pointing to flag side)
            perp_rad = heading_rad + (side_sign * math.pi / 2.0)
            nx, ny = math.cos(perp_rad), math.sin(perp_rad)

            # Tangent vector (parallel to track, along increasing chainage)
            tx, ty = math.cos(heading_rad), math.sin(heading_rad)

            # 1. Draw Flagpole / Centerline
            p_rear = (px - stem_back * nx, py - stem_back * ny)
            p_tip = (px + pole_len * nx, py + pole_len * ny)
            msp.add_line(
                p_rear,
                p_tip,
                dxfattribs={
                    "layer": marker_layer,
                    "color": marker_color,
                    "lineweight": lineweight,
                    "linetype": linetype if (doc and linetype in doc.linetypes) else "Continuous",
                    "ltscale": 1.0,
                },
            )

            # 2. Draw Callout Text & Bracketed Box
            show_callout_text = bool(cfg.get("show_callout_text", cfg.get("show_station_text_only", cfg.get("show_text", True))))
            if show_callout_text:
                line1 = f"C/L OF {st_name}"
                line2 = "RAILWAY STATION,"
                line3 = f"AT Km: {loc_km:.3f}"
                full_text = f"{line1}\\P{line2}\\P{line3}"

                w1 = self._calc_text_width(line1, text_height=base_text_h)
                w2 = self._calc_text_width(line2, text_height=base_text_h)
                w3 = self._calc_text_width(line3, text_height=base_text_h)
                max_line_w = max(w1, w2, w3, 12.0 * anno_scale)
                line_spacing = 1.35
                total_text_h = 3.0 * base_text_h * line_spacing

                # hl is horizontal half-width (X), hw is vertical half-height (Y)
                hl = (max_line_w + 2.0 * box_padding) / 2.0
                hw = (total_text_h + 2.0 * box_padding) / 2.0

                c_box_x = px + box_offset * nx
                c_box_y = py + box_offset * ny

                # Station box and text oriented horizontally (0 deg) per user requirement
                rot_deg = 0.0

                # Horizontal closed rectangular callout box
                box_corners = [
                    (c_box_x - hl, c_box_y - hw),
                    (c_box_x + hl, c_box_y - hw),
                    (c_box_x + hl, c_box_y + hw),
                    (c_box_x - hl, c_box_y + hw),
                ]
                msp.add_lwpolyline(
                    box_corners,
                    close=True,
                    dxfattribs={
                        "layer": marker_layer,
                        "color": marker_color,
                        "lineweight": lineweight,
                    },
                )

                # Centered MText with background mask
                add_masked_mtext(
                    msp=msp,
                    text=full_text,
                    insert=(c_box_x, c_box_y),
                    char_height=base_text_h,
                    rotation_deg=rot_deg,
                    layer=text_layer,
                    color=text_color,
                    style="ESP_STYLE",
                    attachment_point=MTextEntityAlignment.MIDDLE_CENTER,
                    bg_scale=1.10,
                )

                if collision_mgr is not None:
                    box_poly = Polygon(box_corners)
                    collision_mgr.register_box(box_poly)

            # 3. Draw Waving Station Pennant Flag (matching user reference)
            base_x = p_tip[0] - flag_h * nx
            base_y = p_tip[1] - flag_h * ny

            n_steps = 35
            # Top curve: arches up from pole tip to crest, then curves down to top-right tip
            t_top = np.linspace(0.0, 1.0, n_steps)
            u_top = t_top * flag_w
            v_top = flag_h + 0.35 * flag_h * np.sin(t_top * math.pi) - 0.30 * flag_h * (t_top**1.2)
            pts_top = [(base_x + u * tx + v * nx, base_y + u * ty + v * ny) for u, v in zip(u_top, v_top)]

            # Outer swallowtail crescent cutout: curves deeply inward
            t_c = np.linspace(0.0, 1.0, n_steps)
            p0 = np.array([u_top[-1], v_top[-1]])
            p1 = np.array([0.65 * flag_w, 0.38 * flag_h])
            p2 = np.array([0.90 * flag_w, 0.08 * flag_h])
            cres = (1.0 - t_c)[:, None]**2 * p0 + 2.0 * (1.0 - t_c)[:, None] * t_c[:, None] * p1 + t_c[:, None]**2 * p2
            pts_cres = [(base_x + u * tx + v * nx, base_y + u * ty + v * ny) for u, v in cres]

            # Bottom curve: curves from bottom-right tip back to pole base
            t_b = np.linspace(0.0, 1.0, n_steps)
            p0_b = p2
            p1_b = np.array([0.35 * flag_w, 0.28 * flag_h])
            p2_b = np.array([0.0, 0.0])
            bot = (1.0 - t_b)[:, None]**2 * p0_b + 2.0 * (1.0 - t_b)[:, None] * t_b[:, None] * p1_b + t_b[:, None]**2 * p2_b
            pts_bot = [(base_x + u * tx + v * nx, base_y + u * ty + v * ny) for u, v in bot]

            # Connect full flag boundary: pole vertical edge -> top crest -> crescent -> bottom curve
            all_flag_pts = [(base_x, base_y)] + pts_top + pts_cres[1:] + pts_bot[1:]
            msp.add_lwpolyline(
                all_flag_pts,
                close=True,
                dxfattribs={
                    "layer": marker_layer,
                    "color": marker_color,
                    "lineweight": lineweight,
                },
            )

            # Central wave ripple line (S-curve crease)
            t_r = np.linspace(0.0, 1.0, 30)
            p0_r = np.array([0.08 * flag_w, 0.65 * flag_h])
            p1_r = np.array([0.30 * flag_w, 0.80 * flag_h])
            p2_r = np.array([0.50 * flag_w, 0.35 * flag_h])
            p3_r = np.array([0.70 * flag_w, 0.40 * flag_h])
            rip = (
                ((1.0 - t_r)**3)[:, None] * p0_r
                + (3.0 * (1.0 - t_r)**2 * t_r)[:, None] * p1_r
                + (3.0 * (1.0 - t_r) * t_r**2)[:, None] * p2_r
                + (t_r**3)[:, None] * p3_r
            )
            pts_rip = [(base_x + u * tx + v * nx, base_y + u * ty + v * ny) for u, v in rip]
            msp.add_lwpolyline(
                pts_rip,
                close=False,
                dxfattribs={
                    "layer": marker_layer,
                    "color": marker_color,
                    "lineweight": 18,
                },
            )
            self.logger.info("Plotted Station Flag for '%s' at Ch. %.2f m (Km: %.3f)", st_name, ch_m, loc_km)

    @staticmethod
    def _format_dms(deg_val: float) -> str:
        """Format decimal degrees into DMS string (e.g. 23°23'48\")."""
        abs_d = abs(float(deg_val))
        d = int(abs_d)
        rem_m = (abs_d - d) * 60.0
        m = int(rem_m)
        s = round((rem_m - m) * 60.0)
        if s >= 60:
            s = 0
            m += 1
        if m >= 60:
            m = 0
            d += 1
        return f"{d}°{m:02d}'{s:02d}\""

    def _get_compensated_grade(self, deg: float) -> str:
        """Calculate compensated gradient on BG curve (0.04% per degree per RDSO norms)."""
        vert_cfg = self.config.get("vertical", {})
        rail_cfg = self.config.get("railway", {})
        r_val = vert_cfg.get("ruling_gradient", rail_cfg.get("ruling_gradient", "1 in 200"))

        ruling_ratio = 200.0
        if isinstance(r_val, (int, float)):
            if float(r_val) > 1.0:
                ruling_ratio = float(r_val)
            elif float(r_val) > 0:
                ruling_ratio = 1.0 / float(r_val)
        elif isinstance(r_val, str):
            m = re.search(r"(\d+(?:\.\d+)?)", r_val)
            if m:
                ruling_ratio = float(m.group(1))

        ruling_grade = 1.0 / ruling_ratio if ruling_ratio > 0 else 1.0 / 200.0
        grade_comp = 0.04 * deg / 100.0
        comp_grade = ruling_grade - grade_comp
        if comp_grade > 1e-6:
            n_comp = int(round(1.0 / comp_grade))
            return f"1 IN {n_comp}"
        return "LEVEL"

    def _draw_curve_table(
        self,
        msp: Any,
        ip_df: pd.DataFrame,
        x: float,
        y: float,
        anno_scale: float,
    ) -> None:
        """Render the official Indian Railways Proposed Curve Details Table on the sheet.

        Standard 16-Column Layout:
          1. Curve No.
          2. Direction (LHS / RHS)
          3. Chainage of TTP-1 (TS)
          4. Chainage of TP-1 (SC)
          5. Chainage of TP-2 (CS)
          6. Chainage of TTP-2 (ST)
          7. Deflection Angle (DMS format DD°MM'SS")
          8. Radius (in m)
          9. Degree of curvature (1750/R)
          10. Actual Cant Ca (mm)
          11. Cd(mm)
          12. Transition length (in m)
          13. Tangent length (in m)
          14. Length of curve (in m)
          15. Maximum Speed(Kmph)
          16. Compensated Grade (1 IN N)
        """
        valid_rows = [r for _, r in ip_df.iterrows() if float(r.get("radius_m", 0)) > 0]
        if not valid_rows:
            return

        headers_lines: list[list[str]] = [
            ["Curve", "No."],
            ["Direction"],
            ["TPTC-1"],
            ["TPCC-1"],
            ["TPCC-2"],
            ["TPTC-2"],
            ["Deflection", "Angle"],
            ["Radius (in", "m)"],
            ["Degree of", "curvature"],
            ["Actual Cant", "Ca (mm)"],
            ["Cd(mm)"],
            ["Transition", "length (in m)"],
            ["Tangent", "length (in m)"],
            ["Length of", "curve (in m)"],
            ["Maximum", "Speed(Kmph)"],
            ["Compensated", "Grade"],
        ]

        col_widths_base = [
            14.0,  # Curve No.
            14.0,  # Direction
            22.0,  # Chainage of TTP-1
            22.0,  # Chainage of TP-1
            22.0,  # Chainage of TP-2
            22.0,  # Chainage of TTP-2
            20.0,  # Deflection Angle
            16.0,  # Radius (in m)
            18.0,  # Degree of curvature
            18.0,  # Actual Cant Ca (mm)
            16.0,  # Cd(mm)
            20.0,  # Transition length (in m)
            20.0,  # Tangent length (in m)
            20.0,  # Length of curve (in m)
            18.0,  # Maximum Speed(Kmph)
            20.0,  # Compensated Grade
        ]

        col_w = [w * anno_scale for w in col_widths_base]
        total_w = sum(col_w)

        title_h = 7.0 * anno_scale
        header_h = 11.0 * anno_scale
        row_h = 6.0 * anno_scale
        total_h = title_h + header_h + (len(valid_rows) * row_h)

        # 1. Outer Box Border
        msp.add_lwpolyline(
            [(x, y), (x + total_w, y), (x + total_w, y + total_h), (x, y + total_h), (x, y)],
            dxfattribs={"layer": "CURVE_TABLE", "color": self.COL_RED, "lineweight": 25},
        )

        # 2. Title Header Banner
        proj_name = str(self.config.get("project", {}).get("name", "KATRA TO AYODHYA")).strip()
        title_text = f"PROPOSED CURVE DETAILS ({proj_name.upper()})" if proj_name else "PROPOSED CURVE DETAILS"

        msp.add_text(
            title_text,
            dxfattribs={
                "height": 3.0 * anno_scale,
                "layer": "CURVE_TABLE",
                "style": "ESP_STYLE",
                "color": self.COL_RED,
            },
        ).set_placement((x + total_w / 2.0, y + total_h - (title_h / 2.0)), align=TextEntityAlignment.MIDDLE_CENTER)

        # Line below Title Banner
        y_below_title = y + total_h - title_h
        msp.add_line(
            (x, y_below_title),
            (x + total_w, y_below_title),
            dxfattribs={"layer": "CURVE_TABLE", "color": self.COL_RED, "lineweight": 25},
        )

        # 3. Column Header Cells
        y_below_header = y_below_title - header_h
        msp.add_line(
            (x, y_below_header),
            (x + total_w, y_below_header),
            dxfattribs={"layer": "CURVE_TABLE", "color": self.COL_RED, "lineweight": 25},
        )

        header_mid_y = (y_below_title + y_below_header) / 2.0

        # Draw Vertical Column Divider Lines (extending from y up to y_below_title)
        cx = x
        for idx_col, (h_lines, w) in enumerate(zip(headers_lines, col_w)):
            col_mid_x = cx + w / 2.0
            if len(h_lines) == 1:
                msp.add_text(
                    h_lines[0],
                    dxfattribs={
                        "height": 1.8 * anno_scale,
                        "layer": "CURVE_TABLE",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((col_mid_x, header_mid_y), align=TextEntityAlignment.MIDDLE_CENTER)
            else:
                msp.add_text(
                    h_lines[0],
                    dxfattribs={
                        "height": 1.7 * anno_scale,
                        "layer": "CURVE_TABLE",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((col_mid_x, header_mid_y + 2.0 * anno_scale), align=TextEntityAlignment.MIDDLE_CENTER)
                msp.add_text(
                    h_lines[1],
                    dxfattribs={
                        "height": 1.7 * anno_scale,
                        "layer": "CURVE_TABLE",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((col_mid_x, header_mid_y - 2.0 * anno_scale), align=TextEntityAlignment.MIDDLE_CENTER)

            cx += w
            if idx_col < len(col_w) - 1:
                msp.add_line(
                    (cx, y),
                    (cx, y_below_title),
                    dxfattribs={"layer": "CURVE_TABLE", "color": self.COL_RED, "lineweight": 18},
                )

        # 4. Data Rows
        ry = y_below_header
        for r_idx, r in enumerate(valid_rows):
            c_no = str(int(r.get("ip_index", r_idx + 1)))
            ip_ch = float(r.get("chainage_m", 0.0))
            rad = float(r.get("radius_m", 0.0))
            deg = float(r.get("degree_of_curve", 1750.0 / rad if rad > 0 else 0.0))
            delta = float(r.get("deflection_deg", r.get("deflection_angle_deg", 0.0)))
            t_len = float(r.get("tangent_length_m", 0.0))
            ls = float(r.get("transition_length_m", 0.0))
            lc = float(r.get("circular_curve_length_m", 0.0))
            cl = float(r.get("curve_length_m", lc + 2.0 * ls))
            tptc = float(r.get("tptc_chainage_m", max(0.0, ip_ch - t_len)))
            tpcc = float(r.get("tpcc_chainage_m", tptc + ls))
            ctpc = float(r.get("ctpc_chainage_m", tpcc + lc))
            ctpt = float(r.get("ctpt_chainage_m", ctpc + ls))
            dir_str = str(r.get("direction", "LHS" if delta > 0 else "RHS")).upper()
            ca = float(r.get("cant_mm", 0.0))
            cd = float(r.get("actual_cd_mm", r.get("cd_mm", 0.0)))
            v_max = float(r.get("permissible_speed_kmph", r.get("speed_kmph", self.config.get("railway", {}).get("design_speed", 100.0))))

            rad_str = f"{int(round(rad))}" if abs(rad - round(rad)) < 0.01 else f"{rad:.2f}"
            ls_str = f"{int(round(ls))}" if abs(ls - round(ls)) < 0.01 else f"{ls:.2f}"
            ca_str = f"{int(round(ca))}" if abs(ca - round(ca)) < 0.01 else f"{ca:.2f}"
            speed_str = f"{int(round(v_max))}"
            comp_grade_str = self._get_compensated_grade(deg)

            vals = [
                c_no,
                dir_str,
                f"{tptc:.3f}",
                f"{tpcc:.3f}",
                f"{ctpc:.3f}",
                f"{ctpt:.3f}",
                self._format_dms(delta),
                rad_str,
                f"{deg:.3f}",
                ca_str,
                f"{cd:.2f}",
                ls_str,
                f"{t_len:.3f}",
                f"{cl:.3f}",
                speed_str,
                comp_grade_str,
            ]

            row_mid_y = ry - (row_h / 2.0)
            cx = x
            for val, w in zip(vals, col_w):
                msp.add_text(
                    val,
                    dxfattribs={
                        "height": 1.8 * anno_scale,
                        "layer": "CURVE_TABLE",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((cx + w / 2.0, row_mid_y), align=TextEntityAlignment.MIDDLE_CENTER)
                cx += w

            ry -= row_h
            if r_idx < len(valid_rows) - 1:
                msp.add_line(
                    (x, ry),
                    (x + total_w, ry),
                    dxfattribs={"layer": "CURVE_TABLE", "color": self.COL_RED, "lineweight": 18},
                )

    def _draw_gradient_table(
        self,
        msp: Any,
        vip_df: pd.DataFrame,
        x: float,
        y: float,
        anno_scale: float,
        start_ch: float = 0.0,
        end_ch: float = 1e9,
    ) -> None:
        """Render the official Indian Railways Proposed Gradient Details Table on the sheet.

        Standard 8-Column Layout:
          1. Grad. No.
          2. From Ch. (m)
          3. To Ch. (m)
          4. Length (m)
          5. Gradient (1 IN N)
          6. Slope (%)
          7. PVI F.L. (m)
          8. V.C. Length (m)
        """
        if vip_df is None or vip_df.empty:
            return

        vip_sorted = vip_df.sort_values("chainage_m").reset_index(drop=True)
        if len(vip_sorted) < 2:
            return

        # Extract contiguous gradient spans
        valid_spans: list[dict[str, Any]] = []
        for i in range(len(vip_sorted) - 1):
            r_cur = vip_sorted.iloc[i]
            r_nxt = vip_sorted.iloc[i + 1]
            ch_s = float(r_cur.get("chainage_m", 0.0))
            ch_e = float(r_nxt.get("chainage_m", 0.0))
            if ch_e <= start_ch or ch_s >= end_ch:
                continue

            span_len = max(0.0, ch_e - ch_s)
            if span_len <= 1.0:
                continue

            el_s = float(r_cur.get("elevation_m", 0.0))
            el_e = float(r_nxt.get("elevation_m", 0.0))
            dh = el_e - el_s

            # Grade ratio & percent
            g_pct_val = r_cur.get("gradient_to_next_pct")
            if g_pct_val is not None and not (isinstance(g_pct_val, float) and math.isnan(g_pct_val)):
                g_pct = float(g_pct_val)
            else:
                g_pct = (dh / span_len * 100.0) if span_len > 0 else 0.0

            g_str = str(r_cur.get("gradient_ratio_str", "")).strip()
            if not g_str or g_str.upper() in ["NAN", "NONE", ""]:
                if abs(g_pct) < 1e-4:
                    g_str = "LEVEL"
                else:
                    n_ratio = int(round(100.0 / abs(g_pct)))
                    g_str = f"1 IN {n_ratio} {'R' if g_pct > 0 else 'F'}"

            pvi_fl = float(r_nxt.get("elevation_m", 0.0))
            vc_len = float(r_nxt.get("curve_length_m", 0.0))

            valid_spans.append({
                "grad_no": str(len(valid_spans) + 1),
                "from_ch": ch_s,
                "to_ch": ch_e,
                "length_m": span_len,
                "grad_str": g_str,
                "slope_pct": g_pct,
                "pvi_fl": pvi_fl,
                "vc_len": vc_len,
            })

        if not valid_spans:
            return

        headers_lines: list[list[str]] = [
            ["Grad.", "No."],
            ["From", "Ch. (m)"],
            ["To", "Ch. (m)"],
            ["Length", "(in m)"],
            ["Gradient", "(1 IN N)"],
            ["Slope", "(in %)"],
            ["PVI F.L.", "(in m)"],
            ["V.C. Length", "(in m)"],
        ]

        col_widths_base = [
            14.0,  # Grad. No.
            22.0,  # From Ch.
            22.0,  # To Ch.
            18.0,  # Length
            22.0,  # Gradient
            18.0,  # Slope %
            18.0,  # PVI F.L.
            20.0,  # V.C. Length
        ]

        col_w = [w * anno_scale for w in col_widths_base]
        total_w = sum(col_w)

        title_h = 7.0 * anno_scale
        header_h = 11.0 * anno_scale
        row_h = 6.0 * anno_scale
        total_h = title_h + header_h + (len(valid_spans) * row_h)

        # 1. Outer Box Border
        msp.add_lwpolyline(
            [(x, y), (x + total_w, y), (x + total_w, y + total_h), (x, y + total_h), (x, y)],
            dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 25},
        )

        # 2. Title Header Banner
        proj_name = str(self.config.get("project", {}).get("name", "KATRA TO AYODHYA")).strip()
        title_text = f"PROPOSED GRADIENT DETAILS ({proj_name.upper()})" if proj_name else "PROPOSED GRADIENT DETAILS"

        msp.add_text(
            title_text,
            dxfattribs={
                "height": 3.0 * anno_scale,
                "layer": "VGEOM_PLAN",
                "style": "ESP_STYLE",
                "color": self.COL_RED,
            },
        ).set_placement((x + total_w / 2.0, y + total_h - (title_h / 2.0)), align=TextEntityAlignment.MIDDLE_CENTER)

        # Line below Title Banner
        y_below_title = y + total_h - title_h
        msp.add_line(
            (x, y_below_title),
            (x + total_w, y_below_title),
            dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 25},
        )

        # 3. Column Header Cells
        y_below_header = y_below_title - header_h
        msp.add_line(
            (x, y_below_header),
            (x + total_w, y_below_header),
            dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 25},
        )

        header_mid_y = (y_below_title + y_below_header) / 2.0

        # Draw Vertical Column Divider Lines
        cx = x
        for idx_col, (h_lines, w) in enumerate(zip(headers_lines, col_w)):
            col_mid_x = cx + w / 2.0
            if len(h_lines) == 1:
                msp.add_text(
                    h_lines[0],
                    dxfattribs={
                        "height": 1.8 * anno_scale,
                        "layer": "VGEOM_PLAN",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((col_mid_x, header_mid_y), align=TextEntityAlignment.MIDDLE_CENTER)
            else:
                msp.add_text(
                    h_lines[0],
                    dxfattribs={
                        "height": 1.7 * anno_scale,
                        "layer": "VGEOM_PLAN",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((col_mid_x, header_mid_y + 2.0 * anno_scale), align=TextEntityAlignment.MIDDLE_CENTER)
                msp.add_text(
                    h_lines[1],
                    dxfattribs={
                        "height": 1.7 * anno_scale,
                        "layer": "VGEOM_PLAN",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((col_mid_x, header_mid_y - 2.0 * anno_scale), align=TextEntityAlignment.MIDDLE_CENTER)

            cx += w
            if idx_col < len(col_w) - 1:
                msp.add_line(
                    (cx, y),
                    (cx, y_below_title),
                    dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 18},
                )

        # 4. Data Rows
        ry = y_below_header
        for r_idx, span in enumerate(valid_spans):
            sign_str = "+" if span["slope_pct"] > 0 else ("-" if span["slope_pct"] < 0 else "")
            slope_str = f"{sign_str}{abs(span['slope_pct']):.3f}%" if abs(span["slope_pct"]) >= 1e-4 else "0.000%"
            vc_str = f"{span['vc_len']:.1f}" if span["vc_len"] > 0 else "-"

            vals = [
                span["grad_no"],
                f"{span['from_ch']:.3f}",
                f"{span['to_ch']:.3f}",
                f"{span['length_m']:.3f}",
                span["grad_str"],
                slope_str,
                f"{span['pvi_fl']:.3f}",
                vc_str,
            ]

            row_mid_y = ry - (row_h / 2.0)
            cx = x
            for val, w in zip(vals, col_w):
                msp.add_text(
                    val,
                    dxfattribs={
                        "height": 1.8 * anno_scale,
                        "layer": "VGEOM_PLAN",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((cx + w / 2.0, row_mid_y), align=TextEntityAlignment.MIDDLE_CENTER)
                cx += w

            ry -= row_h
            if r_idx < len(valid_spans) - 1:
                msp.add_line(
                    (x, ry),
                    (x + total_w, ry),
                    dxfattribs={"layer": "VGEOM_PLAN", "color": self.COL_RED, "lineweight": 18},
                )

    def _draw_bridge_table(
        self,
        msp: Any,
        bridge_df: pd.DataFrame,
        x: float,
        y: float,
        anno_scale: float,
        start_ch: float = 0.0,
        end_ch: float = 1e9,
    ) -> None:
        """Render the official Indian Railways Proposed Bridge & Structure Details Table on the sheet.

        Standard 6-Column Layout:
          1. S.No.
          2. Bridge No.
          3. Chainage (m)
          4. Type / Span Configuration
          5. Category
          6. Road / Section
        """
        if bridge_df is None or not hasattr(bridge_df, "empty") or bridge_df.empty:
            return

        valid_rows: list[dict[str, Any]] = []
        for idx_raw, (_, row) in enumerate(bridge_df.iterrows()):
            row_dict = {str(k).lower().strip(): v for k, v in row.items()}

            # Extract chainage
            ch = None
            for key in ["approx. chainage (km)", "approx. chainage (m)", "approx. chainage", "chainage_km", "chainage (km)", "chainage_m", "chainage", "ch", "km"]:
                if key in row_dict and pd.notna(row_dict[key]):
                    try:
                        ch_val = float(row_dict[key])
                        # If raw value is in km (< 75.0), convert to metres
                        if 0.0 < ch_val < 75.0:
                            ch = ch_val * 1000.0
                        else:
                            ch = ch_val
                        break
                    except (ValueError, TypeError):
                        pass

            if ch is None:
                continue

            if ch < start_ch or ch > end_ch:
                continue

            # Bridge number
            br_no = ""
            for key in ["bridge no.", "bridge no", "bridge_no", "br_no", "no.", "no"]:
                if key in row_dict and pd.notna(row_dict[key]):
                    br_no = str(row_dict[key]).strip()
                    break

            # Proposed Span
            span_str = ""
            for key in ["proposed span", "proposed_span", "span", "span_config", "size"]:
                if key in row_dict and pd.notna(row_dict[key]):
                    span_str = str(row_dict[key]).strip()
                    break

            # Bridge Type / Material
            br_type = ""
            for key in [
                "bridge type",
                "bridge type / description",
                "type",
                "structure_type",
                "bridge_type_raw",
            ]:
                if key in row_dict and pd.notna(row_dict[key]):
                    br_type = str(row_dict[key]).strip()
                    break
            if not br_type:
                br_type = "CULVERT"

            # Configuration / Classification (MINOR, MAJOR, RUB, ROB, LC)
            config_class = ""
            for key in ["configration", "configuration", "category", "status"]:
                if key in row_dict and pd.notna(row_dict[key]):
                    config_class = str(row_dict[key]).strip().upper()
                    break
            if not config_class:
                config_class = "PROPOSED"

            # Bridge Detail
            detail_str = ""
            for key in ["bridge detail", "bridge_detail", "detail", "details"]:
                if key in row_dict and pd.notna(row_dict[key]):
                    detail_str = str(row_dict[key]).strip()
                    break

            valid_rows.append({
                "s_no": str(len(valid_rows) + 1),
                "br_no": br_no if br_no else f"BR-{len(valid_rows) + 1}",
                "chainage_m": ch,
                "span": span_str if span_str else "-",
                "type_desc": br_type,
                "config": config_class,
                "detail": detail_str if detail_str else "-",
            })

        if not valid_rows:
            return

        headers_lines: list[list[str]] = [
            ["S.", "No."],
            ["Bridge", "No."],
            ["Chainage", "(in m)"],
            ["Proposed", "Span"],
            ["Bridge", "Type"],
            ["Config-", "uration"],
            ["Bridge", "Detail"],
        ]

        min_col_widths = [10.0, 18.0, 20.0, 24.0, 24.0, 20.0, 35.0]

        # Automatically calculate each column's width based on content character lengths
        col_widths_base = []
        for idx_col, h_lines in enumerate(headers_lines):
            max_chars = max(len(line) for line in h_lines)
            for br in valid_rows:
                vals_raw = [
                    br["s_no"],
                    br["br_no"],
                    f"{br['chainage_m']:.3f}",
                    br["span"],
                    br["type_desc"],
                    br["config"],
                    br["detail"],
                ]
                max_chars = max(max_chars, len(str(vals_raw[idx_col])))

            # Dynamic column width: character width factor (1.4 mm) + cell padding (6.0 mm)
            auto_w = max(min_col_widths[idx_col], (max_chars * 1.4) + 6.0)
            col_widths_base.append(auto_w)

        col_w = [w * anno_scale for w in col_widths_base]
        total_w = sum(col_w)

        title_h = 7.0 * anno_scale
        header_h = 11.0 * anno_scale
        row_h = 6.0 * anno_scale
        total_h = title_h + header_h + (len(valid_rows) * row_h)

        # 1. Outer Box Border
        msp.add_lwpolyline(
            [(x, y), (x + total_w, y), (x + total_w, y + total_h), (x, y + total_h), (x, y)],
            dxfattribs={"layer": "STRUCTURES_PLAN", "color": self.COL_RED, "lineweight": 25},
        )

        # 2. Title Header Banner
        title_text = "Proposed Bridge & Structure Details"

        msp.add_text(
            title_text,
            dxfattribs={
                "height": 2.8 * anno_scale,
                "layer": "STRUCTURES_PLAN",
                "style": "ESP_STYLE",
                "color": self.COL_RED,
            },
        ).set_placement((x + total_w / 2.0, y + total_h - (title_h / 2.0)), align=TextEntityAlignment.MIDDLE_CENTER)

        # Line below Title Banner
        y_below_title = y + total_h - title_h
        msp.add_line(
            (x, y_below_title),
            (x + total_w, y_below_title),
            dxfattribs={"layer": "STRUCTURES_PLAN", "color": self.COL_RED, "lineweight": 25},
        )

        # 3. Column Header Cells
        y_below_header = y_below_title - header_h
        msp.add_line(
            (x, y_below_header),
            (x + total_w, y_below_header),
            dxfattribs={"layer": "STRUCTURES_PLAN", "color": self.COL_RED, "lineweight": 25},
        )

        header_mid_y = (y_below_title + y_below_header) / 2.0

        # Draw Vertical Column Divider Lines
        cx = x
        for idx_col, (h_lines, w) in enumerate(zip(headers_lines, col_w)):
            col_mid_x = cx + w / 2.0
            if len(h_lines) == 1:
                msp.add_text(
                    h_lines[0],
                    dxfattribs={
                        "height": 1.8 * anno_scale,
                        "layer": "STRUCTURES_PLAN",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((col_mid_x, header_mid_y), align=TextEntityAlignment.MIDDLE_CENTER)
            else:
                msp.add_text(
                    h_lines[0],
                    dxfattribs={
                        "height": 1.7 * anno_scale,
                        "layer": "STRUCTURES_PLAN",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((col_mid_x, header_mid_y + 2.0 * anno_scale), align=TextEntityAlignment.MIDDLE_CENTER)
                msp.add_text(
                    h_lines[1],
                    dxfattribs={
                        "height": 1.7 * anno_scale,
                        "layer": "STRUCTURES_PLAN",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((col_mid_x, header_mid_y - 2.0 * anno_scale), align=TextEntityAlignment.MIDDLE_CENTER)

            cx += w
            if idx_col < len(col_w) - 1:
                msp.add_line(
                    (cx, y),
                    (cx, y_below_title),
                    dxfattribs={"layer": "STRUCTURES_PLAN", "color": self.COL_RED, "lineweight": 18},
                )

        # 4. Data Rows
        ry = y_below_header
        for r_idx, br in enumerate(valid_rows):
            vals = [
                br["s_no"],
                br["br_no"],
                f"{br['chainage_m']:.3f}",
                br["span"],
                br["type_desc"],
                br["config"],
                br["detail"],
            ]

            row_mid_y = ry - (row_h / 2.0)
            cx = x
            for idx_v, (val, w) in enumerate(zip(vals, col_w)):
                # Dynamically fit text height so content never touches or overflows cell borders
                cell_w_paper = w / anno_scale
                char_len = max(1, len(val))
                if char_len * 1.35 > (cell_w_paper - 2.0):
                    t_h = max(1.2 * anno_scale, min(1.8 * anno_scale, ((cell_w_paper - 2.0) / (char_len * 0.85)) * anno_scale))
                else:
                    t_h = 1.8 * anno_scale

                msp.add_text(
                    val,
                    dxfattribs={
                        "height": t_h,
                        "layer": "STRUCTURES_PLAN",
                        "style": "ESP_STYLE",
                        "color": self.COL_RED,
                    },
                ).set_placement((cx + w / 2.0, row_mid_y), align=TextEntityAlignment.MIDDLE_CENTER)
                cx += w

            ry -= row_h
            if r_idx < len(valid_rows) - 1:
                msp.add_line(
                    (x, ry),
                    (x + total_w, ry),
                    dxfattribs={"layer": "STRUCTURES_PLAN", "color": self.COL_RED, "lineweight": 18},
                )

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
        """Render standard Indian Railways Title Block."""
        w, h = 320.0 * anno_scale, 55.0 * anno_scale
        x = sheet_w - margin - w
        y = margin

        msp.add_lwpolyline(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
            dxfattribs={"layer": "TITLE_BLOCK", "color": self.COL_RED, "lineweight": 35},
        )
        msp.add_line((x, y + h - (14.0 * anno_scale)), (x + w, y + h - (14.0 * anno_scale)), dxfattribs={"layer": "TITLE_BLOCK", "color": self.COL_RED})

        msp.add_text(
            "INDIAN RAILWAYS \u2014 ALIGNMENT PLAN (WORKING PLAN)",
            dxfattribs={"height": 3.8 * anno_scale, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_RED},
        ).set_placement((x + w / 2.0, y + h - (7.0 * anno_scale)), align=TextEntityAlignment.MIDDLE_CENTER)

        proj_name = str(self.config.get("project", {}).get("name", "RAILWAY CORRIDOR DESIGN"))
        items = [
            f"PROJECT: {proj_name}",
            f"SHEET {sheet_index} OF {total_sheets} (KM {start_ch/1000.0:.2f} TO KM {end_ch/1000.0:.2f})",
            f"SCALE: 1:{int(self.plan_scale)} (HORIZONTAL)",
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

    def _draw_plan_view_header(
        self,
        msp: Any,
        origin_x: float,
        plan_top_y: float,
        draw_len_mm: float,
        from_station: str = "KANTI",
        to_station: str = "EGRA",
        theta: float = 0.0,
    ) -> None:
        """Render standard Indian Railways Plan View Header strip (Station direction arrows, Underlined PLAN title, and North Arrow)."""
        y_header = plan_top_y - 10.0

        # ── 1. Left Station & Direction Arrow (◄────── KANTI / KM:) ──
        x_left = origin_x + 35.0
        # Horizontal line
        msp.add_line(
            (x_left - 18.0, y_header),
            (x_left + 18.0, y_header),
            dxfattribs={"layer": "SHEET_BORDER", "color": self.COL_BLACK, "lineweight": 25}
        )
        # Left Arrowhead (closed/filled)
        msp.add_lwpolyline(
            [(x_left - 18.0, y_header), (x_left - 13.5, y_header + 1.8), (x_left - 13.5, y_header - 1.8), (x_left - 18.0, y_header)],
            dxfattribs={"layer": "SHEET_BORDER", "color": self.COL_BLACK, "lineweight": 25}
        )
        # Top label: Station Name (e.g. KANTI)
        txt_from = msp.add_text(
            from_station.upper(),
            dxfattribs={"height": 3.0, "layer": "SHEET_BORDER", "style": "ESP_STYLE", "color": self.COL_BLACK}
        )
        txt_from.set_placement((x_left + 1.0, y_header + 2.0), align=TextEntityAlignment.BOTTOM_CENTER)
        # Bottom label: KM:
        txt_km_left = msp.add_text(
            "KM:",
            dxfattribs={"height": 2.5, "layer": "SHEET_BORDER", "style": "ESP_STYLE", "color": self.COL_BLACK}
        )
        txt_km_left.set_placement((x_left + 1.0, y_header - 5.0), align=TextEntityAlignment.BOTTOM_CENTER)

        # ── 2. Center Title: <u>PLAN</u> ──
        x_mid = origin_x + draw_len_mm / 2.0
        txt_plan = msp.add_text(
            "PLAN",
            dxfattribs={"height": 3.8, "layer": "SHEET_BORDER", "style": "ESP_STYLE", "color": self.COL_BLACK}
        )
        txt_plan.set_placement((x_mid, y_header), align=TextEntityAlignment.MIDDLE_CENTER)
        # Underline below PLAN
        msp.add_line(
            (x_mid - 9.0, y_header - 3.0),
            (x_mid + 9.0, y_header - 3.0),
            dxfattribs={"layer": "SHEET_BORDER", "color": self.COL_BLACK, "lineweight": 25}
        )

        # ── 3. Right Station & Direction Arrow (EGRA / ──────► / KM:) ──
        x_right = origin_x + draw_len_mm - 60.0
        # Horizontal line
        msp.add_line(
            (x_right - 18.0, y_header),
            (x_right + 18.0, y_header),
            dxfattribs={"layer": "SHEET_BORDER", "color": self.COL_BLACK, "lineweight": 25}
        )
        # Right Arrowhead (closed/filled)
        msp.add_lwpolyline(
            [(x_right + 18.0, y_header), (x_right + 13.5, y_header + 1.8), (x_right + 13.5, y_header - 1.8), (x_right + 18.0, y_header)],
            dxfattribs={"layer": "SHEET_BORDER", "color": self.COL_BLACK, "lineweight": 25}
        )
        # Top label: Station Name (e.g. EGRA)
        txt_to = msp.add_text(
            to_station.upper(),
            dxfattribs={"height": 3.0, "layer": "SHEET_BORDER", "style": "ESP_STYLE", "color": self.COL_BLACK}
        )
        txt_to.set_placement((x_right - 1.0, y_header + 2.0), align=TextEntityAlignment.BOTTOM_CENTER)
        # Bottom label: KM:
        txt_km_right = msp.add_text(
            "KM:",
            dxfattribs={"height": 2.5, "layer": "SHEET_BORDER", "style": "ESP_STYLE", "color": self.COL_BLACK}
        )
        txt_km_right.set_placement((x_right - 1.0, y_header - 5.0), align=TextEntityAlignment.BOTTOM_CENTER)

        # ── 4. North Arrow (Compass star with 'N' letter) on Far Right ──
        x_north = origin_x + draw_len_mm - 20.0
        y_north = y_header
        # North arrow star rotated by -theta relative to paper (+Y true north)
        north_ang_deg = math.degrees(-theta + math.pi / 2.0)
        rad_n = math.radians(north_ang_deg)
        cos_n, sin_n = math.cos(rad_n), math.sin(rad_n)
        ux, uy = cos_n, sin_n       # North unit vector
        vx, vy = -sin_n, cos_n      # East unit vector

        star_len = 8.5
        star_w = 2.8
        p_top = (x_north + star_len * ux, y_north + star_len * uy)
        p_bot = (x_north - star_len * 0.6 * ux, y_north - star_len * 0.6 * uy)
        p_right = (x_north + star_w * vx, y_north + star_w * vy)
        p_left = (x_north - star_w * vx, y_north - star_w * vy)

        # Compass 4-point star
        msp.add_lwpolyline(
            [p_top, p_left, (x_north, y_north), p_right, p_top],
            dxfattribs={"layer": "NORTH_ARROW", "color": self.COL_BLACK, "lineweight": 25}
        )
        msp.add_lwpolyline(
            [p_bot, p_left, (x_north, y_north), p_right, p_bot],
            dxfattribs={"layer": "NORTH_ARROW", "color": self.COL_BLACK, "lineweight": 18}
        )
        msp.add_line(p_top, p_bot, dxfattribs={"layer": "NORTH_ARROW", "color": self.COL_BLACK, "lineweight": 25})
        msp.add_line(p_left, p_right, dxfattribs={"layer": "NORTH_ARROW", "color": self.COL_BLACK, "lineweight": 18})

        # 'N' letter annotation
        txt_n_pos = (x_north + (star_len + 3.2) * ux, y_north + (star_len + 3.2) * uy)
        txt_n = msp.add_text(
            "N",
            dxfattribs={"height": 2.8, "layer": "NORTH_ARROW", "style": "ESP_STYLE", "color": self.COL_BLACK}
        )
        txt_n.set_placement(txt_n_pos, align=TextEntityAlignment.MIDDLE_CENTER)

    def _draw_north_arrow_and_scale(self, msp: Any, x: float, y: float, anno_scale: float) -> None:
        """Render North Arrow symbol and graphic scale bar."""
        # North Arrow
        msp.add_line((x, y - (15.0 * anno_scale)), (x, y + (15.0 * anno_scale)), dxfattribs={"layer": "NORTH_ARROW", "color": self.COL_BLACK, "lineweight": 25})
        msp.add_lwpolyline(
            [(x, y + (15.0 * anno_scale)), (x - (4.0 * anno_scale), y + (5.0 * anno_scale)), (x + (4.0 * anno_scale), y + (5.0 * anno_scale)), (x, y + (15.0 * anno_scale))],
            dxfattribs={"layer": "NORTH_ARROW", "color": self.COL_BLACK, "lineweight": 25},
        )
        msp.add_text(
            "N",
            dxfattribs={"height": 4.5 * anno_scale, "layer": "NORTH_ARROW", "style": "ESP_STYLE", "color": self.COL_RED},
        ).set_placement((x, y + (19.0 * anno_scale)), align=TextEntityAlignment.MIDDLE_CENTER)

        # Scale Text
        msp.add_text(
            f"PLAN SCALE 1:{int(self.plan_scale)}",
            dxfattribs={"height": 2.5 * anno_scale, "layer": "TITLE_BLOCK", "style": "ESP_STYLE", "color": self.COL_BLACK},
        ).set_placement((x, y - (22.0 * anno_scale)), align=TextEntityAlignment.MIDDLE_CENTER)


    def log_curve_details(self, ip_df: pd.DataFrame) -> None:
        """Log a comprehensive tabular summary of all horizontal curve details to the logger."""
        if ip_df is None or ip_df.empty:
            self.logger.info("No horizontal curves to display.")
            return

        sep = "=" * 135
        header_sep = "-" * 135
        header_str = (
            f"{'C.No':<5} | {'IP CH (m)':<10} | {'Delta':<10} | {'Dir':<4} | {'R (m)':<8} | "
            f"{'Deg (°)':<7} | {'T (m)':<8} | {'Ls (m)':<7} | {'Lc (m)':<8} | "
            f"{'TP-TC (m)':<11} | {'TP-CC (m)':<11} | {'CT-PC (m)':<11} | {'CT-PT (m)':<11}"
        )
        self.logger.info("\n%s\n  OFFICIAL HORIZONTAL CURVE DETAILS (RDSO / BG)\n%s\n%s\n%s",
                         sep, sep, header_str, header_sep)

        for _, r in ip_df.iterrows():
            if float(r.get("radius_m", 0.0)) <= 0:
                continue
            c_no = int(r.get("ip_index", 1))
            ip_ch = float(r.get("chainage_m", 0.0))
            delta = float(r.get("deflection_deg", 0.0))
            dir_str = str(r.get("direction", "RHS"))
            rad = float(r.get("radius_m", 0.0))
            deg = float(r.get("degree_of_curve", 1750.0 / rad if rad > 0 else 0.0))
            t = float(r.get("tangent_length_m", 0.0))
            ls = float(r.get("transition_length_m", 0.0))
            lc = float(r.get("circular_curve_length_m", 0.0))
            tptc = float(r.get("tptc_chainage_m", max(0.0, ip_ch - t)))
            tpcc = float(r.get("tpcc_chainage_m", tptc + ls))
            ctpc = float(r.get("ctpc_chainage_m", tpcc + lc))
            ctpt = float(r.get("ctpt_chainage_m", ctpc + ls))

            row_str = (
                f"{c_no:<5} | {ip_ch:<10.2f} | {abs(delta):<9.2f}° | {dir_str:<4} | {rad:<8.1f} | "
                f"{deg:<7.2f} | {t:<8.2f} | {ls:<7.1f} | {lc:<8.2f} | "
                f"{tptc:<11.2f} | {tpcc:<11.2f} | {ctpc:<11.2f} | {ctpt:<11.2f}"
            )
            self.logger.info(row_str)

        self.logger.info(sep)

    # Backward-compatible alias (logs via self.logger instead of stdout)
    print_curve_details = log_curve_details


if __name__ == "__main__":
    import yaml
    from modules.read_kml import KMLReader
    from modules.horizontal_geometry import HorizontalGeometry

    config_path = _PROJECT_ROOT / "config" / "config.yaml"
    cfg: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    logger.info("Initializing PlanDrawer with configuration from: %s", config_path)
    
    kml_str = cfg.get("paths", {}).get("kml_path", "D:/ESP/OP_2_update.kml")
    kml_file = Path(kml_str)
    if not kml_file.is_absolute():
        kml_file = _PROJECT_ROOT / kml_file

    if not kml_file.exists():
        raise FileNotFoundError(f"Configured KML file not found: {kml_file}")

    logger.info("Loading actual alignment KML from config: %s", kml_file)
    kml_reader = KMLReader(cfg)
    align_data = kml_reader.read(kml_file)
    logger.info("Computing horizontal curve geometry for alignment...")
    hgeom_engine = HorizontalGeometry(cfg)
    hgeom_data = hgeom_engine.analyze(align_data)
    
    output_name = kml_file.stem + "_alignment_plan"

    drawer_instance = PlanDrawer(cfg)

    # Load bridge schedule if configured
    configured_br_path = cfg.get("paths", {}).get("bridge_list_path", "D:/ESP/Bridge_List_OP2.xlsx")
    bridge_list_file = Path(configured_br_path)
    if not bridge_list_file.is_absolute():
        bridge_list_file = _PROJECT_ROOT / bridge_list_file

    bridge_df = None
    if bridge_list_file.exists():
        logger.info("Loading Bridge List from: %s", bridge_list_file)
        if bridge_list_file.suffix.lower() in [".xlsx", ".xls"]:
            try:
                bridge_df = pd.read_excel(bridge_list_file, sheet_name="Bridge List", header=0)
            except Exception:
                try:
                    bridge_df = pd.read_excel(bridge_list_file, header=0)
                except Exception as exc:
                    logger.warning("Could not read Excel bridge list: %s", exc)
        else:
            bridge_df = pd.read_csv(bridge_list_file)

        if bridge_df is not None:
            bridge_df.columns = [str(c).strip() for c in bridge_df.columns]
            logger.info("Loaded %d bridge records from %s.", len(bridge_df), bridge_list_file.name)

    # Print full curve details table to console
    if hgeom_data.ip_dataframe is not None:
        drawer_instance.print_curve_details(hgeom_data.ip_dataframe)

    result = drawer_instance.draw_alignment_plan(
        align_data, hgeom_data, bridge_df=bridge_df, output_prefix=output_name
    )
    logger.info("Alignment Plan DXF successfully generated at: %s", result.dxf_path)
    for sheet_path in result.dxf_paths:
        logger.info("  Sheet DXF: %s", sheet_path)
