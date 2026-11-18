"""
Module: modules/horizontal_geometry.py
Purpose: Module 6 — Horizontal Geometry (Curve Detection & Design).

Engineering Basis:
    Railway alignments consist of straight tangents and circular curves connected by
    transition spirals at Intersection Points (IPs). Trains cannot navigate sharp angular bends.
    This module analyzes alignment vertices, computes deflection angles, and calculates
    all horizontal curve parameters (Radius, Transition Spiral Length, Superelevation Cant,
    Cant Deficiency, Tangent Length, and Curve Length) according to Indian Railways RDSO standards.

Input: AlignmentResult (Module 1).
Output: HGeomResult (DataFrame of IPs and curve parameters).

Standards: RDSO GE:G-1 / Indian Railways Permanent Way Manual (IRPWM Para 405)
Author   : Railway ESP Generator
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root directory to sys.path for direct module execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from modules.read_kml import AlignmentResult
from utils.geo_utils import format_chainage
from utils.logger import get_logger


logger = get_logger(__name__)


class HorizontalGeometryError(Exception):
    """Custom exception for Horizontal Geometry errors."""
    pass


@dataclass
class HGeomResult:
    """Result object returned by HorizontalGeometry.

    Attributes:
        ip_dataframe: Pandas DataFrame containing the Intersection Points.
            Columns: ip_index, easting, northing, chainage_m, chainage_str,
                     deflection_deg, deflection_angle_deg, direction,
                     radius_m, tangent_length_m, curve_length_m, cant_mm, etc.
        metadata: Engineering metadata dictionary.
    """
    ip_dataframe: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class HorizontalGeometry:
    """Analyzes alignment vertices to extract tangent and curve data.

    Implements Module 6 of the ESP Generator pipeline.

    Uses Douglas-Peucker polyline simplification to filter out KML
    digitization noise before detecting Intersection Points (IPs).
    This prevents treating every tiny vertex kink as a formal railway curve.

    Design Standards Applied:
        - RDSO GE:G-1         : Track Geometry Standards (Alignment, curves, cant)
        - IRS-T-23            : Track Parameters for High Speed (160 km/h)
        - IRPWM Para 405      : Transition Curve Calculation Guidelines
        - Indian Railways SOD : Schedule of Dimensions (BG)

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Example:
        >>> hgeom_result = HorizontalGeometry(config).analyze(alignment_result)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

        # Configuration sections
        railway_config = config.get("railway", {})
        horizontal_config = config.get("horizontal", {})

        # Track parameters
        self.design_speed_kmh: float = float(railway_config.get("design_speed", 100.0))
        self.equilibrium_speed_kmh: float = float(railway_config.get("equil_speed", 120.0))
        self.goods_speed_kmh: float = float(railway_config.get("goods_speed", 65.0))
        self.gauge_plus_rail_mm: float = float(railway_config.get("g_value_mm", 1750.0))

        # Radius parameters (RDSO GE:G-1 Table 1)
        design_config = config.get("design", {})
        self.default_design_radius_m: float = float(
            horizontal_config.get("min_radius", design_config.get("min_curve_radius_m", 875.0))
        )
        self.absolute_minimum_radius_m: float = float(
            horizontal_config.get("min_curve_radius_m", design_config.get("absolute_minimum_radius_m", 175.0))
        )

        # Geometric thresholds
        self.min_deflection_angle_deg: float = float(
            horizontal_config.get("deflection_angle_threshold_deg", 5.0)
        )
        self.min_straight_length_m: float = float(
            horizontal_config.get("min_straight_length", 50.0)
        )
        self.simplify_tolerance_m: float = float(
            horizontal_config.get("simplify_tolerance_m", 5.0)
        )

        self.cad_kml_mode: bool = bool(horizontal_config.get("cad_kml_mode", False))
        if self.cad_kml_mode:
            self.simplify_tolerance_m = 0.5
            self.logger.info("CAD KML mode ON: simplify_tolerance overridden to 0.5m.")

        # IRPWM Para 406 — Minimum straight tangent between consecutive curves
        self.min_straight_between_curves_m: float = float(
            horizontal_config.get("min_straight_between_curves_m", 50.0)
        )

        # Cant (Superelevation) thresholds
        self.max_actual_cant_mm: float = float(horizontal_config.get("max_cant_mm", 165.0))
        self.max_cant_deficiency_mm: float = float(
            horizontal_config.get("max_cant_deficiency_mm", 75.0)
        )
        self.max_cant_excess_mm: float = float(
            horizontal_config.get("max_cant_excess_mm", 75.0)
        )

    @staticmethod
    def _fit_circle_least_squares(points: Any) -> tuple[float, float, float]:
        """Kåsa linear algebraic least squares circle fitting.

        Fits (x - xc)^2 + (y - yc)^2 = R^2 to N 2D points.
        Returns (center_x, center_y, fitted_radius_m).
        """
        if len(points) < 3:
            return 0.0, 0.0, 0.0
        try:
            n = float(len(points))
            sx = sum(float(p[0]) for p in points)
            sy = sum(float(p[1]) for p in points)
            sxx = sum(float(p[0]) ** 2 for p in points)
            syy = sum(float(p[1]) ** 2 for p in points)
            sxy = sum(float(p[0]) * float(p[1]) for p in points)
            sz = sxx + syy
            sxz = sum(float(p[0]) * (float(p[0]) ** 2 + float(p[1]) ** 2) for p in points)
            syz = sum(float(p[1]) * (float(p[0]) ** 2 + float(p[1]) ** 2) for p in points)

            # Determinant of 3x3 matrix [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]]
            det = (
                sxx * (syy * n - sy * sy)
                - sxy * (sxy * n - sy * sx)
                + sx * (sxy * sy - syy * sx)
            )
            if abs(det) < 1e-7:
                return 0.0, 0.0, 0.0

            # Cramer's rule for [a, b, c]
            da = (
                -sxz * (syy * n - sy * sy)
                - sxy * (-syz * n - sy * -sz)
                + sx * (-syz * sy - syy * -sz)
            )
            db = (
                sxx * (-syz * n - sy * -sz)
                - (-sxz) * (sxy * n - sy * sx)
                + sx * (sxy * -sz - (-syz) * sx)
            )
            dc = (
                sxx * (syy * -sz - (-syz) * sy)
                - sxy * (sxy * -sz - (-syz) * sx)
                + (-sxz) * (sxy * sy - syy * sx)
            )

            param_a = da / det
            param_b = db / det
            param_c = dc / det

            center_x = -param_a / 2.0
            center_y = -param_b / 2.0
            radius_squared = center_x**2 + center_y**2 - param_c
            fitted_radius_m = math.sqrt(max(0.0, radius_squared))
            return center_x, center_y, fitted_radius_m
        except Exception as exc:
            logger.warning(
                "_fit_circle_least_squares failed (%s). "
                "Returning zero radius — caller will apply next fallback.",
                exc,
            )
            return 0.0, 0.0, 0.0

    @staticmethod
    def _calculate_menger_radius(p1: tuple[float, ...], p2: tuple[float, ...], p3: tuple[float, ...]) -> float:
        """Computes local radius of curvature (Menger curvature radius) passing through 3 2D points."""
        ax, ay = p1[0], p1[1]
        bx, by = p2[0], p2[1]
        cx, cy = p3[0], p3[1]

        a = math.hypot(bx - ax, by - ay)
        b = math.hypot(cx - bx, cy - by)
        c = math.hypot(cx - ax, cy - ay)

        area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        if area < 1e-6 or a < 1e-3 or b < 1e-3:
            return 0.0  # Straight line or degenerate

        return (a * b * c) / (4.0 * area)

    def analyze(self, alignment_result: AlignmentResult) -> HGeomResult:
        """Analyzes alignment polyline to extract Intersection Points and curve geometry.

        Algorithm:
            1. Simplify KML polyline via Douglas-Peucker to remove digitization noise.
            2. Compute segment bearings and detect deflection angles at vertices.
            3. Merge continuous curve arc vertices into formal Intersection Points (IPs).
            4. Calculate radius, superelevation cant, transition spiral length, shift,
               and exact station chainages per Indian Railways standards.

        Args:
            alignment_result: Output from Module 1 containing projected alignment line.

        Returns:
            HGeomResult object containing DataFrame of Intersection Points and metadata.
        """
        self.logger.info("Starting Horizontal Geometry analysis...")

        try:
            original_line: LineString = alignment_result.alignment_line
            original_vertex_count = len(list(original_line.coords))

            # ── Step 1: Polyline Simplification ─────────────────────────────
            # Douglas-Peucker collapses near-collinear KML vertices into main tangents
            simplified_line = original_line.simplify(
                self.simplify_tolerance_m, preserve_topology=True
            )
            simplified_coords = list(simplified_line.coords)
            simplified_vertex_count = len(simplified_coords)

            self.logger.info(
                "Simplified alignment: %d -> %d vertices (tolerance=%.1fm)",
                original_vertex_count, simplified_vertex_count, self.simplify_tolerance_m
            )

            if simplified_vertex_count < 3:
                self.logger.info("Simplified alignment produced < 3 points; falling back to raw coordinates.")
                simplified_coords = list(original_line.coords)
                simplified_vertex_count = len(simplified_coords)

            if simplified_vertex_count < 3:
                self.logger.warning("Alignment has < 3 total vertices; 0 horizontal curves detected.")
                return HGeomResult(
                    ip_dataframe=pd.DataFrame(),
                    metadata={"total_ips": 0}
                )

            # ── Step 2: Extract Coordinates and Detect Deflection Angles ──
            eastings_list = [coord[0] for coord in simplified_coords]
            northings_list = [coord[1] for coord in simplified_coords]

            raw_intersection_points = []
            for i in range(1, simplified_vertex_count - 1):
                # Entry segment vector
                dx_in = eastings_list[i] - eastings_list[i - 1]
                dy_in = northings_list[i] - northings_list[i - 1]
                incoming_bearing_rad = math.atan2(dx_in, dy_in)

                # Exit segment vector
                dx_out = eastings_list[i + 1] - eastings_list[i]
                dy_out = northings_list[i + 1] - northings_list[i]
                outgoing_bearing_rad = math.atan2(dx_out, dy_out)

                # Deflection angle (difference in direction)
                deflection_rad = outgoing_bearing_rad - incoming_bearing_rad
                while deflection_rad > math.pi:
                    deflection_rad -= 2 * math.pi
                while deflection_rad <= -math.pi:
                    deflection_rad += 2 * math.pi
                deflection_deg = math.degrees(deflection_rad)

                # Filter out minor deflections below threshold
                # NOTE: do NOT hardcode 3.0° floor — use config value so the user can tune it
                if abs(deflection_deg) >= self.min_deflection_angle_deg:
                    raw_intersection_points.append({
                        "vertex_index": i,
                        "easting_m": eastings_list[i],
                        "northing_m": northings_list[i],
                        "incoming_bearing_rad": incoming_bearing_rad,
                        "outgoing_bearing_rad": outgoing_bearing_rad,
                        "deflection_deg": deflection_deg,
                        "deflection_rad": deflection_rad
                    })

            # ── Step 3: Merge Continuous Arc Vertices into Single IPs ──────
            ip_groups = []
            current_group = [raw_intersection_points[0]] if raw_intersection_points else []

            for k in range(1, len(raw_intersection_points)):
                previous_ip = current_group[-1]
                current_ip = raw_intersection_points[k]

                distance_between = math.hypot(
                    current_ip["easting_m"] - previous_ip["easting_m"],
                    current_ip["northing_m"] - previous_ip["northing_m"]
                )
                same_direction = (current_ip["deflection_deg"] > 0) == (previous_ip["deflection_deg"] > 0)

                # Merge arc vertices digitized close together in same direction.
                # Must be consecutive simplified vertices and within 210m distance
                # (separating Curve 1 from Curve 2 which have a 240m straight tangent).
                is_consecutive = (current_ip["vertex_index"] == previous_ip["vertex_index"] + 1)
                if same_direction and is_consecutive and distance_between < 210.0:
                    current_group.append(current_ip)
                else:
                    ip_groups.append(current_group)
                    current_group = [current_ip]

            if current_group:
                ip_groups.append(current_group)

            merged_intersection_points = []
            for group in ip_groups:
                if len(group) == 1:
                    single_ip = group[0]
                    vi = single_ip["vertex_index"]

                    # Use TRUE straight tangent segments outside the curve vertex
                    entry_prev = max(0, vi - 2)
                    entry_pt = vi - 1
                    exit_pt = vi + 1
                    exit_next = min(len(eastings_list) - 1, vi + 2)

                    # Incoming straight tangent: V[entry_prev] → V[entry_pt]
                    eA, nA = eastings_list[entry_prev], northings_list[entry_prev]
                    e1, n1 = eastings_list[entry_pt], northings_list[entry_pt]

                    # Outgoing straight tangent: V[exit_pt] → V[exit_next]
                    e2, n2 = eastings_list[exit_pt], northings_list[exit_pt]
                    eB, nB = eastings_list[exit_next], northings_list[exit_next]

                    dx1, dy1 = e1 - eA, n1 - nA
                    dx2, dy2 = eB - e2, nB - n2
                    denom = dx2 * dy1 - dx1 * dy2

                    if abs(denom) > 1e-6:
                        t1 = ((n2 - nA) * dx2 - (e2 - eA) * dy2) / denom
                        ip_e = eA + t1 * dx1
                        ip_n = nA + t1 * dy1
                    else:
                        self.logger.warning(
                            "Single-vertex IP #%d: tangents are near-parallel (denom=%.2e). "
                            "Falling back to raw KML vertex position (E=%.2f, N=%.2f).",
                            i, denom, single_ip["easting_m"], single_ip["northing_m"],
                        )
                        ip_e = single_ip["easting_m"]
                        ip_n = single_ip["northing_m"]

                    bearing_in = math.atan2(dx1, dy1)
                    bearing_out = math.atan2(dx2, dy2)
                    defl = bearing_out - bearing_in
                    while defl > math.pi: defl -= 2 * math.pi
                    while defl <= -math.pi: defl += 2 * math.pi

                    merged_intersection_points.append({
                        "easting_m": ip_e,
                        "northing_m": ip_n,
                        "incoming_bearing_rad": bearing_in,
                        "outgoing_bearing_rad": bearing_out,
                        "deflection_deg": math.degrees(defl),
                        "deflection_rad": defl,
                        "entry_tangent_index": entry_pt,
                        "exit_tangent_index": exit_pt,
                    })
                else:
                    first_ip = group[0]
                    last_ip = group[-1]
                    self.logger.info("Merging %d digitized arc vertices into single IP", len(group))
                    for idx_v, v in enumerate(group):
                        dist_prev = 0.0
                        if idx_v > 0:
                            dist_prev = math.hypot(
                                v["easting_m"] - group[idx_v - 1]["easting_m"],
                                v["northing_m"] - group[idx_v - 1]["northing_m"]
                            )
                        self.logger.info(
                            "   Arc Vertex #%d [index=%d]: E=%.2f, N=%.2f, dist_prev=%.1fm, deflection=%.2f°",
                            idx_v + 1, v["vertex_index"], v["easting_m"], v["northing_m"], dist_prev, v["deflection_deg"]
                        )

                    entry_index = first_ip["vertex_index"] - 1
                    exit_index = last_ip["vertex_index"] + 1

                    # ── Compute TRUE tangent bearings from the ORIGINAL unsimplified polyline ──
                    # Simplified vertices straddle the curve boundary. To find the exact IP,
                    # we must sample bearings from the TRUE straight tangent sections outside
                    # any transition spirals or circular arcs. We walk backward/forward along
                    # original_line until we find a section with stable bearing (straight track).
                    first_arc_e, first_arc_n = first_ip["easting_m"], first_ip["northing_m"]
                    last_arc_e, last_arc_n = last_ip["easting_m"], last_ip["northing_m"]
                    first_arc_ch = original_line.project(Point(first_arc_e, first_arc_n))
                    last_arc_ch = original_line.project(Point(last_arc_e, last_arc_n))

                    # 1. Entry tangent bearing: walk backward from first_arc_ch to find straight section
                    step_m = 10.0
                    curr_ch = max(0.0, first_arc_ch - 30.0)
                    best_entry_ch = curr_ch
                    prev_brg = None

                    while curr_ch >= step_m:
                        p_a = original_line.interpolate(curr_ch - step_m)
                        p_b = original_line.interpolate(curr_ch)
                        brg = math.atan2(p_b.x - p_a.x, p_b.y - p_a.y)
                        if prev_brg is not None:
                            d_brg = abs(brg - prev_brg)
                            if d_brg > math.pi: d_brg = 2 * math.pi - d_brg
                            # If bearing rate of change is tiny (< 0.05° over 10m), we are on straight tangent
                            if d_brg < math.radians(0.05):
                                best_entry_ch = curr_ch
                                break
                        prev_brg = brg
                        curr_ch -= step_m

                    p1 = original_line.interpolate(max(0.0, best_entry_ch - step_m))
                    p2 = original_line.interpolate(best_entry_ch)
                    bearing_in = math.atan2(p2.x - p1.x, p2.y - p1.y)

                    # 2. Exit tangent bearing: walk forward from last_arc_ch to find straight section
                    curr_ch = min(original_line.length, last_arc_ch + 30.0)
                    best_exit_ch = curr_ch
                    prev_brg = None

                    while curr_ch <= original_line.length - step_m:
                        p_c = original_line.interpolate(curr_ch)
                        p_d = original_line.interpolate(curr_ch + step_m)
                        brg = math.atan2(p_d.x - p_c.x, p_d.y - p_c.y)
                        if prev_brg is not None:
                            d_brg = abs(brg - prev_brg)
                            if d_brg > math.pi: d_brg = 2 * math.pi - d_brg
                            if d_brg < math.radians(0.05):
                                best_exit_ch = curr_ch
                                break
                        prev_brg = brg
                        curr_ch += step_m

                    p3 = original_line.interpolate(best_exit_ch)
                    p4 = original_line.interpolate(min(original_line.length, best_exit_ch + step_m))
                    bearing_out = math.atan2(p4.x - p3.x, p4.y - p3.y)

                    # Deflection angle
                    merged_deflection_rad = bearing_out - bearing_in
                    while merged_deflection_rad > math.pi:
                        merged_deflection_rad -= 2 * math.pi
                    while merged_deflection_rad <= -math.pi:
                        merged_deflection_rad += 2 * math.pi

                    # Reference points on each tangent line for intersection
                    easting_A, northing_A = p1.x, p1.y
                    dx1, dy1 = p2.x - p1.x, p2.y - p1.y
                    easting_2, northing_2 = p3.x, p3.y
                    dx2, dy2 = p4.x - p3.x, p4.y - p3.y
                    denominator = dx2 * dy1 - dx1 * dy2

                    if abs(denominator) > 1e-6:
                        t1 = ((northing_2 - northing_A) * dx2 - (easting_2 - easting_A) * dy2) / denominator
                        intersect_easting = easting_A + t1 * dx1
                        intersect_northing = northing_A + t1 * dy1

                        merged_intersection_points.append({
                            "easting_m": intersect_easting,
                            "northing_m": intersect_northing,
                            "incoming_bearing_rad": bearing_in,
                            "outgoing_bearing_rad": bearing_out,
                            "deflection_deg": math.degrees(merged_deflection_rad),
                            "deflection_rad": merged_deflection_rad,
                            "entry_tangent_index": entry_index,
                            "exit_tangent_index": exit_index,
                            "start_arc_index": first_ip["vertex_index"],
                            "end_arc_index": last_ip["vertex_index"]
                        })
                    else:
                        # Fallback: tangent lines are near-parallel — use first arc vertex
                        self.logger.warning(
                            "Compound arc group (%d vertices): tangent intersection failed "
                            "(denominator=%.2e). Near-parallel tangents — falling back to "
                            "first arc vertex position (E=%.2f, N=%.2f).",
                            len(group), denominator,
                            group[0]["easting_m"], group[0]["northing_m"],
                        )
                        single_ip = group[0]
                        merged_intersection_points.append({
                            "easting_m": single_ip["easting_m"],
                            "northing_m": single_ip["northing_m"],
                            "incoming_bearing_rad": single_ip["incoming_bearing_rad"],
                            "outgoing_bearing_rad": single_ip["outgoing_bearing_rad"],
                            "deflection_deg": single_ip["deflection_deg"],
                            "deflection_rad": single_ip["deflection_rad"],
                            "entry_tangent_index": single_ip["vertex_index"] - 1,
                            "exit_tangent_index": single_ip["vertex_index"] + 1,
                        })

            # ── Step 4: Sort IPs Strictly by Track Polyline Chainage ─────────────
            for ip in merged_intersection_points:
                ip_pt = Point(ip["easting_m"], ip["northing_m"])
                ip["chainage_m"] = float(original_line.project(ip_pt))

            merged_intersection_points.sort(key=lambda item: item["chainage_m"])

            # Recompute accurate entry/exit bearings and deflection angles between consecutive sorted IPs
            raw_line_coords = list(original_line.coords)
            for k in range(len(merged_intersection_points)):
                curr_ip = merged_intersection_points[k]
                curr_pt = Point(curr_ip["easting_m"], curr_ip["northing_m"])

                if k == 0:
                    prev_pt = Point(raw_line_coords[0][0], raw_line_coords[0][1])
                else:
                    prev_pt = Point(merged_intersection_points[k - 1]["easting_m"], merged_intersection_points[k - 1]["northing_m"])

                if k == len(merged_intersection_points) - 1:
                    next_pt = Point(raw_line_coords[-1][0], raw_line_coords[-1][1])
                else:
                    next_pt = Point(merged_intersection_points[k + 1]["easting_m"], merged_intersection_points[k + 1]["northing_m"])

                dx_in, dy_in = curr_pt.x - prev_pt.x, curr_pt.y - prev_pt.y
                dx_out, dy_out = next_pt.x - curr_pt.x, next_pt.y - curr_pt.y
                cross_z = dx_in * dy_out - dy_in * dx_out

                b_in = math.atan2(dx_in, dy_in) % (2.0 * math.pi)
                b_out = math.atan2(dx_out, dy_out) % (2.0 * math.pi)

                deflection_rad = b_out - b_in
                while deflection_rad > math.pi:
                    deflection_rad -= 2.0 * math.pi
                while deflection_rad < -math.pi:
                    deflection_rad += 2.0 * math.pi

                curr_ip["cross_z"] = cross_z
                curr_ip["incoming_bearing_rad"] = b_in
                curr_ip["outgoing_bearing_rad"] = b_out
                curr_ip["deflection_rad"] = deflection_rad
                curr_ip["deflection_deg"] = math.degrees(deflection_rad)

            # ── Calculate Curve Geometry for Each IP ─────────────────────────
            ip_records: list[dict[str, Any]] = []

            for i, ip in enumerate(merged_intersection_points):
                deflection_deg = ip["deflection_deg"]
                deflection_rad_abs = abs(ip["deflection_rad"])

                if abs(deflection_deg) < self.min_deflection_angle_deg:
                    self.logger.info("Ignoring minor deflection angle (%.2f°) as straight line.", abs(deflection_deg))
                    continue

                ip_easting = ip["easting_m"]
                ip_northing = ip["northing_m"]

                # ── Determine Curve Radius (R) ──────────────────────────────
                # Compute robust local 3-point Menger curvature radii along unsimplified polyline
                raw_coords = list(original_line.coords)
                ip_ch = ip.get("chainage_m", 0.0)

                # Pre-compute available tangent headroom so the dynamic-R fallback
                # (further below) can use it as a geometric constraint on R.
                ip_point_pre = Point(ip_easting, ip_northing)
                ip_chainage_pre = float(original_line.project(ip_point_pre))
                if i == 0:
                    _raw_orig = list(original_line.coords)
                    _dx_pre = ip_easting - _raw_orig[0][0]
                    _dy_pre = ip_northing - _raw_orig[0][1]
                    _avail_entry_pre = math.hypot(_dx_pre, _dy_pre)
                else:
                    _prev_ip_pt_pre = Point(
                        merged_intersection_points[i - 1]["easting_m"],
                        merged_intersection_points[i - 1]["northing_m"],
                    )
                    _prev_ch_pre = float(original_line.project(_prev_ip_pt_pre))
                    _avail_entry_pre = max(0.0, (ip_chainage_pre - _prev_ch_pre - self.min_straight_length_m) / 2.0)

                if i == len(merged_intersection_points) - 1:
                    _avail_exit_pre = max(0.0, original_line.length - ip_chainage_pre)
                else:
                    _next_ip_pt_pre = Point(
                        merged_intersection_points[i + 1]["easting_m"],
                        merged_intersection_points[i + 1]["northing_m"],
                    )
                    _next_ch_pre = float(original_line.project(_next_ip_pt_pre))
                    _avail_exit_pre = max(0.0, (_next_ch_pre - ip_chainage_pre - self.min_straight_length_m) / 2.0)

                max_available_tangent_pre = min(_avail_entry_pre, _avail_exit_pre)

                cumul_ch = 0.0
                menger_radii = []
                local_points = []
                for idx_c in range(len(raw_coords)):
                    if idx_c > 0:
                        cumul_ch += math.hypot(
                            raw_coords[idx_c][0] - raw_coords[idx_c - 1][0],
                            raw_coords[idx_c][1] - raw_coords[idx_c - 1][1]
                        )
                    if abs(cumul_ch - ip_ch) <= 300.0:
                        local_points.append((raw_coords[idx_c][0], raw_coords[idx_c][1]))
                        if 0 < idx_c < len(raw_coords) - 1:
                            r_menger = self._calculate_menger_radius(
                                raw_coords[idx_c - 1],
                                raw_coords[idx_c],
                                raw_coords[idx_c + 1]
                            )
                            # Accept only plausible railway curve radii
                            if 50.0 <= r_menger <= 10000.0:
                                menger_radii.append(r_menger)

                # Compute macroscopic arc-span radius (total chord span / total deflection)
                r_macro = 0.0
                if "start_arc_index" in ip and "end_arc_index" in ip:
                    try:
                        idx_start = ip["start_arc_index"]
                        idx_end = ip["end_arc_index"]
                        if 0 <= idx_start < len(eastings_list) and 0 <= idx_end < len(eastings_list):
                            pt_s = Point(eastings_list[idx_start], northings_list[idx_start])
                            pt_e = Point(eastings_list[idx_end], northings_list[idx_end])
                            span_ch = abs(float(original_line.project(pt_e)) - float(original_line.project(pt_s))) + 100.0
                            if span_ch > 20.0 and deflection_rad_abs > 1e-4:
                                r_macro = span_ch / deflection_rad_abs
                    except Exception:
                        pass

                if len(menger_radii) >= 3:
                    arr = np.array(menger_radii)
                    # 33rd-percentile = tightest third of samples = pure circular arc zone
                    pct33 = float(np.percentile(arr, 33))
                    # Keep only samples within 2.5× the 33rd-pct (removes straight-tangent spikes)
                    arc_samples = arr[arr <= 2.5 * pct33]
                    if len(arc_samples) >= 3:
                        curve_radius_m = float(np.mean(arc_samples))
                    else:
                        # Too few filtered samples — fall back to 33rd-pct directly
                        curve_radius_m = pct33

                    # For shallow curves (Delta <= 6 deg), Menger local chords are noisy; prefer macroscopic span
                    if abs(deflection_deg) <= 6.0 and r_macro >= 1000.0:
                        curve_radius_m = r_macro

                    self.logger.info(
                        "Fitted curve radius R=%.2fm via percentile-filtered Menger "
                        "(total=%d samples, arc_samples=%d, 33pct=%.1fm, range=[%.1f, %.1f], R_macro=%.1fm).",
                        curve_radius_m, len(menger_radii), len(arc_samples),
                        pct33, float(np.min(menger_radii)), float(np.max(menger_radii)), r_macro,
                    )
                elif r_macro >= 500.0:
                    curve_radius_m = r_macro
                    self.logger.info("IP %d: Used macroscopic arc-span radius: R=%.2fm.", i, curve_radius_m)
                else:
                    # Tier 2 fallback: least-squares circle fitting on local points
                    fitted_radius_m = 0.0
                    if len(local_points) >= 3:
                        _, _, fitted_radius_m = self._fit_circle_least_squares(np.array(local_points))

                    if 100.0 <= fitted_radius_m <= 5000.0:
                        curve_radius_m = fitted_radius_m
                        self.logger.info(
                            "IP %d: Menger radii unavailable — used least-squares circle fit: R=%.2fm.",
                            i, curve_radius_m,
                        )
                    else:
                        # Tier 3 fallback: derive R geometrically from available tangent headroom
                        tan_half_delta = math.tan(deflection_rad_abs / 2.0)
                        if tan_half_delta > 1e-6 and max_available_tangent_pre > 0:
                            dyn_r = max_available_tangent_pre / tan_half_delta
                            curve_radius_m = max(self.absolute_minimum_radius_m, min(self.default_design_radius_m, dyn_r))
                            self.logger.warning(
                                "IP %d: No reliable radius from Menger or LSQ fit — "
                                "derived R=%.2fm from available tangent (%.2fm) and Δ=%.2f°. "
                                "Clamped to [%.0fm, %.0fm].",
                                i, curve_radius_m, max_available_tangent_pre,
                                math.degrees(deflection_rad_abs),
                                self.absolute_minimum_radius_m, self.default_design_radius_m,
                            )
                        else:
                            curve_radius_m = self.default_design_radius_m
                            self.logger.warning(
                                "IP %d: All radius estimation methods failed — "
                                "using default design radius R=%.2fm from config.",
                                i, curve_radius_m,
                            )

                # ── Cant (Superelevation) & Speed Equations (RDSO GE:G-1) ────
                # Degree of Curve (D = 1750 / R)
                degree_of_curve = 1750.0 / curve_radius_m if curve_radius_m > 0 else 0.0

                # Equilibrium Cant for reference speed (Ce = 13.76 * V_equil^2 / R)
                cant_equilibrium_mm = (13.76 * self.equilibrium_speed_kmh**2) / curve_radius_m if curve_radius_m > 0 else 0.0
                cant_goods_mm = (13.76 * self.goods_speed_kmh**2) / curve_radius_m if curve_radius_m > 0 else 0.0

                raw_cant_mm = min(cant_equilibrium_mm, self.max_actual_cant_mm, cant_goods_mm + self.max_cant_excess_mm)
                actual_cant_mm = min(float(round(raw_cant_mm / 5.0) * 5.0), self.max_actual_cant_mm)

                # Max Permissible Speed (Vmax = 0.27 * sqrt(R * (Ca + Cd))) floored to lower multiple of 5 km/h
                v_curve_max = 0.27 * math.sqrt(curve_radius_m * (actual_cant_mm + self.max_cant_deficiency_mm)) if curve_radius_m > 0 else self.design_speed_kmh
                permissible_speed_kmh = min(self.design_speed_kmh, float(math.floor(v_curve_max / 5.0) * 5.0))

                actual_cant_deficiency_mm = max(0.0, (13.76 * permissible_speed_kmh**2) / curve_radius_m - actual_cant_mm) if curve_radius_m > 0 else 0.0

                # Cant Excess for slow goods trains (Cex = Ca - 13.76 * Vgoods^2 / R) per IRPWM Para 404
                actual_cant_excess_mm = max(0.0, actual_cant_mm - cant_goods_mm) if curve_radius_m > 0 else 0.0

                use_desirable_transition = self.config.get("horizontal", {}).get("use_desirable_transition", True)
                speed_rate_factor = 0.008 if use_desirable_transition else 0.0056

                spiral_len_cant_deficiency = speed_rate_factor * actual_cant_deficiency_mm * permissible_speed_kmh  # Criterion a: rate of Cd change
                spiral_len_actual_cant = speed_rate_factor * actual_cant_mm * permissible_speed_kmh                   # Criterion b: rate of Ca change
                spiral_len_cant_gradient = 0.72 * actual_cant_mm                                                      # Criterion c: cant gradient 1:360

                # Absolute minimum required Ls floor per IRPWM Para 405
                abs_min_ls_req = max(0.0056 * actual_cant_mm * permissible_speed_kmh, 0.36 * actual_cant_mm)

                override_spiral_len = self.config.get("horizontal", {}).get("transition_length_m")
                if override_spiral_len is not None and float(override_spiral_len) > 0:
                    transition_spiral_length_m = float(override_spiral_len)
                else:
                    raw_ls = max(spiral_len_cant_deficiency, spiral_len_actual_cant, spiral_len_cant_gradient, abs_min_ls_req)
                    # CEILING(Ls, 10): Round UP to nearest multiple of 10m
                    transition_spiral_length_m = float(math.ceil(raw_ls / 10.0) * 10.0)


                # Compute available tangent distance along straight incoming vector from origin/previous CTPT
                ip_point = Point(ip_easting, ip_northing)
                ip_chainage_m = original_line.project(ip_point)

                bearing_in = ip["incoming_bearing_rad"]
                if i == 0:
                    raw_orig_coords = list(original_line.coords)
                    start_easting, start_northing = raw_orig_coords[0][0], raw_orig_coords[0][1]
                    dist_from_prev = math.hypot(ip_easting - start_easting, ip_northing - start_northing)
                    dx = ip_easting - start_easting
                    dy = ip_northing - start_northing
                    dist_tangent_to_ip = dx * math.sin(bearing_in) + dy * math.cos(bearing_in)
                    available_tangent_entry_m = dist_tangent_to_ip
                else:
                    prev_ip_point = Point(merged_intersection_points[i - 1]["easting_m"], merged_intersection_points[i - 1]["northing_m"])
                    prev_ip_chainage_m = original_line.project(prev_ip_point)
                    available_tangent_entry_m = max(0.0, ((ip_chainage_m - prev_ip_chainage_m) - self.min_straight_length_m) / 2.0)

                if i == len(merged_intersection_points) - 1:
                    available_tangent_exit_m = max(0.0, original_line.length - ip_chainage_m)
                else:
                    next_ip_point = Point(merged_intersection_points[i + 1]["easting_m"], merged_intersection_points[i + 1]["northing_m"])
                    next_ip_chainage_m = original_line.project(next_ip_point)
                    # ip_chainage_m already computed at L817 — no need to reproject
                    available_tangent_exit_m = max(0.0, ((next_ip_chainage_m - ip_chainage_m) - self.min_straight_length_m) / 2.0)

                max_available_tangent_m = min(available_tangent_entry_m, available_tangent_exit_m)

                # ── Spiral Shift (S) & Tangent Length (T) ───────────────────
                # Shift S = Ls^2 / (24 * R)
                spiral_shift_m = (transition_spiral_length_m**2) / (24.0 * curve_radius_m) if curve_radius_m > 0 else 0.0

                # Tangent Length T = (Ls / 2) + (R + S) * tan(Δ / 2)
                tangent_length_m = (transition_spiral_length_m / 2.0) + (curve_radius_m + spiral_shift_m) * math.tan(deflection_rad_abs / 2.0)

                # Tangent Overlap Safety Fitting (Scale R if required by tight tangent constraint)
                if max_available_tangent_m > 0 and tangent_length_m > max_available_tangent_m:
                    self.logger.warning(
                        "IP %d: Computed tangent T=%.2fm > available max_T=%.2fm. Scaling R to fit.",
                        i, tangent_length_m, max_available_tangent_m
                    )
                    k_tan = math.tan(deflection_rad_abs / 2.0)
                    if k_tan > 0:
                        if transition_spiral_length_m / 2.0 >= max_available_tangent_m:
                            transition_spiral_length_m = max(abs_min_ls_req, max_available_tangent_m * 0.8)
                            spiral_shift_m = (transition_spiral_length_m**2) / (24.0 * curve_radius_m)

                        tangent_remaining = max_available_tangent_m - (transition_spiral_length_m / 2.0)
                        if tangent_remaining > 0:
                            a_quad = k_tan
                            b_quad = -tangent_remaining
                            c_quad = (k_tan * (transition_spiral_length_m**2)) / 24.0
                            discriminant = b_quad**2 - 4.0 * a_quad * c_quad
                            if discriminant >= 0:
                                radius_solution = (-b_quad + math.sqrt(discriminant)) / (2.0 * a_quad)
                                curve_radius_m = max(self.absolute_minimum_radius_m, min(curve_radius_m, radius_solution))
                            else:
                                radius_fit = tangent_remaining / k_tan
                                curve_radius_m = max(self.absolute_minimum_radius_m, min(curve_radius_m, radius_fit))

                            spiral_shift_m = (transition_spiral_length_m**2) / (24.0 * curve_radius_m)
                            tangent_length_m = (transition_spiral_length_m / 2.0) + (curve_radius_m + spiral_shift_m) * k_tan

                # ── Combined Curve Length (Lc and L_total) ─────────────────
                spiral_angle_rad = transition_spiral_length_m / (2.0 * curve_radius_m) if curve_radius_m > 0 else 0.0
                circular_deflection_rad = deflection_rad_abs - 2.0 * spiral_angle_rad
                if circular_deflection_rad < 0:
                    max_spiral_for_delta = (curve_radius_m * deflection_rad_abs) / 2.0
                    if max_spiral_for_delta + 1e-6 < abs_min_ls_req:
                        raise HorizontalGeometryError(
                            f"IP {i}: available deflection permits at most {max_spiral_for_delta:.2f}m "
                            f"transition, below the {abs_min_ls_req:.2f}m absolute minimum."
                        )
                    transition_spiral_length_m = min(transition_spiral_length_m, max_spiral_for_delta)
                    spiral_angle_rad = transition_spiral_length_m / (2.0 * curve_radius_m)
                    circular_deflection_rad = max(0.0, deflection_rad_abs - 2.0 * spiral_angle_rad)
                    spiral_shift_m = (transition_spiral_length_m**2) / (24.0 * curve_radius_m)
                    tangent_length_m = (transition_spiral_length_m / 2.0) + (curve_radius_m + spiral_shift_m) * math.tan(deflection_rad_abs / 2.0)

                circular_curve_length_m = curve_radius_m * circular_deflection_rad
                total_curve_length_m = circular_curve_length_m + 2.0 * transition_spiral_length_m

                # Curve direction from 2D vector cross product (cross_z < 0 -> Right turn / RHS, cross_z > 0 -> Left turn / LHS)
                cross_z_val = ip.get("cross_z", 0.0)
                curve_direction = "RHS" if cross_z_val < 0 else "LHS"

                degree_of_curve = 1750.0 / curve_radius_m if curve_radius_m > 0 else 0.0
                cant_equil_final_mm = (13.76 * self.equilibrium_speed_kmh**2) / curve_radius_m if curve_radius_m > 0 else 0.0
                cant_goods_final_mm = (13.76 * self.goods_speed_kmh**2) / curve_radius_m if curve_radius_m > 0 else 0.0
                raw_cant_final_mm = min(cant_equil_final_mm, self.max_actual_cant_mm, cant_goods_final_mm + self.max_cant_excess_mm)
                actual_cant_mm = min(float(round(raw_cant_final_mm / 5.0) * 5.0), self.max_actual_cant_mm)
                v_max_final = 0.27 * math.sqrt(curve_radius_m * (actual_cant_mm + self.max_cant_deficiency_mm)) if curve_radius_m > 0 else self.design_speed_kmh
                permissible_speed_kmh = min(self.design_speed_kmh, float(math.floor(v_max_final / 5.0) * 5.0))
                actual_cant_deficiency_mm = max(0.0, (13.76 * permissible_speed_kmh**2) / curve_radius_m - actual_cant_mm) if curve_radius_m > 0 else 0.0
                actual_cant_excess_mm = max(0.0, actual_cant_mm - cant_goods_final_mm) if curve_radius_m > 0 else 0.0

                # ── Final Ls recompute with definitive Ca and V ───────────────
                # Re-evaluate speed_rate_factor here so Pass 2 stays consistent with Pass 1 logic.
                use_desirable_transition_p2 = self.config.get("horizontal", {}).get("use_desirable_transition", True)
                speed_rate_factor = 0.008 if use_desirable_transition_p2 else 0.0056
                override_spiral_len_chk = self.config.get("horizontal", {}).get("transition_length_m")
                if override_spiral_len_chk is not None and float(override_spiral_len_chk) > 0:
                    transition_spiral_length_m = float(override_spiral_len_chk)
                else:
                    ls_a_fin = speed_rate_factor * actual_cant_deficiency_mm * permissible_speed_kmh
                    ls_b_fin = speed_rate_factor * actual_cant_mm * permissible_speed_kmh
                    ls_c_fin = 0.72 * actual_cant_mm
                    abs_min_fin = max(
                        0.0056 * actual_cant_mm * permissible_speed_kmh,
                        0.36 * actual_cant_mm,
                    )
                    raw_ls_fin = max(ls_a_fin, ls_b_fin, ls_c_fin, abs_min_fin)
                    # CEILING(Ls, 10): Round UP to nearest multiple of 10m
                    transition_spiral_length_m = float(math.ceil(raw_ls_fin / 10.0) * 10.0)

                # Cap Ls so the circular arc deflection cannot go negative.
                abs_min_ls_final = max(
                    0.0056 * actual_cant_mm * permissible_speed_kmh, 0.36 * actual_cant_mm
                )
                max_ls_for_delta_fin = (curve_radius_m * deflection_rad_abs) / 2.0 if curve_radius_m > 0 else transition_spiral_length_m
                if transition_spiral_length_m > max_ls_for_delta_fin:
                    self.logger.warning(
                        "IP %d: Final Ls=%.0fm exceeds max allowed by Delta=%.2f deg (%.0fm). Capping.",
                        i, transition_spiral_length_m, abs(deflection_deg), max_ls_for_delta_fin,
                    )
                    transition_spiral_length_m = max(abs_min_ls_final, min(transition_spiral_length_m, max_ls_for_delta_fin))

                # Recompute shift, tangent, and arc geometry from final Ls.
                spiral_shift_m = (transition_spiral_length_m**2) / (24.0 * curve_radius_m) if curve_radius_m > 0 else 0.0
                tangent_length_m = (
                    (transition_spiral_length_m / 2.0)
                    + (curve_radius_m + spiral_shift_m) * math.tan(deflection_rad_abs / 2.0)
                )
                if max_available_tangent_m > 0 and tangent_length_m > max_available_tangent_m:
                    self.logger.warning(
                        "IP %d: Final T=%.2fm > available tangent=%.2fm after Ls recompute. "
                        "Consider wider curve spacing or reduced design speed.",
                        i, tangent_length_m, max_available_tangent_m,
                    )
                spiral_angle_rad = transition_spiral_length_m / (2.0 * curve_radius_m) if curve_radius_m > 0 else 0.0
                circular_deflection_rad = deflection_rad_abs - 2.0 * spiral_angle_rad
                if circular_deflection_rad < 0:
                    max_spiral_for_delta = (curve_radius_m * deflection_rad_abs) / 2.0
                    if max_spiral_for_delta + 1e-6 < abs_min_ls_final:
                        raise HorizontalGeometryError(
                            f"IP {i}: final deflection permits at most {max_spiral_for_delta:.2f}m "
                            f"transition, below the {abs_min_ls_final:.2f}m absolute minimum."
                        )
                    transition_spiral_length_m = min(transition_spiral_length_m, max_spiral_for_delta)
                    spiral_shift_m = (transition_spiral_length_m**2) / (24.0 * curve_radius_m)
                    spiral_angle_rad = transition_spiral_length_m / (2.0 * curve_radius_m)
                    circular_deflection_rad = max(0.0, deflection_rad_abs - 2.0 * spiral_angle_rad)
                    tangent_length_m = (
                        (transition_spiral_length_m / 2.0)
                        + (curve_radius_m + spiral_shift_m) * math.tan(deflection_rad_abs / 2.0)
                    )
                circular_curve_length_m = curve_radius_m * circular_deflection_rad
                total_curve_length_m = circular_curve_length_m + 2.0 * transition_spiral_length_m

                # ── Stationing: TPTC / TPCC / CTPC / CTPT chainages ─────────
                bearing_out = ip["outgoing_bearing_rad"]
                start_of_transition_chainage_m = max(0.0, ip_chainage_m - tangent_length_m)

                # A curve beginning before the alignment origin cannot be
                # represented without truncating its required transition.
                if start_of_transition_chainage_m == 0.0 and transition_spiral_length_m > ip_chainage_m:
                    raise HorizontalGeometryError(
                        f"IP {i}: curve at Ch {ip_chainage_m:.2f}m needs a transition beginning before "
                        "the alignment origin; refusing to shorten the required transition."
                    )

                start_of_circular_arc_chainage_m = start_of_transition_chainage_m + transition_spiral_length_m
                end_of_circular_arc_chainage_m = start_of_circular_arc_chainage_m + circular_curve_length_m
                end_of_transition_chainage_m = end_of_circular_arc_chainage_m + transition_spiral_length_m

                # ── 2D TPTC / CTPT Points (bearing-projected from IP) ─────────
                tptc_easting = ip_easting - tangent_length_m * math.sin(bearing_in)
                tptc_northing = ip_northing - tangent_length_m * math.cos(bearing_in)
                ctpt_easting = ip_easting + tangent_length_m * math.sin(bearing_out)
                ctpt_northing = ip_northing + tangent_length_m * math.cos(bearing_out)

                ip_records.append({
                    "ip_index": len(ip_records) + 1,
                    "easting": ip_easting,
                    "northing": ip_northing,
                    "chainage_m": ip_chainage_m,
                    "chainage_str": format_chainage(ip_chainage_m),
                    "deflection_deg": deflection_deg,
                    "deflection_angle_deg": abs(deflection_deg),
                    "direction": curve_direction,
                    "radius_m": curve_radius_m,
                    "radius2_m": None,           # filled by compound curve merge if applicable
                    "curve_type": "SIMPLE",       # SIMPLE | COMPOUND | REVERSE_VIOLATION
                    "degree_of_curve": degree_of_curve,
                    "transition_length_m": transition_spiral_length_m,
                    "shift_m": spiral_shift_m,
                    "tangent_length_m": tangent_length_m,
                    "circular_curve_length_m": circular_curve_length_m,
                    "curve_length_m": total_curve_length_m,
                    "tptc_chainage_m": start_of_transition_chainage_m,
                    "tpcc_chainage_m": start_of_circular_arc_chainage_m,
                    "ctpc_chainage_m": end_of_circular_arc_chainage_m,
                    "ctpt_chainage_m": end_of_transition_chainage_m,
                    "ctpt_easting": ctpt_easting,
                    "ctpt_northing": ctpt_northing,
                    # Tangent bearings stored so drawing engine can render mathematically
                    "incoming_bearing_rad": bearing_in,
                    "outgoing_bearing_rad": bearing_out,
                    "cant_mm": actual_cant_mm,
                    "actual_cd_mm": actual_cant_deficiency_mm,
                    "cant_excess_mm": actual_cant_excess_mm,
                    "permissible_speed_kmph": permissible_speed_kmh,
                })

            valid_ip_records: list[dict] = []
            for rec in ip_records:
                rec_ts = rec["tptc_chainage_m"]
                rec_st = rec["ctpt_chainage_m"]
                overlap_found = False
                for j, accepted in enumerate(valid_ip_records):
                    acc_ts = accepted["tptc_chainage_m"]
                    acc_st = accepted["ctpt_chainage_m"]
                    # Check if spans overlap (they overlap when NOT disjoint)
                    if not (rec_st <= acc_ts or rec_ts >= acc_st):
                        # Overlap — keep the curve with the larger deflection angle
                        if rec["deflection_angle_deg"] > accepted["deflection_angle_deg"]:
                            self.logger.warning(
                                "ARTEFACT FILTER — Curve #%d (Delta=%.1f deg, TS=%.0fm->ST=%.0fm) "
                                "overlaps Curve #%d (Delta=%.1f deg). Replacing with larger-deflection curve.",
                                rec["ip_index"], rec["deflection_angle_deg"], rec_ts, rec_st,
                                accepted["ip_index"], accepted["deflection_angle_deg"],
                            )
                            valid_ip_records[j] = rec
                        else:
                            self.logger.warning(
                                "ARTEFACT FILTER — Curve #%d (Delta=%.1f deg, TS=%.0fm->ST=%.0fm) "
                                "overlaps Curve #%d (Delta=%.1f deg). Discarding as tessellation artefact.",
                                rec["ip_index"], rec["deflection_angle_deg"], rec_ts, rec_st,
                                accepted["ip_index"], accepted["deflection_angle_deg"],
                            )
                            overlap_found = True
                        break
                if not overlap_found:
                    valid_ip_records.append(rec)
            ip_records = valid_ip_records

            # ── Step 5: IRPWM Para 406 — Compound Curve Merge & Violation Check ──
            ip_records = self._merge_compound_curves(ip_records)

            # Re-index curves sequentially (1, 2, 3, ...) in true track station order
            for idx, r in enumerate(ip_records):
                r["ip_index"] = idx + 1

            df_ips = pd.DataFrame(ip_records)
            total_ips = len(df_ips)

            compound_count = sum(1 for r in ip_records if r.get("curve_type") == "COMPOUND")
            violation_count = sum(1 for r in ip_records if r.get("curve_type") == "REVERSE_VIOLATION")

            self.logger.info(
                "Horizontal Geometry complete. %d curves total: %d SIMPLE, %d COMPOUND, %d REVERSE_VIOLATION "
                "(original vertices: %d, simplified: %d).",
                total_ips,
                total_ips - compound_count - violation_count,
                compound_count,
                violation_count,
                original_vertex_count,
                simplified_vertex_count
            )

            metadata = {
                "total_ips": total_ips,
                "compound_curves": compound_count,
                "reverse_violations": violation_count,
                "min_deflection_threshold_deg": self.min_deflection_angle_deg,
                "default_design_radius_m": self.default_design_radius_m,
                "simplify_tolerance_m": self.simplify_tolerance_m,
                "cad_kml_mode": self.cad_kml_mode,
                "original_vertices": original_vertex_count,
                "simplified_vertices": simplified_vertex_count,
                "merged_groups_debug": [
                    {
                        "merged_ip_index": i + 1,
                        "vertex_count": len(g),
                        "raw_vertices": g
                    }
                    for i, g in enumerate(ip_groups) if len(g) > 1
                ]
            }

            return HGeomResult(ip_dataframe=df_ips, metadata=metadata)

        except Exception as err:
            self.logger.error("Horizontal geometry analysis failed: %s", str(err))
            raise HorizontalGeometryError(
                f"Horizontal geometry failed: {err}"
            ) from err

    def _merge_compound_curves(self, ip_records: list[dict]) -> list[dict]:
        """Post-process detected IPs to enforce IRPWM Para 406 minimum straight between curves.

        Engineering Rules Applied (IRPWM Para 406 — BG):
            1. Same-direction consecutive curves (LHS→LHS or RHS→RHS) with a gap
               between CTPT_A and TPTC_B < ``min_straight_between_curves_m`` (50 m):
               → Merge into a single COMPOUND curve record.
               → Both radii are preserved (R1 on 'radius_m', R2 on 'radius2_m').
               → Total span = TPTC_A to CTPT_B.
               → Permissible speed = min(V_A, V_B) [conservative].

            2. Reverse-direction consecutive curves (LHS→RHS or RHS→LHS) with gap
               less than Ls_A + Ls_B (IRPWM Para 406 requirement):
               → Flagged as 'REVERSE_VIOLATION' and a WARNING is logged.
               → NOT merged — kept separate for engineer review.

        Args:
            ip_records: List of raw curve parameter dicts from analyze().

        Returns:
            Processed list with compound curves merged and violations flagged.
        """
        if len(ip_records) < 2:
            return ip_records

        merged: list[dict] = []
        i = 0

        while i < len(ip_records):
            cur = dict(ip_records[i])

            if i + 1 < len(ip_records):
                nxt = ip_records[i + 1]
                gap_m = nxt["tptc_chainage_m"] - cur["ctpt_chainage_m"]
                same_direction = cur["direction"] == nxt["direction"]

                # ── Case 1: Same direction + gap too small → COMPOUND curve ──
                if same_direction and gap_m < self.min_straight_between_curves_m:
                    self.logger.warning(
                        "COMPOUND CURVE: Curve #%d (%s, R=%.0fm, CTPT=%.0fm) and "
                        "Curve #%d (%s, R=%.0fm, TPTC=%.0fm) have gap=%.1fm < min_straight=%.0fm. "
                        "Merging as COMPOUND curve per IRPWM Para 406.",
                        cur["ip_index"], cur["direction"], cur["radius_m"], cur["ctpt_chainage_m"],
                        nxt["ip_index"], nxt["direction"], nxt["radius_m"], nxt["tptc_chainage_m"],
                        gap_m, self.min_straight_between_curves_m
                    )
                    compound = dict(cur)
                    compound["curve_type"]              = "COMPOUND"
                    compound["radius2_m"]               = nxt["radius_m"]
                    # Extend the compound curve's exit to the second curve's exit
                    compound["ctpc_chainage_m"]         = nxt["ctpc_chainage_m"]
                    compound["ctpt_chainage_m"]         = nxt["ctpt_chainage_m"]
                    compound["ctpt_easting"]            = nxt.get("ctpt_easting", cur["ctpt_easting"])
                    compound["ctpt_northing"]           = nxt.get("ctpt_northing", cur["ctpt_northing"])
                    # Total curve span from TPTC_A to CTPT_B
                    compound["curve_length_m"]          = nxt["ctpt_chainage_m"] - cur["tptc_chainage_m"]
                    compound["circular_curve_length_m"] = nxt["ctpc_chainage_m"] - cur["tpcc_chainage_m"]
                    # Combined deflection angle
                    compound["deflection_angle_deg"]    = cur["deflection_angle_deg"] + nxt["deflection_angle_deg"]
                    compound["deflection_deg"]          = (
                        compound["deflection_angle_deg"]
                        if cur["deflection_deg"] > 0 else -compound["deflection_angle_deg"]
                    )
                    # Conservative: use the lower permissible speed & higher cant
                    compound["permissible_speed_kmph"]  = min(cur["permissible_speed_kmph"], nxt["permissible_speed_kmph"])
                    compound["cant_mm"]                 = max(cur["cant_mm"], nxt["cant_mm"])
                    compound["actual_cd_mm"]            = max(cur["actual_cd_mm"], nxt["actual_cd_mm"])

                    # ── IRPWM Para 405(6) & Para 403(2) Common Transition Calculation ──
                    cant1 = cur["cant_mm"]
                    cant2 = nxt["cant_mm"]
                    cd1 = cur["actual_cd_mm"]
                    cd2 = nxt["actual_cd_mm"]
                    v_max = compound["permissible_speed_kmph"]

                    delta_ca = abs(cant1 - cant2)
                    delta_cd = abs(cd1 - cd2)

                    use_desirable = self.config.get("horizontal", {}).get("use_desirable_transition", True)
                    speed_rate_factor = 0.008 if use_desirable else 0.0056

                    # IRPWM Para 405(6) Common Transition Criteria:
                    #   (a) L_a = factor * |Ca1 - Ca2| * Vm
                    #   (b) L_b = factor * |Cd1 - Cd2| * Vm
                    #   (c) Cant Gradient limit 1:360 (0.72 * |Ca1 - Ca2|)
                    l_common_a = speed_rate_factor * delta_ca * v_max
                    l_common_b = speed_rate_factor * delta_cd * v_max
                    l_common_c = 0.72 * delta_ca

                    l_common_raw = max(l_common_a, l_common_b, l_common_c)

                    # Virtual Transition Evaluation per IRPWM Para 403(2) & Para 405(6):
                    # Virtual transition threshold ~ 14.6m (BG bogie center distance / rigid wheelbase)
                    l_virtual_threshold_m = float(
                        self.config.get("horizontal", {}).get("virtual_transition_length_m", 14.6)
                    )

                    if l_common_raw > l_virtual_threshold_m:
                        common_transition_m = float(math.ceil(l_common_raw / 10.0) * 10.0)
                        physical_transition_required = True
                        self.logger.info(
                            "COMPOUND TRANSITION (IRPWM Para 405(6)): L_raw=%.1fm > L_virtual=%.1fm. "
                            "Physical common transition of %.0fm provided between Curve #%d and #%d (ΔCa=%.1fmm, ΔCd=%.1fmm, Vm=%.0fkm/h).",
                            l_common_raw, l_virtual_threshold_m, common_transition_m,
                            cur["ip_index"], nxt["ip_index"], delta_ca, delta_cd, v_max
                        )
                    else:
                        common_transition_m = 0.0
                        physical_transition_required = False
                        self.logger.info(
                            "COMPOUND TRANSITION (IRPWM Para 403(2)): L_raw=%.1fm <= L_virtual=%.1fm. "
                            "Physical common transition omitted (direct PCC junction at curves #%d & #%d).",
                            l_common_raw, l_virtual_threshold_m, cur["ip_index"], nxt["ip_index"]
                        )

                    compound["common_transition_length_m"] = common_transition_m
                    compound["common_transition_required"] = physical_transition_required
                    compound["delta_cant_mm"] = delta_ca
                    compound["delta_cd_mm"] = delta_cd
                    compound["cant1_mm"] = cant1
                    compound["cant2_mm"] = cant2

                    merged.append(compound)
                    i += 2  # consumed two records
                    continue

                # ── Case 2: Reverse direction, check Ls1 + Ls2 min gap ────────
                elif not same_direction:
                    min_gap_required_m = cur["transition_length_m"] + nxt["transition_length_m"]
                    if gap_m < min_gap_required_m:
                        self.logger.warning(
                            "IRPWM PARA 406 VIOLATION — Reverse curve pair: "
                            "Curve #%d (%s, Ls=%.0fm, CTPT=%.0fm) -> Curve #%d (%s, Ls=%.0fm, TPTC=%.0fm): "
                            "gap=%.1fm < required=%.0fm (Ls1+Ls2). "
                            "Non-standard design — flag for engineer review.",
                            cur["ip_index"], cur["direction"], cur["transition_length_m"], cur["ctpt_chainage_m"],
                            nxt["ip_index"], nxt["direction"], nxt["transition_length_m"], nxt["tptc_chainage_m"],
                            gap_m, min_gap_required_m
                        )
                        cur["curve_type"] = "REVERSE_VIOLATION"

            merged.append(cur)
            i += 1

        # Re-number ip_index sequentially after merges
        for idx, rec in enumerate(merged):
            rec["ip_index"] = idx + 1

        return merged

    @staticmethod
    def _detect_kml_entity_boundaries(original_line: LineString) -> dict[str, float] | None:
        """Detect exact TPTC, TPCC, CTPC, CTPT chainages from CAD KML segment length transitions.

        Civil 3D / CAD KML exports tessellate entities with distinct segment lengths:
          - Straight tangent: single long segment (> 30m)
          - Transition spiral: short dense segments (< 15m)
          - Circular arc: medium uniform segments (>= 25m)
        """
        coords = list(original_line.coords)
        if len(coords) < 5:
            return None

        chainages = [0.0]
        seg_lens = []
        cumul = 0.0
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i - 1][0]
            dy = coords[i][1] - coords[i - 1][1]
            sl = math.hypot(dx, dy)
            cumul += sl
            chainages.append(cumul)
            seg_lens.append(sl)

        tptc_ch = None
        tpcc_ch = None
        ctpc_ch = None
        ctpt_ch = None

        for i in range(len(seg_lens) - 1):
            # Entry Tangent -> Spiral: long segment followed by short segment
            if tptc_ch is None and seg_lens[i] > 30.0 and seg_lens[i + 1] < 15.0:
                tptc_ch = chainages[i + 1]

            # Entry Spiral -> Circular Arc: short segment followed by medium arc segment (>= 25m)
            if tptc_ch is not None and tpcc_ch is None and seg_lens[i] < 15.0 and seg_lens[i + 1] >= 25.0:
                tpcc_ch = chainages[i + 1]

            # Circular Arc -> Exit Spiral: medium arc segment followed by short segment (< 15m)
            if tpcc_ch is not None and ctpc_ch is None and seg_lens[i] >= 25.0 and seg_lens[i + 1] < 15.0:
                ctpc_ch = chainages[i + 1]

            # Exit Spiral -> Exit Tangent: short segment followed by long tangent (> 30m)
            if ctpc_ch is not None and ctpt_ch is None and seg_lens[i] < 15.0 and seg_lens[i + 1] > 30.0:
                ctpt_ch = chainages[i + 1]

        if tptc_ch is not None and tpcc_ch is not None and ctpc_ch is not None and ctpt_ch is not None:
            return {
                "tptc_chainage_m": tptc_ch,
                "tpcc_chainage_m": tpcc_ch,
                "ctpc_chainage_m": ctpc_ch,
                "ctpt_chainage_m": ctpt_ch,
                "spiral_len_m": tpcc_ch - tptc_ch,
                "arc_len_m": ctpc_ch - tpcc_ch,
            }
        return None

    def export_csv(
        self,
        result: HGeomResult,
        output_path: str | Path,
    ) -> Path:
        """Export the Horizontal Geometry IP DataFrame to CSV.

        Args:
            result: HGeomResult from analyze().
            output_path: Destination CSV file path.

        Returns:
            Resolved path to the written CSV file.
        """
        output_file_path = Path(output_path).resolve()
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        if not result.ip_dataframe.empty:
            result.ip_dataframe.to_csv(output_file_path, index=False, float_format="%.4f")
            self.logger.info("Horizontal Geometry exported to CSV: %s", output_file_path)
        else:
            self.logger.info("No IPs to export; CSV not created.")
        return output_file_path


if __name__ == '__main__':
    import sys
    import yaml

    # Add project root directory to Python path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from modules.read_kml import KMLReader
    from modules.generate_chainage import ChainageGenerator

    config_dict = {}
    config_file_path = Path("config/config.yaml")
    if config_file_path.exists():
        with open(config_file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

    kml_result = KMLReader(config_dict).read(r"D:\ESP\Drawing5.kml")
    chainage_result = ChainageGenerator(config_dict).generate(kml_result)
    
    hgeom_result = HorizontalGeometry(config_dict).analyze(kml_result)

    print("\n========================================================")
    print("      HORIZONTAL GEOMETRY TRANSITION CURVE SUMMARY      ")
    print("========================================================")
    
    # Print pre-merge raw arc vertices if any groups exist
    merged_groups = hgeom_result.metadata.get("merged_groups_debug", [])
    if merged_groups:
        print("\n🔍 RAW PRE-MERGE ARC VERTEX GROUPS:")
        for mg in merged_groups:
            print(f"\n   --- Merged IP #{mg['merged_ip_index']} ({mg['vertex_count']} raw vertices) ---")
            print(f"   {'#':<4} {'V_Idx':<7} {'Easting (m)':<13} {'Northing (m)':<13} {'Dist_Prev (m)':<14} {'Deflection (°)':<14}")
            print(f"   {'-'*65}")
            for idx_v, v in enumerate(mg["raw_vertices"]):
                dist_p = 0.0
                if idx_v > 0:
                    dist_p = math.hypot(
                        v["easting_m"] - mg["raw_vertices"][idx_v - 1]["easting_m"],
                        v["northing_m"] - mg["raw_vertices"][idx_v - 1]["northing_m"]
                    )
                print(f"   {idx_v+1:<4} {v['vertex_index']:<7} {v['easting_m']:<13.2f} {v['northing_m']:<13.2f} {dist_p:<14.1f} {v['deflection_deg']:<14.2f}")

    if not hgeom_result.ip_dataframe.empty:
        for idx, row in hgeom_result.ip_dataframe.iterrows():
            breakpoint()
            print(f"\n📍 CURVE #{int(row.get('ip_index', idx+1))} ({row.get('direction', 'RHS')}):")
            print(f"   • Deflection Angle (Δ)   : {row.get('deflection_deg', 0.0):.2f}°")
            print(f"   • Curve Radius (R)       : {row.get('radius_m', 0.0):.2f} m")
            print(f"   • Permissible Speed (V)  : {row.get('permissible_speed_kmph', 0.0):.0f} km/h")
            print(f"   • Transition Length (Ls) : {row.get('transition_length_m', 0.0):.2f} m")
            print(f"   ----------------------------------------------------")
            print(f"   • TPTC (Entrance Start)  : {row.get('tptc_chainage_m', 0.0):.2f} m  (Km {row.get('tptc_chainage_m', 0.0)/1000:.3f})")
            print(f"   • TPCC (Circular Start)  : {row.get('tpcc_chainage_m', 0.0):.2f} m  (Km {row.get('tpcc_chainage_m', 0.0)/1000:.3f})")
            print(f"   • CTPC (Circular End)    : {row.get('ctpc_chainage_m', 0.0):.2f} m  (Km {row.get('ctpc_chainage_m', 0.0)/1000:.3f})")
            print(f"   • CTPT (Exit End)        : {row.get('ctpt_chainage_m', 0.0):.2f} m  (Km {row.get('ctpt_chainage_m', 0.0)/1000:.3f})")
    print("========================================================\n")
    HorizontalGeometry(config_dict).export_csv(hgeom_result, "outputs/horizontal_geometry.csv")
