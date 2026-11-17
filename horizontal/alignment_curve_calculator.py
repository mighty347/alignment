"""
Module: future_alignment_studio/horizontal/alignment_curve_calculator.py
Purpose: Complete Railway Alignment Curve Geometry Calculator (RDSO / IRPWM Compliant).

Engineering Logic:
1. Straight Tangent Segments (Deflection Angle Δ = 0):
   - Retains strictly 2 points (Start Point -> End Point).
   - No redundant intermediate vertices.
2. Curve Feature (Deflection Angle Δ > 0 at a Point of Intersection):
   - Entry Transition Spiral (TS -> SC):
     Continuous curvature transition (0 -> 1/R).
     Rendered as multiple chord segments (RDSO Cubic Parabola: y = x^3 / 6RL).
   - Circular Arc (SC -> CS):
     Constant curvature (1/R).
     Represented as true circular arc geometry (Center, Radius R, Start/End angles, Length Lc).
   - Exit Transition Spiral (CS -> ST):
     Continuous curvature transition (1/R -> 0).
     Rendered as multiple chord segments leading smoothly into the next tangent.
   - Following Tangent:
     Continues from ST with strictly 2 points to the next TS or corridor end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

try:
    from alignment.geospatial.kml_ingestor import AlignmentStationPoint
    from alignment.horizontal._alignment_sampler import (
        discretize_arc,
        extract_centerline_coords,
        generate_transition_points,
        sample_alignment_stations,
    )
    from alignment.horizontal.curve_solver import (
        CurveDirection,
        HorizontalCurveSolver,
        RDSOHorizontalCurve,
    )
    from alignment.utils.logger import get_logger
except ImportError:
    try:
        from geospatial.kml_ingestor import AlignmentStationPoint
        from horizontal._alignment_sampler import (
            discretize_arc,
            extract_centerline_coords,
            generate_transition_points,
            sample_alignment_stations,
        )
        from horizontal.curve_solver import (
            CurveDirection,
            HorizontalCurveSolver,
            RDSOHorizontalCurve,
        )
        from utils.logger import get_logger
    except ImportError:
        from future_alignment_studio.geospatial.kml_ingestor import AlignmentStationPoint
        from future_alignment_studio.horizontal._alignment_sampler import (
            discretize_arc,
            extract_centerline_coords,
            generate_transition_points,
            sample_alignment_stations,
        )
        from future_alignment_studio.horizontal.curve_solver import (
            CurveDirection,
            HorizontalCurveSolver,
            RDSOHorizontalCurve,
        )
        from future_alignment_studio.utils.logger import get_logger

logger = get_logger("AlignmentCurveCalculator")


class AlignmentSegmentType(str, Enum):
    """Classification of horizontal alignment geometry segments."""
    TANGENT = "TANGENT"
    TRANSITION_ENTRY = "TRANSITION_ENTRY"
    CIRCULAR_ARC = "CIRCULAR_ARC"
    TRANSITION_EXIT = "TRANSITION_EXIT"


@dataclass(frozen=True)
class ArcGeometry:
    """Exact mathematical parameters for a true circular arc in CAD model space."""
    center_easting_m: float
    center_northing_m: float
    radius_m: float
    start_angle_rad: float
    end_angle_rad: float
    start_angle_deg: float
    end_angle_deg: float
    sweep_angle_deg: float
    direction: CurveDirection
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    length_m: float


@dataclass
class AlignmentElement:
    """A distinct structural geometric element along the railway alignment."""
    element_id: str
    element_type: AlignmentSegmentType
    start_chainage_m: float
    end_chainage_m: float
    length_m: float
    points: List[Tuple[float, float]]
    arc_geom: Optional[ArcGeometry] = None
    curve_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TangentPointStation:
    """RDSO tangent point marker along the horizontal alignment (TPTC-1, TPCC-1, TPCC-2, TPTC-2)."""
    curve_id: str
    curve_index: int
    station_code: str                  # "TS", "SC", "CS", "ST"
    tp_label: str                      # "TPTC-1", "TPCC-1", "TPCC-2", "TPTC-2"
    chainage_m: float
    easting_m: float
    northing_m: float
    direction: CurveDirection
    tangent_unit: Tuple[float, float]
    normal_unit: Tuple[float, float]   # Outward unit normal (away from center of curvature)
    radius_m: float
    transition_length_m: float
    has_transition: bool


@dataclass
class CalculatedAlignment:
    """Complete alignment geometry result containing classified elements and solved curves."""
    total_length_m: float
    elements: List[AlignmentElement]
    curves: List[RDSOHorizontalCurve]
    critical_stations: List[Dict[str, Any]]
    summary: Dict[str, Any]
    tangent_points: List[TangentPointStation] = field(default_factory=list)
    start_chainage_m: float = 0.0
    epsg: Optional[int] = None


class AlignmentCurveCalculator:
    """
    Computes horizontal alignment geometry from a network of tangent intersection points (PIs).
    Enforces 2-point straight tangents and 3-phase curves (Entry Transition -> Arc -> Exit Transition).
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config_path = config_path
            self.config = config
        else:
            if config_path is None:
                config_path = (
                    Path(__file__).resolve().parent.parent / "config/alignment_config.yaml"
                )
            self.config_path = config_path
            with self.config_path.open(encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        railway_std = self.config.get("railway_standards", {})
        curve_std = self.config.get("curve_standards", {})

        self.chainage_interval_m = float(railway_std.get("chainage_interval_m", 20.0))
        self.dynamic_gauge_mm = float(railway_std.get("dynamic_gauge_mm", 1750.0))
        self.default_speed_kmh = float(railway_std.get("default_design_speed_kmh", 160.0))
        self.max_cant_mm = float(curve_std.get("max_cant_actual_mm", 165.0))
        self.max_cant_deficiency_mm = float(curve_std.get("max_cant_deficiency_mm", 75.0))
        self.transition_chord_step_m = float(curve_std.get("transition_chord_step_m", 5.0))
        self.min_deflection_threshold_deg = float(
            curve_std.get("min_deflection_threshold_deg", 0.05)
        )
        self.min_transition_deflection_deg = float(
            curve_std.get("min_transition_deflection_deg", 0.5)
        )
        self.max_transition_spiral_ratio = float(
            curve_std.get("max_transition_spiral_ratio", 0.425)
        )
        self.default_radius_m = float(curve_std.get("default_radius_m", 1000.0))
        self.radius_round_interval_m = float(curve_std.get("radius_round_interval_m", 25.0))
        self.min_fallback_radius_m = float(curve_std.get("min_fallback_radius_m", 175.0))
        self.fallback_tangent_headroom_ratio = float(
            curve_std.get("fallback_tangent_headroom_ratio", 0.90)
        )

        tp_std = self.config.get("tangent_point_standards", {})
        self.tp_labels: Dict[str, str] = {
            "TS": str(tp_std.get("entry_tangent_to_transition_label", "TPTC-1")),
            "SC": str(tp_std.get("entry_transition_to_circular_label", "TPCC-1")),
            "CS": str(tp_std.get("exit_circular_to_transition_label", "TPCC-2")),
            "ST": str(tp_std.get("exit_transition_to_tangent_label", "TPTC-2")),
        }
        self.chainage_precision = int(tp_std.get("chainage_precision", 3))

        self.solver = HorizontalCurveSolver(
            dynamic_gauge_mm=self.dynamic_gauge_mm,
            default_speed_kmh=self.default_speed_kmh,
            max_cant_mm=self.max_cant_mm,
            max_cant_deficiency_mm=self.max_cant_deficiency_mm,
            min_transition_deflection_deg=self.min_transition_deflection_deg,
            max_transition_spiral_ratio=self.max_transition_spiral_ratio,
        )

    def _make_tangent_element(
        self,
        elem_idx: int,
        start_ch: float,
        p_start: Tuple[float, float],
        p_end: Tuple[float, float],
        length_m: float,
    ) -> AlignmentElement:
        """Build a strictly 2-point straight tangent element."""
        return AlignmentElement(
            element_id=f"TANGENT-{elem_idx}",
            element_type=AlignmentSegmentType.TANGENT,
            start_chainage_m=round(start_ch, self.chainage_precision),
            end_chainage_m=round(start_ch + length_m, self.chainage_precision),
            length_m=round(length_m, self.chainage_precision),
            points=[p_start, p_end],  # STRICTLY 2 POINTS
            metadata={"bearing_deg": round(self._calculate_bearing(p_start, p_end), 2)},
        )

    def calculate_alignment(
        self,
        control_points: Sequence[Tuple[float, float]],
        curve_specs: Optional[Dict[int, Dict[str, Any]]] = None,
        start_chainage_m: float = 0.0,
        epsg: Optional[int] = None,
    ) -> CalculatedAlignment:
        """Calculate horizontal alignment elements from traverse control points [P_start, PI_1, ..., P_end]."""
        if len(control_points) < 2:
            raise ValueError("Alignment requires at least two control points (Start and End).")
        if not math.isfinite(start_chainage_m):
            raise ValueError(f"start_chainage_m must be finite; got {start_chainage_m!r}")

        for idx, (x, y) in enumerate(control_points):
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"Control point {idx} contains non-finite coordinates: ({x}, {y})")

        if curve_specs is None:
            curve_specs = {}

        elements: List[AlignmentElement] = []
        solved_curves: List[RDSOHorizontalCurve] = []
        critical_stations: List[Dict[str, Any]] = []
        tangent_points: List[TangentPointStation] = []

        # If only 2 points, pure tangent line (Deflection Δ = 0)
        if len(control_points) == 2:
            p0, p1 = control_points[0], control_points[1]
            dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if dist <= 0.0:
                raise ValueError("Control points must be distinct.")

            elements.append(self._make_tangent_element(1, start_chainage_m, p0, p1, dist))
            ch_prec = self.chainage_precision
            return CalculatedAlignment(
                total_length_m=round(dist, ch_prec),
                elements=elements,
                curves=[],
                critical_stations=[
                    {"name": "START", "chainage_m": round(start_chainage_m, ch_prec), "coord": p0},
                    {"name": "END", "chainage_m": round(start_chainage_m + dist, ch_prec), "coord": p1},
                ],
                summary={
                    "tangents_count": 1,
                    "curves_count": 0,
                    "tangent_points_count": 0,
                    "total_length_m": round(dist, ch_prec),
                },
                tangent_points=[],
                start_chainage_m=round(start_chainage_m, ch_prec),
                epsg=epsg,
            )

        current_chainage = start_chainage_m
        last_exit_point = control_points[0]
        critical_stations.append(
            {"name": "START", "chainage_m": round(current_chainage, self.chainage_precision), "coord": last_exit_point}
        )

        n = len(control_points)
        elem_idx = 1
        curve_counter = 1

        for i in range(1, n - 1):
            p_prev = control_points[i - 1]
            pi = control_points[i]
            p_next = control_points[i + 1]

            dx1, dy1 = pi[0] - p_prev[0], pi[1] - p_prev[1]
            len1 = math.hypot(dx1, dy1)
            dx2, dy2 = p_next[0] - pi[0], p_next[1] - pi[1]
            len2 = math.hypot(dx2, dy2)

            if len1 <= 1e-6 or len2 <= 1e-6:
                raise ValueError(f"Zero-length traverse leg at PI {i}.")

            u1x, u1y = dx1 / len1, dy1 / len1
            u2x, u2y = dx2 / len2, dy2 / len2

            cross = u1x * u2y - u1y * u2x
            dot = u1x * u2x + u1y * u2y
            delta_rad = math.atan2(cross, dot)
            delta_deg = math.degrees(delta_rad)
            abs_delta_deg = abs(delta_deg)

            if abs_delta_deg >= 179.9:
                raise ValueError(f"PI {i}: 180-degree reversal cannot form a railway curve.")

            # Case A: Deflection angle below threshold -> Pure 2-point Tangent Leg
            if abs_delta_deg <= self.min_deflection_threshold_deg:
                logger.info(
                    "PI %d: Deflection angle %.3f° <= threshold %.3f°. Kept as 2-point tangent.",
                    i, abs_delta_deg, self.min_deflection_threshold_deg
                )
                tangent_dist = math.hypot(pi[0] - last_exit_point[0], pi[1] - last_exit_point[1])
                if tangent_dist > 1e-4:
                    elements.append(
                        self._make_tangent_element(elem_idx, current_chainage, last_exit_point, pi, tangent_dist)
                    )
                    elem_idx += 1
                    current_chainage += tangent_dist
                    last_exit_point = pi
                continue

            # Case B: Curve Feature
            direction = CurveDirection.LEFT if delta_deg > 0 else CurveDirection.RIGHT
            spec = curve_specs.get(i, {})
            has_explicit_curve = (
                spec.get("radius_m") is not None or spec.get("degree_of_curve") is not None
            )
            radius_m = spec.get("radius_m", self.default_radius_m)
            degree_of_curve = spec.get("degree_of_curve")
            speed_kmh = spec.get("speed_kmh", self.default_speed_kmh)

            curve_index = curve_counter
            curve_id = f"PI-{curve_index}"
            curve_counter += 1

            dist_to_pi = math.hypot(pi[0] - last_exit_point[0], pi[1] - last_exit_point[1])
            approx_pi_chainage = current_chainage + dist_to_pi

            solved_curve = self.solver.solve_curve(
                curve_id=curve_id,
                pi_chainage=approx_pi_chainage,
                deflection_angle_deg=abs_delta_deg,
                direction=direction,
                radius_m=radius_m if degree_of_curve is None else None,
                degree_of_curve=degree_of_curve,
                speed_kmh=speed_kmh,
                bound_transition=True,
            )

            avail_in = (pi[0] - last_exit_point[0]) * u1x + (pi[1] - last_exit_point[1]) * u1y
            headroom = min(avail_in, len2) if i == n - 2 else avail_in
            if not has_explicit_curve and headroom > 0.0 and solved_curve.tangent_length_m > headroom:
                tan_half = max(math.tan(abs(delta_rad) / 2.0), 1e-6)
                target_t = headroom * self.fallback_tangent_headroom_ratio
                est_r = (target_t - (solved_curve.transition_length_m / 2.0)) / tan_half
                if est_r <= self.min_fallback_radius_m:
                    est_r = target_t / tan_half
                if self.radius_round_interval_m > 0:
                    est_r = math.floor(est_r / self.radius_round_interval_m) * self.radius_round_interval_m
                r_fallback = max(self.min_fallback_radius_m, est_r)
                if r_fallback < solved_curve.radius_m:
                    logger.info(
                        "Curve %s: Default R=%.1fm tangent (%.2fm) exceeds leg headroom (%.2fm); using R=%.1fm.",
                        curve_id, solved_curve.radius_m, solved_curve.tangent_length_m, headroom, r_fallback,
                    )
                    solved_curve = self.solver.solve_curve(
                        curve_id=curve_id,
                        pi_chainage=approx_pi_chainage,
                        deflection_angle_deg=abs_delta_deg,
                        direction=direction,
                        radius_m=r_fallback,
                        speed_kmh=speed_kmh,
                        bound_transition=True,
                    )

            t_len = solved_curve.tangent_length_m
            lt = solved_curve.transition_length_m
            r = solved_curve.radius_m
            theta_s_rad = (lt / (2.0 * r)) if lt > 0.0 else 0.0
            delta_c_rad = max(0.0, abs(delta_rad) - (2.0 * theta_s_rad))

            # 1. Critical Coordinates: TS and ST
            ts_coord = (pi[0] - t_len * u1x, pi[1] - t_len * u1y)
            st_coord = (pi[0] + t_len * u2x, pi[1] + t_len * u2y)

            # Guard against overlapping consecutive curves (never emit backward tangents)
            signed_tan_before = (ts_coord[0] - last_exit_point[0]) * u1x + (ts_coord[1] - last_exit_point[1]) * u1y
            if signed_tan_before < -1e-3:
                raise ValueError(
                    f"Curve {curve_id} TS overlaps previous alignment end/ST by {-signed_tan_before:.2f}m "
                    "along incoming tangent. Reduce curve radius/transition length or increase PI spacing."
                )

            tangent_before_len = max(0.0, signed_tan_before)
            if tangent_before_len > 1e-3:
                elements.append(
                    self._make_tangent_element(elem_idx, current_chainage, last_exit_point, ts_coord, tangent_before_len)
                )
                elem_idx += 1
                current_chainage += tangent_before_len

            ch_ts = current_chainage
            self._record_tangent_point(
                tangent_points=tangent_points,
                critical_stations=critical_stations,
                curve_id=curve_id,
                curve_index=curve_index,
                station_code="TS",
                chainage_m=ch_ts,
                coord=ts_coord,
                direction=direction,
                tangent_unit=(u1x, u1y),
                radius_m=r,
                transition_length_m=lt,
            )

            # Inward normal vector at TS
            if direction == CurveDirection.LEFT:
                n1x, n1y = -u1y, u1x
            else:
                n1x, n1y = u1y, -u1x

            # 2. Entry Transition Spiral (TS -> SC): MULTIPLE SEGMENTS
            if lt > 0.0:
                entry_points, sc_coord = self._generate_transition_points(
                    origin=ts_coord,
                    tangent_unit=(u1x, u1y),
                    normal_unit=(n1x, n1y),
                    transition_len=lt,
                    radius=r,
                    is_entry=True,
                )
                entry_points[0] = ts_coord
                entry_points[-1] = sc_coord

                elements.append(
                    AlignmentElement(
                        element_id=f"{curve_id}_TRANSITION_ENTRY",
                        element_type=AlignmentSegmentType.TRANSITION_ENTRY,
                        start_chainage_m=round(current_chainage, 3),
                        end_chainage_m=round(current_chainage + lt, 3),
                        length_m=round(lt, 3),
                        points=entry_points,  # MULTIPLE SEGMENTS
                        curve_id=curve_id,
                        metadata={"segment_count": len(entry_points) - 1, "transition_length_m": lt},
                    )
                )
                elem_idx += 1
                current_chainage += lt
            else:
                sc_coord = ts_coord

            # 3. Circular Curve (SC -> CS): TRUE CIRCULAR ARC
            # Calculate Arc Center C and exact tangents at SC & CS
            bearing_incoming_rad = math.atan2(u1y, u1x)
            bearing_at_sc = (
                bearing_incoming_rad + theta_s_rad
                if direction == CurveDirection.LEFT
                else bearing_incoming_rad - theta_s_rad
            )

            ch_sc = current_chainage
            self._record_tangent_point(
                tangent_points=tangent_points,
                critical_stations=critical_stations,
                curve_id=curve_id,
                curve_index=curve_index,
                station_code="SC",
                chainage_m=ch_sc,
                coord=sc_coord,
                direction=direction,
                tangent_unit=(math.cos(bearing_at_sc), math.sin(bearing_at_sc)),
                radius_m=r,
                transition_length_m=lt,
            )

            radial_angle_sc = (
                bearing_at_sc + math.pi / 2.0
                if direction == CurveDirection.LEFT
                else bearing_at_sc - math.pi / 2.0
            )
            center_x = sc_coord[0] + r * math.cos(radial_angle_sc)
            center_y = sc_coord[1] + r * math.sin(radial_angle_sc)

            start_angle_rad = math.atan2(sc_coord[1] - center_y, sc_coord[0] - center_x)
            sweep_angle_rad = delta_c_rad if direction == CurveDirection.LEFT else -delta_c_rad
            end_angle_rad = start_angle_rad + sweep_angle_rad

            if lt > 0.0:
                cs_coord = (center_x + r * math.cos(end_angle_rad), center_y + r * math.sin(end_angle_rad))
            else:
                cs_coord = st_coord
            arc_len = r * delta_c_rad

            arc_points = self._discretize_arc(
                center=(center_x, center_y),
                radius=r,
                start_angle_rad=start_angle_rad,
                sweep_angle_rad=sweep_angle_rad,
                arc_length_m=arc_len,
            )
            arc_points[0] = sc_coord
            arc_points[-1] = cs_coord

            ch_prec = self.chainage_precision
            arc_geom = ArcGeometry(
                center_easting_m=round(center_x, 4),
                center_northing_m=round(center_y, 4),
                radius_m=round(r, 3),
                start_angle_rad=start_angle_rad,
                end_angle_rad=end_angle_rad,
                start_angle_deg=round(math.degrees(start_angle_rad) % 360.0, 3),
                end_angle_deg=round(math.degrees(end_angle_rad) % 360.0, 3),
                sweep_angle_deg=round(math.degrees(delta_c_rad), 3),
                direction=direction,
                start_point=sc_coord,
                end_point=cs_coord,
                length_m=round(arc_len, 3),
            )

            elements.append(
                AlignmentElement(
                    element_id=f"{curve_id}_CIRCULAR_ARC",
                    element_type=AlignmentSegmentType.CIRCULAR_ARC,
                    start_chainage_m=round(current_chainage, ch_prec),
                    end_chainage_m=round(current_chainage + arc_len, ch_prec),
                    length_m=round(arc_len, ch_prec),
                    points=arc_points,
                    arc_geom=arc_geom,
                    curve_id=curve_id,
                    metadata={"radius_m": r, "arc_length_m": round(arc_len, ch_prec)},
                )
            )
            elem_idx += 1
            current_chainage += arc_len
            ch_cs = current_chainage

            bearing_at_cs = bearing_at_sc + sweep_angle_rad
            self._record_tangent_point(
                tangent_points=tangent_points,
                critical_stations=critical_stations,
                curve_id=curve_id,
                curve_index=curve_index,
                station_code="CS",
                chainage_m=ch_cs,
                coord=cs_coord,
                direction=direction,
                tangent_unit=(math.cos(bearing_at_cs), math.sin(bearing_at_cs)),
                radius_m=r,
                transition_length_m=lt,
            )

            # 4. Exit Transition Spiral (CS -> ST): MULTIPLE SEGMENTS
            if lt > 0.0:
                n2x, n2y = (-u2y, u2x) if direction == CurveDirection.LEFT else (u2y, -u2x)
                exit_points, _ = self._generate_transition_points(
                    origin=st_coord,
                    tangent_unit=(u2x, u2y),
                    normal_unit=(n2x, n2y),
                    transition_len=lt,
                    radius=r,
                    is_entry=False,
                )
                exit_points[0] = cs_coord
                exit_points[-1] = st_coord

                elements.append(
                    AlignmentElement(
                        element_id=f"{curve_id}_TRANSITION_EXIT",
                        element_type=AlignmentSegmentType.TRANSITION_EXIT,
                        start_chainage_m=round(current_chainage, ch_prec),
                        end_chainage_m=round(current_chainage + lt, ch_prec),
                        length_m=round(lt, ch_prec),
                        points=exit_points,
                        curve_id=curve_id,
                        metadata={"segment_count": len(exit_points) - 1, "transition_length_m": lt},
                    )
                )
                elem_idx += 1
                current_chainage += lt
            else:
                st_coord = cs_coord

            ch_st = current_chainage
            self._record_tangent_point(
                tangent_points=tangent_points,
                critical_stations=critical_stations,
                curve_id=curve_id,
                curve_index=curve_index,
                station_code="ST",
                chainage_m=ch_st,
                coord=st_coord,
                direction=direction,
                tangent_unit=(u2x, u2y),
                radius_m=r,
                transition_length_m=lt,
            )
            solved_curves.append(
                replace(
                    solved_curve,
                    chainage_pi=round(ch_ts + t_len, ch_prec),
                    chainage_ts=round(ch_ts, ch_prec),
                    chainage_sc=round(ch_sc, ch_prec),
                    chainage_cs=round(ch_cs, ch_prec),
                    chainage_st=round(ch_st, ch_prec),
                    circular_length_m=round(arc_len, ch_prec),
                    total_curve_length_m=round((2.0 * lt) + arc_len, ch_prec),
                )
            )
            last_exit_point = st_coord

        # Final Straight Tangent to End Point (Strictly 2 Points)
        p_final = control_points[-1]
        last_dx = p_final[0] - control_points[-2][0]
        last_dy = p_final[1] - control_points[-2][1]
        last_leg_len = math.hypot(last_dx, last_dy)
        if last_leg_len > 1e-6:
            u_last_x, u_last_y = last_dx / last_leg_len, last_dy / last_leg_len
            signed_final = (p_final[0] - last_exit_point[0]) * u_last_x + (p_final[1] - last_exit_point[1]) * u_last_y
            if signed_final < -1e-3:
                raise ValueError(
                    f"Final curve exit ST overshoots corridor end point by {-signed_final:.2f}m. "
                    "Reduce curve radius/transition length or extend the final tangent leg."
                )
            final_tangent_len = max(0.0, signed_final)
        else:
            final_tangent_len = math.hypot(p_final[0] - last_exit_point[0], p_final[1] - last_exit_point[1])

        if final_tangent_len > 1e-4:
            elements.append(
                self._make_tangent_element(elem_idx, current_chainage, last_exit_point, p_final, final_tangent_len)
            )
            current_chainage += final_tangent_len

        ch_prec = self.chainage_precision
        critical_stations.append(
            {"name": "END", "chainage_m": round(current_chainage, ch_prec), "coord": p_final}
        )

        total_length = current_chainage - start_chainage_m
        summary = {
            "total_length_m": round(total_length, ch_prec),
            "elements_count": len(elements),
            "curves_count": len(solved_curves),
            "tangent_points_count": len(tangent_points),
            "tangents_count": sum(1 for e in elements if e.element_type == AlignmentSegmentType.TANGENT),
            "transitions_count": sum(
                1 for e in elements if e.element_type in (AlignmentSegmentType.TRANSITION_ENTRY, AlignmentSegmentType.TRANSITION_EXIT)
            ),
            "arcs_count": sum(1 for e in elements if e.element_type == AlignmentSegmentType.CIRCULAR_ARC),
        }

        logger.info(
            "Calculated alignment: TotalLen=%.2fm, %d tangents (2-pt), %d arcs, %d transitions, %d tangent points.",
            total_length,
            summary["tangents_count"],
            summary["arcs_count"],
            summary["transitions_count"],
            summary["tangent_points_count"],
        )

        return CalculatedAlignment(
            total_length_m=round(total_length, ch_prec),
            elements=elements,
            curves=solved_curves,
            critical_stations=critical_stations,
            summary=summary,
            tangent_points=tangent_points,
            start_chainage_m=round(start_chainage_m, ch_prec),
            epsg=epsg,
        )

    def to_centerline_coords(self, calculated: CalculatedAlignment) -> List[Tuple[float, float]]:
        """Extract deduplicated (Easting, Northing) vertices along solved elements in travel order."""
        return extract_centerline_coords(calculated)

    def sample_stations(
        self,
        calculated: CalculatedAlignment,
        interval_m: Optional[float] = None,
        epsg: Optional[int] = None,
    ) -> List[AlignmentStationPoint]:
        """Resample uniform chainage stations along solved alignment elements (tangents, spirals, arcs)."""
        step = float(interval_m if interval_m is not None else self.chainage_interval_m)
        return sample_alignment_stations(
            calculated=calculated,
            interval_m=step,
            chainage_precision=self.chainage_precision,
            epsg=epsg,
        )

    def _record_tangent_point(
        self,
        *,
        tangent_points: List[TangentPointStation],
        critical_stations: List[Dict[str, Any]],
        curve_id: str,
        curve_index: int,
        station_code: str,
        chainage_m: float,
        coord: Tuple[float, float],
        direction: CurveDirection,
        tangent_unit: Tuple[float, float],
        radius_m: float,
        transition_length_m: float,
    ) -> None:
        """Record an RDSO tangent point (TPTC-1, TPCC-1, TPCC-2, TPTC-2) with outward normal."""
        t_len = math.hypot(tangent_unit[0], tangent_unit[1]) or 1.0
        tx, ty = tangent_unit[0] / t_len, tangent_unit[1] / t_len

        # Outward normal points away from curve center of curvature
        if direction == CurveDirection.LEFT:
            nx, ny = ty, -tx
        else:
            nx, ny = -ty, tx

        ch_rounded = round(chainage_m, self.chainage_precision)
        tp_label = self.tp_labels.get(station_code, station_code)
        t_unit = (round(tx, 8), round(ty, 8))
        n_unit = (round(nx, 8), round(ny, 8))
        has_transition = transition_length_m > 0.0

        tp_station = TangentPointStation(
            curve_id=curve_id,
            curve_index=curve_index,
            station_code=station_code,
            tp_label=tp_label,
            chainage_m=ch_rounded,
            easting_m=round(coord[0], 4),
            northing_m=round(coord[1], 4),
            direction=direction,
            tangent_unit=t_unit,
            normal_unit=n_unit,
            radius_m=radius_m,
            transition_length_m=transition_length_m,
            has_transition=has_transition,
        )
        tangent_points.append(tp_station)
        critical_stations.append(
            {
                "name": f"{curve_id}_{station_code}",
                "tp_label": tp_label,
                "station_code": station_code,
                "curve_id": curve_id,
                "curve_index": curve_index,
                "chainage_m": ch_rounded,
                "coord": coord,
                "direction": direction.value,
                "tangent_unit": t_unit,
                "normal_unit": n_unit,
                "radius_m": radius_m,
                "transition_length_m": transition_length_m,
                "has_transition": has_transition,
            }
        )

    def _generate_transition_points(
        self,
        origin: Tuple[float, float],
        tangent_unit: Tuple[float, float],
        normal_unit: Tuple[float, float],
        transition_len: float,
        radius: float,
        is_entry: bool,
    ) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
        return generate_transition_points(
            origin=origin,
            tangent_unit=tangent_unit,
            normal_unit=normal_unit,
            transition_len=transition_len,
            radius=radius,
            step_m=self.transition_chord_step_m,
            is_entry=is_entry,
        )

    def _discretize_arc(
        self,
        center: Tuple[float, float],
        radius: float,
        start_angle_rad: float,
        sweep_angle_rad: float,
        arc_length_m: float,
    ) -> List[Tuple[float, float]]:
        return discretize_arc(
            center=center,
            radius=radius,
            start_angle_rad=start_angle_rad,
            sweep_angle_rad=sweep_angle_rad,
            arc_length_m=arc_length_m,
            step_m=self.transition_chord_step_m,
        )

    @staticmethod
    def _calculate_bearing(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate whole circle bearing in degrees (0 to 360) from North."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

