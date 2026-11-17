"""
Module: future_alignment_studio/horizontal/curve_solver.py
Purpose: Pure Railway Horizontal Curve Geometric Engine.

Strictly RDSO / IRPWM Compliant:
- User inputs Degree of Curve (D) OR Radius (R).
- Automatic Cant, Cant Deficiency, and Transition Spiral Length (Lt).
- Curve Shift (S), Tangent Length (T), and Critical Tangent Stations (TS, SC, CS, ST).
- RDSO Cubic Parabola transition geometry: y = x^3 / (6 * R * Lt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import yaml

try:
    from alignment.utils.logger import get_logger
except ImportError:
    try:
        from utils.logger import get_logger
    except ImportError:
        from future_alignment_studio.utils.logger import get_logger

logger = get_logger("CurveSolver")
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config/alignment_config.yaml"


class CurveDirection(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STRAIGHT = "STRAIGHT"


@dataclass
class RDSOHorizontalCurve:
    """Holds complete geometric solution for a railway horizontal curve."""
    curve_id: str
    direction: CurveDirection
    radius_m: float
    degree_of_curve: float
    deflection_angle_deg: float      # Total turn angle Δ
    cant_actual_mm: float            # Ca
    cant_deficiency_mm: float        # Cd
    transition_length_m: float       # Lt
    shift_m: float                   # S = Lt^2 / (24*R)
    tangent_length_m: float         # T = (R+S)*tan(Δ/2) + Lt/2
    circular_length_m: float        # Lc
    total_curve_length_m: float     # 2*Lt + Lc
    chainage_pi: float              # Chainage at Point of Intersection
    chainage_ts: float              # Tangent to Spiral (Start of Transition 1)
    chainage_sc: float              # Spiral to Curve (Start of Circular Arc)
    chainage_cs: float              # Curve to Spiral (End of Circular Arc)
    chainage_st: float              # Spiral to Tangent (End of Transition 2)
    spiral_angle_deg: float         # θs = Lt / (2*R) in degrees
    apex_distance_m: float          # Distance from PI to curve apex
    max_permissible_speed_kmh: Optional[float] = None  # Vmax = 0.27 * sqrt(R * (Ca + Cd)) per IR Sheet
    cant_excess_mm: Optional[float] = None             # Ce for goods trains (65 km/h) per IR Sheet
    warnings: List[str] = field(default_factory=list)


@lru_cache(maxsize=4)
def _read_solver_yaml(config_path: Path, mtime_ns: int) -> dict[str, Any]:
    """Read solver sections from YAML; keyed by mtime so edits reload automatically."""
    del mtime_ns
    with config_path.open(encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


class HorizontalCurveSolver:
    """
    Solves railway horizontal curves given user-specified Radius or Degree.
    """

    def __init__(
        self,
        dynamic_gauge_mm: Optional[float] = None,
        default_speed_kmh: Optional[float] = None,
        max_cant_mm: Optional[float] = None,
        max_cant_deficiency_mm: Optional[float] = None,
        min_transition_deflection_deg: Optional[float] = None,
        max_transition_spiral_ratio: Optional[float] = None,
        max_cant_excess_mm: Optional[float] = None,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        raw_cfg: dict[str, Any] = {}
        resolved = (config_path or _DEFAULT_CONFIG_PATH).resolve()
        if resolved.exists():
            raw_cfg = _read_solver_yaml(resolved, resolved.stat().st_mtime_ns)

        railway_std: dict[str, Any] = dict(raw_cfg.get("railway_standards", {}))
        curve_std: dict[str, Any] = dict(raw_cfg.get("curve_standards", {}))
        tp_std: dict[str, Any] = dict(raw_cfg.get("tangent_point_standards", {}))
        if config is not None:
            if isinstance(config.get("railway_standards"), Mapping):
                railway_std.update(config["railway_standards"])
            if isinstance(config.get("curve_standards"), Mapping):
                curve_std.update(config["curve_standards"])
            if isinstance(config.get("tangent_point_standards"), Mapping):
                tp_std.update(config["tangent_point_standards"])

        self.dynamic_gauge_mm = float(
            railway_std["dynamic_gauge_mm"] if dynamic_gauge_mm is None else dynamic_gauge_mm
        )
        self.default_speed_kmh = float(
            railway_std["default_design_speed_kmh"] if default_speed_kmh is None else default_speed_kmh
        )
        self.max_cant_mm = float(curve_std["max_cant_actual_mm"] if max_cant_mm is None else max_cant_mm)
        self.max_cant_deficiency_mm = float(
            curve_std["max_cant_deficiency_mm"] if max_cant_deficiency_mm is None else max_cant_deficiency_mm
        )
        self.max_cant_excess_mm = float(
            curve_std["max_cant_excess_mm"] if max_cant_excess_mm is None else max_cant_excess_mm
        )
        self.min_transition_deflection_deg = float(
            curve_std["min_transition_deflection_deg"]
            if min_transition_deflection_deg is None
            else min_transition_deflection_deg
        )
        self.max_transition_spiral_ratio = float(
            curve_std["max_transition_spiral_ratio"]
            if max_transition_spiral_ratio is None
            else max_transition_spiral_ratio
        )

        self.cant_formula_divisor = float(curve_std["cant_formula_divisor"])
        self.cant_round_step_mm = float(curve_std["cant_round_step_mm"])
        self.high_speed_threshold_kmh = float(curve_std["high_speed_threshold_kmh"])
        self.cant_rate_coeff_high_speed = float(curve_std["cant_rate_coeff_high_speed"])
        self.cant_rate_coeff_normal_speed = float(curve_std["cant_rate_coeff_normal_speed"])
        self.cant_gradient_transition_factor = float(curve_std["cant_gradient_transition_factor"])
        self.min_transition_length_m = float(curve_std["min_transition_length_m"])
        self.transition_round_step_m = float(curve_std["transition_round_step_m"])
        self.min_bounded_transition_length_m = float(curve_std["min_bounded_transition_length_m"])
        self.vmax_speed_coeff = float(curve_std["vmax_speed_coeff"])
        self.goods_train_speed_kmh = float(curve_std["goods_train_speed_kmh"])
        self.chainage_precision = int(tp_std["chainage_precision"])

    def radius_to_degree(self, radius_m: float) -> float:
        """RDSO standard: D = G / R."""
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("Radius must be finite and greater than 0.")
        return round(self.dynamic_gauge_mm / radius_m, 4)

    def degree_to_radius(self, degree_deg: float) -> float:
        """RDSO standard: R = G / D."""
        if not math.isfinite(degree_deg) or degree_deg <= 0.0:
            raise ValueError("Degree of curve must be finite and greater than 0.")
        return round(self.dynamic_gauge_mm / degree_deg, 3)

    def _resolve_radius_and_degree(
        self,
        radius_m: Optional[float],
        degree_of_curve: Optional[float],
    ) -> Tuple[float, float]:
        """Resolve exact radius (m) and degree of curve from caller input."""
        if radius_m is None and degree_of_curve is None:
            raise ValueError("Either radius_m or degree_of_curve must be specified.")
        if radius_m is not None and degree_of_curve is None:
            r = float(radius_m)
            return r, self.radius_to_degree(r)
        if degree_of_curve is not None and radius_m is None:
            d = float(degree_of_curve)
            return self.degree_to_radius(d), d
        r = float(radius_m)  # type: ignore[arg-type]
        return r, self.radius_to_degree(r)

    def _compute_cant_and_speed(
        self,
        radius_m: float,
        speed_kmh: float,
    ) -> Tuple[float, float, float, float]:
        """Compute rounded cant (Ca), deficiency (Cd), goods cant excess (Ce), and Vmax.

        Under RDSO IRPWM Para 406 & Indian Railways Curve Calculation Sheet (Sheet C-10):
        1. Actual Cant (Ca) is designed for Equilibrium Speed (V_equim = 120 km/h for Group A lines,
           or min(speed_kmh, 120.0)), rather than full max speed, to prevent excessive Cant Excess (Ce)
           and inner-rail crushing on slow freight/goods traffic (65 km/h).
        2. Ca must satisfy:
           - Ca >= C_eq(Vmax) - max_cant_deficiency_mm (to ensure Cd <= 75mm)
           - Ca is clamped to max_cant_mm (165mm).
        """
        c_eq = (self.dynamic_gauge_mm * (speed_kmh ** 2)) / (self.cant_formula_divisor * radius_m)
        c_goods = (self.dynamic_gauge_mm * (self.goods_train_speed_kmh ** 2)) / (
            self.cant_formula_divisor * radius_m
        )

        # Equilibrium speed: default 120 km/h for high speed / Group A (per Excel Sheet Row 11/25), or speed_kmh
        v_equim = min(speed_kmh, 120.0)
        c_equim = (self.dynamic_gauge_mm * (v_equim ** 2)) / (self.cant_formula_divisor * radius_m)

        # Provide cant for equilibrium speed, but ensure at least (C_eq - 75mm) so Cd <= 75mm
        ca_min_req = max(0.0, c_eq - self.max_cant_deficiency_mm)
        ca_target = max(ca_min_req, c_equim)

        step_mm = self.cant_round_step_mm
        ca_rounded = round(ca_target / step_mm) * step_mm if step_mm > 0.0 else ca_target
        ca_rounded = min(ca_rounded, self.max_cant_mm)

        cd = max(0.0, c_eq - ca_rounded)
        effective_cd = min(cd, self.max_cant_deficiency_mm)
        v_max = round(self.vmax_speed_coeff * math.sqrt(radius_m * (ca_rounded + effective_cd)), 1)
        ce = max(0.0, ca_rounded - c_goods)
        return ca_rounded, cd, ce, v_max

    def _compute_transition_length(
        self,
        curve_id: str,
        radius_m: float,
        speed_kmh: float,
        ca_mm: float,
        cd_mm: float,
        delta_deg: float,
        delta_rad: float,
        bound_transition: bool,
    ) -> float:
        """Compute RDSO transition spiral length Lt with optional deflection bounding."""
        rate_coeff = (
            self.cant_rate_coeff_high_speed
            if speed_kmh >= self.high_speed_threshold_kmh
            else self.cant_rate_coeff_normal_speed
        )
        lt_1 = rate_coeff * cd_mm * speed_kmh
        lt_2 = rate_coeff * ca_mm * speed_kmh
        lt_3 = self.cant_gradient_transition_factor * ca_mm
        lt_raw = max(lt_1, lt_2, lt_3)
        step_m = self.transition_round_step_m
        lt_rounded = math.ceil(lt_raw / step_m) * step_m if step_m > 0.0 else lt_raw
        lt = max(self.min_transition_length_m, lt_rounded)

        if not bound_transition:
            return lt

        max_lt = self.max_transition_spiral_ratio * radius_m * delta_rad
        if lt <= max_lt:
            return lt

        if delta_deg < self.min_transition_deflection_deg or max_lt < self.min_bounded_transition_length_m:
            logger.info(
                "Curve %s: Deflection %.2f° too small for spiral (max Lt=%.2fm); using simple circular curve.",
                curve_id, delta_deg, max_lt,
            )
            return 0.0

        adjusted_lt = max(self.min_bounded_transition_length_m, round(max_lt, 2))
        logger.info(
            "Curve %s: Theoretical Lt=%.1fm bounded by deflection angle (%.2f°) to Lt=%.2fm.",
            curve_id, lt, delta_deg, adjusted_lt,
        )
        return adjusted_lt

    def solve_curve(
        self,
        curve_id: str,
        pi_chainage: float,
        deflection_angle_deg: float,
        direction: CurveDirection,
        radius_m: Optional[float] = None,
        degree_of_curve: Optional[float] = None,
        speed_kmh: Optional[float] = None,
        bound_transition: bool = False,
    ) -> RDSOHorizontalCurve:
        """
        Solves complete horizontal curve geometry.
        User must provide EITHER radius_m OR degree_of_curve.
        """
        if not math.isfinite(pi_chainage):
            raise ValueError(f"pi_chainage must be finite; got {pi_chainage!r}.")
        if not math.isfinite(deflection_angle_deg):
            raise ValueError(f"deflection_angle_deg must be finite; got {deflection_angle_deg!r}.")

        r, d = self._resolve_radius_and_degree(radius_m, degree_of_curve)
        v = float(speed_kmh if speed_kmh is not None else self.default_speed_kmh)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"speed_kmh must be finite and greater than 0; got {v!r}.")

        delta_deg = abs(deflection_angle_deg)
        if delta_deg == 0.0 or delta_deg >= 180.0:
            raise ValueError("Deflection angle must be strictly between 0 and 180 degrees for a curve.")
        delta_rad = math.radians(delta_deg)

        ca_rounded, cd, ce, v_max = self._compute_cant_and_speed(r, v)
        effective_cd = min(cd, self.max_cant_deficiency_mm)
        operative_speed = min(v, v_max)

        warnings: List[str] = []
        if cd > self.max_cant_deficiency_mm:
            logger.warning(
                "Curve %s: Cant deficiency %.1fmm exceeds max %.1fmm at %.1f km/h (speed restricted to Vmax=%.1f km/h)",
                curve_id, cd, self.max_cant_deficiency_mm, v, v_max,
            )
            warnings.append(f"Cant deficiency ({cd:.1f}mm) exceeds limit ({self.max_cant_deficiency_mm:.1f}mm); speed restricted to Vmax={v_max:.0f} km/h.")
        if ce > self.max_cant_excess_mm:
            warnings.append(f"Cant excess ({ce:.1f}mm) exceeds limit ({self.max_cant_excess_mm:.1f}mm).")

        # In IR Curve calculation sheet (Cell O45), transition length uses effective Cd (<= 75mm) and operative speed
        lt = self._compute_transition_length(
            curve_id=curve_id,
            radius_m=r,
            speed_kmh=operative_speed,
            ca_mm=ca_rounded,
            cd_mm=effective_cd,
            delta_deg=delta_deg,
            delta_rad=delta_rad,
            bound_transition=bound_transition,
        )

        # Shift S = Lt^2 / (24*R), Spiral Angle theta_s = Lt / (2*R), Tangent T = (R+S)*tan(delta/2) + Lt/2
        shift = (lt ** 2) / (24.0 * r) if lt > 0.0 else 0.0
        theta_s_rad = (lt / (2.0 * r)) if lt > 0.0 else 0.0
        theta_s_deg = math.degrees(theta_s_rad)
        tangent_len = (r + shift) * math.tan(delta_rad / 2.0) + (lt / 2.0)

        delta_c_rad = max(0.0, delta_rad - (2.0 * theta_s_rad))
        if delta_c_rad <= 0.0:
            logger.warning(
                "Curve %s: Deflection angle (%.2f°) too small for transition length (%.1fm).",
                curve_id, delta_deg, lt,
            )
            warnings.append("Transitions consume the circular angle; curve is geometrically infeasible.")
            delta_c_rad = 0.0

        circular_len = r * delta_c_rad
        total_curve_len = (2.0 * lt) + circular_len
        apex_dist = ((r + shift) / math.cos(delta_rad / 2.0)) - r

        ch_ts = pi_chainage - tangent_len
        ch_sc = ch_ts + lt
        ch_cs = ch_sc + circular_len
        ch_st = ch_cs + lt
        if ch_ts < 0.0:
            warnings.append("Transition starts before chainage zero.")

        ch_prec = self.chainage_precision
        curve = RDSOHorizontalCurve(
            curve_id=curve_id,
            direction=direction,
            radius_m=round(r, 2),
            degree_of_curve=round(d, 3),
            deflection_angle_deg=round(delta_deg, 4),
            cant_actual_mm=ca_rounded,
            cant_deficiency_mm=round(cd, 2),
            transition_length_m=round(lt, 2),
            shift_m=round(shift, 4),
            tangent_length_m=round(tangent_len, 3),
            circular_length_m=round(circular_len, 3),
            total_curve_length_m=round(total_curve_len, 3),
            chainage_pi=round(pi_chainage, ch_prec),
            chainage_ts=round(ch_ts, ch_prec),
            chainage_sc=round(ch_sc, ch_prec),
            chainage_cs=round(ch_cs, ch_prec),
            chainage_st=round(ch_st, ch_prec),
            spiral_angle_deg=round(theta_s_deg, 4),
            apex_distance_m=round(apex_dist, 3),
            max_permissible_speed_kmh=v_max,
            cant_excess_mm=round(ce, 2),
            warnings=warnings,
        )

        logger.info(
            "Solved Curve %s: R=%.1fm, D=%.2f°, Lt=%.1fm, Shift=%.3fm, Tangent=%.2fm, TotalLen=%.2fm",
            curve_id, r, d, lt, shift, tangent_len, total_curve_len,
        )
        return curve

    @staticmethod
    def calculate_cubic_parabola_y(x: float, radius_m: float, transition_len_m: float) -> float:
        """
        RDSO Cubic Parabola formula: y = x^3 / (6 * R * Lt).
        x = distance along transition from TS (0 <= x <= Lt).
        y = perpendicular offset from tangent line.
        """
        if x <= 0.0 or radius_m <= 0.0 or transition_len_m <= 0.0:
            return 0.0
        return (x ** 3) / (6.0 * radius_m * transition_len_m)

