"""Identify candidate tangent intersections in a projected survey polyline.

The caller chooses a maximum simplification deviation and a minimum turn angle.
Candidates need engineering review: a sampled curved centerline does not uniquely
determine its design tangents or their intersection. Chainages are survey
chainages at retained samples, not design tangent-chainages.

For continuous circular arcs digitized as multiple chord segments, contiguous
bends turning in the same direction are merged into a single design PI apex
with incoming/outgoing straight tangent intersection and estimated radius.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml

try:
    from alignment.geospatial.kml_ingestor import AlignmentStationPoint
    from alignment.horizontal.curve_solver import CurveDirection
    from alignment.utils.logger import get_logger
except ImportError:
    try:
        from geospatial.kml_ingestor import AlignmentStationPoint
        from horizontal.curve_solver import CurveDirection
        from utils.logger import get_logger
    except ImportError:
        from future_alignment_studio.geospatial.kml_ingestor import AlignmentStationPoint
        from future_alignment_studio.horizontal.curve_solver import CurveDirection
        from future_alignment_studio.utils.logger import get_logger

logger = get_logger("PIDetector")
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config/alignment_config.yaml"


@dataclass(frozen=True)
class PointOfIntersection:
    """Candidate at a retained survey point or merged design PI apex; coordinates and chainage in metres."""

    pi_id: str
    station_index: int
    survey_chainage_m: float
    easting_m: float
    northing_m: float
    deflection_angle_deg: float
    direction: CurveDirection
    estimated_radius_m: float | None = None
    sub_bends_count: int = 1
    arc_start_chainage_m: float | None = None
    arc_end_chainage_m: float | None = None


@lru_cache(maxsize=4)
def _read_curve_standards_yaml(config_path: Path, mtime_ns: int) -> dict[str, Any]:
    """Read `curve_standards` from YAML; keyed by mtime so edits reload automatically."""
    del mtime_ns  # Cache key only
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return dict(loaded.get("curve_standards", {}))


def _load_curve_standards(
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Resolve curve-detection parameters from YAML and optional caller overrides."""
    merged: dict[str, Any] = {}
    resolved_path = (config_path or _DEFAULT_CONFIG_PATH).resolve()
    if resolved_path.exists():
        merged.update(_read_curve_standards_yaml(resolved_path, resolved_path.stat().st_mtime_ns))
    if config is not None:
        override = config.get("curve_standards", config)
        if isinstance(override, Mapping):
            merged.update(override)
    return merged


def _find_station_pos(chainages: Sequence[float], target_ch: float) -> int:
    """Locate the positional index in strictly increasing `chainages` nearest to `target_ch`."""
    pos = bisect_left(chainages, target_ch)
    if pos <= 0:
        return 0
    if pos >= len(chainages):
        return len(chainages) - 1
    if (chainages[pos] - target_ch) < (target_ch - chainages[pos - 1]):
        return pos
    return pos - 1


def _fit_circle_radius(
    points: Sequence[tuple[float, float]],
    *,
    min_radius_m: float,
    max_radius_m: float,
) -> float | None:
    """Fit an algebraic circle (Kasa method) to a set of 2D points using linear least-squares.

    Returns the fitted radius in metres, or None if points are collinear or insufficient (< 3).
    """
    if len(points) < 3:
        return None
    try:
        x = np.array([p[0] for p in points], dtype=np.float64)
        y = np.array([p[1] for p in points], dtype=np.float64)
        x_m = np.mean(x)
        y_m = np.mean(y)
        u = x - x_m
        v = y - y_m
        a_mat = np.column_stack([2.0 * u, 2.0 * v, np.ones_like(u)])
        b_vec = u * u + v * v
        sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        uc, vc, c_val = sol
        r_sq = c_val + uc * uc + vc * vc
        if r_sq > 0:
            radius = float(math.sqrt(r_sq))
            if min_radius_m <= radius <= max_radius_m:
                return radius
    except Exception as exc:
        logger.debug("Circle fit failed: %s", exc)
    return None


def _angular_diff_rad(bearing_a: float, bearing_b: float) -> float:
    """Smallest positive angle between two bearings in radians."""
    diff = abs(bearing_a - bearing_b)
    return (2.0 * math.pi - diff) if diff > math.pi else diff


def _find_tangent_vector(
    stations: Sequence[AlignmentStationPoint],
    bearings: Sequence[float],
    start_idx: int,
    direction: str,
    *,
    min_segments: int,
    angle_tol_deg: float,
    max_steps: int,
    fallback_window_steps: int,
) -> tuple[float, float, float, float]:
    """Walk backward or forward from a curve station along the surveyed centerline to anchor
    a stable straight tangent line. Returns (dx, dy, anchor_x, anchor_y).
    """
    tol_rad = math.radians(angle_tol_deg)
    min_steps = max(1, min_segments)
    fb_win = max(1, fallback_window_steps)
    n = len(bearings)

    if direction == "backward":
        count = 0
        upper = min(n - 1, start_idx)
        lower = max(0, upper - max_steps)
        for i in range(upper, lower, -1):
            if _angular_diff_rad(bearings[i], bearings[i - 1]) < tol_rad:
                count += 1
                if count >= min_steps:
                    s_a = stations[i - 1]
                    s_b = stations[min(len(stations) - 1, i + min_steps)]
                    return (
                        s_b.easting_m - s_a.easting_m,
                        s_b.northing_m - s_a.northing_m,
                        s_b.easting_m,
                        s_b.northing_m,
                    )
            else:
                count = 0
        fb_end = max(1, min(len(stations) - 1, start_idx))
        fb_start = max(0, fb_end - fb_win)
        s_a, s_b = stations[fb_start], stations[fb_end]
        return (
            s_b.easting_m - s_a.easting_m,
            s_b.northing_m - s_a.northing_m,
            s_b.easting_m,
            s_b.northing_m,
        )

    count = 0
    lo = max(1, start_idx)
    hi = min(n, lo + max_steps)
    for i in range(lo, hi):
        if _angular_diff_rad(bearings[i], bearings[i - 1]) < tol_rad:
            count += 1
            if count >= min_steps:
                s_c = stations[max(0, i - min_steps)]
                s_d = stations[min(len(stations) - 1, i + 1)]
                return (
                    s_d.easting_m - s_c.easting_m,
                    s_d.northing_m - s_c.northing_m,
                    s_c.easting_m,
                    s_c.northing_m,
                )
        else:
            count = 0
    fb_start = max(0, min(len(stations) - 2, start_idx))
    fb_end = min(len(stations) - 1, fb_start + fb_win)
    s_c, s_d = stations[fb_start], stations[fb_end]
    return (
        s_d.easting_m - s_c.easting_m,
        s_d.northing_m - s_c.northing_m,
        s_c.easting_m,
        s_c.northing_m,
    )


def _simplify_rdp_indices(
    stations: Sequence[AlignmentStationPoint],
    tolerance_m: float,
) -> list[int]:
    """Iterative Ramer-Douglas-Peucker simplification returning retained station indices."""
    retained = {0, len(stations) - 1}
    pending = [(0, len(stations) - 1)]
    while pending:
        start, end = pending.pop()
        a, b = stations[start], stations[end]
        dx, dy = b.easting_m - a.easting_m, b.northing_m - a.northing_m
        length_squared = dx * dx + dy * dy
        max_distance, farthest = -1.0, None
        for i in range(start + 1, end):
            point = stations[i]
            px, py = point.easting_m - a.easting_m, point.northing_m - a.northing_m
            fraction = max(0.0, min(1.0, (px * dx + py * dy) / length_squared)) if length_squared else 0.0
            distance = math.hypot(px - fraction * dx, py - fraction * dy)
            if distance > max_distance:
                max_distance, farthest = distance, i
        if farthest is not None and max_distance > tolerance_m:
            retained.add(farthest)
            pending.extend(((start, farthest), (farthest, end)))
    return sorted(retained)


def _intersect_tangents(
    dx1: float,
    dy1: float,
    x1: float,
    y1: float,
    dx2: float,
    dy2: float,
    x2: float,
    y2: float,
) -> Optional[tuple[float, float, float, CurveDirection, tuple[float, float]]]:
    """Intersect incoming/outgoing tangent rays; return (pi_x, pi_y, defl_deg, direction, u1)."""
    len1 = math.hypot(dx1, dy1)
    len2 = math.hypot(dx2, dy2)
    if len1 <= 1e-9 or len2 <= 1e-9:
        return None
    u1x, u1y = dx1 / len1, dy1 / len1
    u2x, u2y = dx2 / len2, dy2 / len2
    denom = u2x * u1y - u1x * u2y
    if abs(denom) <= 1e-6:
        return None
    t1 = ((y2 - y1) * u2x - (x2 - x1) * u2y) / denom
    pi_x = x1 + t1 * u1x
    pi_y = y1 + t1 * u1y
    cross = u1x * u2y - u1y * u2x
    dot = u1x * u2x + u1y * u2y
    angle_deg = math.degrees(math.atan2(cross, dot))
    direction = CurveDirection.LEFT if angle_deg > 0 else CurveDirection.RIGHT
    return (pi_x, pi_y, abs(angle_deg), direction, (u1x, u1y))


def detect_pis(
    stations: Sequence[AlignmentStationPoint],
    *,
    simplification_tolerance_m: float,
    minimum_deflection_deg: float,
    merge_continuous_curves: Optional[bool] = None,
    max_group_gap_m: Optional[float] = None,
    straight_tangent_min_segments: Optional[int] = None,
    straight_tangent_angle_tol_deg: Optional[float] = None,
    radius_round_interval_m: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> list[PointOfIntersection]:
    """Return candidate bends in travel order using polyline simplification.

    Tolerance is the maximum point-to-segment deviation in ground metres for
    each simplification step. A zero tolerance preserves all non-collinear
    vertices. The deflection threshold is inclusive, in degrees, and must be
    positive. Neither setting is a railway design standard.

    When merge_continuous_curves is True, consecutive chord vertices digitized
    along a circular arc in the same deflection direction are merged into a single
    true design PI apex point with accurate incoming and outgoing straight
    tangent intersection and estimated surveyed radius.
    """
    if not math.isfinite(simplification_tolerance_m) or simplification_tolerance_m < 0:
        raise ValueError("simplification_tolerance_m must be finite and non-negative")
    if not math.isfinite(minimum_deflection_deg) or not 0 < minimum_deflection_deg < 180:
        raise ValueError("minimum_deflection_deg must be finite and between 0 and 180")
    if len(stations) < 2:
        raise ValueError("At least two survey stations are required")
    for i, station in enumerate(stations):
        if not all(math.isfinite(v) for v in (station.chainage_m, station.easting_m, station.northing_m)):
            raise ValueError("Station chainage and projected coordinates must be finite")
        if i:
            previous = stations[i - 1]
            if station.chainage_m <= previous.chainage_m:
                raise ValueError("Survey chainages must be strictly increasing")
            if (station.easting_m, station.northing_m) == (previous.easting_m, previous.northing_m):
                raise ValueError("Consecutive survey coordinates must be distinct")

    cfg = _load_curve_standards(config=config, config_path=config_path)
    use_merge = bool(cfg["merge_continuous_curves"] if merge_continuous_curves is None else merge_continuous_curves)
    gap_limit_m = float(cfg["max_curve_cluster_gap_m"] if max_group_gap_m is None else max_group_gap_m)
    tan_min_segs = int(
        cfg["straight_tangent_min_segments"]
        if straight_tangent_min_segments is None
        else straight_tangent_min_segments
    )
    tan_angle_tol_deg = float(
        cfg["straight_tangent_angle_tol_deg"]
        if straight_tangent_angle_tol_deg is None
        else straight_tangent_angle_tol_deg
    )
    r_round_m = float(cfg["radius_round_interval_m"] if radius_round_interval_m is None else radius_round_interval_m)
    min_fit_r_m = float(cfg["min_fitted_radius_m"])
    max_fit_r_m = float(cfg["max_fitted_radius_m"])
    single_window_m = float(cfg["single_bend_circle_window_m"])
    max_search_steps = int(cfg["tangent_search_max_steps"])
    fb_window_steps = int(cfg["tangent_fallback_window_steps"])

    indices = _simplify_rdp_indices(stations, simplification_tolerance_m)
    raw_candidates: list[PointOfIntersection] = []
    for left, middle, right in zip(indices, indices[1:], indices[2:]):
        a, b, c = stations[left], stations[middle], stations[right]
        ux, uy = b.easting_m - a.easting_m, b.northing_m - a.northing_m
        vx, vy = c.easting_m - b.easting_m, c.northing_m - b.northing_m
        if (ux == 0 and uy == 0) or (vx == 0 and vy == 0):
            raise ValueError("Simplification produced a zero-length tangent; lower the tolerance")
        cross, dot = ux * vy - uy * vx, ux * vx + uy * vy
        angle = math.degrees(math.atan2(cross, dot))
        if abs(angle) >= 180.0:
            raise ValueError("A 180-degree reversal cannot define a railway curve PI")
        if abs(angle) >= minimum_deflection_deg:
            raw_candidates.append(PointOfIntersection(
                pi_id=f"PI-{len(raw_candidates) + 1}",
                station_index=b.station_index,
                survey_chainage_m=b.chainage_m,
                easting_m=b.easting_m,
                northing_m=b.northing_m,
                deflection_angle_deg=abs(angle),
                direction=CurveDirection.LEFT if angle > 0 else CurveDirection.RIGHT,
            ))

    if not use_merge or len(raw_candidates) <= 1:
        return raw_candidates

    station_chainages = [s.chainage_m for s in stations]
    bearings: list[float] = [
        math.atan2(s2.easting_m - s1.easting_m, s2.northing_m - s1.northing_m)
        for s1, s2 in zip(stations, stations[1:])
    ]

    # Cluster consecutive candidates that turn in the same direction within gap_limit_m
    groups: list[list[PointOfIntersection]] = []
    current_group = [raw_candidates[0]]
    for cand in raw_candidates[1:]:
        prev_cand = current_group[-1]
        same_direction = (cand.direction == prev_cand.direction)
        chainage_gap = cand.survey_chainage_m - prev_cand.survey_chainage_m
        if same_direction and chainage_gap <= gap_limit_m:
            current_group.append(cand)
        else:
            groups.append(current_group)
            current_group = [cand]
    if current_group:
        groups.append(current_group)

    merged_pis: list[PointOfIntersection] = []
    for idx, grp in enumerate(groups):
        ch_start = grp[0].survey_chainage_m
        ch_end = grp[-1].survey_chainage_m

        # Estimate circle radius from stations spanning the arc (sliced via bisect for O(log N))
        lo_ch = ch_start - single_window_m if len(grp) == 1 else ch_start
        hi_ch = ch_end + single_window_m if len(grp) == 1 else ch_end
        arc_lo = bisect_left(station_chainages, lo_ch)
        arc_hi = bisect_right(station_chainages, hi_ch + 1e-9)
        arc_points = [(s.easting_m, s.northing_m) for s in stations[arc_lo:arc_hi]]

        r_fitted = _fit_circle_radius(
            arc_points,
            min_radius_m=min_fit_r_m,
            max_radius_m=max_fit_r_m,
        )
        if r_fitted is not None and r_round_m > 0:
            r_estimated = round(r_fitted / r_round_m) * r_round_m
        else:
            r_estimated = r_fitted

        if len(grp) == 1:
            single_cand = grp[0]
            merged_pis.append(PointOfIntersection(
                pi_id=f"PI-{idx + 1}",
                station_index=single_cand.station_index,
                survey_chainage_m=single_cand.survey_chainage_m,
                easting_m=single_cand.easting_m,
                northing_m=single_cand.northing_m,
                deflection_angle_deg=single_cand.deflection_angle_deg,
                direction=single_cand.direction,
                estimated_radius_m=r_estimated,
                sub_bends_count=1,
                arc_start_chainage_m=ch_start,
                arc_end_chainage_m=ch_end,
            ))
            continue

        idx_start = _find_station_pos(station_chainages, ch_start)
        idx_end = _find_station_pos(station_chainages, ch_end)

        dx1, dy1, x1, y1 = _find_tangent_vector(
            stations,
            bearings,
            idx_start,
            "backward",
            min_segments=tan_min_segs,
            angle_tol_deg=tan_angle_tol_deg,
            max_steps=max_search_steps,
            fallback_window_steps=fb_window_steps,
        )
        dx2, dy2, x2, y2 = _find_tangent_vector(
            stations,
            bearings,
            idx_end,
            "forward",
            min_segments=tan_min_segs,
            angle_tol_deg=tan_angle_tol_deg,
            max_steps=max_search_steps,
            fallback_window_steps=fb_window_steps,
        )

        intersected = _intersect_tangents(dx1, dy1, x1, y1, dx2, dy2, x2, y2)
        s_entry = stations[idx_start]
        if intersected is not None:
            pi_x, pi_y, defl_deg, direction, (u1x, u1y) = intersected
            proj_entry = (pi_x - s_entry.easting_m) * u1x + (pi_y - s_entry.northing_m) * u1y
            d_entry = proj_entry if proj_entry > 0.0 else math.hypot(pi_x - s_entry.easting_m, pi_y - s_entry.northing_m)
        else:
            mid_pt = grp[len(grp) // 2]
            pi_x, pi_y = mid_pt.easting_m, mid_pt.northing_m
            defl_deg = sum(c.deflection_angle_deg for c in grp)
            direction = grp[0].direction
            d_entry = math.hypot(pi_x - s_entry.easting_m, pi_y - s_entry.northing_m)

        pi_chainage = s_entry.chainage_m + d_entry

        merged_pis.append(PointOfIntersection(
            pi_id=f"PI-{idx + 1}",
            station_index=grp[len(grp) // 2].station_index,
            survey_chainage_m=round(pi_chainage, 3),
            easting_m=round(pi_x, 3),
            northing_m=round(pi_y, 3),
            deflection_angle_deg=round(defl_deg, 3),
            direction=direction,
            estimated_radius_m=r_estimated,
            sub_bends_count=len(grp),
            arc_start_chainage_m=ch_start,
            arc_end_chainage_m=ch_end,
        ))

    logger.info(
        "PI Detection complete: clustered %d raw bend vertices into %d design PIs.",
        len(raw_candidates),
        len(merged_pis),
    )
    return merged_pis
