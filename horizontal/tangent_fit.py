"""Fit two survey tangents and intersect them in projected ground metres.

Orthogonal least squares is a geometric estimate, not a railway design rule.
The engineer must select straight survey ranges and review the fit. This does
not establish design chainage or check curve feasibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

try:
    from alignment.horizontal.curve_solver import CurveDirection
except ImportError:
    try:
        from horizontal.curve_solver import CurveDirection
    except ImportError:
        from future_alignment_studio.horizontal.curve_solver import CurveDirection


@dataclass(frozen=True)
class SurveyPoint:
    chainage_m: float
    easting_m: float
    northing_m: float


@dataclass(frozen=True)
class FittedPI:
    easting_m: float
    northing_m: float
    deflection_angle_deg: float
    direction: CurveDirection
    incoming_rms_m: float
    outgoing_rms_m: float
    incoming_extrapolation_m: float
    outgoing_extrapolation_m: float
    incoming_count: int
    outgoing_count: int
    incoming_start_easting_m: float
    incoming_start_northing_m: float
    outgoing_end_easting_m: float
    outgoing_end_northing_m: float
    pi_before_incoming_end: bool
    pi_after_outgoing_start: bool


def _fit(points: Sequence[SurveyPoint]) -> tuple[float, float, float, float, float]:
    if len(points) < 2:
        raise ValueError('Each tangent range must contain at least two survey stations')
    x = sum(p.easting_m for p in points) / len(points)
    y = sum(p.northing_m for p in points) / len(points)
    xx = sum((p.easting_m - x) ** 2 for p in points)
    yy = sum((p.northing_m - y) ** 2 for p in points)
    xy = sum((p.easting_m - x) * (p.northing_m - y) for p in points)
    if xx + yy == 0 or math.hypot(xx - yy, 2 * xy) <= (xx + yy) * 1e-12:
        raise ValueError('Tangent points do not define a stable line')
    theta = math.atan2(2 * xy, xx - yy) / 2
    ux, uy = math.cos(theta), math.sin(theta)
    dx = points[-1].easting_m - points[0].easting_m
    dy = points[-1].northing_m - points[0].northing_m
    along = ux * dx + uy * dy
    if abs(along) <= math.hypot(dx, dy) * 1e-12:
        raise ValueError('Tangent direction cannot be determined from survey order')
    if along < 0:
        ux, uy = -ux, -uy
    rms = math.sqrt(sum(((p.easting_m - x) * uy - (p.northing_m - y) * ux) ** 2 for p in points) / len(points))
    return x, y, ux, uy, rms


def fit_pi(stations: Sequence[SurveyPoint], incoming_start: float, incoming_end: float, outgoing_start: float, outgoing_end: float) -> FittedPI:
    """Fit disjoint, increasing inclusive chainage ranges and intersect lines."""
    bounds = (incoming_start, incoming_end, outgoing_start, outgoing_end)
    if not all(math.isfinite(v) for v in bounds) or not incoming_start < incoming_end < outgoing_start < outgoing_end:
        raise ValueError('Tangent ranges must be non-overlapping and in travel order')
    if len(stations) < 4:
        raise ValueError('At least four survey stations are required')
    for index, p in enumerate(stations):
        if not all(math.isfinite(v) for v in (p.chainage_m, p.easting_m, p.northing_m)):
            raise ValueError('Survey stations must contain finite coordinates and chainages')
        if index and p.chainage_m <= stations[index - 1].chainage_m:
            raise ValueError('Survey chainages must be strictly increasing')
    incoming = [p for p in stations if incoming_start <= p.chainage_m <= incoming_end]
    outgoing = [p for p in stations if outgoing_start <= p.chainage_m <= outgoing_end]
    ax, ay, ux, uy, rms_a = _fit(incoming)
    bx, by, vx, vy, rms_b = _fit(outgoing)
    cross = ux * vy - uy * vx
    if abs(cross) <= 1e-8:
        raise ValueError('Tangents are parallel or too nearly parallel to find a stable PI')
    distance = ((bx - ax) * vy - (by - ay) * vx) / cross
    x, y = ax + distance * ux, ay + distance * uy
    angle = math.degrees(math.atan2(cross, ux * vx + uy * vy))
    # Distances along each fitted tangent from its sample interval to the PI.
    def extrapolation(points: Sequence[SurveyPoint], origin_x: float, origin_y: float, tx: float, ty: float) -> float:
        projections = [(p.easting_m - origin_x) * tx + (p.northing_m - origin_y) * ty for p in points]
        pi_projection = (x - origin_x) * tx + (y - origin_y) * ty
        return max(min(projections) - pi_projection, pi_projection - max(projections), 0.0)
    def projected(point: SurveyPoint, origin_x: float, origin_y: float, tx: float, ty: float) -> tuple[float, float]:
        along = (point.easting_m - origin_x) * tx + (point.northing_m - origin_y) * ty
        return origin_x + along * tx, origin_y + along * ty
    in_start_x, in_start_y = projected(incoming[0], ax, ay, ux, uy)
    in_end_x, in_end_y = projected(incoming[-1], ax, ay, ux, uy)
    out_start_x, out_start_y = projected(outgoing[0], bx, by, vx, vy)
    out_end_x, out_end_y = projected(outgoing[-1], bx, by, vx, vy)
    return FittedPI(x, y, abs(angle), CurveDirection.LEFT if angle > 0 else CurveDirection.RIGHT,
                    rms_a, rms_b, extrapolation(incoming, ax, ay, ux, uy),
                    extrapolation(outgoing, bx, by, vx, vy), len(incoming), len(outgoing),
                    in_start_x, in_start_y, out_end_x, out_end_y,
                    (x - in_end_x) * ux + (y - in_end_y) * uy < 0,
                    (x - out_start_x) * vx + (y - out_start_y) * vy > 0)
