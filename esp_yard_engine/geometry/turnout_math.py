"""Turnout (Points & Crossings) and Reverse Curve Mathematical Engine.

Computes exact 2D centerline geometry for Indian Railways standard turnouts
(1 in 8.5, 1 in 12, 1 in 16) and reverse curve transitions into parallel loop lines,
including precise Fouling Mark (FM) clearance (4.265m) and Sand Hump overrun geometry.

References:
  - RDSO Track Design Manual & Standard Drawings for PSC Turnouts
  - Indian Railways Schedule of Dimensions (IRSOD BG 2022)
  - Indian Railways Permanent Way Manual (IRPWM 2020)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any, Optional

from esp_yard_engine.models.yard_schema import (
    TurnoutType,
    TurnoutDirection,
    load_yard_standards,
)

logger = logging.getLogger("ESP_Yard_Engine.TurnoutMath")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class TurnoutGeometryResult:
    """Detailed geometric profile of a turnout and its reverse transition curve."""

    turnout_id: str
    turnout_type: TurnoutType
    direction: TurnoutDirection
    switch_toe: tuple[float, float]
    nose_of_crossing: tuple[float, float]
    reverse_curve_start: tuple[float, float]
    reverse_curve_end: tuple[float, float]
    fouling_mark: tuple[float, float]
    turnout_polyline: list[tuple[float, float]] = field(default_factory=list)
    crossing_angle_deg: float = 0.0
    crossing_angle_rad: float = 0.0
    lead_length_m: float = 0.0
    total_divergence_length_m: float = 0.0
    target_offset_m: float = 0.0
    fouling_mark_clearance_m: float = 4.265


def _get_turnout_params(
    turnout_type: TurnoutType,
    standards: Optional[dict[str, Any]] = None,
    lead_length_m: float | None = None,
    crossing_angle_deg: float | None = None,
) -> tuple[float, float, float, float]:
    """Retrieve switch lead length, crossing angle, radius, and overall length from standards."""
    stds = standards or load_yard_standards()
    t_cfg = stds.get("turnouts", {})
    key = turnout_type.value

    cfg = t_cfg.get(key, {})
    lead_m = float(lead_length_m if lead_length_m is not None else cfg.get("switch_lead_length_m", 39.90 if key == "1_in_12" else 28.625))
    angle_deg = float(crossing_angle_deg if crossing_angle_deg is not None else cfg.get("crossing_angle_deg", 4.76364 if key == "1_in_12" else 6.70984))
    radius_m = float(cfg.get("divergence_radius_m", 441.0 if key == "1_in_12" else 218.0))
    overall_m = float(cfg.get("overall_length_m", 44.50 if key == "1_in_12" else 32.0))
    angle_rad = angle_deg * (math.pi / 180.0)

    return lead_m, angle_deg, angle_rad, radius_m


def calculate_turnout_geometry(
    turnout_id: str,
    turnout_type: TurnoutType,
    direction: TurnoutDirection,
    switch_toe_s: float,
    target_offset_m: float,
    standards: Optional[dict[str, Any]] = None,
    num_sample_points: int = 40,
    lead_length_m: float | None = None,
    crossing_angle_deg: float | None = None,
) -> TurnoutGeometryResult:
    """Compute the full 2D centerline geometry of a turnout connecting a parent track to a loop.

    Coordinate Frame:
      - s: Longitudinal stationing (meters) along the parent track.
      - y: Lateral perpendicular offset (meters, positive = Left, negative = Right).

    Processing:
      1. Switch Lead Curve: Starts at (switch_toe_s, 0) and curves out at radius R_lead
         until the crossing nose is reached at angle alpha.
      2. Intermediate Tangent: Straight segment along angle alpha past the crossing.
      3. Reverse Curve: Circular arc of radius R_rev turning in the opposite direction
         until the track is parallel (angle = 0) at the target track center offset D.
      4. Fouling Mark (FM): Point along the diverging curve where lateral distance equals
         `fouling_mark_clearance_m` (RDSO IRSOD: 4.265m).

    Args:
        turnout_id: Identifier, e.g. "101A".
        turnout_type: Ratio, e.g. TurnoutType.ONE_IN_12.
        direction: FACING_LEFT, FACING_RIGHT, TRAILING_LEFT, or TRAILING_RIGHT.
        switch_toe_s: Longitudinal chainage of the switch toe relative to Station C/L (meters).
        target_offset_m: Parallel loop line lateral offset (e.g. +5.30m or -5.30m).
        standards: Optional pre-loaded yard standards dictionary.
        num_sample_points: Number of polyline interpolation steps.

    Returns:
        TurnoutGeometryResult with all key points, angles, clearances, and polyline coordinates.
    """
    stds = standards or load_yard_standards()
    lead_m, angle_deg, angle_rad, radius_rev = _get_turnout_params(
        turnout_type, stds, lead_length_m, crossing_angle_deg
    )
    if not math.isfinite(lead_m) or lead_m <= 0.0:
        raise ValueError(f"Turnout lead length must be finite and positive; got {lead_m!r}.")
    if not math.isfinite(angle_rad) or not 0.0 < angle_rad < math.pi / 2.0:
        raise ValueError(f"Turnout crossing angle must be between 0 and 90 degrees; got {angle_deg!r}.")
    if not math.isfinite(radius_rev) or radius_rev <= 0.0:
        raise ValueError(f"Turnout divergence radius must be finite and positive; got {radius_rev!r}.")
    fm_clearance = float(stds.get("track_centers", {}).get("fouling_mark_clearance_m", 4.265))

    # Determine longitudinal sign and lateral sign
    # FACING: diverges in increasing chainage (+s)
    # TRAILING: converges in increasing chainage (meaning switch toe is at the outer end, runs into -s or vice versa)
    is_facing = direction in (TurnoutDirection.FACING_LEFT, TurnoutDirection.FACING_RIGHT)
    s_sign = 1.0 if is_facing else -1.0
    lat_sign = 1.0 if target_offset_m >= 0.0 else -1.0
    abs_target_offset = abs(target_offset_m)

    # ── 1. Switch Lead Curve Geometry ──
    # Switch starts at (s_toe, 0.0)
    # Arc of radius R_lead turning through angle alpha
    # Arc length = R_lead * alpha, x_span = R_lead * sin(alpha), y_span = R_lead * (1 - cos(alpha))
    # The standard switch lead is the theoretical toe-to-nose longitudinal
    # distance. Derive the entry-arc radius from that lead and the crossing
    # angle; the configured divergence radius governs the reverse curve.
    radius_lead = lead_m / math.sin(angle_rad)
    x_lead = lead_m
    y_lead = radius_lead * (1.0 - math.cos(angle_rad))

    s_toe = switch_toe_s
    y_toe = 0.0

    s_nc = s_toe + s_sign * x_lead
    y_nc = lat_sign * y_lead

    # ── 2. Reverse Curve & Straight Segment Calculation ──
    # We need total lateral divergence = abs_target_offset
    # The reverse curve of radius R_rev will take out angle alpha:
    # y_rev = R_rev * (1 - cos(alpha))
    # x_rev = R_rev * sin(alpha)
    y_rev = radius_rev * (1.0 - math.cos(angle_rad))
    x_rev = radius_rev * math.sin(angle_rad)

    # Remaining lateral distance between crossing nose and reverse curve
    y_remaining = abs_target_offset - (y_lead + y_rev)

    if y_remaining < 0:
        # If target offset is very narrow, scale reverse radius to fit exactly
        radius_rev = (abs_target_offset - y_lead) / max(0.0001, (1.0 - math.cos(angle_rad)))
        y_rev = radius_rev * (1.0 - math.cos(angle_rad))
        x_rev = radius_rev * math.sin(angle_rad)
        straight_len = 0.0
    else:
        # Intermediate straight length along angle alpha
        straight_len = y_remaining / max(0.0001, math.sin(angle_rad))

    # Point where reverse curve starts
    s_rev_start = s_nc + s_sign * (straight_len * math.cos(angle_rad))
    y_rev_start = lat_sign * (y_lead + straight_len * math.sin(angle_rad))

    # Point where reverse curve ends (track is now parallel at target_offset_m)
    s_rev_end = s_rev_start + s_sign * x_rev
    y_rev_end = target_offset_m

    # ── 3. Discretize Smooth Polyline ──
    points: list[tuple[float, float]] = []
    half_samples = max(8, num_sample_points // 3)

    # Sample Lead Curve
    for i in range(half_samples + 1):
        frac = i / float(half_samples)
        curr_theta = frac * angle_rad
        dx = radius_lead * math.sin(curr_theta)
        dy = radius_lead * (1.0 - math.cos(curr_theta))
        pt_s = s_toe + s_sign * dx
        pt_y = lat_sign * dy
        points.append((pt_s, pt_y))

    # Sample intermediate straight (if exists)
    if straight_len > 0.1:
        steps_straight = max(3, half_samples // 2)
        for i in range(1, steps_straight):
            frac = i / float(steps_straight)
            cur_s = s_nc + s_sign * (frac * straight_len * math.cos(angle_rad))
            cur_y = lat_sign * (y_lead + frac * straight_len * math.sin(angle_rad))
            points.append((cur_s, cur_y))

    # Sample Reverse Curve
    # Starts at (s_rev_start, y_rev_start) with angle alpha and curves to angle 0
    for i in range(1, half_samples + 1):
        frac = i / float(half_samples)
        curr_theta = (1.0 - frac) * angle_rad
        # From end:
        dx = x_rev - radius_rev * math.sin(curr_theta)
        dy = y_rev - radius_rev * (1.0 - math.cos(curr_theta))
        pt_s = s_rev_start + s_sign * dx
        pt_y = lat_sign * (y_lead + straight_len * math.sin(angle_rad) + dy)
        points.append((pt_s, pt_y))

    # Ensure end point is exactly parallel at target_offset_m
    points[-1] = (s_rev_end, target_offset_m)

    # ── 4. Locate Fouling Mark (FM) at lateral clearance = 4.265m ──
    fm_pt = calculate_fouling_mark_location(
        polyline=points,
        required_clearance_m=fm_clearance,
        target_offset_m=target_offset_m,
    )

    total_len = abs(s_rev_end - s_toe)

    logger.debug(
        "Computed Turnout %s (%s): Toe=(%.2f, %.2f), FM=(%.2f, %.2f), End=(%.2f, %.2f), Span=%.2fm",
        turnout_id,
        turnout_type.value,
        s_toe,
        y_toe,
        fm_pt[0],
        fm_pt[1],
        s_rev_end,
        y_rev_end,
        total_len,
    )

    return TurnoutGeometryResult(
        turnout_id=turnout_id,
        turnout_type=turnout_type,
        direction=direction,
        switch_toe=(s_toe, y_toe),
        nose_of_crossing=(s_nc, y_nc),
        reverse_curve_start=(s_rev_start, y_rev_start),
        reverse_curve_end=(s_rev_end, y_rev_end),
        fouling_mark=fm_pt,
        turnout_polyline=points,
        crossing_angle_deg=angle_deg,
        crossing_angle_rad=angle_rad,
        lead_length_m=lead_m,
        total_divergence_length_m=total_len,
        target_offset_m=target_offset_m,
        fouling_mark_clearance_m=fm_clearance,
    )


def calculate_fouling_mark_location(
    polyline: list[tuple[float, float]],
    required_clearance_m: float = 4.265,
    target_offset_m: float = 5.30,
) -> tuple[float, float]:
    """Find the exact (s, y) coordinate along the diverging polyline where lateral clearance = 4.265m.

    Under IRSOD (BG) Chapter II, the Fouling Mark is fixed where track centers converge to 4.265m.
    Interpolates linearly between polyline vertices.
    """
    if not polyline:
        return (0.0, 0.0)

    # If full target offset is less than clearance (rare edge case), place at midpoint
    if abs(target_offset_m) <= required_clearance_m:
        mid_idx = len(polyline) // 2
        return polyline[mid_idx]

    # Search for segment crossing required_clearance_m
    for i in range(len(polyline) - 1):
        s1, y1 = polyline[i]
        s2, y2 = polyline[i + 1]
        ay1, ay2 = abs(y1), abs(y2)

        if (ay1 <= required_clearance_m <= ay2) or (ay2 <= required_clearance_m <= ay1):
            denom = max(0.0001, abs(ay2 - ay1))
            t = (required_clearance_m - ay1) / denom
            fm_s = s1 + t * (s2 - s1)
            fm_y = math.copysign(required_clearance_m, target_offset_m)
            return (fm_s, fm_y)

    # Fallback: return end point
    return polyline[-1]


def calculate_sand_hump_geometry(
    loop_offset_m: float,
    trailing_point_s: float,
    length_m: Optional[float] = None,
    standards: Optional[dict[str, Any]] = None,
) -> list[tuple[float, float]]:
    """Compute the 2D centerline coordinates for a safety Overrun Line / Sand Hump.

    Under RDSO standards:
      - Length = 120m standard
      - Extends straight along the loop line offset past the trailing point
      - Ends with an RDSO Buffer Stop.
    """
    stds = standards or load_yard_standards()
    hump_len = length_m or float(stds.get("safety_structures", {}).get("sand_hump", {}).get("length_m", 120.0))

    start_s = trailing_point_s
    end_s = start_s + hump_len

    # Return start, midpoint, and buffer stop location
    return [
        (start_s, loop_offset_m),
        (start_s + hump_len * 0.5, loop_offset_m),
        (end_s, loop_offset_m),
    ]
