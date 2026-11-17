"""Private sampling & centerline extraction helpers for AlignmentCurveCalculator."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

from pyproj import Transformer

try:
    from alignment.geospatial.kml_ingestor import AlignmentStationPoint
    from alignment.horizontal.curve_solver import CurveDirection
except ImportError:
    try:
        from geospatial.kml_ingestor import AlignmentStationPoint
        from horizontal.curve_solver import CurveDirection
    except ImportError:
        from future_alignment_studio.geospatial.kml_ingestor import AlignmentStationPoint
        from future_alignment_studio.horizontal.curve_solver import CurveDirection

if TYPE_CHECKING:
    try:
        from alignment.horizontal.alignment_curve_calculator import (
            AlignmentElement,
            CalculatedAlignment,
        )
    except ImportError:
        from horizontal.alignment_curve_calculator import (
            AlignmentElement,
            CalculatedAlignment,
        )


def extract_centerline_coords(calculated: CalculatedAlignment) -> List[Tuple[float, float]]:
    """Extract deduplicated (Easting, Northing) vertices along solved elements in travel order."""
    if not calculated.elements:
        raise ValueError("CalculatedAlignment has no geometry elements.")

    coords: List[Tuple[float, float]] = []
    for elem in calculated.elements:
        for pt in elem.points:
            e_pt = (float(pt[0]), float(pt[1]))
            if not coords or math.hypot(e_pt[0] - coords[-1][0], e_pt[1] - coords[-1][1]) > 1e-6:
                coords.append(e_pt)
    return coords


def evaluate_element_point(elem: AlignmentElement, chainage_m: float) -> Tuple[float, float]:
    """Evaluate exact (Easting, Northing) coordinate at chainage_m inside an AlignmentElement."""
    if elem.length_m <= 1e-9 or not elem.points:
        return (float(elem.points[0][0]), float(elem.points[0][1]))

    s_local = max(0.0, min(elem.length_m, chainage_m - elem.start_chainage_m))
    if elem.element_type.value == "CIRCULAR_ARC" and elem.arc_geom is not None:
        ag = elem.arc_geom
        sweep_rad = math.radians(ag.sweep_angle_deg) * (1.0 if ag.direction == CurveDirection.LEFT else -1.0)
        ang = ag.start_angle_rad + (s_local / elem.length_m) * sweep_rad
        return (
            ag.center_easting_m + ag.radius_m * math.cos(ang),
            ag.center_northing_m + ag.radius_m * math.sin(ang),
        )

    pts = elem.points
    if len(pts) == 2:
        frac = s_local / elem.length_m
        return (pts[0][0] + frac * (pts[1][0] - pts[0][0]), pts[0][1] + frac * (pts[1][1] - pts[0][1]))

    cum_d = [0.0]
    for p1, p2 in zip(pts, pts[1:]):
        cum_d.append(cum_d[-1] + math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if cum_d[-1] <= 1e-9:
        return (float(pts[0][0]), float(pts[0][1]))

    target_d = (s_local / elem.length_m) * cum_d[-1]
    for k in range(len(cum_d) - 1):
        if target_d <= cum_d[k + 1] or k == len(cum_d) - 2:
            seg = cum_d[k + 1] - cum_d[k]
            frac = ((target_d - cum_d[k]) / seg) if seg > 1e-9 else 0.0
            return (
                pts[k][0] + frac * (pts[k + 1][0] - pts[k][0]),
                pts[k][1] + frac * (pts[k + 1][1] - pts[k][1]),
            )
    return (float(pts[-1][0]), float(pts[-1][1]))


def sample_alignment_stations(
    calculated: CalculatedAlignment,
    interval_m: float,
    chainage_precision: int = 3,
    epsg: Optional[int] = None,
) -> List[AlignmentStationPoint]:
    """Resample uniform chainage stations along solved alignment elements (tangents, spirals, arcs)."""
    if not calculated.elements:
        raise ValueError("CalculatedAlignment has no geometry elements to sample.")
    if not math.isfinite(interval_m) or interval_m <= 0.0:
        raise ValueError(f"interval_m must be finite and positive; got {interval_m!r}.")

    target_epsg = epsg if epsg is not None else calculated.epsg
    rev_transformer = (
        Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)
        if target_epsg is not None
        else None
    )

    start_ch = float(calculated.start_chainage_m)
    end_ch = start_ch + float(calculated.total_length_m)
    chainages: List[float] = []
    curr = start_ch
    while curr <= end_ch + 1e-6:
        chainages.append(min(curr, end_ch))
        curr += interval_m
    if not chainages or (end_ch - chainages[-1]) > 1e-3:
        chainages.append(end_ch)

    stations: List[AlignmentStationPoint] = []
    elem_idx = 0
    num_elems = len(calculated.elements)
    for idx, ch in enumerate(chainages):
        while elem_idx < num_elems - 1 and ch > calculated.elements[elem_idx].end_chainage_m + 1e-6:
            elem_idx += 1
        east, north = evaluate_element_point(calculated.elements[elem_idx], ch)
        lon_deg, lat_deg = rev_transformer.transform(east, north) if rev_transformer else (0.0, 0.0)
        km = int(ch // 1000.0)
        rem = ch % 1000.0
        stations.append(
            AlignmentStationPoint(
                station_index=idx,
                chainage_m=round(ch, chainage_precision),
                chainage_display=f"{km}+{rem:06.2f}",
                easting_m=round(east, 3),
                northing_m=round(north, 3),
                lon_deg=round(lon_deg, 7),
                lat_deg=round(lat_deg, 7),
            )
        )
    return stations


def generate_transition_points(
    origin: Tuple[float, float],
    tangent_unit: Tuple[float, float],
    normal_unit: Tuple[float, float],
    transition_len: float,
    radius: float,
    step_m: float,
    is_entry: bool,
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """Discretize an RDSO Cubic Parabola transition (y = x^3 / 6RL)."""
    if transition_len <= 1e-4:
        return [origin, origin], origin

    ux, uy = tangent_unit
    nx, ny = normal_unit
    num_steps = max(4, math.ceil(transition_len / step_m))
    coords: List[Tuple[float, float]] = []

    indices = range(num_steps + 1) if is_entry else range(num_steps, -1, -1)
    dir_sign = 1.0 if is_entry else -1.0
    for k in indices:
        s = min(transition_len, k * (transition_len / num_steps))
        x = s - (s ** 5) / (40.0 * (radius ** 2) * (transition_len ** 2))
        y = (s ** 3) / (6.0 * radius * transition_len)
        coords.append((origin[0] + dir_sign * x * ux + y * nx, origin[1] + dir_sign * x * uy + y * ny))

    if is_entry:
        coords[0] = origin
    else:
        coords[-1] = origin
    return coords, coords[-1]


def discretize_arc(
    center: Tuple[float, float],
    radius: float,
    start_angle_rad: float,
    sweep_angle_rad: float,
    arc_length_m: float,
    step_m: float,
) -> List[Tuple[float, float]]:
    """Discretize a circular arc for polyline previewers."""
    num_steps = max(4, math.ceil(arc_length_m / step_m))
    pts: List[Tuple[float, float]] = []
    for k in range(num_steps + 1):
        angle = start_angle_rad + (k / num_steps) * sweep_angle_rad
        pts.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
    return pts

