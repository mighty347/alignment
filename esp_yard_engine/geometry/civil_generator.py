"""Civil Infrastructure Geometry Generator.

Computes 2D real-world coordinates for:
  - High-level passenger platforms (coping lines at 1.670m clearance, end ramps at 1 in 30)
  - Foot Over Bridge (FOB) spans and staircases connecting platforms
  - Station Building footprints placed behind the home platform

Follows Indian Railways Schedule of Dimensions (IRSOD BG 2022) Chapter II.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Optional

from esp_yard_engine.models.yard_schema import (
    YardConfig,
    PlatformSpec,
    PlatformPlacement,
    StationBuildingSpec,
    FOBSpec,
    load_yard_standards,
)
from esp_yard_engine.geometry.alignment_projector import AlignmentProjector

logger = logging.getLogger("ESP_Yard_Engine.CivilGenerator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProjectedPlatform:
    """A passenger platform projected into real-world (X, Y) coordinates."""
    platform_id: str
    name: str
    placement: PlatformPlacement
    adjacent_track_id: str
    coping_edge_world: list[tuple[float, float]]
    boundary_polygon_world: list[tuple[float, float]]
    facing_ramp_polygon_world: Optional[list[tuple[float, float]]]
    trailing_ramp_polygon_world: Optional[list[tuple[float, float]]]
    label_pos_world: tuple[float, float]
    length_m: float
    width_m: float
    coping_offset_m: float


@dataclass
class ProjectedFOB:
    """A Foot Over Bridge (FOB) projected into real-world (X, Y) coordinates."""
    fob_id: str
    name: str
    bridge_span_polygon_world: list[tuple[float, float]]
    stair_polygons_world: list[list[tuple[float, float]]]
    center_pos_world: tuple[float, float]
    width_m: float
    span_length_m: float


@dataclass
class ProjectedStationBuilding:
    """A Station Building footprint projected into real-world (X, Y) coordinates."""
    name: str
    boundary_polygon_world: list[tuple[float, float]]
    label_pos_world: tuple[float, float]
    length_m: float
    width_m: float


@dataclass
class ProjectedCivilLayout:
    """Container of all world-projected platforms, buildings, and bridges."""
    platforms: list[ProjectedPlatform] = field(default_factory=list)
    fobs: list[ProjectedFOB] = field(default_factory=list)
    station_building: Optional[ProjectedStationBuilding] = None


def generate_projected_civil_elements(
    yard_config: YardConfig,
    projector: AlignmentProjector,
    standards: Optional[dict[str, Any]] = None,
    step_interval_m: float = 10.0,
) -> ProjectedCivilLayout:
    """Generate all world-projected civil structures (platforms, FOBs, station buildings).

    Args:
        yard_config: The YardConfig specification.
        projector: AlignmentProjector initialized with the route centerline.
        standards: Optional pre-loaded yard standards dictionary.
        step_interval_m: Sampling distance for curved platform edges.

    Returns:
        ProjectedCivilLayout with closed polygon boundaries and coping lines.
    """
    if not math.isfinite(step_interval_m) or step_interval_m <= 0.0:
        raise ValueError(f"step_interval_m must be finite and positive; got {step_interval_m!r}.")

    stds = standards or load_yard_standards()
    station_ch = yard_config.centerline_chainage_m
    track_lookup = {t.track_id: t for t in yard_config.tracks}

    projected_platforms: list[ProjectedPlatform] = []
    projected_fobs: list[ProjectedFOB] = []
    projected_building: Optional[ProjectedStationBuilding] = None

    # ── 1. Project Passenger Platforms ──
    # IRSOD BG 2022: Coping line is exactly 1.670m from track centerline.
    # Ramp length is defined by platform height (0.84m) and max slope ratio (1 in 30)
    pf_stds = stds.get("platforms", {}).get("high_level", {})
    ramp_length_m = float(pf_stds.get("ramp_length_m", 20.0))

    for pf in yard_config.platforms:
        track = track_lookup.get(pf.adjacent_track_id)
        track_offset = track.lateral_offset_m if track else 0.0

        # Placement side: Left = +normal, Right = -normal
        is_left = pf.placement == PlatformPlacement.LEFT_SIDE
        sign = 1.0 if is_left else -1.0

        coping_offset = track_offset + sign * pf.coping_edge_offset_m
        back_wall_offset = track_offset + sign * (pf.coping_edge_offset_m + pf.width_m)

        abs_start = station_ch + pf.start_chainage_rel_m
        abs_end = station_ch + pf.end_chainage_rel_m
        pf_len = abs(abs_end - abs_start)

        # Discretize coping line (front edge facing train)
        coping_pts: list[tuple[float, float]] = []
        back_pts: list[tuple[float, float]] = []

        cur_ch = min(abs_start, abs_end)
        end_ch = max(abs_start, abs_end)

        while cur_ch <= end_ch:
            coping_pts.append(projector.project_point(cur_ch, coping_offset))
            back_pts.append(projector.project_point(cur_ch, back_wall_offset))
            cur_ch += step_interval_m

        if cur_ch - step_interval_m < end_ch:
            coping_pts.append(projector.project_point(end_ch, coping_offset))
            back_pts.append(projector.project_point(end_ch, back_wall_offset))

        # Closed boundary polygon (coping forward, back wall reversed)
        boundary_poly = list(coping_pts) + list(reversed(back_pts))
        boundary_poly.append(coping_pts[0])  # Close polygon

        # Facing and trailing 1 in 30 end ramps
        facing_ramp_poly = None
        trailing_ramp_poly = None

        if pf.has_end_ramps:
            # Facing ramp (before start of platform)
            r_start_ch = min(abs_start, abs_end) - ramp_length_m
            p_ramp_start_c = projector.project_point(r_start_ch, coping_offset)
            p_ramp_start_b = projector.project_point(r_start_ch, back_wall_offset)
            p_ramp_end_c = coping_pts[0]
            p_ramp_end_b = back_pts[0]
            facing_ramp_poly = [p_ramp_start_c, p_ramp_end_c, p_ramp_end_b, p_ramp_start_b, p_ramp_start_c]

            # Trailing ramp (after end of platform)
            r_end_ch = max(abs_start, abs_end) + ramp_length_m
            p_tram_start_c = coping_pts[-1]
            p_tram_start_b = back_pts[-1]
            p_tram_end_c = projector.project_point(r_end_ch, coping_offset)
            p_tram_end_b = projector.project_point(r_end_ch, back_wall_offset)
            trailing_ramp_poly = [p_tram_start_c, p_tram_end_c, p_tram_end_b, p_tram_start_b, p_tram_start_c]

        # Centroid for label text
        mid_ch = (abs_start + abs_end) / 2.0
        mid_offset = (coping_offset + back_wall_offset) / 2.0
        label_pos = projector.project_point(mid_ch, mid_offset)

        projected_platforms.append(
            ProjectedPlatform(
                platform_id=pf.platform_id,
                name=pf.name,
                placement=pf.placement,
                adjacent_track_id=pf.adjacent_track_id,
                coping_edge_world=coping_pts,
                boundary_polygon_world=boundary_poly,
                facing_ramp_polygon_world=facing_ramp_poly,
                trailing_ramp_polygon_world=trailing_ramp_poly,
                label_pos_world=label_pos,
                length_m=pf_len,
                width_m=pf.width_m,
                coping_offset_m=coping_offset,
            )
        )

    # ── 2. Project Station Building ──
    if yard_config.station_building:
        sb = yard_config.station_building
        sb_ch = station_ch + sb.center_chainage_rel_m
        half_l = sb.length_m / 2.0
        half_w = sb.width_m / 2.0

        b_start_ch = sb_ch - half_l
        b_end_ch = sb_ch + half_l

        # Lateral bounding offsets
        b_near_offset = sb.lateral_offset_m - math.copysign(half_w, sb.lateral_offset_m)
        b_far_offset = sb.lateral_offset_m + math.copysign(half_w, sb.lateral_offset_m)

        p1 = projector.project_point(b_start_ch, b_near_offset)
        p2 = projector.project_point(b_end_ch, b_near_offset)
        p3 = projector.project_point(b_end_ch, b_far_offset)
        p4 = projector.project_point(b_start_ch, b_far_offset)
        sb_poly = [p1, p2, p3, p4, p1]

        lbl_pos = projector.project_point(sb_ch, sb.lateral_offset_m)

        projected_building = ProjectedStationBuilding(
            name=sb.name,
            boundary_polygon_world=sb_poly,
            label_pos_world=lbl_pos,
            length_m=sb.length_m,
            width_m=sb.width_m,
        )

    # ── 3. Project Foot Over Bridge (FOB) ──
    # Span reaches from Platform 1 across intermediate tracks to Platform 2
    for fob in yard_config.fobs:
        fob_ch = station_ch + fob.chainage_rel_m
        half_fob_w = fob.width_m / 2.0

        # Determine span lateral extent
        if projected_platforms:
            min_offset = min(pf.coping_offset_m for pf in projected_platforms)
            max_offset = max(pf.coping_offset_m for pf in projected_platforms)
            # Add extra reach over platform deck
            if min_offset < 0:
                min_offset -= 3.0
            else:
                min_offset = -3.0
            if max_offset > 0:
                max_offset += 3.0
            else:
                max_offset = 3.0
        else:
            min_offset, max_offset = -10.0, 10.0

        # Main span rectangular walkway box
        s_ch1 = fob_ch - half_fob_w
        s_ch2 = fob_ch + half_fob_w

        p1 = projector.project_point(s_ch1, min_offset)
        p2 = projector.project_point(s_ch2, min_offset)
        p3 = projector.project_point(s_ch2, max_offset)
        p4 = projector.project_point(s_ch1, max_offset)
        span_poly = [p1, p2, p3, p4, p1]

        # Stair landings on each platform (12m x 2.44m box aligned with platform)
        stair_polys: list[list[tuple[float, float]]] = []
        stair_len = float(stds.get("civil_structures", {}).get("foot_over_bridge", {}).get("stair_length_m", 12.0))
        stair_w = float(stds.get("civil_structures", {}).get("foot_over_bridge", {}).get("stair_width_m", 2.44))

        for pf in projected_platforms:
            # Landing starts at FOB and runs along platform
            st_start_ch = fob_ch + half_fob_w
            st_end_ch = st_start_ch + stair_len
            st_lat_mid = (pf.coping_offset_m + (pf.coping_offset_m + math.copysign(pf.width_m, pf.coping_offset_m))) / 2.0
            st_half_w = stair_w / 2.0

            sp1 = projector.project_point(st_start_ch, st_lat_mid - st_half_w)
            sp2 = projector.project_point(st_end_ch, st_lat_mid - st_half_w)
            sp3 = projector.project_point(st_end_ch, st_lat_mid + st_half_w)
            sp4 = projector.project_point(st_start_ch, st_lat_mid + st_half_w)
            stair_polys.append([sp1, sp2, sp3, sp4, sp1])

        fob_center = projector.project_point(fob_ch, (min_offset + max_offset) / 2.0)
        span_len = abs(max_offset - min_offset)

        projected_fobs.append(
            ProjectedFOB(
                fob_id=fob.fob_id,
                name=fob.name,
                bridge_span_polygon_world=span_poly,
                stair_polygons_world=stair_polys,
                center_pos_world=fob_center,
                width_m=fob.width_m,
                span_length_m=span_len,
            )
        )

    logger.info(
        "Successfully generated civil infrastructure: %d platforms, %d FOBs, building=%s",
        len(projected_platforms),
        len(projected_fobs),
        "Yes" if projected_building else "No",
    )

    return ProjectedCivilLayout(
        platforms=projected_platforms,
        fobs=projected_fobs,
        station_building=projected_building,
    )
