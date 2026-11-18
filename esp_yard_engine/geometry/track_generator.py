"""Track Polyline and Turnout Projection Generator.

Orchestrates the conversion of YardConfig and TurnoutGeometry into fully projected
world-coordinate polylines (UTM / Paper space) ready for CAD rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Optional

from esp_yard_engine.models.yard_schema import (
    YardConfig,
    TrackSpec,
    TurnoutSpec,
    TrackRole,
    TurnoutType,
    TurnoutDirection,
    load_yard_standards,
)
from esp_yard_engine.geometry.alignment_projector import AlignmentProjector
from esp_yard_engine.geometry.turnout_math import (
    TurnoutGeometryResult,
    calculate_turnout_geometry,
    calculate_sand_hump_geometry,
)

logger = logging.getLogger("ESP_Yard_Engine.TrackGenerator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProjectedTrack:
    """A railway track centerline projected into real-world (X, Y) coordinates."""
    track_id: str
    name: str
    role: TrackRole
    polyline_world: list[tuple[float, float]]
    start_chainage_m: float
    end_chainage_m: float
    lateral_offset_m: float


@dataclass
class ProjectedTurnout:
    """A turnout divergence curve and safety markers projected into world coordinates."""
    turnout_id: str
    turnout_type: TurnoutType
    direction: TurnoutDirection
    switch_toe_world: tuple[float, float]
    nose_of_crossing_world: tuple[float, float]
    reverse_curve_end_world: tuple[float, float]
    turnout_polyline_world: list[tuple[float, float]]
    fouling_mark_world: tuple[float, float]
    fouling_indicator_bar_world: tuple[tuple[float, float], tuple[float, float]]
    crossing_angle_deg: float
    lead_length_m: float
    parent_track_id: str
    branch_track_id: str


@dataclass
class ProjectedOverrun:
    """A safety overrun line / sand hump projected into world coordinates."""
    hump_id: str
    track_id: str
    polyline_world: list[tuple[float, float]]
    buffer_stop_bar_world: tuple[tuple[float, float], tuple[float, float]]
    length_m: float


@dataclass
class ProjectedStationMarker:
    """Station Building Centerline flagpole and alignment reference."""
    station_name: str
    chainage_m: float
    track_point_world: tuple[float, float]
    flagpole_top_world: tuple[float, float]
    heading_rad: float
    normal_vector: tuple[float, float]


@dataclass
class ProjectedYardLayout:
    """Complete container of all projected yard trackwork and infrastructure."""
    station_name: str
    station_chainage_m: float
    csr_length_m: float
    tracks: list[ProjectedTrack] = field(default_factory=list)
    turnouts: list[ProjectedTurnout] = field(default_factory=list)
    overruns: list[ProjectedOverrun] = field(default_factory=list)
    station_marker: Optional[ProjectedStationMarker] = None


def generate_projected_tracks(
    yard_config: YardConfig,
    projector: AlignmentProjector,
    standards: Optional[dict[str, Any]] = None,
    step_interval_m: float = 5.0,
) -> ProjectedYardLayout:
    """Generate all world-projected tracks, turnouts, fouling marks, and safety structures.

    Args:
        yard_config: The YardConfig specification (from models/yard_schema.py).
        projector: AlignmentProjector initialized with the route centerline.
        standards: Optional pre-loaded yard standards dictionary.
        step_interval_m: Sampling distance for continuous track polylines (default: 5.0m).

    Returns:
        ProjectedYardLayout containing all (X, Y) CAD-ready entities.
    """
    if not math.isfinite(step_interval_m) or step_interval_m <= 0.0:
        raise ValueError(f"step_interval_m must be finite and positive; got {step_interval_m!r}.")

    stds = standards or load_yard_standards()
    station_ch = yard_config.centerline_chainage_m

    projected_tracks: list[ProjectedTrack] = []
    projected_turnouts: list[ProjectedTurnout] = []
    projected_overruns: list[ProjectedOverrun] = []

    # ── 1. Project Continuous Track Centerlines ──
    for t_spec in yard_config.tracks:
        abs_start = station_ch + t_spec.start_chainage_rel_m
        abs_end = station_ch + t_spec.end_chainage_rel_m
        offset = t_spec.lateral_offset_m

        # Discretize along chainage
        pts: list[tuple[float, float]] = []
        cur_ch = min(abs_start, abs_end)
        end_ch = max(abs_start, abs_end)

        while cur_ch <= end_ch:
            pts.append(projector.project_point(cur_ch, offset))
            cur_ch += step_interval_m

        if pts and cur_ch - step_interval_m < end_ch:
            pts.append(projector.project_point(end_ch, offset))

        projected_tracks.append(
            ProjectedTrack(
                track_id=t_spec.track_id,
                name=t_spec.name,
                role=t_spec.role,
                polyline_world=pts,
                start_chainage_m=abs_start,
                end_chainage_m=abs_end,
                lateral_offset_m=offset,
            )
        )

    # ── 2. Project Turnouts (Points & Crossings) ──
    track_lookup = {t.track_id: t for t in yard_config.tracks}

    for to_spec in yard_config.turnouts:
        branch_track = track_lookup.get(to_spec.branch_track_id)
        target_offset = branch_track.lateral_offset_m if branch_track else 5.30

        # Calculate local turnout mathematics
        res_local: TurnoutGeometryResult = calculate_turnout_geometry(
            turnout_id=to_spec.turnout_id,
            turnout_type=to_spec.turnout_type,
            direction=to_spec.direction,
            switch_toe_s=to_spec.switch_toe_chainage_rel_m,
            target_offset_m=target_offset,
            standards=stds,
            lead_length_m=to_spec.lead_length_m,
            crossing_angle_deg=to_spec.crossing_angle_deg,
        )

        # Project local polyline into world coordinates
        world_poly = projector.project_polyline(
            points_s_offset=res_local.turnout_polyline,
            base_chainage_m=station_ch,
        )

        # Project key reference points
        toe_w = projector.project_point(station_ch + res_local.switch_toe[0], res_local.switch_toe[1])
        nc_w = projector.project_point(station_ch + res_local.nose_of_crossing[0], res_local.nose_of_crossing[1])
        end_w = projector.project_point(station_ch + res_local.reverse_curve_end[0], res_local.reverse_curve_end[1])
        fm_w = projector.project_point(station_ch + res_local.fouling_mark[0], res_local.fouling_mark[1])

        # Fouling Mark Indicator Bar: a small 2.0m cross-line centered on the FM point
        _, _, norm_fm, _ = projector.get_frame(station_ch + res_local.fouling_mark[0])
        bar_half = 1.0 * projector.scale_factor
        fm_bar_p1 = (fm_w[0] - bar_half * norm_fm[0], fm_w[1] - bar_half * norm_fm[1])
        fm_bar_p2 = (fm_w[0] + bar_half * norm_fm[0], fm_w[1] + bar_half * norm_fm[1])

        projected_turnouts.append(
            ProjectedTurnout(
                turnout_id=to_spec.turnout_id,
                turnout_type=to_spec.turnout_type,
                direction=to_spec.direction,
                switch_toe_world=toe_w,
                nose_of_crossing_world=nc_w,
                reverse_curve_end_world=end_w,
                turnout_polyline_world=world_poly,
                fouling_mark_world=fm_w,
                fouling_indicator_bar_world=(fm_bar_p1, fm_bar_p2),
                crossing_angle_deg=res_local.crossing_angle_deg,
                lead_length_m=res_local.lead_length_m,
                parent_track_id=to_spec.parent_track_id,
                branch_track_id=to_spec.branch_track_id,
            )
        )

    # ── 3. Project Safety Overrun Lines / Sand Humps ──
    sand_hump_len = float(stds.get("safety_structures", {}).get("sand_hump", {}).get("length_m", 120.0))
    buffer_w = float(stds.get("safety_structures", {}).get("sand_hump", {}).get("buffer_stop_width_m", 3.0))

    # For each trailing turnout, build safety sand hump extending past the trailing switch
    for pt in projected_turnouts:
        if pt.direction in (TurnoutDirection.TRAILING_LEFT, TurnoutDirection.TRAILING_RIGHT):
            branch_t = track_lookup.get(pt.branch_track_id)
            if branch_t:
                hump_offset = branch_t.lateral_offset_m
                # Extend 120m along loop
                cur_s = (pt.switch_toe_world[0])  # Approximate anchor
                # Project along loop chainage from the switch toe outwards
                to_spec = next((t for t in yard_config.turnouts if t.turnout_id == pt.turnout_id), None)
                if to_spec:
                    start_s = to_spec.switch_toe_chainage_rel_m
                    end_s = start_s + sand_hump_len

                    hump_local = [
                        (start_s, hump_offset),
                        (start_s + sand_hump_len * 0.5, hump_offset),
                        (end_s, hump_offset),
                    ]
                    hump_world = projector.project_polyline(hump_local, base_chainage_m=station_ch)

                    # Buffer Stop Bar (perpendicular cross bar at end)
                    end_ch = station_ch + end_s
                    end_pos = hump_world[-1]
                    _, _, norm_end, _ = projector.get_frame(end_ch)
                    b_half = (buffer_w / 2.0) * projector.scale_factor
                    b1 = (end_pos[0] - b_half * norm_end[0], end_pos[1] - b_half * norm_end[1])
                    b2 = (end_pos[0] + b_half * norm_end[0], end_pos[1] + b_half * norm_end[1])

                    projected_overruns.append(
                        ProjectedOverrun(
                            hump_id=f"HUMP_{pt.turnout_id}",
                            track_id=pt.branch_track_id,
                            polyline_world=hump_world,
                            buffer_stop_bar_world=(b1, b2),
                            length_m=sand_hump_len,
                        )
                    )

    # ── 4. Project Station Centerline Flagpole ──
    st_pos, _, st_norm, st_heading = projector.get_frame(station_ch)
    flagpole_h = 75.0 * projector.scale_factor if projector.scale_factor != 1.0 else 75.0
    flag_top = (st_pos[0] + flagpole_h * st_norm[0], st_pos[1] + flagpole_h * st_norm[1])

    marker = ProjectedStationMarker(
        station_name=yard_config.station_name,
        chainage_m=station_ch,
        track_point_world=st_pos,
        flagpole_top_world=flag_top,
        heading_rad=st_heading,
        normal_vector=st_norm,
    )

    logger.info(
        "Successfully projected yard '%s' (Ch %.2f m): %d tracks, %d turnouts, %d overruns.",
        yard_config.station_name,
        station_ch,
        len(projected_tracks),
        len(projected_turnouts),
        len(projected_overruns),
    )

    return ProjectedYardLayout(
        station_name=yard_config.station_name,
        station_chainage_m=station_ch,
        csr_length_m=yard_config.csr_length_m,
        tracks=projected_tracks,
        turnouts=projected_turnouts,
        overruns=projected_overruns,
        station_marker=marker,
    )
