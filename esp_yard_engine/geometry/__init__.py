"""Geometry and mathematical modeling package for Railway ESP Yard Engine."""
from __future__ import annotations

from .turnout_math import (
    TurnoutGeometryResult,
    calculate_turnout_geometry,
    calculate_fouling_mark_location,
    calculate_sand_hump_geometry,
)
from .alignment_projector import AlignmentProjector
from .track_generator import (
    ProjectedTrack,
    ProjectedTurnout,
    ProjectedOverrun,
    ProjectedStationMarker,
    ProjectedYardLayout,
    generate_projected_tracks,
)
from .civil_generator import (
    ProjectedPlatform,
    ProjectedFOB,
    ProjectedStationBuilding,
    ProjectedCivilLayout,
    generate_projected_civil_elements,
)

__all__ = [
    "TurnoutGeometryResult",
    "calculate_turnout_geometry",
    "calculate_fouling_mark_location",
    "calculate_sand_hump_geometry",
    "AlignmentProjector",
    "ProjectedTrack",
    "ProjectedTurnout",
    "ProjectedOverrun",
    "ProjectedStationMarker",
    "ProjectedYardLayout",
    "generate_projected_tracks",
    "ProjectedPlatform",
    "ProjectedFOB",
    "ProjectedStationBuilding",
    "ProjectedCivilLayout",
    "generate_projected_civil_elements",
]
