"""Railway ESP Yard Engine.

A standalone, portable Indian Railways Station Yard (Engineering Scale Plan)
geometry and CAD drafting package conforming to RDSO and IRSOD (BG) standards.
"""
from __future__ import annotations

from .models.yard_schema import (
    TurnoutType,
    TurnoutDirection,
    TrackRole,
    PlatformPlacement,
    TurnoutSpec,
    TrackSpec,
    PlatformSpec,
    StationBuildingSpec,
    FOBSpec,
    FoulingMarkSpec,
    YardConfig,
    load_yard_standards,
)
from .geometry.alignment_projector import AlignmentProjector
from .geometry.turnout_math import (
    TurnoutGeometryResult,
    calculate_turnout_geometry,
    calculate_fouling_mark_location,
    calculate_sand_hump_geometry,
)
from .geometry.track_generator import (
    ProjectedTrack,
    ProjectedTurnout,
    ProjectedOverrun,
    ProjectedStationMarker,
    ProjectedYardLayout,
    generate_projected_tracks,
)
from .geometry.civil_generator import (
    ProjectedPlatform,
    ProjectedFOB,
    ProjectedStationBuilding,
    ProjectedCivilLayout,
    generate_projected_civil_elements,
)
from .renderers.dxf_renderer import YardDXFRenderer

__version__ = "0.1.0"

__all__ = [
    "TurnoutType",
    "TurnoutDirection",
    "TrackRole",
    "PlatformPlacement",
    "TurnoutSpec",
    "TrackSpec",
    "PlatformSpec",
    "StationBuildingSpec",
    "FOBSpec",
    "FoulingMarkSpec",
    "YardConfig",
    "load_yard_standards",
    "AlignmentProjector",
    "TurnoutGeometryResult",
    "calculate_turnout_geometry",
    "calculate_fouling_mark_location",
    "calculate_sand_hump_geometry",
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
    "YardDXFRenderer",
]
