"""
Package: alignment.vertical
Purpose: RDSO / IRPWM-compliant Vertical Alignment, Formation Level (PFL),
         Proposed Rail Level (PRL), and Proposed Gradient Schedule engine.
"""
from __future__ import annotations

from .bridge_constraint_loader import (
    classify_bridge_row,
    load_bridge_constraints_from_excel,
    load_bridge_constraints_from_records,
    load_station_zones_from_records,
    parse_span_length_m,
    resolve_bridge_chainage_m,
)
from .formation_level_solver import (
    FormationLevelError,
    FormationLevelResult,
    FormationLevelSolver,
    HorizontalCurveInterval,
    PVIPoint,
    ProfileStationOutput,
    ProposedGradientSegment,
    StationZone,
    TerrainStation,
    VerificationCheck,
    VerticalConstraint,
    classify_gradient,
    format_gradient_label,
)

__all__ = [
    "FormationLevelError",
    "FormationLevelResult",
    "FormationLevelSolver",
    "HorizontalCurveInterval",
    "PVIPoint",
    "ProfileStationOutput",
    "ProposedGradientSegment",
    "StationZone",
    "TerrainStation",
    "VerificationCheck",
    "VerticalConstraint",
    "classify_bridge_row",
    "classify_gradient",
    "format_gradient_label",
    "load_bridge_constraints_from_excel",
    "load_bridge_constraints_from_records",
    "load_station_zones_from_records",
    "parse_span_length_m",
    "resolve_bridge_chainage_m",
]

