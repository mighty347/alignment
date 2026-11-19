"""
Module: alignment/vertical_router.py
Purpose: Unified FastAPI APIRouter for Railway Formation Level (PFL), Proposed
         Rail Level (PRL), and Proposed Gradient Schedule generation.

Single Unified Endpoint:
    POST /api/v1/alignment-studio/vertical/generate-formation
        Accepts all 5 engineering inputs in one request:
          1. Ground Level (EGL) & Existing Rail Level (ERL / EFL) stations
          2. Railway Stations (Station_Excel.xlsx headers or explicit yard limits)
          3. Bridge List (Bridge_List.xlsx column headers)
          4. Horizontal Curves (directly compatible with /detect output)
          5. Optional manual PVIs or existing station PFL (for PFL -> Gradient mode)

Standards: RDSO GE:G-1 / IRPWM Para 408-409 / IRBM Para 102 & 106 / IRS SOD (BG)
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator
import yaml

try:
    from alignment.utils.logger import get_logger
    from alignment.vertical import (
        FormationLevelError,
        FormationLevelSolver,
        HorizontalCurveInterval,
        TerrainStation,
        VerticalConstraint,
    )
except ImportError:
    from utils.logger import get_logger
    from vertical import (
        FormationLevelError,
        FormationLevelSolver,
        HorizontalCurveInterval,
        TerrainStation,
        VerticalConstraint,
    )

logger = get_logger("VerticalServiceRouter")

CONFIG_PATH = Path(__file__).resolve().parent / "config/alignment_config.yaml"
with CONFIG_PATH.open(encoding="utf-8") as _cfg_fh:
    CONFIG: dict[str, Any] = yaml.safe_load(_cfg_fh)

MAX_API_STATIONS = int(CONFIG.get("vertical_standards", {}).get("max_api_stations", 100000))

router = APIRouter(tags=["Railway Vertical Alignment & Formation Level"])


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class VerticalDesignMode(str, Enum):
    AUTO = "auto"
    INTERACTIVE_PVI = "interactive_pvi"
    FROM_PFL = "from_pfl"


class TerrainType(str, Enum):
    PLAINS = "plains"
    HILLS = "hills"
    DFC = "dfc"


class StationInput(BaseModel):
    """Per-station Ground Level (EGL) with optional Existing Rail/Formation or Proposed FL."""
    chainage_m: float = Field(..., description="Continuous alignment chainage in meters")
    egl_m: float = Field(..., description="Existing Ground Level (EGL) in meters above MSL")
    erl_m: Optional[float] = Field(None, description="Existing Rail Level (ERL) in meters")
    efl_m: Optional[float] = Field(None, description="Existing Formation Level (EFL) in meters")
    pfl_m: Optional[float] = Field(None, description="Pre-existing Proposed Formation Level (for from_pfl mode)")


class HorizontalCurveInput(BaseModel):
    """Horizontal curve interval compatible with both /detect output and canonical fields."""
    ts_chainage_m: Optional[float] = Field(None, description="Tangent-to-Spiral chainage (m)")
    sc_chainage_m: Optional[float] = Field(None, description="Spiral-to-Curve chainage (m)")
    cs_chainage_m: Optional[float] = Field(None, description="Curve-to-Spiral chainage (m)")
    st_chainage_m: Optional[float] = Field(None, description="Spiral-to-Tangent chainage (m)")
    # Direct aliases emitted by /api/v1/alignment-studio/detect (RDSOHorizontalCurve)
    chainage_ts: Optional[float] = None
    chainage_sc: Optional[float] = None
    chainage_cs: Optional[float] = None
    chainage_st: Optional[float] = None
    radius_m: float = Field(..., gt=0.0, description="Circular curve radius in meters")
    curve_id: str = ""

    @model_validator(mode="after")
    def resolve_curve_chainages(self) -> HorizontalCurveInput:
        if self.ts_chainage_m is None:
            self.ts_chainage_m = self.chainage_ts
        if self.sc_chainage_m is None:
            self.sc_chainage_m = self.chainage_sc
        if self.cs_chainage_m is None:
            self.cs_chainage_m = self.chainage_cs
        if self.st_chainage_m is None:
            self.st_chainage_m = self.chainage_st
        if self.ts_chainage_m is None or self.st_chainage_m is None:
            raise ValueError("Horizontal curve requires ts_chainage_m/chainage_ts and st_chainage_m/chainage_st.")
        if self.st_chainage_m <= self.ts_chainage_m:
            raise ValueError("Horizontal curve end chainage (ST) must exceed start chainage (TS).")
        return self


class VerticalConstraintInput(BaseModel):
    """Optional explicit vertical anchor or clearance override."""
    chainage_m: float
    constraint_type: str = Field(..., description="MAJOR_BRIDGE | MINOR_BRIDGE | BOX_CULVERT | RUB_NH | RUB_LOCAL | ROB | TIE_IN")
    name: str = ""
    hfl_m: Optional[float] = None
    min_pfl_m: Optional[float] = None
    exact_pfl_m: Optional[float] = None
    bed_level_m: Optional[float] = None
    clearance_m: Optional[float] = None
    structural_depth_m: Optional[float] = None
    span_length_m: float = 0.0


class ManualPVIInput(BaseModel):
    """Engineer-placed Vertical Point of Intersection for interactive_pvi mode."""
    chainage_m: float
    elevation_m: float


class GenerateFormationRequest(BaseModel):
    """Unified request combining EGL, ERL, Railway Stations, Bridge List & Horizontal Curves."""
    stations: list[StationInput] = Field(..., min_length=2, description="Ordered station profile with EGL and optional ERL/EFL/PFL")
    railway_stations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Railway stations using Station_Excel.xlsx headers or {name, chainage_m/location_km, start_chainage_m?, end_chainage_m?}",
    )
    bridge_list: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Bridge schedule rows using Bridge_List.xlsx headers (Bridge No., Bridge Type, Approx. Chainage (km), Proposed Span, Configration, Bed Level, HFL, Clearance)",
    )
    horizontal_curves: list[HorizontalCurveInput] = Field(
        default_factory=list,
        description="Horizontal curves from /detect for IRPWM Para 408 grade compensation & Para 409 VC-spiral check",
    )
    constraints: list[VerticalConstraintInput] = Field(
        default_factory=list,
        description="Optional additional explicit tie-in or clearance constraints",
    )
    manual_pvis: Optional[list[ManualPVIInput]] = Field(
        None,
        description="Required when mode='interactive_pvi'",
    )
    mode: VerticalDesignMode = VerticalDesignMode.AUTO
    terrain_type: TerrainType = TerrainType.PLAINS
    tie_in_to_existing: bool = Field(
        True,
        description="Automatically pin start/end formation level to EFL (= ERL - 0.762m) when ERL/EFL is supplied",
    )

    @model_validator(mode="after")
    def validate_mode_inputs(self) -> GenerateFormationRequest:
        if len(self.stations) > MAX_API_STATIONS:
            raise ValueError(f"Station count ({len(self.stations)}) exceeds maximum limit ({MAX_API_STATIONS}).")
        if self.mode == VerticalDesignMode.INTERACTIVE_PVI:
            if not self.manual_pvis or len(self.manual_pvis) < 2:
                raise ValueError("mode='interactive_pvi' requires at least 2 entries in 'manual_pvis'.")
        if self.mode == VerticalDesignMode.FROM_PFL:
            if any(st.pfl_m is None for st in self.stations):
                raise ValueError("mode='from_pfl' requires 'pfl_m' on every station.")
        return self


# ---------------------------------------------------------------------------
# Conversion Helpers
# ---------------------------------------------------------------------------

def _to_solver_inputs(
    payload: GenerateFormationRequest,
) -> tuple[list[TerrainStation], list[HorizontalCurveInterval], list[VerticalConstraint]]:
    """Map Pydantic request schemas into pure vertical solver dataclasses."""
    stations = [
        TerrainStation(
            chainage_m=st.chainage_m,
            egl_m=st.egl_m,
            pfl_m=st.pfl_m,
            erl_m=st.erl_m,
            efl_m=st.efl_m,
        )
        for st in payload.stations
    ]
    curves = [
        HorizontalCurveInterval(
            ts_chainage_m=float(c.ts_chainage_m),  # type: ignore[arg-type]
            st_chainage_m=float(c.st_chainage_m),  # type: ignore[arg-type]
            radius_m=c.radius_m,
            sc_chainage_m=c.sc_chainage_m,
            cs_chainage_m=c.cs_chainage_m,
            curve_id=c.curve_id,
        )
        for c in payload.horizontal_curves
    ]
    constraints = [
        VerticalConstraint(
            chainage_m=vc.chainage_m,
            constraint_type=vc.constraint_type,
            name=vc.name,
            hfl_m=vc.hfl_m,
            min_pfl_m=vc.min_pfl_m,
            exact_pfl_m=vc.exact_pfl_m,
            bed_level_m=vc.bed_level_m,
            clearance_m=vc.clearance_m,
            structural_depth_m=vc.structural_depth_m,
            span_length_m=vc.span_length_m,
        )
        for vc in payload.constraints
    ]
    return stations, curves, constraints


# ---------------------------------------------------------------------------
# Unified Formation Level & Proposed Gradient Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/api/v1/alignment-studio/vertical/generate-formation",
    status_code=status.HTTP_200_OK,
)
def generate_formation_level(payload: GenerateFormationRequest) -> dict[str, Any]:
    """Generate Proposed Formation Level (PFL), PRL, PVIs & Proposed Gradient Schedule.

    Considers Ground Level (EGL), Existing Rail Level (ERL/EFL), Railway Station
    yard flat limits, Bridge List (HFL/Clearance/Span/RUB/LC), and Horizontal Curves
    in a single unified call with full RDSO/IRPWM Engineering Verification Gate output.
    """
    solver = FormationLevelSolver(config=CONFIG)
    stations, curves, constraints = _to_solver_inputs(payload)

    try:
        if payload.mode == VerticalDesignMode.FROM_PFL:
            result = solver.design_from_pfl(
                stations=stations,
                horizontal_curves=curves,
                constraints=constraints,
                terrain_type=payload.terrain_type.value,
                bridge_records=payload.bridge_list,
                railway_station_records=payload.railway_stations,
                tie_in_to_existing=False,
            )
        else:
            manual_pvis = (
                [(p.chainage_m, p.elevation_m) for p in payload.manual_pvis]
                if payload.mode == VerticalDesignMode.INTERACTIVE_PVI and payload.manual_pvis
                else None
            )
            result = solver.solve(
                stations=stations,
                horizontal_curves=curves,
                constraints=constraints,
                manual_pvis=manual_pvis,
                terrain_type=payload.terrain_type.value,
                bridge_records=payload.bridge_list,
                railway_station_records=payload.railway_stations,
                tie_in_to_existing=payload.tie_in_to_existing,
            )
        return result.to_dict()
    except FormationLevelError as exc:
        logger.warning("Formation level design rejected input: %s", exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during formation level generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Formation level generation failed: {exc}",
        ) from exc

