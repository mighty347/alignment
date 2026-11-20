"""
Pydantic Schemas and Enums for Railway L-Section Sheet Layout Templates.
Standards: RDSO / Indian Railways BG CAD Standards.
Supports 4 dynamic layout compartments:
  1. plan_viewport
  2. profile_viewport
  3. data_bands
  4. sidebar
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PlanViewportConfig(BaseModel):
    """Configuration for Top Region: 2D Track Alignment Plan Viewport."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    x: float = Field(25.0, description="X offset in paper mm from left sheet margin")
    y: float = Field(20.0, description="Y offset in paper mm from sheet margin")
    w: float = Field(964.0, description="Width of plan viewport in paper mm")
    h: float = Field(360.0, description="Height of plan viewport in paper mm")
    scale: float = Field(5000.0, description="Plan horizontal scale denominator (e.g. 5000 for 1:5000)")
    tick_interval_m: float = Field(100.0, alias="tickIntervalM", description="Stationing/chainage tick interval in ground meters")
    show_north_arrow: bool = Field(True, alias="showNorthArrow", description="Whether to render the North Arrow block")

    @field_validator("w", "h", "scale", "tick_interval_m")
    @classmethod
    def validate_positive_dimensions(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be strictly positive, got {v}")
        return float(v)


class ProfileViewportConfig(BaseModel):
    """Configuration for Bottom Region: Elevation Profile Viewport."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    x: float = Field(25.0, description="X offset in paper mm")
    y: float = Field(390.0, description="Y offset in paper mm")
    w: float = Field(964.0, description="Width of profile viewport in paper mm")
    h: float = Field(220.0, description="Height of profile viewport in paper mm")
    h_scale: float = Field(5000.0, alias="hScale", description="Profile horizontal scale (e.g. 5000 for 1:5000)")
    v_scale: float = Field(500.0, alias="vScale", description="Profile vertical scale (e.g. 500 for 1:500)")
    datum_round_step_m: float = Field(5.0, alias="datumRoundStepM", description="Datum level rounding step in meters (e.g. 5.0m)")
    datum_clearance_m: float = Field(20.0, alias="datumClearanceM", description="Buffer clearance below lowest EGL in meters")

    @field_validator("w", "h", "h_scale", "v_scale", "datum_round_step_m")
    @classmethod
    def validate_positive_values(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be strictly positive, got {v}")
        return float(v)

    @property
    def vertical_exaggeration(self) -> float:
        """Vertical exaggeration ratio = h_scale / v_scale (typically 10.0x for Indian Railways)."""
        return self.h_scale / self.v_scale


class DataBandItem(BaseModel):
    """Single strip in the ordered L-Section Profile Data Bands."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    key: str = Field(..., description="Unique technical identifier for band (e.g. proposed_rail_level, gradient)")
    label: str = Field(..., description="Human-readable title drawn in the left header label column")
    height_mm: float = Field(18.0, alias="heightMm", description="Row height in paper mm (standard 18mm or 24mm)")
    enabled: bool = Field(True, description="Whether this band is actively rendered on the sheet")

    @field_validator("height_mm")
    @classmethod
    def validate_height(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Band height must be positive, got {v}")
        return float(v)


class SidebarNotes(BaseModel):
    """General Engineering Notes clauses (RDSO Clauses 1 to 11)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    clauses: List[str] = Field(default_factory=list, description="Ordered list of numbered engineering clauses")


class SidebarRect(BaseModel):
    """Bounding box for schedules in the right sidebar (Curve Schedule, Bridge Schedule)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    enabled: bool = Field(True, description="Toggle schedule visibility")
    x: float = Field(1004.0, description="X coordinate in paper mm")
    y: float = Field(350.0, description="Y coordinate in paper mm")
    w: float = Field(165.0, description="Width in paper mm")
    h: float = Field(160.0, description="Height in paper mm")


class SidebarTitleBlock(BaseModel):
    """Zonal Railway and Division title block metadata."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    zonal_railway: str = Field("INDIAN RAILWAYS", alias="zonalRailway")
    division: str = Field("HEADQUARTERS", alias="division")
    project_name: str = Field("NEW B.G. CORRIDOR", alias="projectName")
    drawing_title: str = Field("COMPOSITE PLAN AND LONGITUDINAL SECTION", alias="drawingTitle")


class SidebarConfig(BaseModel):
    """Configuration for the 185mm RDSO Right Sidebar."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    notes: SidebarNotes = Field(default_factory=SidebarNotes)
    curve_schedule: Optional[SidebarRect] = Field(default=None, alias="curveSchedule")
    bridge_schedule: Optional[SidebarRect] = Field(default=None, alias="bridgeSchedule")
    title_block: SidebarTitleBlock = Field(default_factory=SidebarTitleBlock, alias="titleBlock")


class LSectionSheetTemplate(BaseModel):
    """Complete Model for RDSO L-Section Sheet Template.
    Enforces multi-tenancy attributes and the 4 dynamic layout compartments.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(..., description="Unique template identifier (e.g. rdso-a0-greenfield-standard)")
    user_id: Optional[str] = Field(None, alias="userId")
    organization_id: str = Field("SYSTEM", alias="organizationId")
    name: str = Field(..., description="Template title")
    kind: str = Field("Organisation", description="'Starter' | 'Organisation' | 'Saved'")
    created_by: str = Field("Organisation member", alias="createdBy")
    description: str = Field("", description="Template description")
    corridor_type: str = Field("greenfield", alias="corridorType", description="'greenfield' | 'doubling' | 'quadrupling'")
    paper: str = Field("A0", description="'A0' | 'A1' | 'Roll-915'")

    # Physical sheet dimensions in mm
    sheet_width_mm: float = Field(1189.0, alias="sheetWidthMm", description="A0 standard = 1189 mm")
    sheet_height_mm: float = Field(841.0, alias="sheetHeightMm", description="A0 standard = 841 mm")
    sidebar_width_mm: float = Field(185.0, alias="sidebarWidthMm", description="RDSO standard right panel = 185 mm")

    # 4 Dynamic Layout Compartments
    plan_viewport: PlanViewportConfig = Field(default_factory=PlanViewportConfig, alias="planViewport")
    profile_viewport: ProfileViewportConfig = Field(default_factory=ProfileViewportConfig, alias="profileViewport")
    data_bands: List[DataBandItem] = Field(default_factory=list, alias="dataBands")
    sidebar: SidebarConfig = Field(default_factory=SidebarConfig)

    # Status & Flags
    is_public: bool = Field(False, alias="isPublic")
    is_default: bool = Field(False, alias="isDefault")

    @field_validator("sheet_width_mm", "sheet_height_mm", "sidebar_width_mm")
    @classmethod
    def validate_positive_sheet_dimensions(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be strictly positive, got {v}")
        return float(v)

    @model_validator(mode="after")
    def validate_layout_consistency(self) -> LSectionSheetTemplate:
        """Validate that viewports and sidebar fit within sheet dimensions."""
        if self.sidebar_width_mm >= self.sheet_width_mm:
            raise ValueError(
                f"sidebar_width_mm ({self.sidebar_width_mm}) must be strictly less than sheet_width_mm ({self.sheet_width_mm})"
            )

        # Plan viewport boundary check
        if self.plan_viewport.x + self.plan_viewport.w > self.sheet_width_mm:
            raise ValueError(
                f"plan_viewport horizontal boundary ({self.plan_viewport.x + self.plan_viewport.w} mm) "
                f"exceeds sheet width ({self.sheet_width_mm} mm)"
            )

        # Profile viewport boundary check
        if self.profile_viewport.x + self.profile_viewport.w > self.sheet_width_mm:
            raise ValueError(
                f"profile_viewport horizontal boundary ({self.profile_viewport.x + self.profile_viewport.w} mm) "
                f"exceeds sheet width ({self.sheet_width_mm} mm)"
            )

        return self

    def get_enabled_bands(self) -> List[DataBandItem]:
        """Return only actively enabled data bands."""
        return [band for band in self.data_bands if band.enabled]

    def calculate_total_bands_height_mm(self) -> float:
        """Calculate total vertical height occupied by enabled data bands in mm."""
        return sum(band.height_mm for band in self.get_enabled_bands())

    def calculate_datum_elevation(self, lowest_egl: float) -> float:
        """Calculate RDSO datum elevation in meters based on configured clearance and rounding step.
        Formula: floor((lowest_egl - datum_clearance_m) / datum_round_step_m) * datum_round_step_m
        """
        step = self.profile_viewport.datum_round_step_m
        clearance = self.profile_viewport.datum_clearance_m
        return math.floor((lowest_egl - clearance) / step) * step

    def get_available_drawing_length_m(self) -> float:
        """Returns maximum ground length in meters that can be plotted on one sheet
        at the configured horizontal scale.
        """
        # Drawing area width in mm * (h_scale / 1000) = ground meters
        return (self.profile_viewport.w * self.profile_viewport.h_scale) / 1000.0

