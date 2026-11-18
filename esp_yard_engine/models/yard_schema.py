"""Strictly typed data models and schemas for Railway Engineering Scale Plans (ESP).

Follows Indian Railways RDSO / IRSOD (BG) 2022 standards.
"""
from __future__ import annotations

from enum import Enum
import logging
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field, field_validator

# Use standard logging
logger = logging.getLogger("ESP_Yard_Engine.Models")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_DEFAULT_STANDARDS_PATH = Path(__file__).resolve().parent.parent / "config" / "yard_standards.yaml"
_CACHED_STANDARDS: Optional[dict[str, Any]] = None


def load_yard_standards(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load and cache the RDSO yard standards YAML configuration."""
    global _CACHED_STANDARDS
    if _CACHED_STANDARDS is not None and config_path is None:
        return _CACHED_STANDARDS

    path = Path(config_path) if config_path else _DEFAULT_STANDARDS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Yard standards configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if config_path is None:
        _CACHED_STANDARDS = data

    logger.debug("Loaded yard standards from %s", path)
    return data


class TurnoutType(str, Enum):
    """RDSO Standard Broad Gauge Turnout Ratios."""
    ONE_IN_8_5 = "1_in_8_5"
    ONE_IN_12 = "1_in_12"
    ONE_IN_16 = "1_in_16"


class TurnoutDirection(str, Enum):
    """Direction and orientation of turnout relative to increasing chainage."""
    FACING_LEFT = "FACING_LEFT"        # Points diverge to Left in increasing chainage
    FACING_RIGHT = "FACING_RIGHT"      # Points diverge to Right in increasing chainage
    TRAILING_LEFT = "TRAILING_LEFT"    # Points converge from Left in increasing chainage
    TRAILING_RIGHT = "TRAILING_RIGHT"  # Points converge from Right in increasing chainage


class TrackRole(str, Enum):
    """Operational classification of track inside station yard limits."""
    MAIN = "MAIN"                      # Single track main line
    UP_MAIN = "UP_MAIN"                # UP direction main line
    DN_MAIN = "DN_MAIN"                # DOWN direction main line
    LOOP = "LOOP"                      # General loop line
    COMMON_LOOP = "COMMON_LOOP"        # Bi-directional crossing loop
    GOODS_SIDING = "GOODS_SIDING"      # Dedicated freight loading/stabling siding
    SAND_HUMP = "SAND_HUMP"            # Safety dead end / overrun line


class PlatformPlacement(str, Enum):
    """Location of passenger platform relative to tracks."""
    LEFT_SIDE = "LEFT_SIDE"            # Left side of track (facing increasing chainage)
    RIGHT_SIDE = "RIGHT_SIDE"          # Right side of track
    ISLAND = "ISLAND"                  # Situated between two parallel tracks


class TurnoutSpec(BaseModel):
    """Specification of an individual point & crossing (turnout)."""
    turnout_id: str = Field(..., description="Unique turnout identifier, e.g., '101A'")
    turnout_type: TurnoutType = Field(default=TurnoutType.ONE_IN_12)
    direction: TurnoutDirection = Field(..., description="Facing/Trailing orientation")
    parent_track_id: str = Field(..., description="ID of the track this turnout branches from (usually MAIN)")
    branch_track_id: str = Field(..., description="ID of the track receiving the diverted train")
    switch_toe_chainage_rel_m: float = Field(
        ...,
        description="Chainage of Theoretical Toe of Switch relative to Station Centerline (meters, negative = before station)",
    )
    lead_length_m: Optional[float] = Field(
        default=None,
        description="Switch lead length. If None, derived from yard standards configuration.",
    )
    crossing_angle_deg: Optional[float] = Field(
        default=None,
        description="Crossing angle in degrees. If None, derived from yard standards.",
    )


class TrackSpec(BaseModel):
    """Specification of a track centerline in the yard."""
    track_id: str = Field(..., description="Unique track ID, e.g., 'LINE_1_MAIN'")
    name: str = Field(..., description="Display name, e.g., 'MAIN LINE' or 'UP LOOP'")
    role: TrackRole = Field(default=TrackRole.MAIN)
    lateral_offset_m: float = Field(
        default=0.0,
        description="Perpendicular offset from alignment centerline in meters (0 = Main, + = Left, - = Right)",
    )
    start_chainage_rel_m: float = Field(
        ...,
        description="Start chainage relative to Station Centerline (meters)",
    )
    end_chainage_rel_m: float = Field(
        ...,
        description="End chainage relative to Station Centerline (meters)",
    )
    is_electrified: bool = Field(default=True)


class FoulingMarkSpec(BaseModel):
    """Specification of a Fouling Mark (FM) safety boundary."""
    fm_id: str = Field(..., description="Fouling mark ID, e.g., 'FM_101A'")
    parent_track_id: str = Field(..., description="Parent track ID")
    branch_track_id: str = Field(..., description="Branching track ID")
    chainage_rel_m: float = Field(
        ...,
        description="Chainage where track centers reach 4.265m clearance relative to Station C/L",
    )
    lateral_offset_m: float = Field(
        ...,
        description="Lateral midpoint coordinate between the converging tracks",
    )


class PlatformSpec(BaseModel):
    """Specification of a passenger platform."""
    platform_id: str = Field(..., description="Platform identifier, e.g., 'PF_1'")
    name: str = Field(..., description="Display label, e.g., 'PLATFORM NO. 1'")
    placement: PlatformPlacement = Field(default=PlatformPlacement.LEFT_SIDE)
    adjacent_track_id: str = Field(..., description="Track ID alongside this platform")
    start_chainage_rel_m: float = Field(
        ...,
        description="Platform start chainage relative to Station Centerline (meters)",
    )
    end_chainage_rel_m: float = Field(
        ...,
        description="Platform end chainage relative to Station Centerline (meters)",
    )
    width_m: float = Field(
        default=7.0,
        description="Platform width in meters",
    )
    coping_edge_offset_m: float = Field(
        default=1.670,
        description="Distance from adjacent track centerline to coping edge (RDSO standard: 1.670m)",
    )
    has_end_ramps: bool = Field(
        default=True,
        description="Whether 1 in 30 end ramps are included at both ends",
    )

    @property
    def length_m(self) -> float:
        return abs(self.end_chainage_rel_m - self.start_chainage_rel_m)


class StationBuildingSpec(BaseModel):
    """Specification of the Station Building footprint."""
    name: str = Field(default="STATION BUILDING")
    length_m: float = Field(default=35.0, description="Building length along the track direction")
    width_m: float = Field(default=12.0, description="Building width perpendicular to track")
    center_chainage_rel_m: float = Field(default=0.0, description="Usually 0.0 (aligned with Station C/L)")
    lateral_offset_m: float = Field(
        default=12.0,
        description="Distance from main track centerline to building center",
    )


class FOBSpec(BaseModel):
    """Specification of a Foot Over Bridge (FOB) connecting platforms."""
    fob_id: str = Field(default="FOB_1")
    name: str = Field(default="FOOT OVER BRIDGE")
    chainage_rel_m: float = Field(
        default=0.0,
        description="Location along track relative to Station C/L (meters)",
    )
    width_m: float = Field(default=3.66, description="Clear walkway width in meters")
    connected_platform_ids: list[str] = Field(default_factory=list)


class YardConfig(BaseModel):
    """Complete Station Yard Engineering Scale Plan (ESP) configuration."""
    station_name: str = Field(..., description="Official station name, e.g. 'SATMILE'")
    station_code: str = Field(default="", description="Short IR station code, e.g. 'SMI'")
    station_class: str = Field(default="B", description="Station classification: 'A', 'B', 'C'")
    centerline_chainage_m: float = Field(
        ...,
        description="Absolute chainage of Station Building Centerline along main alignment (meters)",
    )
    csr_length_m: float = Field(
        default=715.0,
        description="Clear Standing Room (CSR) between fouling marks in meters (RDSO: 715m for goods, 686m for passenger)",
    )
    tracks: list[TrackSpec] = Field(default_factory=list)
    turnouts: list[TurnoutSpec] = Field(default_factory=list)
    fouling_marks: list[FoulingMarkSpec] = Field(default_factory=list)
    platforms: list[PlatformSpec] = Field(default_factory=list)
    station_building: Optional[StationBuildingSpec] = Field(default=None)
    fobs: list[FOBSpec] = Field(default_factory=list)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("station_name")
    @classmethod
    def uppercase_station_name(cls, v: str) -> str:
        return v.strip().upper()

    @classmethod
    def create_standard_2_line_yard(
        cls,
        station_name: str,
        centerline_chainage_m: float,
        station_code: str = "",
        station_class: str = "B",
        loop_side: str = "LEFT",
        csr_length_m: float = 715.0,
        platform_length_m: float = 600.0,
        standards: Optional[dict[str, Any]] = None,
    ) -> "YardConfig":
        """Factory method to generate a standard Indian Railways 2-Line Crossing Station (Main + 1 Loop).

        Layout:
          - Line 1: Main Line (Offset = 0.0m)
          - Line 2: Common Loop (Offset = +5.3m or -5.3m)
          - 1:12 Turnouts at facing and trailing ends
          - Standard CSR (715m) between fouling marks
          - 600m High Level Passenger Platform alongside Main Line
          - Station Building at Station Centerline
        """
        stds = standards or load_yard_standards()
        track_center = float(stds.get("track_centers", {}).get("min_track_center_m", 5.30))
        t12_lead = float(stds.get("turnouts", {}).get("1_in_12", {}).get("switch_lead_length_m", 39.90))
        t12_angle = float(stds.get("turnouts", {}).get("1_in_12", {}).get("crossing_angle_deg", 4.7636))
        fm_clearance = float(stds.get("track_centers", {}).get("fouling_mark_clearance_m", 4.265))

        sign = 1.0 if loop_side.upper() == "LEFT" else -1.0
        loop_offset = sign * track_center

        # Geometry anchors
        half_csr = csr_length_m / 2.0
        # Fouling marks sit at ± half_csr
        fm_facing_rel = -half_csr
        fm_trailing_rel = half_csr

        # Turnout lead and toe calculation:
        # Distance from crossing to where tracks converge to 4.265m
        crossing_rad = t12_angle * (3.141592653589793 / 180.0)
        dist_fm_to_crossing = (track_center - fm_clearance) / max(0.001, crossing_rad)
        turnout_total_span = t12_lead + dist_fm_to_crossing + 15.0

        toe_facing_rel = fm_facing_rel - turnout_total_span
        toe_trailing_rel = fm_trailing_rel + turnout_total_span

        # Tracks
        main_track = TrackSpec(
            track_id="LINE_1_MAIN",
            name="MAIN LINE",
            role=TrackRole.MAIN,
            lateral_offset_m=0.0,
            start_chainage_rel_m=toe_facing_rel - 150.0,
            end_chainage_rel_m=toe_trailing_rel + 150.0,
        )

        loop_track = TrackSpec(
            track_id="LINE_2_LOOP",
            name="COMMON LOOP",
            role=TrackRole.COMMON_LOOP,
            lateral_offset_m=loop_offset,
            start_chainage_rel_m=toe_facing_rel,
            end_chainage_rel_m=toe_trailing_rel,
        )

        # Turnouts
        dir_facing = TurnoutDirection.FACING_LEFT if sign > 0 else TurnoutDirection.FACING_RIGHT
        dir_trailing = TurnoutDirection.TRAILING_LEFT if sign > 0 else TurnoutDirection.TRAILING_RIGHT

        turnouts = [
            TurnoutSpec(
                turnout_id="101A",
                turnout_type=TurnoutType.ONE_IN_12,
                direction=dir_facing,
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_2_LOOP",
                switch_toe_chainage_rel_m=toe_facing_rel,
                lead_length_m=t12_lead,
                crossing_angle_deg=t12_angle,
            ),
            TurnoutSpec(
                turnout_id="102A",
                turnout_type=TurnoutType.ONE_IN_12,
                direction=dir_trailing,
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_2_LOOP",
                switch_toe_chainage_rel_m=toe_trailing_rel,
                lead_length_m=t12_lead,
                crossing_angle_deg=t12_angle,
            ),
        ]

        # Fouling Marks
        fms = [
            FoulingMarkSpec(
                fm_id="FM_101A",
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_2_LOOP",
                chainage_rel_m=fm_facing_rel,
                lateral_offset_m=loop_offset / 2.0,
            ),
            FoulingMarkSpec(
                fm_id="FM_102A",
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_2_LOOP",
                chainage_rel_m=fm_trailing_rel,
                lateral_offset_m=loop_offset / 2.0,
            ),
        ]

        # Passenger Platform (placed opposite to common loop or on home side)
        pf_placement = PlatformPlacement.RIGHT_SIDE if sign > 0 else PlatformPlacement.LEFT_SIDE
        half_pf = platform_length_m / 2.0
        platform = PlatformSpec(
            platform_id="PF_1",
            name="PLATFORM NO. 1",
            placement=pf_placement,
            adjacent_track_id="LINE_1_MAIN",
            start_chainage_rel_m=-half_pf,
            end_chainage_rel_m=half_pf,
            width_m=float(stds.get("platforms", {}).get("high_level", {}).get("standard_width_m", 7.0)),
            coping_edge_offset_m=float(stds.get("platforms", {}).get("high_level", {}).get("edge_offset_m", 1.670)),
        )

        # Station Building behind Platform 1
        building_sign = -sign
        sb_spec = StationBuildingSpec(
            name=f"{station_name} STATION BUILDING",
            length_m=float(stds.get("civil_structures", {}).get("station_building", {}).get("default_length_m", 35.0)),
            width_m=float(stds.get("civil_structures", {}).get("station_building", {}).get("default_width_m", 12.0)),
            center_chainage_rel_m=0.0,
            lateral_offset_m=building_sign * (platform.width_m + 8.0),
        )

        return cls(
            station_name=station_name,
            station_code=station_code,
            station_class=station_class,
            centerline_chainage_m=centerline_chainage_m,
            csr_length_m=csr_length_m,
            tracks=[main_track, loop_track],
            turnouts=turnouts,
            fouling_marks=fms,
            platforms=[platform],
            station_building=sb_spec,
            fobs=[],
        )

    @classmethod
    def create_standard_3_line_yard(
        cls,
        station_name: str,
        centerline_chainage_m: float,
        station_code: str = "",
        station_class: str = "B",
        csr_length_m: float = 715.0,
        platform_length_m: float = 600.0,
        standards: Optional[dict[str, Any]] = None,
    ) -> "YardConfig":
        """Factory method for Standard Indian Railways 3-Line Station (Main Line + Up Loop + Down Loop)."""
        stds = standards or load_yard_standards()
        track_center = float(stds.get("track_centers", {}).get("min_track_center_m", 5.30))
        t12_lead = float(stds.get("turnouts", {}).get("1_in_12", {}).get("switch_lead_length_m", 39.90))
        t12_angle = float(stds.get("turnouts", {}).get("1_in_12", {}).get("crossing_angle_deg", 4.7636))

        half_csr = csr_length_m / 2.0
        fm_facing_rel = -half_csr
        fm_trailing_rel = half_csr

        crossing_rad = t12_angle * (3.141592653589793 / 180.0)
        dist_fm = (track_center - 4.265) / max(0.001, crossing_rad)
        turnout_span = t12_lead + dist_fm + 15.0

        toe_facing = fm_facing_rel - turnout_span
        toe_trailing = fm_trailing_rel + turnout_span

        tracks = [
            TrackSpec(
                track_id="LINE_1_MAIN",
                name="MAIN LINE",
                role=TrackRole.MAIN,
                lateral_offset_m=0.0,
                start_chainage_rel_m=toe_facing - 150.0,
                end_chainage_rel_m=toe_trailing + 150.0,
            ),
            TrackSpec(
                track_id="LINE_2_UP_LOOP",
                name="UP LOOP",
                role=TrackRole.LOOP,
                lateral_offset_m=track_center,  # Left
                start_chainage_rel_m=toe_facing,
                end_chainage_rel_m=toe_trailing,
            ),
            TrackSpec(
                track_id="LINE_3_DN_LOOP",
                name="DN LOOP",
                role=TrackRole.LOOP,
                lateral_offset_m=-track_center,  # Right
                start_chainage_rel_m=toe_facing,
                end_chainage_rel_m=toe_trailing,
            ),
        ]

        turnouts = [
            TurnoutSpec(
                turnout_id="101A",
                turnout_type=TurnoutType.ONE_IN_12,
                direction=TurnoutDirection.FACING_LEFT,
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_2_UP_LOOP",
                switch_toe_chainage_rel_m=toe_facing,
                lead_length_m=t12_lead,
                crossing_angle_deg=t12_angle,
            ),
            TurnoutSpec(
                turnout_id="101B",
                turnout_type=TurnoutType.ONE_IN_12,
                direction=TurnoutDirection.FACING_RIGHT,
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_3_DN_LOOP",
                switch_toe_chainage_rel_m=toe_facing + 45.0,  # Staggered turnout
                lead_length_m=t12_lead,
                crossing_angle_deg=t12_angle,
            ),
            TurnoutSpec(
                turnout_id="102A",
                turnout_type=TurnoutType.ONE_IN_12,
                direction=TurnoutDirection.TRAILING_LEFT,
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_2_UP_LOOP",
                switch_toe_chainage_rel_m=toe_trailing,
                lead_length_m=t12_lead,
                crossing_angle_deg=t12_angle,
            ),
            TurnoutSpec(
                turnout_id="102B",
                turnout_type=TurnoutType.ONE_IN_12,
                direction=TurnoutDirection.TRAILING_RIGHT,
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_3_DN_LOOP",
                switch_toe_chainage_rel_m=toe_trailing - 45.0,  # Staggered turnout
                lead_length_m=t12_lead,
                crossing_angle_deg=t12_angle,
            ),
        ]

        fms = [
            FoulingMarkSpec(
                fm_id="FM_101A",
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_2_UP_LOOP",
                chainage_rel_m=fm_facing_rel,
                lateral_offset_m=track_center / 2.0,
            ),
            FoulingMarkSpec(
                fm_id="FM_101B",
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_3_DN_LOOP",
                chainage_rel_m=fm_facing_rel + 45.0,
                lateral_offset_m=-track_center / 2.0,
            ),
            FoulingMarkSpec(
                fm_id="FM_102A",
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_2_UP_LOOP",
                chainage_rel_m=fm_trailing_rel,
                lateral_offset_m=track_center / 2.0,
            ),
            FoulingMarkSpec(
                fm_id="FM_102B",
                parent_track_id="LINE_1_MAIN",
                branch_track_id="LINE_3_DN_LOOP",
                chainage_rel_m=fm_trailing_rel - 45.0,
                lateral_offset_m=-track_center / 2.0,
            ),
        ]

        half_pf = platform_length_m / 2.0
        pf_width = float(stds.get("platforms", {}).get("high_level", {}).get("standard_width_m", 7.0))
        pf_edge_offset = float(stds.get("platforms", {}).get("high_level", {}).get("edge_offset_m", 1.670))

        # Two Passenger Platforms for the two Loop Lines:
        # Platform 1 on UP Loop (Left side), Platform 2 on DN Loop (Right side)
        platforms = [
            PlatformSpec(
                platform_id="PF_1",
                name="PLATFORM NO. 1",
                placement=PlatformPlacement.LEFT_SIDE,
                adjacent_track_id="LINE_2_UP_LOOP",
                start_chainage_rel_m=-half_pf,
                end_chainage_rel_m=half_pf,
                width_m=pf_width,
                coping_edge_offset_m=pf_edge_offset,
            ),
            PlatformSpec(
                platform_id="PF_2",
                name="PLATFORM NO. 2",
                placement=PlatformPlacement.RIGHT_SIDE,
                adjacent_track_id="LINE_3_DN_LOOP",
                start_chainage_rel_m=-half_pf,
                end_chainage_rel_m=half_pf,
                width_m=pf_width,
                coping_edge_offset_m=pf_edge_offset,
            ),
        ]

        # Station Building behind Platform 1 on the UP side
        sb = StationBuildingSpec(
            name=f"{station_name} STATION BUILDING",
            length_m=float(stds.get("civil_structures", {}).get("station_building", {}).get("default_length_m", 35.0)),
            width_m=float(stds.get("civil_structures", {}).get("station_building", {}).get("default_width_m", 12.0)),
            center_chainage_rel_m=0.0,
            lateral_offset_m=track_center + pf_edge_offset + pf_width + 6.0,
        )

        # Foot Over Bridge (FOB) connecting Platform 1 and Platform 2 across all 3 tracks
        fob = FOBSpec(
            fob_id="FOB_1",
            name="FOOT OVER BRIDGE",
            chainage_rel_m=0.0,
            width_m=float(stds.get("civil_structures", {}).get("foot_over_bridge", {}).get("standard_width_m", 3.66)),
            connected_platform_ids=["PF_1", "PF_2"],
        )

        return cls(
            station_name=station_name,
            station_code=station_code,
            station_class=station_class,
            centerline_chainage_m=centerline_chainage_m,
            csr_length_m=csr_length_m,
            tracks=tracks,
            turnouts=turnouts,
            fouling_marks=fms,
            platforms=platforms,
            station_building=sb,
            fobs=[fob],
        )

    @classmethod
    def create_standard_class_b_yard(
        cls,
        station_name: str,
        centerline_chainage_m: float,
        station_code: str = "",
        station_class: str = "B",
        csr_length_m: float = 715.0,
        platform_length_m: float = 600.0,
        standards: Optional[dict[str, Any]] = None,
    ) -> "YardConfig":
        """Standard Indian Railways Class B Station with 2 Loop Lines and 2 Platforms.

        Layout:
          - Line 1: Central Main Line (Offset = 0.0m) for non-stopping high-speed traffic
          - Line 2: UP Loop (Offset = +5.30m) with Platform No. 1 (Home Platform)
          - Line 3: DN Loop (Offset = -5.30m) with Platform No. 2
          - Foot Over Bridge (FOB) spanning across all 3 tracks connecting PF 1 and PF 2
          - 1:12 Turnouts at facing and trailing ends
          - Station Building located on Platform 1 side
        """
        return cls.create_standard_3_line_yard(
            station_name=station_name,
            centerline_chainage_m=centerline_chainage_m,
            station_code=station_code,
            station_class=station_class,
            csr_length_m=csr_length_m,
            platform_length_m=platform_length_m,
            standards=standards,
        )

    def save_to_yaml(self, path: str | Path) -> None:
        """Serialize configuration to a clean YAML file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Use mode='json' so that Enums and Path objects are dumped as standard JSON/YAML primitives
        data = self.model_dump(mode="json")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info("Saved yard configuration to: %s", out_path)

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "YardConfig":
        """Load configuration from a YAML file."""
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {in_path}")
        with open(in_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
