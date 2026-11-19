"""
Module: alignment/vertical/_vertical_models.py
Purpose: Typed dataclasses, YAML config loader, and pure RDSO gradient/constraint
         helpers for the Formation Level & Proposed Gradient engine.
Standards: RDSO GE:G-1 / IRPWM Para 408-409 / IRBM Para 106 / IRS SOD (BG)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml

try:
    from alignment.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger("VerticalModels")
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config/alignment_config.yaml"


class FormationLevelError(Exception):
    """Raised when vertical alignment / formation level inputs or constraints fail."""


# ---------------------------------------------------------------------------
# Public Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerrainStation:
    """Single station along the alignment profile."""
    chainage_m: float
    egl_m: float
    pfl_m: Optional[float] = None
    erl_m: Optional[float] = None
    efl_m: Optional[float] = None


@dataclass(frozen=True)
class HorizontalCurveInterval:
    """Horizontal curve overlap span used for IRPWM Para 408/409 checks."""
    ts_chainage_m: float
    st_chainage_m: float
    radius_m: float
    sc_chainage_m: Optional[float] = None
    cs_chainage_m: Optional[float] = None
    curve_id: str = ""


@dataclass(frozen=True)
class VerticalConstraint:
    """Bridge, RUB, culvert, HFL, or tie-in elevation constraint at a chainage."""
    chainage_m: float
    constraint_type: str
    name: str = ""
    hfl_m: Optional[float] = None
    min_pfl_m: Optional[float] = None
    exact_pfl_m: Optional[float] = None
    bed_level_m: Optional[float] = None
    clearance_m: Optional[float] = None
    structural_depth_m: Optional[float] = None
    span_length_m: float = 0.0
    bridge_type: str = ""
    proposed_span_text: str = ""


@dataclass(frozen=True)
class StationZone:
    """Station or yard limit requiring flatter track gradient per IRSOD."""
    start_chainage_m: float
    end_chainage_m: float
    name: str = ""
    max_gradient: Optional[float] = None


@dataclass(frozen=True)
class ResolvedConstraint:
    """Resolved vertical constraint with concrete required PFL at chainage."""
    ch_m: float
    req_pfl_m: float
    is_exact: bool
    constraint_type: str
    name: str
    span_length_m: float = 0.0


@dataclass
class PVIPoint:
    """Vertical Point of Intersection with IRPWM Para 409 parabolic VC data."""
    pvi_index: int
    chainage_m: float
    elevation_m: float
    grade_in: float
    grade_out: float
    grade_change: float
    curve_type: str
    vc_length_m: float
    rv_m: float
    pvc_chainage_m: float
    pvc_elevation_m: float
    pvt_chainage_m: float
    pvt_elevation_m: float


@dataclass
class ProposedGradientSegment:
    """Continuous Proposed Gradient tangent derived from Proposed Formation Level."""
    seg_no: int
    start_chainage_m: float
    end_chainage_m: float
    length_m: float
    start_pfl_m: float
    end_pfl_m: float
    start_prl_m: float
    end_prl_m: float
    gradient_decimal: float
    gradient_pct: float
    slope_ratio: float
    direction: str
    gradient_label: str
    max_allowed_gradient: float
    is_curve_compensated: bool
    is_station_zone: bool
    vpi_chainage_m: Optional[float] = None
    vpi_elevation_m: Optional[float] = None
    vc_length_m: float = 0.0


@dataclass
class ProfileStationOutput:
    """Station-level vertical design output (EGL, PFL, PRL, ERL/EFL, Cut/Fill, Gradient)."""
    chainage_m: float
    egl_m: float
    pfl_m: float
    prl_m: float
    bank_m: float
    cutting_m: float
    proposed_gradient_decimal: float
    proposed_gradient_pct: float
    proposed_gradient_label: str
    in_vertical_curve: bool
    erl_m: Optional[float] = None
    efl_m: Optional[float] = None
    diff_rail_level_m: Optional[float] = None


@dataclass
class VerificationCheck:
    """Individual RDSO / IRPWM Engineering Verification Gate assertion."""
    name: str
    passed: bool
    detail: str


@dataclass
class FormationLevelResult:
    """Complete vertical alignment, formation level, and proposed gradient result."""
    mode: str
    stations: list[ProfileStationOutput]
    pvis: list[PVIPoint]
    gradient_schedule: list[ProposedGradientSegment]
    verification_status: str
    checks: list[VerificationCheck] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result into JSON-safe dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Config Loading & Pure Engineering Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _read_vertical_yaml(config_path: Path, mtime_ns: int) -> dict[str, Any]:
    """Read YAML configuration keyed by mtime so config edits reload cleanly."""
    del mtime_ns
    with config_path.open(encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def load_vertical_config(
    config: Optional[Mapping[str, Any]],
    config_path: Optional[Path],
) -> dict[str, Any]:
    """Load vertical_standards from YAML with optional caller overrides."""
    raw: dict[str, Any] = {}
    resolved = (config_path or _DEFAULT_CONFIG_PATH).resolve()
    if resolved.exists():
        raw = _read_vertical_yaml(resolved, resolved.stat().st_mtime_ns)

    vcfg: dict[str, Any] = dict(raw.get("vertical_standards", {}))
    if config is not None:
        if isinstance(config.get("vertical_standards"), Mapping):
            vcfg.update(config["vertical_standards"])
        elif "track_buildup_m" in config:
            vcfg.update(config)

    required_keys = (
        "track_buildup_m",
        "ruling_gradient_plains",
        "min_gradient",
        "station_max_gradient",
        "level_gradient_threshold",
        "grade_compensation_factor",
        "standard_gradient_denominators",
        "min_tangent_length_m",
        "max_tangent_length_m",
        "min_grade_change_for_vc",
        "min_vc_length_m",
        "min_rv_summit_m",
        "min_rv_sag_m",
        "vc_overlap_headroom_ratio",
        "min_drainage_clearance_m",
        "max_excess_cut_m",
        "max_excess_fill_m",
        "bridge_clearances_m",
    )
    missing = [k for k in required_keys if k not in vcfg]
    if missing:
        raise FormationLevelError(
            f"Missing required keys in 'vertical_standards' config: {', '.join(missing)}"
        )
    return vcfg


def classify_gradient(gradient: float, level_threshold: float) -> tuple[str, float, str]:
    """Return (direction, slope_ratio, Indian Railways CAD label) for a decimal grade."""
    if abs(gradient) <= level_threshold:
        return "LEVEL", 0.0, "LEVEL"
    ratio = 1.0 / abs(gradient)
    direction = "RISE" if gradient > 0.0 else "FALL"
    suffix = "R" if gradient > 0.0 else "F"
    return direction, round(ratio, 2), f"1 IN {int(round(ratio))} {suffix}"


def format_gradient_label(gradient: float, level_threshold: float) -> str:
    """Format a decimal slope as '1 IN X R', '1 IN X F', or 'LEVEL'."""
    return classify_gradient(gradient, level_threshold)[2]


def validate_stations(stations: Sequence[TerrainStation]) -> tuple[np.ndarray, np.ndarray]:
    """Validate station list and return profile-frame arrays (p_chainages, p_egls)."""
    if len(stations) < 2:
        raise FormationLevelError("At least 2 terrain stations are required.")

    p_chainages = np.asarray([float(s.chainage_m) for s in stations], dtype=float)
    p_egls = np.asarray([float(s.egl_m) for s in stations], dtype=float)

    if not (np.all(np.isfinite(p_chainages)) and np.all(np.isfinite(p_egls))):
        raise FormationLevelError("Station chainage_m and egl_m values must all be finite.")
    if np.any(np.diff(p_chainages) <= 0.0):
        raise FormationLevelError("Station chainages must be strictly increasing.")

    return p_chainages, p_egls


def resolve_ruling_gradient(vcfg: Mapping[str, Any], terrain_type: Optional[str]) -> float:
    """Resolve ruling gradient from config for plains, hills, or DFC."""
    terrain = (terrain_type or str(vcfg.get("terrain_type", "plains"))).strip().lower()
    mapping = {
        "plains": float(vcfg["ruling_gradient_plains"]),
        "hills": float(vcfg["ruling_gradient_hills"]),
        "dfc": float(vcfg["ruling_gradient_dfc"]),
    }
    if terrain not in mapping:
        raise FormationLevelError(
            f"Unsupported terrain_type '{terrain}'. Expected one of: {sorted(mapping)}."
        )
    return mapping[terrain]


def build_standard_ratios(vcfg: Mapping[str, Any]) -> list[float]:
    """Convert config gradient denominators into sorted decimal slopes."""
    ratios: list[float] = []
    for denom in vcfg["standard_gradient_denominators"]:
        d_val = float(denom)
        ratios.append(0.0 if d_val <= 0.0 else 1.0 / d_val)
    return sorted(set(ratios))


def compute_segment_max_gradient(
    seg_start: float,
    seg_end: float,
    ruling_grad: float,
    min_grad: float,
    grade_comp_factor: float,
    curves: Sequence[HorizontalCurveInterval],
    station_zones: Sequence[StationZone],
    station_max_grad: float,
) -> tuple[float, bool, bool]:
    """Compute IRPWM Para 408 curve-compensated & station-capped max gradient."""
    max_grad = ruling_grad
    is_comp = False
    is_stn = False

    for curve in curves:
        if curve.radius_m <= 0.0:
            continue
        if min(seg_end, curve.st_chainage_m) > max(seg_start, curve.ts_chainage_m):
            comp_decimal = (grade_comp_factor / curve.radius_m) / 100.0
            comp_limit = max(min_grad, ruling_grad - comp_decimal)
            if comp_limit < max_grad:
                max_grad = comp_limit
                is_comp = True

    for zone in station_zones:
        if min(seg_end, zone.end_chainage_m) > max(seg_start, zone.start_chainage_m):
            zone_cap = float(zone.max_gradient) if zone.max_gradient is not None else station_max_grad
            if zone_cap < max_grad:
                max_grad = zone_cap
                is_stn = True

    return max_grad, is_comp, is_stn


def find_drainage_violations(
    p_chainages: np.ndarray,
    p_egls: np.ndarray,
    p_pfls: np.ndarray,
    min_clearance: float,
    tol: float,
    max_transition_span_m: float,
) -> list[int]:
    """Identify indices in shallow waterlogging bank (excluding clean cut/fill crossings)."""
    fill_h = p_pfls - p_egls
    n = len(fill_h)
    bad: list[int] = []
    idx = 0
    while idx < n:
        if 0.0 <= fill_h[idx] < (min_clearance - tol):
            run_start = idx
            while idx + 1 < n and 0.0 <= fill_h[idx + 1] < (min_clearance - tol):
                idx += 1
            run_end = idx
            left_is_cut = run_start > 0 and fill_h[run_start - 1] < 0.0
            right_is_cut = run_end + 1 < n and fill_h[run_end + 1] < 0.0
            run_span_m = float(p_chainages[run_end] - p_chainages[run_start])
            is_valid_crossing = (left_is_cut or right_is_cut) and (run_span_m <= max_transition_span_m)
            if not is_valid_crossing:
                bad.extend(range(run_start, run_end + 1))
        idx += 1
    return bad
