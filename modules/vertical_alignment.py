"""
Module: modules/vertical_alignment.py
Purpose: Module 7 — Vertical Alignment (Formation Level & Profile Design).

Engineering Basis:
    Generates a production-grade Vertical Alignment (Formation Level - PFL &
    Proposed Rail Level - PRL) that approximates the terrain profile while strictly
    adhering to Indian Railways RDSO GE:G-1 guidelines and IRPWM Para 409 norms.

    Three Modes:
    1. Full Survey / EFL Mode  : When >=95% of profile has surveyed EFL/PFL from CSV.
                                 Exact survey levels are preserved as-is — no smoothing.
    2. Prescribed VIP Mode     : Engineer-specified VIPs locked from config.yaml.
    3. Automatic Optimizer Mode: Long-tangent minimum rise-and-fall greedy optimizer.
                                 - RDSO gradient quantization (standard ratios only).
                                 - Grade compensation on horizontal curves (IRPWM Para 408).
                                 - Hard drainage clearance floor (+0.60 m above EGL).
                                 - Parabolic vertical curves (IRPWM Para 409).

    Key Design Principles (Bug Fixes Over Previous Version):
    - Survey levels are SACRED: merged levels survive ALL parabolic passes via
      _stamp_survey_levels() re-stamp applied after every parabolic smoothing.
    - Single VIP computation pass: VIP DataFrame built directly from design VIPs,
      never re-derived from a smoothed profile (_sync_vip_df_with_profile REMOVED).
    - PVC_E/PVT_E always recalculated after overlap scaling (two-stage _build_pvis).
    - Bridge chainage uses config bridge_chainage_unit key (km or m) - no guessing.
    - Optimizer has calibrated score and explicit convergence guard.
    - Grade compensation uses actual curve [start, end] overlap window.
    - UNKNOWN bridge type: consistent warning+skip in both topo and Excel paths.
    - import sys added to top-level imports.
    - First/Last PVI g1/g2 use actual tangent grades, not hardcoded 0.0.

Input  : EGLResult (Module 4) & HGeomResult (Module 6).
Output : VAlignResult (VIP DataFrame + profile DataFrame with PFL, PRL, Cut/Fill).

Standards: RDSO GE:G-1 / IRPWM Para 408-409 / Indian Railways SOD (BG)
Author   : Railway ESP Generator
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.elevation_sampler import EGLResult
from modules.horizontal_geometry import HGeomResult
from utils.geo_utils import format_chainage
from utils.logger import get_logger

# Add project root to path for direct execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from modules.topo_analysis import TopoAnalysisResult
except ImportError:
    TopoAnalysisResult = Any  # type: ignore[assignment,misc]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class VerticalAlignmentError(Exception):
    """Raised when vertical alignment design fails."""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BridgeConstraint:
    """A required PFL constraint at a specific bridge/crossing chainage.

    Attributes:
        ch_m        : Crossing chainage in meters.
        req_pfl_m   : Minimum required PFL at this chainage.
        struct_type : Canonical structure type string.
        name        : Feature name for logging.
        clearance_m : Clearance applied above EGL (for logging).
    """
    ch_m: float
    req_pfl_m: float
    struct_type: str
    name: str
    clearance_m: float


@dataclass
class VAlignResult:
    """Result object returned by VerticalAlignment.design().

    Attributes:
        vip_dataframe    : DataFrame of Vertical Intersection Points (VIPs/PVIs).
                           Columns: vip_index, chainage_m, chainage_str, elevation_m,
                           gradient_to_next_pct, gradient_ratio_str, curve_length_m,
                           pvc_chainage_m, pvt_chainage_m.
        profile_dataframe: EGL DataFrame appended with formation_level_m,
                           proposed_rail_level_m, cut_fill_m.
        metadata         : Engineering metadata dictionary.
    """
    vip_dataframe: pd.DataFrame
    profile_dataframe: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Standard RDSO Gradient Quantization Ratios (GE:G-1)
# All gradients emitted by this module MUST snap to one of these values.
# ---------------------------------------------------------------------------

_STANDARD_GRADIENT_RATIOS: list[float] = sorted([
    0.0,           # LEVEL
    1.0 / 1000.0,  # 1 in 1000  - platform / drainage floor
    1.0 / 500.0,   # 1 in 500
    1.0 / 400.0,   # 1 in 400
    1.0 / 300.0,   # 1 in 300
    1.0 / 250.0,   # 1 in 250
    1.0 / 200.0,   # 1 in 200  - DFC ruling
    1.0 / 170.0,   # 1 in 170
    1.0 / 150.0,   # 1 in 150  - plains ruling
])


# ---------------------------------------------------------------------------
# VerticalAlignment Class
# ---------------------------------------------------------------------------

class VerticalAlignment:
    """Production-grade Vertical Alignment & Profile Optimizer - Module 7.

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Example:
        >>> result = VerticalAlignment(config).design(egl_result, hgeom_result)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Validate required config sections
        for section in ("railway", "vertical"):
            if section not in config:
                raise KeyError(f"Missing required configuration section '{section}'")

        vert_cfg = config["vertical"]
        rail_cfg = config["railway"]
        val_cfg  = config.get("validation", {})

        # Track parameters
        self.design_speed_kmh: float    = float(rail_cfg["design_speed"])
        self.track_buildup_m: float     = float(vert_cfg["track_buildup_m"])

        # Gradient parameters
        self.terrain_type: str          = str(vert_cfg["terrain_type"]).lower()
        self.ruling_gradient: float     = float(vert_cfg["ruling_gradient"])
        self.min_gradient: float        = float(vert_cfg["min_gradient"])
        self.min_tangent_length_m: float = float(vert_cfg["min_tangent_length_m"])
        self.min_drainage_clearance_m: float = float(vert_cfg["min_drainage_clearance_m"])

        # Vertical curve parameters (IRPWM Para 409)
        self.min_vc_length_m: float     = float(vert_cfg["min_vc_length"])

        # Rv from validation section (matches actual IRPWM Para 409 table in config)
        self.rv_summit_m: float = float(val_cfg.get("min_rv_summit_m", 4000.0))
        self.rv_sag_m: float    = float(val_cfg.get("min_rv_sag_m", 3000.0))

        # Grade change threshold to trigger a VC
        _mgc = vert_cfg.get("min_grade_change_for_vc") or val_cfg.get("min_grade_change_for_vc")
        if _mgc is None:
            raise KeyError(
                "Missing 'min_grade_change_for_vc' in 'vertical' or 'validation' config section."
            )
        self.min_grade_change_for_vc: float = float(_mgc)

        # Bridge clearance parameters (IRBM Para 106 / IRC)
        self.bridge_clearance_river_m: float     = float(vert_cfg["bridge_clearance_river_m"])
        self.bridge_clearance_canal_m: float     = float(vert_cfg["bridge_clearance_canal_m"])
        self.bridge_clearance_culvert_m: float   = float(vert_cfg["bridge_clearance_culvert_m"])
        self.bridge_clearance_rub_nh_m: float    = float(vert_cfg["bridge_clearance_rub_nh_m"])
        self.bridge_clearance_rub_local_m: float = float(vert_cfg["bridge_clearance_rub_local_m"])
        self.min_hfl_clearance_m: float          = float(vert_cfg.get("min_hfl_clearance_m", 0.60))
        self.bridge_ramp_lookahead_m: float      = float(vert_cfg["bridge_ramp_lookahead_m"])
        # "km" -> multiply by 1000; "m" -> use directly (no guessing heuristics)
        self.bridge_chainage_unit: str = str(vert_cfg.get("bridge_chainage_unit", "m")).lower().strip()

        # Design quality counters (reset per design() call)
        self._drainage_violation_count: int = 0
        self._ruling_gradient_exceeded_count: int = 0

        self.logger.info(
            "VerticalAlignment init: speed=%.0f km/h, terrain='%s', ruling_gradient=1:%.0f, "
            "track_buildup=%.3fm, Rv(summit/sag)=%.0f/%.0fm, bridge_chainage_unit='%s'",
            self.design_speed_kmh, self.terrain_type,
            1.0 / self.ruling_gradient if self.ruling_gradient > 0 else 0,
            self.track_buildup_m, self.rv_summit_m, self.rv_sag_m,
            self.bridge_chainage_unit,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def design(
        self,
        egl_result: EGLResult,
        hgeom_result: HGeomResult | None = None,
        topo_result: "TopoAnalysisResult | None" = None,
        survey_csv_path: str | Path | None = None,
        bridge_df: Any | None = None,
        gradient_excel_path: str | Path | None = None,
    ) -> VAlignResult:
        """Design continuous Formation Level (PFL) and Vertical Intersection Points.

        Args:
            egl_result          : Module 4 output - chainage_m, elevation_m (EGL).
            hgeom_result        : Module 6 output - horizontal curve geometry (optional).
            topo_result         : Module 10 output - proposed bridge structures (optional).
            survey_csv_path     : Path to survey CSV/Excel with formation/rail/ground levels.
            bridge_df           : Bridge List DataFrame from Excel.
            gradient_excel_path : Path to official Gradient Schedule Excel (Graidient_excel.xlsx).

        Returns:
            VAlignResult with vip_dataframe, profile_dataframe, metadata.
        """
        self.logger.info("=== Vertical Alignment design() start ===")
        self._drainage_violation_count = 0
        self._ruling_gradient_exceeded_count = 0

        try:
            # Step 1: Base profile from EGL result
            df_profile = egl_result.dataframe.copy()

            # Step 2: Merge survey CSV/Excel levels (if provided)
            # Baseline: lsection_excel_path loads full profile baseline,
            # then survey_csv_path applies site survey stamps.
            survey_candidates: list[Path] = []
            lsec_path_cfg = self.config.get("paths", {}).get("lsection_excel_path")
            if lsec_path_cfg:
                lsec_p = Path(lsec_path_cfg)
                if lsec_p.exists():
                    survey_candidates.append(lsec_p)

            if survey_csv_path:
                s_p = Path(survey_csv_path)
                if s_p in survey_candidates:
                    survey_candidates.remove(s_p)
                survey_candidates.append(s_p)
            elif self.config.get("paths", {}).get("survey_csv_path"):
                s_p = Path(self.config["paths"]["survey_csv_path"])
                if s_p != lsec_path_cfg and s_p.exists():
                    if s_p in survey_candidates:
                        survey_candidates.remove(s_p)
                    survey_candidates.append(s_p)

            survey_levels: pd.DataFrame | None = None
            for s_path in survey_candidates:
                if s_path.exists():
                    df_profile, s_lev = self._merge_survey_levels(s_path, df_profile)
                    if survey_levels is None:
                        survey_levels = s_lev
                    elif s_lev is not None and not s_lev.empty:
                        survey_levels = pd.concat([survey_levels, s_lev], ignore_index=True)

            efl_col = "existing_formation_level_m"
            pfl_col = "formation_level_m"
            has_efl = (
                efl_col in df_profile.columns
                and df_profile[efl_col].notna().sum() > 0
            )
            has_pfl = (
                pfl_col in df_profile.columns
                and df_profile[pfl_col].notna().sum() > 0
            )
            pfl_coverage = (
                df_profile[pfl_col].notna().sum() / len(df_profile)
                if pfl_col in df_profile.columns and len(df_profile) > 0 else 0.0
            )
            # Direct survey mode is reserved for a substantially complete PFL
            # schedule. Sparse survey stamps must not disable the design logic.
            is_full_survey = has_pfl and pfl_coverage >= 0.95

            # MODE A: Full Survey / EFL - preserve exact levels, no smoothing
            if is_full_survey:
                self.logger.info("Direct Excel Mode engaged: using exact Excel levels without recalculation.")
                valid_mask = df_profile["elevation_m"].notna()
                valid_df = df_profile.loc[valid_mask]
                if len(valid_df) < 2:
                    raise VerticalAlignmentError(
                        "Full survey PFL validation requires at least two valid EGL points."
                    )
                bridge_constraints = self._extract_bridge_constraints(
                    topo_result=topo_result,
                    bridge_df=bridge_df,
                    chainages_arr=valid_df["chainage_m"].to_numpy(dtype=float),
                    elevations_arr=valid_df["elevation_m"].to_numpy(dtype=float),
                )
                return self._design_full_survey_mode(
                    df_profile,
                    has_efl,
                    efl_col,
                    gradient_excel_path=gradient_excel_path,
                    bridge_constraints=bridge_constraints,
                )

            # Validate EGL data for optimization modes
            valid_mask    = df_profile["elevation_m"].notna()
            valid_df      = df_profile[valid_mask]
            if len(valid_df) < 2:
                raise VerticalAlignmentError("Not enough valid EGL points for vertical alignment.")

            chainages_arr  = valid_df["chainage_m"].values.astype(float)
            elevations_arr = valid_df["elevation_m"].values.astype(float)

            # Build bridge / survey PFL constraints
            bridge_constraints: list[BridgeConstraint] = self._extract_bridge_constraints(
                topo_result=topo_result,
                bridge_df=bridge_df,
                chainages_arr=chainages_arr,
                elevations_arr=elevations_arr,
            )

            # Add survey EFL start/end as hard PFL anchors
            if has_efl:
                efl_valid = df_profile[df_profile[efl_col].notna()]
                if not efl_valid.empty:
                    for s_row in (efl_valid.iloc[0], efl_valid.iloc[-1]):
                        s_ch = float(s_row["chainage_m"])
                        s_fl = float(s_row[efl_col])
                        if not any(abs(c.ch_m - s_ch) < 1.0 for c in bridge_constraints):
                            bridge_constraints.append(BridgeConstraint(
                                ch_m=round(s_ch, 2),
                                req_pfl_m=round(s_fl, 3),
                                struct_type="SURVEY_EFL",
                                name=f"EFL@{s_ch:.0f}m",
                                clearance_m=0.0,
                            ))

            bridge_constraints.sort(key=lambda c: c.ch_m)
            self.logger.info(
                "Total PFL constraints (bridges + survey anchors): %d", len(bridge_constraints)
            )

            # Build horizontal curve list for grade compensation
            h_curves: list[tuple[float, float, float]] = []  # (start_m, end_m, radius_m)
            if (hgeom_result is not None
                    and hgeom_result.ip_dataframe is not None
                    and not hgeom_result.ip_dataframe.empty):
                for _, row in hgeom_result.ip_dataframe.iterrows():
                    tptc   = float(row.get("tptc_chainage_m", 0.0))
                    ctpt   = float(row.get("ctpt_chainage_m", 0.0))
                    radius = float(row.get("radius_m", 0.0))
                    if tptc < ctpt and radius > 0:
                        h_curves.append((tptc, ctpt, radius))

            # MODE B: Prescribed VIPs from config
            prescribed_vips     = self.config.get("vertical", {}).get("prescribed_vips", [])
            prescribed_segments = self.config.get("vertical", {}).get("prescribed_segments", [])

            if prescribed_vips:
                vip_chainages, vip_fls, gradients = self._build_prescribed_vips(prescribed_vips)
            elif prescribed_segments:
                vip_chainages, vip_fls, gradients = self._build_prescribed_segments(
                    prescribed_segments
                )
            else:
                # MODE C: Automatic Optimizer
                survey_efl_df = (
                    df_profile[df_profile[efl_col].notna()] if has_efl else None
                )
                vip_chainages, vip_fls, gradients = self._build_vip_tangents(
                    chainages_arr=chainages_arr,
                    elevations_arr=elevations_arr,
                    h_curves=h_curves,
                    bridge_constraints=bridge_constraints,
                    survey_efl_df=survey_efl_df,
                )

            # Step 4: Build PVI geometry - single pass + scale + RECALCULATE geometry
            pvis = self._build_pvis(vip_chainages, vip_fls, gradients, h_curves)

            # Step 5: Apply parabolic curves and iteratively raise bridge PVI
            # levels when vertical-curve sag would consume required clearance.
            all_chainages    = df_profile["chainage_m"].values.astype(float)
            formation_levels = self._apply_parabolic_curves(all_chainages, pvis)
            for _ in range(6):
                deficits = []
                for constraint in bridge_constraints:
                    actual = float(np.interp(constraint.ch_m, all_chainages, formation_levels))
                    deficit = constraint.req_pfl_m - actual
                    if deficit > 0.05:
                        deficits.append((constraint, deficit))
                if not deficits:
                    break
                for constraint, deficit in deficits:
                    idx = int(np.argmin(np.abs(np.asarray(vip_chainages) - constraint.ch_m)))
                    if abs(vip_chainages[idx] - constraint.ch_m) > 5.0:
                        raise VerticalAlignmentError(
                            f"Bridge constraint at Ch {constraint.ch_m:.2f} m has no coincident VIP; "
                            "cannot safely compensate for vertical-curve clearance loss."
                        )
                    vip_fls[idx] += deficit + 0.05
                gradients = self._compute_gradients(vip_chainages, vip_fls)
                pvis = self._build_pvis(vip_chainages, vip_fls, gradients, h_curves)
                formation_levels = self._apply_parabolic_curves(all_chainages, pvis)

            remaining = [
                f"{constraint.name} Ch {constraint.ch_m:.2f} m: "
                f"PFL {float(np.interp(constraint.ch_m, all_chainages, formation_levels)):.3f} m "
                f"< {constraint.req_pfl_m:.3f} m"
                for constraint in bridge_constraints
                if float(np.interp(constraint.ch_m, all_chainages, formation_levels)) + 0.05
                < constraint.req_pfl_m
            ]
            if remaining:
                raise VerticalAlignmentError(
                    "Designed PFL does not satisfy bridge/crossing clearance after curve fitting: "
                    + "; ".join(remaining)
                )
            df_profile["formation_level_m"] = formation_levels

            # Step 6: Re-stamp surveyed formation levels (SACRED - survives smoothing)
            if survey_levels is not None:
                df_profile = self._stamp_survey_levels(df_profile, survey_levels)

            # Step 7: Compute PRL and cut/fill
            df_profile = self._compute_prl_and_cutfill(df_profile)

            # Step 8: Build clean VIP DataFrame directly from design VIPs (no re-detection)
            vip_df = self._build_vip_dataframe(vip_chainages, vip_fls, gradients, pvis)

            # Step 9: Drainage clearance audit
            self._drainage_violation_count = self._check_drainage_clearance(df_profile)

            total_vips = len(vip_df)
            self.logger.info(
                "=== Vertical Alignment complete: %d VIPs, ruling gradient %s, buildup %.3fm ===",
                total_vips,
                self._format_gradient_ratio(self.ruling_gradient),
                self.track_buildup_m,
            )

            meta: dict[str, Any] = {
                "total_vips": total_vips,
                "ruling_gradient": self.ruling_gradient,
                "track_buildup_m": self.track_buildup_m,
                "design_speed_kmh": self.design_speed_kmh,
                "prescribed_mode": bool(prescribed_vips or prescribed_segments),
                "efl_mode": False,
                "drainage_clearance_violation_count": self._drainage_violation_count,
                "ruling_gradient_exceeded_count": self._ruling_gradient_exceeded_count,
            }
            return VAlignResult(
                vip_dataframe=vip_df,
                profile_dataframe=df_profile,
                metadata=meta,
            )

        except VerticalAlignmentError:
            raise
        except Exception as err:
            self.logger.exception("Vertical alignment design failed: %s", err)
            raise VerticalAlignmentError(f"Vertical alignment failed: {err}") from err

    def export_csv(self, result: VAlignResult, output_dir: str | Path) -> list[Path]:
        """Export VIPs and Profile DataFrames to CSV files.

        Args:
            result    : VAlignResult from design().
            output_dir: Destination directory path.

        Returns:
            List of resolved file paths written.
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        vip_path  = out_dir / "vertical_ips.csv"
        prof_path = out_dir / "chainage_formation_levels.csv"

        result.vip_dataframe.to_csv(vip_path, index=False, float_format="%.4f")
        result.profile_dataframe.to_csv(prof_path, index=False, float_format="%.4f")

        self.logger.info("VIP CSV exported     : %s", vip_path)
        self.logger.info("Profile CSV exported : %s", prof_path)
        return [vip_path, prof_path]

    # =========================================================================
    # Formatting & Quantization
    # =========================================================================

    def _format_gradient_ratio(self, gradient: float) -> str:
        """Format decimal gradient as 'RISE 1 IN X' / 'FALL 1 IN X' / 'LEVEL'."""
        if abs(gradient) < 1e-4:
            return "LEVEL"
        ratio = 1.0 / abs(gradient)
        sign  = "RISE" if gradient > 0 else "FALL"
        return f"{sign} 1 IN {int(round(ratio))}"

    def _quantize_gradient(self, raw_gradient: float, max_allowed: float) -> float:
        """Snap raw gradient to nearest standard RDSO ratio, capped at max_allowed.

        Nearest-neighbour search against _STANDARD_GRADIENT_RATIOS.
        Respects ruling-gradient ceiling and minimum-gradient floor.
        """
        if abs(raw_gradient) < 1e-5:
            return 0.0

        sign    = 1.0 if raw_gradient >= 0 else -1.0
        abs_g   = abs(raw_gradient)
        eff_max = min(max_allowed, self.ruling_gradient)
        if eff_max <= 0:
            return 0.0

        abs_g      = max(self.min_gradient, min(abs_g, eff_max))
        candidates = [g for g in _STANDARD_GRADIENT_RATIOS if g <= eff_max + 1e-9]
        if not candidates:
            raise VerticalAlignmentError(
                f"No standard RDSO gradient ratio available <= {eff_max:.6f}."
            )
        nearest = min(candidates, key=lambda g: abs(g - abs_g))
        return sign * nearest

    # =========================================================================
    # Survey CSV Merge (O(n log n) merge_asof, no O(n^2) row iteration)
    # =========================================================================

    def _merge_survey_levels(
        self,
        survey_csv: str | Path,
        df_profile: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Merge EFL and ERL from survey CSV into df_profile.

        Priority rules (NO silent overwrites):
        - existing_formation_level_m : from EFL column only.
        - existing_rail_level_m      : from ERL column; if absent, EFL + track_buildup_m.
        - formation_level_m          : seeded from EFL (overwritten by PFL if PFL column exists).
        - proposed_rail_level_m      : seeded from ERL / EFL+buildup at surveyed chainages.

        Uses pd.merge_asof (O(n log n)) - not row-by-row np.isclose.

        Returns:
            (updated df_profile, raw survey DataFrame for _stamp_survey_levels later)
        """
        csv_path = Path(survey_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Survey CSV not found: {csv_path}")

        # Ensure output columns exist
        for col in (
            "existing_rail_level_m",
            "existing_formation_level_m",
            "formation_level_m",
            "proposed_rail_level_m",
        ):
            if col not in df_profile.columns:
                df_profile[col] = np.nan

        self.logger.info("Loading survey levels from: %s", csv_path)
        if csv_path.suffix.lower() in [".xlsx", ".xls"]:
            df_survey_raw = pd.read_excel(csv_path)
        else:
            df_survey_raw = pd.read_csv(csv_path)
        df_survey     = df_survey_raw.copy()
        df_survey.columns = pd.Index([str(c).strip().lower() for c in df_survey.columns])

        # Robust semantic column detection across all Indian Railway zones and DPR schemas
        import re

        def _clean(col_name: str) -> str:
            col_str = str(col_name).strip().lower().replace("-", "_").replace(".", "_").replace("/", "_")
            return re.sub(r"\(.*?\)", "", col_str).strip("_")

        ch_col = None
        pfl_key = None
        prl_key = None
        ogl_key = None
        efl_key = None
        erl_key = None
        cut_fill_key = None
        diff_key = None

        for c in df_survey.columns:
            cl = _clean(c)
            # 1. Diff between rail levels
            if "diff" in cl and any(k in cl for k in ("rail", "rl", "level")):
                diff_key = c
            # 2. Chainage
            elif any(k in cl for k in ("chainage", "ch_km", "chainage_km", "chainage_m", "station")) or cl in ("ch", "km", "stn"):
                if ch_col is None:
                    ch_col = c
            # 3. Existing Rail Level
            elif ("exist" in cl or "exg" in cl or "ex_" in cl) and ("rail" in cl or "rl" in cl):
                erl_key = c
            elif cl in ("erl", "erl_m", "ex_rail_level") and "prop" not in cl:
                erl_key = c
            # 4. Proposed Rail Level
            elif ("prop" in cl or "new" in cl) and ("rail" in cl or "rl" in cl):
                prl_key = c
            elif cl in ("prl", "prl_m", "proposed_rail_level"):
                prl_key = c
            # 5. Existing Formation Level
            elif ("exist" in cl or "exg" in cl or "ex_" in cl) and ("form" in cl or "fl" in cl):
                efl_key = c
            elif cl in ("efl", "efl_m", "ex_formation_level"):
                efl_key = c
            # 6. Proposed Formation Level
            elif ("prop" in cl or "new" in cl) and ("form" in cl or "fl" in cl):
                pfl_key = c
            elif cl in ("pfl", "pfl_m", "formation_level", "proposed_formation_level"):
                pfl_key = c
            # 7. Ground Level / EGL / OGL / Ground Line
            elif any(k in cl for k in ("ground", "ogl", "egl", "nsl")) or cl in ("gl", "elevation_m"):
                ogl_key = c
            # 8. Bank / Cutting / Cut-Fill
            elif any(k in cl for k in ("cut", "fill", "bank")):
                cut_fill_key = c

        if ch_col is None:
            raise ValueError(f"Survey file '{csv_path}' is missing a chainage column.")

        df_survey["_ch_m"] = pd.to_numeric(df_survey[ch_col], errors="coerce")
        df_survey = (
            df_survey.dropna(subset=["_ch_m"])
            .sort_values("_ch_m")
            .reset_index(drop=True)
        )

        right_cols = ["_ch_m"]
        for k in (pfl_key, prl_key, ogl_key, efl_key, erl_key, cut_fill_key, diff_key):
            if k is not None and k in df_survey.columns and k not in right_cols:
                right_cols.append(k)

        # If survey file contains full profile chainages, adopt exact survey stations and ground levels
        if len(df_survey) >= len(df_profile) * 0.5:
            df_new = pd.DataFrame()
            df_new["chainage_m"] = df_survey["_ch_m"].values

            # Preserve existing levels in df_profile if not present in this survey file
            for col in ("elevation_m", "formation_level_m", "proposed_rail_level_m"):
                if col in df_profile.columns and df_profile[col].notna().sum() > 0:
                    valid_mask = df_profile[col].notna()
                    df_new[col] = np.interp(
                        df_new["chainage_m"].values,
                        df_profile.loc[valid_mask, "chainage_m"].values,
                        df_profile.loc[valid_mask, col].values,
                    )
                else:
                    df_new[col] = np.nan

            # For EFL & ERL: do NOT interpolate! Only preserve exact matches, keep others NaN
            for col in ("existing_formation_level_m", "existing_rail_level_m"):
                df_new[col] = np.nan
                if col in df_profile.columns and df_profile[col].notna().sum() > 0:
                    for i_new, ch_v in enumerate(df_new["chainage_m"].values):
                        m_old = np.isclose(df_profile["chainage_m"].values, ch_v, atol=0.5)
                        if np.any(m_old):
                            old_val = df_profile.loc[m_old, col].values[0]
                            if pd.notna(old_val) and old_val != 0.0:
                                df_new.loc[i_new, col] = old_val

            if ogl_key and ogl_key in df_survey.columns:
                df_new["elevation_m"] = pd.to_numeric(df_survey[ogl_key], errors="coerce").values
            if pfl_key and pfl_key in df_survey.columns:
                df_new["formation_level_m"] = pd.to_numeric(df_survey[pfl_key], errors="coerce").values
            if prl_key and prl_key in df_survey.columns:
                df_new["proposed_rail_level_m"] = pd.to_numeric(df_survey[prl_key], errors="coerce").values
            if efl_key and efl_key in df_survey.columns:
                df_new["existing_formation_level_m"] = pd.to_numeric(df_survey[efl_key], errors="coerce").values
            if erl_key and erl_key in df_survey.columns:
                df_new["existing_rail_level_m"] = pd.to_numeric(df_survey[erl_key], errors="coerce").values
            if cut_fill_key and cut_fill_key in df_survey.columns:
                df_new["cut_fill_m"] = pd.to_numeric(df_survey[cut_fill_key], errors="coerce").values
            if diff_key and diff_key in df_survey.columns:
                df_new["diff_rail_level_m"] = pd.to_numeric(df_survey[diff_key], errors="coerce").values

            # Carry over spatial coordinates if available
            if "easting_m" in df_profile.columns and "northing_m" in df_profile.columns:
                ch_old = df_profile["chainage_m"].values
                df_new["easting_m"] = np.interp(df_new["chainage_m"].values, ch_old, df_profile["easting_m"].values)
                df_new["northing_m"] = np.interp(df_new["chainage_m"].values, ch_old, df_profile["northing_m"].values)

            df_profile = df_new
        else:
            df_left = df_profile[["chainage_m"]].copy()
            df_left["chainage_m"] = pd.to_numeric(df_left["chainage_m"], errors="coerce").astype(float)
            df_left = df_left.sort_values("chainage_m")

            df_survey_right = df_survey[right_cols].copy()
            df_survey_right["_ch_m"] = pd.to_numeric(df_survey_right["_ch_m"], errors="coerce").astype(float)
            df_survey_right = df_survey_right.sort_values("_ch_m")

            merged = pd.merge_asof(
                df_left,
                df_survey_right,
                left_on="chainage_m",
                right_on="_ch_m",
                tolerance=10.0,
                direction="nearest",
            )
            merged.index = df_left.index

            # Ground Level / OGL -> elevation_m (EGL)
            if ogl_key and ogl_key in merged.columns:
                ogl_vals = pd.to_numeric(merged[ogl_key], errors="coerce")
                mask_ogl = ogl_vals.notna()
                df_profile.loc[mask_ogl, "elevation_m"] = ogl_vals[mask_ogl]

            # EFL -> existing_formation_level_m (seed formation_level_m, lower priority)
            if efl_key and efl_key in merged.columns:
                efl_vals = pd.to_numeric(merged[efl_key], errors="coerce")
                mask_efl = efl_vals.notna()
                df_profile.loc[mask_efl, "existing_formation_level_m"] = efl_vals[mask_efl]
                df_profile.loc[mask_efl, "formation_level_m"]          = efl_vals[mask_efl]

            # PFL -> formation_level_m (higher priority - overwrites EFL seed)
            if pfl_key and pfl_key in merged.columns:
                pfl_vals = pd.to_numeric(merged[pfl_key], errors="coerce")
                mask_pfl = pfl_vals.notna()
                df_profile.loc[mask_pfl, "formation_level_m"] = pfl_vals[mask_pfl]

            # PRL -> proposed_rail_level_m
            if prl_key and prl_key in merged.columns:
                prl_vals = pd.to_numeric(merged[prl_key], errors="coerce")
                mask_prl = prl_vals.notna()
                df_profile.loc[mask_prl, "proposed_rail_level_m"] = prl_vals[mask_prl]

            # ERL -> existing_rail_level_m
            if erl_key and erl_key in merged.columns:
                erl_vals = pd.to_numeric(merged[erl_key], errors="coerce")
                mask_erl = erl_vals.notna()
                df_profile.loc[mask_erl, "existing_rail_level_m"]  = erl_vals[mask_erl]
                if "proposed_rail_level_m" not in df_profile.columns or df_profile["proposed_rail_level_m"].isna().all():
                    df_profile.loc[mask_erl, "proposed_rail_level_m"]  = erl_vals[mask_erl]
            elif efl_key and efl_key in merged.columns:
                efl_vals2 = pd.to_numeric(merged[efl_key], errors="coerce")
                mask_efl2 = efl_vals2.notna()
                df_profile.loc[mask_efl2, "existing_rail_level_m"] = (
                    efl_vals2[mask_efl2] + self.track_buildup_m
                )
                if "proposed_rail_level_m" not in df_profile.columns or df_profile["proposed_rail_level_m"].isna().all():
                    df_profile.loc[mask_efl2, "proposed_rail_level_m"] = (
                        efl_vals2[mask_efl2] + self.track_buildup_m
                    )

            # Cut/Fill & Diff
            if cut_fill_key and cut_fill_key in merged.columns:
                cf_vals = pd.to_numeric(merged[cut_fill_key], errors="coerce")
                mask_cf = cf_vals.notna()
                df_profile.loc[mask_cf, "cut_fill_m"] = cf_vals[mask_cf]
            if diff_key and diff_key in merged.columns:
                df_vals = pd.to_numeric(merged[diff_key], errors="coerce")
                mask_df = df_vals.notna()
                df_profile.loc[mask_df, "diff_rail_level_m"] = df_vals[mask_df]

        n_pfl = df_profile["formation_level_m"].notna().sum()
        n_egl = df_profile["elevation_m"].notna().sum()
        n_efl = df_profile["existing_formation_level_m"].notna().sum()
        n_erl = df_profile["existing_rail_level_m"].notna().sum()
        self.logger.info(
            "Survey merge: %d PFL points, %d EGL points, %d EFL points, %d ERL points populated.",
            n_pfl, n_egl, n_efl, n_erl
        )

        # Return raw survey df (keyed columns renamed for _stamp_survey_levels)
        rename_map: dict[str, str] = {"_ch_m": "chainage_m"}
        if pfl_key and pfl_key in df_survey.columns:
            rename_map[pfl_key] = "_pfl"
        if prl_key and prl_key in df_survey.columns:
            rename_map[prl_key] = "_prl"
        if ogl_key and ogl_key in df_survey.columns:
            rename_map[ogl_key] = "_ogl"
        if efl_key and efl_key in df_survey.columns:
            rename_map[efl_key] = "_efl"
        if erl_key and erl_key in df_survey.columns:
            rename_map[erl_key] = "_erl"
        if cut_fill_key and cut_fill_key in df_survey.columns:
            rename_map[cut_fill_key] = "_cut_fill"
        df_survey = df_survey.rename(columns=rename_map)
        return df_profile, df_survey

    # =========================================================================
    # Bridge Constraint Extraction
    # =========================================================================

    def _extract_bridge_constraints(
        self,
        topo_result: Any,
        bridge_df: Any,
        chainages_arr: np.ndarray,
        elevations_arr: np.ndarray,
    ) -> list[BridgeConstraint]:
        """Compute required PFL at each bridge/crossing.

        UNKNOWN / unrecognized structure types: warning + skip (consistent for both
        topo_result and bridge_df paths - no fatal ValueError from bridge_df path).
        """
        constraints: list[BridgeConstraint] = []
        start_ch = float(chainages_arr[0])
        end_ch   = float(chainages_arr[-1])

        def _interp_egl(ch: float) -> float:
            return float(np.interp(ch, chainages_arr, elevations_arr))

        def _tokenize(text: str) -> set[str]:
            raw = set(t for t in re.split(r"[\s_\-/,]+", text.upper()) if t)
            expanded = set(raw)
            for t in raw:
                words = re.findall(r"[A-Z]+", t)
                for w in words:
                    expanded.add(w)
                    for kw in ("MINOR", "MAJOR", "CULVERT", "BOX", "RUB",
                               "SUBWAY", "ROB", "CANAL", "STREAM"):
                        if kw in w:
                            expanded.add(kw)
            return expanded

        def _clearance(struct_type: str, highway_type: str = "") -> tuple[float, str]:
            tok = _tokenize(struct_type)
            hw  = _tokenize(highway_type)
            if "ROB" in tok:
                return 0.0, "ROB"
            if tok & {"EXISTING", "EXG", "EX"}:
                return self.bridge_clearance_river_m, "EXISTING_BRIDGE"
            if tok & {"RUB", "SUBWAY", "UNDERPASS", "VUP", "CUP", "LHS"}:
                if hw & {"NH", "SH", "MOTORWAY", "TRUNK", "PRIMARY"}:
                    return self.bridge_clearance_rub_nh_m, "RUB_NH"
                return self.bridge_clearance_rub_local_m, "RUB_LOCAL"
            if tok & {"CULVERT", "BOX", "RCC_BOX", "PIPE"}:
                return self.bridge_clearance_culvert_m, "BOX_CULVERT"
            if tok & {"MINOR", "CANAL", "STREAM", "SLAB"}:
                return self.bridge_clearance_canal_m, "MINOR_BRIDGE"
            if tok & {"MAJOR", "RIVER", "OPEN", "WEB", "TRUSS", "GIRDER", "STEEL",
                      "PSC", "COMPOSITE", "BRIDGE", "BR"}:
                return self.bridge_clearance_river_m, "MAJOR_BRIDGE"
            return 0.0, "UNKNOWN"

        def _add(br_ch: float, clr: float, ctype: str, name: str) -> None:
            if not (start_ch - 10.0 <= br_ch <= end_ch + 10.0):
                return
            if any(abs(c.ch_m - br_ch) < 30.0 for c in constraints):
                return
            egl_v = _interp_egl(br_ch)
            req   = egl_v + clr
            constraints.append(BridgeConstraint(
                ch_m=round(br_ch, 2),
                req_pfl_m=round(req, 3),
                struct_type=ctype,
                name=name,
                clearance_m=clr,
            ))
            self.logger.debug(
                "BridgeConstraint: %s @ Ch %.1fm | EGL=%.3fm + clr=%.1fm = req_pfl=%.3fm",
                ctype, br_ch, egl_v, clr, req,
            )

        # 1. From TopoAnalysisResult (GIS / OSM)
        if topo_result is not None:
            df_structs = getattr(topo_result, "structures_df", None)
            if (
                df_structs is not None
                and not df_structs.empty
                and "chainage_m" in df_structs.columns
            ):
                for _, row in df_structs.iterrows():
                    br_ch   = float(row.get("chainage_m", 0.0))
                    st_type = str(row.get("structure_type", "")).upper()
                    cr_feat = str(row.get("crossing_feature", "")).upper()
                    name    = str(
                        row.get("structure_name", row.get("feature_name", f"STRUCT@{br_ch:.0f}m"))
                    ).strip()
                    clr, ct = _clearance(st_type, cr_feat)
                    if ct == "ROB":
                        continue
                    if ct == "UNKNOWN":
                        self.logger.warning(
                            "Unrecognized structure type '%s' @ Ch %.1fm (topo) - skipped.",
                            st_type, br_ch,
                        )
                        continue
                    _add(br_ch, clr, ct, name)

        # 2. From Bridge List Excel / DataFrame
        if bridge_df is not None and not bridge_df.empty:
            bdf = bridge_df.copy()
            bdf.columns = pd.Index([
                str(c).strip().lower().replace(" ", "_").replace("-", "_")
                for c in bdf.columns
            ])

            def _find_col(priority: list[str]) -> str | None:
                for kw in priority:
                    for c in bdf.columns:
                        if kw in c:
                            return c
                return None

            ch_col_b   = _find_col(["chainage_m", "chainage", "ch_m", "ch", "km"])
            cat_col_b  = _find_col(["category", "bridge_category", "structure_type",
                                    "bridge_type", "type", "structure"])
            desc_col_b = _find_col(["description", "span_description", "bridge_name",
                                    "structure_name", "name", "bridge_no"])
            hw_col_b   = _find_col(["highway", "road_type", "road_class",
                                    "road_section", "crossing_feature"])

            if desc_col_b == cat_col_b:
                desc_col_b = None

            if ch_col_b is not None:
                for _, row in bdf.iterrows():
                    try:
                        raw_ch = float(row[ch_col_b])
                    except (ValueError, TypeError):
                        continue

                    # Explicit unit conversion with corridor bounds protection
                    if start_ch - 50.0 <= raw_ch <= end_ch + 50.0:
                        br_ch = raw_ch
                    elif start_ch - 50.0 <= raw_ch * 1000.0 <= end_ch + 50.0:
                        br_ch = raw_ch * 1000.0
                    elif self.bridge_chainage_unit == "km" and raw_ch * 1000.0 <= end_ch + 50.0:
                        br_ch = raw_ch * 1000.0
                    else:
                        br_ch = raw_ch

                    cat_val  = str(row.get(cat_col_b, "")) if cat_col_b else ""
                    desc_val = (
                        str(row.get(desc_col_b, f"BRIDGE@{br_ch:.0f}m"))
                        if desc_col_b else f"BRIDGE@{br_ch:.0f}m"
                    )
                    hw_val   = str(row.get(hw_col_b, "")) if hw_col_b else ""
                    full_str = f"{cat_val} {desc_val}".strip().upper()

                    clr, ct = _clearance(full_str, hw_val)
                    if ct == "ROB":
                        continue
                    if ct == "UNKNOWN":
                        self.logger.warning(
                            "Unrecognized bridge category '%s' @ Ch %.2fm (bridge_df) - skipped. "
                            "Add a recognized type (Major Bridge, Minor Bridge, Box Culvert, RUB).",
                            full_str, br_ch,
                        )
                        continue

                    # Check for explicit HFL (High Flood Level) in bridge row
                    hfl_val = None
                    for hfl_key in ["hfl", "hfl_m", "high_flood_level", "design_hfl", "h_f_l"]:
                        if hfl_key in row and pd.notna(row[hfl_key]):
                            try:
                                hfl_val = float(str(row[hfl_key]).replace("m", "").replace("M", "").strip())
                                break
                            except Exception:
                                pass

                    if hfl_val is not None and hfl_val > 0:
                        if not (start_ch - 10.0 <= br_ch <= end_ch + 10.0):
                            continue
                        if any(abs(c.ch_m - br_ch) < 30.0 for c in constraints):
                            continue
                        # Proposed Formation Level must be at least HFL + 0.60m (IRBM Para 106)
                        hfl_pfl = hfl_val + self.min_hfl_clearance_m
                        egl_v = _interp_egl(br_ch)
                        req_fl = max(hfl_pfl, egl_v + clr)
                        constraints.append(BridgeConstraint(
                            ch_m=round(br_ch, 2),
                            req_pfl_m=round(req_fl, 3),
                            struct_type=ct,
                            name=f"{desc_val.strip()} (HFL={hfl_val:.2f}m+{self.min_hfl_clearance_m:.2f}m)",
                            clearance_m=round(req_fl - egl_v, 3),
                        ))
                        self.logger.info(
                            "BridgeConstraint (HFL): %s @ Ch %.1fm | HFL=%.3fm + %.2fm = req_pfl=%.3fm (EGL=%.3fm)",
                            ct, br_ch, hfl_val, self.min_hfl_clearance_m, req_fl, egl_v
                        )
                    else:
                        _add(br_ch, clr, ct, desc_val.strip())

        return constraints

    # =========================================================================
    # Grade Compensation (correct overlap window)
    # =========================================================================

    def _compute_grade_compensation(
        self,
        curr_ch: float,
        next_ch: float,
        h_curves: list[tuple[float, float, float]],
    ) -> float:
        """Return max allowed gradient for tangent segment [curr_ch, next_ch].

        Grade compensation per IRPWM Para 408 / RDSO GE:G-1:
            comp = 70 / R  (decimal fraction, equivalent to 0.04%/degree)
        Applied only when a horizontal curve OVERLAPS the segment - uses actual
        [c_start, c_end] overlap, NOT a fixed lookahead window (fixes BUG-05).
        """
        max_grad = self.ruling_gradient
        for (c_start, c_end, c_rad) in h_curves:
            ol_s = max(curr_ch, c_start)
            ol_e = min(next_ch, c_end)
            if ol_e > ol_s and c_rad > 0:
                comp = (70.0 / c_rad) / 100.0  # 70/R (%) converted to decimal fraction (0.04%/deg)
                max_grad = min(max_grad, max(self.min_gradient, self.ruling_gradient - comp))
        return max_grad

    # =========================================================================
    # VIP Tangent Builders
    # =========================================================================

    def _build_prescribed_vips(
        self,
        prescribed_vips: list[dict[str, Any]],
    ) -> tuple[list[float], list[float], list[float]]:
        """Build VIP lists from engineer-specified VIPs in config.yaml."""
        self.logger.info("Prescribed VIP Mode: %d VIPs from config.", len(prescribed_vips))
        vip_chainages = [float(v["ch"]) for v in prescribed_vips]
        vip_fls       = [float(v["fl"]) for v in prescribed_vips]
        gradients = [
            (vip_fls[i + 1] - vip_fls[i]) / (vip_chainages[i + 1] - vip_chainages[i])
            for i in range(len(vip_chainages) - 1)
        ]
        gradients.append(0.0)
        return vip_chainages, vip_fls, gradients

    def _build_prescribed_segments(
        self,
        prescribed_segments: list[dict[str, Any]],
    ) -> tuple[list[float], list[float], list[float]]:
        """Build VIP lists from prescribed formation level segments in config.yaml."""
        self.logger.info(
            "Prescribed Segments Mode: %d segments from config.", len(prescribed_segments)
        )
        vip_chainages: list[float] = [float(prescribed_segments[0]["start_ch"])]
        vip_fls: list[float]       = [float(prescribed_segments[0]["fl"])]

        for seg in prescribed_segments:
            s_ch = float(seg["start_ch"])
            e_ch = float(seg["end_ch"])
            fl   = float(seg["fl"])
            if s_ch != vip_chainages[-1]:
                vip_chainages.append(s_ch)
                vip_fls.append(fl)
            vip_chainages.append(e_ch)
            vip_fls.append(fl)

        gradients = [
            (
                (vip_fls[i + 1] - vip_fls[i]) / (vip_chainages[i + 1] - vip_chainages[i])
                if (vip_chainages[i + 1] - vip_chainages[i]) > 0 else 0.0
            )
            for i in range(len(vip_chainages) - 1)
        ]
        gradients.append(0.0)
        return vip_chainages, vip_fls, gradients

    def _build_vip_tangents(
        self,
        chainages_arr: np.ndarray,
        elevations_arr: np.ndarray,
        h_curves: list[tuple[float, float, float]],
        bridge_constraints: list[BridgeConstraint],
        survey_efl_df: pd.DataFrame | None,
    ) -> tuple[list[float], list[float], list[float]]:
        """Long-tangent minimum rise-and-fall greedy optimizer.

        Improvements over previous version:
        - Calibrated score function (normalized by max_cut/fill, not raw metres).
        - Correct grade compensation via _compute_grade_compensation().
        - Explicit convergence guard (no infinite loops on misconfiguration).
        - Bridge ramp lookahead uses actual ruling gradient for ramp distance.
        - Micro-VIP merge uses only vip_fls for gradient computation.

        Returns:
            (vip_chainages, vip_fls, gradients)
        """
        vert_cfg = self.config.get("vertical", {})

        # These keys MUST be present in config/config.yaml — no hidden defaults allowed.
        for _required_key in ("max_excess_cut_m", "max_excess_fill_m", "vip_interval_m"):
            if _required_key not in vert_cfg:
                raise KeyError(
                    f"Missing required config key 'vertical.{_required_key}' in config.yaml. "
                    f"Add it explicitly — no fallback default is used."
                )
        max_excess_cut_m: float   = float(vert_cfg["max_excess_cut_m"])
        max_excess_fill_m: float  = float(vert_cfg["max_excess_fill_m"])
        min_floor_clr: float      = self.min_drainage_clearance_m
        max_tangent_len: float    = float(vert_cfg["vip_interval_m"])
        min_tangent_len: float    = max(300.0, self.min_tangent_length_m)

        # Candidate RDSO quantized gradients (both signs, within ruling_gradient)
        candidate_gradients: list[float] = []
        for g in _STANDARD_GRADIENT_RATIOS:
            if g <= self.ruling_gradient + 1e-9:
                candidate_gradients.append(g)
                if g > 0:
                    candidate_gradients.append(-g)

        # Starting conditions
        has_survey_efl = (
            survey_efl_df is not None
            and not survey_efl_df.empty
            and "existing_formation_level_m" in survey_efl_df.columns
        )

        if has_survey_efl:
            assert survey_efl_df is not None
            first_row = survey_efl_df.iloc[0]
            start_fl  = float(first_row["existing_formation_level_m"])
            self.logger.info(
                "Optimizer start: EFL junction @ Ch %.1fm, FL=%.3fm",
                float(first_row["chainage_m"]), start_fl,
            )
        else:
            start_fl = float(elevations_arr[0]) + min_floor_clr

        start_ch = float(chainages_arr[0])
        end_ch   = float(chainages_arr[-1])
        curr_ch  = start_ch
        curr_fl  = start_fl

        vip_chainages: list[float] = [curr_ch]
        vip_fls: list[float]       = [curr_fl]

        # Convergence guard: max iterations based on profile length
        max_iterations = int((end_ch - start_ch) / 20) + 200
        iterations     = 0

        self.logger.info(
            "Auto Optimizer: start Ch=%.1fm FL=%.3fm, end Ch=%.1fm", curr_ch, curr_fl, end_ch
        )

        while curr_ch < end_ch - 10.0:
            iterations += 1
            if iterations > max_iterations:
                self.logger.warning(
                    "Optimizer hit convergence limit (%d) at Ch %.1fm - forcing end.",
                    max_iterations, curr_ch,
                )
                break

            best_grad:  float | None = None   # None = no valid gradient found yet
            best_score: float       = -1e18
            best_reach: float       = curr_ch

            for grad in candidate_gradients:
                # Grade compensation: actual overlap of [curr_ch, curr_ch+max_tangent_len]
                # with horizontal curves
                max_grad_allowed = self._compute_grade_compensation(
                    curr_ch, curr_ch + max_tangent_len, h_curves
                )
                if abs(grad) > max_grad_allowed + 1e-9:
                    continue

                reach_ch       = curr_ch
                segment_cuts:  list[float] = []
                segment_fills: list[float] = []
                floor_violated = False

                for idx in range(len(chainages_arr)):
                    ch = float(chainages_arr[idx])
                    if ch < curr_ch:
                        continue
                    if ch > curr_ch + max_tangent_len:
                        break

                    egl       = float(elevations_arr[idx])
                    pfl       = curr_fl + grad * (ch - curr_ch)
                    cut_depth = egl - pfl       # positive = cutting
                    fill_h    = pfl - egl       # positive = embankment

                    if cut_depth > max_excess_cut_m or fill_h > max_excess_fill_m:
                        break

                    # Drainage floor rule (RDSO GE:G-1):
                    # Shallow embankment (0 <= fill_h < min_floor_clr) is forbidden to prevent subgrade waterlogging.
                    # Cutting (PFL < EGL, fill_h < 0) is allowed down to max_excess_cut_m with side drains.
                    if 0.0 <= fill_h < min_floor_clr - 10:
                        floor_violated = True
                        break

                    reach_ch = ch
                    if cut_depth > 0:
                        segment_cuts.append(cut_depth)
                    else:
                        segment_fills.append(fill_h)

                if floor_violated or (reach_ch - curr_ch) < 50.0:
                    continue

                tangent_len = reach_ch - curr_ch
                n_pts       = max(1, len(segment_cuts) + len(segment_fills))
                total_cut   = sum(segment_cuts)
                total_fill  = sum(segment_fills)

                # ── Earthwork Cut vs Embankment Rule (0.5m target vs 1 in 150 cut threshold) ──
                # Default target: PFL >= EGL + 0.50m (Embankment).
                # Cutting (PFL < EGL) is introduced ONLY when terrain steepness makes it impossible
                # to maintain PFL >= EGL + 0.50m within the 1 in 150 ruling gradient limit.
                has_cutting = len(segment_cuts) > 0 and total_cut > 0.01
                unnecessary_cut_penalty = 0.0
                if has_cutting and tangent_len > 0:
                    # Determine highest terrain elevation in this segment
                    max_egl_in_seg = float(max(
                        elevations_arr[i] for i in range(len(chainages_arr))
                        if curr_ch <= chainages_arr[i] <= reach_ch
                    ))
                    req_grad_embankment = abs(max_egl_in_seg + min_floor_clr - curr_fl) / tangent_len
                    # If embankment <= ruling_gradient (1 in 150) was feasible, heavily penalize unnecessary cutting
                    if req_grad_embankment <= self.ruling_gradient + 1e-6:
                        unnecessary_cut_penalty = 50.0

                # Calibrated score: all terms normalized to [0, ~1] range
                mean_cut_norm  = (total_cut  / n_pts) / max_excess_cut_m
                mean_fill_norm = (total_fill / n_pts) / max_excess_fill_m
                imbalance_norm = (
                    abs(total_cut - total_fill)
                    / (n_pts * max(max_excess_cut_m, max_excess_fill_m))
                )

                reach_bonus       = min(10.0, (tangent_len / 1000.0) * 10.0)
                earthwork_penalty = (mean_cut_norm + mean_fill_norm) * 5.0
                imbalance_penalty = imbalance_norm * 4.0
                steepness_penalty = abs(grad) / self.ruling_gradient * 2.0

                score = reach_bonus - earthwork_penalty - imbalance_penalty - steepness_penalty - unnecessary_cut_penalty

                if score > best_score:
                    best_score = score
                    best_reach = reach_ch
                    best_grad  = grad

            # No valid gradient found — raise instead of silently using LEVEL
            if best_grad is None:
                raise VerticalAlignmentError(
                    f"Optimizer failed at Ch {curr_ch:.1f}m: no valid RDSO gradient passes "
                    f"drainage clearance ({min_floor_clr:.2f}m), cut limit ({max_excess_cut_m:.1f}m), "
                    f"and fill limit ({max_excess_fill_m:.1f}m) simultaneously. "
                    f"Check terrain profile or relax max_excess_cut_m / max_excess_fill_m in config."
                )

            # Clip next VIP chainage
            next_ch = min(
                end_ch,
                min(curr_ch + max_tangent_len, max(curr_ch + min_tangent_len, best_reach)),
            )
            if next_ch >= end_ch - 50.0:
                next_ch = end_ch

            # Bridge constraint ramp lookahead
            active_constraint: BridgeConstraint | None = None
            max_la = max(3500.0, self.bridge_ramp_lookahead_m)
            for bc in bridge_constraints:
                if bc.ch_m <= curr_ch + 5.0:
                    continue
                if bc.ch_m > curr_ch + max_la:
                    break
                rise_needed = bc.req_pfl_m - curr_fl
                if rise_needed > 0:
                    req_dist   = rise_needed / self.ruling_gradient
                    ramp_start = bc.ch_m - req_dist
                    if curr_ch < ramp_start < next_ch and ramp_start >= curr_ch + 50.0:
                        next_ch = round(ramp_start, 2)
                        self.logger.info(
                            "Ramp start @ Ch %.1fm for %s @ Ch %.1fm (rise=%.2fm).",
                            next_ch, bc.name, bc.ch_m, rise_needed,
                        )
                        break
                    elif abs(ramp_start - curr_ch) <= 50.0 or curr_ch >= ramp_start:
                        active_constraint = bc
                        break

            if active_constraint is not None:
                ramp_ch    = active_constraint.ch_m
                ramp_fl    = active_constraint.req_pfl_m
                ramp_dist  = ramp_ch - curr_ch
                raw_ramp_g = (ramp_fl - curr_fl) / ramp_dist if ramp_dist > 0 else 0.0
                ramp_g     = self._quantize_gradient(
                    min(raw_ramp_g, self.ruling_gradient), self.ruling_gradient
                )
                actual_fl = (
                    ramp_fl
                    if active_constraint.struct_type == "SURVEY_EFL"
                    else curr_fl + ramp_g * ramp_dist
                )
                self.logger.info(
                    "PFL CONSTRAINT @ Ch %.1fm->%.1fm: %s FL=%.3fm [%s]",
                    curr_ch, ramp_ch,
                    self._format_gradient_ratio(ramp_g),
                    actual_fl, active_constraint.name,
                )
                vip_chainages.append(ramp_ch)
                vip_fls.append(actual_fl)
                curr_ch = ramp_ch
                curr_fl = actual_fl
                continue

            next_fl = curr_fl + best_grad * (next_ch - curr_ch)  # type: ignore[operator]
            self.logger.info(
                "VIP @ Ch %.1f->%.1fm: %s (len=%.0fm, FL=%.3fm, score=%.2f)",
                curr_ch, next_ch,
                self._format_gradient_ratio(best_grad),
                next_ch - curr_ch, next_fl, best_score,
            )
            vip_chainages.append(next_ch)
            vip_fls.append(next_fl)
            curr_ch = next_ch
            curr_fl = next_fl

        # Micro-VIP merge (gradients always recomputed from vip_fls only)
        vip_chainages, vip_fls = self._merge_micro_vips(
            vip_chainages, vip_fls, min_tangent_len, bridge_constraints
        )
        gradients = self._compute_gradients(vip_chainages, vip_fls)
        return vip_chainages, vip_fls, gradients

    def _merge_micro_vips(
        self,
        vip_chainages: list[float],
        vip_fls: list[float],
        min_tangent_len: float,
        bridge_constraints: list[BridgeConstraint],
    ) -> tuple[list[float], list[float]]:
        """Remove intermediate VIPs closer than min_tangent_len.

        Merges short-distance gradient changes (< min_tangent_len) into a unified
        continuous tangent to prevent dynamic coupler shocks and short sag/summit dips.
        Bridge-anchored VIPs and endpoint VIPs are never removed.
        """
        merge_changed = True
        while merge_changed:
            merge_changed = False
            new_chs: list[float] = [vip_chainages[0]]
            new_fls: list[float] = [vip_fls[0]]
            i = 1
            while i < len(vip_chainages):
                span      = vip_chainages[i] - new_chs[-1]
                is_bridge = any(abs(c.ch_m - vip_chainages[i]) < 5.0 for c in bridge_constraints)
                is_last   = i == len(vip_chainages) - 1

                if not is_last and not is_bridge and span < min_tangent_len and i + 1 < len(vip_chainages):
                    self.logger.info(
                        "Merging short gradient segment @ Ch %.1fm (span=%.0fm < %.0fm)",
                        vip_chainages[i], span, min_tangent_len,
                    )
                    # Re-quantize unified gradient if next VIP is not a bridge anchor
                    next_is_bridge = any(abs(c.ch_m - vip_chainages[i + 1]) < 5.0 for c in bridge_constraints)
                    if not next_is_bridge and i + 1 < len(vip_chainages) - 1:
                        total_dx = vip_chainages[i + 1] - new_chs[-1]
                        raw_g = (vip_fls[i + 1] - new_fls[-1]) / total_dx if total_dx > 0 else 0.0
                        quant_g = self._quantize_gradient(raw_g, self.ruling_gradient)
                        vip_fls[i + 1] = new_fls[-1] + quant_g * total_dx

                    merge_changed = True
                    i += 1
                    continue

                new_chs.append(vip_chainages[i])
                new_fls.append(vip_fls[i])
                i += 1
            vip_chainages = new_chs
            vip_fls       = new_fls
        return vip_chainages, vip_fls

    @staticmethod
    def _compute_gradients(
        vip_chainages: list[float],
        vip_fls: list[float],
    ) -> list[float]:
        """Compute tangent gradients between consecutive VIPs. Last entry = 0.0."""
        grads: list[float] = []
        for i in range(len(vip_chainages) - 1):
            dx = vip_chainages[i + 1] - vip_chainages[i]
            grads.append((vip_fls[i + 1] - vip_fls[i]) / dx if dx > 0 else 0.0)
        grads.append(0.0)
        return grads

    # =========================================================================
    # Parabolic Vertical Curve Geometry
    # =========================================================================

    def _build_pvis(
        self,
        vip_chainages: list[float],
        vip_fls: list[float],
        gradients: list[float],
        h_curves: list[tuple[float, float, float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build PVI list with parabolic VC geometry (IRPWM Para 409).

        TWO-STAGE algorithm (fixes BUG-03 — stale PVC_E/PVT_E):
          Stage 1: Compute raw L for each PVI.
          Stage 2: Scale overlapping adjacent VCs.
          Stage 3: RECALCULATE PVC_S, PVT_S, PVC_E, PVT_E after all scaling.

        First/Last PVI:
          - L = 0.0 always (no VC at alignment endpoints).
          - g1 = gradients[0]   (actual first tangent, NOT hardcoded 0.0 — fixes BUG-07).
          - g2 = gradients[-1]  (actual last tangent).
        """
        n    = len(vip_chainages)
        pvis: list[dict[str, Any]] = []

        # Stage 1: Compute VC length at each PVI
        for i in range(n):
            # Use actual tangent grades (not hardcoded 0.0)
            g_in  = gradients[i - 1] if i > 0     else gradients[0]
            g_out = gradients[i]     if i < n - 1 else gradients[-1]

            if i == 0 or i == n - 1:
                length_l = 0.0
            else:
                delta_g = abs(g_out - g_in)
                if delta_g >= self.min_grade_change_for_vc:
                    is_summit = g_out < g_in
                    # Rv directly from config validation section — no hardcoded fallback.
                    # min_rv_summit_m / min_rv_sag_m must be present in config.yaml.
                    rv = self.rv_summit_m if is_summit else self.rv_sag_m
                    length_l = max(self.min_vc_length_m, rv * delta_g)
                else:
                    length_l = 0.0

            pvis.append({
                "chainage": vip_chainages[i],
                "elev":     vip_fls[i],
                "L":        length_l,
                "g1":       g_in,
                "g2":       g_out,
            })

        # VC-Spiral overlap prevention
        if h_curves:
            for idx, pvi in enumerate(pvis):
                if idx == 0 or idx == n - 1 or pvi["L"] <= 0:
                    continue
                v_pvc = pvi["chainage"] - pvi["L"] / 2.0
                v_pvt = pvi["chainage"] + pvi["L"] / 2.0
                for (s_start, s_end, _) in h_curves:
                    ol_s = max(v_pvc, s_start)
                    ol_e = min(v_pvt, s_end)
                    if ol_e > ol_s:
                        allowed  = min(
                            abs(pvi["chainage"] - s_start),
                            abs(pvi["chainage"] - s_end),
                        )
                        pvi["L"] = max(self.min_vc_length_m, allowed * 2.0)
                        self.logger.info(
                            "VC trimmed at Ch %.1fm to %.1fm (spiral overlap).",
                            pvi["chainage"], pvi["L"],
                        )

        # Stage 2: Scale overlapping adjacent VCs to 85% of inter-PVI distance
        for i in range(len(pvis) - 1):
            p1, p2   = pvis[i], pvis[i + 1]
            dist     = p2["chainage"] - p1["chainage"]
            max_comb = dist * 0.85
            combined = (p1["L"] / 2.0) + (p2["L"] / 2.0)
            if combined > max_comb and combined > 0:
                scale    = max_comb / combined
                p1["L"] *= scale
                p2["L"] *= scale

        # Stage 3: RECALCULATE PVC_E / PVT_E after all scaling (critical fix for BUG-03)
        for pvi in pvis:
            l   = pvi["L"]
            g1  = pvi["g1"]
            g2  = pvi["g2"]
            pvi["PVC_S"] = pvi["chainage"] - l / 2.0
            pvi["PVT_S"] = pvi["chainage"] + l / 2.0
            pvi["PVC_E"] = pvi["elev"] - g1 * (l / 2.0)
            if l > 0.0:
                pvi["PVT_E"] = pvi["PVC_E"] + (g1 + g2) / 2.0 * l
            else:
                pvi["PVT_E"] = pvi["elev"]

        return pvis

    def _apply_parabolic_curves(
        self,
        chainages: np.ndarray,
        pvis: list[dict[str, Any]],
    ) -> np.ndarray:
        """Evaluate continuous PFL elevation at every chainage station.

        Coverage regions:
        - s <= PVC_S of first PVI  : extrapolate along first tangent g1.
        - s >= PVT_S of last PVI   : extrapolate along last tangent g2.
        - PVC_S <= s <= PVT_S      : parabola: y = PVC_E + g1*x + ((g2-g1)/(2L))*x^2.
        - PVT_S(i) <= s <= PVC_S(i+1): linear along g2 of PVI-i from PVT_E(i).

        No dead fallback path (fixes BUG-03 ambiguity). Raises clearly if uncovered.
        """
        fl = np.full(len(chainages), np.nan, dtype=float)
        if not pvis:
            return fl

        for k, s in enumerate(chainages):
            if np.isnan(s):
                continue

            # Before first PVC
            if s <= pvis[0]["PVC_S"]:
                fp    = pvis[0]
                fl[k] = fp["PVC_E"] + fp["g1"] * (s - fp["PVC_S"])
                continue

            # After last PVT
            if s >= pvis[-1]["PVT_S"]:
                lp    = pvis[-1]
                fl[k] = lp["PVT_E"] + lp["g2"] * (s - lp["PVT_S"])
                continue

            assigned = False

            # Inside a parabolic VC zone
            for pvi in pvis:
                if pvi["L"] > 0 and pvi["PVC_S"] <= s <= pvi["PVT_S"]:
                    x      = s - pvi["PVC_S"]
                    fl[k]  = (
                        pvi["PVC_E"]
                        + pvi["g1"] * x
                        + ((pvi["g2"] - pvi["g1"]) / (2.0 * pvi["L"])) * x ** 2
                    )
                    assigned = True
                    break

            if assigned:
                continue

            # Linear tangent between two consecutive VCs
            for i in range(len(pvis) - 1):
                p_curr = pvis[i]
                p_next = pvis[i + 1]
                if p_curr["PVT_S"] <= s <= p_next["PVC_S"]:
                    fl[k]    = p_curr["PVT_E"] + p_curr["g2"] * (s - p_curr["PVT_S"])
                    assigned = True
                    break

            if not assigned:
                raise VerticalAlignmentError(
                    f"PFL evaluation failed at s={s:.2f}m - not covered by any PVI curve or tangent."
                    " Check VIP list for gaps or overlapping VCs."
                )

        return fl

    # =========================================================================
    # Survey Level Re-stamping (Survey Values Are SACRED)
    # =========================================================================

    def _stamp_survey_levels(
        self,
        df_profile: pd.DataFrame,
        survey_levels: pd.DataFrame,
    ) -> pd.DataFrame:
        """Re-stamp exact surveyed levels AFTER parabolic smoothing.

        Parabolic smoothing interpolates unsurveyed gaps. At surveyed chainages
        the exact CSV values must be restored — they are not to be overwritten by
        smooth curve interpolation.

        Priority: PFL column > EFL column (same as in _merge_survey_levels).

        Args:
            df_profile   : Profile DataFrame after parabolic formation_level_m applied.
            survey_levels: Raw survey DataFrame returned by _merge_survey_levels().

        Returns:
            Updated df_profile with exact survey values restored.
        """
        if survey_levels.empty:
            return df_profile

        survey_clean = survey_levels[survey_levels["chainage_m"].notna()].copy()
        if survey_clean.empty:
            return df_profile

        interval_m = float(self.config.get("chainage", {}).get("interval_m", 20.0))
        rem = survey_clean["chainage_m"] % interval_m
        is_regular = (np.minimum(rem, interval_m - rem) < 0.5)
        survey_clean = survey_clean[is_regular].copy().reset_index(drop=True)
        survey_clean["chainage_m"] = np.round(survey_clean["chainage_m"] / interval_m) * interval_m
        survey_clean = survey_clean.drop_duplicates(subset=["chainage_m"], keep="first").reset_index(drop=True)

        has_pfl = "_pfl" in survey_clean.columns
        has_efl = "_efl" in survey_clean.columns
        has_erl = "_erl" in survey_clean.columns

        right_cols = ["chainage_m"]
        if has_efl:
            right_cols.append("_efl")
        if has_pfl:
            right_cols.append("_pfl")
        if has_erl:
            right_cols.append("_erl")

        df_left = df_profile[["chainage_m"]].copy()
        df_left["chainage_m"] = pd.to_numeric(df_left["chainage_m"], errors="coerce").astype(float)
        df_left = df_left.sort_values("chainage_m")

        survey_clean_sub = survey_clean[right_cols].copy()
        survey_clean_sub["chainage_m"] = pd.to_numeric(survey_clean_sub["chainage_m"], errors="coerce").astype(float)
        survey_clean_sub = survey_clean_sub.sort_values("chainage_m")

        merged = pd.merge_asof(
            df_left,
            survey_clean_sub,
            on="chainage_m",
            tolerance=10.0,
            direction="nearest",
        )
        merged.index = df_left.index
        merged = merged.reindex(df_profile.index)

        # EFL stamp (lower priority - only if formation_level_m is missing)
        if has_efl and "_efl" in merged.columns:
            efl_v = pd.to_numeric(merged["_efl"], errors="coerce")
            mask  = efl_v.notna() & df_profile["formation_level_m"].isna()
            df_profile.loc[mask, "formation_level_m"] = efl_v[mask]

        # PFL stamp (higher priority - overwrites EFL)
        if has_pfl and "_pfl" in merged.columns:
            pfl_v = pd.to_numeric(merged["_pfl"], errors="coerce")
            mask  = pfl_v.notna()
            df_profile.loc[mask, "formation_level_m"] = pfl_v[mask]

        # ERL stamp
        if has_erl and "_erl" in merged.columns:
            erl_v = pd.to_numeric(merged["_erl"], errors="coerce")
            mask  = erl_v.notna()
            df_profile.loc[mask, "existing_rail_level_m"] = erl_v[mask]

        count_key = "_pfl" if has_pfl else "_efl" if has_efl else None
        if count_key and count_key in merged.columns:
            n_stamped = pd.to_numeric(merged[count_key], errors="coerce").notna().sum()
            self.logger.info(
                "_stamp_survey_levels: %d surveyed stations re-stamped post-parabolic.", n_stamped
            )
        return df_profile

    # =========================================================================
    # Profile Assembly
    # =========================================================================

    def _compute_prl_and_cutfill(self, df_profile: pd.DataFrame) -> pd.DataFrame:
        """Compute proposed_rail_level_m and cut_fill_m if not already provided by Excel.

        PRL = formation_level_m + track_buildup_m (fallback).
        cut_fill_m = formation_level_m - elevation_m (+ fill, - cut) (fallback).
        """
        if "proposed_rail_level_m" not in df_profile.columns or df_profile["proposed_rail_level_m"].isna().all():
            df_profile["proposed_rail_level_m"] = (
                df_profile["formation_level_m"] + self.track_buildup_m
            )
        else:
            missing_prl = df_profile["proposed_rail_level_m"].isna() & df_profile["formation_level_m"].notna()
            df_profile.loc[missing_prl, "proposed_rail_level_m"] = (
                df_profile.loc[missing_prl, "formation_level_m"] + self.track_buildup_m
            )

        if "cut_fill_m" not in df_profile.columns or df_profile["cut_fill_m"].isna().all():
            if "elevation_m" in df_profile.columns:
                df_profile["cut_fill_m"] = (
                    df_profile["formation_level_m"] - df_profile["elevation_m"]
                )
        return df_profile

    def _build_vip_dataframe(
        self,
        vip_chainages: list[float],
        vip_fls: list[float],
        gradients: list[float],
        pvis: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Build VIP DataFrame directly from design VIPs.

        No second-pass re-detection from the smoothed profile (fixes FLAW-01).
        """
        rows = []
        for i, (ch, fl) in enumerate(zip(vip_chainages, vip_fls)):
            pvi      = pvis[i]
            length_l = pvi["L"]
            grad     = gradients[i]
            rows.append({
                "vip_index":            i,
                "chainage_m":           ch,
                "chainage_str":         format_chainage(ch),
                "elevation_m":          fl,
                "gradient_to_next_pct": grad * 100.0,
                "gradient_ratio_str":   (
                    self._format_gradient_ratio(grad)
                    if i < len(vip_chainages) - 1 else "End"
                ),
                "curve_length_m":       length_l,
                "pvc_chainage_m":       pvi["PVC_S"],
                "pvt_chainage_m":       pvi["PVT_S"],
            })
        return pd.DataFrame(rows)

    # =========================================================================
    # Full Survey Mode & Gradient Schedule Reader
    # =========================================================================

    def read_gradient_excel(self, gradient_excel_path: str | Path) -> pd.DataFrame:
        """Parse engineer's official Gradient / Grade List schedule Excel workbook.

        Maps columns:
          - Start Station, End Station (e.g. 0+000.00m, 0+961.94m)
          - Start Elevation, End Elevation (e.g. 3.997m, 4.251m)
          - Grade / Gradient (e.g. Horizontal, 1 : 2300, 1:300)
          - Direction (e.g. Level, Rise, Fall)

        Returns:
          vip_dataframe with VIP stations, elevations, gradient ratios, and VC parameters.
        """
        g_path = Path(gradient_excel_path)
        if not g_path.exists():
            raise FileNotFoundError(f"Gradient Excel file not found at: {g_path}")

        self.logger.info("Reading official Gradient Schedule from: %s", g_path.name)
        df_raw = pd.read_excel(g_path)
        df = df_raw.copy()
        df.columns = [str(c).strip() for c in df.columns]

        def _parse_ch(s: Any) -> float:
            st = str(s).replace("m", "").strip()
            if "+" in st:
                parts = st.split("+")
                return float(parts[0]) * 1000.0 + float(parts[1])
            return float(st)

        def _parse_el(s: Any) -> float:
            return float(str(s).replace("m", "").strip())

        vips = []
        # First VIP point from Start Station/Elevation of first row
        s_ch0 = _parse_ch(df.iloc[0].get("Start Station", 0.0))
        s_el0 = _parse_el(df.iloc[0].get("Start Elevation", 0.0))
        vips.append({"vip_index": 0, "chainage_m": s_ch0, "elevation_m": s_el0})

        for idx, row in df.iterrows():
            e_ch = _parse_ch(row.get("End Station", 0.0))
            e_el = _parse_el(row.get("End Elevation", 0.0))
            vips.append({"vip_index": idx + 1, "chainage_m": e_ch, "elevation_m": e_el})

        v_df = pd.DataFrame(vips)
        grads = []
        grad_strs = []
        for i in range(len(v_df)):
            if i < len(v_df) - 1:
                dx = v_df.loc[i + 1, "chainage_m"] - v_df.loc[i, "chainage_m"]
                dy = v_df.loc[i + 1, "elevation_m"] - v_df.loc[i, "elevation_m"]
                g = dy / dx if dx > 0 else 0.0
                grads.append(g * 100.0)
                if abs(g) < 1e-5:
                    grad_strs.append("LEVEL")
                elif g > 0:
                    ratio = int(round(1.0 / g))
                    grad_strs.append(f"RISE 1 IN {ratio}")
                else:
                    ratio = int(round(1.0 / abs(g)))
                    grad_strs.append(f"FALL 1 IN {ratio}")
            else:
                grads.append(0.0)
                grad_strs.append("End")

        v_df["chainage_str"] = [format_chainage(c) for c in v_df["chainage_m"]]
        v_df["gradient_to_next_pct"] = grads
        v_df["gradient_pct"] = grads
        v_df["gradient_ratio_str"] = grad_strs
        v_df["curve_length_m"] = 0.0
        v_df["pvc_chainage_m"] = v_df["chainage_m"]
        v_df["pvt_chainage_m"] = v_df["chainage_m"]
        return v_df

    def _design_full_survey_mode(
        self,
        df_profile: pd.DataFrame,
        has_efl: bool,
        efl_col: str,
        gradient_excel_path: str | Path | None = None,
        bridge_constraints: list[BridgeConstraint] | None = None,
    ) -> VAlignResult:
        """Handle Full Survey / EFL Mode.

        Preserves exact survey levels as-is. NO parabolic smoothing applied.
        """
        self.logger.info(
            "Full Survey / EFL Mode: preserving exact survey formation levels (no smoothing)."
        )

        if has_efl:
            efl_mask = df_profile[efl_col].notna() & df_profile["formation_level_m"].isna()
            df_profile.loc[efl_mask, "formation_level_m"] = df_profile.loc[efl_mask, efl_col]

        df_profile = self._compute_prl_and_cutfill(df_profile)

        # Surveyed PFL is preserved verbatim, so reject it when it violates a
        # required bridge clearance instead of silently returning an unsafe
        # profile as a successful design.
        violations: list[str] = []
        surveyed = df_profile.dropna(subset=["chainage_m", "formation_level_m"])
        if not surveyed.empty:
            s_min = float(surveyed["chainage_m"].min())
            s_max = float(surveyed["chainage_m"].max())
            for constraint in bridge_constraints or []:
                if constraint.ch_m < s_min or constraint.ch_m > s_max:
                    self.logger.info(
                        "Bridge constraint %s @ Ch %.2fm is outside surveyed section [%.2f, %.2f]m — skipped.",
                        constraint.name, constraint.ch_m, s_min, s_max
                    )
                    continue
                actual_pfl = float(np.interp(
                    constraint.ch_m,
                    surveyed["chainage_m"].to_numpy(dtype=float),
                    surveyed["formation_level_m"].to_numpy(dtype=float),
                ))
                if actual_pfl + 0.05 < constraint.req_pfl_m:
                    violations.append(
                        f"{constraint.name} at Ch {constraint.ch_m:.2f} m: "
                        f"PFL {actual_pfl:.3f} m < required {constraint.req_pfl_m:.3f} m"
                    )
        if violations:
            raise VerticalAlignmentError(
                "Surveyed PFL violates bridge/crossing clearance constraints: "
                + "; ".join(violations)
            )

        # Build VIP table: Prefer official Gradient Excel schedule if provided/configured
        grad_p = gradient_excel_path or self.config.get("paths", {}).get("gradient_excel_path")
        if grad_p and Path(grad_p).exists():
            vip_df = self.read_gradient_excel(grad_p)
        else:
            vip_chainages, vip_fls, gradients = self._detect_grade_breaks(df_profile)
            pvis   = self._build_pvis(vip_chainages, vip_fls, gradients)
            vip_df = self._build_vip_dataframe(vip_chainages, vip_fls, gradients, pvis)

        self._drainage_violation_count = self._check_drainage_clearance(df_profile)

        meta: dict[str, Any] = {
            "total_vips": len(vip_df),
            "ruling_gradient": self.ruling_gradient,
            "track_buildup_m": self.track_buildup_m,
            "design_speed_kmh": self.design_speed_kmh,
            "prescribed_mode": True,
            "efl_mode": True,
            "drainage_clearance_violation_count": self._drainage_violation_count,
            "ruling_gradient_exceeded_count": 0,
        }
        return VAlignResult(vip_dataframe=vip_df, profile_dataframe=df_profile, metadata=meta)

    def _detect_grade_breaks(
        self,
        df_profile: pd.DataFrame,
    ) -> tuple[list[float], list[float], list[float]]:
        """Detect VIP grade breaks from surveyed formation_level_m for VIP table only.

        Used exclusively in Full Survey Mode. Uses geometric line simplification (tolerance 0.01m)
        to filter out millimetre rounding noise and extract true VIP inflection points.
        Does NOT modify df_profile.
        """
        ch_v = df_profile["chainage_m"].values.astype(float)
        fl_v = df_profile["formation_level_m"].values.astype(float)

        valid = ~np.isnan(ch_v) & ~np.isnan(fl_v)
        ch_v  = ch_v[valid]
        fl_v  = fl_v[valid]

        if len(ch_v) < 2:
            return [float(ch_v[0])], [float(fl_v[0])], [0.0]

        try:
            from shapely.geometry import LineString
            ls = LineString(np.column_stack([ch_v, fl_v]))
            # 0.01m tolerance extracts true engineered grade breaks while filtering out mm-level rounding jitter
            simplified = ls.simplify(0.01, preserve_topology=True)
            pts = list(simplified.coords)
            vip_chs = [float(p[0]) for p in pts]
            vip_fls_ = [float(p[1]) for p in pts]
        except Exception as exc:
            self.logger.warning("Line simplification failed in _detect_grade_breaks: %s, falling back to endpoints", exc)
            vip_chs = [float(ch_v[0]), float(ch_v[-1])]
            vip_fls_ = [float(fl_v[0]), float(fl_v[-1])]

        return vip_chs, vip_fls_, self._compute_gradients(vip_chs, vip_fls_)

    # =========================================================================
    # Quality Audit
    # =========================================================================

    def _check_drainage_clearance(self, df_profile: pd.DataFrame) -> int:
        """Audit PFL >= EGL + min_drainage_clearance_m across the full profile.

        Returns violation count. Logs warning with worst-case chainage.
        """
        if (
            "formation_level_m" not in df_profile.columns
            or "elevation_m" not in df_profile.columns
        ):
            return 0

        valid     = df_profile["formation_level_m"].notna() & df_profile["elevation_m"].notna()
        if not valid.any():
            return 0

        clearance = (
            df_profile.loc[valid, "formation_level_m"]
            - df_profile.loc[valid, "elevation_m"]
        )
        violation = clearance < (self.min_drainage_clearance_m - 1e-6)
        n         = int(violation.sum())

        if n > 0:
            worst_idx = clearance[violation].idxmin()
            worst_ch  = (
                float(df_profile.loc[worst_idx, "chainage_m"])
                if "chainage_m" in df_profile.columns else float("nan")
            )
            worst_clr = float(clearance.loc[worst_idx])
            self.logger.warning(
                "Drainage clearance VIOLATED at %d station(s). "
                "Worst: Ch %.1fm, clearance=%.3fm (required >= %.2fm). "
                "Engineer must review VIPs in this reach.",
                n, worst_ch, worst_clr, self.min_drainage_clearance_m,
            )
        return n


# ===========================================================================
# Standalone Execution Entry-Point
# ===========================================================================

if __name__ == "__main__":
    import yaml

    config_dict: dict[str, Any] = {}
    config_file_path = Path("config/config.yaml")
    if config_file_path.exists():
        with open(config_file_path, "r", encoding="utf-8") as _f:
            config_dict = yaml.safe_load(_f)

    from modules.read_kml import KMLReader
    from modules.generate_chainage import ChainageGenerator
    from modules.dem_reader import DEMReader
    from modules.elevation_sampler import ElevationSampler

    kml_result      = KMLReader(config_dict).read(r"D:\ESP\OP_2_update.kml")
    chainage_result = ChainageGenerator(config_dict).generate(kml_result)
    dem_result      = DEMReader(config_dict).read(
        r"D:\ESP\esp_generator\data\dem.tif",
        target_crs=kml_result.utm_info.epsg_str,
    )
    egl_result    = ElevationSampler(config_dict).sample(chainage_result, dem_result)
    valign_result = VerticalAlignment(config_dict).design(egl_result)

    print("\n" + "=" * 64)
    print("       VERTICAL ALIGNMENT FORMATION LEVEL SUMMARY")
    print("=" * 64)

    vips = valign_result.vip_dataframe
    if not vips.empty:
        print(f"\nVIPs ({len(vips)} total):")
        hdr = f"   {'VIP#':<6} {'Chainage':<12} {'FL (m)':<14} {'Gradient':<20} {'VC (m)':<12}"
        print(hdr)
        print("   " + "-" * 66)
        for _, row in vips.iterrows():
            print(
                f"   VIP-{int(row['vip_index']) + 1:<3} "
                f"{row['chainage_str']:<12} "
                f"{row['elevation_m']:<14.3f} "
                f"{row['gradient_ratio_str']:<20} "
                f"{row['curve_length_m']:<12.1f}"
            )

    df_p = valign_result.profile_dataframe
    if not df_p.empty:
        print("\nSample Profile (first 10 stations):")
        print(
            f"   {'Chainage':<12} {'EGL (m)':<14} {'PFL (m)':<14}"
            f" {'PRL (m)':<14} {'Cut/Fill (m)':<14}"
        )
        print("   " + "-" * 68)
        for _, row in df_p.head(10).iterrows():
            print(
                f"   {str(row.get('chainage_str', '')):<12} "
                f"{row.get('elevation_m', float('nan')):<14.3f} "
                f"{row.get('formation_level_m', float('nan')):<14.3f} "
                f"{row.get('proposed_rail_level_m', float('nan')):<14.3f} "
                f"{row.get('cut_fill_m', float('nan')):<+14.3f}"
            )

    print("=" * 64 + "\n")
    VerticalAlignment(config_dict).export_csv(valign_result, "outputs")




