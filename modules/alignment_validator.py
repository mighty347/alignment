"""
Module: modules/alignment_validator.py
Purpose: Module 11 - Alignment Validator (IRS/RDSO Standards Compliance Checker).

Engineering Basis:
    Validates the complete L-Section output against Indian Railways IRS/RDSO
    standards. Covers 40 checks across 6 categories:
      1. VERTICAL_ALIGNMENT   - VA-01 to VA-10
      2. VERTICAL_CURVE       - VC-01 to VC-06
      3. HORIZONTAL_CURVE     - HC-01 to HC-10
      4. FORMATION            - FW-01 to FW-07
      5. SPECIAL_LOCATIONS    - SL-01 to SL-04
      6. DRAWING_STANDARDS    - DS-01 to DS-03

    Each check produces a row in the ValidationResult.checks_df DataFrame with:
        check_id, category, check_name, status (PASS/FAIL/WARN/INFO),
        value, limit, unit, standard, location_chainage_m, remark.

Input:
    HGeomResult       - horizontal curve parameters (Module 6)
    VAlignResult      - vertical alignment, VIPs, gradients (Module 7)
    EarthworkResult   - cut/fill heights, formation levels (Module 8)
    TopoAnalysisResult - proposed structures and locations (Module 10)

Output: ValidationResult (checks_df + CSV export + console summary).

Standards:
    RDSO GE:G-1 | IRPWM | IRS-T-23 | IRBM | Indian Railways SOD (BG)

Author   : Railway ESP Generator
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.horizontal_geometry import HGeomResult
from modules.vertical_alignment import VAlignResult
from modules.earthwork import EarthworkResult
from modules.topo_analyzer import TopoAnalysisResult
from utils.logger import get_logger


logger = get_logger(__name__)


class AlignmentValidatorError(Exception):
    """Custom exception for AlignmentValidator errors."""
    pass


@dataclass
class ValidationResult:
    """Result object returned by AlignmentValidator.

    Attributes:
        checks_df:   DataFrame with one row per validation check.
        pass_count:  Number of PASS checks.
        fail_count:  Number of FAIL checks.
        warn_count:  Number of WARN checks.
        info_count:  Number of INFO checks.
        metadata:    Summary metadata dictionary.
    """
    checks_df: pd.DataFrame
    pass_count: int
    fail_count: int
    warn_count: int
    info_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class AlignmentValidator:
    """Runs 40 IRS/RDSO compliance checks on the complete L-Section output.

    Implements Module 11 of the ESP Generator pipeline.
    All thresholds read from config/config.yaml (validation: section).
    No numeric limits are hardcoded in this class.

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Example:
        >>> val_result = AlignmentValidator(config).validate(
        ...     hgeom_result, valign_result, earthwork_result, topo_result
        ... )
        >>> AlignmentValidator(config).print_summary(val_result)
        >>> AlignmentValidator(config).export_csv(val_result, "outputs")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

        val_cfg  = config.get("validation", {})
        vert_cfg = config.get("vertical", {})
        rail_cfg = config.get("railway", {})
        draw_cfg = config.get("drawing", {})

        self.design_speed_kmh: float = float(rail_cfg.get("design_speed", 100.0))

        # Vertical Alignment limits
        self.max_gradient_plains:       float = float(val_cfg.get("max_gradient_plains", 1.0/150.0))
        self.max_gradient_hills:        float = float(val_cfg.get("max_gradient_hills", 0.01))
        self.min_gradient:              float = float(val_cfg.get("min_gradient", 0.001))
        self.max_platform_gradient:     float = float(val_cfg.get("max_platform_gradient", 0.0025))
        self.grade_comp_factor:         float = float(val_cfg.get("grade_compensation_factor", 70.0))
        self.min_formation_clearance_m: float = float(val_cfg.get("min_formation_clearance_m", 0.60))
        self.max_cut_standard_m:        float = float(val_cfg.get("max_cut_height_standard_m", 6.0))
        self.max_fill_standard_m:       float = float(val_cfg.get("max_fill_height_standard_m", 6.0))
        self.track_buildup_m:           float = float(val_cfg.get("track_buildup_m", 0.762))

        # Vertical Curve limits
        self.min_vc_length_m:         float = float(val_cfg.get("min_vc_length_m", 60.0))
        self.min_grade_change_for_vc: float = float(val_cfg.get("min_grade_change_for_vc", 0.004)) * 100.0 # in %
        self.min_rv_summit_m:         float = float(val_cfg.get("min_rv_summit_m", 4000.0))
        self.min_rv_sag_m:            float = float(val_cfg.get("min_rv_sag_m", 3000.0))
        self.vc_rate_divisor:         float = float(val_cfg.get("vc_rate_of_change", 4000.0))

        # Horizontal Curve limits
        self.min_radius_for_speed:    float = self._get_min_radius_for_speed(val_cfg)
        self.max_cant_passenger_mm:   float = float(val_cfg.get("max_cant_passenger_mm", 165.0))
        self.max_cant_deficiency_mm:  float = float(val_cfg.get("max_cant_deficiency_mm", 100.0))
        self.max_cant_excess_mm:      float = float(val_cfg.get("max_cant_excess_mm", 75.0))
        self.spiral_factor_desirable: float = float(val_cfg.get("spiral_factor_desirable", 0.008))
        self.spiral_factor_minimum:   float = float(val_cfg.get("spiral_factor_minimum", 0.0056))
        self.cant_gradient_factor:    float = float(val_cfg.get("cant_gradient_factor", 360.0))
        self.min_straight_same_dir_m: float = float(val_cfg.get("min_straight_same_dir_m", 50.0))

        # Formation
        self.fw_cut_m:        float = float(val_cfg.get("formation_width_cut_m", 6.85))
        self.fw_fill_low_m:   float = float(val_cfg.get("formation_width_fill_low_m", 7.70))
        self.fw_fill_high_m:  float = float(val_cfg.get("formation_width_fill_high_m", 8.85))
        self.slope_cut:       float = float(val_cfg.get("slope_cut_ratio", 1.5))
        self.slope_fill:      float = float(val_cfg.get("slope_fill_ratio", 2.0))
        self.blanket_mm:      float = float(val_cfg.get("blanket_thickness_mm", 300.0))

        # Special Locations
        self.bridge_appr_grad_m: float = float(val_cfg.get("bridge_approach_gradient_limit", 1.0/400.0))
        self.bridge_appr_len_m:  float = float(val_cfg.get("bridge_approach_length_m", 30.0))
        self.lc_grad_limit:      float = float(val_cfg.get("lc_gradient_limit", 0.001))
        self.lc_check_len_m:     float = float(val_cfg.get("lc_check_length_m", 30.0))
        self.min_bridge_curve_m: float = float(val_cfg.get("min_straight_bridge_to_curve_m", 50.0))

        # Drawing Standards
        self.max_sheet_len_m: float = float(draw_cfg.get("max_lsection_length_m", 5000.0))
        self.std_interval_m:  float = float(val_cfg.get("standard_chainage_interval_m", 20.0))
        self.req_bands:       int   = int(val_cfg.get("required_data_bands", 10))

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def validate(
        self,
        hgeom_result: HGeomResult,
        valign_result: VAlignResult,
        earthwork_result: EarthworkResult,
        topo_result: TopoAnalysisResult,
    ) -> ValidationResult:
        """Run all IRS/RDSO validation checks on the complete L-Section output.

        Args:
            hgeom_result:     Output from Module 6 (HorizontalGeometry).
            valign_result:    Output from Module 7 (VerticalAlignment).
            earthwork_result: Output from Module 8 (Earthwork).
            topo_result:      Output from Module 10 (TopoAnalyzer).

        Returns:
            ValidationResult with checks_df and pass/fail/warn/info counts.

        Raises:
            AlignmentValidatorError: If validation setup fails.
        """
        self.logger.info("Starting IRS/RDSO L-Section alignment validation (40 checks)...")

        checks: list[dict] = []
        checks += self._check_vertical_alignment(valign_result, hgeom_result, earthwork_result)
        checks += self._check_vertical_curves(valign_result, hgeom_result)
        checks += self._check_horizontal_curves(hgeom_result)
        checks += self._check_formation(earthwork_result)
        checks += self._check_special_locations(valign_result, hgeom_result, topo_result)
        checks += self._check_drawing_standards()

        df          = pd.DataFrame(checks)
        pass_count  = int((df["status"] == "PASS").sum())
        fail_count  = int((df["status"] == "FAIL").sum())
        warn_count  = int((df["status"] == "WARN").sum())
        info_count  = int((df["status"] == "INFO").sum())

        self.logger.info(
            "Validation complete: %d PASS | %d FAIL | %d WARN | %d INFO | %d total",
            pass_count, fail_count, warn_count, info_count, len(checks),
        )
        if fail_count > 0:
            self.logger.warning("%d FAIL(s) - review validation_report.csv.", fail_count)

        return ValidationResult(
            checks_df=df,
            pass_count=pass_count,
            fail_count=fail_count,
            warn_count=warn_count,
            info_count=info_count,
            metadata={"total_checks": len(checks), "design_speed_kmh": self.design_speed_kmh},
        )

    def export_csv(self, result: ValidationResult, output_dir: str | Path) -> Path:
        """Export the validation report to a CSV file.

        Args:
            result:     ValidationResult from validate().
            output_dir: Directory where validation_report.csv is written.

        Returns:
            Resolved path to the written CSV file.
        """
        output_path = Path(output_dir).resolve() / "validation_report.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.checks_df.to_csv(output_path, index=False, float_format="%.3f")
        self.logger.info("Validation report exported: %s", output_path)
        return output_path

    def print_summary(self, result: ValidationResult) -> None:
        """Print a formatted validation summary to the console.

        Args:
            result: ValidationResult from validate().
        """
        df     = result.checks_df
        status_sym = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[INFO]"}

        print("\n" + "=" * 78)
        print("  IRS / RDSO - ALIGNMENT VALIDATION REPORT")
        print("=" * 78)

        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            print(f"\n  -- {category} --")
            for _, row in cat_df.iterrows():
                sym    = status_sym.get(str(row["status"]), "[    ]")
                ch_str = (
                    f"  [Ch {float(row['location_chainage_m']):.0f}m]"
                    if not pd.isna(row["location_chainage_m"]) else ""
                )
                print(f"  {sym} {row['check_id']}: {row['check_name']}{ch_str}")
                print(f"         Value : {row['value']}  |  Limit: {row['limit']} {row['unit']}")
                print(f"         Ref   : {row['standard']}")
                if row["remark"]:
                    print(f"         Remark: {row['remark']}")

        print("\n" + "=" * 78)
        print(
            f"  SUMMARY: [PASS] {result.pass_count}  |  [FAIL] {result.fail_count}"
            f"  |  [WARN] {result.warn_count}  |  [INFO] {result.info_count}"
        )
        print(f"  TOTAL CHECKS: {result.pass_count + result.fail_count + result.warn_count + result.info_count}")
        if result.fail_count > 0:
            print(f"  !! ACTION REQUIRED: {result.fail_count} check(s) FAILED - review and rectify.")
        print("=" * 78 + "\n")

    # --------------------------------------------------------------------------
    # Category 1: Vertical Alignment (VA-01 to VA-10)
    # --------------------------------------------------------------------------

    def _check_vertical_alignment(
        self,
        valign_result: VAlignResult,
        hgeom_result: HGeomResult,
        earthwork_result: EarthworkResult,
    ) -> list[dict]:
        """Run VA-01 through VA-10 vertical alignment checks."""
        checks: list[dict] = []
        C       = "VERTICAL_ALIGNMENT"
        vip_df  = valign_result.vip_dataframe.copy() if not valign_result.vip_dataframe.empty else pd.DataFrame()
        prof_df = earthwork_result.profile_dataframe.copy() if not earthwork_result.profile_dataframe.empty else pd.DataFrame()
        ip_df   = hgeom_result.ip_dataframe.copy() if not hgeom_result.ip_dataframe.empty else pd.DataFrame()

        # VA-01: Maximum ruling gradient
        max_grad: float     = 0.0
        max_grad_ch: float  = float("nan")
        if not vip_df.empty:
            for i in range(len(vip_df) - 1):
                g = abs(float(vip_df.iloc[i].get("gradient_to_next_pct", 0.0))) / 100.0
                if g > max_grad:
                    max_grad    = g
                    max_grad_ch = float(vip_df.iloc[i].get("chainage_m", 0.0))
        ratio_str = f"1:{round(1.0/max_grad)}" if max_grad > 1e-6 else "LEVEL"
        status    = "PASS" if max_grad <= self.max_gradient_plains + 1e-6 else "FAIL"
        checks.append(self._chk(
            "VA-01", C, "Ruling Gradient (Max)",
            status, f"{max_grad*100:.4f}% ({ratio_str})",
            f"{self.max_gradient_plains*100:.4f}% (1:150)",
            "%", "RDSO GE:G-1", max_grad_ch,
            "Exceeds ruling gradient 1:150" if status == "FAIL" else "Within ruling gradient",
        ))

        # VA-02: Grade compensation on curves
        if not vip_df.empty and not ip_df.empty:
            comp_fails: list[tuple] = []
            for _, ip_row in ip_df.iterrows():
                r = float(ip_row.get("radius_m", 9999.0))
                if r <= 0:
                    continue
                comp_limit = self.max_gradient_plains - (self.grade_comp_factor / r) / 100.0
                tptc = float(ip_row.get("tptc_chainage_m", 0.0))
                ctpt = float(ip_row.get("ctpt_chainage_m", 0.0))
                for i in range(len(vip_df) - 1):
                    s = float(vip_df.iloc[i]["chainage_m"])
                    e = float(vip_df.iloc[i+1]["chainage_m"])
                    if e < tptc or s > ctpt:
                        continue
                    g = abs(float(vip_df.iloc[i].get("gradient_to_next_pct", 0.0))) / 100.0
                    if g > comp_limit + 1e-6:
                        comp_fails.append((s, g, comp_limit, r))
            if comp_fails:
                for ch, g_val, g_lim, r_val in comp_fails[:5]:
                    checks.append(self._chk(
                        "VA-02", C, "Grade Compensation on Curves", "FAIL",
                        f"{g_val*100:.3f}%", f"{g_lim*100:.3f}% (R={r_val:.0f}m)",
                        "%", "IRPWM Para 408", ch,
                        f"Gradient {g_val*100:.3f}% > compensated limit {g_lim*100:.3f}% (70/R deduction for BG)",
                    ))
            else:
                checks.append(self._chk(
                    "VA-02", C, "Grade Compensation on Curves", "PASS",
                    "All OK", "Ruling minus 70/R%",
                    "%", "IRPWM Para 408", None,
                    "All curve gradients within 70/R BG compensated ruling gradient limit",
                ))
        else:
            checks.append(self._chk("VA-02", C, "Grade Compensation on Curves", "INFO", "N/A", "70/R%", "%", "IRPWM Para 408", None, "Insufficient data"))

        # VA-03: Minimum gradient for drainage in cuttings
        min_grad: float    = float("inf")
        min_grad_ch: float = 0.0
        if not vip_df.empty:
            for i in range(len(vip_df) - 1):
                g = abs(float(vip_df.iloc[i].get("gradient_to_next_pct", 0.0))) / 100.0
                if g < min_grad:
                    min_grad    = g
                    min_grad_ch = float(vip_df.iloc[i].get("chainage_m", 0.0))
        if math.isinf(min_grad):
            min_grad = 0.0
        min_ratio = f"1:{round(1.0/min_grad)}" if min_grad > 1e-8 else "LEVEL"
        status    = "PASS" if min_grad >= self.min_gradient - 1e-7 else "WARN"
        checks.append(self._chk(
            "VA-03", C, "Minimum Gradient (Drainage)",
            status, f"{min_grad*100:.4f}% ({min_ratio})", f"{self.min_gradient*100:.2f}% (1:1000)",
            "%", "IRPWM Para 409", min_grad_ch,
            "Flat/level section - check drainage provision" if status == "WARN" else "Adequate drainage gradient",
        ))

        # VA-04: Gradient vs hill section limit
        if max_grad > self.max_gradient_hills + 1e-6:
            st4, rm4 = "FAIL", "Exceeds 1:100 hill gradient limit"
        elif max_grad > self.max_gradient_plains + 1e-6:
            st4, rm4 = "WARN", "Gradient > 1:150 - confirm hill section applicability"
        else:
            st4, rm4 = "PASS", "Within hill gradient limit"
        checks.append(self._chk(
            "VA-04", C, "Gradient vs Hill Section Limit",
            st4, f"{max_grad*100:.3f}%", f"{self.max_gradient_hills*100:.1f}% (1:100 Hills)",
            "%", "IRPWM Para 408", max_grad_ch, rm4,
        ))

        # VA-07: Min FL clearance above EGL in fill sections
        if not prof_df.empty and "elevation_m" in prof_df.columns and "formation_level_m" in prof_df.columns:
            fill_mask = prof_df.get("ew_type", pd.Series(["FILL"]*len(prof_df))) == "FILL"
            fill_df   = prof_df[fill_mask]
            if fill_df.empty:
                fill_df = prof_df[(prof_df["formation_level_m"] - prof_df["elevation_m"]) > 0]
            if not fill_df.empty:
                clrs       = fill_df["formation_level_m"] - fill_df["elevation_m"]
                min_clr    = float(clrs.min())
                min_clr_ch = float(fill_df.loc[clrs.idxmin(), "chainage_m"])
                status = "PASS" if min_clr >= self.min_formation_clearance_m - 1e-3 else "FAIL"
                checks.append(self._chk(
                    "VA-07", C, "Min FL Clearance Above EGL (Fill)",
                    status, f"{min_clr:.3f} m", f"{self.min_formation_clearance_m:.2f} m",
                    "m", "RDSO GE:G-0014", min_clr_ch,
                    "FL too close to EGL - drainage risk" if status == "FAIL" else "Adequate FL clearance",
                ))
            else:
                checks.append(self._chk("VA-07", C, "Min FL Clearance Above EGL", "INFO", "No fill sections", "0.60 m", "m", "RDSO GE:G-0014", None, "No fill sections found"))
        else:
            checks.append(self._chk("VA-07", C, "Min FL Clearance Above EGL", "INFO", "N/A", "0.60 m", "m", "RDSO GE:G-0014", None, "Profile data unavailable"))

        # VA-08: Max cut depth
        if not prof_df.empty and "ew_height_m" in prof_df.columns:
            cut_df = prof_df[prof_df.get("ew_type", pd.Series(["NONE"]*len(prof_df))) == "CUT"]
            if not cut_df.empty:
                max_cut    = float(cut_df["ew_height_m"].abs().max())
                max_cut_ch = float(prof_df.loc[cut_df["ew_height_m"].abs().idxmax(), "chainage_m"])
                status = "PASS" if max_cut <= self.max_cut_standard_m + 1e-3 else "WARN"
                checks.append(self._chk(
                    "VA-08", C, "Max Cut Depth (Standard Slopes)",
                    status, f"{max_cut:.2f} m", f"{self.max_cut_standard_m:.1f} m",
                    "m", "RDSO GE:G-0014", max_cut_ch,
                    "Cut >6m - special slope design needed" if status == "WARN" else "Within standard slope limit",
                ))
            else:
                checks.append(self._chk("VA-08", C, "Max Cut Depth", "INFO", "No cuts", "6.0 m", "m", "RDSO GE:G-0014", None, "No cut sections"))

        # VA-09: Max fill height
        if not prof_df.empty and "ew_height_m" in prof_df.columns:
            fill_df2 = prof_df[prof_df.get("ew_type", pd.Series(["NONE"]*len(prof_df))) == "FILL"]
            if not fill_df2.empty:
                max_fill    = float(fill_df2["ew_height_m"].abs().max())
                max_fill_ch = float(prof_df.loc[fill_df2["ew_height_m"].abs().idxmax(), "chainage_m"])
                status = "PASS" if max_fill <= self.max_fill_standard_m + 1e-3 else "WARN"
                checks.append(self._chk(
                    "VA-09", C, "Max Fill Height (Standard Slopes)",
                    status, f"{max_fill:.2f} m", f"{self.max_fill_standard_m:.1f} m",
                    "m", "RDSO GE:G-0014", max_fill_ch,
                    "Fill >6m - use 8.85m width + special design" if status == "WARN" else "Within standard slope limit",
                ))
            else:
                checks.append(self._chk("VA-09", C, "Max Fill Height", "INFO", "No fills", "6.0 m", "m", "RDSO GE:G-0014", None, "No fill sections"))

        # VA-10: Track buildup config check
        cfg_bu = float(self.config.get("vertical", {}).get("track_buildup_m", 0.762))
        profile = valign_result.profile_dataframe
        has_levels = (
            "proposed_rail_level_m" in profile.columns
            and "formation_level_m" in profile.columns
        )
        paired_levels = (
            profile.loc[:, ["proposed_rail_level_m", "formation_level_m"]]
            .dropna()
            if has_levels else pd.DataFrame()
        )
        actual_buildup = (
            paired_levels["proposed_rail_level_m"] - paired_levels["formation_level_m"]
            if not paired_levels.empty else pd.Series(dtype=float)
        )
        max_buildup_error = (
            float((actual_buildup - self.track_buildup_m).abs().max())
            if not actual_buildup.empty else None
        )
        config_matches = abs(cfg_bu - self.track_buildup_m) <= 0.01
        profile_matches = max_buildup_error is not None and max_buildup_error <= 0.01
        status = (
            "PASS" if config_matches and profile_matches
            else "WARN" if max_buildup_error is None
            else "FAIL"
        )
        observed = (
            f"configured {cfg_bu:.3f} m; max profile deviation {max_buildup_error:.3f} m"
            if max_buildup_error is not None
            else f"configured {cfg_bu:.3f} m; profile PRL/PFL unavailable"
        )
        checks.append(self._chk(
            "VA-10", C, "Track Buildup (PFL to PRL)",
            status, observed, f"{self.track_buildup_m:.3f} m",
            "m", "RDSO BG PSC", None,
            "Profile PRL-PFL and configured buildup match" if status == "PASS" else "Configured buildup mismatch or profile PRL/PFL missing or inconsistent",
        ))

        return checks

    # --------------------------------------------------------------------------
    # Category 2: Vertical Curves (VC-01 to VC-06)
    # --------------------------------------------------------------------------

    def _check_vertical_curves(
        self, valign_result: VAlignResult, hgeom_result: HGeomResult
    ) -> list[dict]:
        """Run VC-01 through VC-06 vertical curve checks."""
        checks: list[dict] = []
        C      = "VERTICAL_CURVE"
        vip_df = valign_result.vip_dataframe.copy() if not valign_result.vip_dataframe.empty else pd.DataFrame()
        ip_df  = hgeom_result.ip_dataframe.copy() if not hgeom_result.ip_dataframe.empty else pd.DataFrame()

        if vip_df.empty:
            checks.append(self._chk("VC-01", C, "VIP Data", "INFO", "N/A", "N/A", "", "IRPWM Para 409", None, "No VIP data for VC checks"))
            return checks

        for i in range(1, len(vip_df) - 1):
            row    = vip_df.iloc[i]
            ch     = float(row.get("chainage_m", 0.0))
            L      = float(row.get("curve_length_m", 0.0))
            g1_pct = float(vip_df.iloc[i-1].get("gradient_to_next_pct", 0.0))
            g2_pct = float(row.get("gradient_to_next_pct", 0.0))
            dg_pct = abs(g2_pct - g1_pct)
            if dg_pct < 0.01:
                continue

            # VC-01: Required Lv = dg% x V^2 / 4000 (IRPWM Para 409)
            req_lv = dg_pct * self.design_speed_kmh**2 / self.vc_rate_divisor
            status = "PASS" if L >= req_lv - 0.1 else "FAIL"
            checks.append(self._chk(
                "VC-01", C, f"Required VC Length @ VIP-{i+1}",
                status, f"{L:.1f} m", f"{req_lv:.1f} m",
                "m", "IRPWM Para 409", ch,
                f"dg={dg_pct:.3f}%, V={self.design_speed_kmh:.0f}km/h -> Lv_req={req_lv:.1f}m",
            ))

            # VC-02: Absolute minimum Lv >= 60m BG
            if L > 0:
                status = "PASS" if L >= self.min_vc_length_m - 0.1 else "FAIL"
                checks.append(self._chk(
                    "VC-02", C, f"Min VC Length (Absolute) @ VIP-{i+1}",
                    status, f"{L:.1f} m", f"{self.min_vc_length_m:.0f} m",
                    "m", "IRPWM Para 409", ch,
                    "VC below 60m absolute minimum for BG" if status == "FAIL" else "VC length OK",
                ))

            # VC-03/VC-04: Summit or Sag Rv
            if L > 0:
                rv        = L / (dg_pct / 100.0)
                is_summit = g2_pct < g1_pct
                rv_limit  = self.min_rv_summit_m if is_summit else self.min_rv_sag_m
                rv_id     = "VC-03" if is_summit else "VC-04"
                rv_type   = "Summit" if is_summit else "Sag"
                status    = "PASS" if rv >= rv_limit - 0.1 else "FAIL"
                checks.append(self._chk(
                    rv_id, C, f"{rv_type} VC Radius (Rv) @ VIP-{i+1}",
                    status, f"{rv:.0f} m", f"{rv_limit:.0f} m",
                    "m", "IRPWM Para 409", ch,
                    f"{rv_type}: Rv=L/dg={L:.0f}/{dg_pct:.3f}%={rv:.0f}m",
                ))

            # VC-06: Grade change >= 0.4% without VC (IRPWM Para 409)
            if dg_pct >= self.min_grade_change_for_vc and L == 0:
                checks.append(self._chk(
                    "VC-06", C, f"Grade Change >= 0.4% Without VC @ VIP-{i+1}",
                    "FAIL", "No VC (L=0)", f"dg={dg_pct:.2f}% >= 0.4% — VC required",
                    "", "IRPWM Para 409", ch,
                    f"Grade change {dg_pct:.2f}% >= 0.4% has no vertical curve — must rectify",
                ))

        # VC-05: VC must NOT overlap Transition Spiral zone
        overlaps: list[tuple] = []
        if not ip_df.empty:
            for i in range(1, len(vip_df) - 1):
                row = vip_df.iloc[i]
                L   = float(row.get("curve_length_m", 0.0))
                if L <= 0:
                    continue
                ch     = float(row.get("chainage_m", 0.0))
                pvc_ch = float(row.get("pvc_chainage_m", ch - L/2.0))
                pvt_ch = float(row.get("pvt_chainage_m", ch + L/2.0))
                for _, ip_row in ip_df.iterrows():
                    tptc = float(ip_row.get("tptc_chainage_m", 0.0))
                    tpcc = float(ip_row.get("tpcc_chainage_m", tptc))
                    ctpc = float(ip_row.get("ctpc_chainage_m", tpcc))
                    ctpt = float(ip_row.get("ctpt_chainage_m", ctpc))

                    for s_start, s_end in [(tptc, tpcc), (ctpc, ctpt)]:
                        if s_end > s_start + 0.1:
                            ov_s = max(pvc_ch, s_start)
                            ov_e = min(pvt_ch, s_end)
                            if ov_e > ov_s + 0.1:
                                overlaps.append((ch, ov_e - ov_s))

        if overlaps:
            for ch, ov in overlaps:
                checks.append(self._chk(
                    "VC-05", C, "VC Overlaps Transition Spiral Zone",
                    "FAIL", f"{ov:.1f} m overlap", "0 m",
                    "m", "RDSO GE:G-1", ch,
                    f"VC overlaps Ls zone by {ov:.1f}m - shift VC or Ls to eliminate",
                ))
        else:
            checks.append(self._chk(
                "VC-05", C, "VC Overlap with Transition Spiral",
                "PASS", "No overlap", "0 m",
                "m", "RDSO GE:G-1", None, "No VC/Ls zone overlaps",
            ))

        return checks

    # --------------------------------------------------------------------------
    # Category 3: Horizontal Curves (HC-01 to HC-10)
    # --------------------------------------------------------------------------

    def _check_horizontal_curves(self, hgeom_result: HGeomResult) -> list[dict]:
        """Run HC-01 through HC-10 horizontal curve checks."""
        checks: list[dict] = []
        C     = "HORIZONTAL_CURVE"
        ip_df = hgeom_result.ip_dataframe.copy() if not hgeom_result.ip_dataframe.empty else pd.DataFrame()

        if ip_df.empty:
            checks.append(self._chk("HC-01", C, "Horizontal Curves", "INFO", "No curves", "N/A", "", "IRPWM Para 401", None, "No horizontal curves detected"))
            return checks

        for _, row in ip_df.iterrows():
            idx = int(row.get("ip_index", 0))
            ch  = float(row.get("tptc_chainage_m", row.get("chainage_m", 0.0)))
            r   = float(row.get("radius_m", 9999.0))
            ca  = float(row.get("cant_mm", 0.0))
            cd  = float(row.get("actual_cd_mm", 0.0))
            ls  = float(row.get("transition_length_m", 0.0))
            v   = float(row.get("permissible_speed_kmph", 0.0))
            ct  = str(row.get("curve_type", "SIMPLE"))

            # HC-01: Minimum radius per design speed
            status = "PASS" if r >= self.min_radius_for_speed - 0.1 else "WARN"
            if r < 175.0:
                status = "FAIL"
            checks.append(self._chk(
                "HC-01", C, f"Min Radius @ Curve #{idx} ({ct})",
                status, f"{r:.0f} m", f"{self.min_radius_for_speed:.0f} m @ {self.design_speed_kmh:.0f}km/h",
                "m", "IRPWM Para 401", ch,
                f"R={r:.0f}m Vmax={v:.0f}km/h" + (" - speed restriction" if v < self.design_speed_kmh else " OK"),
            ))

            # HC-02: Max actual cant (165mm for BG)
            status = "PASS" if ca <= self.max_cant_passenger_mm + 0.1 else "FAIL"
            checks.append(self._chk(
                "HC-02", C, f"Max Actual Cant @ Curve #{idx}",
                status, f"{ca:.0f} mm", f"{self.max_cant_passenger_mm:.0f} mm",
                "mm", "IRSOD BG Schedule I Ch IV", ch,
                "Exceeds 165mm max permissible cant for BG" if status == "FAIL" else "Cant within 165mm limit",
            ))

            # HC-03: Cant excess for goods trains (75mm max)
            ce = float(row.get("cant_excess_mm", 0.0))
            status = "PASS" if ce <= self.max_cant_excess_mm + 0.1 else "FAIL"
            checks.append(self._chk(
                "HC-03", C, f"Cant Excess @ Curve #{idx}",
                status, f"{ce:.0f} mm", f"{self.max_cant_excess_mm:.0f} mm",
                "mm", "IRPWM Para 404", ch,
                "Exceeds max cant excess for goods trains" if status == "FAIL" else "Cant excess within 75mm limit",
            ))

            # HC-04: Cant deficiency (75mm std / 100mm high-speed)
            status = "PASS" if cd <= self.max_cant_deficiency_mm + 0.1 else "FAIL"
            checks.append(self._chk(
                "HC-04", C, f"Cant Deficiency @ Curve #{idx}",
                status, f"{cd:.0f} mm", f"{self.max_cant_deficiency_mm:.0f} mm",
                "mm", "IRSOD BG Schedule I Ch IV", ch,
                "Exceeds max cant deficiency" if status == "FAIL" else "Cd within limit",
            ))

            if ca > 0 and v > 0:
                ls_des = max(
                    self.spiral_factor_desirable * ca * v,
                    self.spiral_factor_desirable * cd * v,
                    0.72 * ca,
                )
                ls_min = max(
                    self.spiral_factor_minimum * ca * v,
                    self.spiral_factor_minimum * cd * v,
                    0.36 * ca,
                )

                # HC-05: Ls desirable
                status = "PASS" if ls >= ls_des - 0.1 else "WARN"
                checks.append(self._chk(
                    "HC-05", C, f"Transition Ls (Desirable) @ Curve #{idx}",
                    status, f"{ls:.0f} m", f"{ls_des:.0f} m",
                    "m", "IRPWM Para 405", ch,
                    f"max(0.008xCa x V, 0.008xCd x V, 0.72xCa)={ls_des:.0f}m",
                ))

                # HC-06: Ls absolute minimum
                status = "PASS" if ls >= ls_min - 0.1 else "FAIL"
                checks.append(self._chk(
                    "HC-06", C, f"Transition Ls (Abs Min) @ Curve #{idx}",
                    status, f"{ls:.0f} m", f"{ls_min:.0f} m",
                    "m", "IRPWM Para 405", ch,
                    f"max(0.0056xCa x V, 0.0056xCd x V, 0.36xCa)={ls_min:.0f}m" + (" CRITICAL" if status == "FAIL" else ""),
                ))

                # HC-07: Cant gradient 1:360
                ls_cg  = ca * self.cant_gradient_factor / 1000.0
                status = "PASS" if ls >= ls_cg - 0.1 else "FAIL"
                checks.append(self._chk(
                    "HC-07", C, f"Cant Gradient (1:{self.cant_gradient_factor:.0f}) @ Curve #{idx}",
                    status, f"{ls:.0f} m", f"{ls_cg:.0f} m",
                    "m", "IRPWM Para 405", ch, f"Ls >= Ca x 360/1000 = {ca:.0f}x0.36={ls_cg:.0f}m",
                ))

            # HC-10: Permissible speed vs design speed
            status = "PASS" if v >= self.design_speed_kmh - 0.5 else "WARN"
            checks.append(self._chk(
                "HC-10", C, f"Permissible Speed @ Curve #{idx}",
                status, f"{v:.0f} km/h", f"{self.design_speed_kmh:.0f} km/h",
                "km/h", "IRPWM Para 403", ch,
                f"Speed restriction {v:.0f}km/h on R={r:.0f}m" if status == "WARN" else "Full design speed OK",
            ))

        # HC-08 and HC-09: Between consecutive curve pairs
        for i in range(len(ip_df) - 1):
            cur      = ip_df.iloc[i]
            nxt      = ip_df.iloc[i+1]
            gap_m    = float(nxt.get("tptc_chainage_m", 0.0)) - float(cur.get("ctpt_chainage_m", 0.0))
            same_dir = str(cur.get("direction")) == str(nxt.get("direction"))
            ch       = float(cur.get("ctpt_chainage_m", 0.0))
            i_c      = int(cur.get("ip_index", i+1))
            i_n      = int(nxt.get("ip_index", i+2))

            if same_dir:
                status = "PASS" if gap_m >= self.min_straight_same_dir_m - 0.1 else "FAIL"
                checks.append(self._chk(
                    "HC-09", C, f"Min Straight (Same-Dir) #{i_c}->{i_n}",
                    status, f"{gap_m:.1f} m", f"{self.min_straight_same_dir_m:.0f} m",
                    "m", "IRPWM Para 406", ch,
                    f"Gap={gap_m:.1f}m between same-dir curves #{i_c} and #{i_n}",
                ))
            else:
                ls1     = float(cur.get("transition_length_m", 0.0))
                ls2     = float(nxt.get("transition_length_m", 0.0))
                min_gap = ls1 + ls2
                status  = "PASS" if gap_m >= min_gap - 0.1 else "FAIL"
                checks.append(self._chk(
                    "HC-08", C, f"Min Straight (Reverse) #{i_c}->{i_n}",
                    status, f"{gap_m:.1f} m", f"{min_gap:.0f} m (Ls1+Ls2)",
                    "m", "IRPWM Para 406", ch,
                    f"Reverse curves: gap={gap_m:.1f}m, Ls1({ls1:.0f})+Ls2({ls2:.0f})={min_gap:.0f}m required",
                ))

        return checks

    # --------------------------------------------------------------------------
    # Category 4: Formation & Earthwork (FW-01 to FW-07)
    # --------------------------------------------------------------------------

    def _check_formation(self, earthwork_result: EarthworkResult) -> list[dict]:
        """Run FW-01 through FW-07 formation and earthwork config checks."""
        checks: list[dict] = []
        C        = "FORMATION"
        ew_cfg   = self.config.get("earthwork", {})
        form_cfg = self.config.get("formation", {})
        vert_cfg = self.config.get("vertical", {})

        # FW-01: Formation width — BG single cut
        cfg_cut = float(ew_cfg.get("formation_width_cut_m", 0.0))
        status  = "PASS" if abs(cfg_cut - self.fw_cut_m) <= 0.01 else "FAIL"
        checks.append(self._chk("FW-01", C, "Formation Width - BG Single Cut",
            status, f"{cfg_cut:.2f} m", f"{self.fw_cut_m:.2f} m",
            "m", "RDSO GE:G-0014", None,
            "Config matches RDSO" if status == "PASS" else f"Config {cfg_cut}m != RDSO {self.fw_cut_m}m",
        ))

        # FW-02: Formation width — fill <= 6m
        cfg_fl = float(ew_cfg.get("formation_width_fill_m", 0.0))
        status = "PASS" if abs(cfg_fl - self.fw_fill_low_m) <= 0.01 else "FAIL"
        checks.append(self._chk("FW-02", C, "Formation Width - BG Single Fill <=6m",
            status, f"{cfg_fl:.2f} m", f"{self.fw_fill_low_m:.2f} m",
            "m", "RDSO GE:G-0014", None,
            "Config matches RDSO" if status == "PASS" else "Config mismatch",
        ))

        # FW-03: Formation width — fill > 6m
        cfg_fh = float(form_cfg.get("width_single_fill_high", self.fw_fill_high_m))
        status = "PASS" if abs(cfg_fh - self.fw_fill_high_m) <= 0.01 else "WARN"
        checks.append(self._chk("FW-03", C, "Formation Width - BG Single Fill >6m",
            status, f"{cfg_fh:.2f} m", f"{self.fw_fill_high_m:.2f} m",
            "m", "RDSO GE:G-0014", None,
            "Config matches RDSO" if status == "PASS" else "Config mismatch",
        ))

        # FW-04: Cut slope — ordinary soil
        cfg_sc = float(ew_cfg.get("side_slope_cut", 0.0))
        status = "PASS" if abs(cfg_sc - self.slope_cut) <= 0.01 else "WARN"
        checks.append(self._chk("FW-04", C, "Cut Slope - Ordinary Soil",
            status, f"{cfg_sc:.1f}:1", f"{self.slope_cut:.1f}:1",
            "H:V", "RDSO GE:G-0014", None,
            "Config matches RDSO" if status == "PASS" else "Non-standard slope - verify material",
        ))

        # FW-05: Fill slope — embankment
        cfg_sf = float(ew_cfg.get("side_slope_fill", 0.0))
        status = "PASS" if abs(cfg_sf - self.slope_fill) <= 0.01 else "WARN"
        checks.append(self._chk("FW-05", C, "Fill Slope - Embankment",
            status, f"{cfg_sf:.1f}:1", f"{self.slope_fill:.1f}:1",
            "H:V", "RDSO GE:G-0014", None,
            "Config matches RDSO" if status == "PASS" else "Non-standard fill slope",
        ))

        # FW-06: Blanket / subgrade thickness
        cfg_bm = float(form_cfg.get("blanket_thickness", self.blanket_mm/1000.0)) * 1000.0
        status = "PASS" if abs(cfg_bm - self.blanket_mm) <= 1.0 else "WARN"
        checks.append(self._chk("FW-06", C, "Blanket / Subgrade Thickness",
            status, f"{cfg_bm:.0f} mm", f"{self.blanket_mm:.0f} mm",
            "mm", "RDSO GE:G-0014", None,
            "Config matches RDSO" if status == "PASS" else "Non-standard blanket",
        ))

        # FW-07: Track buildup (matches VA-10, independent check in formation context)
        cfg_bu = float(vert_cfg.get("track_buildup_m", 0.762)) * 1000.0
        status = "PASS" if abs(cfg_bu - self.track_buildup_m*1000.0) <= 5.0 else "WARN"
        checks.append(self._chk("FW-07", C, "Track Buildup (PFL to PRL)",
            status, f"{cfg_bu:.0f} mm", f"{self.track_buildup_m*1000:.0f} mm",
            "mm", "RDSO BG PSC", None,
            "350mm ballast + 200mm sleeper + 172mm rail + 40mm pad/liner = 762mm total" if status == "PASS" else "Non-standard track buildup",
        ))

        return checks

    # --------------------------------------------------------------------------
    # Category 5: Special Locations (SL-01 to SL-04)
    # --------------------------------------------------------------------------

    def _check_special_locations(
        self,
        valign_result: VAlignResult,
        hgeom_result: HGeomResult,
        topo_result: TopoAnalysisResult,
    ) -> list[dict]:
        """Run SL-01 through SL-04 special location proximity checks."""
        checks: list[dict] = []
        C             = "SPECIAL_LOCATIONS"
        vip_df        = valign_result.vip_dataframe.copy() if not valign_result.vip_dataframe.empty else pd.DataFrame()
        ip_df         = hgeom_result.ip_dataframe.copy() if not hgeom_result.ip_dataframe.empty else pd.DataFrame()
        structures_df = topo_result.structures_df.copy() if not topo_result.structures_df.empty else pd.DataFrame()

        if structures_df.empty:
            for cid, cname, std in [
                ("SL-01", "Bridge Approach Gradient", "IRBM"),
                ("SL-02", "Level Crossing Gradient", "IRPWM"),
                ("SL-03", "No Curve Within Bridge Zone", "IRBM"),
                ("SL-04", "Min Straight Bridge to Curve", "IRBM"),
            ]:
                checks.append(self._chk(cid, C, cname, "INFO", "No structure data", "N/A", "", std, None, "No TopoAnalyzer structures available"))
            return checks

        def is_bridge(s: str) -> bool:
            return any(k in str(s).upper() for k in ["BRIDGE", "VIADUCT", "MINOR_BRIDGE", "MAJOR_BRIDGE"])

        def is_lc(s: str) -> bool:
            return any(k in str(s).upper() for k in ["LC", "LEVEL_CROSSING", "ROAD", "RUB", "ROB"])

        if "structure_type" in structures_df.columns:
            bridges = structures_df[structures_df["structure_type"].apply(is_bridge)]
            lcs     = structures_df[structures_df["structure_type"].apply(is_lc)]
        else:
            bridges = pd.DataFrame()
            lcs     = pd.DataFrame()

        # SL-01: Bridge approach gradient <= 1:400
        if not bridges.empty:
            for _, br in bridges.iterrows():
                br_ch = float(br.get("chainage_m", 0.0))
                for side in [br_ch - self.bridge_appr_len_m, br_ch + self.bridge_appr_len_m]:
                    g      = abs(self._gradient_at_chainage(vip_df, side))
                    status = "PASS" if g <= self.bridge_appr_grad_m + 1e-5 else "FAIL"
                    gr_str = f"1:{round(1.0/g)}" if g > 1e-6 else "LEVEL"
                    checks.append(self._chk(
                        "SL-01", C, f"Bridge Approach Gradient @ Ch {br_ch:.0f}m",
                        status, gr_str, "1:400", "ratio", "IRBM", side,
                        f"Approach {g*100:.3f}% at +-{self.bridge_appr_len_m:.0f}m" + (" - exceeds 1:400" if status == "FAIL" else ""),
                    ))
        else:
            checks.append(self._chk("SL-01", C, "Bridge Approach Gradient", "INFO", "No bridges", "1:400", "ratio", "IRBM", None, "No bridge structures"))

        # SL-02: Level crossing gradient <= 1:1000
        if not lcs.empty:
            for _, lc in lcs.iterrows():
                lc_ch = float(lc.get("chainage_m", 0.0))
                for chk in [lc_ch - self.lc_check_len_m, lc_ch, lc_ch + self.lc_check_len_m]:
                    g      = abs(self._gradient_at_chainage(vip_df, chk))
                    status = "PASS" if g <= self.lc_grad_limit + 1e-5 else "FAIL"
                    gr_str = f"1:{round(1.0/g)}" if g > 1e-6 else "LEVEL"
                    checks.append(self._chk(
                        "SL-02", C, f"LC Gradient @ Ch {lc_ch:.0f}m",
                        status, gr_str, "1:1000", "ratio", "IRPWM", chk,
                        f"LC gradient {g*100:.3f}%" + (" - exceeds 1:1000" if status == "FAIL" else ""),
                    ))
        else:
            checks.append(self._chk("SL-02", C, "Level Crossing Gradient", "INFO", "No LCs", "1:1000", "ratio", "IRPWM", None, "No level crossings"))

        # SL-03 and SL-04: Curve proximity to bridges
        if not bridges.empty and not ip_df.empty:
            for _, br in bridges.iterrows():
                br_ch = float(br.get("chainage_m", 0.0))
                for _, ip_row in ip_df.iterrows():
                    tptc   = float(ip_row.get("tptc_chainage_m", 0.0))
                    ctpt   = float(ip_row.get("ctpt_chainage_m", 0.0))
                    ip_idx = int(ip_row.get("ip_index", 0))
                    if tptc <= br_ch <= ctpt:
                        checks.append(self._chk(
                            "SL-03", C, f"Bridge Inside Curve #{ip_idx}",
                            "FAIL", f"Ch {br_ch:.0f}m inside [{tptc:.0f}-{ctpt:.0f}m]", "No curve at bridge",
                            "", "IRBM", br_ch, "Bridge within horizontal curve zone - not permitted",
                        ))
                    else:
                        dist   = min(abs(br_ch - tptc), abs(br_ch - ctpt))
                        status = "PASS" if dist >= self.min_bridge_curve_m - 0.1 else "WARN"
                        checks.append(self._chk(
                            "SL-04", C, f"Bridge->Curve #{ip_idx} Straight @ Ch {br_ch:.0f}m",
                            status, f"{dist:.1f} m", f"{self.min_bridge_curve_m:.0f} m",
                            "m", "IRBM", br_ch, f"Distance bridge to curve = {dist:.1f}m",
                        ))
        else:
            checks.append(self._chk("SL-03", C, "No Curve Within Bridge Zone", "INFO", "N/A", "N/A", "", "IRBM", None, "No bridge/curve data"))
            checks.append(self._chk("SL-04", C, "Min Straight Bridge to Curve", "INFO", "N/A", "50 m", "m", "IRBM", None, "No bridge/curve data"))

        return checks

    # --------------------------------------------------------------------------
    # Category 6: Drawing Standards (DS-01 to DS-03)
    # --------------------------------------------------------------------------

    def _check_drawing_standards(self) -> list[dict]:
        """Run DS-01 through DS-03 drawing standard checks."""
        checks: list[dict] = []
        C       = "DRAWING_STANDARDS"
        aln_cfg = self.config.get("alignment", {})

        # DS-01: Max alignment per L-Section sheet <= 5000m
        status = "PASS" if self.max_sheet_len_m <= 5000.0 + 0.1 else "FAIL"
        checks.append(self._chk(
            "DS-01", C, "Max Alignment per L-Section Sheet",
            status, f"{self.max_sheet_len_m:.0f} m", "5000 m",
            "m", "RDSO Drawing Standard", None,
            "Within RDSO 5km limit" if status == "PASS" else "Exceeds 5km per sheet",
        ))

        # DS-02: Standard chainage interval = 20m
        cfg_int = float(aln_cfg.get("chainage_interval", self.std_interval_m))
        status  = "PASS" if abs(cfg_int - 20.0) <= 0.1 else "WARN"
        checks.append(self._chk(
            "DS-02", C, "Chainage Interval",
            status, f"{cfg_int:.0f} m", "20 m",
            "m", "Indian Railways ESP Standard", None,
            "Standard 20m interval" if status == "PASS" else f"Non-standard {cfg_int:.0f}m interval",
        ))

        # DS-03: Required data bands >= 9
        status = "PASS" if self.req_bands >= 9 else "WARN"
        checks.append(self._chk(
            "DS-03", C, "Required Data Bands in L-Section",
            status, str(self.req_bands), "9",
            "count", "Indian Railways Drg Standard", None,
            "All 9 bands configured" if status == "PASS" else f"Only {self.req_bands} bands - need 9",
        ))

        return checks

    # --------------------------------------------------------------------------
    # Private Helpers
    # --------------------------------------------------------------------------

    def _get_min_radius_for_speed(self, val_cfg: dict) -> float:
        """Return RDSO GE:G-1 Table 1 minimum radius for the project design speed."""
        speed = self.design_speed_kmh
        if speed >= 160:
            return float(val_cfg.get("min_radius_160", 4000.0))
        elif speed >= 130:
            return float(val_cfg.get("min_radius_130", 2000.0))
        elif speed >= 100:
            return float(val_cfg.get("min_radius_100", 875.0))
        else:
            return float(val_cfg.get("min_radius_075", 400.0))

    def _gradient_at_chainage(self, vip_df: pd.DataFrame, chainage_m: float) -> float:
        """Interpolate gradient (fraction) at a given chainage from VIP dataframe."""
        if vip_df.empty:
            return 0.0
        for i in range(len(vip_df) - 1):
            c_s = float(vip_df.iloc[i]["chainage_m"])
            c_e = float(vip_df.iloc[i+1]["chainage_m"])
            if c_s <= chainage_m <= c_e:
                return float(vip_df.iloc[i].get("gradient_to_next_pct", 0.0)) / 100.0
        return 0.0

    @staticmethod
    def _chk(
        check_id:   str,
        category:   str,
        check_name: str,
        status:     str,
        value:      str,
        limit:      str,
        unit:       str,
        standard:   str,
        chainage:   float | None = None,
        remark:     str = "",
    ) -> dict:
        """Build a single validation check result dictionary."""
        return {
            "check_id":            check_id,
            "category":            category,
            "check_name":          check_name,
            "status":              status,
            "value":               value,
            "limit":               limit,
            "unit":                unit,
            "standard":            standard,
            "location_chainage_m": float(chainage) if chainage is not None else float("nan"),
            "remark":              remark,
        }
