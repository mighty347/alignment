"""
Module: alignment/vertical/formation_level_solver.py
Purpose: Pure RDSO / IRPWM-compliant Formation Level (PFL), Proposed Rail Level
         (PRL), and Proposed Gradient Schedule engine for Railway Alignment Design.

Modes:
    1. Auto Mode        : Generates optimal long-tangent RDSO-quantized PVIs,
                          PFL, PRL, and Proposed Gradient Schedule from EGL +
                          horizontal curves + bridge/station/tie-in constraints.
    2. Interactive Mode : Evaluates engineer-supplied PVIs with parabolic VCs,
                          proposed gradients, and live RDSO verification.
    3. PFL-Driven Mode  : Derives continuous Proposed Gradient Schedule & PVIs
                          directly from any station-by-station PFL series.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    from alignment.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

from .bridge_constraint_loader import (
    load_bridge_constraints_from_excel,
    load_bridge_constraints_from_records,
    load_station_zones_from_records,
)
from ._gradient_and_vc import (
    build_gradient_schedule_from_pvis,
    build_pvis,
    evaluate_profile_and_slopes,
    extract_pvis_and_gradients_from_pfl,
    lift_pvis_for_bridge_clearance,
    run_verification_gate,
)
from ._vertical_models import (
    FormationLevelError,
    FormationLevelResult,
    HorizontalCurveInterval,
    PVIPoint,
    ProfileStationOutput,
    ProposedGradientSegment,
    ResolvedConstraint,
    StationZone,
    TerrainStation,
    VerificationCheck,
    VerticalConstraint,
    build_standard_ratios,
    classify_gradient,
    compute_segment_max_gradient,
    find_drainage_violations,
    format_gradient_label,
    load_vertical_config,
    resolve_ruling_gradient,
    validate_stations,
)

logger = get_logger("FormationLevelSolver")


# ---------------------------------------------------------------------------
# Constraint Resolution & Auto-Optimizer Helpers
# ---------------------------------------------------------------------------

def _resolve_constraints(
    constraints: Sequence[VerticalConstraint],
    p_chainages: np.ndarray,
    p_egls: np.ndarray,
    vcfg: Mapping[str, Any],
) -> list[ResolvedConstraint]:
    """Resolve bridge, RUB, HFL, structural depth, and tie-in constraints to concrete PFLs."""
    clearances = {str(k).upper(): float(v) for k, v in vcfg["bridge_clearances_m"].items()}
    min_hfl_clr = float(vcfg["min_hfl_clearance_m"])
    min_drain_clr = float(vcfg["min_drainage_clearance_m"])
    dedup_m = float(vcfg["bridge_dedup_spacing_m"])
    s_min, s_max = float(p_chainages[0]), float(p_chainages[-1])

    resolved: list[ResolvedConstraint] = []
    for item in constraints:
        ch = float(item.chainage_m)
        if not math.isfinite(ch) or ch < s_min - 1.0 or ch > s_max + 1.0:
            continue

        ctype = item.constraint_type.strip().upper()
        if ctype in ("ROB", "LC") and item.min_pfl_m is None and item.exact_pfl_m is None:
            continue

        p_egl = float(np.interp(ch, p_chainages, p_egls))
        is_exact = item.exact_pfl_m is not None
        if is_exact:
            req_pfl = float(item.exact_pfl_m)  # type: ignore[arg-type]
        else:
            if ctype not in clearances and item.min_pfl_m is None and item.hfl_m is None:
                logger.warning("Skipping unrecognized vertical constraint '%s' at Ch %.2fm", ctype, ch)
                continue

            struct_d = float(item.structural_depth_m) if item.structural_depth_m is not None else 0.0
            if item.hfl_m is not None:
                # Hydraulic study governs: PFL >= HFL + Clearance + Structural Depth
                clr_hfl = float(item.clearance_m) if item.clearance_m is not None else min_hfl_clr
                req_pfl = max(float(item.hfl_m) + clr_hfl + struct_d, p_egl + min_drain_clr)
                # For Hume Pipe (H.P) culverts with bed level & pipe dia, ensure pipe fits above bed
                btype_clean = item.bridge_type.replace(".", "").strip().upper()
                if btype_clean in ("HP", "HUMEPIPE") and item.bed_level_m is not None and item.span_length_m > 0.0:
                    req_pfl = max(req_pfl, float(item.bed_level_m) + item.span_length_m + struct_d)
            else:
                # Fallback feasibility/RUB clearance above EGL when HFL is absent
                clr = clearances.get(ctype, 0.0)
                req_pfl = p_egl + clr

            if item.min_pfl_m is not None:
                req_pfl = max(req_pfl, float(item.min_pfl_m))

        candidate = ResolvedConstraint(
            ch_m=round(ch, 3),
            req_pfl_m=round(req_pfl, 4),
            is_exact=is_exact,
            constraint_type=ctype,
            name=item.name or f"{ctype}@{ch:.0f}m",
            span_length_m=float(item.span_length_m),
        )
        merged = False
        for idx, existing in enumerate(resolved):
            if abs(existing.ch_m - candidate.ch_m) < dedup_m:
                if candidate.is_exact or (not existing.is_exact and candidate.req_pfl_m > existing.req_pfl_m):
                    resolved[idx] = candidate
                merged = True
                break
        if not merged:
            resolved.append(candidate)

    resolved.sort(key=lambda c: c.ch_m)
    return resolved


def _compute_interval_grade_caps(
    p_chainages: np.ndarray,
    ruling_grad: float,
    curves: Sequence[HorizontalCurveInterval],
    station_zones: Sequence[StationZone],
    standard_ratios: Sequence[float],
    vcfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-interval (exact_cap, quantized_cap) arrays across all station steps."""
    min_grad = float(vcfg["min_gradient"])
    comp_factor = float(vcfg["grade_compensation_factor"])
    stn_cap = float(vcfg["station_max_gradient"])

    n_seg = len(p_chainages) - 1
    g_caps = np.empty(n_seg, dtype=float)
    q_caps = np.empty(n_seg, dtype=float)
    for k in range(n_seg):
        cap, _, _ = compute_segment_max_gradient(
            float(p_chainages[k]), float(p_chainages[k + 1]),
            ruling_grad, min_grad, comp_factor, curves, station_zones, stn_cap,
        )
        g_caps[k] = cap
        legal = [g for g in standard_ratios if g <= cap + 1e-9]
        q_caps[k] = max(legal) if legal else 0.0
    return g_caps, q_caps


def _compute_bridge_min_envelope(
    p_chainages: np.ndarray,
    q_caps: np.ndarray,
    constraints: Sequence[ResolvedConstraint],
    vcfg: Mapping[str, Any],
) -> np.ndarray:
    """Propagate all bridge/RUB/HFL floors backward at legal quantized grades.

    Guarantees that from any station staying >= envelope, every future bridge/RUB
    can be reached without violating curve-compensated or LC/approach gradient caps.
    """
    n_pts = len(p_chainages)
    env = np.full(n_pts, -np.inf, dtype=float)
    vc_reserve = float(vcfg["bridge_summit_lift_step_m"]) * 2.0

    for bc in constraints:
        if bc.is_exact:
            continue
        half = 0.5 * bc.span_length_m
        s_lo, s_hi = bc.ch_m - half, bc.ch_m + half
        target = bc.req_pfl_m + vc_reserve

        idx_lo = max(0, int(np.searchsorted(p_chainages, s_lo, side="right")) - 1)
        idx_hi = min(n_pts - 1, int(np.searchsorted(p_chainages, s_hi, side="left")))
        for k in range(idx_lo, idx_hi + 1):
            env[k] = max(env[k], target)

    # Backward pass: propagate required climb at legal quantized grades
    for k in range(n_pts - 2, -1, -1):
        if math.isfinite(env[k + 1]):
            ds = float(p_chainages[k + 1] - p_chainages[k])
            env[k] = max(env[k], env[k + 1] - float(q_caps[k]) * ds)

    return env


def _compute_tie_in_cone(
    p_chainages: np.ndarray,
    g_caps: np.ndarray,
    constraints: Sequence[ResolvedConstraint],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float]]]:
    """Propagate backward (min, max) reachability cone for exact tie-in anchors."""
    n_pts = len(p_chainages)
    lo = np.full(n_pts, -np.inf, dtype=float)
    hi = np.full(n_pts, np.inf, dtype=float)
    exact_targets: list[tuple[int, float]] = []

    for bc in constraints:
        if not bc.is_exact:
            continue
        idx = int(np.argmin(np.abs(p_chainages - bc.ch_m)))
        if idx == 0:
            continue
        exact_targets.append((idx, bc.req_pfl_m))
        lo[idx] = max(lo[idx], bc.req_pfl_m)
        hi[idx] = min(hi[idx], bc.req_pfl_m)

    for k in range(n_pts - 2, -1, -1):
        ds = float(p_chainages[k + 1] - p_chainages[k])
        cap = float(g_caps[k])
        if math.isfinite(lo[k + 1]):
            lo[k] = max(lo[k], lo[k + 1] - cap * ds)
        if math.isfinite(hi[k + 1]):
            hi[k] = min(hi[k], hi[k + 1] + cap * ds)

    exact_targets.sort(key=lambda item: item[0])
    return lo, hi, exact_targets


def _extract_existing_levels(
    stations: Sequence[TerrainStation],
    buildup_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract existing rail (ERL) and formation (EFL) arrays using IRS SOD buildup."""
    n = len(stations)
    p_erls = np.full(n, np.nan, dtype=float)
    p_efls = np.full(n, np.nan, dtype=float)
    for k, st in enumerate(stations):
        has_erl = st.erl_m is not None and math.isfinite(float(st.erl_m))
        has_efl = st.efl_m is not None and math.isfinite(float(st.efl_m))
        if has_erl and has_efl:
            p_erls[k] = float(st.erl_m)  # type: ignore[arg-type]
            p_efls[k] = float(st.efl_m)  # type: ignore[arg-type]
        elif has_erl:
            erl = float(st.erl_m)  # type: ignore[arg-type]
            p_erls[k] = erl
            p_efls[k] = erl - buildup_m
        elif has_efl:
            efl = float(st.efl_m)  # type: ignore[arg-type]
            p_efls[k] = efl
            p_erls[k] = efl + buildup_m
    return p_erls, p_efls


def _derive_existing_tie_ins(
    p_chainages: np.ndarray,
    p_efls: np.ndarray,
    existing_constraints: Sequence[VerticalConstraint],
) -> list[VerticalConstraint]:
    """Create start/end TIE_IN anchors from existing rail/formation levels if present."""
    valid_idx = np.flatnonzero(np.isfinite(p_efls))
    if len(valid_idx) == 0:
        return []

    tie_ins: list[VerticalConstraint] = []
    candidate_indices = [int(valid_idx[0])]
    if len(valid_idx) > 1 and int(valid_idx[-1]) == len(p_chainages) - 1:
        candidate_indices.append(int(valid_idx[-1]))

    for idx in candidate_indices:
        ch = float(p_chainages[idx])
        # Only auto-pin boundary junction stations (start or end of corridor)
        if idx not in (0, len(p_chainages) - 1):
            continue
        if any(abs(c.chainage_m - ch) < 1.0 and c.exact_pfl_m is not None for c in existing_constraints):
            continue
        tie_ins.append(
            VerticalConstraint(
                chainage_m=ch,
                constraint_type="TIE_IN",
                name=f"Existing Track Tie-In @ Ch {ch:.0f}m",
                exact_pfl_m=round(float(p_efls[idx]), 4),
            )
        )
    return tie_ins


def _score_candidate_gradient(
    grad: float,
    curr_idx: int,
    curr_fl: float,
    p_chainages: np.ndarray,
    p_egls: np.ndarray,
    g_caps: np.ndarray,
    bridge_env: np.ndarray,
    tie_lo: np.ndarray,
    tie_hi: np.ndarray,
    ruling_grad: float,
    vcfg: Mapping[str, Any],
) -> Optional[tuple[float, int]]:
    """Evaluate one candidate gradient from curr_idx and return (score, reach_idx)."""
    max_t = float(vcfg["max_tangent_length_m"])
    min_reach = float(vcfg["min_optimizer_reach_m"])
    min_clr = float(vcfg["min_drainage_clearance_m"])
    drain_tol = float(vcfg["drainage_tolerance_m"])
    max_trans_m = float(vcfg["max_cut_fill_transition_m"])
    max_cut = float(vcfg["max_excess_cut_m"])
    max_fill = float(vcfg["max_excess_fill_m"])
    br_tol = float(vcfg["bridge_clearance_tolerance_m"])
    weights = vcfg["scoring"]

    curr_ch = float(p_chainages[curr_idx])
    end_idx = len(p_chainages) - 1
    horizon_ch = min(float(p_chainages[end_idx]), curr_ch + max_t)
    max_idx = int(np.searchsorted(p_chainages, horizon_ch, side="right")) - 1
    max_idx = min(end_idx, max(curr_idx + 1, max_idx))

    seg_ch = p_chainages[curr_idx : max_idx + 1]
    seg_egl = p_egls[curr_idx : max_idx + 1]
    seg_env = bridge_env[curr_idx : max_idx + 1]
    seg_lo = tie_lo[curr_idx : max_idx + 1]
    seg_hi = tie_hi[curr_idx : max_idx + 1]
    seg_pfl = curr_fl + grad * (seg_ch - curr_ch)
    fill_arr = seg_pfl - seg_egl
    cut_arr = -fill_arr

    valid_rel = 0
    for rel in range(len(seg_ch)):
        if rel > 0 and abs(grad) > float(g_caps[curr_idx + rel - 1]) + 1e-9:
            break
        if cut_arr[rel] > max_cut or fill_arr[rel] > max_fill:
            break
        if math.isfinite(seg_env[rel]) and seg_pfl[rel] + br_tol < seg_env[rel]:
            break
        if math.isfinite(seg_lo[rel]) and seg_pfl[rel] + br_tol < seg_lo[rel]:
            break
        if math.isfinite(seg_hi[rel]) and seg_pfl[rel] - br_tol > seg_hi[rel]:
            break
        valid_rel = rel

    if valid_rel == 0:
        return None

    bad_drain = find_drainage_violations(
        seg_ch[: valid_rel + 1], seg_egl[: valid_rel + 1], seg_pfl[: valid_rel + 1],
        min_clr, drain_tol, max_trans_m,
    )
    if bad_drain:
        if bad_drain[0] == 0:
            return None
        valid_rel = bad_drain[0] - 1

    reach_idx = curr_idx + valid_rel
    reach_ch = float(p_chainages[reach_idx])
    if (reach_ch - curr_ch) < min_reach and reach_idx < end_idx:
        # Allow shorter tangent if stopping right at a flat-zone boundary (LC / station / bridge)
        next_is_zone_change = (
            reach_idx < end_idx
            and abs(grad) > float(g_caps[reach_idx]) + 1e-9
        )
        if not next_is_zone_change:
            return None

    sub_ch = seg_ch[: valid_rel + 1]
    sub_egl = seg_egl[: valid_rel + 1]
    sub_pfl = seg_pfl[: valid_rel + 1]
    if find_drainage_violations(sub_ch, sub_egl, sub_pfl, min_clr, drain_tol, max_trans_m):
        return None

    sub_fill = np.maximum(0.0, sub_pfl - sub_egl)
    sub_cut = np.maximum(0.0, sub_egl - sub_pfl)
    tangent_len = reach_ch - curr_ch
    n_pts = max(1, len(sub_egl))

    unnecessary_cut_pen = 0.0
    if float(np.sum(sub_cut)) > 0.01 and tangent_len > 0.0:
        req_embank_g = abs(float(np.max(sub_egl)) + min_clr - curr_fl) / tangent_len
        if req_embank_g <= ruling_grad + 1e-6:
            unnecessary_cut_pen = float(weights["unnecessary_cut_penalty"])

    mean_cut_norm = float(np.mean(sub_cut)) / max_cut
    mean_fill_norm = float(np.mean(sub_fill)) / max_fill
    imbalance_norm = abs(float(np.sum(sub_cut)) - float(np.sum(sub_fill))) / (
        n_pts * max(max_cut, max_fill)
    )
    reach_bonus = min(
        float(weights["reach_weight"]),
        (tangent_len / float(weights["reach_reference_m"])) * float(weights["reach_weight"]),
    )
    score = (
        reach_bonus
        - (mean_cut_norm + mean_fill_norm) * float(weights["earthwork_weight"])
        - imbalance_norm * float(weights["imbalance_weight"])
        - (abs(grad) / ruling_grad) * float(weights["steepness_weight"])
        - unnecessary_cut_pen
    )
    return score, reach_idx


def _merge_micro_pvis(
    vip_ch: list[float],
    vip_fl: list[float],
    min_tangent_len: float,
    constraints: Sequence[ResolvedConstraint],
    p_chainages: np.ndarray,
    p_egls: np.ndarray,
    bridge_env: np.ndarray,
    ruling_grad: float,
    curves: Sequence[HorizontalCurveInterval],
    station_zones: Sequence[StationZone],
    vcfg: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    """Merge collinear or short non-anchored interior PVIs when safe."""
    min_grad = float(vcfg["min_gradient"])
    comp_factor = float(vcfg["grade_compensation_factor"])
    stn_cap = float(vcfg["station_max_gradient"])
    min_clr = float(vcfg["min_drainage_clearance_m"])
    drain_tol = float(vcfg["drainage_tolerance_m"])
    max_trans_m = float(vcfg["max_cut_fill_transition_m"])
    max_cut = float(vcfg["max_excess_cut_m"])
    max_fill = float(vcfg["max_excess_fill_m"])
    merge_tol = float(vcfg["gradient_merge_tolerance"])
    br_tol = float(vcfg["bridge_clearance_tolerance_m"])

    changed = True
    while changed and len(vip_ch) > 2:
        changed = False
        new_ch = [vip_ch[0]]
        new_fl = [vip_fl[0]]
        idx = 1
        while idx < len(vip_ch):
            span = vip_ch[idx] - new_ch[-1]
            is_last = idx == len(vip_ch) - 1
            is_anchor = any(abs(c.ch_m - vip_ch[idx]) < 5.0 for c in constraints)
            if not is_last and not is_anchor and idx + 1 < len(vip_ch):
                g_prev = (vip_fl[idx] - new_fl[-1]) / span if span > 0.0 else 0.0
                next_dx = vip_ch[idx + 1] - vip_ch[idx]
                g_next = (vip_fl[idx + 1] - vip_fl[idx]) / next_dx if next_dx > 0.0 else 0.0
                is_collinear = abs(g_next - g_prev) <= merge_tol

                if is_collinear:
                    changed = True
                    idx += 1
                    continue

                if span < min_tangent_len:
                    n_ch, n_fl = vip_ch[idx + 1], vip_fl[idx + 1]
                    chord_dx = n_ch - new_ch[-1]
                    chord_g = (n_fl - new_fl[-1]) / chord_dx if chord_dx > 0.0 else 0.0
                    seg_cap, _, _ = compute_segment_max_gradient(
                        new_ch[-1], n_ch, ruling_grad, min_grad, comp_factor, curves, station_zones, stn_cap
                    )
                    mask = (p_chainages >= new_ch[-1]) & (p_chainages <= n_ch)
                    m_ch, m_egl, m_env = p_chainages[mask], p_egls[mask], bridge_env[mask]
                    m_pfl = new_fl[-1] + chord_g * (m_ch - new_ch[-1])
                    env_ok = bool(np.all(
                        (~np.isfinite(m_env)) | (m_pfl + br_tol >= m_env)
                    ))
                    chord_safe = (
                        abs(chord_g) <= seg_cap + 1e-6
                        and len(m_ch) > 0
                        and env_ok
                        and float(np.max(m_egl - m_pfl)) <= max_cut
                        and float(np.max(m_pfl - m_egl)) <= max_fill
                        and not find_drainage_violations(m_ch, m_egl, m_pfl, min_clr, drain_tol, max_trans_m)
                    )
                    if chord_safe:
                        changed = True
                        idx += 1
                        continue
            new_ch.append(vip_ch[idx])
            new_fl.append(vip_fl[idx])
            idx += 1
        vip_ch, vip_fl = new_ch, new_fl
    return vip_ch, vip_fl


# ---------------------------------------------------------------------------
# Main Public Engine Class
# ---------------------------------------------------------------------------

class FormationLevelSolver:
    """RDSO / IRPWM Formation Level (PFL) & Proposed Gradient Schedule Engine."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.vcfg = load_vertical_config(config, config_path)
        self.standard_ratios = build_standard_ratios(self.vcfg)
        self.logger = get_logger(self.__class__.__name__)

    def solve(
        self,
        stations: Sequence[TerrainStation],
        horizontal_curves: Sequence[HorizontalCurveInterval] = (),
        constraints: Sequence[VerticalConstraint] = (),
        station_zones: Sequence[StationZone] = (),
        manual_pvis: Optional[Sequence[tuple[float, float]]] = None,
        terrain_type: Optional[str] = None,
        bridge_records: Sequence[Mapping[str, Any]] = (),
        bridge_excel_path: Optional[str | Path] = None,
        railway_station_records: Sequence[Mapping[str, Any]] = (),
        tie_in_to_existing: bool = True,
    ) -> FormationLevelResult:
        """Design Formation Level (PFL), Rail Level (PRL), and Proposed Gradients."""
        p_chainages, p_egls = validate_stations(stations)
        buildup_m = float(self.vcfg["track_buildup_m"])
        p_erls, p_efls = _extract_existing_levels(stations, buildup_m)
        ruling_grad = resolve_ruling_gradient(self.vcfg, terrain_type)

        all_constraints, all_zones = self._merge_inputs(
            constraints, station_zones, bridge_records, bridge_excel_path,
            railway_station_records, p_chainages, p_efls, tie_in_to_existing,
        )
        resolved_constraints = _resolve_constraints(all_constraints, p_chainages, p_egls, self.vcfg)

        if manual_pvis is not None:
            mode = "INTERACTIVE_PVI"
            vip_ch, vip_fl = self._validate_manual_pvis(manual_pvis, p_chainages)
            pvis, warnings = build_pvis(vip_ch, vip_fl, horizontal_curves, self.vcfg)
            p_pfls, p_slopes, in_vc = evaluate_profile_and_slopes(p_chainages, pvis)
        else:
            mode = "AUTO_OPTIMIZER"
            vip_ch, vip_fl = self._optimize_pvis(
                p_chainages, p_egls, ruling_grad, horizontal_curves, resolved_constraints, all_zones
            )
            vip_ch, vip_fl, pvis, warnings, p_pfls, p_slopes, in_vc = lift_pvis_for_bridge_clearance(
                vip_ch, vip_fl, resolved_constraints, horizontal_curves, p_chainages, self.vcfg,
                ruling_grad=ruling_grad, station_zones=all_zones,
            )

        schedule = build_gradient_schedule_from_pvis(
            pvis, ruling_grad, horizontal_curves, all_zones, self.vcfg
        )
        return self._assemble_result(
            mode, p_chainages, p_egls, p_pfls, p_slopes, in_vc,
            pvis, schedule, resolved_constraints, warnings, ruling_grad,
            p_erls=p_erls, p_efls=p_efls,
        )

    def design_from_pfl(
        self,
        stations: Sequence[TerrainStation],
        horizontal_curves: Sequence[HorizontalCurveInterval] = (),
        constraints: Sequence[VerticalConstraint] = (),
        station_zones: Sequence[StationZone] = (),
        terrain_type: Optional[str] = None,
        bridge_records: Sequence[Mapping[str, Any]] = (),
        bridge_excel_path: Optional[str | Path] = None,
        railway_station_records: Sequence[Mapping[str, Any]] = (),
        tie_in_to_existing: bool = False,
    ) -> FormationLevelResult:
        """Derive Proposed Gradient Schedule & run RDSO verification from existing station PFL."""
        p_chainages, p_egls = validate_stations(stations)
        if any(s.pfl_m is None or not math.isfinite(float(s.pfl_m)) for s in stations):
            raise FormationLevelError("All stations must supply a finite 'pfl_m' for PFL-driven mode.")

        buildup_m = float(self.vcfg["track_buildup_m"])
        p_erls, p_efls = _extract_existing_levels(stations, buildup_m)
        p_pfls = np.asarray([float(s.pfl_m) for s in stations], dtype=float)  # type: ignore[arg-type]
        ruling_grad = resolve_ruling_gradient(self.vcfg, terrain_type)
        all_constraints, all_zones = self._merge_inputs(
            constraints, station_zones, bridge_records, bridge_excel_path,
            railway_station_records, p_chainages, p_efls, tie_in_to_existing,
        )
        resolved_constraints = _resolve_constraints(all_constraints, p_chainages, p_egls, self.vcfg)

        pvis, schedule = extract_pvis_and_gradients_from_pfl(
            p_chainages, p_pfls, ruling_grad, horizontal_curves, all_zones, self.vcfg
        )

        seg_slopes = np.diff(p_pfls) / np.diff(p_chainages)
        p_slopes = np.append(seg_slopes, seg_slopes[-1])
        in_vc = np.zeros(len(p_chainages), dtype=bool)
        for pvi in pvis:
            if pvi.vc_length_m > 0.0:
                in_vc |= (p_chainages >= pvi.pvc_chainage_m) & (p_chainages <= pvi.pvt_chainage_m)

        return self._assemble_result(
            "PFL_DRIVEN_GRADIENT", p_chainages, p_egls, p_pfls, p_slopes, in_vc,
            pvis, schedule, resolved_constraints, [], ruling_grad,
            p_erls=p_erls, p_efls=p_efls,
        )

    def _merge_inputs(
        self,
        constraints: Sequence[VerticalConstraint],
        station_zones: Sequence[StationZone],
        bridge_records: Sequence[Mapping[str, Any]],
        bridge_excel_path: Optional[str | Path],
        railway_station_records: Sequence[Mapping[str, Any]],
        p_chainages: np.ndarray,
        p_efls: np.ndarray,
        tie_in_to_existing: bool,
    ) -> tuple[list[VerticalConstraint], list[StationZone]]:
        """Combine explicit constraints/zones with Bridge_List, Station_Excel, and ERL tie-ins."""
        start_ch_m, end_ch_m = float(p_chainages[0]), float(p_chainages[-1])
        merged_c = list(constraints)
        merged_z = list(station_zones)

        if railway_station_records:
            merged_z.extend(
                load_station_zones_from_records(
                    railway_station_records, start_ch_m, end_ch_m, config=self.vcfg
                )
            )

        if bridge_records:
            rec_c, rec_z = load_bridge_constraints_from_records(
                bridge_records, start_ch_m, end_ch_m, config=self.vcfg
            )
            merged_c.extend(rec_c)
            merged_z.extend(rec_z)

        if bridge_excel_path is not None:
            xl_c, xl_z = load_bridge_constraints_from_excel(
                bridge_excel_path, start_ch_m, end_ch_m, config=self.vcfg
            )
            merged_c.extend(xl_c)
            merged_z.extend(xl_z)

        if tie_in_to_existing:
            merged_c.extend(_derive_existing_tie_ins(p_chainages, p_efls, merged_c))

        return merged_c, merged_z

    def _validate_manual_pvis(
        self,
        manual_pvis: Sequence[tuple[float, float]],
        p_chainages: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        """Validate engineer-supplied PVI chainages and elevations."""
        if len(manual_pvis) < 2:
            raise FormationLevelError("Manual PVI mode requires at least 2 PVIs.")
        vip_ch = [float(ch) for ch, _ in manual_pvis]
        vip_fl = [float(fl) for _, fl in manual_pvis]
        if any(not (math.isfinite(c) and math.isfinite(e)) for c, e in zip(vip_ch, vip_fl)):
            raise FormationLevelError("All manual PVI chainages and elevations must be finite.")
        if any(b <= a for a, b in zip(vip_ch, vip_ch[1:])):
            raise FormationLevelError("Manual PVI chainages must be strictly increasing.")
        if vip_ch[0] > float(p_chainages[0]) + 1.0 or vip_ch[-1] < float(p_chainages[-1]) - 1.0:
            self.logger.warning(
                "Manual PVIs [%.1f, %.1f] do not span full alignment [%.1f, %.1f]; endpoints will extrapolate.",
                vip_ch[0], vip_ch[-1], float(p_chainages[0]), float(p_chainages[-1]),
            )
        return vip_ch, vip_fl

    def _optimize_pvis(
        self,
        p_chainages: np.ndarray,
        p_egls: np.ndarray,
        ruling_grad: float,
        curves: Sequence[HorizontalCurveInterval],
        constraints: Sequence[ResolvedConstraint],
        station_zones: Sequence[StationZone],
    ) -> tuple[list[float], list[float]]:
        """Greedy long-tangent RDSO minimum rise-and-fall PVI optimizer with global bridge envelope."""
        min_t = float(self.vcfg["min_tangent_length_m"])
        target_clr = float(self.vcfg["target_embankment_clearance_m"])
        br_tol = float(self.vcfg["bridge_clearance_tolerance_m"])

        g_caps, q_caps = _compute_interval_grade_caps(
            p_chainages, ruling_grad, curves, station_zones, self.standard_ratios, self.vcfg
        )
        bridge_env = _compute_bridge_min_envelope(p_chainages, q_caps, constraints, self.vcfg)
        tie_lo, tie_hi, exact_targets = _compute_tie_in_cone(p_chainages, g_caps, constraints)

        candidates: list[float] = []
        for g in self.standard_ratios:
            if g <= ruling_grad + 1e-9:
                candidates.append(g)
                if g > 0.0:
                    candidates.append(-g)

        start_ch, end_idx = float(p_chainages[0]), len(p_chainages) - 1
        start_anchor = next((c for c in constraints if abs(c.ch_m - start_ch) <= 1.0 and c.is_exact), None)
        curr_idx = 0
        base_start = start_anchor.req_pfl_m if start_anchor else float(p_egls[0]) + target_clr
        curr_fl = max(base_start, float(bridge_env[0])) if (not start_anchor and math.isfinite(bridge_env[0])) else base_start

        vip_ch: list[float] = [float(p_chainages[0])]
        vip_fl: list[float] = [curr_fl]
        max_iters = len(p_chainages) + 100

        for _ in range(max_iters):
            if curr_idx >= end_idx:
                break

            # Direct chord check into an upcoming exact tie-in target when feasible
            next_exact = next(((t_idx, t_fl) for t_idx, t_fl in exact_targets if t_idx > curr_idx), None)
            if next_exact is not None:
                t_idx, t_fl = next_exact
                dx_exact = float(p_chainages[t_idx] - p_chainages[curr_idx])
                if dx_exact <= float(self.vcfg["max_tangent_length_m"]) and dx_exact > 0.0:
                    g_exact = (t_fl - curr_fl) / dx_exact
                    if abs(g_exact) <= float(np.min(g_caps[curr_idx:t_idx])) + 1e-9:
                        scored_exact = _score_candidate_gradient(
                            g_exact, curr_idx, curr_fl, p_chainages, p_egls,
                            g_caps, bridge_env, tie_lo, tie_hi, ruling_grad, self.vcfg,
                        )
                        if scored_exact is not None and scored_exact[1] >= t_idx:
                            vip_ch.append(float(p_chainages[t_idx]))
                            vip_fl.append(t_fl)
                            curr_idx = t_idx
                            curr_fl = t_fl
                            continue

            best_grad: Optional[float] = None
            best_score = -1e18
            best_reach_idx = curr_idx

            for grad in candidates:
                if abs(grad) > float(g_caps[curr_idx]) + 1e-9:
                    continue
                scored = _score_candidate_gradient(
                    grad, curr_idx, curr_fl, p_chainages, p_egls,
                    g_caps, bridge_env, tie_lo, tie_hi, ruling_grad, self.vcfg,
                )
                if scored is not None and scored[0] > best_score:
                    best_score, best_reach_idx = scored
                    best_grad = grad

            if best_grad is None:
                step_g = float(q_caps[curr_idx])
                ds = float(p_chainages[curr_idx + 1] - p_chainages[curr_idx])
                if math.isfinite(tie_hi[curr_idx + 1]) and (curr_fl - step_g * ds - br_tol <= tie_hi[curr_idx + 1]):
                    best_grad = -step_g if (curr_fl > tie_hi[curr_idx + 1]) else step_g
                    best_reach_idx = curr_idx + 1
                elif math.isfinite(bridge_env[curr_idx + 1]) and (curr_fl + step_g * ds + br_tol >= bridge_env[curr_idx + 1]):
                    best_grad = step_g
                    best_reach_idx = curr_idx + 1
                else:
                    raise FormationLevelError(
                        f"No legal RDSO gradient found from Ch {float(p_chainages[curr_idx]):.1f}m "
                        f"(FL={curr_fl:.2f}m) within ruling gradient 1:{int(round(1.0 / ruling_grad))} and cut/fill caps."
                    )

            next_ch = float(p_chainages[best_reach_idx])
            next_fl = curr_fl + best_grad * (next_ch - float(p_chainages[curr_idx]))
            vip_ch.append(next_ch)
            vip_fl.append(next_fl)
            curr_idx = best_reach_idx
            curr_fl = next_fl

        return _merge_micro_pvis(
            vip_ch, vip_fl, min_t, constraints, p_chainages, p_egls, bridge_env,
            ruling_grad, curves, station_zones, self.vcfg,
        )

    def _assemble_result(
        self,
        mode: str,
        p_chainages: np.ndarray,
        p_egls: np.ndarray,
        p_pfls: np.ndarray,
        p_slopes: np.ndarray,
        in_vc: np.ndarray,
        pvis: list[PVIPoint],
        schedule: list[ProposedGradientSegment],
        constraints: Sequence[ResolvedConstraint],
        warnings: list[str],
        ruling_grad: float,
        p_erls: Optional[np.ndarray] = None,
        p_efls: Optional[np.ndarray] = None,
    ) -> FormationLevelResult:
        """Compute PRL, Cut/Fill, station rows, and run the Engineering Verification Gate."""
        buildup_m = float(self.vcfg["track_buildup_m"])
        level_thr = float(self.vcfg["level_gradient_threshold"])
        p_prls = p_pfls + buildup_m

        status, checks, violations = run_verification_gate(
            p_chainages, p_egls, p_pfls, p_prls, pvis, schedule, constraints, self.vcfg
        )

        station_rows: list[ProfileStationOutput] = []
        for k in range(len(p_chainages)):
            diff = float(p_pfls[k] - p_egls[k])
            slope_k = float(p_slopes[k])
            erl_val = (
                round(float(p_erls[k]), 4)
                if p_erls is not None and math.isfinite(float(p_erls[k]))
                else None
            )
            efl_val = (
                round(float(p_efls[k]), 4)
                if p_efls is not None and math.isfinite(float(p_efls[k]))
                else None
            )
            diff_rl = round(float(p_prls[k]) - erl_val, 4) if erl_val is not None else None
            station_rows.append(
                ProfileStationOutput(
                    chainage_m=round(float(p_chainages[k]), 3),
                    egl_m=round(float(p_egls[k]), 4),
                    pfl_m=round(float(p_pfls[k]), 4),
                    prl_m=round(float(p_prls[k]), 4),
                    bank_m=round(max(0.0, diff), 4),
                    cutting_m=round(max(0.0, -diff), 4),
                    proposed_gradient_decimal=round(slope_k, 6),
                    proposed_gradient_pct=round(slope_k * 100.0, 4),
                    proposed_gradient_label=format_gradient_label(slope_k, level_thr),
                    in_vertical_curve=bool(in_vc[k]),
                    erl_m=erl_val,
                    efl_m=efl_val,
                    diff_rail_level_m=diff_rl,
                )
            )

        self.logger.info(
            "FormationLevelSolver [%s] complete: %d stations, %d PVIs, %d gradient spans | Gate=%s",
            mode, len(station_rows), len(pvis), len(schedule), status,
        )
        return FormationLevelResult(
            mode=mode,
            stations=station_rows,
            pvis=pvis,
            gradient_schedule=schedule,
            verification_status=status,
            checks=checks,
            violations=violations,
            warnings=warnings,
            metadata={
                "total_stations": len(station_rows),
                "total_pvis": len(pvis),
                "total_gradient_spans": len(schedule),
                "ruling_gradient": ruling_grad,
                "track_buildup_m": buildup_m,
                "max_bank_m": round(float(np.max(np.maximum(0.0, p_pfls - p_egls))), 3),
                "max_cutting_m": round(float(np.max(np.maximum(0.0, p_egls - p_pfls))), 3),
            },
        )
