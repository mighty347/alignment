"""
Module: alignment/vertical/_gradient_and_vc.py
Purpose: Parabolic Vertical Curve (IRPWM Para 409) builder & evaluator,
         Proposed Gradient Schedule extraction from PVIs or station PFLs,
         and the mandatory RDSO Engineering Verification Gate.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._vertical_models import (
    FormationLevelError,
    HorizontalCurveInterval,
    PVIPoint,
    ProposedGradientSegment,
    ResolvedConstraint,
    StationZone,
    VerificationCheck,
    classify_gradient,
    compute_segment_max_gradient,
    find_drainage_violations,
)


# ---------------------------------------------------------------------------
# Parabolic Vertical Curves (IRPWM Para 409)
# ---------------------------------------------------------------------------

def build_pvis(
    vip_ch: Sequence[float],
    vip_fl: Sequence[float],
    curves: Sequence[HorizontalCurveInterval],
    vcfg: Mapping[str, Any],
) -> tuple[list[PVIPoint], list[str]]:
    """Build 3-stage IRPWM Para 409 PVI & parabolic VC geometry."""
    n = len(vip_ch)
    if n < 2:
        raise FormationLevelError("At least 2 PVIs are required to define a vertical alignment.")

    min_dg = float(vcfg["min_grade_change_for_vc"])
    min_vc = float(vcfg["min_vc_length_m"])
    rv_summit = float(vcfg["min_rv_summit_m"])
    rv_sag = float(vcfg["min_rv_sag_m"])
    headroom = float(vcfg["vc_overlap_headroom_ratio"])
    warnings: list[str] = []

    tangents = [
        (vip_fl[i + 1] - vip_fl[i]) / (vip_ch[i + 1] - vip_ch[i])
        for i in range(n - 1)
    ]

    vc_lengths: list[float] = []
    rv_list: list[float] = []
    types: list[str] = []
    g_ins: list[float] = []
    g_outs: list[float] = []

    for i in range(n):
        g_in = tangents[i - 1] if i > 0 else tangents[0]
        g_out = tangents[i] if i < n - 1 else tangents[-1]
        g_ins.append(g_in)
        g_outs.append(g_out)

        if i == 0 or i == n - 1:
            vc_lengths.append(0.0)
            rv_list.append(0.0)
            types.append("ENDPOINT")
            continue

        dg = g_out - g_in
        if abs(dg) >= min_dg:
            is_summit = dg < 0.0
            rv = rv_summit if is_summit else rv_sag
            vc_lengths.append(max(min_vc, rv * abs(dg)))
            rv_list.append(rv)
            types.append("SUMMIT" if is_summit else "SAG")
        else:
            vc_lengths.append(0.0)
            rv_list.append(0.0)
            types.append("NONE")

    # Stage 2: Prevent adjacent VC overlaps (headroom cap)
    for i in range(n - 1):
        span = vip_ch[i + 1] - vip_ch[i]
        max_comb = span * headroom
        combined = 0.5 * vc_lengths[i] + 0.5 * vc_lengths[i + 1]
        if combined > max_comb and combined > 0.0:
            scale = max_comb / combined
            vc_lengths[i] *= scale
            vc_lengths[i + 1] *= scale

    # Stage 3: Finalize PVC/PVT coordinates & check horizontal spiral overlap
    pvis: list[PVIPoint] = []
    for i in range(n):
        length_l = vc_lengths[i]
        pvc_s = vip_ch[i] - 0.5 * length_l
        pvt_s = vip_ch[i] + 0.5 * length_l
        pvc_e = vip_fl[i] - g_ins[i] * (0.5 * length_l)
        pvt_e = (
            pvc_e + 0.5 * (g_ins[i] + g_outs[i]) * length_l
            if length_l > 0.0
            else vip_fl[i]
        )

        if length_l > 0.0:
            for curve in curves:
                spirals: list[tuple[float, float]] = []
                if curve.sc_chainage_m is not None and curve.sc_chainage_m > curve.ts_chainage_m:
                    spirals.append((curve.ts_chainage_m, curve.sc_chainage_m))
                if curve.cs_chainage_m is not None and curve.st_chainage_m > curve.cs_chainage_m:
                    spirals.append((curve.cs_chainage_m, curve.st_chainage_m))
                if not spirals:
                    spirals.append((curve.ts_chainage_m, curve.st_chainage_m))
                for sp_start, sp_end in spirals:
                    if min(pvt_s, sp_end) > max(pvc_s, sp_start):
                        warnings.append(
                            f"PVI {i} (Ch {vip_ch[i]:.2f}m) VC [{pvc_s:.1f}, {pvt_s:.1f}] "
                            f"overlaps horizontal curve/spiral [{sp_start:.1f}, {sp_end:.1f}]."
                        )
                        break

        pvis.append(
            PVIPoint(
                pvi_index=i,
                chainage_m=round(vip_ch[i], 3),
                elevation_m=round(vip_fl[i], 4),
                grade_in=g_ins[i],
                grade_out=g_outs[i],
                grade_change=g_outs[i] - g_ins[i],
                curve_type=types[i],
                vc_length_m=round(length_l, 3),
                rv_m=rv_list[i],
                pvc_chainage_m=round(pvc_s, 3),
                pvc_elevation_m=round(pvc_e, 4),
                pvt_chainage_m=round(pvt_s, 3),
                pvt_elevation_m=round(pvt_e, 4),
            )
        )

    return pvis, warnings


def evaluate_profile_and_slopes(
    p_chainages: np.ndarray,
    pvis: Sequence[PVIPoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate continuous PFL, instantaneous tangent slope, and VC mask at stations."""
    n_pts = len(p_chainages)
    p_pfls = np.empty(n_pts, dtype=float)
    p_slopes = np.empty(n_pts, dtype=float)
    in_vc = np.zeros(n_pts, dtype=bool)

    for k, s in enumerate(p_chainages):
        s_val = float(s)
        if s_val <= pvis[0].pvc_chainage_m:
            fp = pvis[0]
            p_pfls[k] = fp.pvc_elevation_m + fp.grade_in * (s_val - fp.pvc_chainage_m)
            p_slopes[k] = fp.grade_in
            continue
        if s_val >= pvis[-1].pvt_chainage_m:
            lp = pvis[-1]
            p_pfls[k] = lp.pvt_elevation_m + lp.grade_out * (s_val - lp.pvt_chainage_m)
            p_slopes[k] = lp.grade_out
            continue

        placed = False
        for pvi in pvis:
            if pvi.vc_length_m > 0.0 and pvi.pvc_chainage_m <= s_val <= pvi.pvt_chainage_m:
                x = s_val - pvi.pvc_chainage_m
                rate = (pvi.grade_out - pvi.grade_in) / pvi.vc_length_m
                p_pfls[k] = pvi.pvc_elevation_m + pvi.grade_in * x + 0.5 * rate * x * x
                p_slopes[k] = pvi.grade_in + rate * x
                in_vc[k] = True
                placed = True
                break
        if placed:
            continue

        for i in range(len(pvis) - 1):
            p_curr, p_next = pvis[i], pvis[i + 1]
            if p_curr.pvt_chainage_m - 1e-6 <= s_val <= p_next.pvc_chainage_m + 1e-6:
                p_pfls[k] = p_curr.pvt_elevation_m + p_curr.grade_out * (s_val - p_curr.pvt_chainage_m)
                p_slopes[k] = p_curr.grade_out
                placed = True
                break

        if not placed:
            raise FormationLevelError(
                f"Failed to evaluate continuous PFL at chainage {s_val:.2f}m; check PVI ordering."
            )

    return p_pfls, p_slopes, in_vc


def _rebalance_ramp_after_lift(
    vip_ch: list[float],
    vip_fl: list[float],
    idx: int,
    constraints: Sequence[ResolvedConstraint],
    ruling_grad: float,
    curves: Sequence[HorizontalCurveInterval],
    station_zones: Sequence[StationZone],
    vcfg: Mapping[str, Any],
) -> None:
    """Slide adjacent non-anchor ramp PVIs outward if a summit lift steepens a ramp > seg_cap."""
    min_grad = float(vcfg["min_gradient"])
    comp_factor = float(vcfg["grade_compensation_factor"])
    stn_cap = float(vcfg["station_max_gradient"])

    # Incoming ramp (idx - 1 -> idx)
    if idx >= 2:
        prev_is_anchor = any(abs(c.ch_m - vip_ch[idx - 1]) < 5.0 for c in constraints)
        dx_in = vip_ch[idx] - vip_ch[idx - 1]
        if not prev_is_anchor and dx_in > 0.0:
            seg_cap, _, _ = compute_segment_max_gradient(
                vip_ch[idx - 1], vip_ch[idx], ruling_grad, min_grad, comp_factor, curves, station_zones, stn_cap
            )
            g_in = (vip_fl[idx] - vip_fl[idx - 1]) / dx_in
            if abs(g_in) > seg_cap:
                sign_in = 1.0 if g_in > 0.0 else -1.0
                target_g = sign_in * seg_cap
                dx_prev = vip_ch[idx - 1] - vip_ch[idx - 2]
                g_prev = (vip_fl[idx - 1] - vip_fl[idx - 2]) / dx_prev if dx_prev > 0.0 else 0.0
                if abs(g_prev - target_g) > 1e-9:
                    new_s = (
                        vip_fl[idx] - vip_fl[idx - 2] + g_prev * vip_ch[idx - 2] - target_g * vip_ch[idx]
                    ) / (g_prev - target_g)
                    if vip_ch[idx - 2] + 10.0 < new_s < vip_ch[idx] - 10.0:
                        vip_ch[idx - 1] = round(new_s, 3)
                        vip_fl[idx - 1] = vip_fl[idx - 2] + g_prev * (new_s - vip_ch[idx - 2])

    # Outgoing ramp (idx -> idx + 1)
    if idx + 2 < len(vip_ch):
        next_is_anchor = any(abs(c.ch_m - vip_ch[idx + 1]) < 5.0 for c in constraints)
        dx_out = vip_ch[idx + 1] - vip_ch[idx]
        if not next_is_anchor and dx_out > 0.0:
            seg_cap, _, _ = compute_segment_max_gradient(
                vip_ch[idx], vip_ch[idx + 1], ruling_grad, min_grad, comp_factor, curves, station_zones, stn_cap
            )
            g_out = (vip_fl[idx + 1] - vip_fl[idx]) / dx_out
            if abs(g_out) > seg_cap:
                sign_out = 1.0 if g_out > 0.0 else -1.0
                target_g = sign_out * seg_cap
                dx_next = vip_ch[idx + 2] - vip_ch[idx + 1]
                g_next = (vip_fl[idx + 2] - vip_fl[idx + 1]) / dx_next if dx_next > 0.0 else 0.0
                if abs(g_next - target_g) > 1e-9:
                    new_s = (
                        vip_fl[idx] - vip_fl[idx + 2] + g_next * vip_ch[idx + 2] - target_g * vip_ch[idx]
                    ) / (g_next - target_g)
                    if vip_ch[idx] + 10.0 < new_s < vip_ch[idx + 2] - 10.0:
                        vip_ch[idx + 1] = round(new_s, 3)
                        vip_fl[idx + 1] = vip_fl[idx + 2] + g_next * (new_s - vip_ch[idx + 2])


def _constraint_check_chainages(bc: ResolvedConstraint, s_min: float, s_max: float) -> list[float]:
    """Return deck abutment & center chainages to check for a bridge/vertical constraint."""
    if bc.span_length_m > 0.0 and not bc.is_exact:
        half = 0.5 * bc.span_length_m
        pts = [max(s_min, bc.ch_m - half), bc.ch_m, min(s_max, bc.ch_m + half)]
        return sorted(set(round(p, 3) for p in pts))
    return [bc.ch_m]


def lift_pvis_for_bridge_clearance(
    vip_ch: list[float],
    vip_fl: list[float],
    constraints: Sequence[ResolvedConstraint],
    curves: Sequence[HorizontalCurveInterval],
    p_chainages: np.ndarray,
    vcfg: Mapping[str, Any],
    ruling_grad: float,
    station_zones: Sequence[StationZone] = (),
) -> tuple[list[float], list[float], list[PVIPoint], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively lift summit PVIs if parabolic sag reduces PFL below bridge clearance."""
    tol = float(vcfg["bridge_clearance_tolerance_m"])
    step = float(vcfg["bridge_summit_lift_step_m"])
    max_iter = int(vcfg["max_summit_lift_iterations"])
    s_min, s_max = float(p_chainages[0]), float(p_chainages[-1])

    pvis, warnings = build_pvis(vip_ch, vip_fl, curves, vcfg)
    p_pfls, p_slopes, in_vc = evaluate_profile_and_slopes(p_chainages, pvis)

    for _ in range(max_iter):
        deficits: list[tuple[ResolvedConstraint, float]] = []
        for bc in constraints:
            if bc.is_exact:
                continue
            check_pts = _constraint_check_chainages(bc, s_min, s_max)
            min_actual = min(float(np.interp(pt, p_chainages, p_pfls)) for pt in check_pts)
            deficit = bc.req_pfl_m - min_actual
            if deficit > tol:
                deficits.append((bc, deficit))
        if not deficits:
            break

        for bc, deficit in deficits:
            idx = int(np.argmin(np.abs(np.asarray(vip_ch) - bc.ch_m)))
            if abs(vip_ch[idx] - bc.ch_m) <= max(5.0, 0.5 * bc.span_length_m + 5.0):
                vip_fl[idx] += deficit + step
                _rebalance_ramp_after_lift(
                    vip_ch, vip_fl, idx, constraints, ruling_grad, curves, station_zones, vcfg
                )
            else:
                insert_pos = int(np.searchsorted(vip_ch, bc.ch_m))
                vip_ch.insert(insert_pos, bc.ch_m)
                vip_fl.insert(insert_pos, bc.req_pfl_m + step)
                _rebalance_ramp_after_lift(
                    vip_ch, vip_fl, insert_pos, constraints, ruling_grad, curves, station_zones, vcfg
                )

        pvis, warnings = build_pvis(vip_ch, vip_fl, curves, vcfg)
        p_pfls, p_slopes, in_vc = evaluate_profile_and_slopes(p_chainages, pvis)

    return vip_ch, vip_fl, pvis, warnings, p_pfls, p_slopes, in_vc


# ---------------------------------------------------------------------------
# Proposed Gradient Schedule Generation (From PVIs OR Station PFL Series)
# ---------------------------------------------------------------------------

def build_gradient_schedule_from_pvis(
    pvis: Sequence[PVIPoint],
    ruling_grad: float,
    curves: Sequence[HorizontalCurveInterval],
    station_zones: Sequence[StationZone],
    vcfg: Mapping[str, Any],
) -> list[ProposedGradientSegment]:
    """Emit continuous ProposedGradientSegment schedule directly from PVI tangents."""
    buildup_m = float(vcfg["track_buildup_m"])
    min_grad = float(vcfg["min_gradient"])
    comp_factor = float(vcfg["grade_compensation_factor"])
    stn_cap = float(vcfg["station_max_gradient"])
    level_thr = float(vcfg["level_gradient_threshold"])

    segments: list[ProposedGradientSegment] = []
    for i in range(len(pvis) - 1):
        p1, p2 = pvis[i], pvis[i + 1]
        length_m = p2.chainage_m - p1.chainage_m
        if length_m <= 0.0:
            continue

        grad = (p2.elevation_m - p1.elevation_m) / length_m
        direction, ratio, label = classify_gradient(grad, level_thr)
        max_allowed, is_comp, is_stn = compute_segment_max_gradient(
            p1.chainage_m,
            p2.chainage_m,
            ruling_grad,
            min_grad,
            comp_factor,
            curves,
            station_zones,
            stn_cap,
        )
        segments.append(
            ProposedGradientSegment(
                seg_no=len(segments) + 1,
                start_chainage_m=round(p1.chainage_m, 3),
                end_chainage_m=round(p2.chainage_m, 3),
                length_m=round(length_m, 3),
                start_pfl_m=round(p1.elevation_m, 4),
                end_pfl_m=round(p2.elevation_m, 4),
                start_prl_m=round(p1.elevation_m + buildup_m, 4),
                end_prl_m=round(p2.elevation_m + buildup_m, 4),
                gradient_decimal=round(grad, 6),
                gradient_pct=round(grad * 100.0, 4),
                slope_ratio=ratio,
                direction=direction,
                gradient_label=label,
                max_allowed_gradient=round(max_allowed, 6),
                is_curve_compensated=is_comp,
                is_station_zone=is_stn,
                vpi_chainage_m=p2.chainage_m if i < len(pvis) - 2 else None,
                vpi_elevation_m=p2.elevation_m if i < len(pvis) - 2 else None,
                vc_length_m=p2.vc_length_m if i < len(pvis) - 2 else 0.0,
            )
        )
    return segments


def extract_pvis_and_gradients_from_pfl(
    p_chainages: np.ndarray,
    p_pfls: np.ndarray,
    ruling_grad: float,
    curves: Sequence[HorizontalCurveInterval],
    station_zones: Sequence[StationZone],
    vcfg: Mapping[str, Any],
) -> tuple[list[PVIPoint], list[ProposedGradientSegment]]:
    """Derive condensed PVIs & Proposed Gradient Schedule from any station PFL array."""
    merge_tol = float(vcfg["gradient_merge_tolerance"])
    min_span = float(vcfg["min_gradient_span_m"])

    raw_slopes = np.diff(p_pfls) / np.diff(p_chainages)
    runs: list[tuple[int, int, float]] = []
    run_start = 0
    for idx in range(1, len(raw_slopes)):
        if abs(raw_slopes[idx] - raw_slopes[run_start]) > merge_tol:
            mean_g = float((p_pfls[idx] - p_pfls[run_start]) / (p_chainages[idx] - p_chainages[run_start]))
            runs.append((run_start, idx, mean_g))
            run_start = idx
    last_g = float((p_pfls[-1] - p_pfls[run_start]) / (p_chainages[-1] - p_chainages[run_start]))
    runs.append((run_start, len(p_chainages) - 1, last_g))

    # Keep true constant-grade tangents (>= 2 matching intervals & >= min_span);
    # single-interval changing chords between tangents are parabolic vertical curves.
    tangents = [
        r for r in runs
        if (r[1] - r[0]) >= 2 and (p_chainages[r[1]] - p_chainages[r[0]]) >= min_span
    ]
    if not tangents:
        tangents = [(0, len(p_chainages) - 1, float((p_pfls[-1] - p_pfls[0]) / (p_chainages[-1] - p_chainages[0])))]

    merged: list[tuple[int, int, float]] = [tangents[0]]
    for t_s, t_e, t_g in tangents[1:]:
        p_s, _, p_g = merged[-1]
        if abs(t_g - p_g) <= merge_tol:
            comb_g = float((p_pfls[t_e] - p_pfls[p_s]) / (p_chainages[t_e] - p_chainages[p_s]))
            merged[-1] = (p_s, t_e, comb_g)
        else:
            merged.append((t_s, t_e, t_g))

    vip_ch: list[float] = [float(p_chainages[0])]
    vip_fl: list[float] = [float(p_pfls[0])]
    detected_vc_lengths: dict[int, float] = {}

    for i in range(len(merged) - 1):
        _, a_end, g_a = merged[i]
        b_start, _, g_b = merged[i + 1]
        s_a, e_a = float(p_chainages[a_end]), float(p_pfls[a_end])
        s_b, e_b = float(p_chainages[b_start]), float(p_pfls[b_start])

        if abs(g_a - g_b) > 1e-9:
            s_vpi = (e_b - e_a + g_a * s_a - g_b * s_b) / (g_a - g_b)
            s_vpi = max(s_a, min(s_b, s_vpi)) if s_b > s_a else s_a
            e_vpi = e_a + g_a * (s_vpi - s_a)
        else:
            s_vpi, e_vpi = 0.5 * (s_a + s_b), 0.5 * (e_a + e_b)

        vc_len = max(0.0, s_b - s_a)
        if s_vpi > vip_ch[-1] + 1e-3 and s_vpi < float(p_chainages[-1]) - 1e-3:
            detected_vc_lengths[len(vip_ch)] = round(vc_len, 3)
            vip_ch.append(s_vpi)
            vip_fl.append(e_vpi)

    vip_ch.append(float(p_chainages[-1]))
    vip_fl.append(float(p_pfls[-1]))

    pvis, _ = build_pvis(vip_ch, vip_fl, curves, vcfg)
    for idx, vc_len in detected_vc_lengths.items():
        if idx < len(pvis) and vc_len > 0.0:
            pvi = pvis[idx]
            pvi.vc_length_m = vc_len
            pvi.pvc_chainage_m = round(pvi.chainage_m - 0.5 * vc_len, 3)
            pvi.pvt_chainage_m = round(pvi.chainage_m + 0.5 * vc_len, 3)
            pvi.pvc_elevation_m = round(pvi.elevation_m - pvi.grade_in * 0.5 * vc_len, 4)
            pvi.pvt_elevation_m = round(pvi.elevation_m + pvi.grade_out * 0.5 * vc_len, 4)

    schedule = build_gradient_schedule_from_pvis(pvis, ruling_grad, curves, station_zones, vcfg)
    return pvis, schedule


# ---------------------------------------------------------------------------
# Mandatory Engineering Verification Gate
# ---------------------------------------------------------------------------

def run_verification_gate(
    p_chainages: np.ndarray,
    p_egls: np.ndarray,
    p_pfls: np.ndarray,
    p_prls: np.ndarray,
    pvis: Sequence[PVIPoint],
    schedule: Sequence[ProposedGradientSegment],
    constraints: Sequence[ResolvedConstraint],
    vcfg: Mapping[str, Any],
) -> tuple[str, list[VerificationCheck], list[str]]:
    """Run all mandatory RDSO / IRPWM / IRBM / IRSOD engineering invariant checks."""
    checks: list[VerificationCheck] = []
    violations: list[str] = []

    def _record(name: str, passed: bool, detail: str) -> None:
        checks.append(VerificationCheck(name=name, passed=passed, detail=detail))
        if not passed:
            violations.append(f"{name}: {detail}")

    mono_ok = bool(np.all(np.diff(p_chainages) > 0.0))
    _record(
        "CHAINAGE_MONOTONICITY",
        mono_ok,
        "Strictly increasing station chainages" if mono_ok else "Non-monotonic chainage detected.",
    )

    buildup = float(vcfg["track_buildup_m"])
    buildup_tol = float(vcfg["buildup_tolerance_m"])
    max_err = float(np.max(np.abs((p_prls - p_pfls) - buildup)))
    _record(
        "TRACK_BUILDUP_INVARIANT",
        max_err <= buildup_tol,
        f"Max |PRL - PFL - {buildup:.3f}m| = {max_err:.7f}m (tol {buildup_tol:.6f}m)",
    )

    grad_tol = float(vcfg["gradient_tolerance"])
    bad_grads = [
        f"Seg {s.seg_no} ({s.gradient_label}, |g|={abs(s.gradient_decimal):.5f} > max {s.max_allowed_gradient:.5f})"
        for s in schedule
        if abs(s.gradient_decimal) > s.max_allowed_gradient + grad_tol
    ]
    _record(
        "COMPENSATED_RULING_GRADIENT",
        not bad_grads,
        "All gradient spans comply with IRPWM Para 408 / IRSOD caps." if not bad_grads else "; ".join(bad_grads),
    )

    min_clr = float(vcfg["min_drainage_clearance_m"])
    drain_tol = float(vcfg["drainage_tolerance_m"])
    max_trans_m = float(vcfg["max_cut_fill_transition_m"])
    drain_bad = find_drainage_violations(p_chainages, p_egls, p_pfls, min_clr, drain_tol, max_trans_m)
    _record(
        "DRAINAGE_CLEARANCE",
        not drain_bad,
        f"All fill stations >= EGL + {min_clr:.2f}m."
        if not drain_bad
        else f"{len(drain_bad)} station(s) in shallow waterlogging bank (< {min_clr:.2f}m).",
    )

    br_tol = float(vcfg["bridge_clearance_tolerance_m"])
    s_min, s_max = float(p_chainages[0]), float(p_chainages[-1])
    br_bad = []
    for bc in constraints:
        check_pts = _constraint_check_chainages(bc, s_min, s_max)
        if bc.is_exact:
            actual = float(np.interp(bc.ch_m, p_chainages, p_pfls))
            if abs(actual - bc.req_pfl_m) > br_tol:
                br_bad.append(f"{bc.name} @ Ch {bc.ch_m:.1f}m: PFL {actual:.3f}m != tie-in {bc.req_pfl_m:.3f}m")
        else:
            min_actual = min(float(np.interp(pt, p_chainages, p_pfls)) for pt in check_pts)
            if min_actual + br_tol < bc.req_pfl_m:
                br_bad.append(f"{bc.name} @ Ch {bc.ch_m:.1f}m: PFL {min_actual:.3f}m < req {bc.req_pfl_m:.3f}m")
    _record(
        "BRIDGE_HFL_CLEARANCE",
        not br_bad,
        f"All {len(constraints)} vertical constraint(s) satisfied." if not br_bad else "; ".join(br_bad),
    )

    vc_bad = [
        f"PVI {pvis[i].pvi_index} PVT ({pvis[i].pvt_chainage_m:.2f}m) > PVI {pvis[i + 1].pvi_index} PVC ({pvis[i + 1].pvc_chainage_m:.2f}m)"
        for i in range(len(pvis) - 1)
        if pvis[i].pvt_chainage_m > pvis[i + 1].pvc_chainage_m + 1e-3
    ]
    _record("VC_NON_OVERLAP", not vc_bad, "Zero vertical curve overlaps." if not vc_bad else "; ".join(vc_bad))

    cont_bad = [
        f"Gap between Seg {schedule[i].seg_no} ({schedule[i].end_chainage_m:.2f}m) and Seg {schedule[i + 1].seg_no} ({schedule[i + 1].start_chainage_m:.2f}m)"
        for i in range(len(schedule) - 1)
        if abs(schedule[i].end_chainage_m - schedule[i + 1].start_chainage_m) > 0.05
    ]
    _record(
        "GRADIENT_SPAN_CONTINUITY",
        bool(schedule) and not cont_bad,
        f"{len(schedule)} continuous gradient span(s)."
        if (schedule and not cont_bad)
        else ("; ".join(cont_bad) or "Empty gradient schedule."),
    )

    max_cut_cap = float(vcfg["max_excess_cut_m"])
    max_fill_cap = float(vcfg["max_excess_fill_m"])
    max_cut_obs = float(np.max(np.maximum(0.0, p_egls - p_pfls)))
    max_fill_obs = float(np.max(np.maximum(0.0, p_pfls - p_egls)))
    cf_ok = max_cut_obs <= max_cut_cap + 0.05 and max_fill_obs <= max_fill_cap + 0.05
    _record(
        "CUT_FILL_LIMITS",
        cf_ok,
        f"Max fill={max_fill_obs:.2f}m (<= {max_fill_cap:.1f}m), Max cut={max_cut_obs:.2f}m (<= {max_cut_cap:.1f}m)",
    )

    return ("PASS" if not violations else "FAIL"), checks, violations
