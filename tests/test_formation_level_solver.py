"""
Unit tests for alignment/vertical/formation_level_solver.py.
Standards: RDSO GE:G-1 / IRPWM Para 408-409 / IRBM Para 106 / IRS SOD (BG).
Runs 100% with pure in-memory synthetic data and hand-calculated benchmarks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ALIGNMENT_ROOT = Path(__file__).resolve().parent.parent
if str(_ALIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALIGNMENT_ROOT))

from vertical import (
    FormationLevelError,
    FormationLevelSolver,
    HorizontalCurveInterval,
    StationZone,
    TerrainStation,
    VerticalConstraint,
)


def _make_stations(
    end_ch_m: float,
    step_m: float,
    egl_fn,
    start_ch_m: float = 0.0,
) -> list[TerrainStation]:
    """Build synthetic strictly monotonic terrain stations."""
    stations: list[TerrainStation] = []
    ch = start_ch_m
    while ch <= end_ch_m + 1e-6:
        stations.append(TerrainStation(chainage_m=round(ch, 3), egl_m=round(float(egl_fn(ch)), 4)))
        ch += step_m
    return stations


def test_auto_flat_terrain_level_gradient_and_invariants():
    """Flat plains -> single LEVEL gradient, valid drainage bank, exact 0.762m buildup."""
    stations = _make_stations(2000.0, 20.0, lambda _ch: 100.0)
    solver = FormationLevelSolver()

    result = solver.solve(stations)

    assert result.mode == "AUTO_OPTIMIZER"
    assert result.verification_status == "PASS", result.violations
    assert len(result.gradient_schedule) == 1
    seg = result.gradient_schedule[0]
    assert seg.direction == "LEVEL"
    assert seg.gradient_label == "LEVEL"
    assert seg.start_chainage_m == pytest.approx(0.0)
    assert seg.end_chainage_m == pytest.approx(2000.0)

    for st in result.stations:
        assert st.prl_m - st.pfl_m == pytest.approx(0.762, abs=1e-4)
        assert st.bank_m >= 0.60 - 1e-4
        assert st.cutting_m == pytest.approx(0.0)


def test_curve_grade_compensation_irpwm_para_408():
    """Rising terrain through R=1000m curve must enforce 70/R=0.07% grade compensation."""
    # Terrain climbs at 1 in 160 (0.00625) across 0..2400m
    stations = _make_stations(2400.0, 20.0, lambda ch: 100.0 + 0.0055 * ch)
    curve = HorizontalCurveInterval(
        ts_chainage_m=600.0,
        sc_chainage_m=700.0,
        cs_chainage_m=1700.0,
        st_chainage_m=1800.0,
        radius_m=1000.0,
        curve_id="C1",
    )
    solver = FormationLevelSolver()
    result = solver.solve(stations, horizontal_curves=[curve])

    assert result.verification_status == "PASS", result.violations
    # Hand calculation: ruling 0.006667 - 70/(1000*100) = 0.005967
    expected_cap = 0.006667 - 0.0007
    comp_segs = [s for s in result.gradient_schedule if s.is_curve_compensated]
    assert comp_segs, "Expected at least one curve-compensated gradient segment."
    for seg in comp_segs:
        assert seg.max_allowed_gradient == pytest.approx(expected_cap, abs=1e-5)
        assert abs(seg.gradient_decimal) <= expected_cap + 1e-5


def test_bridge_hfl_clearance_and_summit_lift():
    """Major river bridge with HFL forces ramp and preserves clearance after summit VC."""
    stations = _make_stations(3000.0, 20.0, lambda _ch: 100.0)
    bridge = VerticalConstraint(
        chainage_m=1500.0,
        constraint_type="MAJOR_BRIDGE",
        name="River Br-1",
        hfl_m=104.5,  # req_pfl >= max(100+4.0, 104.5+0.60) = 105.10m
    )
    solver = FormationLevelSolver()
    result = solver.solve(stations, constraints=[bridge])

    assert result.verification_status == "PASS", result.violations
    st_bridge = next(s for s in result.stations if abs(s.chainage_m - 1500.0) < 1e-3)
    assert st_bridge.pfl_m >= 105.10 - 0.02
    assert len(result.gradient_schedule) >= 2
    assert any(seg.direction == "RISE" for seg in result.gradient_schedule)


def test_parabolic_vc_hand_calculated_benchmark_and_proposed_gradients():
    """Hand-calculated IRPWM Para 409 summit VC: PFL(PVI) = E_PVI - |dg|*L/8."""
    stations = _make_stations(2000.0, 20.0, lambda _ch: 98.0)
    manual_pvis = [(0.0, 100.0), (1000.0, 105.0), (2000.0, 100.0)]
    solver = FormationLevelSolver()

    result = solver.solve(stations, manual_pvis=manual_pvis)

    assert result.mode == "INTERACTIVE_PVI"
    assert result.verification_status == "PASS", result.violations

    # Hand benchmark: g1 = +0.005, g2 = -0.005, |dg| = 0.010, Rv = 4000 -> max(60, 40) = 60m
    pvi_mid = result.pvis[1]
    assert pvi_mid.curve_type == "SUMMIT"
    assert pvi_mid.vc_length_m == pytest.approx(60.0, abs=1e-3)
    assert pvi_mid.pvc_chainage_m == pytest.approx(970.0, abs=1e-3)
    assert pvi_mid.pvt_chainage_m == pytest.approx(1030.0, abs=1e-3)

    # Exact parabolic midpoint elevation: 105.0 - (0.010 * 60.0) / 8.0 = 104.925m
    st_mid = next(s for s in result.stations if abs(s.chainage_m - 1000.0) < 1e-3)
    assert st_mid.in_vertical_curve is True
    assert st_mid.pfl_m == pytest.approx(104.925, abs=1e-4)
    assert st_mid.prl_m == pytest.approx(104.925 + 0.762, abs=1e-4)
    assert st_mid.proposed_gradient_decimal == pytest.approx(0.0, abs=1e-5)

    # Exact Proposed Gradient Schedule derived from PFL
    assert len(result.gradient_schedule) == 2
    assert result.gradient_schedule[0].gradient_label == "1 IN 200 R"
    assert result.gradient_schedule[1].gradient_label == "1 IN 200 F"
    assert result.gradient_schedule[0].vpi_chainage_m == pytest.approx(1000.0)
    assert result.gradient_schedule[0].vc_length_m == pytest.approx(60.0)


def test_design_from_pfl_reconstructs_proposed_gradients_and_vpi():
    """Station PFL series (with parabolic VC) cleanly reconstructs 2 tangents & VPI."""
    base_stations = _make_stations(2000.0, 20.0, lambda _ch: 98.0)
    solver = FormationLevelSolver()
    designed = solver.solve(base_stations, manual_pvis=[(0.0, 100.0), (1000.0, 105.0), (2000.0, 100.0)])

    # Feed only (chainage, egl, pfl) back into design_from_pfl()
    pfl_stations = [
        TerrainStation(chainage_m=s.chainage_m, egl_m=s.egl_m, pfl_m=s.pfl_m)
        for s in designed.stations
    ]
    reconstructed = solver.design_from_pfl(pfl_stations)

    assert reconstructed.mode == "PFL_DRIVEN_GRADIENT"
    assert reconstructed.verification_status == "PASS", reconstructed.violations
    assert len(reconstructed.gradient_schedule) == 2
    assert reconstructed.gradient_schedule[0].gradient_label == "1 IN 200 R"
    assert reconstructed.gradient_schedule[1].gradient_label == "1 IN 200 F"
    assert reconstructed.gradient_schedule[0].end_chainage_m == pytest.approx(1000.0, abs=1.0)
    assert reconstructed.gradient_schedule[1].start_chainage_m == pytest.approx(1000.0, abs=1.0)
    assert reconstructed.pvis[1].chainage_m == pytest.approx(1000.0, abs=1.0)
    assert reconstructed.pvis[1].elevation_m == pytest.approx(105.0, abs=0.05)


def test_station_zone_gradient_cap_pass_and_fail():
    """Station zone enforces <= 1 in 400 (0.25%) in auto mode and flags FAIL if breached."""
    stations = _make_stations(2000.0, 20.0, lambda ch: 100.0 + 0.002 * ch)
    stn_zone = StationZone(start_chainage_m=400.0, end_chainage_m=1200.0, name="Test Yard")
    solver = FormationLevelSolver()

    # Auto mode respects station cap
    auto_res = solver.solve(stations, station_zones=[stn_zone])
    assert auto_res.verification_status == "PASS", auto_res.violations
    for seg in auto_res.gradient_schedule:
        if seg.is_station_zone:
            assert abs(seg.gradient_decimal) <= 0.0025 + 1e-5

    # Manual PVI violating station cap (1 in 200 = 0.005 > 0.0025) must FAIL gate
    bad_res = solver.solve(
        stations,
        station_zones=[stn_zone],
        manual_pvis=[(0.0, 102.0), (2000.0, 112.0)],
    )
    assert bad_res.verification_status == "FAIL"
    assert any("COMPENSATED_RULING_GRADIENT" in v for v in bad_res.violations)


def test_drainage_violation_triggers_gate_fail():
    """Shallow embankment (0.20m < 0.60m drainage floor) must FAIL verification gate."""
    stations = _make_stations(1000.0, 20.0, lambda _ch: 100.0)
    solver = FormationLevelSolver()

    res = solver.solve(stations, manual_pvis=[(0.0, 100.20), (1000.0, 100.20)])
    assert res.verification_status == "FAIL"
    assert any("DRAINAGE_CLEARANCE" in v for v in res.violations)


def test_hill_cutting_and_tie_in_anchors():
    """Steep hill mound forces legal cutting with zero false drainage errors; tie-in anchors hold."""
    # Flat 100m plain with a 6m ridge centered at Ch 1200m (width 400m -> slope 3% > 1:150)
    def hill_egl(ch: float) -> float:
        dist = abs(ch - 1200.0)
        return 100.0 + max(0.0, 6.0 * (1.0 - dist / 200.0))

    stations = _make_stations(2400.0, 20.0, hill_egl)
    tie_in_start = VerticalConstraint(
        chainage_m=0.0,
        constraint_type="TIE_IN",
        name="Start EFL Junction",
        exact_pfl_m=101.20,
    )
    solver = FormationLevelSolver()
    result = solver.solve(stations, constraints=[tie_in_start])

    assert result.verification_status == "PASS", result.violations
    assert result.stations[0].pfl_m == pytest.approx(101.20, abs=1e-3)
    assert any(s.cutting_m > 0.5 for s in result.stations), "Expected cutting through steep ridge."
    assert any(s.bank_m >= 0.60 for s in result.stations), "Expected embankment across plains."


def test_invalid_inputs_raise_formation_level_error():
    """Non-monotonic chainages and too-few stations raise FormationLevelError."""
    solver = FormationLevelSolver()
    with pytest.raises(FormationLevelError):
        solver.solve([TerrainStation(chainage_m=0.0, egl_m=100.0)])

    with pytest.raises(FormationLevelError):
        solver.solve([
            TerrainStation(chainage_m=100.0, egl_m=100.0),
            TerrainStation(chainage_m=80.0, egl_m=100.0),
        ])
