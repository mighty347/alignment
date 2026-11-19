"""
Unit tests for alignment/vertical/bridge_constraint_loader.py and Bridge_List.xlsx
column-header integration inside FormationLevelSolver.
Runs with 100% synthetic in-memory data (no external files required).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ALIGNMENT_ROOT = Path(__file__).resolve().parent.parent
if str(_ALIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALIGNMENT_ROOT))

from vertical import (
    FormationLevelSolver,
    TerrainStation,
    classify_bridge_row,
    load_bridge_constraints_from_records,
    parse_span_length_m,
    resolve_bridge_chainage_m,
)


def _flat_stations(end_ch_m: float, egl_m: float = 3.0, step_m: float = 20.0) -> list[TerrainStation]:
    stations: list[TerrainStation] = []
    ch = 0.0
    while ch <= end_ch_m + 1e-6:
        stations.append(TerrainStation(chainage_m=round(ch, 3), egl_m=egl_m))
        ch += step_m
    return stations


def test_parse_span_length_m_simple_and_compound():
    """Verify simple and compound Indian Railways span strings parse accurately."""
    assert parse_span_length_m("1X1.40") == pytest.approx(1.40)
    assert parse_span_length_m("1X18.30") == pytest.approx(18.30)
    assert parse_span_length_m("2x24.4+2X12.20Composite") == pytest.approx(73.20)
    assert parse_span_length_m("1X6.1PSC") == pytest.approx(6.10)
    assert parse_span_length_m("L-Xing\"C\"Class") == pytest.approx(0.0)
    assert parse_span_length_m(None) == pytest.approx(0.0)


def test_resolve_bridge_chainage_m_guards_km_header_holding_meters():
    """'Approx. Chainage (km)' holding 871.23m stays meters; 1.786km converts to 1786m."""
    assert resolve_bridge_chainage_m(871.23, 0.0, 25000.0, "Approx. Chainage (km)") == pytest.approx(871.23)
    assert resolve_bridge_chainage_m(1.786, 0.0, 25000.0, "Approx. Chainage (km)") == pytest.approx(1786.0)


def test_classify_all_bridge_list_types():
    """Every Bridge Type & Configration observed in Bridge_List.xlsx maps cleanly."""
    assert classify_bridge_row("MINOR", "H.P", "1X1.40", "CH-871.23,BRNo-111") == ("MINOR_BRIDGE", "HP")
    assert classify_bridge_row("Minor", "RCCBOX", "1X3.00", "BRNo-1") == ("MINOR_BRIDGE", "RCCBOX")
    assert classify_bridge_row("RUB", "RCCBOX", "1X6.00", "BRNo-2") == ("RUB_LOCAL", "RCCBOX")
    assert classify_bridge_row("minor", "PSCSLAB", "1X6.1", "BRNo-3") == ("MINOR_BRIDGE", "PSCSLAB")
    assert classify_bridge_row("major", "CompositeGirder", "1X18.30", "BRNo-12") == ("MAJOR_BRIDGE", "COMPOSITE_GIRDER")
    assert classify_bridge_row("Major", "Girder+PSC", "2x24.4+2X12.20Composite", "BRNo-20") == ("MAJOR_BRIDGE", "GIRDER_PSC")
    assert classify_bridge_row("LC", "Gate", "L-Xing\"C\"Class", "LC 1") == ("LC", "NONE")
    assert classify_bridge_row("LC", "Xing\"C\"class", "LC", "LC 2") == ("LC", "NONE")


def test_solver_proposes_formation_level_from_exact_bridge_list_headers():
    """End-to-end formation level + proposed gradient generation from Bridge_List.xlsx headers."""
    # Exact column headers from D:\ESP-Backup\inputs\Sample\Bridge_List.xlsx
    bridge_rows = [
        {
            "S.No": 1,
            "Bridge No.": "BRNo-111",
            "Bridge Type": "H.P",
            "Approx. Chainage (km)": 800.0,
            "Proposed Span": "1X1.40",
            "Configration": "MINOR",
            "Bridge Type / Description": "H.P",
            "Bridge Detail": "CH-800.0,BRNo-111,1X1.40MINOR",
            "Bed Level": 3.1561,
            "HFL": 3.451,
            "Clearance": 0.6,
        },
        {
            "S.No": 2,
            "Bridge No.": "LC 1",
            "Bridge Type": "Gate",
            "Approx. Chainage (km)": 1800.0,
            "Proposed Span": "L-Xing\"C\"Class",
            "Configration": "LC",
            "Bridge Type / Description": "Gate",
            "Bridge Detail": "CH-1800.0,LC 1,L-Xing\"C\"ClassLC",
            "Bed Level": None,
            "HFL": None,
            "Clearance": None,
        },
        {
            "S.No": 3,
            "Bridge No.": "BRNo-12",
            "Bridge Type": "CompositeGirder",
            "Approx. Chainage (km)": 3200.0,
            "Proposed Span": "1X18.30",
            "Configration": "major",
            "Bridge Type / Description": "CompositeGirder",
            "Bridge Detail": "CH-3200.0,BRNo-12,1X18.30major",
            "Bed Level": 2.3288,
            "HFL": 3.242,
            "Clearance": 0.6,
        },
        {
            "S.No": 4,
            "Bridge No.": "BRNo-2",
            "Bridge Type": "RCCBOX",
            "Approx. Chainage (km)": 4600.0,
            "Proposed Span": "1X6.00",
            "Configration": "RUB",
            "Bridge Type / Description": "RCCBOX",
            "Bridge Detail": "CH-4600.0,BRNo-2,1X6.00RUB",
            "Bed Level": None,
            "HFL": None,
            "Clearance": None,
        },
    ]

    constraints, flat_zones = load_bridge_constraints_from_records(bridge_rows, 0.0, 5600.0)
    assert len(constraints) == 3  # BRNo-111, BRNo-12, BRNo-2 (LC is a flat zone, not a lift)
    assert len(flat_zones) == 2   # LC 1 (±30m @ 1:1000) + BRNo-12 approach (±9.15+30m @ 1:400)

    stations = _flat_stations(5600.0, egl_m=3.0, step_m=20.0)
    solver = FormationLevelSolver()
    result = solver.solve(stations, bridge_records=bridge_rows)

    assert result.verification_status == "PASS", result.violations
    assert len(result.gradient_schedule) >= 2

    # 1. H.P culvert @ Ch 800: max(HFL 3.451 + 0.6 + 0.5 = 4.551, Bed 3.1561 + 1.4 + 0.5 = 5.0561)
    st_hp = next(s for s in result.stations if abs(s.chainage_m - 800.0) < 1e-3)
    assert st_hp.pfl_m >= 5.0561 - 0.02

    # 2. CompositeGirder MAJOR @ Ch 3200 (span 18.3m): HFL 3.242 + 0.6 + 1.80 = 5.642m across deck
    st_major = next(s for s in result.stations if abs(s.chainage_m - 3200.0) < 1e-3)
    assert st_major.pfl_m >= 5.642 - 0.02

    # 3. RUB @ Ch 4600 (HFL=None -> EGL 3.0 + 5.0 = 8.0m)
    st_rub = next(s for s in result.stations if abs(s.chainage_m - 4600.0) < 1e-3)
    assert st_rub.pfl_m >= 8.00 - 0.02

    # 4. LC 1 @ Ch 1800: active gradient must obey <= 1 in 1000 (0.001)
    st_lc = next(s for s in result.stations if abs(s.chainage_m - 1800.0) < 1e-3)
    assert abs(st_lc.proposed_gradient_decimal) <= 0.001 + 1e-4
