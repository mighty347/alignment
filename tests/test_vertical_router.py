"""
FastAPI TestClient tests for alignment/vertical_router.py.
Tests the single unified endpoint POST /api/v1/alignment-studio/vertical/generate-formation
combining Ground Level (EGL), Existing Rail Level (ERL), Railway Stations, Bridge List,
and Horizontal Curves with 100% synthetic data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ALIGNMENT_ROOT = Path(__file__).resolve().parent.parent
if str(_ALIGNMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALIGNMENT_ROOT))

from main import app

client = TestClient(app)


def _build_station_payload(
    end_ch_m: float,
    egl_fn,
    step_m: float = 20.0,
    start_erl: float | None = None,
    end_erl: float | None = None,
) -> list[dict]:
    rows: list[dict] = []
    ch = 0.0
    while ch <= end_ch_m + 1e-6:
        row: dict = {"chainage_m": round(ch, 3), "egl_m": round(float(egl_fn(ch)), 4)}
        if start_erl is not None and abs(ch) < 1e-6:
            row["erl_m"] = start_erl
        if end_erl is not None and abs(ch - end_ch_m) < 1e-6:
            row["erl_m"] = end_erl
        rows.append(row)
        ch += step_m
    return rows


def test_unified_endpoint_all_five_inputs_pass():
    """Single endpoint combines EGL, ERL tie-ins, Railway Stations, Bridge List & Curves."""
    # 6 km corridor across 3.0m plain; start ERL=4.762 (EFL=4.0), end ERL=5.762 (EFL=5.0)
    stations = _build_station_payload(
        end_ch_m=6000.0,
        egl_fn=lambda _ch: 3.0,
        step_m=20.0,
        start_erl=4.762,
        end_erl=5.762,
    )

    # Exact Station_Excel.xlsx column headers
    railway_stations = [
        {
            "S.NO.": 1,
            "Name of station": "Pansar station",
            "Location (km)": 0.0,
            "Class of station": "B",
            "Intermediate distance": 0.0,
        },
        {
            "S.NO.": 2,
            "Name of station": "Midway Station",
            "Location (km)": 3.0,
            "Class of station": "B",
            "Intermediate distance": 3.0,
        },
    ]

    # Exact Bridge_List.xlsx column headers
    bridge_list = [
        {
            "S.No": 1,
            "Bridge No.": "BRNo-111",
            "Bridge Type": "H.P",
            "Approx. Chainage (km)": 900.0,
            "Proposed Span": "1X1.40",
            "Configration": "MINOR",
            "Bridge Type / Description": "H.P",
            "Bridge Detail": "CH-900.0,BRNo-111,1X1.40MINOR",
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
            "Approx. Chainage (km)": 4200.0,
            "Proposed Span": "1X18.30",
            "Configration": "major",
            "Bridge Type / Description": "CompositeGirder",
            "Bridge Detail": "CH-4200.0,BRNo-12,1X18.30major",
            "Bed Level": 2.3288,
            "HFL": 3.242,
            "Clearance": 0.6,
        },
        {
            "S.No": 4,
            "Bridge No.": "BRNo-2",
            "Bridge Type": "RCCBOX",
            "Approx. Chainage (km)": 5000.0,
            "Proposed Span": "1X6.00",
            "Configration": "RUB",
            "Bridge Type / Description": "RCCBOX",
            "Bridge Detail": "CH-5000.0,BRNo-2,1X6.00RUB",
            "Bed Level": None,
            "HFL": None,
            "Clearance": None,
        },
    ]

    # Direct /detect curve key format (chainage_ts / chainage_st)
    horizontal_curves = [
        {
            "curve_id": "C1",
            "chainage_ts": 1000.0,
            "chainage_sc": 1100.0,
            "chainage_cs": 1500.0,
            "chainage_st": 1600.0,
            "radius_m": 1000.0,
        }
    ]

    resp = client.post(
        "/api/v1/alignment-studio/vertical/generate-formation",
        json={
            "stations": stations,
            "railway_stations": railway_stations,
            "bridge_list": bridge_list,
            "horizontal_curves": horizontal_curves,
            "mode": "auto",
            "terrain_type": "plains",
            "tie_in_to_existing": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["mode"] == "AUTO_OPTIMIZER"
    assert data["verification_status"] == "PASS", data["violations"]
    assert len(data["gradient_schedule"]) >= 2

    # Existing Rail Level tie-in at Start & End: PRL == ERL (diff_rail_level_m == 0)
    st_start = data["stations"][0]
    st_end = data["stations"][-1]
    assert st_start["erl_m"] == pytest.approx(4.762)
    assert st_start["efl_m"] == pytest.approx(4.0)
    assert st_start["pfl_m"] == pytest.approx(4.0, abs=0.02)
    assert st_start["diff_rail_level_m"] == pytest.approx(0.0, abs=0.02)

    assert st_end["erl_m"] == pytest.approx(5.762)
    assert st_end["efl_m"] == pytest.approx(5.0)
    assert st_end["pfl_m"] == pytest.approx(5.0, abs=0.02)
    assert st_end["diff_rail_level_m"] == pytest.approx(0.0, abs=0.02)

    # Railway Station Yard (Midway @ 3000m, ±500m) must stay <= 1 in 400 (0.0025)
    st_yard = next(s for s in data["stations"] if abs(s["chainage_m"] - 3000.0) < 1e-3)
    assert abs(st_yard["proposed_gradient_decimal"]) <= 0.0025 + 1e-4

    # LC 1 @ 1800m must stay <= 1 in 1000 (0.001)
    st_lc = next(s for s in data["stations"] if abs(s["chainage_m"] - 1800.0) < 1e-3)
    assert abs(st_lc["proposed_gradient_decimal"]) <= 0.001 + 1e-4

    # Major Bridge @ 4200m (>= 5.642m) & RUB @ 5000m (>= 8.0m) cleared
    st_major = next(s for s in data["stations"] if abs(s["chainage_m"] - 4200.0) < 1e-3)
    st_rub = next(s for s in data["stations"] if abs(s["chainage_m"] - 5000.0) < 1e-3)
    assert st_major["pfl_m"] >= 5.642 - 0.02
    assert st_rub["pfl_m"] >= 8.00 - 0.02


def test_unified_endpoint_interactive_pvi_and_from_pfl_roundtrip():
    """Interactive PVI computes parabolic VC; from_pfl reconstructs exact Proposed Gradients."""
    stations = _build_station_payload(2000.0, lambda _ch: 98.0)
    resp_pvi = client.post(
        "/api/v1/alignment-studio/vertical/generate-formation",
        json={
            "stations": stations,
            "mode": "interactive_pvi",
            "manual_pvis": [
                {"chainage_m": 0.0, "elevation_m": 100.0},
                {"chainage_m": 1000.0, "elevation_m": 105.0},
                {"chainage_m": 2000.0, "elevation_m": 100.0},
            ],
        },
    )
    assert resp_pvi.status_code == 200, resp_pvi.text
    pvi_data = resp_pvi.json()
    assert pvi_data["verification_status"] == "PASS"
    st_mid = next(s for s in pvi_data["stations"] if abs(s["chainage_m"] - 1000.0) < 1e-3)
    assert st_mid["pfl_m"] == pytest.approx(104.925, abs=1e-4)

    # Round-trip station PFLs back into mode='from_pfl'
    pfl_stations = [
        {"chainage_m": s["chainage_m"], "egl_m": s["egl_m"], "pfl_m": s["pfl_m"]}
        for s in pvi_data["stations"]
    ]
    resp_grad = client.post(
        "/api/v1/alignment-studio/vertical/generate-formation",
        json={"stations": pfl_stations, "mode": "from_pfl"},
    )
    assert resp_grad.status_code == 200, resp_grad.text
    grad_data = resp_grad.json()
    assert grad_data["verification_status"] == "PASS"
    assert len(grad_data["gradient_schedule"]) == 2
    assert grad_data["gradient_schedule"][0]["gradient_label"] == "1 IN 200 R"
    assert grad_data["gradient_schedule"][1]["gradient_label"] == "1 IN 200 F"


def test_unified_endpoint_validation_errors():
    """Invalid mode payloads and non-monotonic chainages return HTTP 422."""
    # Missing manual_pvis for interactive_pvi mode
    r1 = client.post(
        "/api/v1/alignment-studio/vertical/generate-formation",
        json={
            "stations": [{"chainage_m": 0.0, "egl_m": 100.0}, {"chainage_m": 100.0, "egl_m": 100.0}],
            "mode": "interactive_pvi",
        },
    )
    assert r1.status_code == 422

    # Missing pfl_m for from_pfl mode
    r2 = client.post(
        "/api/v1/alignment-studio/vertical/generate-formation",
        json={
            "stations": [{"chainage_m": 0.0, "egl_m": 100.0}, {"chainage_m": 100.0, "egl_m": 100.0}],
            "mode": "from_pfl",
        },
    )
    assert r2.status_code == 422

    # Non-monotonic station chainages
    r3 = client.post(
        "/api/v1/alignment-studio/vertical/generate-formation",
        json={
            "stations": [{"chainage_m": 100.0, "egl_m": 100.0}, {"chainage_m": 50.0, "egl_m": 100.0}],
            "mode": "auto",
        },
    )
    assert r3.status_code == 422

