"""
Module: tests/test_composite_dxf_generation.py
Purpose: Standalone unit test verifying that Composite Plan & Profile DXF generation
         renders BOTH Top Plan View, Bottom L-Section Profile View, and Right Sidebar
         adhering to Indian Railways / RDSO standards.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure alignment root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ezdxf
import numpy as np
import pandas as pd
import pytest

from modules.composite_drawer import CompositeDrawer
from modules.lsection_drawer import LSectionDrawer
from greenfield.greenfield_router import _synthesize_kml_from_band_points


def test_synthesize_kml_from_band_points(tmp_path: Path):
    """Verify KML synthesis when no external alignment KML URL exists."""
    df = pd.DataFrame({
        "chainage_m": [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        "elevation_m": [25.0, 25.5, 26.0, 26.5, 27.0, 27.5],
        "easting_m": [450000.0, 451000.0, 452000.0, 453000.0, 454000.0, 455000.0],
        "northing_m": [2400000.0, 2400050.0, 2400100.0, 2400150.0, 2400200.0, 2400250.0],
    })
    kml_path = tmp_path / "test_synth.kml"
    res = _synthesize_kml_from_band_points(df, kml_path)
    assert res.exists()
    content = res.read_text(encoding="utf-8")
    assert "<coordinates>" in content
    assert "<LineString>" in content
    assert len(content.splitlines()) > 5


def test_vertical_geometry_band_renders_slope_and_divider(tmp_path: Path):
    """Verify LSectionDrawer renders slope tangents, boundary dividers, and labels for all segments."""
    chs = np.arange(0, 5020, 20)
    prof_df = pd.DataFrame({
        "chainage_m": chs,
        "elevation_m": 50.0 + 3.0 * np.sin(chs / 600.0),
        "formation_level_m": 52.0 + 0.005 * chs,
        "proposed_rail_level_m": 52.762 + 0.005 * chs,
    })
    vip_df = pd.DataFrame([
        {
            "startChainage": 0.0,
            "endChainage": 2000.0,
            "slopeRatio": 200.0,
            "direction": "RISING",
            "startRl": 52.0,
            "endRl": 62.0,
        },
        {
            "startChainage": 2000.0,
            "endChainage": 5000.0,
            "slopeRatio": 0.0,
            "direction": "LEVEL",
            "startRl": 62.0,
            "endRl": 62.0,
        },
    ])

    drawer = LSectionDrawer({})
    doc, msp = drawer._setup_document()
    drawer._render_lsection_sheet(msp, prof_df, vip_df, pd.DataFrame(), 0.0, 5000.0, 1, 1)

    geom_entities = [e for e in msp if e.dxf.layer == "ESP_GEOM_TEXT"]
    assert len(geom_entities) >= 4  # Lines + texts for both segments

    texts = [e.dxf.text for e in geom_entities if e.dxftype() == "TEXT"]
    assert any("1 IN 200 RISE" in t for t in texts)
    assert any("LEVEL" in t for t in texts)


def test_composite_drawer_renders_plan_and_profile_views(tmp_path: Path):
    """Verify CompositeDrawer creates full composite DXF with Plan View, Profile, and Sidebar."""
    # 1. Synthesize minimal test KML
    kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Main Line</name>
      <LineString>
        <coordinates>
          87.750000,21.780000,0
          87.760000,21.785000,0
          87.770000,21.790000,0
          87.780000,21.795000,0
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    kml_file = tmp_path / "corridor.kml"
    kml_file.write_text(kml_content, encoding="utf-8")

    # 2. Synthesize Excel files
    lsec_csv = tmp_path / "lsec.csv"
    chs = np.arange(0, 3020, 20)
    pd.DataFrame({
        "chainage_m": chs,
        "elevation_m": 20.0 + 2.0 * np.sin(chs / 400.0),
        "formation_level_m": 22.0 + 0.004 * chs,
        "proposed_rail_level_m": 22.762 + 0.004 * chs,
    }).to_csv(lsec_csv, index=False)

    grad_xlsx = tmp_path / "gradients.xlsx"
    pd.DataFrame([
        {
            "Start Station": "0+000.00m",
            "End Station": "3+000.00m",
            "Start Elevation": "22.000m",
            "End Elevation": "34.000m",
            "Grade ": "1:250",
            "Gradient": "1:250",
            "Direction": "Rise",
        }
    ]).to_excel(grad_xlsx, index=False)

    from greenfield.greenfield_router import BASE_CONFIG
    import copy

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg.setdefault("paths", {})["kml_path"] = str(kml_file)
    cfg["paths"]["lsection_excel_path"] = str(lsec_csv)
    cfg["paths"]["gradient_excel_path"] = str(grad_xlsx)
    cfg["paths"]["curve_excel_path"] = None
    cfg["paths"]["bridge_list_path"] = None
    cfg["paths"]["topo_path"] = None
    cfg["paths"]["station_excel_path"] = None
    cfg["paths"]["outputs_dir"] = str(tmp_path / "outputs")
    cfg.setdefault("drawing", {})["generate_complete_profile"] = False
    cfg["drawing"]["generate_sheet_drawings"] = True
    cfg["drawing"]["export_pdf"] = False
    cfg["drawing"]["sheet_length_m"] = 3000.0

    drawer = CompositeDrawer(cfg)
    res = drawer.generate_from_inputs(output_prefix="test_composite")
    assert res is not None
    comp_res = res.get("composite_result")
    assert comp_res is not None
    assert len(comp_res.dxf_paths) >= 1

    dxf_file = Path(comp_res.dxf_paths[0])
    assert dxf_file.exists()

    doc = ezdxf.readfile(str(dxf_file))
    msp = doc.modelspace()

    # Collect layers
    layers = {e.dxf.layer for e in msp}

    # Verify Plan View layers exist
    plan_layers = [l for l in layers if "PLAN" in l]
    assert len(plan_layers) > 0, f"Expected Plan View layers, found: {sorted(layers)}"

    # Verify Profile layers exist
    prof_layers = [l for l in layers if "PROF" in l or "GEOM" in l]
    assert len(prof_layers) > 0, f"Expected Profile layers, found: {sorted(layers)}"

    # Verify Border and Title Block exist
    assert "BORDER" in layers or "SHEET_BORDER" in layers
    assert "TITLE_BLOCK" in layers
