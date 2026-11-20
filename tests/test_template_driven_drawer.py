"""
Module: tests/test_template_driven_drawer.py
Purpose: Unit tests verifying that CompositeDrawer dynamically binds LSectionSheetTemplate
         for physical sheet sizing, data bands ordering & heights, datum elevation,
         and sidebar notes/title block rendering.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import copy
import ezdxf
import numpy as np
import pandas as pd
import pytest

from template_models import LSectionSheetTemplate
from modules.composite_drawer import CompositeDrawer
from greenfield.greenfield_router import BASE_CONFIG


def test_template_driven_composite_dxf(tmp_path: Path):
    """Verify that a custom LSectionSheetTemplate properly alters the generated DXF sheet dimensions,
    data bands, datum elevation, and right sidebar content.
    """
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

    # 2. Synthesize Excel data files
    lsec_csv = tmp_path / "lsec.csv"
    chs = np.arange(0, 2020, 20)
    pd.DataFrame({
        "chainage_m": chs,
        "elevation_m": 45.0 + 2.0 * np.sin(chs / 300.0),
        "formation_level_m": 48.0 + 0.003 * chs,
        "proposed_rail_level_m": 48.762 + 0.003 * chs,
    }).to_csv(lsec_csv, index=False)

    grad_xlsx = tmp_path / "gradients.xlsx"
    pd.DataFrame([
        {
            "Start Station": "0+000.00m",
            "End Station": "2+000.00m",
            "Start Elevation": "48.000m",
            "End Elevation": "54.000m",
            "Grade ": "1:300",
            "Gradient": "1:300",
            "Direction": "Rise",
        }
    ]).to_excel(grad_xlsx, index=False)

    # 3. Create Custom Template with specific 4 bands, custom dimensions and notes
    tpl_payload = {
        "id": "test-scr-a0-custom",
        "name": "SCR Custom A0 Template",
        "corridorType": "greenfield",
        "paper": "A0",
        "sheetWidthMm": 1189.0,
        "sheetHeightMm": 841.0,
        "sidebarWidthMm": 185.0,
        "planViewport": {
            "x": 25.0,
            "y": 20.0,
            "w": 964.0,
            "h": 360.0,
            "scale": 5000,
            "tickIntervalM": 100,
            "showNorthArrow": True,
        },
        "profileViewport": {
            "x": 25.0,
            "y": 390.0,
            "w": 964.0,
            "h": 220.0,
            "hScale": 5000,
            "vScale": 500,
            "datumRoundStepM": 5.0,
            "datumClearanceM": 20.0,
        },
        "dataBands": [
            {"key": "proposed_rail_level", "label": "CUSTOM PRL (M)", "heightMm": 18.0, "enabled": True},
            {"key": "existing_ground_level", "label": "CUSTOM EGL (M)", "heightMm": 18.0, "enabled": True},
            {"key": "cutting_filling", "label": "CUSTOM CUT/FILL", "heightMm": 18.0, "enabled": True},
            {"key": "gradient", "label": "CUSTOM GRADIENT & VC", "heightMm": 24.0, "enabled": True},
        ],
        "sidebar": {
            "notes": {
                "clauses": [
                    "1. SOUTH CENTRAL RAILWAY SPECIAL SPECIFICATION.",
                    "2. TRACK FORMATION WIDTH = 7.85 METRES.",
                ]
            },
            "titleBlock": {
                "zonalRailway": "SOUTH CENTRAL RAILWAY",
                "division": "SECUNDERABAD DIVISION",
                "projectName": "WADI-GUNTAKAL THIRD LINE",
                "drawingTitle": "DYNAMIC L-SECTION AND PLAN",
            },
        },
    }
    template = LSectionSheetTemplate(**tpl_payload)

    # 4. Configure pipeline
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
    cfg["drawing"]["sheet_length_m"] = 2000.0

    # 5. Instantiate CompositeDrawer with template
    drawer = CompositeDrawer(cfg, template=template)
    assert drawer.template is not None
    assert drawer.template.id == "test-scr-a0-custom"

    res = drawer.generate_from_inputs(output_prefix="test_template_run")
    assert res is not None
    comp_res = res.get("composite_result")
    assert comp_res is not None
    assert len(comp_res.dxf_paths) >= 1

    dxf_file = Path(comp_res.dxf_paths[0])
    assert dxf_file.exists()

    doc = ezdxf.readfile(str(dxf_file))
    msp = doc.modelspace()

    # Collect texts across drawing
    all_texts = [e.dxf.text for e in msp if e.dxftype() == "TEXT"]

    # Verify custom labels are present
    assert any("CUSTOM PRL" in t for t in all_texts), f"Expected 'CUSTOM PRL', got: {all_texts}"
    assert any("CUSTOM EGL" in t for t in all_texts), f"Expected 'CUSTOM EGL', got: {all_texts}"

    # Verify custom notes clauses from sidebar
    assert any("SOUTH CENTRAL RAILWAY SPECIAL SPECIFICATION" in t for t in all_texts)
    assert any("TRACK FORMATION WIDTH" in t for t in all_texts)

    # Verify custom title block
    assert any("SECUNDERABAD DIVISION" in t for t in all_texts)
    assert any("WADI-GUNTAKAL THIRD LINE" in t for t in all_texts)
    assert any("DYNAMIC L-SECTION AND PLAN" in t for t in all_texts)
