"""
Unit tests for alignment microservice.
Standards: RDSO / Indian Railways BG
Tests run with pure in-memory synthetic data without external network/file dependencies.
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

# Ensure alignment directory is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

import pytest
from fastapi.testclient import TestClient

from main import app
from alignment_router import _detect, router
from alignment_models import PlanProfileResponse


SYNTHETIC_KML = b"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Synthetic Corridor</name>
    <Placemark>
      <name>Main Line Alignment</name>
      <LineString>
        <coordinates>
          87.750000,21.780000,0
          87.755000,21.782000,0
          87.760000,21.788000,0
          87.765000,21.792000,0
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""


def test_detect_worker_synthetic():
    """Verify _detect parses KML, detects PIs, and computes geometry."""
    res = _detect(
        data=SYNTHETIC_KML,
        is_kmz=False,
        tolerance=5.0,
        angle=0.5,
        line_index=0,
        start_chainage_m=0.0,
    )
    assert res.detected_files is not None
    det = res.detected_files
    assert det["epsg"] == 32645
    assert det["total_length_m"] > 1000.0
    assert len(det["stations"]) > 10
    assert len(det["raw_lines"]) >= 3
    assert len(det["raw_vertices"]) == 4


def test_detect_api_endpoint():
    """Verify POST /detect endpoint parses uploaded KML and returns detection results synchronously."""
    client = TestClient(app)
    response = client.post(
        "/detect",
        files={"file": ("corridor.kml", io.BytesIO(SYNTHETIC_KML), "application/vnd.google-earth.kml+xml")},
        data={"simplification_tolerance_m": "5.0", "minimum_deflection_deg": "0.5"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "detected_files" in data
    det = data["detected_files"]
    assert det["epsg"] == 32645
    assert det["total_length_m"] > 1000.0
    assert len(det["stations"]) > 0


def test_api_routes_registered():
    """Verify that the FastAPI app contains the expected endpoints."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert any("/detect" in ep for ep in data["endpoints"])
    assert "POST /api/v1/alignment-studio/vertical/generate-formation" in data["endpoints"]
