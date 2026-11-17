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
from job_manager import job_manager
from alignment_models import DTMQueryRequest, JobStatus
from modules.survey_reader import BrownfieldSurveyReader


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

SYNTHETIC_SURVEY_CSV = """Chainage,Easting,Northing,Elevation
0,450000.0,2400000.0,25.5
20,450020.0,2400000.0,25.6
40,450040.0,2400000.0,25.7
60,450060.0,2400000.0,25.8
80,450080.0,2400000.0,25.9
100,450100.0,2400000.0,26.0
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


def test_survey_reader_synthetic():
    """Verify BrownfieldSurveyReader extracts centerline from CSV buffer."""
    buf = io.BytesIO(SYNTHETIC_SURVEY_CSV.encode("utf-8"))
    buf.name = "survey.csv"

    from alignment_router import CONFIG
    reader = BrownfieldSurveyReader(CONFIG)
    result = reader.extract_centerline_from_survey(
        survey_source=buf,
        start_chainage_m=0.0,
        chainage_interval_m=20.0,
    )
    assert result["total_length_m"] == pytest.approx(100.0, abs=1.0)
    assert len(result["stations"]) >= 5


def test_job_manager_lifecycle():
    """Verify in-memory job manager lifecycle."""
    job = job_manager.create_job(project_name="Test Corridor")
    assert job.status == JobStatus.QUEUED
    assert job.job_id.startswith("job_")

    job_manager.emit_event(job.job_id, "status", {"progress_percent": 50, "step": "Interpolating"})
    retrieved = job_manager.get_job(job.job_id)
    assert retrieved is not None
    assert retrieved.progress_percent == 50
    assert retrieved.current_step == "Interpolating"


def test_api_routes_registered():
    """Verify that the FastAPI app contains the expected endpoints."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert len(data["endpoints"]) == 6
