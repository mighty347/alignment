"""
Standalone synthetic tests for the Greenfield Composite Microservice (greenfield/).
Runs completely independently of plan_profile_generator/ with zero live DB or S3.
"""
from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient

_ALIGN_ROOT = Path(__file__).resolve().parent.parent
if str(_ALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALIGN_ROOT))

import main as gf_main
from greenfield import greenfield_router as gf_router
from modules.greenfield_db_repository import (
    GreenfieldDBBundle,
    GreenfieldProjectMeta,
    GreenfieldVersionMeta,
    SheetWriteRecord,
    VersionNotFoundError,
)


def _build_bundle(
    version_id: str,
    verification_status: str = "PASS",
    warnings: Optional[list[str]] = None,
) -> GreenfieldDBBundle:
    ls_id = str(uuid.uuid4())
    proj_id = str(uuid.uuid4())
    return GreenfieldDBBundle(
        project_meta=GreenfieldProjectMeta(
            project_id=proj_id,
            project_name="Kanti - Egra Greenfield BG",
            railway_zone="South Eastern Railway",
            division="Kharagpur",
            line_type="GREENFIELD",
            gauge="BG",
            lsection_id=ls_id,
            start_chainage_m=0.0,
            end_chainage_m=5000.0,
        ),
        version_meta=GreenfieldVersionMeta(
            version_id=version_id,
            lsection_id=ls_id,
            name="V1",
            status="DRAFT",
            band_headers=[
                {"id": "egl", "title": "Ground Level (M)", "source_column": "elevation_m", "height_mm": 18.0}
            ],
            alignment_kml_url="s3://demo-bucket/alignment.kml",
            topo_kml_url=None,
            dxf_url=None,
        ),
        band_points_df=pd.DataFrame([
            {"chainage_m": 0.0, "elevation_m": 16.4, "formation_level_m": 18.0, "proposed_rail_level_m": 18.762},
            {"chainage_m": 2500.0, "elevation_m": 15.8, "formation_level_m": 18.5, "proposed_rail_level_m": 19.262},
            {"chainage_m": 5000.0, "elevation_m": 17.0, "formation_level_m": 19.0, "proposed_rail_level_m": 19.762},
        ]),
        curves_df=pd.DataFrame([
            {
                "curve_no": "C-01",
                "track_type": "PROPOSED",
                "track_name": "Main Line",
                "ts_chainage_m": 1000.0,
                "sc_chainage_m": 1080.0,
                "cs_chainage_m": 1420.0,
                "st_chainage_m": 1500.0,
                "total_length_m": 500.0,
                "circular_length_m": 340.0,
                "transition_length_m": 80.0,
                "radius_m": 1750.0,
                "degree": 1.0,
                "deflection_deg": 12.5,
                "cant_mm": 75.0,
                "direction": "RIGHT",
            }
        ]),
        gradients_df=pd.DataFrame([
            {
                "seg_no": 1,
                "start_chainage_m": 0.0,
                "end_chainage_m": 5000.0,
                "length_m": 5000.0,
                "slope_ratio": 200.0,
                "direction": "RISING",
                "start_rl_m": 18.0,
                "end_rl_m": 43.0,
                "vpi_chainage_m": 5000.0,
                "vpi_elevation_m": 43.0,
                "vc_length_m": 100.0,
            }
        ]),
        stations_df=pd.DataFrame([
            {"name": "KANTI", "chainage_m": 500.0, "location_km": 0.5, "station_class": "B"}
        ]),
        bridges_df=pd.DataFrame([
            {
                "Bridge No": "BR-01",
                "Bridge Name": "Canal Bridge",
                "Chainage": 2200.0,
                "Bridge Type": "Minor Bridge",
                "Status": "PROPOSED",
                "Bed Level": 12.0,
                "HFL": 14.5,
                "Rail Level": 19.262,
                "Formation Level": 18.5,
            }
        ]),
        warnings=warnings or [],
        verification_status=verification_status,
    )


@dataclass
class _FakeCompositeResult:
    dxf_path: Path
    dxf_paths: list[Path] = field(default_factory=list)
    pdf_paths: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class _FakeStorage:
    def download_file(self, source: str, dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("<kml/>", encoding="utf-8")
        return dest_path

    def upload_file(self, local_path: Path, bucket: str, key: str) -> str:
        return f"s3://{bucket}/{key}"


class _FakeDrawer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def generate_from_inputs(self, output_prefix: str = "composite") -> dict[str, Any]:
        out_dir = Path(self.cfg["paths"]["composite_outputs_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        dxf_p = out_dir / f"{output_prefix}_sheet_1.dxf"
        dxf_p.write_text("0\nEOF\n", encoding="utf-8")
        return {
            "composite_result": _FakeCompositeResult(
                dxf_path=dxf_p,
                dxf_paths=[dxf_p],
                pdf_paths=[],
                metadata={
                    "sheets": [
                        {
                            "sheet_index": 1,
                            "start_ch": 0.0,
                            "end_ch": 5000.0,
                            "dxf_path": str(dxf_p),
                        }
                    ]
                },
            )
        }


def test_health_and_root_endpoints() -> None:
    with TestClient(gf_main.app) as client:
        assert client.get("/health").json() == {"status": "HEALTHY"}
        root_data = client.get("/").json()
        assert root_data["status"] == "ONLINE"
        endpoints_joined = "\n".join(root_data["endpoints"])
        assert "/api/v1/alignment-studio/detect" in endpoints_joined
        assert "/api/v1/greenfield/composite/generate" in endpoints_joined
        assert "/api/v1/greenfield/composite/jobs/{job_id}" in endpoints_joined


def test_preview_endpoint_exact_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    vid = str(uuid.uuid4())
    missing_vid = str(uuid.uuid4())
    bundle = _build_bundle(vid)

    class _FakeRepo:
        def load_version_bundle(self, version_id: str) -> GreenfieldDBBundle:
            if version_id == missing_vid:
                raise VersionNotFoundError("Not found")
            return bundle

    monkeypatch.setattr(gf_router, "_create_repository", lambda cfg: _FakeRepo())

    with TestClient(gf_main.app) as client:
        assert client.get("/api/v1/greenfield/composite/preview/bad").status_code == 400
        assert client.get(f"/api/v1/greenfield/composite/preview/{missing_vid}").status_code == 404

        resp = client.get(f"/api/v1/greenfield/composite/preview/{vid}")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"version_id", "ready_to_generate", "verification_status", "warnings"}
        assert body["ready_to_generate"] is True


def test_generate_and_poll_standalone_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vid = str(uuid.uuid4())
    bundle = _build_bundle(vid)
    saved_records: list[SheetWriteRecord] = []

    class _FakeRepo:
        def load_version_bundle(self, version_id: str) -> GreenfieldDBBundle:
            return bundle

        def save_generated_sheets(
            self,
            version_id: str,
            sheets: list[SheetWriteRecord],
            master_dxf_url: Optional[str] = None,
            version_status: Optional[str] = None,
            updated_by: Optional[str] = None,
        ) -> list[str]:
            saved_records.extend(sheets)
            return [str(uuid.uuid4()) for _ in sheets]

    monkeypatch.setattr(gf_router, "_create_repository", lambda cfg: _FakeRepo())
    monkeypatch.setattr(gf_router, "_create_storage_manager", lambda cfg: _FakeStorage())
    monkeypatch.setattr(gf_router, "_create_composite_drawer", lambda cfg: _FakeDrawer(cfg))
    monkeypatch.setattr(gf_router, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(gf_router, "TEMP_UPLOADS_DIR", tmp_path / "temp")

    with TestClient(gf_main.app) as client:
        r_acc = client.post(
            "/api/v1/greenfield/composite/generate",
            json={"version_id": vid, "s3_output_bucket": "esp-out", "s3_output_prefix": "gf/run"},
        )
        assert r_acc.status_code == 202
        acc = r_acc.json()
        assert set(acc.keys()) == {"job_id", "version_id", "status", "status_url", "stream_url"}

        status_data: dict[str, Any] = {}
        for _ in range(50):
            status_data = client.get(acc["status_url"]).json()
            if status_data["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)

    assert status_data["status"] == "completed", status_data.get("error")
    assert set(status_data.keys()) == {
        "job_id",
        "version_id",
        "status",
        "progress_percent",
        "verification_status",
        "warnings",
        "sheets",
        "error",
    }
    assert len(status_data["sheets"]) == 1
    assert status_data["sheets"][0]["datum_rl_m"] == pytest.approx(10.0)
    assert status_data["sheets"][0]["dxf_url"].startswith("s3://esp-out/gf/run/")
    assert len(saved_records) == 1


def test_generate_blocks_when_engineering_gate_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vid = str(uuid.uuid4())
    bundle = _build_bundle(
        vid,
        verification_status="FAIL",
        warnings=["FAIL: Track buildup invariant violated."],
    )

    class _FakeRepo:
        def load_version_bundle(self, version_id: str) -> GreenfieldDBBundle:
            return bundle

    monkeypatch.setattr(gf_router, "_create_repository", lambda cfg: _FakeRepo())
    monkeypatch.setattr(gf_router, "_create_storage_manager", lambda cfg: _FakeStorage())
    monkeypatch.setattr(gf_router, "_create_composite_drawer", lambda cfg: _FakeDrawer(cfg))
    monkeypatch.setattr(gf_router, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(gf_router, "TEMP_UPLOADS_DIR", tmp_path / "temp")

    with TestClient(gf_main.app) as client:
        r_acc = client.post("/api/v1/greenfield/composite/generate", json={"version_id": vid})
        assert r_acc.status_code == 202
        status_url = r_acc.json()["status_url"]

        status_data: dict[str, Any] = {}
        for _ in range(50):
            status_data = client.get(status_url).json()
            if status_data["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)

    assert status_data["status"] == "failed"
    assert status_data["verification_status"] == "FAIL"
    assert "Engineering Verification Gate FAILED" in (status_data.get("error") or "")

