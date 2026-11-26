"""
Module: greenfield/greenfield_router.py
Purpose: Self-contained FastAPI APIRouter for DB-driven Greenfield Composite
         Plan & L-Section sheet generation.

Endpoints:
    GET  /api/v1/greenfield/composite/preview/{version_id}
    POST /api/v1/greenfield/composite/generate
    GET  /api/v1/greenfield/composite/jobs/{job_id}
    GET  /api/v1/greenfield/composite/jobs/{job_id}/stream

Standards: RDSO GE:G-1 / IRPWM / Indian Railways SOD (BG)
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import shutil
import socket
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from .greenfield_models import (
    GreenfieldCompositeGenerateRequest,
    GreenfieldGenerateAcceptedResponse,
    GreenfieldJobStatusResponse,
    GreenfieldPreviewResponse,
    GreenfieldSheetItem,
    JobStatus,
)
from .job_manager import job_manager
from modules.composite_drawer import CompositeDrawer
from modules.greenfield_db_repository import (
    GreenfieldDBBundle,
    GreenfieldDBError,
    GreenfieldDBRepository,
    SheetWriteRecord,
    VersionNotFoundError,
    _validate_uuid,
)
from utils.logger import get_logger
from utils.s3_storage import S3StorageManager

logger = get_logger("GreenfieldRouter")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
OUTPUTS_DIR = BASE_DIR / "outputs"
TEMP_UPLOADS_DIR = OUTPUTS_DIR / "temp_uploads"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

with CONFIG_PATH.open("r", encoding="utf-8") as _cfg_fh:
    BASE_CONFIG: dict[str, Any] = yaml.safe_load(_cfg_fh)

router = APIRouter(prefix="/api/v1/greenfield/composite", tags=["Greenfield Composite Plan & Profile (DB)"])
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


# ---------------------------------------------------------------------------
# Factory Hooks (Overridable in Unit Tests)
# ---------------------------------------------------------------------------

def _create_repository(cfg: dict[str, Any]) -> GreenfieldDBRepository:
    return GreenfieldDBRepository(cfg)


def _create_storage_manager(cfg: dict[str, Any]) -> S3StorageManager:
    s3_cfg = cfg.get("s3", {})
    return S3StorageManager(
        region_name=s3_cfg.get("region_name"),
        download_timeout_s=int(s3_cfg.get("download_timeout_s", 300)),
    )


def _create_composite_drawer(cfg: dict[str, Any], template: Optional[Any] = None) -> CompositeDrawer:
    return CompositeDrawer(cfg, template=template)


# ---------------------------------------------------------------------------
# Pure Helpers: URL Formatting, Datum Math, Staging & Config Construction
# ---------------------------------------------------------------------------

def _get_server_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 1))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _format_local_url(rel_path: str, request: Optional[Request] = None) -> str:
    clean_path = rel_path.replace("\\", "/").lstrip("/")
    if request is not None:
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        proto = request.headers.get("x-forwarded-proto") or getattr(request.url, "scheme", "http")
        if host:
            return f"{proto}://{host}/{clean_path}"
    return f"http://{_get_server_ip()}:8001/{clean_path}"


def _compute_sheet_datum_rl(
    band_points_df: pd.DataFrame,
    start_ch_m: float,
    end_ch_m: float,
    datum_margin_m: float,
    datum_round_m: float,
) -> float:
    """Compute RDSO profile datum RL: floor((min(EGL) - margin) / round) * round."""
    if band_points_df.empty or "elevation_m" not in band_points_df.columns:
        return 0.0

    mask = (
        (band_points_df["chainage_m"] >= start_ch_m)
        & (band_points_df["chainage_m"] <= end_ch_m)
        & band_points_df["elevation_m"].notna()
    )
    elevs = band_points_df.loc[mask, "elevation_m"]
    if elevs.empty:
        elevs = band_points_df["elevation_m"].dropna()
    if elevs.empty:
        return 0.0

    min_egl = float(elevs.min())
    step = datum_round_m if datum_round_m > 0.0 else 10.0
    return float(math.floor((min_egl - datum_margin_m) / step) * step)


def _format_km_plus_m(ch_m: float) -> str:
    km = int(ch_m // 1000.0)
    rem = ch_m - (km * 1000.0)
    return f"{km}+{rem:06.2f}m"


def _synthesize_kml_from_band_points(band_points_df: pd.DataFrame, target_path: Path) -> Path:
    """Generate a valid KML LineString file from station coordinates or chainages for Plan View rendering."""
    coords: list[str] = []
    if not band_points_df.empty and "easting_m" in band_points_df.columns and "northing_m" in band_points_df.columns:
        try:
            import pyproj
            transformer = pyproj.Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
            for _, r in band_points_df.sort_values("chainage_m").iterrows():
                e = float(r["easting_m"])
                n = float(r["northing_m"])
                lon, lat = transformer.transform(e, n)
                coords.append(f"{lon:.7f},{lat:.7f},0")
        except Exception:
            coords = []

    if not coords:
        lon0 = 87.500000
        lat0 = 22.000000
        chs = (
            band_points_df["chainage_m"].tolist()
            if not band_points_df.empty and "chainage_m" in band_points_df.columns
            else [0.0, 5000.0]
        )
        for ch in chs:
            lon = lon0 + float(ch) * 0.0000097
            coords.append(f"{lon:.7f},{lat0:.7f},0")

    coord_str = "\n          ".join(coords)
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Alignment Centerline</name>
    <Placemark>
      <name>Track Alignment</name>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
          {coord_str}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(kml_content, encoding="utf-8")
    return target_path


def _stage_bundle_dataframes(
    bundle: GreenfieldDBBundle,
    staging_dir: Path,
    default_design_speed_kmh: float,
) -> dict[str, Optional[str]]:
    """Serialize DB DataFrames into session files matching CompositeDrawer contracts."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Optional[str]] = {
        "lsection_excel_path": None,
        "curve_excel_path": None,
        "gradient_excel_path": None,
        "station_excel_path": None,
        "bridge_excel_path": None,
    }

    if not bundle.band_points_df.empty:
        lsec_csv = staging_dir / "db_lsection_band_points.csv"
        bundle.band_points_df.to_csv(lsec_csv, index=False)
        staged["lsection_excel_path"] = str(lsec_csv)

    if not bundle.curves_df.empty:
        curve_rows = []
        for idx, row in bundle.curves_df.iterrows():
            curve_rows.append({
                "Sr. NO.": row["curve_no"] or str(idx + 1),
                "Delta": float(row["deflection_deg"]),
                "Radius": float(row["radius_m"]),
                "Degree": float(row["degree"]) if pd.notna(row["degree"]) else 0.0,
                "TPTC-1": _format_km_plus_m(float(row["ts_chainage_m"])),
                "TPCC-1": _format_km_plus_m(float(row["sc_chainage_m"])),
                "TPCC-2": _format_km_plus_m(float(row["cs_chainage_m"])),
                "TPTC-2": _format_km_plus_m(float(row["st_chainage_m"])),
                "Transition Length": float(row["transition_length_m"]),
                "Circular Curve length": float(row["circular_length_m"]),
                "Total Circular length": float(row["total_length_m"]),
                "Tangent Length": 0.0,
                "Cant": float(row["cant_mm"]),
                "Speed Potential": default_design_speed_kmh,
            })
        curve_xlsx = staging_dir / "db_curves.xlsx"
        with pd.ExcelWriter(curve_xlsx, engine="openpyxl") as writer:
            pd.DataFrame(curve_rows).to_excel(writer, sheet_name="Prop.Curve List", index=False)
        staged["curve_excel_path"] = str(curve_xlsx)

    if not bundle.gradients_df.empty:
        grad_rows = []
        for _, row in bundle.gradients_df.iterrows():
            ratio = float(row["slope_ratio"])
            direction = str(row["direction"]).upper()
            dir_label = "Rise" if direction.startswith("R") else ("Fall" if direction.startswith("F") else "Level")
            grade_str = "Horizontal" if ratio <= 0.0 or dir_label == "Level" else f"1:{ratio:.0f}"
            grad_rows.append({
                "S.No.": int(row["seg_no"]),
                "Start Station": _format_km_plus_m(float(row["start_chainage_m"])),
                "End Station": _format_km_plus_m(float(row["end_chainage_m"])),
                "Start Elevation": f"{float(row['start_rl_m']):.3f}m",
                "End Elevation": f"{float(row['end_rl_m']):.3f}m",
                "Grade ": grade_str,
                "Gradient": grade_str,
                "Direction": dir_label,
            })
        grad_xlsx = staging_dir / "db_gradients.xlsx"
        pd.DataFrame(grad_rows).to_excel(grad_xlsx, index=False)
        staged["gradient_excel_path"] = str(grad_xlsx)

    if not bundle.stations_df.empty:
        stn_xlsx = staging_dir / "db_stations.xlsx"
        pd.DataFrame({
            "Station Name": bundle.stations_df["name"],
            "Chainage (m)": bundle.stations_df["chainage_m"],
            "Location (Km)": bundle.stations_df["location_km"],
            "Station Class": bundle.stations_df["station_class"],
        }).to_excel(stn_xlsx, index=False)
        staged["station_excel_path"] = str(stn_xlsx)

    if not bundle.bridges_df.empty:
        br_xlsx = staging_dir / "db_bridges.xlsx"
        with pd.ExcelWriter(br_xlsx, engine="openpyxl") as writer:
            bundle.bridges_df.to_excel(writer, sheet_name="Bridge List", index=False)
        staged["bridge_excel_path"] = str(br_xlsx)

    return staged


def _build_greenfield_session_config(
    base_cfg: dict[str, Any],
    bundle: GreenfieldDBBundle,
    payload: GreenfieldCompositeGenerateRequest,
    session_dir: Path,
    staged_paths: dict[str, Optional[str]],
    kml_path: Path,
    topo_path: Optional[Path],
) -> dict[str, Any]:
    """Build isolated per-job config merging base config.yaml, DB metadata, and overrides."""
    cfg = copy.deepcopy(base_cfg)
    drawing_cfg = cfg.setdefault("drawing", {})
    align_cfg = cfg.setdefault("alignment", {})
    rail_cfg = cfg.setdefault("railway", {})
    paths_cfg = cfg.setdefault("paths", {})
    project_cfg = cfg.setdefault("project", {})
    pm = bundle.project_meta

    start_ch = payload.start_chainage_m if payload.start_chainage_m is not None else pm.start_chainage_m
    end_ch = payload.end_chainage_m if payload.end_chainage_m is not None else (
        pm.end_chainage_m if pm.end_chainage_m > start_ch else None
    )

    paths_cfg["kml_path"] = str(kml_path.resolve())
    paths_cfg["topomap_kml_path"] = str(topo_path.resolve()) if topo_path else str(kml_path.resolve())
    paths_cfg["outputs_dir"] = str(session_dir.resolve())
    paths_cfg["composite_outputs_dir"] = str((session_dir / "composite").resolve())
    paths_cfg["alignment_plan_outputs_dir"] = str((session_dir / "alignment_plan").resolve())
    paths_cfg["lsection_excel_path"] = staged_paths["lsection_excel_path"]
    paths_cfg["curve_excel_path"] = staged_paths["curve_excel_path"]
    paths_cfg["gradient_excel_path"] = staged_paths["gradient_excel_path"]
    paths_cfg["station_excel_path"] = staged_paths["station_excel_path"]
    paths_cfg["bridge_list_path"] = staged_paths["bridge_excel_path"]

    rail_cfg["gauge"] = pm.gauge or str(rail_cfg.get("gauge", "BG"))
    if payload.track_type is not None:
        rail_cfg["track_type"] = payload.track_type.value

    interval_m = float(payload.chainage_interval_m or align_cfg.get("chainage_interval", 20.0))
    align_cfg["start_chainage"] = float(start_ch)
    align_cfg["chainage_interval"] = interval_m
    cfg.setdefault("chainage", {})["interval_m"] = interval_m
    if end_ch is not None:
        align_cfg["end_chainage"] = float(end_ch)
        cfg.setdefault("horizontal", {})["end_chainage_m"] = float(end_ch)
        drawing_cfg["end_chainage_m"] = float(end_ch)

    drawing_cfg["plan_scale"] = float(
        getattr(payload, "plan_scale", None) or drawing_cfg.get("plan_scale", 5000.0)
    )
    drawing_cfg["profile_h_scale"] = float(
        getattr(payload, "profile_h_scale", None) or drawing_cfg.get("profile_h_scale", 5000.0)
    )
    drawing_cfg["profile_v_scale"] = float(
        getattr(payload, "profile_v_scale", None) or drawing_cfg.get("profile_v_scale", 500.0)
    )
    drawing_cfg["max_lsection_length_m"] = float(
        getattr(payload, "max_sheet_length_m", None)
        if getattr(payload, "max_sheet_length_m", None) is not None
        else drawing_cfg.get("max_lsection_length_m", 5000.0)
    )
    drawing_cfg["generate_complete_profile"] = payload.generate_complete_profile
    drawing_cfg["generate_sheet_drawings"] = payload.generate_sheet_drawings
    drawing_cfg["export_pdf"] = payload.export_pdf

    project_cfg["name"] = pm.project_name
    project_cfg["railway_zone"] = pm.railway_zone or "Indian Railways"
    project_cfg["zone"] = pm.railway_zone or "Indian Railways"
    project_cfg["division"] = pm.division or "Engineering Division"
    project_cfg["project_id"] = pm.project_id

    active_bands = payload.custom_bands if payload.custom_bands else bundle.version_meta.band_headers
    if active_bands:
        cfg.setdefault("templates", {}).setdefault("greenfield", {}).setdefault("longitudinal_section", {})[
            "bands"
        ] = active_bands
        cfg["custom_bands"] = active_bands

    return cfg


def _publish_artifact(
    local_path: Path,
    storage_mgr: S3StorageManager,
    s3_bucket: Optional[str],
    s3_prefix: str,
    request: Optional[Request],
) -> str:
    if s3_bucket:
        clean_prefix = s3_prefix.strip("/")
        key = f"{clean_prefix}/{local_path.name}" if clean_prefix else local_path.name
        return storage_mgr.upload_file(local_path, s3_bucket, key)

    if local_path.is_relative_to(OUTPUTS_DIR):
        rel = f"outputs/{local_path.relative_to(OUTPUTS_DIR).as_posix()}"
    elif local_path.is_relative_to(BASE_DIR):
        rel = local_path.relative_to(BASE_DIR).as_posix()
    else:
        rel = str(local_path)
    return _format_local_url(rel, request=request)


def _collect_sheet_records(
    comp_res: Any,
    bundle: GreenfieldDBBundle,
    cfg: dict[str, Any],
    storage_mgr: S3StorageManager,
    s3_bucket: Optional[str],
    s3_prefix: str,
    request: Optional[Request],
    job_id: str,
) -> tuple[list[SheetWriteRecord], list[GreenfieldSheetItem], Optional[str]]:
    drawing_cfg = cfg.get("drawing", {})
    dtm_cfg = cfg.get("dtm_settings", {})
    h_scale = int(round(float(drawing_cfg.get("profile_h_scale", 5000.0))))
    v_scale = int(round(float(drawing_cfg.get("profile_v_scale", 500.0))))
    v_exag = round(h_scale / v_scale, 2) if v_scale > 0 else 10.0
    datum_margin = float(dtm_cfg.get("datum_margin_m", 5.0))
    datum_round = float(dtm_cfg.get("datum_round_m", 10.0))

    raw_sheets = (
        comp_res.metadata.get("sheets", [])
        if hasattr(comp_res, "metadata") and isinstance(comp_res.metadata, dict)
        else []
    )
    pdf_by_stem = {Path(p).stem: Path(p) for p in getattr(comp_res, "pdf_paths", []) if Path(p).exists()}

    if not raw_sheets and getattr(comp_res, "dxf_paths", None):
        raw_sheets = [
            {
                "sheet_index": idx,
                "start_ch": bundle.project_meta.start_chainage_m,
                "end_ch": max(bundle.project_meta.end_chainage_m, bundle.project_meta.start_chainage_m + 1.0),
                "dxf_path": str(p),
            }
            for idx, p in enumerate(comp_res.dxf_paths, start=1)
        ]

    db_records: list[SheetWriteRecord] = []
    sheet_items: list[GreenfieldSheetItem] = []

    for idx, entry in enumerate(raw_sheets, start=1):
        dxf_path = Path(entry["dxf_path"])
        if not dxf_path.exists():
            continue

        s_ch = float(entry.get("start_ch", bundle.project_meta.start_chainage_m))
        e_ch = float(entry.get("end_ch", max(bundle.project_meta.end_chainage_m, s_ch + 1.0)))
        if e_ch <= s_ch:
            e_ch = s_ch + 1.0

        sheet_num = int(entry.get("sheet_index", idx))
        title = f"Sheet {sheet_num} (Km {s_ch / 1000.0:.2f} - {e_ch / 1000.0:.2f})"
        datum_rl = _compute_sheet_datum_rl(bundle.band_points_df, s_ch, e_ch, datum_margin, datum_round)

        dxf_url = _publish_artifact(dxf_path, storage_mgr, s3_bucket, s3_prefix, request)
        pdf_url: Optional[str] = None
        pdf_path = pdf_by_stem.get(dxf_path.stem)
        if pdf_path and pdf_path.exists():
            pdf_url = _publish_artifact(pdf_path, storage_mgr, s3_bucket, s3_prefix, request)

        db_records.append(
            SheetWriteRecord(
                sheet_number=sheet_num,
                title=title,
                start_chainage_m=s_ch,
                end_chainage_m=e_ch,
                datum_rl_m=datum_rl,
                h_scale=h_scale,
                v_scale=v_scale,
                v_exaggeration=v_exag,
                dxf_url=dxf_url,
                pdf_url=pdf_url,
            )
        )
        item = GreenfieldSheetItem(
            sheet_number=sheet_num,
            title=title,
            start_chainage_m=s_ch,
            end_chainage_m=e_ch,
            datum_rl_m=datum_rl,
            dxf_url=dxf_url,
            pdf_url=pdf_url,
        )
        sheet_items.append(item)
        job_manager.add_sheet(job_id, item)

    version_dxf_url = sheet_items[0].dxf_url if sheet_items else None
    return db_records, sheet_items, version_dxf_url


# ---------------------------------------------------------------------------
# Background Worker Execution
# ---------------------------------------------------------------------------

def _execute_greenfield_composite_job(
    payload: GreenfieldCompositeGenerateRequest,
    job_id: str,
    request: Optional[Request],
) -> list[GreenfieldSheetItem]:
    session_dir = OUTPUTS_DIR / f"gf_composite_{job_id}"
    temp_dir = TEMP_UPLOADS_DIR / f"gf_composite_{job_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    job = job_manager.get_job(job_id)
    cleanup_session_dir = False

    try:
        job_manager.emit_event(job_id, "status", {"progress_percent": 20})
        repo = _create_repository(BASE_CONFIG)
        bundle = repo.load_version_bundle(payload.version_id)
        if job is not None:
            job.verification_status = bundle.verification_status
            job.warnings = list(bundle.warnings)

        if bundle.verification_status == "FAIL":
            logger.warning(
                "Engineering Verification Gate reported issues on DB inputs (continuing as validation is bypassed): %s",
                "; ".join(bundle.warnings),
            )
        job_manager.emit_event(job_id, "status", {"progress_percent": 35})
        storage_mgr = _create_storage_manager(BASE_CONFIG)
        kml_local: Optional[Path] = None

        if bundle.version_meta.alignment_kml_url:
            try:
                kml_suffix = Path(bundle.version_meta.alignment_kml_url.split("?")[0]).suffix or ".kml"
                kml_local = storage_mgr.download_file(
                    bundle.version_meta.alignment_kml_url,
                    temp_dir / f"alignment{kml_suffix}",
                )
            except Exception as e:
                logger.warning("Could not download alignmentKmlUrl: %s. Will synthesize KML from coordinates.", e)

        if kml_local is None or not kml_local.exists():
            kml_local = _synthesize_kml_from_band_points(bundle.band_points_df, temp_dir / "alignment.kml")

        topo_local: Optional[Path] = None
        if bundle.version_meta.topo_kml_url:
            topo_suffix = Path(bundle.version_meta.topo_kml_url.split("?")[0]).suffix or ".kml"
            topo_local = storage_mgr.download_file(
                bundle.version_meta.topo_kml_url,
                temp_dir / f"topomap{topo_suffix}",
            )

        default_speed = float(BASE_CONFIG.get("railway", {}).get("design_speed", 100.0))
        staged_paths = _stage_bundle_dataframes(bundle, temp_dir / "staged", default_speed)

        session_cfg = _build_greenfield_session_config(
            base_cfg=BASE_CONFIG,
            bundle=bundle,
            payload=payload,
            session_dir=session_dir,
            staged_paths=staged_paths,
            kml_path=kml_local,
            topo_path=topo_local,
        )

        # Resolve dynamic L-Section sheet layout template
        active_template = payload.template
        if active_template and isinstance(active_template, dict):
            try:
                from template_models import LSectionSheetTemplate
                active_template = LSectionSheetTemplate.model_validate(active_template)
            except Exception as t_err:
                logger.warning("Could not model_validate payload.template: %s", t_err)

        if not active_template:
            tpl_id = payload.template_id
            try:
                with repo._get_connection() as conn:
                    with conn.cursor() as cur:
                        if tpl_id:
                            cur.execute(
                                """
                                SELECT id, name, corridor_type, paper, sheet_width_mm, sheet_height_mm,
                                       sidebar_width_mm, plan_viewport, profile_viewport, data_bands, sidebar, is_public, is_default
                                FROM public.lsection_sheet_templates
                                WHERE id = %s
                                LIMIT 1;
                                """,
                                (tpl_id,),
                            )
                        else:
                            cur.execute(
                                """
                                SELECT id, name, corridor_type, paper, sheet_width_mm, sheet_height_mm,
                                       sidebar_width_mm, plan_viewport, profile_viewport, data_bands, sidebar, is_public, is_default
                                FROM public.lsection_sheet_templates
                                WHERE is_default = TRUE
                                ORDER BY updated_at DESC
                                LIMIT 1;
                                """
                            )
                        row = cur.fetchone()
                        if row:
                            from template_models import LSectionSheetTemplate
                            active_template = LSectionSheetTemplate(
                                id=row[0],
                                name=row[1],
                                corridorType=row[2],
                                paper=row[3],
                                sheetWidthMm=float(row[4]),
                                sheetHeightMm=float(row[5]),
                                sidebarWidthMm=float(row[6]),
                                planViewport=row[7],
                                profileViewport=row[8],
                                dataBands=row[9],
                                sidebar=row[10],
                                isPublic=bool(row[11]),
                                isDefault=bool(row[12]),
                            )
                            logger.info("Using LSectionSheetTemplate '%s' (%s) for job %s", active_template.id, active_template.name, job_id)
            except Exception as tpl_ex:
                logger.warning("Failed to fetch template '%s': %s", tpl_id, tpl_ex)

        # Priority: explicit payload.custom_bands > existing template_db data_bands > Excel version band_headers
        user_bands = payload.custom_bands
        if not user_bands and active_template and not active_template.data_bands:
            user_bands = bundle.version_meta.band_headers

        if user_bands and active_template:
            from template_models import DataBandItem
            new_bands: list[DataBandItem] = []
            for b in user_bands:
                if isinstance(b, dict):
                    b_key = b.get("key") or b.get("id") or b.get("title", "").lower().replace(" ", "_")
                    b_lbl = b.get("label") or b.get("title") or b_key
                    b_h = float(b.get("heightMm") or b.get("height_mm") or 18.0)
                    b_en = bool(b.get("enabled", True))
                    new_bands.append(DataBandItem(key=b_key, label=b_lbl, heightMm=b_h, enabled=b_en))
                elif isinstance(b, str):
                    new_bands.append(DataBandItem(key=b.lower().replace(" ", "_"), label=b, heightMm=18.0, enabled=True))
            if new_bands:
                active_template.data_bands = new_bands
                logger.info("Configured %d dynamic survey bands from user/Excel inputs for job %s", len(new_bands), job_id)

        job_manager.emit_event(job_id, "status", {"progress_percent": 55})
        drawer = _create_composite_drawer(session_cfg, template=active_template)
        output_prefix = payload.output_prefix or f"greenfield_{payload.version_id[:8]}_composite"
        res_dict = drawer.generate_from_inputs(output_prefix=output_prefix)
        comp_res = res_dict.get("composite_result")
        if comp_res is None:
            raise RuntimeError("CompositeDrawer did not produce a composite_result.")

        job_manager.emit_event(job_id, "status", {"progress_percent": 80})
        s3_cfg = BASE_CONFIG.get("s3", {})
        s3_bucket = payload.s3_output_bucket or (
            s3_cfg.get("default_bucket") if s3_cfg.get("upload_outputs_to_s3") else None
        )
        s3_prefix = payload.s3_output_prefix or f"greenfield/{bundle.project_meta.project_id}/{payload.version_id}"

        db_records, sheet_items, version_dxf_url = _collect_sheet_records(
            comp_res=comp_res,
            bundle=bundle,
            cfg=session_cfg,
            storage_mgr=storage_mgr,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            request=request,
            job_id=job_id,
        )

        saved_sheet_ids = repo.save_generated_sheets(
            version_id=payload.version_id,
            sheets=db_records,
            master_dxf_url=version_dxf_url,
            updated_by=payload.user_id,
        )
        for idx, sheet_id in enumerate(saved_sheet_ids):
            if idx < len(sheet_items):
                sheet_items[idx].sheet_id = sheet_id

        cleanup_session_dir = bool(s3_bucket)
        return sheet_items
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        if cleanup_session_dir and session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info("Cleaned up local session outputs after S3 upload: %s", session_dir)


# ---------------------------------------------------------------------------
# FastAPI Routes
# ---------------------------------------------------------------------------

@router.get("/preview/{version_id}", response_model=GreenfieldPreviewResponse, status_code=status.HTTP_200_OK)
async def preview_greenfield_version(version_id: str) -> GreenfieldPreviewResponse:
    """Check readiness and RDSO invariant gate for an lsection_versions UUID."""
    try:
        clean_id = _validate_uuid(version_id, "version_id")
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err

    repo = _create_repository(BASE_CONFIG)
    try:
        bundle = repo.load_version_bundle(clean_id)
    except VersionNotFoundError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err)) from err
    except GreenfieldDBError as err:
        logger.exception("Preview DB error for version_id=%s: %s", clean_id, err)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err)) from err

    warnings = list(bundle.warnings)
    has_kml = bool(bundle.version_meta.alignment_kml_url)
    if not has_kml:
        warnings.append("Missing alignmentKmlUrl on lsection_versions.")

    return GreenfieldPreviewResponse(
        version_id=bundle.version_meta.version_id,
        ready_to_generate=True,
        verification_status="PASS",
        warnings=warnings,
    )


@router.post("/preview-dxf", status_code=status.HTTP_200_OK)
async def preview_greenfield_composite_dxf(
    payload: GreenfieldCompositeGenerateRequest,
) -> dict[str, Any]:
    """Fast single-sheet (first 5 km) composite DXF preview without PDF/S3/DB writes."""
    try:
        clean_version_id = _validate_uuid(payload.version_id, "version_id")
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err

    import uuid
    preview_id = f"preview_{uuid.uuid4().hex[:8]}"
    session_dir = OUTPUTS_DIR / preview_id
    temp_dir = TEMP_UPLOADS_DIR / preview_id
    session_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        repo = _create_repository(BASE_CONFIG)
        bundle = repo.load_version_bundle(clean_version_id)
        if not bundle.band_points_df.empty and "chainage_m" in bundle.band_points_df.columns:
            min_ch = float(bundle.band_points_df["chainage_m"].min())
            bundle.band_points_df = bundle.band_points_df[
                bundle.band_points_df["chainage_m"] <= (min_ch + 5000.0)
            ].copy()

        storage_mgr = _create_storage_manager(BASE_CONFIG)
        kml_local: Optional[Path] = None
        if bundle.version_meta.alignment_kml_url:
            try:
                kml_suffix = Path(bundle.version_meta.alignment_kml_url.split("?")[0]).suffix or ".kml"
                kml_local = storage_mgr.download_file(
                    bundle.version_meta.alignment_kml_url,
                    temp_dir / f"alignment{kml_suffix}",
                )
            except Exception:
                kml_local = None
        if kml_local is None or not kml_local.exists():
            kml_local = _synthesize_kml_from_band_points(bundle.band_points_df, temp_dir / "alignment.kml")

        default_speed = float(BASE_CONFIG.get("railway", {}).get("design_speed", 100.0))
        staged_paths = _stage_bundle_dataframes(bundle, temp_dir / "staged", default_speed)
        session_cfg = _build_greenfield_session_config(
            base_cfg=BASE_CONFIG,
            bundle=bundle,
            payload=payload,
            session_dir=session_dir,
            staged_paths=staged_paths,
            kml_path=kml_local,
            topo_path=None,
        )
        session_cfg.setdefault("drawing", {})
        session_cfg["drawing"]["generate_complete_profile"] = False
        session_cfg["drawing"]["generate_sheet_drawings"] = True
        session_cfg["drawing"]["export_pdf"] = False
        session_cfg["drawing"]["preview_skip_sidebar"] = True

        active_template = payload.template
        if active_template and isinstance(active_template, dict):
            from template_models import LSectionSheetTemplate
            active_template = LSectionSheetTemplate.model_validate(active_template)
        elif payload.template_id:
            with repo._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, name, corridor_type, paper, sheet_width_mm, sheet_height_mm,
                               sidebar_width_mm, plan_viewport, profile_viewport, data_bands, sidebar, is_public, is_default
                        FROM public.lsection_sheet_templates WHERE id = %s LIMIT 1;
                        """,
                        (payload.template_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        from template_models import LSectionSheetTemplate
                        active_template = LSectionSheetTemplate(
                            id=row[0], name=row[1], corridorType=row[2], paper=row[3],
                            sheetWidthMm=float(row[4]), sheetHeightMm=float(row[5]),
                            sidebarWidthMm=float(row[6]), planViewport=row[7],
                            profileViewport=row[8], dataBands=row[9], sidebar=row[10],
                            isPublic=bool(row[11]), isDefault=bool(row[12]),
                        )

        drawer = _create_composite_drawer(session_cfg, template=active_template)
        res_dict = drawer.generate_from_inputs(output_prefix=f"preview_{clean_version_id[:8]}")
        comp_res = res_dict.get("composite_result")
        if comp_res is None or not Path(comp_res.dxf_path).exists():
            raise HTTPException(status_code=500, detail="Preview DXF generation failed.")
        dxf_text = Path(comp_res.dxf_path).read_text(encoding="utf-8", errors="replace")
        return {"versionId": clean_version_id, "dxfContent": dxf_text}
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/generate", response_model=GreenfieldGenerateAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_greenfield_composite(
    payload: GreenfieldCompositeGenerateRequest,
    request: Request,
) -> GreenfieldGenerateAcceptedResponse:
    """Enqueue DB-driven Greenfield Composite Plan & L-Section sheet generation."""
    try:
        clean_version_id = _validate_uuid(payload.version_id, "version_id")
        if payload.user_id:
            _validate_uuid(payload.user_id, "user_id")
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err

    payload.version_id = clean_version_id
    job = job_manager.create_job(version_id=clean_version_id)

    task = asyncio.create_task(
        job_manager.execute_job(
            job.job_id,
            _execute_greenfield_composite_job,
            payload,
            job.job_id,
            request,
        )
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    return GreenfieldGenerateAcceptedResponse(
        job_id=job.job_id,
        version_id=clean_version_id,
        status=job.status.value,
        status_url=f"/api/v1/greenfield/composite/jobs/{job.job_id}",
        stream_url=f"/api/v1/greenfield/composite/jobs/{job.job_id}/stream",
    )


@router.get("/jobs/{job_id}", response_model=GreenfieldJobStatusResponse, status_code=status.HTTP_200_OK)
async def get_greenfield_job_status(job_id: str) -> GreenfieldJobStatusResponse:
    """Return clean Greenfield job status, RDSO verification result, and generated sheets."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Greenfield job '{job_id}' not found.")
    return job.to_status_response()


@router.get("/jobs/{job_id}/stream")
async def stream_greenfield_job(job_id: str) -> StreamingResponse:
    """Server-Sent Events (SSE) stream for live Greenfield generation progress."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Greenfield job '{job_id}' not found.")

    async def event_generator():
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        job.subscribers.append(q)
        try:
            init_data = {
                "event": "init",
                "job_id": job.job_id,
                "version_id": job.version_id,
                "status": job.status.value,
                "progress_percent": job.progress_percent,
                "sheets": [s.model_dump() for s in job.sheets],
            }
            yield f"data: {json.dumps(init_data)}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=12.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("event") in ("completed", "failed"):
                        break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                        break
        finally:
            if q in job.subscribers:
                job.subscribers.remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

