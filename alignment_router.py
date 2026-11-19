"""
Module: service/alignment/alignment_router.py
Purpose: FastAPI APIRouter for Railway Alignment Detection, Fast DTM Surface Sampling,
         and Survey Centerline Extraction.
Standards: RDSO / Indian Railways BG
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import yaml

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

try:
    from alignment.config import __file__ as _cfg_file
    CONFIG_PATH = Path(_cfg_file).resolve().parent / "alignment_config.yaml"
except Exception:
    CONFIG_PATH = Path(__file__).resolve().parent / "config/alignment_config.yaml"

if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).resolve().parents[2] / "future_alignment_studio/config/alignment_config.yaml"

with CONFIG_PATH.open(encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

try:
    from alignment.utils.logger import get_logger
    from alignment.alignment_models import PlanProfileResponse
    from alignment.geospatial.kml_ingestor import KMLAlignmentIngestor
    from alignment.horizontal.pi_detector import detect_pis
    from alignment.horizontal.curve_solver import CurveDirection, HorizontalCurveSolver
    from alignment.horizontal.alignment_curve_calculator import AlignmentCurveCalculator, CalculatedAlignment
except ImportError:
    from utils.logger import get_logger
    from alignment_models import PlanProfileResponse
    from geospatial.kml_ingestor import KMLAlignmentIngestor
    from horizontal.pi_detector import detect_pis
    from horizontal.curve_solver import CurveDirection, HorizontalCurveSolver
    from horizontal.alignment_curve_calculator import AlignmentCurveCalculator, CalculatedAlignment

logger = get_logger("AlignmentServiceRouter")

router = APIRouter(tags=["Railway Alignment Detection Service"])

MIB = 1024 * 1024
MAX_UPLOAD_BYTES = int(os.environ.get("ESP_ALIGNMENT_MAX_UPLOAD_MIB", "100")) * MIB
MAX_KML_BYTES = int(os.environ.get("ESP_ALIGNMENT_MAX_KML_MIB", "100")) * MIB


# ------------------------------------------------------------------------------
# Helpers: KML Corridor Stitching & Geometry Serialization
# ------------------------------------------------------------------------------
def _extract_and_stitch_corridors(data: bytes, is_kmz: bool) -> tuple[list[tuple[str, list[tuple[float, float]]]], list[dict[str, Any]]]:
    """Extract and stitch fragmented LineString coordinates from KML/KMZ into continuous paths."""
    if is_kmz:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = [
                item for item in archive.infolist()
                if item.filename.lower().endswith(".kml") and not item.is_dir()
            ]
            if not entries:
                raise ValueError("KMZ archive contains no .kml document")
            kml_entry = next((e for e in entries if e.filename.lower().endswith("doc.kml")), entries[0])
            if kml_entry.file_size > MAX_KML_BYTES:
                raise ValueError(f"KML inside KMZ exceeds {MAX_KML_BYTES // MIB} MiB limit")
            data = archive.read(kml_entry)

    root = ET.fromstring(data)
    raw_lines: list[tuple[str, str]] = []
    stack = [(root, "")]
    while stack:
        node, parent_name = stack.pop()
        tag = node.tag.rsplit("}", 1)[-1]
        name = parent_name
        if tag in ("Placemark", "Folder", "Document"):
            name = next(
                ((c.text or "").strip() for c in node if c.tag.rsplit("}", 1)[-1] == "name" and (c.text or "").strip()),
                parent_name,
            )
        if tag == "LineString":
            coord_nodes = [c for c in node if c.tag.rsplit("}", 1)[-1] == "coordinates"]
            if len(coord_nodes) == 1 and coord_nodes[0].text and coord_nodes[0].text.strip():
                raw_lines.append((name, coord_nodes[0].text.strip()))
        stack.extend((child, name) for child in reversed(list(node)))

    if not raw_lines:
        raise ValueError("No LineString corridor found in KML file")

    parsed_lines: list[tuple[str, list[tuple[float, float]]]] = []
    for idx, (name, text) in enumerate(raw_lines):
        pts: list[tuple[float, float]] = []
        for chunk in text.split():
            parts = chunk.split(",")
            if len(parts) >= 2:
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                    if math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90:
                        if not pts or (pts[-1][0] != lon or pts[-1][1] != lat):
                            pts.append((lon, lat))
                except ValueError:
                    continue
        if len(pts) >= 2:
            parsed_lines.append((name or f"Line {idx + 1}", pts))

    if not parsed_lines:
        raise ValueError("No valid coordinates found in LineString elements")

    if len(parsed_lines) == 1:
        return [parsed_lines[0]], []

    shapely_geoms = [LineString(pts) for _, pts in parsed_lines]
    try:
        merged = linemerge(shapely_geoms)
        if merged.geom_type == "LineString":
            merged_parts = [merged]
        elif merged.geom_type == "MultiLineString":
            merged_parts = list(merged.geoms)
        else:
            merged_parts = shapely_geoms
    except Exception:
        merged_parts = shapely_geoms

    sorted_parts = sorted(merged_parts, key=lambda g: g.length, reverse=True)
    corridors: list[list[tuple[float, float]]] = []
    while sorted_parts:
        cur = list(sorted_parts.pop(0).coords)
        changed = True
        while changed and sorted_parts:
            changed = False
            best_idx = -1
            best_d = float("inf")
            conn_start = False
            rev = False
            for i, p in enumerate(sorted_parts):
                pc = list(p.coords)
                d1 = (pc[-1][0] - cur[0][0]) ** 2 + (pc[-1][1] - cur[0][1]) ** 2
                if d1 < best_d:
                    best_d = d1; best_idx = i; conn_start = True; rev = False
                d2 = (pc[0][0] - cur[0][0]) ** 2 + (pc[0][1] - cur[0][1]) ** 2
                if d2 < best_d:
                    best_d = d2; best_idx = i; conn_start = True; rev = True
                d3 = (pc[0][0] - cur[-1][0]) ** 2 + (pc[0][1] - cur[-1][1]) ** 2
                if d3 < best_d:
                    best_d = d3; best_idx = i; conn_start = False; rev = False
                d4 = (pc[-1][0] - cur[-1][0]) ** 2 + (pc[-1][1] - cur[-1][1]) ** 2
                if d4 < best_d:
                    best_d = d4; best_idx = i; conn_start = False; rev = True
            if best_d < (0.0005) ** 2:
                matched = sorted_parts.pop(best_idx)
                add_c = list(matched.coords)
                if rev:
                    add_c.reverse()
                if conn_start:
                    cur = add_c[:-1] + cur
                else:
                    cur = cur + add_c[1:]
                changed = True
        corridors.append(cur)

    final_corridors: list[tuple[str, list[tuple[float, float]]]] = []
    options: list[dict[str, Any]] = []
    for idx, c in enumerate(corridors):
        name = f"Corridor {idx + 1}"
        final_corridors.append((name, c))
        if len(corridors) > 1:
            l_deg = sum(math.hypot(c[i+1][0] - c[i][0], c[i+1][1] - c[i][1]) for i in range(len(c)-1))
            options.append({
                "index": idx,
                "name": name,
                "point_count": len(c),
                "length_km": round((l_deg * 111000.0) / 1000.0, 2),
                "start": f"{c[0][0]},{c[0][1]}",
                "end": f"{c[-1][0]},{c[-1][1]}",
            })

    return final_corridors, options


def _serialize_calculated_alignment(calculated: CalculatedAlignment, start_chainage_m: float = 0.0) -> dict[str, Any]:
    """Serialize CalculatedAlignment geometry elements into JSON."""
    geom_elements = []
    for elem in calculated.elements:
        el_dict: dict[str, Any] = {
            "element_id": elem.element_id,
            "element_type": elem.element_type.value,
            "start_chainage_m": elem.start_chainage_m,
            "end_chainage_m": elem.end_chainage_m,
            "length_m": elem.length_m,
            "points": [{"x": pt[0], "y": pt[1]} for pt in elem.points],
            "curve_id": elem.curve_id,
        }
        if elem.arc_geom is not None:
            el_dict["arc_geom"] = {
                "center_x": elem.arc_geom.center_easting_m,
                "center_y": elem.arc_geom.center_northing_m,
                "radius_m": elem.arc_geom.radius_m,
                "start_angle_deg": elem.arc_geom.start_angle_deg,
                "end_angle_deg": elem.arc_geom.end_angle_deg,
                "sweep_angle_deg": elem.arc_geom.sweep_angle_deg,
                "direction": elem.arc_geom.direction.value,
                "start_point": {"x": elem.arc_geom.start_point[0], "y": elem.arc_geom.start_point[1]},
                "end_point": {"x": elem.arc_geom.end_point[0], "y": elem.arc_geom.end_point[1]},
            }
        geom_elements.append(el_dict)

    return {
        "total_length_m": calculated.total_length_m,
        "start_chainage_m": float(start_chainage_m),
        "elements": geom_elements,
        "curves": [asdict(c) for c in calculated.curves],
        "tangent_points": [asdict(tp) for tp in calculated.tangent_points],
        "critical_stations": calculated.critical_stations,
    }


def _detect(
    data: bytes,
    is_kmz: bool,
    tolerance: float,
    angle: float,
    line_index: Optional[int] = None,
    start_chainage_m: float = 0.0,
) -> PlanProfileResponse:
    """Core calculation worker for horizontal alignment PI & curve detection."""
    corridors, options = _extract_and_stitch_corridors(data, is_kmz)
    if not corridors:
        raise ValueError("No LineString corridor found in KML/KMZ file")

    idx = 0 if line_index is None or line_index >= len(corridors) else line_index
    _name, coords = corridors[idx]

    if len(coords) < 2:
        raise ValueError("Selected corridor requires at least two coordinates")

    ingestor = KMLAlignmentIngestor(CONFIG["railway_standards"]["chainage_interval_m"])
    epsg = ingestor.detect_utm_epsg(sum(p[0] for p in coords) / len(coords), sum(p[1] for p in coords) / len(coords))
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    projected = [transformer.transform(*pt) for pt in coords]
    length = sum(math.dist(a, b) for a, b in zip(projected, projected[1:]))

    stations, epsg = ingestor.process_and_resample(
        coords,
        target_epsg=epsg,
        start_chainage_m=start_chainage_m,
    )

    curve_std = CONFIG.get("curve_standards", {})
    candidates = detect_pis(
        stations,
        simplification_tolerance_m=tolerance,
        minimum_deflection_deg=angle,
        merge_continuous_curves=curve_std.get("merge_continuous_curves", True),
        max_group_gap_m=curve_std.get("max_curve_cluster_gap_m", 350.0),
        straight_tangent_min_segments=curve_std.get("straight_tangent_min_segments", 4),
        straight_tangent_angle_tol_deg=curve_std.get("straight_tangent_angle_tol_deg", 0.02),
        radius_round_interval_m=curve_std.get("radius_round_interval_m", 25.0),
    )

    raw_lines = []
    cumulative_m = float(start_chainage_m)
    for seg_idx, (p1, p2) in enumerate(zip(projected, projected[1:])):
        seg_dist = math.dist(p1, p2)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        raw_lines.append({
            "line_id": f"Line {seg_idx + 1}",
            "segment_index": seg_idx,
            "start_easting_m": round(p1[0], 3),
            "start_northing_m": round(p1[1], 3),
            "end_easting_m": round(p2[0], 3),
            "end_northing_m": round(p2[1], 3),
            "length_m": round(seg_dist, 3),
            "bearing_deg": round(bearing, 2),
            "start_chainage_m": round(cumulative_m, 2),
            "end_chainage_m": round(cumulative_m + seg_dist, 2),
        })
        cumulative_m += seg_dist

    raw_vertices = [
        {
            "vertex_index": i,
            "easting_m": round(p[0], 3),
            "northing_m": round(p[1], 3),
            "lon_deg": coords[i][0],
            "lat_deg": coords[i][1],
        }
        for i, p in enumerate(projected)
    ]

    alignment_geometry = None
    try:
        calc = AlignmentCurveCalculator()
        ctrl_pts = [(projected[0][0], projected[0][1])]
        specs = {}
        for c_idx, c in enumerate(candidates):
            ctrl_pts.append((c.easting_m, c.northing_m))
            if c.estimated_radius_m and c.estimated_radius_m > 0:
                specs[c_idx + 1] = {
                    "radius_m": c.estimated_radius_m,
                    "speed_kmh": CONFIG.get("railway_standards", {}).get("default_design_speed_kmh", 160.0),
                }
        ctrl_pts.append((projected[-1][0], projected[-1][1]))
        if len(ctrl_pts) >= 2:
            calc_res = calc.calculate_alignment(
                ctrl_pts,
                curve_specs=specs if specs else None,
                start_chainage_m=start_chainage_m,
            )
            alignment_geometry = _serialize_calculated_alignment(calc_res, start_chainage_m=start_chainage_m)
    except Exception as exc:
        logger.warning("Could not compute full alignment curve geometry: %s", exc)

    result = {
        "epsg": epsg,
        "total_length_m": round(length, 3),
        "raw_vertices": raw_vertices,
        "raw_lines": raw_lines,
        "stations": [asdict(s) for s in stations],
        "candidates": [asdict(p) for p in candidates],
        "alignment_geometry": alignment_geometry,
        "line_options": options if options else None,
        "selected_line_index": idx,
    }
    return PlanProfileResponse(dxf_files_path=[], detected_files=result)


# ------------------------------------------------------------------------------
# Endpoint: /detect (Horizontal Alignment PI & Curve Detection)
# ------------------------------------------------------------------------------
@router.post("/detect", response_model=PlanProfileResponse, status_code=status.HTTP_200_OK)
@router.post("/api/v1/alignment-studio/detect", response_model=PlanProfileResponse, status_code=status.HTTP_200_OK)
async def detect_alignment(
    file: UploadFile = File(...),
    simplification_tolerance_m: float = Form(5.0),
    minimum_deflection_deg: float = Form(1.0),
    start_chainage_m: float = Form(0.0),
    line_index: Optional[int] = Form(None),
):
    """Detect horizontal alignment PIs and curve geometry synchronously from an uploaded KML/KMZ file."""
    name = (file.filename or "").lower()
    if not name.endswith((".kml", ".kmz")):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Please upload a valid .kml or .kmz file")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {MAX_UPLOAD_BYTES // MIB} MiB upload limit")

    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            _detect,
            data,
            name.endswith(".kmz"),
            simplification_tolerance_m,
            minimum_deflection_deg,
            line_index,
            start_chainage_m,
        )
        return res
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    except Exception as exc:
        logger.exception("Error processing alignment detection: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Alignment detection failed: {exc}")
