"""
Module: alignment/vertical/bridge_constraint_loader.py
Purpose: Ingest Bridge Schedule tables (matching Bridge_List.xlsx headers) into
         typed VerticalConstraint and StationZone objects for FormationLevelSolver.

Supported Input Headers (via config bridge_column_aliases):
    S.No | Bridge No. | Bridge Type | Approx. Chainage (km) | Proposed Span |
    Configration | Bridge Type / Description | Bridge Detail | Bed Level |
    HFL | Clearance

Standards:
    - IRBM Para 102 : Major bridge deck & 30m approach gradient cap (<= 1:400).
    - IRBM Para 106 : Waterway bridge soffit clearance above HFL + structural depth.
    - IRC:SP:34     : RUB vertical headroom + box slab clearance above road EGL.
    - IRPWM         : Level Crossing (LC) at-grade flat zone (<= 1:1000, ±30m).
"""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

try:
    from alignment.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

from ._vertical_models import (
    FormationLevelError,
    StationZone,
    VerticalConstraint,
    load_vertical_config,
)

logger = get_logger("BridgeConstraintLoader")

_SPAN_PATTERN = re.compile(r"(\d+)\s*[xX×]\s*(\d+(?:\.\d+)?)")


def _normalize_header(text: str) -> str:
    """Normalize a column header for case- and punctuation-insensitive lookup."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _safe_float(value: Any) -> Optional[float]:
    """Convert raw Excel/JSON cell value to a finite float, else None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        cleaned = str(value).strip()
        if not cleaned or cleaned.lower() in ("nan", "none", "null", "-", "na", "n/a"):
            return None
        cleaned = re.sub(r"(?i)\s*(?:m|mtr|km)$", "", cleaned).strip()
        num = float(cleaned)
        return num if math.isfinite(num) else None
    except (TypeError, ValueError):
        return None


def parse_span_length_m(span_text: Any) -> float:
    """Parse Indian Railways span strings (e.g. '2x24.4+2X12.20Composite') into total meters."""
    if span_text is None:
        return 0.0
    text = str(span_text).strip()
    if not text or text.lower() in ("nan", "none", "lc"):
        return 0.0

    matches = _SPAN_PATTERN.findall(text)
    if not matches:
        return 0.0
    total = sum(int(count) * float(length) for count, length in matches)
    return round(total, 3)


def resolve_bridge_chainage_m(
    raw_ch: float,
    start_ch_m: float,
    end_ch_m: float,
    header_name: str = "",
) -> float:
    """Resolve bridge chainage into meters, guarding against '(km)' headers holding meters."""
    pad = 100.0
    in_m = (start_ch_m - pad) <= raw_ch <= (end_ch_m + pad)
    in_km = (start_ch_m - pad) <= (raw_ch * 1000.0) <= (end_ch_m + pad)

    if in_m and not in_km:
        return round(raw_ch, 3)
    if in_km and not in_m:
        return round(raw_ch * 1000.0, 3)
    if in_m and in_km:
        # Both plausible (very short corridor); check if header explicitly says km and value is small
        if "km" in header_name.lower() and raw_ch < 50.0 and end_ch_m > 500.0:
            return round(raw_ch * 1000.0, 3)
        return round(raw_ch, 3)

    if "km" in header_name.lower() and abs(raw_ch) < 500.0 and end_ch_m > 1000.0:
        return round(raw_ch * 1000.0, 3)
    return round(raw_ch, 3)


def classify_bridge_row(
    configuration: str,
    bridge_type: str,
    span_text: str,
    detail: str,
) -> tuple[str, str]:
    """Map Bridge_List.xlsx row fields to (canonical_constraint_type, structural_depth_key)."""
    cfg_up = re.sub(r"[^A-Z0-9]+", " ", configuration.upper()).strip()
    btype_up = re.sub(r"[^A-Z0-9]+", "", bridge_type.upper())
    combined = f"{cfg_up} {bridge_type.upper()} {span_text.upper()} {detail.upper()}"

    if cfg_up == "LC" or any(tok in combined for tok in ("L-XING", "LXING", "LEVEL CROSSING", "GATE", "XING")):
        return "LC", "NONE"

    if "ROB" in cfg_up.split() or "ROAD OVER BRIDGE" in combined:
        return "ROB", "NONE"

    if "RUB" in cfg_up.split() or "BOXRUB" in btype_up or "RUB" in combined or "UNDERPASS" in combined:
        if any(hw in combined for hw in (" NH", " SH", "HIGHWAY", "TRUNK")):
            return "RUB_NH", "RCCBOX"
        return "RUB_LOCAL", "RCCBOX"

    # Determine structural depth key from Bridge Type
    depth_key = "DEFAULT_MINOR"
    if "OPENWEB" in btype_up or "OWG" in btype_up:
        depth_key = "OPEN_WEB_GIRDER"
    elif "COMPOSITE" in btype_up:
        depth_key = "COMPOSITE_GIRDER"
    elif "GIRDER" in btype_up:
        depth_key = "GIRDER_PSC"
    elif "PSCSLAB" in btype_up:
        depth_key = "PSCSLAB"
    elif "SLAB" in btype_up:
        depth_key = "SLAB"
    elif "RCCBOX" in btype_up or "BOX" in btype_up:
        depth_key = "RCCBOX"
    elif btype_up in ("HP", "HUMEPIPE") or "PIPE" in btype_up:
        depth_key = "HP"

    is_major = (
        "MAJOR" in cfg_up.split()
        or depth_key in ("COMPOSITE_GIRDER", "GIRDER_PSC", "OPEN_WEB_GIRDER")
    )
    if is_major:
        return "MAJOR_BRIDGE", (depth_key if depth_key != "DEFAULT_MINOR" else "DEFAULT_MAJOR")
    if depth_key in ("HP", "RCCBOX") and "MINOR" not in cfg_up.split():
        return "BOX_CULVERT", depth_key
    return "MINOR_BRIDGE", depth_key


def _resolve_column_map(
    keys: Sequence[str],
    aliases: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Match input column headers against config bridge_column_aliases."""
    norm_to_orig = {_normalize_header(k): k for k in keys}
    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for alias in candidates:
            norm_a = _normalize_header(alias)
            if norm_a in norm_to_orig:
                resolved[canonical] = norm_to_orig[norm_a]
                break
    return resolved


def load_bridge_constraints_from_records(
    records: Sequence[Mapping[str, Any]],
    start_ch_m: float,
    end_ch_m: float,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> tuple[list[VerticalConstraint], list[StationZone]]:
    """Parse bridge dict rows (using Bridge_List.xlsx headers) into vertical constraints & flat zones."""
    if not records:
        return [], []

    vcfg = load_vertical_config(config, config_path)
    aliases = vcfg.get("bridge_column_aliases", {})
    struct_depths = {str(k).upper(): float(v) for k, v in vcfg.get("bridge_structural_depths_m", {}).items()}
    br_approach_grad = float(vcfg.get("bridge_approach_gradient_limit", 0.0025))
    br_approach_len = float(vcfg.get("bridge_approach_length_m", 30.0))
    lc_grad = float(vcfg.get("lc_gradient_limit", 0.001))
    lc_len = float(vcfg.get("lc_check_length_m", 30.0))

    all_keys: list[str] = []
    for rec in records:
        for k in rec.keys():
            if k not in all_keys:
                all_keys.append(str(k))
    col_map = _resolve_column_map(all_keys, aliases)

    if "chainage" not in col_map:
        raise FormationLevelError(
            "Bridge records are missing a recognizable chainage column "
            "(expected e.g. 'Approx. Chainage (km)' or 'chainage_m')."
        )

    ch_header = col_map["chainage"]
    constraints: list[VerticalConstraint] = []
    flat_zones: list[StationZone] = []

    for idx, row in enumerate(records, start=1):
        raw_ch = _safe_float(row.get(ch_header))
        if raw_ch is None:
            continue

        ch_m = resolve_bridge_chainage_m(raw_ch, start_ch_m, end_ch_m, ch_header)
        if ch_m < start_ch_m - 1.0 or ch_m > end_ch_m + 1.0:
            continue

        br_no = str(row.get(col_map.get("bridge_no", ""), "") or f"BR-{idx}").strip()
        br_type = str(row.get(col_map.get("bridge_type", ""), "") or "").strip()
        cfg_val = str(row.get(col_map.get("configuration", ""), "") or "").strip()
        span_txt = str(row.get(col_map.get("proposed_span", ""), "") or "").strip()
        detail = str(row.get(col_map.get("detail", ""), "") or "").strip()

        ctype, depth_key = classify_bridge_row(cfg_val, br_type, span_txt, detail)
        span_len_m = parse_span_length_m(span_txt)
        bed_lvl = _safe_float(row.get(col_map.get("bed_level", "")))
        hfl_val = _safe_float(row.get(col_map.get("hfl", "")))
        clr_val = _safe_float(row.get(col_map.get("clearance", "")))
        struct_depth = struct_depths.get(depth_key, 0.0) if depth_key != "NONE" else 0.0

        if ctype == "LC":
            flat_zones.append(
                StationZone(
                    start_chainage_m=round(max(start_ch_m, ch_m - lc_len), 3),
                    end_chainage_m=round(min(end_ch_m, ch_m + lc_len), 3),
                    name=f"{br_no} (LC)",
                    max_gradient=lc_grad,
                )
            )
            continue

        if ctype == "ROB":
            continue

        constraints.append(
            VerticalConstraint(
                chainage_m=ch_m,
                constraint_type=ctype,
                name=f"{br_no} ({br_type or ctype})",
                hfl_m=hfl_val,
                bed_level_m=bed_lvl,
                clearance_m=clr_val,
                structural_depth_m=struct_depth,
                span_length_m=span_len_m,
                bridge_type=br_type,
                proposed_span_text=span_txt,
            )
        )

        # IRBM Para 102: Major bridges require flat deck & <= 1:400 within 30m of abutments
        if ctype == "MAJOR_BRIDGE":
            half_span = 0.5 * span_len_m
            flat_zones.append(
                StationZone(
                    start_chainage_m=round(max(start_ch_m, ch_m - half_span - br_approach_len), 3),
                    end_chainage_m=round(min(end_ch_m, ch_m + half_span + br_approach_len), 3),
                    name=f"{br_no} Approach (IRBM Para 102)",
                    max_gradient=br_approach_grad,
                )
            )

    logger.info(
        "Loaded %d bridge vertical constraint(s) and %d flat approach/LC zone(s) from %d record(s).",
        len(constraints), len(flat_zones), len(records),
    )
    return constraints, flat_zones


def load_bridge_constraints_from_excel(
    excel_path: str | Path,
    start_ch_m: float,
    end_ch_m: float,
    sheet_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> tuple[list[VerticalConstraint], list[StationZone]]:
    """Read Bridge_List.xlsx (or CSV) and convert rows into vertical constraints & flat zones."""
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Bridge schedule file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        target_sheet: Any = sheet_name if sheet_name is not None else 0
        if sheet_name is None:
            xl = pd.ExcelFile(path)
            for candidate in xl.sheet_names:
                if "bridge" in candidate.lower():
                    target_sheet = candidate
                    break
        df = pd.read_excel(path, sheet_name=target_sheet)
    else:
        raise FormationLevelError(f"Unsupported bridge schedule format '{suffix}'. Use .xlsx, .xls, or .csv.")

    records = df.where(pd.notna(df), None).to_dict(orient="records")
    return load_bridge_constraints_from_records(
        records=records,
        start_ch_m=start_ch_m,
        end_ch_m=end_ch_m,
        config=config,
        config_path=config_path,
    )


def load_station_zones_from_records(
    records: Sequence[Mapping[str, Any]],
    start_ch_m: float,
    end_ch_m: float,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> list[StationZone]:
    """Parse railway station rows (Station_Excel.xlsx headers or explicit limits) into StationZones."""
    if not records:
        return []

    vcfg = load_vertical_config(config, config_path)
    aliases = vcfg.get("station_column_aliases", {})
    default_half_m = float(vcfg.get("station_yard_half_length_m", 500.0))
    by_class = {
        str(k).upper(): float(v)
        for k, v in vcfg.get("station_yard_half_lengths_by_class_m", {}).items()
    }
    default_max_g = float(vcfg["station_max_gradient"])

    all_keys: list[str] = []
    for rec in records:
        for k in rec.keys():
            if k not in all_keys:
                all_keys.append(str(k))
    col_map = _resolve_column_map(all_keys, aliases)

    zones: list[StationZone] = []
    for idx, row in enumerate(records, start=1):
        name = str(row.get(col_map.get("name", ""), "") or f"Station-{idx}").strip()
        stn_cls = str(row.get(col_map.get("station_class", ""), "") or "").strip().upper()
        max_g = _safe_float(row.get(col_map.get("max_gradient", "")))
        zone_max_g = max_g if max_g is not None else default_max_g

        s_explicit = _safe_float(row.get(col_map.get("start_chainage_m", "")))
        e_explicit = _safe_float(row.get(col_map.get("end_chainage_m", "")))
        if s_explicit is not None and e_explicit is not None and e_explicit > s_explicit:
            z_start = max(start_ch_m, s_explicit)
            z_end = min(end_ch_m, e_explicit)
        else:
            raw_m = _safe_float(row.get(col_map.get("chainage_m", "")))
            raw_km = _safe_float(row.get(col_map.get("location_km", "")))
            if raw_m is not None:
                center_m = resolve_bridge_chainage_m(raw_m, start_ch_m, end_ch_m, col_map.get("chainage_m", ""))
            elif raw_km is not None:
                center_m = resolve_bridge_chainage_m(raw_km, start_ch_m, end_ch_m, col_map.get("location_km", "km"))
            else:
                continue

            if center_m < start_ch_m - default_half_m or center_m > end_ch_m + default_half_m:
                continue

            half_m = by_class.get(stn_cls, default_half_m)
            z_start = max(start_ch_m, center_m - half_m)
            z_end = min(end_ch_m, center_m + half_m)

        if z_end > z_start:
            zones.append(
                StationZone(
                    start_chainage_m=round(z_start, 3),
                    end_chainage_m=round(z_end, 3),
                    name=f"{name} (Class {stn_cls})" if stn_cls else name,
                    max_gradient=zone_max_g,
                )
            )

    logger.info("Loaded %d railway station yard zone(s) from %d record(s).", len(zones), len(records))
    return zones

