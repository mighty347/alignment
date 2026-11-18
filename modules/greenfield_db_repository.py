"""
Module: modules/greenfield_db_repository.py
Purpose: PostgreSQL Repository (Reader & Writer) for Greenfield Composite Plan
         and Longitudinal Section (L-Section) generation.

Maps the live platform PostgreSQL schema (database_schema.sql) directly into
typed dataclasses and DataFrames compatible with CompositeDrawer, PlanDrawer,
LSectionDrawer, HorizontalGeometry, and VerticalAlignment:
    Read:
      - projects, project_lsections, lsection_versions
      - lsection_band_points   (replaces L-Section Excel)
      - lsection_curves        (replaces Prop. Curve Schedule Excel)
      - lsection_gradients     (replaces Gradient Schedule Excel)
      - lsection_stations      (replaces Station Schedule Excel)
      - sites, bridges, bridge_types, site_versions, site_version_bridges,
        lsections, hydrology, bridge_metadata (replaces Bridge List Excel)
    Write:
      - lsection_sheets        (persists generated multi-sheet DXF/PDF metadata)
      - lsection_versions      (updates master dxfUrl, status, updatedAt)

Standards: RDSO GE:G-1 / IRPWM / Indian Railways SOD (BG)
"""
from __future__ import annotations

import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional

import numpy as np
import pandas as pd

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]
    HAS_PSYCOPG2 = False

from utils.logger import get_logger

logger = get_logger(__name__)


class GreenfieldDBError(Exception):
    """Base exception raised by GreenfieldDBRepository."""


class VersionNotFoundError(GreenfieldDBError):
    """Raised when an lsection_versions record does not exist or is inactive."""


# ---------------------------------------------------------------------------
# Dataclasses (Typed Interface Contracts)
# ---------------------------------------------------------------------------

@dataclass
class GreenfieldProjectMeta:
    """Project and corridor boundary metadata from projects + project_lsections."""

    project_id: str
    project_name: str
    railway_zone: str
    division: str
    line_type: str
    gauge: str
    lsection_id: str
    start_chainage_m: float
    end_chainage_m: float


@dataclass
class GreenfieldVersionMeta:
    """Version metadata from lsection_versions."""

    version_id: str
    lsection_id: str
    name: str
    status: str
    band_headers: list[dict[str, Any]] = field(default_factory=list)
    alignment_kml_url: Optional[str] = None
    topo_kml_url: Optional[str] = None
    dxf_url: Optional[str] = None


@dataclass
class SheetWriteRecord:
    """Generated composite sheet record to persist into lsection_sheets."""

    sheet_number: int
    title: str
    start_chainage_m: float
    end_chainage_m: float
    datum_rl_m: float
    h_scale: int
    v_scale: int
    v_exaggeration: float
    dxf_url: Optional[str] = None
    pdf_url: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class GreenfieldDBBundle:
    """Complete DB-hydrated corridor bundle ready for CompositeDrawer."""

    project_meta: GreenfieldProjectMeta
    version_meta: GreenfieldVersionMeta
    band_points_df: pd.DataFrame
    curves_df: pd.DataFrame
    gradients_df: pd.DataFrame
    stations_df: pd.DataFrame
    bridges_df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    verification_status: str = "NOT VERIFIED"


# ---------------------------------------------------------------------------
# Pure Parsing & Normalization Helpers
# ---------------------------------------------------------------------------

def _validate_uuid(value: str, field_name: str) -> str:
    """Validate that a string is a canonical UUID and return normalized text."""
    if not value or not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty UUID string.")
    try:
        return str(uuid.UUID(value.strip()))
    except ValueError as err:
        raise ValueError(f"Invalid UUID for {field_name}: '{value}'") from err


def _safe_float(value: Any) -> Optional[float]:
    """Convert numeric or string value to finite float, else None."""
    if value is None:
        return None
    try:
        num = float(value)
        return num if math.isfinite(num) else None
    except (TypeError, ValueError):
        return None


def _parse_chainage_to_meters(raw: Any) -> Optional[float]:
    """Parse float meters or Indian Railways 'Km 1+250.50' / 'CH-02+040' to float meters."""
    direct = _safe_float(raw)
    if direct is not None:
        return direct
    if not isinstance(raw, str):
        return None

    text = raw.strip().replace(",", "")
    text = re.sub(r"(?i)^(?:km|ch)[\s.:_-]*", "", text)
    text = re.sub(r"(?i)\s*(?:km|mtr|m)\.?$", "", text).strip()

    if "+" in text:
        parts = text.split("+", 1)
        km_str = parts[0].strip()
        m_val = _safe_float(parts[1].strip())
        km_val = _safe_float(km_str)
        if km_val is not None and m_val is not None:
            sign = -1.0 if km_str.startswith("-") else 1.0
            return sign * (abs(km_val) * 1000.0 + abs(m_val))
    return _safe_float(text)


def _normalize_key(key: str) -> str:
    """Normalize a JSON band key for case/punctuation-insensitive alias lookup."""
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _extract_band_point_row(
    chainage_m: float,
    band_values: Any,
    alias_map: dict[str, list[str]],
) -> dict[str, Any]:
    """Flatten one lsection_band_points JSON row into canonical + raw columns."""
    raw_dict: dict[str, Any] = {}
    if isinstance(band_values, str):
        try:
            parsed = json.loads(band_values)
            if isinstance(parsed, dict):
                raw_dict = parsed
        except json.JSONDecodeError as err:
            logger.warning("Malformed bandValues JSON at chainage %.3f: %s", chainage_m, err)
    elif isinstance(band_values, dict):
        raw_dict = band_values

    normalized_lookup = {_normalize_key(k): v for k, v in raw_dict.items()}
    row: dict[str, Any] = {"chainage_m": float(chainage_m)}

    for canonical_col, aliases in alias_map.items():
        matched_val = None
        for alias in aliases:
            norm_alias = _normalize_key(alias)
            if norm_alias in normalized_lookup:
                matched_val = _safe_float(normalized_lookup[norm_alias])
                break
        row[canonical_col] = matched_val if matched_val is not None else np.nan

    for orig_key, orig_val in raw_dict.items():
        if orig_key not in row:
            row[orig_key] = orig_val

    return row


def _build_band_points_dataframe(
    rows: list[dict[str, Any]],
    alias_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Convert lsection_band_points SQL rows into a sorted profile DataFrame."""
    base_cols = ["chainage_m", *alias_map.keys()]
    if not rows:
        return pd.DataFrame(columns=base_cols)

    records: list[dict[str, Any]] = []
    for item in rows:
        ch_m = _safe_float(item.get("chainage"))
        if ch_m is None:
            continue
        records.append(_extract_band_point_row(ch_m, item.get("bandValues"), alias_map))

    if not records:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(records).sort_values("chainage_m").reset_index(drop=True)
    return df


def _build_curves_dataframe(rows: list[dict[str, Any]], g_value_m: float) -> pd.DataFrame:
    """Convert lsection_curves SQL rows into IRPWM/RDSO curve DataFrame."""
    columns = [
        "curve_no",
        "track_type",
        "track_name",
        "ts_chainage_m",
        "sc_chainage_m",
        "cs_chainage_m",
        "st_chainage_m",
        "total_length_m",
        "circular_length_m",
        "transition_length_m",
        "radius_m",
        "degree",
        "deflection_deg",
        "cant_mm",
        "direction",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for item in rows:
        ts_m = _safe_float(item.get("startChainage")) or 0.0
        st_m = _safe_float(item.get("endChainage")) or 0.0
        total_len = _safe_float(item.get("lengthM"))
        if total_len is None:
            total_len = max(0.0, st_m - ts_m)
        ls_m = _safe_float(item.get("transitionLengthM")) or 0.0
        radius_m = _safe_float(item.get("radiusM")) or 0.0
        degree = _safe_float(item.get("degree"))
        if degree is None and radius_m > 0.0:
            degree = round(g_value_m / radius_m, 4)

        sc_m = ts_m + ls_m
        cs_m = st_m - ls_m
        circular_len = total_len - (2.0 * ls_m)

        records.append({
            "curve_no": str(item.get("curveNo", "")).strip(),
            "track_type": str(item.get("trackType", "")).strip(),
            "track_name": str(item.get("trackName", "")).strip(),
            "ts_chainage_m": ts_m,
            "sc_chainage_m": sc_m,
            "cs_chainage_m": cs_m,
            "st_chainage_m": st_m,
            "total_length_m": total_len,
            "circular_length_m": circular_len,
            "transition_length_m": ls_m,
            "radius_m": radius_m,
            "degree": degree if degree is not None else np.nan,
            "deflection_deg": _safe_float(item.get("deflectionDeg")) or 0.0,
            "cant_mm": _safe_float(item.get("cantMm")) or 0.0,
            "direction": str(item.get("direction", "RIGHT")).strip().upper(),
        })

    return pd.DataFrame(records, columns=columns).sort_values("ts_chainage_m").reset_index(drop=True)


def _build_gradients_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert lsection_gradients SQL rows into VerticalAlignment DataFrame."""
    columns = [
        "seg_no",
        "start_chainage_m",
        "end_chainage_m",
        "length_m",
        "slope_ratio",
        "direction",
        "start_rl_m",
        "end_rl_m",
        "vpi_chainage_m",
        "vpi_elevation_m",
        "vc_length_m",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for item in rows:
        start_ch = _safe_float(item.get("startChainage")) or 0.0
        end_ch = _safe_float(item.get("endChainage")) or 0.0
        length_m = _safe_float(item.get("lengthM"))
        if length_m is None:
            length_m = max(0.0, end_ch - start_ch)

        records.append({
            "seg_no": int(item.get("segNo", 0)),
            "start_chainage_m": start_ch,
            "end_chainage_m": end_ch,
            "length_m": length_m,
            "slope_ratio": _safe_float(item.get("slopeRatio")) or 0.0,
            "direction": str(item.get("direction", "LEVEL")).strip().upper(),
            "start_rl_m": _safe_float(item.get("startRl")) or 0.0,
            "end_rl_m": _safe_float(item.get("endRl")) or 0.0,
            "vpi_chainage_m": _safe_float(item.get("vpiChainage")),
            "vpi_elevation_m": _safe_float(item.get("vpiElevation")),
            "vc_length_m": _safe_float(item.get("vcLength")) or 0.0,
        })

    return pd.DataFrame(records, columns=columns).sort_values("seg_no").reset_index(drop=True)


def _build_stations_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert lsection_stations SQL rows into station flagpole DataFrame."""
    columns = ["name", "chainage_m", "location_km", "station_class"]
    if not rows:
        return pd.DataFrame(columns=columns)

    records = [
        {
            "name": str(item.get("name", "")).strip(),
            "chainage_m": _safe_float(item.get("chainageM")) or 0.0,
            "location_km": _safe_float(item.get("locationKm")) or 0.0,
            "station_class": str(item.get("stationClass") or "").strip(),
        }
        for item in rows
    ]
    return pd.DataFrame(records, columns=columns).sort_values("chainage_m").reset_index(drop=True)


def _build_bridges_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert DB bridge/lsection/hydrology rows into CompositeDrawer bridge_df."""
    columns = [
        "Bridge No",
        "Bridge Name",
        "Chainage",
        "Bridge Type",
        "Status",
        "Bed Level",
        "HFL",
        "Rail Level",
        "Formation Level",
        "Road Level",
        "Skew Angle",
        "Flow Direction",
        "Linear Waterway",
        "Max Scour Level",
        "Metadata",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for item in rows:
        ch_raw = item.get("lsection_chainage") or item.get("site_chainage")
        ch_m = _parse_chainage_to_meters(ch_raw)
        records.append({
            "Bridge No": str(item.get("bridgeNumber") or "").strip(),
            "Bridge Name": str(item.get("bridge_name") or item.get("site_name") or "").strip(),
            "Chainage": ch_m if ch_m is not None else np.nan,
            "Bridge Type": str(item.get("bridge_type_name") or item.get("bridge_type_code") or "").strip(),
            "Status": str(item.get("bridgeStatus") or "PROPOSED").strip(),
            "Bed Level": _safe_float(item.get("bedLevel")),
            "HFL": _safe_float(item.get("hfl")),
            "Rail Level": _safe_float(item.get("railLevel")),
            "Formation Level": _safe_float(item.get("formationLevel")),
            "Road Level": _safe_float(item.get("roadLevel")),
            "Skew Angle": _safe_float(item.get("skewAngle")),
            "Flow Direction": str(item.get("flowDirection") or "").strip(),
            "Linear Waterway": _safe_float(item.get("linearWaterway")),
            "Max Scour Level": _safe_float(item.get("maxScourLevel")),
            "Metadata": item.get("metadataValues") or {},
        })

    df = pd.DataFrame(records, columns=columns)
    return df.sort_values("Chainage", na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Engineering Verification Gate (DB Data Invariants)
# ---------------------------------------------------------------------------

def validate_bundle_invariants(
    bundle: GreenfieldDBBundle,
    track_buildup_m: float,
    buildup_tolerance_m: float,
    cut_fill_tolerance_m: float,
) -> tuple[list[str], str]:
    """Verify RDSO railway engineering invariants on loaded DB data.

    Checks:
        1. Chainage strict monotonicity on band_points_df (s_{i+1} > s_i).
        2. Track buildup invariant: PRL - PFL == track_buildup_m (0.762 m BG).
        3. Horizontal curve geometry: R > 0, ST > TS, and circular arc Lc >= 0.
        4. Vertical gradient spans: end_chainage_m > start_chainage_m.
        5. Profile cut/fill consistency: cut_fill_m == PFL - EGL.

    Returns:
        Tuple of (warnings_or_failures, verification_status ['PASS' | 'FAIL']).
    """
    issues: list[str] = []
    has_failure = False

    bp_df = bundle.band_points_df
    if not bp_df.empty and "chainage_m" in bp_df.columns:
        diffs = np.diff(bp_df["chainage_m"].to_numpy(dtype=float))
        if np.any(diffs <= 0.0):
            has_failure = True
            bad_count = int(np.sum(diffs <= 0.0))
            issues.append(
                f"FAIL: Chainage monotonicity violated at {bad_count} station(s) in lsection_band_points."
            )

        if {"proposed_rail_level_m", "formation_level_m"}.issubset(bp_df.columns):
            valid_mask = bp_df["proposed_rail_level_m"].notna() & bp_df["formation_level_m"].notna()
            if valid_mask.any():
                actual_buildup = (
                    bp_df.loc[valid_mask, "proposed_rail_level_m"]
                    - bp_df.loc[valid_mask, "formation_level_m"]
                )
                max_dev = float(np.max(np.abs(actual_buildup - track_buildup_m)))
                if max_dev > buildup_tolerance_m:
                    has_failure = True
                    issues.append(
                        f"FAIL: Track buildup invariant (PRL - PFL = {track_buildup_m:.3f}m) "
                        f"violated with max deviation {max_dev:.4f}m."
                    )

        if {"cut_fill_m", "formation_level_m", "elevation_m"}.issubset(bp_df.columns):
            cf_mask = (
                bp_df["cut_fill_m"].notna()
                & bp_df["formation_level_m"].notna()
                & bp_df["elevation_m"].notna()
            )
            if cf_mask.any():
                expected_cf = bp_df.loc[cf_mask, "formation_level_m"] - bp_df.loc[cf_mask, "elevation_m"]
                max_cf_dev = float(np.max(np.abs(bp_df.loc[cf_mask, "cut_fill_m"] - expected_cf)))
                if max_cf_dev > cut_fill_tolerance_m:
                    issues.append(
                        f"WARNING: cut_fill_m deviates from (PFL - EGL) by up to {max_cf_dev:.3f}m."
                    )

    c_df = bundle.curves_df
    if not c_df.empty:
        bad_r = c_df[c_df["radius_m"] <= 0.0]
        if not bad_r.empty:
            has_failure = True
            issues.append(f"FAIL: {len(bad_r)} curve(s) have non-positive radiusM.")

        bad_span = c_df[c_df["st_chainage_m"] <= c_df["ts_chainage_m"]]
        if not bad_span.empty:
            has_failure = True
            issues.append(f"FAIL: {len(bad_span)} curve(s) have endChainage <= startChainage.")

        bad_lc = c_df[c_df["circular_length_m"] < -1e-6]
        if not bad_lc.empty:
            has_failure = True
            issues.append(
                f"FAIL: {len(bad_lc)} curve(s) have negative circular length (2 * transitionLengthM > lengthM)."
            )

    g_df = bundle.gradients_df
    if not g_df.empty:
        bad_g = g_df[g_df["end_chainage_m"] <= g_df["start_chainage_m"]]
        if not bad_g.empty:
            has_failure = True
            issues.append(f"FAIL: {len(bad_g)} gradient segment(s) have endChainage <= startChainage.")

    status = "FAIL" if has_failure else "PASS"
    return issues, status


# ---------------------------------------------------------------------------
# PostgreSQL Repository Class
# ---------------------------------------------------------------------------

class GreenfieldDBRepository:
    """Reads Greenfield corridor/bridge inputs from PostgreSQL and writes lsection_sheets."""

    def __init__(
        self,
        config: dict[str, Any],
        connection_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary loaded from config/config.yaml.")
        self.config = config
        self._db_cfg = config.get("greenfield_db", {})
        self._vertical_cfg = config.get("vertical", {})
        self._railway_cfg = config.get("railway", {})
        self._drawing_cfg = config.get("drawing", {})
        self._connection_factory = connection_factory

        self.track_buildup_m = float(self._vertical_cfg.get("track_buildup_m", 0.762))
        self.buildup_tolerance_m = float(self._db_cfg.get("buildup_tolerance_m", 0.005))
        self.cut_fill_tolerance_m = float(self._db_cfg.get("cut_fill_tolerance_m", 0.02))
        self.g_value_m = float(self._railway_cfg.get("g_value_mm", 1750.0))
        self.soft_delete_sheets = bool(self._db_cfg.get("soft_delete_sheets", True))
        self.default_version_status = str(self._db_cfg.get("default_version_status", "GENERATED"))
        self.alias_map: dict[str, list[str]] = self._db_cfg.get("band_value_aliases", {})

    @contextmanager
    def _get_connection(self) -> Generator[Any, None, None]:
        """Yield a PostgreSQL connection from injected factory or psycopg2 config."""
        if self._connection_factory is not None:
            conn = self._connection_factory()
            try:
                yield conn
            finally:
                close_fn = getattr(conn, "close", None)
                if callable(close_fn):
                    close_fn()
            return

        if not HAS_PSYCOPG2:
            raise GreenfieldDBError("psycopg2 is required for PostgreSQL database access.")

        url_env = str(self._db_cfg.get("database_url_env", "DATABASE_URL"))
        dsn = os.environ.get(url_env)
        timeout_s = int(self._db_cfg.get("connect_timeout_s", 10))

        if dsn:
            conn = psycopg2.connect(dsn, connect_timeout=timeout_s)
        else:
            pg = self._db_cfg.get("postgres", {})
            conn = psycopg2.connect(
                host=pg.get("host", "localhost"),
                port=int(pg.get("port", 5432)),
                dbname=pg.get("dbname", "esp_db"),
                user=pg.get("user", "postgres"),
                password=pg.get("password", "postgres"),
                connect_timeout=timeout_s,
            )

        try:
            yield conn
        finally:
            conn.close()

    def _open_cursor(self, conn: Any) -> Any:
        """Open a dict-returning cursor (works with both psycopg2 and mock connections)."""
        if RealDictCursor is not None and self._connection_factory is None:
            return conn.cursor(cursor_factory=RealDictCursor)
        return conn.cursor()

    def _fetch_version_and_project(
        self, cur: Any, version_id: str
    ) -> tuple[GreenfieldProjectMeta, GreenfieldVersionMeta]:
        """Fetch version, corridor bounds, and parent project metadata."""
        sql = """
            SELECT
                v.id                  AS version_id,
                v."lsectionId"        AS lsection_id,
                v.name                AS version_name,
                v.status              AS version_status,
                v."bandHeaders"       AS band_headers,
                v."alignmentKmlUrl"   AS alignment_kml_url,
                v."topoKmlUrl"        AS topo_kml_url,
                v."dxfUrl"            AS dxf_url,
                pl."startChainage"    AS start_chainage,
                pl."endChainage"      AS end_chainage,
                p.id                  AS project_id,
                p.name                AS project_name,
                p."railwayZone"       AS railway_zone,
                p.division            AS division,
                p."lineType"          AS line_type,
                p."proposedGauge"     AS proposed_gauge,
                p."existingGauge"     AS existing_gauge
            FROM public.lsection_versions v
            INNER JOIN public.project_lsections pl
                ON pl.id = v."lsectionId" AND pl."auditStatus" = 'ACTIVE'
            INNER JOIN public.projects p
                ON p.id = pl."projectId" AND p."auditStatus" = 'ACTIVE'
            WHERE v.id = %s
              AND v."auditStatus" = 'ACTIVE'
            LIMIT 1
        """
        cur.execute(sql, (version_id,))
        row = cur.fetchone()
        if not row:
            raise VersionNotFoundError(
                f"Active lsection_versions record not found for version_id='{version_id}'."
            )

        raw_headers = row.get("band_headers")
        if isinstance(raw_headers, str):
            raw_headers = json.loads(raw_headers)
        band_headers = raw_headers if isinstance(raw_headers, list) else []

        gauge = str(
            row.get("proposed_gauge")
            or row.get("existing_gauge")
            or self._railway_cfg.get("gauge", "BG")
        ).strip()

        project_meta = GreenfieldProjectMeta(
            project_id=str(row["project_id"]),
            project_name=str(row["project_name"]),
            railway_zone=str(row.get("railway_zone") or ""),
            division=str(row.get("division") or ""),
            line_type=str(row.get("line_type") or "GREENFIELD"),
            gauge=gauge,
            lsection_id=str(row["lsection_id"]),
            start_chainage_m=float(row.get("start_chainage") or 0.0),
            end_chainage_m=float(row.get("end_chainage") or 0.0),
        )
        version_meta = GreenfieldVersionMeta(
            version_id=str(row["version_id"]),
            lsection_id=str(row["lsection_id"]),
            name=str(row["version_name"]),
            status=str(row["version_status"]),
            band_headers=band_headers,
            alignment_kml_url=row.get("alignment_kml_url"),
            topo_kml_url=row.get("topo_kml_url"),
            dxf_url=row.get("dxf_url"),
        )
        return project_meta, version_meta

    def _fetch_band_points(self, cur: Any, version_id: str) -> pd.DataFrame:
        """Fetch ordered station band points from lsection_band_points."""
        sql = """
            SELECT chainage, "bandValues"
            FROM public.lsection_band_points
            WHERE "versionId" = %s
              AND "auditStatus" = 'ACTIVE'
            ORDER BY chainage ASC
        """
        cur.execute(sql, (version_id,))
        return _build_band_points_dataframe(list(cur.fetchall()), self.alias_map)

    def _fetch_curves(self, cur: Any, version_id: str) -> pd.DataFrame:
        """Fetch horizontal track curves from lsection_curves."""
        sql = """
            SELECT
                "trackType", "trackName", "curveNo",
                "startChainage", "endChainage", "lengthM",
                "radiusM", degree, "deflectionDeg",
                "cantMm", "transitionLengthM", direction
            FROM public.lsection_curves
            WHERE "versionId" = %s
              AND "auditStatus" = 'ACTIVE'
            ORDER BY "startChainage" ASC
        """
        cur.execute(sql, (version_id,))
        return _build_curves_dataframe(list(cur.fetchall()), self.g_value_m)

    def _fetch_gradients(self, cur: Any, version_id: str) -> pd.DataFrame:
        """Fetch vertical gradient segments from lsection_gradients."""
        sql = """
            SELECT
                "segNo", "startChainage", "endChainage", "lengthM",
                "slopeRatio", direction, "startRl", "endRl",
                "vpiChainage", "vpiElevation", "vcLength"
            FROM public.lsection_gradients
            WHERE "versionId" = %s
              AND "auditStatus" = 'ACTIVE'
            ORDER BY "segNo" ASC
        """
        cur.execute(sql, (version_id,))
        return _build_gradients_dataframe(list(cur.fetchall()))

    def _fetch_stations(self, cur: Any, lsection_id: str) -> pd.DataFrame:
        """Fetch railway stations from lsection_stations."""
        sql = """
            SELECT name, "chainageM", "locationKm", "stationClass"
            FROM public.lsection_stations
            WHERE "lsectionId" = %s
              AND "auditStatus" = 'ACTIVE'
            ORDER BY "chainageM" ASC
        """
        cur.execute(sql, (lsection_id,))
        return _build_stations_dataframe(list(cur.fetchall()))

    def _fetch_project_bridges(self, cur: Any, project_id: str) -> pd.DataFrame:
        """Fetch active bridges and latest version levels/hydrology for a project."""
        sql = """
            SELECT DISTINCT ON (b.id)
                b."bridgeNumber",
                b.name                  AS bridge_name,
                b."bridgeStatus",
                s.name                  AS site_name,
                s.chainage              AS site_chainage,
                bt.code                 AS bridge_type_code,
                bt.name                 AS bridge_type_name,
                ls.chainage             AS lsection_chainage,
                ls."bedLevel",
                ls.hfl,
                ls."railLevel",
                ls."formationLevel",
                ls."roadLevel",
                ls."skewAngle",
                ls."flowDirection",
                hy."linearWaterway",
                hy."maxScourLevel",
                bm."metadataValues"
            FROM public.sites s
            INNER JOIN public.bridges b
                ON b."siteId" = s.id AND b."auditStatus" = 'ACTIVE'
            INNER JOIN public.bridge_types bt
                ON bt.id = b."bridgeTypeId"
            LEFT JOIN public.site_versions sv
                ON sv."siteId" = s.id AND sv."auditStatus" = 'ACTIVE'
            LEFT JOIN public.site_version_bridges svb
                ON svb."siteVersionId" = sv.id
               AND svb."bridgeId" = b.id
               AND svb."auditStatus" = 'ACTIVE'
            LEFT JOIN public.lsections ls
                ON ls.id = svb."lsectionId"
            LEFT JOIN public.hydrology hy
                ON hy.id = svb."hydrologyId"
            LEFT JOIN public.bridge_metadata bm
                ON bm.id = svb."bridgeMetadataId"
            WHERE s."projectId" = %s
              AND s."auditStatus" = 'ACTIVE'
            ORDER BY b.id, sv."versionNumber" DESC NULLS LAST
        """
        cur.execute(sql, (project_id,))
        return _build_bridges_dataframe(list(cur.fetchall()))

    def load_version_bundle(self, version_id: str) -> GreenfieldDBBundle:
        """Load complete Greenfield corridor dataset from PostgreSQL for a version."""
        clean_version_id = _validate_uuid(version_id, "version_id")
        logger.info("Loading Greenfield DB bundle for version_id=%s", clean_version_id)

        try:
            with self._get_connection() as conn:
                with self._open_cursor(conn) as cur:
                    project_meta, version_meta = self._fetch_version_and_project(cur, clean_version_id)
                    band_points_df = self._fetch_band_points(cur, clean_version_id)
                    curves_df = self._fetch_curves(cur, clean_version_id)
                    gradients_df = self._fetch_gradients(cur, clean_version_id)
                    stations_df = self._fetch_stations(cur, project_meta.lsection_id)
                    bridges_df = self._fetch_project_bridges(cur, project_meta.project_id)
        except VersionNotFoundError:
            raise
        except Exception as err:
            logger.exception("Failed loading Greenfield DB bundle for version_id=%s: %s", clean_version_id, err)
            raise GreenfieldDBError(f"Database read failed for version '{clean_version_id}': {err}") from err

        bundle = GreenfieldDBBundle(
            project_meta=project_meta,
            version_meta=version_meta,
            band_points_df=band_points_df,
            curves_df=curves_df,
            gradients_df=gradients_df,
            stations_df=stations_df,
            bridges_df=bridges_df,
        )

        warnings, verification_status = validate_bundle_invariants(
            bundle=bundle,
            track_buildup_m=self.track_buildup_m,
            buildup_tolerance_m=self.buildup_tolerance_m,
            cut_fill_tolerance_m=self.cut_fill_tolerance_m,
        )
        bundle.warnings = warnings
        bundle.verification_status = verification_status

        logger.info(
            "Loaded bundle for '%s' (version=%s): %d band points, %d curves, "
            "%d gradients, %d stations, %d bridges [Gate=%s]",
            project_meta.project_name,
            version_meta.name,
            len(band_points_df),
            len(curves_df),
            len(gradients_df),
            len(stations_df),
            len(bridges_df),
            verification_status,
        )
        return bundle

    def save_generated_sheets(
        self,
        version_id: str,
        sheets: list[SheetWriteRecord],
        master_dxf_url: Optional[str] = None,
        version_status: Optional[str] = None,
        updated_by: Optional[str] = None,
    ) -> list[str]:
        """Atomically persist generated composite sheets into lsection_sheets and update version."""
        clean_version_id = _validate_uuid(version_id, "version_id")
        if not sheets:
            raise ValueError("sheets list must contain at least one SheetWriteRecord.")

        clean_user_id = _validate_uuid(updated_by, "updated_by") if updated_by else None
        target_status = version_status or self.default_version_status
        inserted_ids: list[str] = []

        try:
            with self._get_connection() as conn:
                try:
                    with self._open_cursor(conn) as cur:
                        if self.soft_delete_sheets:
                            cur.execute(
                                """
                                UPDATE public.lsection_sheets
                                SET "auditStatus" = 'DELETED',
                                    "updatedAt" = CURRENT_TIMESTAMP,
                                    "updatedBy" = COALESCE(%s, "updatedBy")
                                WHERE "versionId" = %s
                                  AND "auditStatus" = 'ACTIVE'
                                """,
                                (clean_user_id, clean_version_id),
                            )
                        else:
                            cur.execute(
                                'DELETE FROM public.lsection_sheets WHERE "versionId" = %s',
                                (clean_version_id,),
                            )

                        insert_sql = """
                            INSERT INTO public.lsection_sheets (
                                id, "versionId", "sheetNumber", title,
                                "startChainage", "endChainage", "datumRl",
                                "hScale", "vScale", "vExaggeration",
                                "dxfUrl", "pdfUrl", "createdBy", "updatedBy", notes
                            )
                            VALUES (
                                %s, %s, %s, %s,
                                %s, %s, %s,
                                %s, %s, %s,
                                %s, %s, %s, %s, %s
                            )
                        """
                        for sheet in sheets:
                            if sheet.end_chainage_m <= sheet.start_chainage_m:
                                raise ValueError(
                                    f"Sheet {sheet.sheet_number} end_chainage_m must exceed start_chainage_m."
                                )
                            sheet_id = str(uuid.uuid4())
                            cur.execute(
                                insert_sql,
                                (
                                    sheet_id,
                                    clean_version_id,
                                    int(sheet.sheet_number),
                                    sheet.title,
                                    float(sheet.start_chainage_m),
                                    float(sheet.end_chainage_m),
                                    float(sheet.datum_rl_m),
                                    int(sheet.h_scale),
                                    int(sheet.v_scale),
                                    float(sheet.v_exaggeration),
                                    sheet.dxf_url,
                                    sheet.pdf_url,
                                    clean_user_id,
                                    clean_user_id,
                                    sheet.notes,
                                ),
                            )
                            inserted_ids.append(sheet_id)

                        cur.execute(
                            """
                            UPDATE public.lsection_versions
                            SET status = %s,
                                "dxfUrl" = COALESCE(%s, "dxfUrl"),
                                "updatedAt" = CURRENT_TIMESTAMP,
                                "updatedBy" = COALESCE(%s, "updatedBy")
                            WHERE id = %s
                              AND "auditStatus" = 'ACTIVE'
                            """,
                            (target_status, master_dxf_url, clean_user_id, clean_version_id),
                        )

                    conn.commit()
                    logger.info(
                        "Saved %d composite sheet(s) to lsection_sheets for version_id=%s",
                        len(inserted_ids),
                        clean_version_id,
                    )
                    return inserted_ids
                except Exception:
                    conn.rollback()
                    raise
        except ValueError:
            raise
        except Exception as err:
            logger.exception("Failed saving generated sheets for version_id=%s: %s", clean_version_id, err)
            raise GreenfieldDBError(f"Database write failed for version '{clean_version_id}': {err}") from err
