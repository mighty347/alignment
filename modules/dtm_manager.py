"""
Module: modules/dtm_manager.py
Purpose: Module 40 — DTM Cloud Ingestion, PostgreSQL/PostGIS Spatial Metadata Registry,
         and Fast Runtime Surface Sampling Engine.

Engineering Basis:
    Digital Terrain Models (DEM/DTM in GeoTIFF, ESRI ASCII Grid, or DEM formats)
    provide the 3D topographic surface Z = f(X, Y) required for Longitudinal
    Section (L-Section) profiles.
    - Large terrain rasters uploaded to S3 or direct web uploads are ingested and stored
      into a persistent PostgreSQL / PostGIS database (esp_db).
    - Spatial footprints are indexed via PostGIS GIST geometry for instant spatial lookups
      along railway corridors.
    - Compressed elevation grids are stored in PostgreSQL binary columns (BYTEA) allowing
      on-demand reconstruction into full GeoTIFF files or instant in-memory DEMResult objects.
    - For real-time railway CAD alignment editing, station elevation querying uses
      sub-pixel bilinear interpolation to return accurate ground levels (EGL) in <30ms,
      avoiding stepped nearest-neighbor artifacts.
    - Calculates recommended profile datum: floor(min(EGL) - datum_margin_m) rounded
      down to datum_round_m per Indian Railways RDSO practice.

Standards: RDSO / Indian Railways BG
Author: Railway ESP Generator
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sqlite3
import time
import uuid
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.interpolate import RegularGridInterpolator
import yaml

try:
    import psycopg2
    from psycopg2.extras import DictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from alignment.utils.logger import get_logger
except ImportError:
    try:
        from utils.logger import get_logger
    except ImportError:
        import logging
        def get_logger(name: str):
            return logging.getLogger(name)

logger = get_logger("DTMManager")

LOCAL_ALIGNMENT_CONFIG = Path(__file__).resolve().parent.parent / "config/alignment_config.yaml"
CONFIG_PATH_DEFAULT = LOCAL_ALIGNMENT_CONFIG if LOCAL_ALIGNMENT_CONFIG.exists() else Path(__file__).resolve().parents[1] / "config/config.yaml"
ALT_CONFIG_PATH = LOCAL_ALIGNMENT_CONFIG


@dataclass
class DTMRecord:
    """Metadata record representing an ingested Digital Terrain Model."""
    project_id: str
    source_filename: str
    crs_epsg: int
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_elevation_m: float
    max_elevation_m: float
    resolution_x_m: float
    resolution_y_m: float
    status: str
    created_at: float
    geom_wkt: Optional[str] = None


class DTMManager:
    """
    Manages DTM ingestion, S3 pre-signed upload URLs, asynchronous background processing,
    PostgreSQL/PostGIS registry persistence with GeoTIFF export, and fast sub-pixel
    runtime surface sampling.
    """

    def __init__(self, config_path: Optional[Path] = None, db_backend: Optional[str] = None):
        cfg_file = config_path or CONFIG_PATH_DEFAULT
        if not cfg_file.exists() and ALT_CONFIG_PATH.exists():
            cfg_file = ALT_CONFIG_PATH

        self.config: Dict[str, Any] = {}
        if cfg_file.exists():
            with cfg_file.open(encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        dtm_cfg = self.config.get("dtm_settings", {})
        osm_pg_cfg = self.config.get("osm_postgis", {})
        base_dir = Path(__file__).resolve().parents[1]

        rel_storage = dtm_cfg.get("storage_dir", "storage/dtm")
        self.storage_dir = (base_dir / rel_storage).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        rel_db = dtm_cfg.get("db_path", "storage/dtm/dtm_registry.sqlite")
        self.db_path = (base_dir / rel_db).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Database configuration
        self.db_backend = (db_backend or dtm_cfg.get("db_backend", "postgres")).lower()
        pg_cfg = dtm_cfg.get("postgres", {})
        self.pg_host = os.environ.get("PGHOST", pg_cfg.get("host", osm_pg_cfg.get("host", "localhost")))
        self.pg_port = int(os.environ.get("PGPORT", pg_cfg.get("port", osm_pg_cfg.get("port", 5432))))
        self.pg_dbname = os.environ.get("PGDATABASE", pg_cfg.get("dbname", osm_pg_cfg.get("dbname", "esp_db")))
        self.pg_user = os.environ.get("PGUSER", pg_cfg.get("user", osm_pg_cfg.get("user", "postgres")))
        self.pg_password = os.environ.get("PGPASSWORD", pg_cfg.get("password", osm_pg_cfg.get("password", "postgres")))

        self.default_s3_bucket = dtm_cfg.get("default_s3_bucket", "esp-railway-dtm-bucket")
        self.default_s3_region = dtm_cfg.get("default_s3_region", "ap-south-1")
        self.presigned_expiry = int(dtm_cfg.get("presigned_url_expiration_seconds", 1800))
        self.datum_margin_m = float(dtm_cfg.get("datum_margin_m", 5.0))
        self.datum_round_m = float(dtm_cfg.get("datum_round_m", 10.0))
        self.max_sampling_points = int(dtm_cfg.get("max_sampling_points", 50000))

        # In-memory raster cache for rapid hot-loop sampling (< 30ms)
        self._interpolator_cache: Dict[str, Tuple[RegularGridInterpolator, tuple, float]] = {}

        self._use_postgres = False
        if self.db_backend == "postgres":
            if not HAS_PSYCOPG2:
                logger.warning("psycopg2 is not installed; falling back to SQLite registry.")
            else:
                try:
                    with psycopg2.connect(
                        host=self.pg_host,
                        port=self.pg_port,
                        dbname=self.pg_dbname,
                        user=self.pg_user,
                        password=self.pg_password,
                        connect_timeout=3,
                    ) as test_conn:
                        pass
                    self._use_postgres = True
                    logger.info(
                        "Connected to PostgreSQL database '%s' at %s:%d for DTM storage.",
                        self.pg_dbname,
                        self.pg_host,
                        self.pg_port,
                    )
                except Exception as pg_err:
                    logger.warning(
                        "Could not connect to PostgreSQL (%s). Falling back to SQLite registry at %s",
                        pg_err,
                        self.db_path,
                    )

        self._init_db()
        logger.info(
            "Initialized DTMManager with backend=%s, storage_dir=%s",
            "PostgreSQL" if self._use_postgres else "SQLite",
            self.storage_dir,
        )

    @contextlib.contextmanager
    def _get_pg_conn(self) -> Generator[Any, None, None]:
        """Context manager yielding an active PostgreSQL connection."""
        if not HAS_PSYCOPG2:
            raise RuntimeError("psycopg2 is required for PostgreSQL DTM operations.")
        conn = psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            dbname=self.pg_dbname,
            user=self.pg_user,
            password=self.pg_password,
        )
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize PostgreSQL/PostGIS spatial metadata registry and raster tables, or SQLite fallback."""
        if self._use_postgres:
            try:
                with self._get_pg_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS dtm_records (
                                project_id VARCHAR(128) PRIMARY KEY,
                                source_filename VARCHAR(255) NOT NULL,
                                crs_epsg INTEGER NOT NULL,
                                min_x DOUBLE PRECISION NOT NULL,
                                min_y DOUBLE PRECISION NOT NULL,
                                max_x DOUBLE PRECISION NOT NULL,
                                max_y DOUBLE PRECISION NOT NULL,
                                min_elevation_m DOUBLE PRECISION NOT NULL,
                                max_elevation_m DOUBLE PRECISION NOT NULL,
                                resolution_x_m DOUBLE PRECISION NOT NULL,
                                resolution_y_m DOUBLE PRECISION NOT NULL,
                                status VARCHAR(64) NOT NULL,
                                created_at DOUBLE PRECISION NOT NULL,
                                geom geometry(Polygon, 4326)
                            );
                            CREATE INDEX IF NOT EXISTS idx_dtm_records_geom ON dtm_records USING GIST (geom);

                            -- Drop legacy unused path columns if present
                            ALTER TABLE dtm_records DROP COLUMN IF EXISTS s3_key;
                            ALTER TABLE dtm_records DROP COLUMN IF EXISTS local_path;

                            CREATE TABLE IF NOT EXISTS dtm_rasters (
                                project_id VARCHAR(128) PRIMARY KEY REFERENCES dtm_records(project_id) ON DELETE CASCADE,
                                width INTEGER NOT NULL,
                                height INTEGER NOT NULL,
                                nodata_val DOUBLE PRECISION,
                                transform_json TEXT NOT NULL,
                                raster_blob BYTEA NOT NULL,
                                updated_at DOUBLE PRECISION NOT NULL
                            );
                            """
                        )
                    conn.commit()
                return
            except Exception as exc:
                logger.error("Failed to initialize PostgreSQL DTM tables: %s. Reverting to SQLite.", exc)
                self._use_postgres = False

        # SQLite Fallback Registry
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dtm_records (
                    project_id TEXT PRIMARY KEY,
                    source_filename TEXT NOT NULL,
                    crs_epsg INTEGER NOT NULL,
                    min_x REAL NOT NULL,
                    min_y REAL NOT NULL,
                    max_x REAL NOT NULL,
                    max_y REAL NOT NULL,
                    min_elevation_m REAL NOT NULL,
                    max_elevation_m REAL NOT NULL,
                    resolution_x_m REAL NOT NULL,
                    resolution_y_m REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            # Migrate SQLite table if legacy columns exist
            try:
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(dtm_records)")
                cols = [row[1] for row in cur.fetchall()]
                if "s3_key" in cols or "local_path" in cols:
                    cur.execute(
                        """
                        CREATE TABLE dtm_records_new (
                            project_id TEXT PRIMARY KEY,
                            source_filename TEXT NOT NULL,
                            crs_epsg INTEGER NOT NULL,
                            min_x REAL NOT NULL,
                            min_y REAL NOT NULL,
                            max_x REAL NOT NULL,
                            max_y REAL NOT NULL,
                            min_elevation_m REAL NOT NULL,
                            max_elevation_m REAL NOT NULL,
                            resolution_x_m REAL NOT NULL,
                            resolution_y_m REAL NOT NULL,
                            status TEXT NOT NULL,
                            created_at REAL NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        """
                        INSERT INTO dtm_records_new
                        SELECT project_id, source_filename, crs_epsg,
                               min_x, min_y, max_x, max_y, min_elevation_m, max_elevation_m,
                               resolution_x_m, resolution_y_m, status, created_at
                        FROM dtm_records
                        """
                    )
                    cur.execute("DROP TABLE dtm_records")
                    cur.execute("ALTER TABLE dtm_records_new RENAME TO dtm_records")
            except Exception as mig_err:
                logger.debug("SQLite column migration notice: %s", mig_err)

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dtm_rasters (
                    project_id TEXT PRIMARY KEY,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    nodata_val REAL,
                    transform_json TEXT NOT NULL,
                    raster_blob BLOB NOT NULL
                )
                """
            )
            conn.commit()

    def generate_presigned_upload_url(
        self, project_id: str, filename: str
    ) -> Dict[str, Any]:
        """
        Generates an S3 pre-signed upload URL for direct-to-S3 client uploads.
        Falls back to local direct upload URL if AWS credentials are not configured.
        """
        clean_name = Path(filename).name
        s3_key = f"projects/{project_id}/dtm/{clean_name}"
        has_aws_creds = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )

        if has_aws_creds:
            try:
                import boto3  # type: ignore

                s3_client = boto3.client(
                    "s3",
                    region_name=self.default_s3_region,
                )
                upload_url = s3_client.generate_presigned_url(
                    ClientMethod="put_object",
                    Params={
                        "Bucket": self.default_s3_bucket,
                        "Key": s3_key,
                        "ContentType": "application/octet-stream",
                    },
                    ExpiresIn=self.presigned_expiry,
                )
                return {
                    "mode": "s3",
                    "project_id": project_id,
                    "s3_key": s3_key,
                    "bucket": self.default_s3_bucket,
                    "upload_url": upload_url,
                    "expires_in_seconds": self.presigned_expiry,
                }
            except Exception as exc:
                logger.warning(
                    "Could not generate AWS S3 presigned URL (%s). Falling back to local upload route.",
                    exc,
                )

        # Local development / offline fallback route
        local_upload_route = f"/api/v1/alignment-studio/dtm/upload-direct?project_id={project_id}&filename={clean_name}"
        return {
            "mode": "local_fallback",
            "project_id": project_id,
            "s3_key": s3_key,
            "bucket": "local-storage",
            "upload_url": local_upload_route,
            "expires_in_seconds": self.presigned_expiry,
        }

    def get_raw_upload_path(self, project_id: str, filename: str) -> Path:
        """Returns local staging path for a direct DTM upload."""
        return self.storage_dir / f"{project_id}_raw_{Path(filename).name}"

    def save_direct_upload(
        self, project_id: str, filename: str, data: bytes
    ) -> Path:
        """Saves a direct file upload to local staging storage."""
        target_path = self.get_raw_upload_path(project_id, filename)
        target_path.write_bytes(data)
        logger.info(
            "Saved %d bytes for project '%s' to %s",
            len(data),
            project_id,
            target_path,
        )
        return target_path

    def process_dtm_job(
        self,
        job_id: str,
        project_id: str,
        s3_key_or_path: str,
        target_epsg: Optional[int] = None,
        job_manager_instance: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Long-running background worker task:
        1. Reads raw GeoTIFF or ESRI ASCII Grid (.asc).
        2. Validates CRS & reprojects to metric UTM space if needed.
        3. Converts into an optimized tiled GeoTIFF in storage.
        4. Calculates spatial bounds and min/max elevation.
        5. Saves metadata & compressed elevation grid to PostgreSQL/PostGIS registry.
        """
        logger.info(
            "Starting DTM ingestion job %s for project '%s' (Source: %s)",
            job_id,
            project_id,
            s3_key_or_path,
        )

        def emit(progress: int, step_msg: str) -> None:
            if job_manager_instance:
                job_manager_instance.emit_event(
                    job_id,
                    "progress",
                    {"progress_percent": progress, "step": step_msg},
                )

        emit(10, "Locating terrain dataset...")
        source_path = Path(s3_key_or_path)

        # Download from S3 if s3_key is an actual S3 path and AWS is configured
        if s3_key_or_path.startswith("projects/") and not source_path.exists():
            emit(15, "Streaming terrain data from S3 storage...")
            try:
                import boto3  # type: ignore

                local_dl = self.storage_dir / f"{project_id}_s3_download.tif"
                s3_client = boto3.client("s3", region_name=self.default_s3_region)
                s3_client.download_file(self.default_s3_bucket, s3_key_or_path, str(local_dl))
                source_path = local_dl
            except Exception as exc:
                raise RuntimeError(f"Could not download DTM from S3 key '{s3_key_or_path}': {exc}") from exc

        if not source_path.exists():
            raise FileNotFoundError(f"DTM dataset file not found: {source_path}")

        emit(30, "Reading raster dataset and inspecting CRS...")
        optimized_path = self.storage_dir / f"{project_id}_optimized.tif"

        with rasterio.open(str(source_path)) as src:
            src_crs = src.crs
            src_epsg = src_crs.to_epsg() if src_crs else None

            # Determine target EPSG: caller specified, or source EPSG, or fallback to India UTM Zone 44N (32644)
            final_epsg = target_epsg or src_epsg or 32644
            target_crs_str = f"EPSG:{final_epsg}"

            # If src_crs is None (e.g. ASCII Grid .asc), coordinates are assumed to be in target UTM
            if src_crs is None:
                needs_reprojection = False
            else:
                needs_reprojection = (src_epsg != final_epsg)

            if needs_reprojection:
                emit(50, f"Reprojecting DTM from {src_crs} to metric {target_crs_str}...")
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs_str, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()
                kwargs.update({
                    "crs": target_crs_str,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "driver": "GTiff",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                })

                with rasterio.open(str(optimized_path), "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs_str,
                            resampling=Resampling.bilinear,
                        )
            else:
                emit(60, "Writing optimized tiled GeoTIFF...")
                kwargs = src.meta.copy()
                kwargs.update({
                    "crs": target_crs_str,
                    "driver": "GTiff",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                })
                with rasterio.open(str(optimized_path), "w", **kwargs) as dst:
                    for _, window in src.block_windows(1):
                        for i in range(1, src.count + 1):
                            dst.write(src.read(i, window=window), i, window=window)

        emit(80, "Trimming DTM strictly to available elevation data & computing statistics...")
        with rasterio.open(str(optimized_path)) as opt_src:
            band_raw = opt_src.read(1).astype(np.float32, copy=False)
            nodata_clean = float(opt_src.nodata) if opt_src.nodata is not None else -9999.0
            res_x, res_y = opt_src.res
            src_transform = opt_src.transform

            # Mask valid, available elevation data (exclude NoData, NaN, Inf)
            valid_mask = (band_raw != nodata_clean) & np.isfinite(band_raw)

            if not np.any(valid_mask):
                valid_data = np.array([0.0], dtype=np.float32)
                min_elev, max_elev = 0.0, 0.0
                band_valid = band_raw
                final_transform = src_transform
                width_val = opt_src.width
                height_val = opt_src.height
                b_left, b_bottom, b_right, b_top = opt_src.bounds.left, opt_src.bounds.bottom, opt_src.bounds.right, opt_src.bounds.top
            else:
                valid_data = band_raw[valid_mask]
                min_elev = float(np.min(valid_data))
                max_elev = float(np.max(valid_data))

                # Crop strictly to the minimal bounding box containing actual available data
                # Discards all empty outer NoData rows/columns, unmapped regions, and reprojection margins
                rows = np.any(valid_mask, axis=1)
                cols = np.any(valid_mask, axis=0)
                r_min, r_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
                c_min, c_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

                band_valid = band_raw[r_min : r_max + 1, c_min : c_max + 1]
                height_val, width_val = band_valid.shape
                final_transform = (src_transform @ Affine.translation(c_min, r_min)) if hasattr(src_transform, "__matmul__") else (src_transform * Affine.translation(c_min, r_min))
                b_left, b_bottom, b_right, b_top = rasterio.transform.array_bounds(
                    height_val, width_val, final_transform
                )

                trimmed_pixels = (opt_src.width * opt_src.height) - (width_val * height_val)
                if trimmed_pixels > 0:
                    logger.info(
                        "Trimmed DTM for '%s' to available data: %dx%d -> %dx%d (discarded %d unavailable/padding pixels)",
                        project_id,
                        opt_src.width,
                        opt_src.height,
                        width_val,
                        height_val,
                        trimmed_pixels,
                    )

            compressed_grid = zlib.compress(band_valid.tobytes())
            transform_vals = list(final_transform)[:6]
            del band_raw, valid_mask, valid_data, band_valid

        emit(90, f"Recording DTM into {'PostgreSQL/PostGIS' if self._use_postgres else 'SQLite'} registry...")

        record = DTMRecord(
            project_id=project_id,
            source_filename=Path(s3_key_or_path).name,
            crs_epsg=final_epsg,
            min_x=round(b_left, 3),
            min_y=round(b_bottom, 3),
            max_x=round(b_right, 3),
            max_y=round(b_top, 3),
            min_elevation_m=round(min_elev, 3),
            max_elevation_m=round(max_elev, 3),
            resolution_x_m=round(res_x, 3),
            resolution_y_m=round(res_y, 3),
            status="READY",
            created_at=time.time(),
        )

        if self._use_postgres:
            # Transform UTM bounding box to WGS84 EPSG:4326 for PostGIS geometry
            try:
                transformer = Transformer.from_crs(f"EPSG:{final_epsg}", "EPSG:4326", always_xy=True)
                min_lon, min_lat = transformer.transform(b_left, b_bottom)
                max_lon, max_lat = transformer.transform(b_right, b_top)
            except Exception:
                min_lon, min_lat, max_lon, max_lat = 0.0, 0.0, 0.0, 0.0

            with self._get_pg_conn() as conn:
                with conn.cursor() as cur:
                    # 1. Store metadata & PostGIS envelope
                    cur.execute(
                        """
                        INSERT INTO dtm_records (
                            project_id, source_filename, crs_epsg,
                            min_x, min_y, max_x, max_y, min_elevation_m, max_elevation_m,
                            resolution_x_m, resolution_y_m, status, created_at, geom
                        ) VALUES (
                            %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                        )
                        ON CONFLICT (project_id) DO UPDATE SET
                            source_filename = EXCLUDED.source_filename,
                            crs_epsg = EXCLUDED.crs_epsg,
                            min_x = EXCLUDED.min_x,
                            min_y = EXCLUDED.min_y,
                            max_x = EXCLUDED.max_x,
                            max_y = EXCLUDED.max_y,
                            min_elevation_m = EXCLUDED.min_elevation_m,
                            max_elevation_m = EXCLUDED.max_elevation_m,
                            resolution_x_m = EXCLUDED.resolution_x_m,
                            resolution_y_m = EXCLUDED.resolution_y_m,
                            status = EXCLUDED.status,
                            created_at = EXCLUDED.created_at,
                            geom = EXCLUDED.geom;
                        """,
                        (
                            record.project_id,
                            record.source_filename,
                            record.crs_epsg,
                            record.min_x,
                            record.min_y,
                            record.max_x,
                            record.max_y,
                            record.min_elevation_m,
                            record.max_elevation_m,
                            record.resolution_x_m,
                            record.resolution_y_m,
                            record.status,
                            record.created_at,
                            min_lon,
                            min_lat,
                            max_lon,
                            max_lat,
                        ),
                    )
                    # 2. Store compressed raster grid in PostgreSQL BYTEA
                    cur.execute(
                        """
                        INSERT INTO dtm_rasters (
                            project_id, width, height, nodata_val, transform_json, raster_blob, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (project_id) DO UPDATE SET
                            width = EXCLUDED.width,
                            height = EXCLUDED.height,
                            nodata_val = EXCLUDED.nodata_val,
                            transform_json = EXCLUDED.transform_json,
                            raster_blob = EXCLUDED.raster_blob,
                            updated_at = EXCLUDED.updated_at;
                        """,
                        (
                            project_id,
                            width_val,
                            height_val,
                            nodata_clean,
                            json.dumps(transform_vals),
                            psycopg2.Binary(compressed_grid),
                            time.time(),
                        ),
                    )
                conn.commit()
        else:
            # SQLite Fallback Storage
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dtm_records (
                        project_id, source_filename, crs_epsg,
                        min_x, min_y, max_x, max_y, min_elevation_m, max_elevation_m,
                        resolution_x_m, resolution_y_m, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.project_id,
                        record.source_filename,
                        record.crs_epsg,
                        record.min_x,
                        record.min_y,
                        record.max_x,
                        record.max_y,
                        record.min_elevation_m,
                        record.max_elevation_m,
                        record.resolution_x_m,
                        record.resolution_y_m,
                        record.status,
                        record.created_at,
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dtm_rasters (
                        project_id, width, height, nodata_val, transform_json, raster_blob
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        width_val,
                        height_val,
                        nodata_clean,
                        json.dumps(transform_vals),
                        compressed_grid,
                    ),
                )
                conn.commit()

        # Evict any stale cache entry for this project
        self._interpolator_cache.pop(project_id, None)

        emit(100, "DTM processing complete. Surface is ready for runtime queries.")
        logger.info(
            "DTM successfully ingested for project '%s' (Elevations: %.2fm to %.2fm, CRS: EPSG:%d)",
            project_id,
            min_elev,
            max_elev,
            final_epsg,
        )
        return asdict(record)

    def get_dtm_record(self, project_id: str) -> Optional[DTMRecord]:
        """Fetch the DTM registry record from PostgreSQL or SQLite for a given project."""
        if self._use_postgres:
            with self._get_pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT project_id, source_filename, crs_epsg,
                               min_x, min_y, max_x, max_y, min_elevation_m, max_elevation_m,
                               resolution_x_m, resolution_y_m, status, created_at,
                               ST_AsText(geom) AS geom_wkt
                        FROM dtm_records WHERE project_id = %s
                        """,
                        (project_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    return DTMRecord(*row)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT project_id, source_filename, crs_epsg,
                       min_x, min_y, max_x, max_y, min_elevation_m, max_elevation_m,
                       resolution_x_m, resolution_y_m, status, created_at
                FROM dtm_records WHERE project_id = ?
                """,
                (project_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return DTMRecord(*row)

    def list_all_dtm_records(self) -> List[Dict[str, Any]]:
        """List all ingested DTM datasets stored in the registry."""
        if self._use_postgres:
            with self._get_pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT project_id, source_filename, crs_epsg,
                               min_x, min_y, max_x, max_y, min_elevation_m, max_elevation_m,
                               resolution_x_m, resolution_y_m, status, created_at,
                               ST_AsText(geom) AS geom_wkt
                        FROM dtm_records ORDER BY created_at DESC
                        """
                    )
                    rows = cur.fetchall()
                    return [asdict(DTMRecord(*r)) for r in rows]

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT project_id, source_filename, crs_epsg,
                       min_x, min_y, max_x, max_y, min_elevation_m, max_elevation_m,
                       resolution_x_m, resolution_y_m, status, created_at
                FROM dtm_records ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()
            return [asdict(DTMRecord(*r)) for r in rows]

    def delete_dtm_record(self, project_id: str) -> bool:
        """Deletes a DTM record and its stored raster from the database and cache."""
        self._interpolator_cache.pop(project_id, None)
        if self._use_postgres:
            with self._get_pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM dtm_rasters WHERE project_id = %s", (project_id,))
                    cur.execute("DELETE FROM dtm_records WHERE project_id = %s", (project_id,))
                conn.commit()
            return True

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM dtm_rasters WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM dtm_records WHERE project_id = ?", (project_id,))
            conn.commit()
        return True

    def get_raster_grid(self, project_id: str) -> Tuple[np.ndarray, Affine, float, int]:
        """
        Retrieves raw 2D elevation matrix and georeferencing transform directly from the database.
        Returns:
            (elevation_array: np.ndarray, transform: Affine, nodata: float, crs_epsg: int)
        """
        record = self.get_dtm_record(project_id)
        if not record or record.status != "READY":
            raise ValueError(f"DTM for project '{project_id}' is not ready or does not exist in the database.")

        raster_row = None
        if self._use_postgres:
            with self._get_pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT width, height, nodata_val, transform_json, raster_blob FROM dtm_rasters WHERE project_id = %s",
                        (project_id,),
                    )
                    raster_row = cur.fetchone()
        else:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT width, height, nodata_val, transform_json, raster_blob FROM dtm_rasters WHERE project_id = ?",
                    (project_id,),
                )
                raster_row = cursor.fetchone()

        if raster_row:
            width, height, nodata_val, transform_json, raster_blob = raster_row
            nodata = float(nodata_val) if nodata_val is not None else -9999.0
            raw_blob = bytes(raster_blob)
            decompressed = zlib.decompress(raw_blob)
            data = np.frombuffer(decompressed, dtype=np.float32).reshape((height, width))
            t_vals = json.loads(transform_json)
            transform = Affine(*t_vals)
            return data, transform, nodata, record.crs_epsg

        # Fallback to local GeoTIFF file if present
        local_opt = self.storage_dir / f"{project_id}_optimized.tif"
        if local_opt.exists():
            with rasterio.open(str(local_opt)) as src:
                data = src.read(1).astype(np.float32)
                nodata = float(src.nodata) if src.nodata is not None else -9999.0
                transform = src.transform
                return data, transform, nodata, record.crs_epsg

        raise FileNotFoundError(f"DTM raster missing from database for project '{project_id}'")

    def export_to_geotiff(
        self, project_id: str, output_path: Optional[Path | str] = None
    ) -> Path:
        """
        Reconstruct and export the DTM from the PostgreSQL database back into a fully georeferenced GeoTIFF (.tif).
        This allows external GIS tools (QGIS, Civil 3D) and downstream CAD/engineering pipelines to reuse the stored data.
        """
        record = self.get_dtm_record(project_id)
        if not record:
            raise ValueError(f"DTM record not found in database for project '{project_id}'")

        data, transform, nodata, crs_epsg = self.get_raster_grid(project_id)
        height, width = data.shape

        target_path = Path(output_path) if output_path else self.storage_dir / f"{project_id}_exported.tif"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": f"EPSG:{crs_epsg}",
            "transform": transform,
            "nodata": nodata,
        }
        if width >= 256 and height >= 256:
            profile.update({
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            })

        with rasterio.open(str(target_path), "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)

        logger.info(
            "Exported stored DTM for project '%s' (%dx%d, EPSG:%d) to %s",
            project_id,
            width,
            height,
            crs_epsg,
            target_path,
        )
        return target_path

    def get_dem_result(self, project_id: str, target_crs: Optional[str] = None) -> Any:
        """
        Constructs a standard DEMResult directly from the database-stored DTM.
        Enables instant zero-disk integration with Module 4 (ElevationSampler), Module 5 (LongitudinalProfile),
        and Module 9D (CompositeDrawer) without needing local raster files.
        """
        record = self.get_dtm_record(project_id)
        if not record:
            raise ValueError(f"DTM record not found in database for project '{project_id}'")

        data, transform, nodata, crs_epsg = self.get_raster_grid(project_id)
        height, width = data.shape

        xs = np.array([rasterio.transform.xy(transform, 0, c)[0] for c in range(width)], dtype=np.float64)
        ys = np.array([rasterio.transform.xy(transform, r, 0)[1] for r in range(height)], dtype=np.float64)

        # Invert Y if stored north-to-south
        if len(ys) > 1 and ys[0] > ys[-1]:
            ys = ys[::-1]
            data = data[::-1, :]

        # Guarantee a valid local GeoTIFF path for any components requiring a file path
        geotiff_path = self.storage_dir / f"{project_id}_dtm.tif"
        if not geotiff_path.exists():
            geotiff_path = self.export_to_geotiff(project_id, output_path=geotiff_path)

        try:
            from modules.dem_reader import DEMResult
        except ImportError:
            from plan_profile_generator.modules.dem_reader import DEMResult

        return DEMResult(
            dataset_path=geotiff_path,
            crs=f"EPSG:{record.crs_epsg}",
            nodata=nodata,
            resolution=(record.resolution_x_m, record.resolution_y_m),
            bounds=(record.min_x, record.min_y, record.max_x, record.max_y),
            xs=xs,
            ys=ys,
            data=data,
            metadata={
                "project_id": project_id,
                "source_filename": record.source_filename,
                "backend": "PostgreSQL esp_db" if self._use_postgres else "SQLite",
                "min_elevation_m": record.min_elevation_m,
                "max_elevation_m": record.max_elevation_m,
            },
        )

    def find_matching_dtms(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> List[Dict[str, Any]]:
        """
        Find stored DTM datasets overlapping the given WGS84 geographic bounding box.
        Uses PostGIS spatial index (GIST) for instant spatial query execution.
        """
        if self._use_postgres:
            with self._get_pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT project_id, source_filename, crs_epsg,
                               min_x, min_y, max_x, max_y, min_elevation_m, max_elevation_m,
                               resolution_x_m, resolution_y_m, status, created_at,
                               ST_AsText(geom) AS geom_wkt
                        FROM dtm_records
                        WHERE geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                          AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
                        ORDER BY created_at DESC;
                        """,
                        (min_lon, min_lat, max_lon, max_lat, min_lon, min_lat, max_lon, max_lat),
                    )
                    rows = cur.fetchall()
                    return [asdict(DTMRecord(*r)) for r in rows]

        # SQLite bounding box fallback
        all_records = self.list_all_dtm_records()
        matches: List[Dict[str, Any]] = []
        for rec in all_records:
            try:
                inv_trans = Transformer.from_crs(f"EPSG:{rec['crs_epsg']}", "EPSG:4326", always_xy=True)
                r_min_lon, r_min_lat = inv_trans.transform(rec["min_x"], rec["min_y"])
                r_max_lon, r_max_lat = inv_trans.transform(rec["max_x"], rec["max_y"])
                if not (r_max_lon < min_lon or r_min_lon > max_lon or r_max_lat < min_lat or r_min_lat > max_lat):
                    matches.append(rec)
            except Exception:
                matches.append(rec)
        return matches

    def _get_interpolator(
        self, project_id: str
    ) -> Tuple[RegularGridInterpolator, tuple, float]:
        """
        Builds or retrieves a cached scipy RegularGridInterpolator for sub-pixel bilinear sampling.
        Caches in RAM for sub-30ms execution during alignment drag events.
        """
        if project_id in self._interpolator_cache:
            return self._interpolator_cache[project_id]

        record = self.get_dtm_record(project_id)
        if not record or record.status != "READY":
            raise ValueError(f"DTM for project '{project_id}' is not ready or does not exist.")

        data, transform, nodata, _ = self.get_raster_grid(project_id)
        height, width = data.shape

        # Coordinates for rows (Y) and columns (X) using rasterio.transform.xy
        xs = np.array([rasterio.transform.xy(transform, 0, c)[0] for c in range(width)], dtype=np.float64)
        ys = np.array([rasterio.transform.xy(transform, r, 0)[1] for r in range(height)], dtype=np.float64)

        # Invert Y if raster is stored north-to-south (standard descending)
        if len(ys) > 1 and ys[0] > ys[-1]:
            ys = ys[::-1]
            data = data[::-1, :]

        # Clean nodata values for interpolation
        clean_data = np.where(data == nodata, np.nan, data)

        interp = RegularGridInterpolator(
            (ys, xs),
            clean_data,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        bounds = (record.min_x, record.min_y, record.max_x, record.max_y)
        cached_val = (interp, bounds, float(nodata))
        self._interpolator_cache[project_id] = cached_val
        return cached_val

    def sample_stations(
        self, project_id: str, stations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fast runtime user query feature:
        Takes a list of railway stations (chainage_m, easting_m, northing_m),
        performs bilinear sub-pixel interpolation against the stored DTM,
        and computes the recommended profile datum level.
        Execution speed: < 30ms for typical 5,000 station corridors.
        """
        t0 = time.time()
        if len(stations) > self.max_sampling_points:
            raise ValueError(
                f"Station count ({len(stations)}) exceeds configured maximum limit of {self.max_sampling_points}"
            )

        interpolator, bounds, nodata_val = self._get_interpolator(project_id)
        min_x, min_y, max_x, max_y = bounds

        samples: List[Dict[str, Any]] = []
        valid_elevations: List[float] = []

        # Batch query for speed
        coords = np.array([[s["northing_m"], s["easting_m"]] for s in stations])
        elevations = interpolator(coords)

        for idx, s in enumerate(stations):
            ch = float(s["chainage_m"])
            east = float(s["easting_m"])
            north = float(s["northing_m"])
            elev = float(elevations[idx])

            if math.isnan(elev) or elev == nodata_val:
                quality = "outside_extent" if not (min_x <= east <= max_x and min_y <= north <= max_y) else "nodata"
                elev_out = None
            else:
                quality = "ok"
                elev_out = round(elev, 3)
                valid_elevations.append(elev_out)

            samples.append({
                "chainage_m": ch,
                "easting_m": east,
                "northing_m": north,
                "egl_m": elev_out,
                "quality": quality,
            })

        min_egl = min(valid_elevations) if valid_elevations else 0.0
        max_egl = max(valid_elevations) if valid_elevations else 0.0

        # RDSO Recommended Datum: floor(min_egl - datum_margin) rounded down to datum_round_m
        suggested_datum = (
            math.floor((min_egl - self.datum_margin_m) / self.datum_round_m) * self.datum_round_m
            if valid_elevations
            else 0.0
        )

        query_time_ms = round((time.time() - t0) * 1000.0, 2)
        logger.info(
            "Sampled %d stations for project '%s' in %.2fms (EGL: %.2fm to %.2fm, Datum: %.1fm)",
            len(stations),
            project_id,
            query_time_ms,
            min_egl,
            max_egl,
            suggested_datum,
        )

        return {
            "project_id": project_id,
            "query_time_ms": query_time_ms,
            "station_count": len(stations),
            "min_egl_m": round(min_egl, 3),
            "max_egl_m": round(max_egl, 3),
            "suggested_datum_m": round(suggested_datum, 1),
            "samples": samples,
        }

    def upload_and_ingest(
        self,
        project_id: str,
        filename: str,
        file_bytes: bytes,
        target_epsg: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous direct upload & ingest:
        Saves file bytes, reprojects/optimizes, and stores metadata + compressed elevation grid
        directly in the PostgreSQL database.
        Returns complete DTMRecord dictionary.
        """
        clean_name = Path(filename).name
        raw_path = self.save_direct_upload(project_id, clean_name, file_bytes)
        job_id = f"job_upload_{uuid.uuid4().hex[:8]}"
        return self.process_dtm_job(
            job_id=job_id,
            project_id=project_id,
            s3_key_or_path=str(raw_path),
            target_epsg=target_epsg,
        )

    def sample_surface_query(
        self,
        project_id: str,
        stations: Optional[List[Dict[str, Any]]] = None,
        coordinates: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Query surface profile on the basis of KML point changes:
        - If `stations` are provided (with chainage_m, easting_m, northing_m in metric UTM), samples them.
        - If `coordinates` are provided (raw [[lon, lat], ...] from KML polyline edits),
          automatically projects WGS84 coordinates into the DTM's UTM projection, computes
          chainage along the path, and samples the ground elevations.
        """
        if coordinates is not None and len(coordinates) > 0:
            record = self.get_dtm_record(project_id)
            if not record:
                raise ValueError(f"DTM for project '{project_id}' is not ready or does not exist.")

            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{record.crs_epsg}", always_xy=True)
            station_list: List[Dict[str, Any]] = []
            cum_dist = 0.0
            prev_pt = None

            for idx, pt in enumerate(coordinates):
                lon, lat = float(pt[0]), float(pt[1])
                east, north = transformer.transform(lon, lat)
                if prev_pt is not None:
                    cum_dist += math.hypot(east - prev_pt[0], north - prev_pt[1])
                prev_pt = (east, north)
                station_list.append({
                    "station_index": idx,
                    "chainage_m": round(cum_dist, 2),
                    "easting_m": round(east, 3),
                    "northing_m": round(north, 3),
                    "lon_deg": round(lon, 7),
                    "lat_deg": round(lat, 7),
                })

            res = self.sample_stations(project_id, station_list)
            res["query_mode"] = "kml_coordinates"
            return res

        if stations is not None and len(stations) > 0:
            res = self.sample_stations(project_id, stations)
            res["query_mode"] = "stations"
            return res

        raise ValueError("Either 'stations' or 'coordinates' must be provided in the query payload.")
