"""
Module: services/lsection_kafka_service.py
Purpose: Kafka Consumer & DXF Export Service for Longitudinal Section (L-Section).
Listens to 'lsection-dxf-request', queries PostgreSQL, generates per-sheet DXF
drawings using LSectionDrawer, uploads to S3/MinIO (or local storage fallback),
and publishes results to 'lsection-dxf-response'.

Standards: RDSO GE:G-1 / IRPWM / Indian Railways SOD (BG)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Auto-configure GDAL/PROJ
if "GDAL_DATA" not in os.environ:
    gdal_cand = Path(sys.prefix) / "Library" / "share" / "gdal"
    if gdal_cand.exists():
        os.environ["GDAL_DATA"] = str(gdal_cand)

try:
    from confluent_kafka import Consumer, Producer, KafkaError
    HAS_KAFKA = True
except ImportError:
    Consumer = None  # type: ignore[assignment, misc]
    Producer = None  # type: ignore[assignment, misc]
    KafkaError = None  # type: ignore[assignment, misc]
    HAS_KAFKA = False

import boto3
from botocore.exceptions import ClientError

from modules.lsection_drawer import LSectionDrawer
from utils.logger import get_logger

logger = get_logger("LSectionKafkaService")


def _load_dotenv() -> None:
    """Load key-value pairs from .env file into os.environ if present."""
    base_dir = Path(__file__).resolve().parent.parent
    cand = base_dir / ".env"
    if not cand.exists():
        cand = Path.cwd() / ".env"
    if cand.exists():
        try:
            with open(cand, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"\'')
                    if k and v:
                        os.environ[k] = v
        except Exception:
            pass


_load_dotenv()


class LSectionKafkaService:
    """Kafka Background Service for processing L-Section DXF Generation."""

    _instance: Optional[LSectionKafkaService] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        # Kafka configuration
        self.bootstrap_servers = (
            os.getenv("KAFKA_BOOTSTRAP_SERVERS")
            or os.getenv("KAFKA_BROKER")
            or "kafka:14004"
        )
        self.sasl_username = os.getenv("KAFKA_SASL_USERNAME", "admin")
        self.sasl_password = os.getenv("KAFKA_SASL_PASSWORD", "admin-secret")
        self.sasl_mechanism = os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
        self.security_protocol = os.getenv(
            "KAFKA_SECURITY_PROTOCOL",
            "SASL_PLAINTEXT" if self.sasl_username else "PLAINTEXT",
        )

        self.request_topic = os.getenv("KAFKA_LSECTION_TOPIC_REQUEST", "lsection-dxf-request")
        self.response_topic = os.getenv("KAFKA_LSECTION_TOPIC_RESPONSE", "lsection-dxf-response")
        self.group_id = os.getenv("KAFKA_LSECTION_GROUP_ID", "lsection-service-group")

        # Helper to clean and strip env vars
        def _clean_str(val: Optional[str]) -> Optional[str]:
            if not val:
                return None
            cleaned = str(val).strip().strip('"\'').strip()
            return cleaned if cleaned else None

        # Database configuration
        raw_db_url = os.getenv("DATABASE_URL")
        if not raw_db_url:
            raise RuntimeError("DATABASE_URL environment variable is required.")
        # Strip ?schema=... to avoid psycopg2 syntax error
        self.db_url = re.sub(r"\?.*$", "", raw_db_url)

        # S3 / Cloud Storage configuration (Strictly from environment, no hardcoded fallbacks)
        self.s3_bucket = (
            _clean_str(os.getenv("AWS_BUCKET_NAME"))
            or _clean_str(os.getenv("AWS_S3_BUCKET_NAME"))
        )
        self.aws_region = _clean_str(os.getenv("AWS_REGION")) or "ap-south-1"
        self.aws_access_key = _clean_str(os.getenv("AWS_ACCESS_KEY_ID"))
        self.aws_secret_key = _clean_str(os.getenv("AWS_SECRET_ACCESS_KEY"))
        self.s3_url_base = _clean_str(os.getenv("AWS_S3_URL"))

        # Output directory
        base_dir = Path(__file__).resolve().parent.parent
        self.outputs_dir = base_dir / "greenfield" / "outputs" / "lsection_dxf"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.producer: Optional[Producer] = None
        self.consumer: Optional[Consumer] = None
        self.consumer_thread: Optional[threading.Thread] = None
        self.running = False

        logger.info(
            "Initialized LSectionKafkaService: broker=%s, req_topic=%s, res_topic=%s, group=%s",
            self.bootstrap_servers,
            self.request_topic,
            self.response_topic,
            self.group_id,
        )

    @classmethod
    def get_instance(cls) -> LSectionKafkaService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _get_common_kafka_config(self) -> dict[str, Any]:
        conf: dict[str, Any] = {
            "bootstrap.servers": self.bootstrap_servers,
        }
        if self.sasl_username and self.sasl_password:
            conf.update({
                "security.protocol": self.security_protocol,
                "sasl.mechanisms": self.sasl_mechanism,
                "sasl.username": self.sasl_username,
                "sasl.password": self.sasl_password,
            })
        return conf

    def start(self) -> None:
        """Start Kafka consumer thread."""
        if not HAS_KAFKA:
            logger.error("confluent-kafka is not installed. Kafka consumer cannot start.")
            return

        if self.running:
            logger.info("LSectionKafkaService is already running.")
            return

        # Initialize Producer
        try:
            prod_conf = self._get_common_kafka_config()
            self.producer = Producer(prod_conf)
            logger.info("L-Section Kafka Producer initialized.")
        except Exception as e:
            logger.error("Failed to initialize Kafka Producer: %s", e, exc_info=True)

        self.running = True
        self.consumer_thread = threading.Thread(target=self._run_consumer, daemon=True)
        self.consumer_thread.start()
        logger.info("L-Section Kafka Consumer thread started.")

    def stop(self) -> None:
        """Stop Kafka consumer and flush producer."""
        logger.info("Stopping LSectionKafkaService...")
        self.running = False
        if self.consumer:
            try:
                self.consumer.close()
            except Exception as e:
                logger.error("Error closing Kafka consumer: %s", e)
        if self.producer:
            try:
                self.producer.flush(timeout=5.0)
            except Exception as e:
                logger.error("Error flushing Kafka producer: %s", e)
        logger.info("LSectionKafkaService stopped.")

    def _run_consumer(self) -> None:
        """Consumer polling loop."""
        conf = self._get_common_kafka_config()
        conf.update({
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        })

        try:
            self.consumer = Consumer(conf)
            self.consumer.subscribe([self.request_topic])
            logger.info("Subscribed to request topic: %s", self.request_topic)
        except Exception as e:
            logger.error("Failed to subscribe Kafka Consumer: %s", e, exc_info=True)
            self.running = False
            return

        while self.running:
            try:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("Kafka consumer error: %s", msg.error())
                    continue

                raw_val = msg.value().decode("utf-8").strip()
                logger.info("Received L-Section DXF generation request: %s", raw_val)

                try:
                    payload = json.loads(raw_val)
                    self.process_request(payload)
                except Exception as ex:
                    logger.exception("Failed to process L-Section request: %s", ex)
            except Exception as loop_ex:
                logger.error("Error in consumer loop: %s", loop_ex, exc_info=True)
                time.sleep(1.0)

    # -------------------------------------------------------------------------
    # Core Processing Logic
    # -------------------------------------------------------------------------

    def _get_db_connection(self):
        return psycopg2.connect(self.db_url, connect_timeout=5)

    def fetch_alignment_data(self, version_id: str, corridor_id: str) -> dict[str, Any]:
        """Fetch all required entities from PostgreSQL."""
        with self._get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 1. Sheets
                cur.execute("""
                    SELECT id, "sheetNumber", "startChainage", "endChainage", "datumRl", "hScale", "vScale", "dxfUrl"
                    FROM public.lsection_sheets
                    WHERE "versionId" = %s AND "auditStatus" = 'ACTIVE'
                    ORDER BY "sheetNumber" ASC;
                """, (version_id,))
                sheets = cur.fetchall()

                # 2. Vertical Alignment / Gradients
                cur.execute("""
                    SELECT "segNo", "startChainage", "endChainage", "lengthM", "slopeRatio", direction,
                           "startRl", "endRl", "vpiChainage", "vpiElevation", "vcLength"
                    FROM public.lsection_gradients
                    WHERE "versionId" = %s AND "auditStatus" = 'ACTIVE'
                    ORDER BY "segNo" ASC;
                """, (version_id,))
                gradients = cur.fetchall()

                # 3. Horizontal Alignment / Curves
                cur.execute("""
                    SELECT "curveNo", "trackType", "trackName", "startChainage", "endChainage",
                           "lengthM", "radiusM", degree, "deflectionDeg", "cantMm", "transitionLengthM", direction
                    FROM public.lsection_curves
                    WHERE "versionId" = %s AND "auditStatus" = 'ACTIVE'
                    ORDER BY "startChainage" ASC;
                """, (version_id,))
                curves = cur.fetchall()

                # 4. Survey Band Points (Ground & Formation Levels)
                cur.execute("""
                    SELECT chainage, "bandValues"
                    FROM public.lsection_band_points
                    WHERE "versionId" = %s AND "auditStatus" = 'ACTIVE'
                    ORDER BY chainage ASC;
                """, (version_id,))
                band_points = cur.fetchall()

                # 5. Stations
                cur.execute("""
                    SELECT name, "chainageM", "locationKm", "stationClass"
                    FROM public.lsection_stations
                    WHERE "lsectionId" = %s AND "auditStatus" = 'ACTIVE'
                    ORDER BY "chainageM" ASC;
                """, (corridor_id,))
                stations = cur.fetchall()

        return {
            "sheets": sheets,
            "gradients": gradients,
            "curves": curves,
            "band_points": band_points,
            "stations": stations,
        }

    def parse_band_points(self, band_points: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse JSONB bandValues into structured DataFrame."""
        records = []
        for bp in band_points:
            ch = float(bp["chainage"])
            vals = bp.get("bandValues") or {}
            if isinstance(vals, str):
                try:
                    vals = json.loads(vals)
                except Exception:
                    vals = {}

            # Normalized lookup
            norm_vals = {str(k).lower().replace("_", ""): v for k, v in vals.items()}

            egl = (
                vals.get("groundLevel")
                or vals.get("elevation_m")
                or vals.get("egl")
                or norm_vals.get("groundlevel")
                or norm_vals.get("elevationm")
                or norm_vals.get("existinggroundlevel")
                or 0.0
            )
            pfl = (
                vals.get("proposedFormation")
                or vals.get("formation_level_m")
                or vals.get("pfl")
                or norm_vals.get("proposedformation")
                or norm_vals.get("formationlevelm")
                or norm_vals.get("formationlevel")
                or egl
            )
            prl = (
                vals.get("proposedRail")
                or vals.get("proposed_rail_level_m")
                or vals.get("prl")
                or norm_vals.get("proposedrail")
                or (float(pfl) + 0.762)
            )
            erl = (
                vals.get("existingRailLevel")
                or vals.get("existing_rail_level_m")
                or norm_vals.get("existingraillevel")
                or np.nan
            )
            efl = (
                vals.get("existingFormationLevel")
                or vals.get("existing_formation_level_m")
                or norm_vals.get("existingformationlevel")
                or np.nan
            )

            records.append({
                "chainage_m": ch,
                "elevation_m": float(egl),
                "formation_level_m": float(pfl),
                "proposed_rail_level_m": float(prl),
                "existing_rail_level_m": float(erl) if pd.notna(erl) else np.nan,
                "existing_formation_level_m": float(efl) if pd.notna(efl) else np.nan,
            })

        if not records:
            return pd.DataFrame(columns=["chainage_m", "elevation_m", "formation_level_m"])

        df = pd.DataFrame(records).sort_values("chainage_m").reset_index(drop=True)
        return df

    def upload_dxf(self, local_path: Path, version_id: str, sheet_num: int) -> str:
        """Upload DXF directly to AWS S3. No credentials in logs, no fallback."""
        if not self.s3_bucket:
            raise RuntimeError("AWS_BUCKET_NAME is not configured in environment.")

        s3_key = f"lsection/{version_id}/sheet_{sheet_num}.dxf"
        logger.info("Uploading %s to s3://%s/%s...", local_path.name, self.s3_bucket, s3_key)

        s3_kwargs: dict[str, Any] = {"region_name": self.aws_region}
        if self.aws_access_key and self.aws_secret_key:
            s3_kwargs["aws_access_key_id"] = self.aws_access_key
            s3_kwargs["aws_secret_access_key"] = self.aws_secret_key

        s3_client = boto3.client("s3", **s3_kwargs)
        s3_client.upload_file(
            str(local_path),
            self.s3_bucket,
            s3_key,
            ExtraArgs={"ContentType": "application/dxf"},
        )

        if self.s3_url_base:
            s3_url = f"{self.s3_url_base.rstrip('/')}/{s3_key}"
        else:
            s3_url = f"https://{self.s3_bucket}.s3.{self.aws_region}.amazonaws.com/{s3_key}"

        logger.info("S3 upload successful: %s", s3_url)
        return s3_url

    def process_request(self, payload: dict[str, Any]) -> None:
        """Process incoming request: Query DB -> Render Complete Composite DXF -> Upload -> Emit Response."""
        version_id = (
            payload.get("versionId")
            or payload.get("siteVersionId")
            or payload.get("version_id")
        )
        corridor_id = (
            payload.get("corridorId")
            or payload.get("lsectionId")
            or payload.get("corridor_id")
        )

        if not version_id:
            logger.error("Missing versionId in payload: %s", payload)
            return

        logger.info(
            "=== Starting Composite Plan & Profile DXF Generation via Channel B for versionId=%s, corridorId=%s ===",
            version_id,
            corridor_id,
        )

        try:
            from greenfield.greenfield_models import GreenfieldCompositeGenerateRequest
            from greenfield.greenfield_router import _execute_greenfield_composite_job

            req = GreenfieldCompositeGenerateRequest(
                version_id=str(version_id),
                s3_output_bucket=self.s3_bucket,
                generate_sheet_drawings=True,
                export_pdf=False,
            )
            job_id = f"kafka_{uuid.uuid4().hex[:12]}"

            logger.info("Running Composite Plan & Profile engine for Job %s...", job_id)
            sheet_items = _execute_greenfield_composite_job(req, job_id, request=None)

            generated_sheets: list[dict[str, Any]] = []
            for item in sheet_items:
                final_dxf_url = item.dxf_url
                if not final_dxf_url and item.local_dxf_path and self.s3_bucket:
                    try:
                        final_dxf_url = self.upload_dxf(Path(item.local_dxf_path), version_id, item.sheet_number)
                    except Exception as up_err:
                        logger.warning("Could not upload sheet %d to S3: %s", item.sheet_number, up_err)

                generated_sheets.append({
                    "sheetNumber": item.sheet_number,
                    "sheetId": item.sheet_id,
                    "startChainage": item.start_chainage_m,
                    "endChainage": item.end_chainage_m,
                    "datumRl": item.datum_rl_m,
                    "dxfUrl": final_dxf_url or item.local_dxf_path,
                })

            response_payload = {
                "versionId": version_id,
                "corridorId": corridor_id,
                "sheets": generated_sheets,
                "status": "SUCCESS",
            }
        except Exception as ex:
            logger.exception("Failed to generate composite DXF for versionId=%s: %s", version_id, ex)
            response_payload = {
                "versionId": version_id,
                "corridorId": corridor_id,
                "status": "FAILED",
                "error": str(ex),
                "sheets": [],
            }

        def _delivery_report(err, msg):
            if err is not None:
                logger.error("Message delivery failed: %s", err)
            else:
                logger.info("Message delivered to %s [%s]", msg.topic(), msg.partition())

        if self.producer:
            self.producer.produce(
                self.response_topic,
                key=str(version_id).encode("utf-8"),
                value=json.dumps(response_payload).encode("utf-8"),
                callback=_delivery_report,
            )
            self.producer.flush(timeout=5.0)
            logger.info("Published response to topic '%s': %s", self.response_topic, response_payload)
        else:
            logger.warning("Producer not initialized. Response not emitted to Kafka.")


# Standalone runner
if __name__ == "__main__":
    service = LSectionKafkaService.get_instance()
    service.start()
    logger.info("L-Section Kafka Service started in standalone mode. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        service.stop()
        logger.info("L-Section Kafka Service stopped.")
