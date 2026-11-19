"""
Module: alignment/main.py
Purpose: Unified FastAPI Microservice for Railway Alignment Studio, Fast DTM
         Elevation Query, Survey Centerline Extraction, and DB-Driven Greenfield
         Composite Plan & L-Section Generation.
Standards: RDSO / Indian Railways BG
Port: 8001 (configurable via PORT environment variable)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure alignment directory is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

from utils.logger import get_logger
from alignment_router import router as alignment_router
from greenfield.greenfield_router import (
    OUTPUTS_DIR as GREENFIELD_OUTPUTS_DIR,
    router as greenfield_router,
)

logger = get_logger("AlignmentService")

app = FastAPI(
    title="Railway Alignment, DTM & Greenfield Plan/Profile Service",
    description=(
        "Unified Microservice for Horizontal Alignment Detection, Real-Time DTM "
        "Elevation Queries, Survey Centerline Extraction, and DB-Driven Greenfield "
        "Composite Plan & L-Section Generation."
    ),
    version="1.0.0",
    docs_url=None,  # Custom relative handler below
    redoc_url=None,  # Custom relative handler below
)

@app.get("/docs", include_in_schema=False)
async def swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="openapi.json",
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url="openapi.json",
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

# Configure CORS
_cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "ESP_CORS_ORIGINS",
        "http://localhost:4200,http://127.0.0.1:4200,https://bridge-gad-dev.aagento.ai,https://gad-engine.web.app",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins + ["*"],
    allow_origin_regex=r"^https?://.*$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount Greenfield generated outputs (DXF/PDF downloads)
GREENFIELD_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=GREENFIELD_OUTPUTS_DIR), name="outputs")

# Register Alignment, DTM & Greenfield Composite Routes
app.include_router(alignment_router)
app.include_router(greenfield_router)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=== Railway Alignment & Greenfield Service Started (Port %s) ===", os.environ.get("PORT", "8001"))
    logger.info("Greenfield outputs directory: %s", GREENFIELD_OUTPUTS_DIR)
    logger.info("Total registered routes: %d", len(app.routes))

    # Optional auto-start of L-Section Kafka Consumer thread
    if os.environ.get("ENABLE_KAFKA_CONSUMER", "true").lower() in ("true", "1", "yes"):
        try:
            from services.lsection_kafka_service import LSectionKafkaService
            LSectionKafkaService.get_instance().start()
            logger.info("L-Section Kafka Consumer background service started.")
        except Exception as e:
            logger.warning("Could not start L-Section Kafka Consumer on startup: %s", e)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    try:
        from services.lsection_kafka_service import LSectionKafkaService
        LSectionKafkaService.get_instance().stop()
        logger.info("L-Section Kafka Consumer stopped cleanly.")
    except Exception as e:
        logger.warning("Error stopping L-Section Kafka Consumer: %s", e)


@app.get("/")
def root():
    return {
        "status": "ONLINE",
        "service": "Railway Alignment, DTM & Greenfield Composite Service",
        "version": "1.0.0",
        "endpoints": [
            "POST /detect",
            "POST /api/v1/alignment-studio/detect",
            "GET  /api/v1/greenfield/composite/preview/{version_id}",
            "POST /api/v1/greenfield/composite/generate",
            "GET  /api/v1/greenfield/composite/jobs/{job_id}",
            "GET  /api/v1/greenfield/composite/jobs/{job_id}/stream",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "HEALTHY"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
