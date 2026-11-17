"""
Module: alignment/main.py
Purpose: Standalone FastAPI Microservice for Railway Alignment Studio & Fast DTM Elevation Query.
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

from utils.logger import get_logger
from alignment_router import router as alignment_router

logger = get_logger("AlignmentService")

app = FastAPI(
    title="Railway Alignment & DTM Query Service",
    description="Standalone Microservice for Horizontal Alignment Detection, Real-Time DTM Elevation Queries, and Survey Centerline Extraction.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

# Register Alignment & DTM Routes
app.include_router(alignment_router)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=== Railway Alignment & DTM Service Started (Port %s) ===", os.environ.get("PORT", "8001"))
    logger.info("Total registered endpoints: %d", len(app.routes))


@app.get("/")
def root():
    return {
        "status": "ONLINE",
        "service": "Railway Alignment & DTM Service",
        "version": "1.0.0",
        "endpoints": [
            "POST /api/v1/alignment-studio/detect",
            "GET  /api/v1/jobs/{job_id}/status",
            "GET  /api/v1/jobs/{job_id}/stream",
            "POST /api/v1/alignment-studio/dtm/query",
            "POST /api/v1/alignment-studio/survey/generate-centerline",
            "POST /api/v1/alignment-studio/survey/generate-centerline-path",
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

