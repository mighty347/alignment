"""
Pydantic Schemas and Enums for Railway Alignment Service.
Standards: RDSO / Indian Railways BG
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryStation(BaseModel):
    chainage_m: float
    easting_m: float
    northing_m: float


class DTMQueryRequest(BaseModel):
    project_id: str = Field(..., description="Project ID matching the stored DTM raster/grid")
    stations: Optional[List[QueryStation]] = Field(None, description="List of stations with chainage and UTM Easting/Northing")
    coordinates: Optional[List[List[float]]] = Field(None, description="Raw [[lon, lat], ...] from KML line vertices")

    @model_validator(mode="after")
    def check_inputs(self) -> DTMQueryRequest:
        if not self.stations and not self.coordinates:
            raise ValueError("Must provide either 'stations' or 'coordinates' to query DTM.")
        return self


class GenerateCenterlinePathRequest(BaseModel):
    file_path: str = Field(..., description="Server path to local survey file (.csv, .xlsx, .xls)")
    start_chainage_m: float = Field(0.0, description="Start chainage in meters")
    chainage_interval_m: float = Field(20.0, description="Chainage interval in meters (default 20.0m)")
    epsg: Optional[int] = Field(None, description="UTM EPSG code for coordinate projection")
    reverse_direction: bool = Field(False, description="Reverse centerline stationing direction")
    track_index: int = Field(0, description="Track column index for multi-track survey sheets")


class PlanProfileResponse(BaseModel):
    dxf_files_path: List[str] = Field(default_factory=list)
    pdf_files_path: Optional[List[str]] = None
    detected_files: Optional[dict[str, Any]] = None
    package_name: Optional[str] = None
    sheets: Optional[List[dict[str, Any]]] = None


class JobRecord:
    """In-memory tracking record for async alignment detection jobs."""

    def __init__(self, job_id: str, user_id: str = "default_user", project_name: str = "Horizontal Alignment Detection"):
        self.job_id = job_id
        self.user_id = user_id
        self.project_name = project_name
        self.status = JobStatus.QUEUED
        self.progress_percent = 0
        self.current_step = "Queued in processing pool"
        self.queue_position = 1
        self.detected_files: dict[str, Any] = {}
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.subscribers: List[asyncio.Queue] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "project_name": self.project_name,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "queue_position": self.queue_position,
            "detected_files": self.detected_files,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
