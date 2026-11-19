"""
Module: greenfield/greenfield_models.py
Purpose: Clean Pydantic request/response schemas and job state models for the
         Standalone Greenfield Composite Plan & L-Section Microservice.
Standards: RDSO GE:G-1 / IRPWM / Indian Railways SOD (BG)
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrackType(str, Enum):
    SINGLE = "single"
    DOUBLE = "double"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GreenfieldCompositeGenerateRequest(BaseModel):
    """Request payload to generate Greenfield Composite Plan & L-Section sheets from DB."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "version_id": "20d5f3c0-1111-4222-8333-444455556666",
                "track_type": "single",
                "chainage_interval_m": 20.0,
                "generate_complete_profile": False,
                "generate_sheet_drawings": True,
                "export_pdf": False,
            }
        },
    )

    version_id: str = Field(..., description="UUID of lsection_versions row to generate")
    user_id: Optional[str] = Field(None, description="Optional UUID of requesting user for audit columns")
    output_prefix: Optional[str] = Field(None, description="Optional filename prefix for generated DXF/PDF sheets")
    track_type: Optional[TrackType] = Field(None, description="Optional override: single or double track")
    chainage_interval_m: Optional[float] = Field(None, ge=5.0, le=100.0)
    start_chainage_m: Optional[float] = Field(None)
    end_chainage_m: Optional[float] = Field(None)
    generate_complete_profile: bool = Field(False)
    generate_sheet_drawings: bool = Field(True)
    export_pdf: bool = Field(False)
    s3_output_bucket: Optional[str] = Field(None, description="Optional S3 bucket for uploading generated DXF/PDFs")
    s3_output_prefix: Optional[str] = Field(None, description="Optional S3 key prefix for uploaded sheets")
    custom_bands: Optional[list[dict[str, Any]]] = Field(
        None, description="Optional UIBandItem[] override; defaults to lsection_versions.bandHeaders"
    )


class GreenfieldPreviewResponse(BaseModel):
    """Minimal pre-flight readiness & RDSO gate summary."""

    version_id: str
    ready_to_generate: bool
    verification_status: str
    warnings: list[str] = Field(default_factory=list)


class GreenfieldGenerateAcceptedResponse(BaseModel):
    """Immediate 202 Accepted response for async Greenfield generation job."""

    job_id: str
    version_id: str
    status: str
    status_url: str
    stream_url: str


class GreenfieldSheetItem(BaseModel):
    """Generated composite sheet metadata aligned 1-to-1 with public.lsection_sheets."""

    sheet_id: Optional[str] = None
    sheet_number: int
    title: str
    start_chainage_m: float
    end_chainage_m: float
    datum_rl_m: float
    dxf_url: Optional[str] = None
    pdf_url: Optional[str] = None


class GreenfieldJobStatusResponse(BaseModel):
    """Clean job status and sheet manifest for Greenfield composite generation."""

    job_id: str
    version_id: str
    status: str
    progress_percent: int
    verification_status: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    sheets: list[GreenfieldSheetItem] = Field(default_factory=list)
    error: Optional[str] = None


class GreenfieldJobRecord:
    """In-memory runtime state for an async Greenfield generation job."""

    def __init__(self, job_id: str, version_id: str) -> None:
        self.job_id = job_id
        self.version_id = version_id
        self.status = JobStatus.QUEUED
        self.progress_percent = 0
        self.verification_status: Optional[str] = None
        self.warnings: list[str] = []
        self.sheets: list[GreenfieldSheetItem] = []
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def to_status_response(self) -> GreenfieldJobStatusResponse:
        return GreenfieldJobStatusResponse(
            job_id=self.job_id,
            version_id=self.version_id,
            status=self.status.value,
            progress_percent=self.progress_percent,
            verification_status=self.verification_status,
            warnings=self.warnings,
            sheets=self.sheets,
            error=self.error,
        )

