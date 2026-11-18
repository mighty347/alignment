"""
Module: greenfield/job_manager.py
Purpose: Standalone asynchronous job queue & SSE event emitter for Greenfield
         Composite Plan & L-Section sheet generation.
Standards: RDSO / Indian Railways BG
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Optional

from .greenfield_models import GreenfieldJobRecord, GreenfieldSheetItem, JobStatus
from utils.logger import get_logger

logger = get_logger("GreenfieldJobManager")


class GreenfieldJobManager:
    """Concurrency-limited background worker pool with SSE subscriber streaming."""

    def __init__(self, max_concurrent_workers: int = 4) -> None:
        self.jobs: dict[str, GreenfieldJobRecord] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_workers)
        self.max_concurrent_workers = max_concurrent_workers

    def create_job(self, version_id: str) -> GreenfieldJobRecord:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        job = GreenfieldJobRecord(job_id=job_id, version_id=version_id)
        self.jobs[job_id] = job
        logger.info("Created Greenfield job %s for version_id=%s", job_id, version_id)
        return job

    def get_job(self, job_id: str) -> Optional[GreenfieldJobRecord]:
        return self.jobs.get(job_id)

    def emit_event(self, job_id: str, event_type: str, payload: dict[str, Any]) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        job.updated_at = time.time()
        if "progress_percent" in payload:
            job.progress_percent = int(payload["progress_percent"])
        if "status" in payload:
            job.status = JobStatus(payload["status"])
        if "error" in payload:
            job.error = str(payload["error"])

        event = {"event": event_type, "job_id": job_id, **payload}
        for q in list(job.subscribers):
            try:
                q.put_nowait(event)
            except Exception as err:
                logger.warning("Failed pushing SSE event for job %s: %s", job_id, err)

    def add_sheet(self, job_id: str, sheet: GreenfieldSheetItem) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        if not any(s.sheet_number == sheet.sheet_number for s in job.sheets):
            job.sheets.append(sheet)
        self.emit_event(job_id, "sheet_ready", {
            "sheet": sheet.model_dump(),
            "progress_percent": min(95, 45 + len(job.sheets) * 12),
        })

    async def execute_job(
        self,
        job_id: str,
        task_fn: Callable[..., list[GreenfieldSheetItem]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        self.emit_event(job_id, "status", {"status": JobStatus.QUEUED.value, "progress_percent": 5})

        async with self.semaphore:
            job.status = JobStatus.PROCESSING
            self.emit_event(job_id, "status", {"status": JobStatus.PROCESSING.value, "progress_percent": 15})
            try:
                loop = asyncio.get_running_loop()
                sheets = await loop.run_in_executor(None, task_fn, *args, **kwargs)
                job.sheets = sheets
                job.status = JobStatus.COMPLETED
                job.progress_percent = 100
                self.emit_event(job_id, "completed", {
                    "status": JobStatus.COMPLETED.value,
                    "progress_percent": 100,
                    "sheets": [s.model_dump() for s in sheets],
                    "verification_status": job.verification_status,
                    "warnings": job.warnings,
                })
            except Exception as exc:
                logger.exception("Greenfield job %s failed: %s", job_id, exc)
                job.status = JobStatus.FAILED
                job.error = str(exc)
                self.emit_event(job_id, "failed", {
                    "status": JobStatus.FAILED.value,
                    "error": str(exc),
                })


job_manager = GreenfieldJobManager(max_concurrent_workers=4)

