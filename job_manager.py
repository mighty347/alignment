"""
Module: service/alignment/job_manager.py
Purpose: Lightweight asynchronous background job tracker with SSE support.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Optional

try:
    from alignment.utils.logger import get_logger
    from alignment.alignment_models import JobRecord, JobStatus
except ImportError:
    from utils.logger import get_logger
    from alignment_models import JobRecord, JobStatus

logger = get_logger("AlignmentJobManager")


class JobManager:
    """Asynchronous Job Registry for Horizontal Alignment Detection."""

    def __init__(self, max_concurrent_workers: int = 4):
        self.jobs: dict[str, JobRecord] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_workers)
        self.max_concurrent_workers = max_concurrent_workers

    def create_job(self, project_name: str = "Horizontal Alignment Detection", user_id: str = "default_user") -> JobRecord:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        job = JobRecord(job_id=job_id, user_id=user_id, project_name=project_name)
        active_or_queued = sum(1 for j in self.jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING))
        job.queue_position = max(1, active_or_queued + 1)
        self.jobs[job_id] = job
        logger.info("Created Job %s (Queue Position: %d)", job_id, job.queue_position)
        return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    def emit_event(self, job_id: str, event_type: str, payload: dict[str, Any]) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        job.updated_at = time.time()
        if "progress_percent" in payload:
            job.progress_percent = int(payload["progress_percent"])
        if "step" in payload:
            job.current_step = str(payload["step"])
        if "status" in payload:
            job.status = JobStatus(payload["status"])
        if "error" in payload:
            job.error = str(payload["error"])

        event = {"event": event_type, "timestamp": time.time(), "job_id": job_id, **payload}
        for q in list(job.subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    async def execute_job(self, job_id: str, task_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Execute heavy worker in a background thread with concurrency protection."""
        job = self.jobs.get(job_id)
        if not job:
            return

        async with self.semaphore:
            job.status = JobStatus.PROCESSING
            job.current_step = "Processing alignment geometry..."
            job.progress_percent = 15
            self.emit_event(job_id, "status", {"status": "processing", "progress_percent": 15, "step": job.current_step})

            try:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(None, lambda: task_fn(*args, **kwargs))

                job.status = JobStatus.COMPLETED
                job.progress_percent = 100
                job.current_step = "Completed"
                if hasattr(res, "detected_files") and res.detected_files:
                    job.detected_files = res.detected_files
                elif isinstance(res, dict):
                    job.detected_files = res

                self.emit_event(job_id, "completed", {
                    "status": "completed",
                    "progress_percent": 100,
                    "step": "Completed",
                    "detected_files": job.detected_files,
                })
                logger.info("Job %s completed successfully", job_id)
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.current_step = f"Failed: {exc}"
                self.emit_event(job_id, "failed", {
                    "status": "failed",
                    "progress_percent": 100,
                    "error": str(exc),
                    "step": f"Failed: {exc}"
                })
                logger.exception("Job %s failed: %s", job_id, exc)


job_manager = JobManager()
