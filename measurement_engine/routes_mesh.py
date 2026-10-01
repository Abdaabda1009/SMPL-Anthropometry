"""routes_mesh.py — POST /measure/mesh (upload a scan, get a job back) plus
the two GET routes used to poll a job: /measure/{job_id} and its /overlay."""
import hashlib
import logging
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import fit_cache, jobs
from .auth import require_api_key
from .config import MAX_UPLOAD_BYTES
from .fit_worker import run as run_fit
from .models import JobStatus
from .validation import validate

logger = logging.getLogger("measurement_engine")
router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/measure/mesh", status_code=202)
def measure_mesh(
    mesh: UploadFile = File(...),
    height_cm: float | None = Form(None),
    gender: str = Form("NEUTRAL"),
    model_type: str = Form("smplx"),
    # TEMP: defaulted to True for the current Gradio-UI experiment, which
    # has no control to pass overlay=true explicitly. Revert to Form(False)
    # once that UI exposes the option.
    overlay: bool = Form(True),
) -> dict:
    """Declared `def` (not async) so the upload copy to disk runs on
    FastAPI's threadpool instead of blocking the event loop."""
    gender, model_type = validate(gender, model_type)
    jobs.evict_expired()

    # stream the upload to disk in chunks instead of reading it all into RAM,
    # hashing as we go so the cache lookup below costs no extra I/O
    suffix = Path(mesh.filename or "mesh.glb").suffix or ".glb"
    hasher = hashlib.sha256()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="m3d_engine_") as tmp:
        mesh_path = tmp.name
        too_large = False
        while chunk := mesh.file.read(1024 * 1024):
            tmp.write(chunk)
            hasher.update(chunk)
            if tmp.tell() > MAX_UPLOAD_BYTES:
                too_large = True
                break
    if too_large:
        Path(mesh_path).unlink(missing_ok=True)
        raise HTTPException(413, f"mesh exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")

    cache_key = fit_cache.key_for(hasher.hexdigest(), model_type, gender)
    job_id = uuid.uuid4().hex

    # Cache check and in-flight registration happen as one atomic step, so a
    # fit can't complete (and populate the cache) in the gap between them —
    # otherwise this request could miss a hit that finished a moment earlier
    # and start a redundant fit.
    cached, is_owner = fit_cache.lookup_or_join(cache_key, job_id, height_cm, overlay)

    if cached is not None:
        # identical mesh + params already fit: no registration needed, just
        # apply this request's own height_cm/overlay to the cached result
        Path(mesh_path).unlink(missing_ok=True)
        overlay_path = cached.get("overlay_path") if overlay else None
        if overlay_path and not Path(overlay_path).exists():
            overlay_path = None  # evicted/cleaned up since the entry was cached
        result = fit_cache.build_result(cached, gender, model_type, height_cm,
                                        overlay_path, cache_hit=True)
        jobs.create(job_id, status="complete", result=result, error=None,
                    overlay_path=overlay_path, finished_at=time.time())
        logger.info("job %s served from cache (model_type=%s gender=%s)",
                    job_id, model_type, gender)
        return {"job_id": job_id, "status": "complete"}

    jobs.create(job_id, status="pending", result=None, error=None)

    if is_owner:
        jobs.executor.submit(run_fit, job_id, mesh_path, model_type, gender,
                              overlay, cache_key)
        logger.info("job %s queued (model_type=%s gender=%s height_cm=%s overlay=%s)",
                    job_id, model_type, gender, height_cm, overlay)
    else:
        # same mesh+params already fitting for another job_id: don't run a
        # second fit, just wait to be fanned out to when that one finishes
        Path(mesh_path).unlink(missing_ok=True)
        logger.info("job %s aliased to in-flight fit (model_type=%s gender=%s)",
                    job_id, model_type, gender)
    return {"job_id": job_id, "status": "pending"}


@router.get("/measure/{job_id}", response_model=JobStatus)
async def get_measure_job(job_id: str) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return JobStatus(job_id=job_id, status=job["status"],
                     result=job["result"], error=job["error"])


@router.get("/measure/{job_id}/overlay")
async def get_measure_overlay(job_id: str) -> FileResponse:
    """QA overlay HTML for a completed job (request with overlay=true)."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    path = job.get("overlay_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "no overlay for this job (submit with overlay=true)")
    return FileResponse(path, media_type="text/html")
