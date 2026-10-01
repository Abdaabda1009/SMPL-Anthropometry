"""routes_health.py — unauthenticated liveness/readiness probes."""
from fastapi import APIRouter, HTTPException

from . import jobs
from .config import APP_DIR, GENDERS

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict:
    """Readiness: model files are mounted and the job queue is reported."""
    smplx_dir = APP_DIR / "data" / "smplx"
    missing = [f"SMPLX_{g}.pkl" for g in sorted(GENDERS)
               if not (smplx_dir / f"SMPLX_{g}.pkl").exists()]
    if missing:
        raise HTTPException(503, f"model files missing in {smplx_dir}: {missing}")
    return {"status": "ready", "jobs": jobs.counts_by_status()}
