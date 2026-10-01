"""jobs.py — the in-memory job store: status/result per job_id, plus the
thread pool that runs fits.

Job state is in-process memory (see app.py's lifespan docstring): run
exactly ONE uvicorn worker per instance, or two processes would each have
their own, unsynchronized copy of `_jobs`.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .config import JOB_TTL_SECONDS, MAX_WORKERS

# Fits are CPU-bound (several minutes each); a small pool lets a couple run
# concurrently without starving each other on a typical dev machine. Shared
# (not private to one module) because routes_mesh.py submits to it directly.
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

_jobs: dict[str, dict] = {}
_lock = Lock()


def create(job_id: str, **fields) -> None:
    """Register a new job_id with the given initial fields (status, result,
    error, ...)."""
    with _lock:
        _jobs[job_id] = fields


def update(job_id: str, **fields) -> None:
    """Merge fields into an existing job's record."""
    with _lock:
        _jobs[job_id].update(fields)


def get(job_id: str) -> dict | None:
    """The full record for one job, or None if it doesn't exist (never ran,
    or was already evicted)."""
    with _lock:
        return _jobs.get(job_id)


def evict_expired() -> None:
    """Drop finished job records older than JOB_TTL_SECONDS.

    Only the small per-job status/result dict is dropped here. Overlay HTML
    on disk is named after the fit (mesh+params), not the job — several job
    ids can share one cached fit — so its lifetime is tied to the fit
    cache's own LRU eviction (see fit_cache.py), not to any single job's TTL.
    """
    cutoff = time.time() - JOB_TTL_SECONDS
    with _lock:
        expired = [jid for jid, j in _jobs.items()
                   if j["status"] in ("complete", "failed")
                   and j.get("finished_at", time.time()) < cutoff]
        for jid in expired:
            _jobs.pop(jid)


def counts_by_status() -> dict[str, int]:
    """Job counts by status, for GET /ready."""
    counts = {s: 0 for s in ("pending", "running", "complete", "failed")}
    with _lock:
        for j in _jobs.values():
            counts[j["status"]] += 1
    return counts
