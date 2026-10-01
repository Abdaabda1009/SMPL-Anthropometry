"""app.py — assembles the FastAPI app: startup/shutdown lifecycle plus the
route modules. This is what `uvicorn measurement_engine.app:app` (or the
thin root api.py re-export) serves.

Job state is in-process memory (see jobs.py): run exactly ONE uvicorn worker
per instance — run_api.sh and the Dockerfile both already pin --workers 1.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Import first: config.py loads .env (if present) as an import-time side
# effect. fit_mesh/measure below read their own env vars (e.g.
# SMPL_MODEL_CACHE in measure.py) at import time too, so they must be
# imported *after* this or they'd miss values that only exist in .env.
from .config import API_KEY, MAX_UPLOAD_BYTES, MAX_WORKERS, PRELOAD_GENDERS

from fit_mesh import build_model
from measure import _load_topology

from . import jobs, routes_health, routes_mesh, routes_shape

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("measurement_engine")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Warm the per-model-type topology (faces + segmentation) so the first
    # request doesn't pay the pkl load; optionally preload full body models.
    _load_topology("smplx", os.path.join("data", "smplx"))
    for gender in PRELOAD_GENDERS:
        build_model("smplx", gender)
    logger.info("ready (workers=%d, max_upload=%dMB, auth=%s, preloaded=%s)",
                MAX_WORKERS, MAX_UPLOAD_BYTES // (1024 * 1024),
                "on" if API_KEY else "off", PRELOAD_GENDERS or "none")
    yield
    jobs.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="M3d Measurement Engine API", version="1.0.0",
              lifespan=_lifespan)
app.include_router(routes_mesh.router)
app.include_router(routes_shape.router)
app.include_router(routes_health.router)
