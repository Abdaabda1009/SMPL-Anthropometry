"""config.py — environment-derived settings and constants for the
Measurement Engine API. Reading these has no side effects beyond loading
.env into os.environ and the one os.chdir below, so any module can import
this cheaply and first.

Values are resolved in this order (highest priority first):
  1. a real environment variable, e.g. `API_KEY=x uvicorn ...` or `docker run -e`
  2. the repo-root `.env` file (see .env_example), if one exists
  3. the hardcoded default below
Real env vars always win — load_dotenv(override=False) never overwrites one
that's already set — so a `.env` file is safe to commit-adjacent (it isn't;
see .gitignore) without it fighting a deployment's own -e flags.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# fit_mesh.py/measure.py resolve `data/` relative to the process's cwd, so we
# chdir to the repo root as a side effect of importing this module — every
# other module in this package imports from here, directly or transitively,
# so this always runs before any request-time code needs a "data/..." path.
APP_DIR = Path(__file__).resolve().parent.parent
os.chdir(APP_DIR)

# No-op (returns False, doesn't raise) if APP_DIR/.env doesn't exist — fine
# for Docker/production, where real env vars are passed in some other way.
load_dotenv(APP_DIR / ".env", override=False)

# if set, /measure/* requires header `X-API-Key: <value>`
API_KEY = os.environ.get("API_KEY") or None

# concurrent fits (each needs ~2 GB RAM)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))

# mesh upload size limit
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024)

# finished job records are evicted after this many seconds (see jobs.py)
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", str(24 * 3600)))

# e.g. "MALE,FEMALE" — load these SMPLX models at startup instead of on first use
PRELOAD_GENDERS = [g.strip().upper() for g in
                   os.environ.get("PRELOAD_GENDERS", "").split(",") if g.strip()]

GENDERS = {"MALE", "FEMALE", "NEUTRAL"}
MODEL_TYPES = {"smpl", "smplx"}

# surface samples per fit; part of the fit's identity (see fit_cache.py)
N_TARGET_POINTS = 12000

# Bump this whenever fit_mesh.fit()'s algorithm, N_TARGET_POINTS, or the
# measurement pass changes in a way that changes output for the same mesh —
# it's mixed into the fit cache key so old (now-stale) entries stop being
# served after a real behavior change.
FIT_VERSION = "1"

# distinct (mesh, model_type, gender) fit results kept in the result cache
FIT_CACHE_SIZE = int(os.environ.get("FIT_CACHE_SIZE", "256"))
