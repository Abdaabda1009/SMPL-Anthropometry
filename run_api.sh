#!/usr/bin/env bash
# Launch the Measurement Engine API with the active Python env (e.g. `conda activate smpl`).
# Override with env vars: PYTHON, HOST, PORT, plus the API settings documented in
# measurement_engine/config.py (PRELOAD_GENDERS, MAX_WORKERS, etc. — set them in
# .env, see .env_example, rather than here; config.py loads it automatically).
set -euo pipefail
cd "$(dirname "$0")"
# Same thread caps as the Dockerfile: with MAX_WORKERS fits running
# concurrently, an unset OMP/MKL thread count lets torch oversubscribe the
# CPU across them. Only sets a default — an already-exported value wins.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "${PYTHON:-python}" -m uvicorn api:app \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-8100}" \
    --workers 1 --timeout-keep-alive 75
