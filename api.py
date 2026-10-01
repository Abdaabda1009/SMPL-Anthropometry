"""api.py — thin entry point for `uvicorn api:app` (see run_api.sh /
Dockerfile). The real implementation lives in measurement_engine/; this file
only exists so the existing run/deploy commands don't need to change.
"""
from measurement_engine.app import app

__all__ = ["app"]
