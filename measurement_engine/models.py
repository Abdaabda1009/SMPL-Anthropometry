"""models.py — pydantic request/response schemas for the Measurement Engine
API. Pure data shapes; no logic lives here."""
from typing import Literal

from pydantic import BaseModel


class MeasureResult(BaseModel):
    """A completed measurement: native cm values, the STANDARD_LABELS subset,
    height-rescaled values (if height_cm was given), and fit/cache
    diagnostics (chamfer error, whether this came from the cache, etc.)."""
    measurements: dict[str, float]
    labeled: dict[str, float]
    height_normalized: dict[str, float] | None
    gender: str
    model_type: str
    meta: dict[str, float]


class JobStatus(BaseModel):
    """Status of one /measure/mesh job, as returned by GET /measure/{job_id}."""
    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    result: MeasureResult | None = None
    error: str | None = None


class ShapeRequest(BaseModel):
    """Body for POST /measure/shape: measure directly from known betas, no
    mesh and no fitting."""
    betas: list[float]
    gender: str = "NEUTRAL"
    model_type: str = "smplx"
    height_cm: float | None = None
