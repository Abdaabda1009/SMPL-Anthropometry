"""routes_shape.py — POST /measure/shape: measure directly from known SMPL/
SMPLX betas. No mesh, no fitting — a forward pass plus the measurement pass."""
import torch
from fastapi import APIRouter, Depends, HTTPException

from .auth import require_api_key
from .measuring import from_betas
from .models import MeasureResult, ShapeRequest
from .validation import validate

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/measure/shape", response_model=MeasureResult)
def measure_shape(req: ShapeRequest) -> MeasureResult:
    """Synchronous: this costs ~1s, not the several-minute registration that
    /measure/mesh runs. Declared `def` rather than `async def` so FastAPI
    runs it on its threadpool instead of blocking the event loop."""
    gender, model_type = validate(req.gender, req.model_type)
    if len(req.betas) != 10:
        raise HTTPException(
            400, f"betas must have exactly 10 values, got {len(req.betas)}")

    betas = torch.tensor([req.betas], dtype=torch.float32)
    measurements, labeled, height_normalized = from_betas(
        model_type, gender, betas, req.height_cm)
    return MeasureResult(
        measurements=measurements,
        labeled=labeled,
        height_normalized=height_normalized,
        gender=gender,
        model_type=model_type,
        meta={"betas_norm": float(betas.norm().item())},
    )
