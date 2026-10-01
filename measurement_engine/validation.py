"""validation.py — shared request validation for gender/model_type, used by
both /measure/mesh and /measure/shape."""
from fastapi import HTTPException

from .config import GENDERS, MODEL_TYPES


def validate(gender: str, model_type: str) -> tuple[str, str]:
    """Normalize + check gender/model_type; raises HTTP 400 on an invalid
    value. Returns the normalized (uppercase gender, lowercase model_type)."""
    gender = gender.upper()
    if gender not in GENDERS:
        raise HTTPException(400, f"gender must be one of {sorted(GENDERS)}")
    model_type = model_type.lower()
    if model_type not in MODEL_TYPES:
        raise HTTPException(400, f"model_type must be one of {sorted(MODEL_TYPES)}")
    return gender, model_type
