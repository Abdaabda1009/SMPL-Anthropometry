"""auth.py — optional API key check, shared as a FastAPI dependency by every
/measure/* route."""
import secrets

from fastapi import Header, HTTPException

from .config import API_KEY


def require_api_key(x_api_key: str | None = Header(None)) -> None:
    """No-op unless API_KEY is set in the environment; then requires a
    matching `X-API-Key` header, compared in constant time."""
    if API_KEY is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "invalid or missing X-API-Key")
