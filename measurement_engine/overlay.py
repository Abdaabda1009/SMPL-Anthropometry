"""overlay.py — API-side wrapper around fit_mesh.write_overlay: adds path
caching (reuse an existing file for the same fit) and makes overlay
generation best-effort (a failure here never fails the measurement itself)."""
import logging
import tempfile
from pathlib import Path

from fit_mesh import write_overlay as _write_overlay_html

logger = logging.getLogger("measurement_engine")

OVERLAY_DIR = Path(tempfile.gettempdir()) / "m3d_engine_overlays"
OVERLAY_DIR.mkdir(exist_ok=True)


def write(name_stub: str, target_pts, posed_verts, faces) -> str | None:
    """Write (or reuse) the overlay for one fit; returns its path, or None on
    failure.

    `name_stub` identifies the FIT (mesh hash + model_type + gender), not any
    one job, so repeat requests for the same mesh reuse the same file —
    including across a process restart, since it's a pure function of the
    input. The API never keeps the full target mesh around (only its sampled
    points), so this always plots the "target (scan)" point-cloud variant of
    the overlay, not the full-surface one fit_mesh.py's CLI can draw.
    """
    path = OVERLAY_DIR / f"{name_stub}.html"
    if path.exists():
        return str(path)
    try:
        return _write_overlay_html(path, target_pts, posed_verts, faces)
    except Exception:
        logger.exception("overlay generation failed for %s", name_stub)
        return None
