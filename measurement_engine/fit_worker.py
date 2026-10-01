"""fit_worker.py — runs one fit (fit_mesh.fit) in a background thread, then
fans the result out to every job waiting on it and caches it for next time.

This is the only place fit_mesh.fit() is actually called from the API; the
route just decides whether a fit is even needed (see routes_mesh.py).
"""
import logging
import time
from pathlib import Path

import torch

from fit_mesh import build_model, fit, load_target_points

from . import fit_cache, jobs, overlay
from .config import N_TARGET_POINTS
from .measuring import from_betas

logger = logging.getLogger("measurement_engine")


def run(job_id: str, mesh_path: str, model_type: str, gender: str,
        overlay_wanted: bool, cache_key: tuple) -> None:
    """Fit `cache_key`'s owner (first) request, then fan the one result out
    to every job_id waiting on it — the owner plus any requests that arrived
    for the same (mesh, model_type, gender) while this was in flight —
    applying each one's own height_cm/overlay preference, and cache the fit
    itself so later requests for this mesh skip fitting entirely.
    """
    jobs.update(job_id, status="running")
    try:
        # per-job generator (same stream as torch.manual_seed(0)) so concurrent
        # fits can't perturb each other's sampling
        generator = torch.Generator().manual_seed(0)
        target_pts, target_normals, _ = load_target_points(
            mesh_path, n_points=N_TARGET_POINTS)
        model = build_model(model_type, gender)
        betas, posed_verts, faces, final_chamfer = fit(
            model, target_pts, target_normals=target_normals, verbose=False,
            generator=generator)

        # native (non-height-normalized) measurements only: height is applied
        # per-waiter below, since different requests for this same mesh can
        # ask for different height_cm.
        measurements, labeled, _ = from_betas(model_type, gender, betas, None)

        overlay_path = None
        if overlay_wanted:
            overlay_path = overlay.write(
                f"{cache_key[0][:16]}_{model_type}_{gender}",
                target_pts, posed_verts, faces)

        # chamfer as % of body height per side: scale-free fit-quality score
        # the backend can threshold (e.g. flag scans above ~2%)
        tgt_h = float(target_pts[:, 1].max() - target_pts[:, 1].min())
        chamfer_pct = float(final_chamfer) / 2.0 / tgt_h * 100.0 if tgt_h > 0 else -1.0

        entry = {
            "measurements": measurements,
            "labeled": labeled,
            "chamfer_error": float(final_chamfer),
            "chamfer_pct_height": chamfer_pct,
            "betas_norm": float(betas.norm().item()),
            "overlay_path": overlay_path,
        }

        waiters, evicted_overlay = fit_cache.store_and_resolve(
            cache_key, entry, job_id, overlay_wanted)
        if evicted_overlay:
            Path(evicted_overlay).unlink(missing_ok=True)

        finished_at = time.time()
        for wid, w_height, w_overlay in waiters:
            w_overlay_path = overlay_path if w_overlay else None
            result = fit_cache.build_result(
                entry, gender, model_type, w_height, w_overlay_path,
                cache_hit=False, deduped=(wid != job_id))
            jobs.update(wid, status="complete", result=result, error=None,
                        overlay_path=w_overlay_path, finished_at=finished_at)
        logger.info("job %s complete (chamfer=%.5f, %.2f%% of height, %d waiter(s))",
                    job_id, final_chamfer, chamfer_pct, len(waiters))
    except Exception as e:
        logger.exception("job %s failed", job_id)
        waiters = fit_cache.resolve_failure(cache_key, job_id, overlay_wanted)
        finished_at = time.time()
        for wid, _height, _overlay in waiters:
            jobs.update(wid, status="failed", error=str(e), finished_at=finished_at)
    finally:
        Path(mesh_path).unlink(missing_ok=True)
