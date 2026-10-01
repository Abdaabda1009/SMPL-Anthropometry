"""fit_cache.py — caches fit results by (mesh hash, model_type, gender) and
coalesces concurrent requests for the same not-yet-cached mesh.

The registration in fit_mesh.fit() is a seeded, deterministic optimization:
the same mesh bytes + model_type + gender always produce the same betas and
measurements. So a repeat upload of the same file can skip the multi-minute
fit entirely and return in ~milliseconds — that's the whole point of this
module. See config.FIT_VERSION for the cache-invalidation rule.
"""
from collections import OrderedDict
from threading import Lock

from .config import FIT_CACHE_SIZE, FIT_VERSION
from .models import MeasureResult

# cache_key -> {measurements, labeled, chamfer_error, chamfer_pct_height,
# betas_norm, overlay_path}. height_cm is deliberately NOT part of the key
# (it's a cheap linear rescale of `measurements`, applied per-request in
# build_result) so different callers requesting the same mesh at different
# heights still share one cached fit.
_cache: "OrderedDict[tuple, dict]" = OrderedDict()

# cache_key -> list of (job_id, height_cm, overlay) for requests currently
# waiting on the one in-flight fit for that key; fanned out to all of them
# (and cached) when that fit finishes.
_inflight: dict[tuple, list[tuple[str, float | None, bool]]] = {}

# Guards both dicts together: a request always needs to check/update them as
# one atomic step (see lookup_or_join), so one lock is simpler and safer than
# two that would have to be taken in a fixed order.
_lock = Lock()


def key_for(file_hash: str, model_type: str, gender: str) -> tuple:
    """The cache key for one (mesh, model_type, gender) fit."""
    return (file_hash, model_type, gender, FIT_VERSION)


def lookup_or_join(key: tuple, job_id: str, height_cm: float | None,
                    overlay: bool) -> tuple[dict | None, bool]:
    """Single atomic step: look up a cached result, or on a miss register
    this job as either the owner of a new fit or a waiter on one already in
    flight.

    Returns (cached_entry_or_None, is_owner). is_owner only matters when the
    entry is None: True means the caller must actually run the fit, False
    means it should just wait to be fanned out to when the owner's finishes.
    """
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)  # LRU touch
            return cached, False
        waiters = _inflight.get(key)
        if waiters is None:
            _inflight[key] = [(job_id, height_cm, overlay)]
            return None, True
        waiters.append((job_id, height_cm, overlay))
        return None, False


def store_and_resolve(key: tuple, entry: dict, owner_job_id: str,
                       owner_overlay: bool) -> tuple[list, str | None]:
    """Called once a fit succeeds: caches `entry`, evicts the LRU entry if
    now over capacity, and pops+returns everyone waiting on this key (owner
    included) so the caller can fan the result out to each of them.

    Returns (waiters, evicted_overlay_path_or_None) — the second value is an
    overlay file that's now unreachable by any future cache hit and should
    be deleted.
    """
    with _lock:
        _cache[key] = entry
        _cache.move_to_end(key)
        evicted_overlay = None
        while len(_cache) > FIT_CACHE_SIZE:
            _, evicted = _cache.popitem(last=False)
            if evicted.get("overlay_path"):
                evicted_overlay = evicted["overlay_path"]
        waiters = _inflight.pop(key, [(owner_job_id, None, owner_overlay)])
        return waiters, evicted_overlay


def resolve_failure(key: tuple, owner_job_id: str, owner_overlay: bool) -> list:
    """Called when a fit raises: pops and returns everyone waiting on this
    key, without caching anything, so the same mesh can be retried from
    scratch by the next request."""
    with _lock:
        return _inflight.pop(key, [(owner_job_id, None, owner_overlay)])


def build_result(entry: dict, gender: str, model_type: str,
                  height_cm: float | None, overlay_path: str | None,
                  cache_hit: bool, deduped: bool = False) -> MeasureResult:
    """Build a MeasureResult from a cached fit entry, applying this
    request's own height_cm and overlay visibility. `overlay_path` is
    resolved by the caller: it may be None even when the underlying fit has
    one, if this particular request didn't ask for it.
    """
    measurements = entry["measurements"]
    height_normalized = None
    if height_cm is not None:
        old_height = measurements["height"]
        height_normalized = {k: v / old_height * height_cm
                              for k, v in measurements.items()}
    return MeasureResult(
        measurements=measurements,
        labeled=entry["labeled"],
        height_normalized=height_normalized,
        gender=gender,
        model_type=model_type,
        meta={
            "chamfer_error": entry["chamfer_error"],
            "chamfer_pct_height": entry["chamfer_pct_height"],
            "betas_norm": entry["betas_norm"],
            "overlay_ready": 1.0 if overlay_path else 0.0,
            "cache_hit": 1.0 if cache_hit else 0.0,
            "deduped": 1.0 if deduped else 0.0,
        },
    )
