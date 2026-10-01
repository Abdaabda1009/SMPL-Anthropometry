"""measurement_engine — Measurement Engine API implementation.

A thin FastAPI wrapper around fit_mesh.py + measure.py (both at the repo
root), split into small single-responsibility modules:

    config.py           env-derived settings & constants
    models.py           pydantic request/response schemas
    auth.py             optional API-key dependency
    validation.py       gender/model_type request validation
    jobs.py             in-memory job store + thread pool
    fit_cache.py        fit result cache + in-flight de-duplication
    measuring.py        measure a body from known betas (no fitting)
    overlay.py          writes the QA overlay HTML for a fit
    fit_worker.py       runs one fit, then fans its result out
    routes_mesh.py      POST /measure/mesh, GET .../{job_id}[/overlay]
    routes_shape.py     POST /measure/shape
    routes_health.py    GET /health, /ready
    app.py              assembles the FastAPI app + lifespan

Entry point: `measurement_engine.app:app`. The root api.py just re-exports
this, so the existing `uvicorn api:app` / run_api.sh / Dockerfile keep
working unchanged.

This file deliberately imports nothing: importing the package itself (e.g.
`import measurement_engine` from a test) should stay cheap and side-effect
free. Import the submodule you actually need instead.
"""
