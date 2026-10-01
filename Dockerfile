# Measurement Engine API — CPU-only image.
#
# Model files are NOT baked in (size + SMPL-X licence). Mount them read-only:
#   docker build -t measure-engine .
#   docker run -p 8100:8100 -v /srv/smplx-models:/app/data:ro \
#       -e API_KEY=change-me measure-engine
# The mounted dir must contain smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.pkl and
# smplx/smplx_body_parts_2_faces.json (and smpl/... if serving model_type=smpl).
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
 && pip install -r requirements.txt

COPY api.py fit_mesh.py measure.py utils.py measurement_definitions.py \
     landmark_definitions.py joint_definitions.py visualize.py ./
COPY measurement_engine/ ./measurement_engine/

RUN useradd --system --uid 1000 app && mkdir -p /app/data && chown app /app
USER app

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8100/health')"

# ONE worker: job state lives in process memory (see measurement_engine/jobs.py).
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8100", \
     "--workers", "1", "--timeout-keep-alive", "75"]
