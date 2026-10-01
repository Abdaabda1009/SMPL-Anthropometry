# Production deployment — Measurement Engine on Linux LXC

Target: `api.py` running as a long-lived service inside an **unprivileged LXC container**
(LXD/Incus or Proxmox), CPU-only, behind a reverse proxy.

All sizing numbers below were **measured on this codebase**, not estimated from defaults.
See §8 for the measurements and what they imply.

---

> **Status (2026-09-26):** B1, B3, B4 (upload cap + optional `API_KEY`), B6 and B7 are
> fixed in code — see `requirements.txt`, `Dockerfile`, and the config block at the top of
> `api.py`. The pkl-loaded-per-request waste in §8 is fixed (topology cached per process).
> Still open: **B2** (in-memory jobs → keep `--workers 1`) and **B5** (L/R landmark fix,
> left out on purpose because it changes measurement output).

## 0. Launch blockers — fix before any traffic

These are not polish items. Each one produces incorrect behaviour in production.

| # | Issue | Why it blocks launch | Where |
|---|---|---|---|
| B1 | **Global RNG seeding inside a thread pool** | `torch.manual_seed(0)` and `np.random.seed(seed)` are process-global but called per job from 2 worker threads. Concurrent jobs interleave their seeding, so **the same mesh returns different measurements depending on what else was running.** Unacceptable for a measurement product. | `api.py:96`, `fit_mesh.py:58` |
| B2 | **Job state is in-process memory** | `_jobs` is a plain dict. Any restart, crash or redeploy **loses every in-flight and completed job**, and clients polling `/measure/{job_id}` get a 404 with no way to recover. | `api.py:40` |
| B3 | **`_jobs` and overlay files grow without bound** | Nothing is ever evicted; overlay HTML in `/tmp` is never deleted. The container will OOM or fill its rootfs given enough uptime. | `api.py:40,44` |
| B4 | **No authentication, no upload size limit** | Every endpoint is open, and `await mesh.read()` loads the entire upload into RAM before any check. Your own `glb/abbe.glb` is 73 MB; a handful of concurrent large uploads exhausts memory before fitting even starts. | `api.py:150` |
| B5 | **Left/right landmark bug** (see `ACCURACY_REPORT.md` §A.5) | Ships measurably wrong arm and shoulder numbers to every customer. Few-character fix, no re-validation needed. | `landmark_definitions.py` |
| B6 | **`requirements.txt` is incomplete — the service cannot start from it** | `fastapi`, `uvicorn`, **`python-multipart`** and `networkx` are all missing. Without `python-multipart`, FastAPI raises on any `UploadFile`/`Form` endpoint — i.e. `/measure/mesh`, the only endpoint that matters. | `docker/requirements.txt` |
| B7 | **`docker/Dockerfile` targets Python 3.7.7** | `torch==2.2.2` and `numpy==1.23.5` both require ≥3.8. This image cannot build. The working env is **Python 3.10.20**. | `docker/Dockerfile` |

B1–B4 are deployment correctness. B5 is accuracy. B6–B7 mean the current packaging is not
installable at all — whatever you do below, start by fixing those two.

I'd treat B1, B2, B6 and B7 as genuinely non-negotiable. B3/B4 you can survive a soft
launch with if you watch the metrics in §9. B5 is your call — it's a known-wrong number,
not an outage.

---

## 1. Fix the packaging first

### 1.1 Corrected `requirements.txt`

Verified against the working `smpl` env and against the actual import graph:

```
# --- runtime: web layer (ALL MISSING TODAY) ---
fastapi==0.139.2
uvicorn[standard]==0.51.0
python-multipart==0.0.32      # REQUIRED by FastAPI for UploadFile/Form
pydantic==2.13.4

# --- runtime: model + geometry ---
torch==2.2.2                  # install the CPU wheel, see 1.2
numpy==1.23.5                 # pinned: newer numpy breaks smplx 0.1.26
smplx==0.1.26
trimesh==3.15.1
networkx==3.4.2               # trimesh graph ops (repair.fix_normals) — MISSING TODAY
scipy==1.10.0
pillow==12.3.0                # trimesh texture handling on .glb
pandas==1.5.3                 # imported at module level by fit_mesh.py
plotly==5.10.0                # NOT optional: measure.py -> visualize.py imports it
tqdm==4.66.1
```

Two notes from checking the import graph rather than assuming:

- **`scikit-learn` can be dropped.** It is in the current file but imported nowhere in the
  runtime path.
- **`plotly` cannot be dropped**, even if you never request an overlay. `measure.py` does
  `from visualize import Visualizer` at module level, and `visualize.py` imports plotly. A
  reasonable cleanup is to make that import lazy, but until then it is a hard dependency.

### 1.2 Install the CPU-only torch wheel

The default `pip install torch` pulls ~2 GB of bundled CUDA libraries you will never use
on a CPU container. Inside the container:

```bash
pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.2.2
pip install --no-cache-dir -r requirements.txt
```

Saves roughly 2 GB of image/rootfs and shortens cold start.

---

## 2. Decide where the model files live — do not bake them in

`data/smplx/` is **1.6 GB** (3 × ~544 MB `.pkl`). Keep it out of the container image:

- **Licensing.** SMPL-X models are distributed under a research/commercial licence tied to
  your registration. Baking them into an image that gets pushed to a registry, copied
  between hosts or shared with a contractor is a licence problem, not just a size problem.
- **Practicality.** A 1.6 GB rootfs delta per rebuild, for files that never change.

Put them on the host once and bind-mount read-only:

```bash
sudo mkdir -p /srv/smplx-models/smplx /srv/smplx-models/smpl
# copy SMPLX_{MALE,FEMALE,NEUTRAL}.pkl + smplx_body_parts_2_faces.json into smplx/
# copy smpl_body_parts_2_faces.json into smpl/
sudo chmod -R a-w /srv/smplx-models
```

⚠️ **`data/` is resolved relative to the process CWD** (`build_model(model_path="data")`,
`MeasureSMPLX.body_model_root = "data"`). The service must run with `WorkingDirectory` set
to the app root, and the mount must land exactly at `<app root>/data`. This is the single
most common way this service fails to start. Worth fixing properly with an env var, but
until then, respect it.

---

## 3. Create the container

### 3.1 LXD / Incus

```bash
lxc launch images:debian/12 measure-engine

# Resources — justified in §8
lxc config set measure-engine limits.cpu 4
lxc config set measure-engine limits.memory 8GiB
lxc config set measure-engine limits.memory.swap false   # avoid swap thrash on a 2.3GB/fit job

# Models, read-only
lxc config device add measure-engine models disk \
    source=/srv/smplx-models \
    path=/opt/measure-engine/data \
    readonly=true
```

For an **unprivileged** container the bind mount needs uid mapping, or the container sees
the files as `nobody:nogroup` and the service fails to read them:

```bash
# map container uid/gid 1000 to the host user that owns /srv/smplx-models
printf 'uid %s 1000\ngid %s 1000\n' "$(id -u smplsvc)" "$(id -g smplsvc)" \
  | lxc config set measure-engine raw.idmap -
lxc restart measure-engine
```

### 3.2 Proxmox

```bash
pct create 110 local:vztmpl/debian-12-standard_12.7-1_amd64.tar.zst \
  --hostname measure-engine \
  --cores 4 --memory 8192 --swap 1024 \
  --rootfs local-lvm:16 \
  --unprivileged 1 \
  --net0 name=eth0,bridge=vmbr0,ip=dhcp

pct set 110 -mp0 /srv/smplx-models,mp=/opt/measure-engine/data,ro=1
```

On Proxmox, an unprivileged container maps container root to host uid **100000**. So the
host directory must be readable by that range:

```bash
sudo chown -R 100000:100000 /srv/smplx-models
sudo chmod -R a-w /srv/smplx-models
```

### 3.3 Rootfs sizing

16 GB is comfortable: ~1.5 GB base Debian, ~2.5 GB venv with the CPU torch wheel, plus
headroom for uploads and overlay files in `/tmp`. Do **not** go below 8 GB — B3 means
`/tmp` grows unbounded until you fix it.

---

## 4. Inside the container

```bash
apt update && apt install -y python3.11 python3.11-venv python3-pip git

useradd -r -m -d /opt/measure-engine -s /usr/sbin/nologin smplsvc
# deploy the repo to /opt/measure-engine/app  (data/ is the read-only mount from §2)

python3.11 -m venv /opt/measure-engine/venv
/opt/measure-engine/venv/bin/pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
/opt/measure-engine/venv/bin/pip install --no-cache-dir -r /opt/measure-engine/app/requirements.txt
```

Debian 12 ships Python 3.11; the dev env is 3.10.20. Both satisfy every pin, but **pick one
and pin it** — `numpy==1.23.5` has no wheels for 3.12+, so do not let the base image drift
forward.

---

## 5. systemd unit

`/etc/systemd/system/measure-engine.service`:

```ini
[Unit]
Description=M3d Measurement Engine API
After=network-online.target

[Service]
Type=exec
User=smplsvc
Group=smplsvc

# CRITICAL: data/ is resolved relative to CWD (see §2)
WorkingDirectory=/opt/measure-engine/app

# CRITICAL: see §7 — without this, torch oversubscribes the container's CPU limit
Environment=OMP_NUM_THREADS=2
Environment=MKL_NUM_THREADS=2
Environment=TOKENIZERS_PARALLELISM=false

ExecStart=/opt/measure-engine/venv/bin/uvicorn api:app \
    --host 127.0.0.1 --port 8100 \
    --workers 1 \
    --timeout-keep-alive 75

Restart=always
RestartSec=5

# Fits are long; never let systemd kill one mid-flight
TimeoutStopSec=600
KillSignal=SIGINT

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/tmp

[Install]
WantedBy=multi-user.target
```

Two deliberate choices:

- **`--workers 1`.** `api.py` keeps job state in a process-global dict (B2). With more than
  one uvicorn worker, a client polling `/measure/{job_id}` hits a *different* process than
  the one that ran the fit and gets a spurious 404. **Do not raise `--workers` until B2 is
  fixed** with shared state (Redis or a database).
- **`--host 127.0.0.1`.** The app binds `0.0.0.0` in `run_api.sh`. Bind to loopback and let
  the reverse proxy (§6) be the only thing exposed — especially given B4, no auth.
- **`PrivateTmp=true`** gives the service its own `/tmp`, so the overlay leak (B3) is
  cleared on every restart. A mitigation, not a fix.

---

## 6. Reverse proxy, TLS and timeouts

nginx on the host or in a sibling container:

```nginx
server {
    listen 443 ssl http2;
    server_name measure.yourdomain.com;
    # ssl_certificate ... (certbot)

    # Your largest test mesh is 73 MB — cap deliberately (B4)
    client_max_body_size 100M;
    client_body_timeout 300s;

    location / {
        proxy_pass http://10.0.3.42:8100;      # container IP
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # POST /measure/mesh returns 202 immediately, so these need only cover
        # the upload itself — not the fit. Clients poll for the result.
        proxy_connect_timeout 15s;
        proxy_read_timeout    300s;
        proxy_send_timeout    300s;
    }
}
```

The async job design is the right call here and it pays off at the proxy: because
`/measure/mesh` returns `202` immediately, **no timeout anywhere has to accommodate the
multi-minute fit.** Keep it that way — don't add a synchronous "measure and wait" endpoint.

Add auth at this layer as the fastest fix for B4 — an API key check in nginx, or mTLS if
only your own backend calls it.

---

## 7. The tuning item that matters most: torch thread count

This is the classic container trap, and it will cost you more than anything else in this
document if you miss it.

**PyTorch sizes its thread pool from the host's CPU count, not the container's cgroup
limit.** On a 32-core host with `limits.cpu 4`, torch starts ~32 OpenMP threads that are
then squeezed onto 4 cores. With `ThreadPoolExecutor(max_workers=2)` running two fits, that
is ~64 threads fighting over 4 cores. Throughput collapses and latency becomes erratic.

The `OMP_NUM_THREADS=2` in §5 is the fix: **container cores ÷ concurrent fits**. With
`limits.cpu 4` and `max_workers=2`, that is 2 threads per fit.

Verify after deploy — do not assume:

```bash
lxc exec measure-engine -- /opt/measure-engine/venv/bin/python -c \
  "import torch, os; print('torch threads:', torch.get_num_threads(), '| os.cpu_count():', os.cpu_count())"
```

If `os.cpu_count()` reports the host's core count rather than your limit, lxcfs is not
masking `/proc/cpuinfo` — which is exactly why the explicit env var is not optional.

---

## 8. Capacity planning (measured, not estimated)

Peak RSS through a single fit, measured on this machine:

| stage | peak RSS |
|---|---|
| interpreter baseline | 8 MB |
| after `import torch/trimesh/smplx` | 194 MB |
| after `load_target_points(12000)` | 407 MB |
| after `build_model()` — 1st pkl load | 1425 MB |
| after `MeasureBody()` — 2nd pkl load | 1617 MB |
| after `from_body_model()` — 3rd pkl load | 1801 MB |
| after one dense `cdist(8000×12000)` + backward | **2308 MB** |

**~2.3 GB per concurrent fit.** Two things fall out of this:

1. **`max_workers=2` needs ~4.6 GB of tensor/model memory plus interpreter overhead.**
   8 GiB is the right container limit with a safety margin; 4 GiB will OOM under
   concurrent load.
2. **The service loads the 544 MB pkl three times per request** — `MeasureBody.__init__`
   (only to read `.faces`), `from_body_model → create_model`, and `fit_mesh.build_model`.
   That is ~1.1 GB of the 2.3 GB peak, and it is pure waste. Caching the models at startup
   (`PERFORMANCE_REPORT.md` §2.1) roughly **halves the memory per fit** and removes ~3 s of
   per-request load time. This is the highest-value production change in the whole
   document, and it is low risk.

**Throughput.** `PERFORMANCE_REPORT.md` measured 155–187 s per fit at 6000 target points.
`api.py` uses **12000**, and the polish stage runs against all of them, so expect roughly
**4–6 minutes per fit** *(estimate, extrapolated — not directly measured at 12000)*.

With `max_workers=2`: **~20–30 fits/hour, sustained.** Plan launch volume against that
number. If you need more, scale **horizontally** — more containers behind the proxy — not
by raising `max_workers`, which is bounded by memory (2.3 GB each) and by CPU contention
(§7). Horizontal scaling also requires B2 fixed first, so that any container can serve a
poll for any job.

---

## 9. Observability

Minimum viable set, given what can actually go wrong here:

- **Health.** `/health` exists and is trivially cheap — wire it to your monitor. Consider a
  deeper readiness check that confirms `data/` is mounted and a model loads, because B-class
  failure mode "container up, models unreadable" returns a healthy `/health` today.
- **Queue depth.** Export `len(_jobs)` by status. With a 4–6 min fit and 2 workers, a
  modest burst queues for a long time and you want to see it building.
- **Fit quality.** `meta.chamfer_pct_height` is already computed per job. **Log it and
  alert on the tail** — it is your only automated signal that a fit went wrong.
  ⚠️ Caveat from `ACCURACY_REPORT.md` §2.1: that metric divides by the mesh's Y extent, which
  is wrong for meshes not oriented as assumed. Treat it as a coarse signal until fixed.
- **Memory.** Alert at 80% of the 8 GiB limit. Given B3, a slow climb across days is the
  expected failure, and it is invisible unless you watch for it.
- **Logs.** `journalctl -u measure-engine`. The app already logs job start/completion with
  chamfer values; ship them off-box so a container restart doesn't erase the evidence.

---

## 10. Launch checklist

**Packaging**
- [ ] `requirements.txt` corrected (§1.1) — `python-multipart` present (B6)
- [ ] `Dockerfile` fixed or deleted; Python pinned to 3.10/3.11 (B7)
- [ ] CPU-only torch wheel confirmed (`torch.version.cuda is None`)

**Correctness (B-list)**
- [ ] RNG made per-job: `np.random.default_rng(seed)`, local `torch.Generator` (B1)
- [ ] Job state externalised, or `--workers 1` enforced and documented (B2)
- [ ] `_jobs` TTL eviction + overlay cleanup (B3)
- [ ] Auth at the proxy + `client_max_body_size` (B4)
- [ ] L/R landmark mirror fix (B5)

**Container**
- [ ] Unprivileged, `limits.memory 8GiB`, `limits.cpu 4`
- [ ] `/srv/smplx-models` bind-mounted **read-only** at `<app>/data`, idmap correct
- [ ] Models **not** baked into the image (licence, §2)
- [ ] `WorkingDirectory` = app root, verified by an actual fit

**Runtime**
- [ ] `OMP_NUM_THREADS` = cores ÷ workers, **verified inside the container** (§7)
- [ ] systemd `TimeoutStopSec=600` so restarts don't kill in-flight fits
- [ ] Proxy timeouts cover upload only, not the fit
- [ ] TLS + auth in front; app bound to 127.0.0.1

**Before real traffic**
- [ ] End-to-end fit inside the container, result matches a known-good local run
- [ ] Two concurrent fits — confirm no OOM and no RNG cross-contamination (B1 regression test)
- [ ] Restart mid-fit; confirm the client gets a sane error, not a hang
- [ ] Soak: 24 h of periodic fits, watch RSS and `/tmp` for the B3 climb

---

## 11. Sequencing suggestion

You have two separate tracks and they are not equally urgent:

**Track 1 — can this run in production at all?** B1, B2, B6, B7, model caching (§8), torch
threads (§7). This is a few days of work and it is all low-risk, well-understood plumbing.

**Track 2 — are the numbers right?** Everything in `ACCURACY_REPORT.md`. Only B5 (the
landmark fix) is both trivial and clearly correct; the rest needs the evaluation harness
before you can tell whether a change helped.

These are independent — Track 1 doesn't depend on Track 2. But it's worth being clear-eyed
that finishing Track 1 gives you a **reliable** service returning numbers with a known
±3.6 cm MAE and a verified waist/hip inversion on at least one real subject. Whether that's
launchable depends entirely on what you're promising customers. A soft launch with stated
tolerances, or an internal-only launch while Track 2 proceeds, are both reasonable; a
public "tailor-grade measurement" claim is not yet supported by the data you have.
