# Measurement Engine — integration guide

How a frontend talks to `api.py`. Upload a body mesh (`.glb`), poll, get measurements in cm.

- **Base URL (dev):** `http://<host>:8100`
- **Start it:** `./run_api.sh` (binds `0.0.0.0:8100`, must run from the repo dir)
- **Shape of the flow:** submit → poll → read result. A fit takes **~4 minutes** (measured: 230s for a 5.6 MB `.glb`), so it is
  asynchronous by design. Never block a UI thread on it.

---

## Read this before you write the client

Three things will bite a station frontend immediately:

1. **There is no CORS header.** A browser `fetch()` straight from the page to port 8100
   fails on the preflight, every time. Fix one of two ways — see §4.
2. **There is no auth and no upload size limit.** Keep the engine on the LAN / loopback
   only. Do not expose port 8100 to the open internet.
3. **Only `model_type=smplx` works.** `smpl` is accepted by the API but the `data/smpl/`
   model files are not present, so it fails inside the job. Leave the default alone.

Also worth knowing: job state lives in process memory. Restart the engine and every
in-flight and finished job is gone (a poll then returns 404).

---

## 1. Submit a mesh

`POST /measure/mesh` — `multipart/form-data` → **202 Accepted**

| Field | Type | Default | Notes |
|---|---|---|---|
| `mesh` | file | *required* | `.glb`, `.obj`, `.ply` |
| `height_cm` | float | `null` | Real height. **Send it** — see §3 |
| `gender` | string | `NEUTRAL` | `MALE` \| `FEMALE` \| `NEUTRAL` |
| `model_type` | string | `smplx` | leave as is |
| `overlay` | bool | `false` | also build a QA overlay page |

```json
{ "job_id": "8f2c1a...", "status": "pending" }
```

Bad `gender` / `model_type` → 400. Note the engine returns 202 as soon as the file is
saved; it has not started fitting yet.

## 1b. Or skip the mesh: measure from shape params

`POST /measure/shape` — JSON → **200**, synchronous, **~1s**. No fitting, no polling.

Use this when you already have SMPL/SMPLX betas (e.g. from a previous fit you stored, or
from your own regressor). It runs only the forward pass and the measurement pass.

```bash
curl -X POST http://localhost:8100/measure/shape -H "Content-Type: application/json" \
  -d '{"betas":[0,0,0,0,0,0,0,0,0,0],"gender":"MALE","height_cm":180}'
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `betas` | float[10] | *required* | exactly 10 values, else 400 |
| `gender` | string | `NEUTRAL` | as above |
| `model_type` | string | `smplx` | as above |
| `height_cm` | float | `null` | as above |

Returns the same `result` object as §2, except `meta` has only `betas_norm`. Missing
`betas` → 422; wrong length → 400.

## 2. Poll for the result

`GET /measure/{job_id}` → **200**, or 404 if unknown/restarted.

Poll every **3–5s**, and give up after ~10 min. `status` is one of
`pending` · `running` · `complete` · `failed`.

```json
{
  "job_id": "8f2c1a...",
  "status": "complete",
  "error": null,
  "result": {
    "measurements":      { "height": 171.29, "chest circumference": 100.76, "...": 0 },
    "labeled":           { "A": 56.5, "B": 36.85, "...": 0 },
    "height_normalized": { "height": 180.0,  "chest circumference": 105.9,  "...": 0 },
    "gender": "MALE",
    "model_type": "smplx",
    "meta": {
      "chamfer_error": 0.0231,
      "chamfer_pct_height": 1.24,
      "betas_norm": 2.81,
      "overlay_ready": 1.0
    }
  }
}
```

- `measurements` — 22 entries keyed by full name, **in SMPL's arbitrary scale**.
- `labeled` — the 16 standard measurements keyed `A`–`P` (`A` head circumference,
  `E` waist, `F` hip, `P` height …; the full map is `STANDARD_LABELS` in
  `measurement_definitions.py`).
- `height_normalized` — same keys as `measurements`, rescaled so `height == height_cm`.
  **`null` when you did not send `height_cm`.**
- `meta.chamfer_pct_height` — fit quality, % of body height. **Flag anything above ~2%**
  as a bad scan rather than showing the numbers.

On `failed`, `result` is `null` and `error` carries the message.

## 3. Which numbers to display

Meshes from photogrammetry / generative reconstruction have **no real-world scale**. So:

- Sent `height_cm` → show **`height_normalized`**. These are real centimetres.
- Did not → `measurements` is in arbitrary units and is not a body measurement in cm.
  Do not put it in front of a user.

Practically: make height a required field in the station UI.

## 4. Getting past CORS

**Option A — proxy (preferred).** The station calls your own backend; the backend calls
the engine server-to-server. No CORS involved, and you get auth and a size limit for free.

**Option B — enable CORS on the engine.** Fine for a kiosk on a trusted LAN. Add to `api.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # the station origin; avoid "*"
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 5. Minimal client

```js
const ENGINE = "http://localhost:8100";

export async function measure(file, { heightCm, gender = "NEUTRAL", signal } = {}) {
  const form = new FormData();
  form.append("mesh", file);
  form.append("gender", gender);
  if (heightCm != null) form.append("height_cm", String(heightCm));

  const res = await fetch(`${ENGINE}/measure/mesh`, { method: "POST", body: form, signal });
  if (!res.ok) throw new Error(`submit failed: ${res.status}`);
  const { job_id } = await res.json();

  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 4000));
    const p = await fetch(`${ENGINE}/measure/${job_id}`, { signal });
    if (p.status === 404) throw new Error("job lost (engine restarted?)");
    const job = await p.json();
    if (job.status === "failed") throw new Error(job.error || "fit failed");
    if (job.status === "complete") return { jobId: job_id, ...job.result };
  }
  throw new Error("timed out");
}
```

Show a progress state for the full ~4 minutes — there is no percentage available, only
`pending` → `running` → `complete`.

## 6. Other endpoints

- `GET /health` → `{"status":"ok"}`. Use it to show engine-reachable in the station UI.
- `GET /measure/{job_id}/overlay` → HTML page, fitted body over the scan. Only exists if
  you submitted with `overlay=true`; otherwise 404. Useful as an operator QA view, drop it
  in an `<iframe>`.

## 7. Capacity

Two fits run concurrently (`ThreadPoolExecutor(max_workers=2)`); further submissions queue.
Each cached body model holds a few hundred MB of RAM. For a single station this is fine —
just don't expect a third simultaneous scan to start right away.
