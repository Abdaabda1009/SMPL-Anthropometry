# M3d — Project Overview

_Generated 2026-07-22, from the current `backend/` code (not the older `system.md` /
`system-overview.md` docs, which describe the pre-migration Blender/Meshy stack)._

M3d turns 2–4 phone photos + a height into body measurements: **upload → 3D
reconstruction → SMPL/SMPLX fit-and-measure**. FastAPI backend, React Native
(Expo) mobile frontend, and a separate measurement-engine microservice.

---

## 1. Main pipeline

```
POST /scans/  (images + height_cm + gender)
      │
      ▼
upload.py        save raw/{front,left,right,back}.jpg, create ScanRecord (pending)
      │            (gender: "Male"/"Female"/"Other" → MALE/FEMALE/NEUTRAL)
      ▼
pipeline.py      process_scan() runs as a FastAPI BackgroundTask
      │
      ├─ 1. preprocess.py   rembg background removal + blur/crop/size checks
      │                      ⚠️ CURRENTLY SKIPPED (_SKIP_PREPROCESS=True, 2026-07-21
      │                         temp flag) — raw JPEGs are fed straight to Tripo
      │                         while testing whether pre-cut RGBA cutouts confuse it
      │
      ├─ 2. tripo.py         Tripo3D multiview-to-3d API (replaces Meshy):
      │                      submit 2–4 views → poll (self-managed retry around
      │                      transient connection drops) → download body.glb
      │
      └─ 3. measurements.py  HTTP client to the standalone Measurement Engine
                             service (SMPL-Anthropometry, separate repo/process):
                             POST mesh+height+gender → poll job → map response
                             onto Measurements → write result/measurements.json
```

Status lifecycle: `pending → processing → complete | failed`, `progress` 10 → 30 → 80 → 100.

---

## 2. Services architecture

Two independent processes:

| Service | Repo | Role |
|---|---|---|
| `backend/` (this repo) | FastAPI app | scan lifecycle, uploads, Tripo3D reconstruction, mobile API contract |
| Measurement Engine | `spin/SMPL-Anthropometry` (sibling repo, conda env `smpl`) | fits SMPL/SMPLX betas to the uploaded mesh (Chamfer optimization via `fit_mesh.py`), measures the fitted body, returns cm values |

`backend/` never imports torch/smplx directly — `app/services/measurements.py`
is a pure `httpx` client against `settings.measurement_engine_url`
(`http://127.0.0.1:8100` by default). This is why Blender and MediaPipe are
gone from `backend/`: that math now lives entirely in the other service.

---

## 3. Directory structure (`backend/`)

```
backend/
  app/
    main.py            FastAPI app, CORS, /health
    config.py           Settings: tripo_api_key, tripo_model_version,
                         measurement_engine_url, scans_dir
    models.py            ScanStatus · Measurements (22 fields) · ScanResult · ScanRecord
    store.py             SQLite (data/scans.db) + manifest.json mirror per scan
    routes/scans.py      POST /, GET status|mesh|mesh/file|result, POST /measure-mesh (dev), DELETE
    services/
      upload.py           save raw images, normalize gender, create ScanRecord
      preprocess.py       rembg cutout + quality flags (currently bypassed, see §1)
      pose.py             tiny alpha-mask head/feet-visibility check (no MediaPipe)
      tripo.py            Tripo3D SDK client — submit/poll/download GLB
      measurements.py     Measurement Engine HTTP client + result assembly
      pipeline.py         process_scan(): the whole orchestration, one function
  data/scans.db          SQLite store
  scans/{id}/             raw/ processed/ mesh/ result/ manifest.json  (per scan)
  Agents/desgin_agent/    separate design-agent experiment, not part of the scan pipeline
  blob_detection/         standalone detector.py, not wired into the pipeline
  measurement_engine_integration_report.md   design doc for the engine split (marked
                                              "pre-implementation" but largely already done)
```

Frontend (`../frontend/`): Expo/React Native app. `camera.jsx` posts
`images` + `height_cm` + `gender` + `weight_kg` + `age` to `POST /scans/`;
`profile.jsx` / `onboarding/setup.jsx` collect gender as `Male|Female|Other`.

---

## 4. Key data model (`app/models.py`)

`Measurements` — 22 fields the mobile UI renders, all default `0.0`:

- **From the engine today:** neck, chest, waist, hips, upper_thigh, calf,
  upper_arm, wrist, shoulder_width, inseam, height, plus additive fields with
  no old equivalent — head_circumference, forearm_circumference,
  arm_length_right/left, ankle_circumference, shoulder_to_crotch_height.
- **No engine equivalent yet** (stay `0.0` + a `missing:<field>` flag):
  under_bust, knee, torso_length, back_length, sleeve_length.

`ScanRecord` also carries `gender` (`MALE|FEMALE|NEUTRAL`, default `NEUTRAL`)
and `mesh_path` (local GLB — the source of truth; Tripo URLs expire).

Field mapping lives in `measurements.py:_FIELD_MAP` — engine's native
measurement names (e.g. `"thigh left circumference"`) → `Measurements`
attribute names. Laterality is fixed by the engine's standard set (thigh/
calf/ankle = left, arm/wrist/forearm = right), surfaced as `laterality:*`
flags rather than a per-request choice.

---

## 5. API summary

| Method | Path | Notes |
|---|---|---|
| POST | `/scans/` | form: `images` (2–4), `height_cm`, `gender` → `{scan_id, status}` (202) |
| GET | `/scans/{id}/status` | `{scan_id, status, progress, mesh_ready}` |
| GET | `/scans/{id}/mesh` | `{glb_url, raw_glb_url, obj_url}` |
| GET | `/scans/{id}/mesh/file` | streams the GLB |
| GET | `/scans/{id}/result` | full `ScanResult` |
| POST | `/scans/measure-mesh` | dev: mesh-in → measurements-out, skips Tripo entirely |
| DELETE | `/scans/{id}` | removes DB row + scan folder |
| GET | `/health` | `{status: ok}` |

---

## 6. Known gaps / in-flight state

- `_SKIP_PREPROCESS = True` in `pipeline.py` is a temporary debug flag (dated
  2026-07-21) — quality flags (blurry/cropped/too-small) are currently not
  being generated at all until this is reverted.
- Limb-circumference accuracy (thigh/bicep/forearm) remains the known weak
  spot across both the old Blender slicer and the new SMPL-Anthropometry
  engine — a measurement-algorithm issue, not a slice-placement or engine
  issue (see memory: slice-calibration, measurement-engine-migration).
- `under_bust`, `knee`, `torso_length`, `back_length`, `sleeve_length` have no
  engine equivalent yet (Phase 2 — custom landmark definitions).
- `measurement_engine_integration_report.md` still says "Status: planning /
  pre-implementation," but the code shows the swap is essentially done
  (Tripo3D live, HTTP measurement client live, gender wired end-to-end) — that
  doc needs a status update.
