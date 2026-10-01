# Measurement Engine — System Design & API Spec

Design doc for using **SMPL-Anthropometry** as the single measurement engine that
handles every body-measurement task in your product. Read this before writing the
engine wrapper — it describes how the underlying system works today, what it can and
cannot do, and the API surface to build on top of it.

---

## 1. What this repo actually is

SMPL-Anthropometry takes a **parametric human body** (SMPL or SMPLX) and returns
tailoring-style measurements (height, chest/waist/hip circumference, arm length, …)
in **centimetres**. It does *not* detect bodies, read photos, or measure arbitrary
scans on its own. It measures a body that is already expressed in SMPL/SMPLX form.

There are exactly **three ways a body can enter the engine**:

| Input | Method | Notes |
|-------|--------|-------|
| Shape params `betas` (+ gender) | `from_body_model(gender, shape)` | Cleanest. Body is generated in canonical T-pose. |
| Raw vertices `(N,3)` | `from_verts(verts)` | N must be 6890 (SMPL) or 10475 (SMPLX), **in SMPL vertex order**. |
| Arbitrary mesh (`.glb/.obj/.ply`) | `fit_mesh.py` → betas → `from_body_model` | Registers an SMPL template to the mesh first (see §5). |

Everything else (a photo, a Meshy AI avatar, a depth scan) has to be converted into
one of those three before it can be measured.

---

## 2. How the measurement actually works

All measurements reduce to two primitive types (`measurement_definitions.py`):

**LENGTH** — Euclidean distance between two landmark vertices.
`measure_length()` looks up 1–2 fixed vertex indices per endpoint (averaging a tuple
of indices when given), then sums segment lengths. Example: `height` = distance from
`HEAD_TOP` to `HEELS`.

**CIRCUMFERENCE** — the body is sliced by a plane and the slice perimeter is measured.
`measure_circumference()`:
1. `plane_origin` = mean of the landmark vertices (e.g. belly-button verts for waist).
2. `plane_normal` = vector between two **regressed joints** (e.g. `pelvis`→`spine3`).
3. `trimesh.intersections.mesh_plane` cuts the body → many little line segments.
4. `filter_body_part_slices` keeps only segments on the intended body part (via the
   face-segmentation JSON in `data/`), discarding stray slices (e.g. the arm when
   slicing the chest).
5. A convex hull of the surviving points gives the perimeter → summed → cm.

Key consequence: **measurements depend on fixed vertex indices + regressed joints.**
That is why an arbitrary mesh can't be measured directly — it has no correspondence to
those indices. It also means measurement quality is only as good as the pose/shape of
the body you feed in.

### Data flow

```
                    betas + gender ─┐
                                    ├─►  MeasureBody  ──► verts (N,3)  + joints
   arbitrary mesh ─► fit_mesh.py ───┘        │                │
        (Chamfer registration)               ▼                ▼
                                        landmark lookup   joint regressor
                                              │                │
                                              └──► measure() ──►  measurements{name: cm}
                                                          │
                                    label_measurements()  ├─► labeled_measurements{A..P}
                                    height_normalize()     └─► height_normalized_measurements
```

---

## 3. Module map

| File | Role |
|------|------|
| `measure.py` | **Core.** `MeasureBody(model_type)` factory → `MeasureSMPL`/`MeasureSMPLX`. Holds `from_body_model`, `from_verts`, `measure`, `label_measurements`, `height_normalize_measurements`, `visualize`. |
| `measurement_definitions.py` | `STANDARD_LABELS` (A–P), `MEASUREMENT_TYPES`, and per-model `LENGTHS` / `CIRCUMFERENCES` / `CIRCUMFERENCE_TO_BODYPARTS`. **Add new measurements here.** |
| `landmark_definitions.py` | `SMPL_LANDMARK_INDICES` / `SMPLX_LANDMARK_INDICES` — named vertex indices. |
| `joint_definitions.py` | Joint name → index maps, joint counts, joint regressor loading. |
| `utils.py` | Face-segmentation loading, slice filtering, convex hull, joint regressor. |
| `evaluate.py` | `evaluate_mae(gt, pred)` → per-measurement absolute error (cm). ⚠️ its `__main__` uses a dead API — ignore it. |
| `visualize.py` | Plotly 3D viz of body + landmarks + measurements. |
| `fit_mesh.py` | Fit SMPL/SMPLX to an arbitrary mesh (§5). |
| `test.py` | End-to-end reference of the current API. Copy patterns from here. |
| `data/{smpl,smplx}/` | `.pkl` body models + `*_body_parts_2_faces.json` segmentation. |

---

## 4. Current API (canonical usage)

> The README is authoritative. `evaluate.py.__main__` shows an **outdated** API
> (`MeasureSMPL(smpl_path=)`, `from_smpl`) — do not copy it.

```python
import torch
from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS

measurer = MeasureBody("smplx")                 # "smpl" or "smplx" (a STRING, not a path)

# --- pick ONE input path ---
measurer.from_body_model(gender="NEUTRAL",      # "MALE"|"FEMALE"|"NEUTRAL"
                         shape=torch.zeros(1, 10))
# or:
# measurer.from_verts(verts=torch.tensor(verts, dtype=torch.float32))  # (10475,3) smplx / (6890,3) smpl

# --- measure ---
measurer.measure(measurer.all_possible_measurements)   # or a subset list of names
measurer.label_measurements(STANDARD_LABELS)

measurer.measurements            # {'height': 175.4, 'waist circumference': 82.1, ...}  (cm)
measurer.labeled_measurements    # {'A': 56.6, 'B': 34.2, ...}  (A–P letters)
measurer.labels2names            # {'A': 'head circumference', ...}

# --- optional: normalize to a known real height ---
measurer.height_normalize_measurements(175.0)          # scales everything so height==175
measurer.height_normalized_measurements

# --- compare two bodies ---
from evaluate import evaluate_mae
mae = evaluate_mae(gt.measurements, pred.measurements)  # {name: abs_error_cm}
```

**All values are in cm.** SMPL's native scale is metric-ish but arbitrary for fitted
meshes — always height-normalize when the source scale is unknown.

### Measurements available

SMPL exposes more than SMPLX. Query at runtime with `measurer.all_possible_measurements`.

- **Both models (16 standard, A–P):** height, head/neck/chest/waist/hip circumference,
  wrist/bicep/forearm right circumference, arm right length, arm left length,
  inside leg height, thigh/calf/ankle left circumference, shoulder breadth,
  shoulder-to-crotch height.
- **SMPL-only extras:** `arm length (shoulder to elbow)`, `arm length (spine to wrist)`
  (⚠️ straight-line, not geodesic — see FIXME in defs), `crotch height`,
  `Hip circumference max height`.

---

## 5. The arbitrary-mesh path (`fit_mesh.py`)

For `.glb`/`.obj` bodies (e.g. Meshy AI, in `glb/`) that have no SMPL correspondence:

```
python fit_mesh.py --input glb/10042.glb --model_type smplx --gender FEMALE \
    --height 170 --export_fit fitted.obj --overlay fit_overlay.html \
    --save_measurements mesh_measurements.csv
```

Pipeline: sample surface points → 3-stage Adam fit minimizing Chamfer distance
(rigid+scale init trying yaw 0 and π → +body_pose regularized toward T/A-pose →
+shape betas regularized) → measure the fitted **betas** in clean T-pose.

Critical caveats (these are why fitted numbers drift from a tape measure):
- Meshy meshes are **not metrically calibrated** → `--height` is effectively required.
- The **pose stage is essential**; without it betas hit extremes to fake limb bends.
- Hair, clothing, surface thickness bias circumferences outward (~1–2 cm on head).
- CPU-only, ~2–3 min per mesh. When piping, stdout buffers — run in background & tail.

---

## 6. Building THE engine (recommended wrapper)

Rather than sprinkling `MeasureBody` calls across your app, wrap the three input paths
behind **one façade** so every measurement task in the product goes through the same
door. Suggested shape (`measurement_engine.py`, to be written):

```python
class MeasurementEngine:
    """Single entry point for all body measurement in the product."""

    def __init__(self, model_type="smplx", default_gender="NEUTRAL"):
        ...

    # --- three input paths, one output contract ---
    def measure_from_betas(self, betas, gender=None, target_height_cm=None) -> "Result": ...
    def measure_from_verts(self, verts, target_height_cm=None) -> "Result": ...
    def measure_from_mesh(self, path, gender=None, target_height_cm=None) -> "Result": ...

    # everything returns the SAME Result object
```

`Result` should always carry: `measurements` (name→cm), `labeled` (A–P), `gender`,
`model_type`, `height_normalized` (if a target height was given), and `meta`
(fit Chamfer error / confidence for the mesh path, `None` otherwise).

Design rules for the engine layer:
1. **One output contract** regardless of input path — callers never branch on source.
2. **Return dataclasses, not raw dicts** — stable field names, easy to serialize to JSON/API.
3. **Height-normalize by policy**: if the caller supplies a real height, always return
   normalized values as the "primary" numbers and keep native as a debug field.
4. **Surface confidence** for the mesh path (final Chamfer + betas magnitude); callers
   should be able to reject a bad fit. `betas` beyond ~±3 or high Chamfer = low trust.
5. **Load models once.** `smplx.create` + segmentation JSON load is not free; construct
   `MeasureBody` per request is OK on CPU but cache the underlying `smplx` model if you
   go high-throughput.
6. **Fitting is slow and blocking** (2–3 min). For an API service, run `measure_from_mesh`
   in a worker/queue, not inline in a request handler.
7. **Validate inputs early**: `from_verts` asserts exact vertex count/order — catch and
   return a clean error, don't let the assert bubble to a 500.

### Turning it into a service (if needed)

- Wrap the engine in FastAPI/Flask; `/measure/betas` and `/measure/verts` can be
  synchronous, `/measure/mesh` must be async (job id + poll, or callback).
- Serialize `Result` to JSON directly (all floats/cm).
- Keep the model `.pkl` files out of the image if large — mount `data/` as a volume.

---

## 7. Environment & setup

- Conda env **`smpl`**: numpy 1.23.5, torch 2.2.2, smplx 0.1.26, trimesh 3.15.1,
  scipy 1.10.0, plotly 5.10.0, pandas 1.5.3 (see `requirements.txt` at the repo
  root). Newer numpy/torch may break smplx 0.1.26 — pin these.
- Models load from `data/{smpl,smplx}/` with `ext="pkl"` (hardcoded). This repo
  currently has **SMPLX** models only (`data/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.pkl`)
  plus both segmentation JSONs. **SMPL `.pkl` models are missing** — add them to
  `data/smpl/` if you want to run the `smpl` path (download from the smplx repo).
- Runs on CPU by default; no GPU required (fitting is just slower). A
  `Dockerfile.gpu` variant exists for eventual GPU deployment — as of this
  writing it accelerates `/measure/shape` only (see `TORCH_DEVICE` in
  `.env_example`); `fit_mesh.py`'s optimization loop is still CPU-only, see
  that file's module docstring for what's left to migrate.

---

## 8. Limitations & gotchas (know these before shipping)

- **No image → body.** Needs an upstream regressor (e.g. SPIN in the parent folder,
  which outputs SMPL, not SMPLX) or the mesh-fitting path.
- **Straight-line arm length** for `arm length (spine to wrist)` — not geodesic
  (FIXME in `measurement_definitions.py`). Don't advertise it as tape-accurate.
- **Circumferences via convex hull** slightly overestimate concave regions.
- **Pose sensitivity**: `from_verts` on a non-T-pose body gives wrong lengths; the
  clean path is always betas measured in canonical pose.
- **Fitted meshes inherit clothing/hair error**; treat mesh-path numbers as estimates.
- **Scale is arbitrary for fitted meshes** — a measurement without a known height is
  only meaningful as a ratio.
- **SMPL vs SMPLX differ**: measurement sets and some joint choices differ (e.g. SMPLX
  wrist circumference uses a different joint pair). Pick one model type per product and
  stick to it for comparability.

---

## 9. Implementation checklist

1. [ ] Confirm env `smpl` activates and `python test.py` runs clean.
2. [ ] Decide model type (recommend **SMPLX**, since that's what's installed).
3. [ ] Add SMPL `.pkl` models to `data/smpl/` only if you need the SMPL path.
4. [ ] Write `measurement_engine.py` with the `MeasurementEngine` + `Result` façade (§6).
5. [ ] Add input validation + error types around `from_verts` / mesh loading.
6. [ ] Decide height-normalization policy and make it the default output.
7. [ ] For mesh input: move fitting to a background worker; expose fit confidence.
8. [ ] Add a thin serialization layer (`Result` → JSON) if exposing an API.
9. [ ] Write regression tests: mean-shape betas should reproduce known cm values;
       `from_verts` of a betas body should match `from_body_model` (~0 cm MAE — see
       the sanity check in `test.py`).
```
