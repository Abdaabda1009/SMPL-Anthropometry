# Measurement Engine — accuracy review of the mesh-fitting path

Scope: `api.py` → `fit_mesh.py` → `measure.py`. Focus is **accuracy of the numbers**,
not speed (see `PERFORMANCE_REPORT.md` for that). No code was changed. Every claim
marked **[verified]** was measured on this machine (env `smpl`, torch 2.2.2, SMPLX/MALE)
during this review; claims marked **[unverified]** are reasoned from the code and are
flagged as such.

---

## 0. TL;DR

The measurement code (`measure.py`) is sound. **Essentially all of the error is in the
fitting stage**, and two specific defects dominate:

1. **The A-pose/T-pose detector and the yaw search are both axis-dependent and both
   misfire on meshes whose subject faces along X.** On `glb/male10033.glb` — the mesh
   your performance report benchmarks — the fitter picks the *worst* of the eight
   init combinations I tested: **3.5× worse rigid-stage loss than the best one.**
2. **The Huber threshold is ~3.4 cm on a 1.7 m body, but the surface deviations that
   encode body shape are 0.2–1 cm.** The robust kernel is quadratically attenuating
   exactly the signal it needs. Tightening it cut round-trip MAE **0.95 → 0.51 cm** and
   waist error **+3.46 → +1.29 cm**, while *also* improving Chamfer.

There is also a systematic **~1.1% scale bias** in the height-normalization denominator,
which multiplies every reported measurement.

And the thing that blocks all of this: **`mesh_measurements.csv` has an empty `real_cm`
column.** There is no ground truth anywhere in the repo, so no accuracy claim about this
pipeline is currently falsifiable. §5 proposes a zero-labelling-cost fix.

---

## 1. How the pipeline actually produces a number

```
.glb ─► sample 12k surface pts ─► fit(): rigid → pose → betas → polish  ─► betas (10)
                                          (Chamfer + Huber + normals)       │
                                                                            ▼
                              MeasureBody.from_body_model(gender, betas)  T-pose verts
                                                                            │
                        landmark vertex indices + regressed joints ─► measure() ─► cm
                                                                            │
                                            height_normalize_measurements(real_height)
```

The critical structural point: **only the 10 betas survive the fit.** Pose, scale and
translation are discarded. So the accuracy ceiling of the whole product is *"how well can
10 SMPLX shape coefficients, recovered by Chamfer descent, reproduce this person."*
Everything in §2 is about why that recovery is currently much worse than it needs to be.

---

## 2. Verified findings, ranked by impact

### 2.1 The pose prior and yaw search misfire on X-facing meshes — **[verified]**

`fit_mesh.py:150` decides T-pose vs A-pose from the **X extent only**:

```python
tgt_w = (target[:, 0].max() - target[:, 0].min()).item()
arms_down = tgt_w < 0.6 * tgt_h
```

But the subject's facing direction is unknown at that point — that is what the yaw search
is *for*, and it runs afterwards. Measured bounding boxes:

| mesh | x | y | z | reading |
|---|---|---|---|---|
| `male10033.glb` | **0.198** | 0.969 | **1.000** | T-pose, arm span along **Z**, faces ±X |
| `10042.glb` | 0.980 | 0.979 | 0.175 | T-pose, arm span along X, faces ±Z |
| `abbe.glb` | 0.385 | 1.135 | 0.218 | A-pose, faces ±Z |

PCA confirms all three are upright (body long axis within 1.6° of +Y) — so this is a
*facing* problem, not an up-axis problem.

For `male10033.glb`, `x/y = 0.20 < 0.6` → the fitter concludes **arms-down and injects a
±1.2 rad shoulder rotation into a mesh that is actually in a T-pose.** Separately, mapping
SMPLX's canonical arm axis (±X) onto this mesh's arm axis (±Z) needs a yaw of **±π/2** —
and `fit()` only ever tries `yaw ∈ {0, π}`.

Rigid-stage loss after 200 iterations, all 8 combinations:

| yaw | arms-down init (what the code picks) | T-pose init |
|---|---|---|
| **0** *(tried)* | 0.04601 | 0.01828 |
| **π/2** *(never tried)* | 0.04601 | **0.01323** ← best |
| **π** *(tried)* | 0.04601 | 0.01828 |
| 3π/2 *(never tried)* | 0.05207 | 0.01828 |

The current code lands on **0.04601 — 3.5× the best achievable.** The wrong pose prior
costs more than the wrong yaw, and the prior keeps pulling the shoulders back toward
A-pose through every subsequent stage (`pose_reg` is applied against `pose_init`).

This also corrupts two downstream numbers that assume Y is the body's long axis:
`init_scale = tgt_h / smpl_h`, and `chamfer_pct_height` in `api.py:118` — the very
fit-quality score the backend is supposed to threshold on.

**Fix direction:** derive the up axis and the facing axis from PCA of the point cloud
(long axis → up, second axis → arm span, third → facing), which also gives the yaw in
closed form; then decide arms-down from the *arm-span* extent rather than X. Search 4
yaws, or better, resolve facing by a cheap front/back asymmetry test. `PERFORMANCE_REPORT.md`
§2.4 already proposed closed-form rigid init for *speed* — it turns out to be an
**accuracy** fix first and a speed fix second.

### 2.2 The Huber threshold is ~5× too large — **[verified]**

`fit_mesh.py:146`: `delta = 0.02 * tgt_h` ≈ **3.4 cm** on a 1.7 m body. Inside the
quadratic region the gradient is `d/delta`, so a 1 cm surface deviation is weighted at
0.29 of its true magnitude, and a 3 mm deviation at 0.09.

That is the wrong regime for this problem. A 3 cm difference in *waist circumference* is
only ≈0.5 cm of *radius* — the geometry that distinguishes a 74 cm waist from a 77 cm
waist lives almost entirely inside the attenuated zone.

Round-trip test (§5), sweeping only `delta`, everything else identical:

| delta | Chamfer | \|betas\| | **MAE** | chest | waist | hip | head |
|---|---|---|---|---|---|---|---|
| 0.0332 m (2.0% h) **← current** | 0.01551 | 0.91 | **0.95 cm** | +2.52 | **+3.46** | +0.63 | −2.85 |
| 0.0166 m (1.0% h) | 0.01512 | 1.15 | 0.72 cm | +1.66 | +2.22 | +0.19 | −2.64 |
| 0.0083 m (0.5% h) | 0.01487 | 1.35 | 0.57 cm | +1.19 | +1.46 | −0.02 | −2.44 |
| 0.0033 m (0.2% h) | 0.0**1477** | **1.44** | **0.51 cm** | +1.09 | **+1.29** | −0.04 | −2.33 |

Monotone improvement, and **Chamfer improves too** — on a clean target this is not a
robustness/accuracy trade-off, it is strictly better. `|betas|` rising 0.91 → 1.44 toward
the ground-truth norm shows the current setting is suppressing shape recovery, not noise.

**Caveat, important:** this target is *clean* (no hair, no clothing, no reconstruction
spikes). On real Meshy/Tripo meshes a tight delta genuinely does lose outlier robustness.
The right change is not a blind constant — it is either a **graduated (annealed) delta**,
starting wide for the rigid/pose stages and tightening through the shape and polish stages,
or a scale-estimated robust kernel (e.g. Geman-McClure with the scale set from the current
residual median). **Annealing must be validated on the real `.glb` set before shipping** —
that part is [unverified].

### 2.3 `height` is not stature, and it is the normalization denominator — **[verified]**

Every user-facing number goes through `height_normalize_measurements`, which divides by
`self.measurements["height"]`. So a bias in that one quantity multiplies **all 22
measurements**. On the mean-shape neutral SMPLX body:

| quantity | value |
|---|---|
| `height` as computed (Euclidean, `HEAD_TOP` → mean of heel verts) | **170.23 cm** |
| same landmarks, vertical component only | 169.70 cm |
| true vertical extent of the body (`max y − min y`) | **172.11 cm** |

Two separate defects:

- **Euclidean, not vertical.** `HEAD_TOP` sits at z = +0.018 and the heel midpoint at
  z = −0.116; that 13 cm horizontal offset inflates the distance by 0.53 cm. Stature is a
  vertical measurement; any horizontal offset between the landmarks is pure error.
- **The heel vertex is not the sole.** Lowest vertex y = −1.3018 vs heel landmark
  y = −1.2824 — the fitter measures from ~1.9 cm *above* the floor.

Net: the engine's `height` is **1.1% smaller than the body's actual stature.** When a user
supplies a real tape stature of 178 cm, the engine scales the body so that its *smaller*
quantity equals 178 — inflating every circumference by ~1.1%. **On a 100 cm chest that is
+1.1 cm of pure systematic bias**, present in every result the API has ever returned.

**Fix direction:** define `height` as the vertical extent of the T-posed mesh. It is a
one-line change but it **shifts every historical number by ~1%**, so it needs a version
bump on the API contract, not a silent patch.

### 2.4 The fitter cannot recover a body it generated itself — **[verified]**

The strongest available test: take known betas, render that exact SMPLX body to a mesh,
feed it through the real `fit()`, and measure. No hair, no clothing, no reconstruction
noise, correct gender, correct model family, perfect T-pose — the friendliest input that
can possibly exist.

Ground truth `betas = [2.0, −1.5, 0.8, 0, 0.5, 0…]` → recovered
`[0.19, −0.85, −0.04, 0.15, 0.15, …]`.

```
measurement                      gt_cm    fit_cm   err_cm   err_%
waist circumference              74.17     77.49    +3.32    +4.5
inside leg height                69.07     71.19    +2.12    +3.1
chest circumference              89.09     91.50    +2.41    +2.7
under bust circumference         78.23     80.30    +2.07    +2.6
shoulder breadth                 31.31     32.12    +0.81    +2.6
head circumference               55.13     52.30    -2.83    -5.1
hip circumference                91.18     91.75    +0.57    +0.6
...
MAE over 22 measurements: 0.93 cm
```

Two things to read off this:

- **This is the error floor.** Every real Meshy/Tripo mesh is worse than this, because it
  adds hair, clothing thickness, recon spikes, an unknown pose and an unknown gender on
  top. A ±3.3 cm waist error on a *synthetic, exactly-representable* target means the
  real-world waist error is larger, currently unknown, and above tailoring tolerance
  (typically ±1 cm).
- **The errors are systematically toward the mean body.** GT waist 74.2 (slim) reads high
  at 77.5; GT head 55.1 reads low at 52.3. The recovered shape is flattened toward
  average. Note `height` error is 0.00 **by construction** — normalization forces it — so
  this MAE *understates* total error.

I tested and **rejected** one hypothesis here: I suspected `betas_reg` was over-regularizing.
Instrumenting the shape stage shows `betas_reg` contributes only **1.4% of the data term**
(0.00016 vs 0.01085). It is not the cause — §2.1 and §2.2 are.

### 2.5 Only 10 of 300 available shape components are used — **[verified]**

`SMPLX_NEUTRAL.pkl` ships `shapedirs` of shape **(10475, 3, 400)** — 300 shape PCs plus
100 expression. `build_model()` and `create_model()` both hardcode `num_betas=10`, so
**97% of the available shape basis is discarded.**

10 betas is the standard choice for pose-estimation-from-images, where the shape signal is
weak. Here you are registering to real 3D geometry, which carries far more shape
information than 10 coefficients can absorb. Raising to 20–50 is the single cheapest way
to lift the representational ceiling in §2.4.

**Caveat [unverified]:** more betas also means more freedom to fit hair and clothing. This
should be raised *together* with the robust-kernel work in §2.2 and validated on real
meshes — and both `fit_mesh.build_model` and `measure.create_model` must be changed in
lockstep, or the measured body will not be the body that was fitted.

### 2.6 Smaller verified issues

- **Limb circumference planes use the torso axis.** Thigh/calf/ankle all use
  `pelvis→spine3` as the plane normal, so limbs are cut at 7.7–8.9° off-perpendicular.
  Measured effect on the mean shape is small — thigh −0.52 cm, calf −0.27 cm, ankle
  +0.21 cm vs using `left_hip→left_knee` / `left_knee→left_ankle`. Real but secondary;
  likely grows on non-mean bodies. **[verified]**
- **Concurrency destroys reproducibility.** `api.py:96` calls `torch.manual_seed(0)` and
  `load_target_points` calls `np.random.seed(seed)` — both **global** — from inside a
  2-worker `ThreadPoolExecutor`. Two concurrent jobs interleave their seeding, so the same
  mesh can return different numbers depending on what else was running. For a measurement
  product this is a correctness issue, not just hygiene. Use `np.random.default_rng(seed)`
  and a local `torch.Generator`. **[verified by code inspection]**
- **The 544 MB SMPLX pkl is loaded three times per request** — `MeasureSMPLX.__init__`
  (just to read `.faces`), `from_body_model → create_model`, and `fit_mesh.build_model`.
  This is the §2.1 item of `PERFORMANCE_REPORT.md`, worse than reported there. **[verified
  by code inspection]**
- **`measure()` uses `pass` where it means `continue`** (`measure.py:89,92`, same in
  `label_measurements`). An unknown measurement name prints "not defined" and then raises
  `KeyError` on the next line instead of skipping; already-computed measurements are
  silently recomputed. **[verified by code inspection]**
- **Unbounded job state.** `api.py:_jobs` never evicts and overlay HTML files in
  `_OVERLAY_DIR` are never cleaned — a long-running service leaks both. **[verified by
  code inspection]**

### 2.7 Suspected, not yet verified

- **The body→target Chamfer term has no outlier gate.** Target→body is trimmed at 5%, but
  body→target is not. If the scan has holes (missing underarm, between legs, cropped
  feet), body vertices there are dragged toward whatever is nearest. A distance threshold
  or a visibility mask would help.
- **Correspondence subsets are fixed per stage, not resampled per iteration.** `si, ti` are
  drawn once per stage, so the optimizer overfits a fixed 6–10k subset rather than
  descending the true Chamfer. Resampling every N iterations is nearly free and gives an
  unbiased gradient.
- **5% trimming may be too little for long hair**, which on a Meshy avatar can exceed 5% of
  sampled surface area.

---

## 3. "Custom fitting" — the options you asked about, ranked

### Option A — measure the target mesh directly, using the fit only for correspondence ★
**This is the highest-ceiling change.** Today the fit is used to produce betas, and the
betas are measured — so the answer is confined to the 10-dimensional SMPLX shape subspace
(§2.4, §2.5). Instead: use the fitted body to *locate* the measurement planes and landmarks,
then transfer those planes onto the **target mesh** and slice the actual reconstructed
geometry.

- **Upside:** measurements describe the real person, not their nearest SMPLX approximation.
  This removes the §2.4 error floor entirely rather than shrinking it.
- **Downside:** you also measure their hair, their clothing and their recon artifacts. The
  SMPLX subspace is currently acting as an (aggressive) denoiser, and this gives that up.
- **Requires:** pose-aware planes (you already have the fitted pose — it is currently
  thrown away), and a reasonably watertight target.
- **Verdict:** highest upside, highest variance. Worth prototyping behind a flag and
  comparing against Option B on the same meshes.

### Option B — SMPL+D: per-vertex offsets on top of betas ★
Fit `betas + pose + per-vertex displacements D`, regularized by a Laplacian/ARAP term so D
stays smooth. Then unpose to T-pose and measure via **`from_verts()`** — which already
exists and already works, because SMPL+D preserves SMPL topology.

- **Upside:** captures real body detail outside the beta subspace while keeping SMPL
  topology, so every existing landmark index and the whole of `measure.py` keeps working
  unchanged. A middle ground between A and today.
- **Requires:** inverse-LBS to unpose the displaced mesh, and a Laplacian regularizer
  strong enough to reject hair without flattening real shape.
- **Watch out:** `from_verts` hardcodes `gender="NEUTRAL"` for the joint regressor
  (`measure.py:335,420`). On a gendered body the regressed joints — and therefore every
  circumference plane normal — would be subtly wrong. Fix that first if you go this route.
- **Verdict:** best effort-to-accuracy ratio of the three. Recommended.

### Option C — anatomical anchor terms in the loss
Add cheap geometric correspondences to the objective: top-of-head to `HEAD_TOP`, floor
contact to the heel verts, fingertips, crotch height. Chamfer alone is translation-sloppy
along the body's long axis — explicit anchors pin down exactly the landmarks that the
LENGTH measurements depend on.

Low effort, no new dependencies, composes with A and B. Also directly attacks §2.3, since
a floor/head anchor makes the height scale observable rather than emergent.

### Option D — better initialization (from §2.1)
Not really "custom fitting", but it is the prerequisite for all of the above: none of A/B/C
can help while the optimizer is starting 90° off with the wrong pose prior. **Do this first.**

---

## 4. Suggested order of work

| # | Change | Expected effect | Risk | Effort |
|---|---|---|---|---|
| 1 | Round-trip eval harness (§5) | none directly — **unblocks everything else** | none | S |
| 2 | PCA-based up/facing init + 4-way yaw + arm-span-based pose detection (§2.1) | large on affected meshes (3.5× loss → parity) | low | M |
| 3 | Fix RNG scoping (§2.6) | reproducibility | none | S |
| 4 | Annealed / scale-estimated robust kernel (§2.2) | MAE −45% on clean targets | **medium — must validate on real meshes** | M |
| 5 | `height` = vertical extent (§2.3) | removes ~1.1% global bias | **breaks API contract — version it** | S |
| 6 | Raise `num_betas` to 20–50, both call sites (§2.5) | lifts representational ceiling | medium (overfits hair) | S |
| 7 | Cache models; load the pkl once (§2.6) | latency/memory only | none | S |
| 8 | Per-limb plane normals (§2.6) | 0.2–0.5 cm on limbs | low | S |
| 9 | Option B (SMPL+D) or Option A | removes the §2.4 floor | high | L |

Items 2–4 are where the accuracy actually is. Item 1 is what makes any of it provable.

---

## 5. The missing piece: there is no ground truth

`mesh_measurements.csv` has an empty `real_cm` column. Nothing in the repo validates a
single number the engine produces, which means every item above is currently an argument
rather than a measurement — and any future change is unfalsifiable.

You don't need tape-measured subjects to fix most of this. The **synthetic round-trip**
used throughout §2 costs nothing to label:

1. Sample known `betas`.
2. Render that body to a mesh and export it.
3. Run the real `fit()` on it.
4. Compare recovered measurements against the known-betas measurements.

Any error is **purely fit-induced**, because the target is exactly representable by the
model. It gave the §2.2 and §2.4 numbers, it runs unattended, and it turns "is this
change better?" into a number. I have a working version in the scratchpad — say the word
and I'll add it as `test_roundtrip.py` with a small battery of body types (slim, heavy,
tall, short, both genders).

That harness measures the *fitter*. It does not measure hair, clothing or reconstruction
error — for those you still need real subjects with tape measurements. Ten scanned people
with a tape would let you calibrate the residual offsets that no amount of fitting work
can remove.

---

## 6. What I did not check

- No end-to-end fit was run on `abbe.glb`, `seyid.glb`, `mesh.glb` or `10042.glb` — the
  bounding-box and PCA analysis in §2.1 covers them, but their fits were not evaluated.
- The annealed-delta schedule (§2.2) is proposed, not tested. Only fixed deltas were swept,
  and only on a clean synthetic target.
- Gender selection: `api.py` defaults to `NEUTRAL`. Measuring a real woman with the neutral
  model is very likely a meaningful chest/hip error, but I did not quantify it.
- SMPL (non-X) path is untestable here — `data/smpl/` has the segmentation JSON but no
  `.pkl` models.
- `visualize.py` was not reviewed.

---
---

# Addendum — validation against real tape measurements

Ground truth supplied by you (handwritten sheet) for one subject, against the predicted
values from the engine. This is the first real validation data in the project, and it
**changes the priority order in §4** — one of my hypotheses is confirmed, one is refuted.

## A.1 The comparison

| measurement | real (tape) | predicted | error | error % |
|---|---|---|---|---|
| under bust circumference | 101 | 101.11 | **+0.11** | +0.1% |
| neck circumference | 41 | 40.79 | **−0.21** | −0.5% |
| knee left circumference | 39 | 39.48 | **+0.48** | +1.2% |
| chest circumference | 111 | 111.58 | **+0.58** | +0.5% |
| head circumference | 58 | 56.78 | −1.22 | −2.1% |
| ankle left circumference | 21 | 25.22 | **+4.22** | **+20.1%** |
| arm right length | 60 | 54.66 | **−5.34** | −8.9% |
| arm left length | 60 | 52.78 | **−7.22** | −12.0% |
| waist circumference | 110 *(see A.2)* | 101.82 | **−8.18** | −7.4% |
| hip circumference | 99 | 107.35 | **+8.35** | +8.4% |
| thigh left circumference | 26 ⚠️ | 53.30 | — | — |

**MAE = 3.59 cm** over the 10 usable measurements; MAPE 6.1% (i.e. "93.9% accurate",
close to your 95% estimate).

But the aggregate hides the real structure: **the results are bimodal.** Five
measurements are at or near tape-measure noise (MAE 0.52 cm over head/neck/chest/under
bust/knee) and four are badly wrong (MAE 7.3 cm over waist/hip/arms). There is no middle.
That is a much more useful signal than a single 95% — it means specific, identifiable
mechanisms are failing, not that the whole pipeline is uniformly "about 5% off".

⚠️ **The 26 cm left thigh is almost certainly a recording error** — that is smaller than
the measured ankle-to-knee taper and anatomically impossible next to a 39 cm knee. Please
re-measure; I excluded it from all statistics.

A note on scoring: percent-of-value flatters large circumferences. An 8 cm hip error
scores as "92%" while a 4 cm ankle error scores as "80%", even though the hip error is
twice as large in the units a tailor actually cares about. **Track absolute cm against a
±1 cm tolerance**, not percentages.

## A.2 One number needs confirmation

Your sheet reads **waist = 110 cm** but is annotated **100%**, and the prediction is
101.82 — those cannot both be right. The two readings tell completely different stories:

- **If waist = 110**, the annotation is just an eyeball slip, and the waist/hip result
  becomes the most diagnostic finding in this whole review (see A.3).
- **If waist = 100**, the waist is fine (+1.8%) and only the hip is wrong.

A.3 argues strongly that **110 is the correct reading**, but please confirm before I act
on it — it decides where the next block of work goes.

## A.3 The headline finding: the waist/hip inversion

Assuming waist = 110:

| | waist | hip | waist-to-hip ratio |
|---|---|---|---|
| **real person** | 110 | 99 | **1.111** |
| **predicted** | 101.82 | 107.35 | **0.949** |
| mean-shape male (reference) | 94.9 | 104.2 | 0.911 |

The engine did not simply get two numbers wrong — **it inverted the body's silhouette.**
The subject's waist is larger than their hips; the prediction says the opposite, and lands
almost exactly on the population average (0.949 vs the mean body's 0.911). The two errors
are nearly equal and opposite (−8.18 / +8.35): total torso volume was roughly conserved
and *redistributed* from waist to hip, toward an average-shaped body.

This is the regression-to-the-mean effect from §2.4, now confirmed on a real subject and
much larger than the synthetic test suggested (±8 cm vs ±3 cm).

## A.4 I tested my own explanation for this, and it was wrong

In §2.5 I argued that using 10 of 300 shape components was limiting, and implied it was
behind this kind of error. **That is not the cause.** I searched the SMPLX shape space for
the maximum achievable waist-to-hip ratio:

| shape space | max achievable WHR | needed |
|---|---|---|
| **10 betas (current)** | **2.182** | 1.111 |
| 50 betas | 3.140 | 1.111 |
| 100 betas | 3.616 | 1.111 |

**10 betas can reach a WHR of 2.18 — nearly double what this subject needs.** The shape
space is not the binding constraint. The betas that would reproduce this person's
silhouette *exist and are reachable*; the fitter simply is not finding them.

That relocates the problem from **representational capacity** to the **objective
function**, and it points directly at §2.2. The arithmetic lines up: an 8 cm
circumference error is only ≈**1.3 cm of radius**, and the Huber threshold is **3.4 cm**.
A 1.3 cm deviation therefore sits deep in the quadratic region and is down-weighted to
~0.38 of its true gradient. The optimizer is not being rewarded for fixing the waist,
because the loss barely registers the difference. The delta sweep in §2.2 already showed
this mechanism: waist error fell from +3.46 to +1.29 cm on tightening delta alone.

**Consequence for §4: raising `num_betas` drops from priority 6 to optional.** The
robust-kernel work (§2.2, priority 4) is now the single highest-value change, because it
is the direct cause of the largest real-world error.

## A.5 Arms: a genuine bug plus a definition mismatch

Predicted 52.78 / 54.66 against a symmetric real 60 / 60. Two independent causes:

**(a) The left/right landmarks are not mirror pairs — [verified].** On the mean-shape
body, which is bilaterally symmetric by construction:

```
arm left  length: 50.77 cm
arm right length: 52.55 cm
L/R difference  : 1.78 cm     <-- should be 0.00
```

The cause is in `landmark_definitions.py`:

| landmark | index used | true mirror vertex | mismatch |
|---|---|---|---|
| `LEFT_SHOULDER` 4442 → `RIGHT_SHOULDER` | **7218** | **7178** | 2.68 cm |
| `LEFT_WRIST` 4823 → `RIGHT_WRIST` | **7449** | **7559** | 4.57 cm |

This is a pure indexing bug, independent of fitting: it puts a fixed ~1.8 cm spurious
asymmetry into every arm measurement the engine has ever produced, and it also inflates
`shoulder breadth` (which uses both shoulder landmarks, 2.68 cm out of plane). It should
be fixed regardless of anything else in this report — it is a few-character change and
needs no re-validation of the fitter.

**(b) A straight chord is not a tape measurement.** Even after (a), both arms are ~5 cm
short, and that gap is not a fitting error — it is a definition mismatch. `arm length` is
the *straight-line Euclidean distance* from shoulder to wrist; a tailor's tape follows the
arm over the elbow. `sleeve length` (which chains shoulder→elbow→wrist) predicts 55.36 —
closer, still 4.6 cm short of 60.

To match a tape you need a **geodesic path over the mesh surface**, which is the open
`FIXME` in `measurement_definitions.py`. Alternatively, calibrate a fixed offset — but
that is a fudge factor and will not generalize across body sizes.

## A.6 Ankle: worst relative error, cause not yet isolated

+4.22 cm (+20.1%) — the largest proportional error in the set. Candidate causes, none yet
confirmed: the `LEFT_ANKLE` landmark may sit too high on the calf taper; the slicing plane
uses the *torso* axis rather than the limb axis (§2.6 — but I measured that as only
+0.21 cm on the mean shape, so it is not the main driver); convex-hull perimeter
overestimates; and ankles/feet are typically the worst-reconstructed region of a
Meshy/Tripo mesh, often fused with shoes or a ground plane. Worth a dedicated look, since
20% is far outside any usable tolerance.

## A.7 What this data does *not* show

Being straight about a prediction of mine that did not appear: §2.3 predicted a systematic
**~+1.1% inflation** on every measurement from the height-normalization denominator. The
five well-fitted measurements average **−0.16%**, not +1.1%. Either the bias is masked by
opposing fitting error, or the normalization height used here differs from my assumption.
**The §2.3 fix is still correct on its own terms** (the engine's `height` genuinely is not
stature), but this dataset does not corroborate the predicted direction, and I should not
claim it does.

Likewise, §2.1 (the yaw/pose-init bug) evidently did **not** damage this particular fit —
chest and under bust are within 0.6 cm, which a badly misaligned fit could not produce.
That bug is a *reliability* problem across inputs, not the cause of this subject's errors.
It still needs fixing, but it should be framed as "some meshes will fail badly and
unpredictably", not as a contributor here.

## A.8 Revised priority order

| # | Change | Evidence | Effort |
|---|---|---|---|
| 1 | **Fix L/R landmark mirror pairs** (A.5a) | verified 1.78 cm bug, affects every result | **XS** |
| 2 | **Annealed/scale-estimated robust kernel** (§2.2) | direct cause of the ±8 cm waist/hip error (A.4) | M |
| 3 | Round-trip + real-subject eval harness (§5) | makes 2 provable; you now have 1 real subject | S |
| 4 | Geodesic arm length, or drop the measurement (A.5b) | 5 cm definition gap | M |
| 5 | Investigate ankle landmark/plane (A.6) | 20% error, cause unknown | S |
| 6 | PCA-based init + 4-way yaw (§2.1) | reliability across meshes, not this subject | M |
| 7 | `height` = vertical extent (§2.3) | correct, but unconfirmed by this data — version it | S |
| 8 | ~~Raise `num_betas`~~ | **deprioritized — refuted in A.4** | — |

## A.9 The limit of a single subject

This is one person, and several conclusions above rest on a single data point — notably
A.3, which is the basis for making the robust kernel priority 2. One subject cannot
distinguish "the fitter systematically flattens silhouettes" from "this particular mesh
reconstructed poorly". Before investing in item 2, it is worth collecting **5–10 subjects
with tape measurements**, deliberately including at least two with a waist-to-hip ratio
above 1.0. If the inversion reproduces across them, A.3 is a systematic finding and the
fix is clearly justified; if it does not, this was one bad mesh.

The synthetic harness in §5 costs nothing and should be built first regardless — it
measures the fitter in isolation, with no tape and no subjects.
