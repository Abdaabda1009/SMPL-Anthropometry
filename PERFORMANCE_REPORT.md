# `fit_mesh.py` performance report — accuracy-preserving options

Scope constraint honored: **no reduction in target/SMPL point count** (the biggest lever,
~7.6x, was rejected because you're deploying to a VM and want max accuracy regardless of
load time). Everything below keeps the same 6000-point Chamfer objective and looks for
speed elsewhere. All numbers are measured on this machine (M3, macOS, `env smpl`, torch
2.2.2), fitting `glb/male10033.glb`, SMPLX/MALE, 6000 target points.

---

## 1. Where the time actually goes

| Step | Time |
|---|---|
| `import torch/smplx/trimesh` | 0.85s |
| `fit_mesh` import | 0.29s |
| `build_model` (load SMPLX pkl) | 1.05s |
| `load_target_points` (load glb + sample surface) | 0.08s |
| **`fit()` (the Adam optimization loop)** | **155–187s** |

`fit()` is >98% of total wall time. Everything else is noise. Within `fit()`,
per-iteration cost breaks down as:

| Component | ms/iter (n=6000) |
|---|---|
| SMPL/SMPLX forward pass (linear blend skinning) | ~2.7 |
| + Chamfer distance forward (`torch.cdist` + `.min()`) | ~68 |
| + backward pass | ~70 |

So the SMPL forward pass itself is negligible — **the entire cost is the pairwise Chamfer
distance (`torch.cdist` on a 6000×6000 matrix) and its autograd**, run ~1120 times
(2×60 direction search + 200 rigid + 300 pose + 500 shape).

---

## 2. Options that keep identical point counts and don't sacrifice accuracy

### 2.1 Cache the SMPL/SMPLX model in `api.py` — free, zero risk
`_run_fit` currently calls `build_model(model_type, gender)` on *every* request (~1s: pkl
deserialization + tensor setup). There are only 6 possible `(model_type, gender)` combos.
Building them once at service startup (or lazily with an in-memory cache) removes ~1s per
request with **no numerical difference whatsoever** — `build_model` is deterministic and
stateless per call. Trivial, no accuracy question even applies here.

### 2.2 Convergence-based early stopping — verified per-stage, mixed results
I printed the loss curve per stage (on a cheaper n=2000 run, curve *shape* is what matters
here, not absolute time):

```
[rigid    0] chamfer=0.08170
[rigid   50] chamfer=0.07997
[rigid  100] chamfer=0.07992   <- plateaued
[rigid  150] chamfer=0.07992
[pose     0] chamfer=0.08122
[pose   250] chamfer=0.05969   <- still improving at last checkpoint
[shape    0] chamfer=0.06793
[shape  200] chamfer=0.05113
[shape  300] chamfer=0.03996   <- big drop right before this
[shape  450] chamfer=0.03883   <- still improving, not converged
```

**Verified finding, not assumption:**
- **Rigid stage (200 iters)** fully plateaus by ~iter 100 → safe to halve with an
  early-stop/tolerance check (e.g. stop if loss delta < 1e-5 over 20 steps). No accuracy
  cost — it has already converged, the extra iterations are wasted compute.
- **Pose stage (300 iters)** and **shape stage (500 iters)** are *still visibly improving*
  near their current iteration caps. Shortening these would be a real accuracy trade-off,
  not a free win — the opposite of what you asked for. Do not touch these without
  re-validating against ground truth.

Net effect of only trimming the rigid stage: modest (rigid is the cheapest of the three
stages already). Real but small.

### 2.3 LBFGS instead of Adam — modest win, verified, with a caveat
Tested LBFGS (`strong_wolfe` line search) vs Adam on the rigid+scale stage, run to full
convergence:

| Optimizer | Iterations/config | Final chamfer | Wall time |
|---|---|---|---|
| Adam | 200 iters | 0.03335 | 25.30s |
| LBFGS | 10 outer × 20 max_iter | 0.03335 (**identical**) | 17.88s |

Same local optimum, ~1.4x faster. **But** LBFGS is not simply "fewer iterations needed" —
when under-provisioned (e.g. 3 outer × 10 max_iter = 36 closure calls), it converges to a
visibly worse local minimum (chamfer 0.148 vs 0.033), because Chamfer's `.min()` makes the
loss landscape non-smooth in a way that hurts line-search methods if stopped early. So this
is a real, verified ~1.4x, low-risk *if* you don't also cut the iteration budget — but it's
not the order-of-magnitude change you're likely hoping for.

### 2.4 Closed-form rigid initialization instead of gradient descent
The current 2-direction "grid search" (yaw 0 and π, 60 Adam iters each = 120 iters just to
pick a facing direction) could be replaced by a closed-form rigid alignment (e.g. Procrustes
/ centroid+PCA-axis alignment, or a few ICP sweeps via `scipy.spatial.cKDTree`) run in
milliseconds instead of ~15s of gradient descent. This is architecturally sound (a rigid
transform has a closed-form least-squares solution; gradient descent is overkill for it) but
I have not benchmarked it end-to-end here — flagging as promising, not yet verified.

### 2.5 Batch concurrent requests instead of speeding up one request
`api.py`'s `ThreadPoolExecutor(max_workers=2)` runs at most 2 fits in parallel, each at
batch size 1. SMPL/SMPLX forward + Chamfer are matrix ops that batch naturally (`betas`
shape `(B,10)` instead of `(1,10)`). If the VM serves multiple scans concurrently, fitting
them as one batched tensor op (instead of N independent single-item fits sharing a thread
pool) uses hardware far more efficiently — this improves **throughput**, not the latency of
any single fit, and changes zero math per-user. Worth considering if concurrent load (not
single-user latency) is the real production constraint.

---

## 3. Hardware-dependent options (matter more once this is actually on a VM)

### 3.1 Apple MPS (this machine only) — tested, minor, and *not relevant to your VM*
Moving the fit loop to `mps` gave only ~1.7x (79ms/iter vs 141ms/iter CPU), with a logged
warning that some ops (padding) fall back to a slower unoptimized path on MPS. **This is
moot for your deployment target anyway** — MPS is Apple Metal only; a Linux/cloud VM won't
have it at all. Mentioned only to close the loop on the earlier ask.

### 3.2 Real CUDA GPU on the VM — the actual high-leverage hardware option
If the target VM has an NVIDIA GPU (common for CV inference boxes — T4/A10/A100 class),
this is a different story than MPS: full CUDA support, no op fallbacks, and access to fused
kernels purpose-built for this exact operation. In particular, **`pytorch3d`'s
`chamfer_distance`/`knn_points`** use a fused CUDA kernel that never materializes the full
6000×6000 distance matrix (unlike the current `torch.cdist` + autograd, which does) —
same math, same accuracy, historically large constant-factor speedups on real GPUs. Not
installed in this env (`ModuleNotFoundError: No module named 'pytorch3d'`), and its
optimized path is CUDA-only — the CPU fallback is not meaningfully faster than what you
have now, so this only pays off if the VM actually has an NVIDIA GPU. Not benchmarked here
since no CUDA hardware is available on this machine to test against.

### 3.3 `torch.compile`
PyTorch 2.x's `torch.compile` can fuse the forward/backward graph and cut per-iteration
Python/dispatch overhead without changing any numerics. Not benchmarked yet — worth a quick
try since it's a one-line wrapper around `forward_verts`/`chamfer_distance` with no
algorithmic change, hence no accuracy risk.

---

## 4. Your actual question: "SMPL 3D model generator instead of fitting" — feed-forward regression

This is the architecturally different option, and it's what "fitting via optimization" is
normally contrasted with in the human-body-recovery literature:

- **Feed-forward regressors** (HMR2.0/4D-Humans, PIXIE, ExPose, CLIFF, ROMP, etc.) predict
  SMPL/SMPLX parameters directly from image(s) in a single forward pass — milliseconds on a
  GPU, no iterative optimization at all. ExPose specifically outputs SMPLX (body+hands+face)
  the way your pipeline needs.
- **Trade-off, stated plainly**: pure feed-forward regression is generally *less metrically
  accurate* than a fully-converged Chamfer fit to an actual reconstructed mesh — it's
  estimating from 2D image cues, not registering to real 3D geometry your Tripo3D
  reconstruction already gives you. Given you explicitly want max accuracy on the VM, a bare
  swap to pure regression would very likely be a step backward in accuracy, not a neutral
  speed win.
- **Note on the sibling `spin/` directory**: an earlier memory note referenced "SPIN, the
  parent folder" as a regressor source. I checked — the `spin/` parent directory currently
  contains only this `SMPL-Anthropometry` repo; no actual SPIN (or other regressor) code is
  present. The directory name suggests this was planned, not that a working regressor
  already exists here — adding one would be new integration work, not a drop-in swap.
- **The accuracy-preserving version of this idea**: use a regressor only to produce a *good
  initial guess* (betas/pose) in milliseconds, then run the existing Chamfer optimization to
  full convergence starting from that guess instead of from a cold T-pose/zero-betas start.
  Final accuracy is identical (same objective, run to the same convergence), but the
  optimizer needs far fewer steps to get there since it starts near the optimum. This is the
  classic "regress then refine" pattern (the same idea SPIN-style pipelines are built
  around). It's the only option in this report that could plausibly get you an
  order-of-magnitude speedup *without* touching accuracy — but it requires integrating a real
  pretrained regressor (e.g. PIXIE or ExPose checkpoints) and wiring its output as the
  initialization for `fit()`'s `new_params()`, which is a real engineering task, not a config
  change.

---

## 5. Summary / suggested priority if you want to proceed

| # | Option | Speedup | Accuracy risk | Effort |
|---|---|---|---|---|
| 1 | Cache SMPL/SMPLX model in `api.py` | small, free | none | trivial |
| 2 | Early-stop the rigid stage only | small | none (verified plateaued) | small |
| 3 | LBFGS for rigid stage | ~1.4x (that stage only) | none if fully converged | small |
| 4 | Closed-form rigid init (skip gradient-descent direction search) | unverified, likely meaningful | none (closed-form solves same problem exactly) | medium |
| 5 | Batch concurrent requests | throughput only, not latency | none | medium |
| 6 | `torch.compile` | unverified | none (same numerics) | small, try first |
| 7 | Real CUDA GPU + `pytorch3d` fused Chamfer kernel | large, VM-hardware-dependent | none (same math) | medium, needs GPU VM to test |
| 8 | Regress-then-refine (pretrained regressor as init) | potentially large | none if refinement runs to convergence | large — real new component |

None of these individually match the ~7.6x from cutting point count, but stacking
1+2+3+6 costs little and is risk-free; 7 and 8 are where the real order-of-magnitude
gains live if you have GPU hardware or engineering time to invest.
