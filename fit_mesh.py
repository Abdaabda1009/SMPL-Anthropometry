"""
fit_mesh.py — fitting of an SMPL/SMPLX body to an arbitrary 3D mesh
(e.g. a .glb from Meshy AI / Tripo3D), so it can then be measured by
SMPL-Anthropometry.

Why this exists
---------------
SMPL-Anthropometry measures a body via *hardcoded SMPL/SMPLX vertex indices*.
An arbitrary mesh (different vertex count/order) has no correspondence to those
indices, so it cannot be measured directly. This script registers the SMPL/SMPLX
template to your mesh by optimizing its parameters (shape betas + body pose +
global rotation/translation/scale) to minimize a robust Chamfer distance to your
mesh surface. The fitted `betas` are then measured in a clean T-pose via
`from_body_model`.

Accuracy features+
-----------------
* Robust (Huber) point losses: hair, accessories and reconstruction spikes stop
  dragging the shape outward.
* Trimmed target->body term: the worst 5% of target points (hair tufts, props)
  are ignored each step.
* Normal-compatible correspondence: a body point only matches a target point
  whose surface normal roughly agrees, which stops arm points snapping to the
  torso (the classic limb-circumference failure mode).
* Staged sampling: coarse stages run on few points for speed, the final polish
  stage runs dense samples at a low learning rate.
* Assumes the target is a roughly upright, T-/A-posed, full, bare body.
* Meshes from generative recon are not metrically calibrated, so absolute cm
  are arbitrary. Use --height to height-normalize to a real-world height.

Usage
-----
    python fit_mesh.py --input glb/abbe/tripo3d.glb --gender MALE --height 176.50
    python fit_mesh.py --input glb/abbe.glb --model_type smplx --gender male \
                       --height 176
"""

import argparse
import threading

import numpy as np
import torch
import trimesh

from measure import MeasureBody, create_model
from measurement_definitions import STANDARD_LABELS


# trimesh.sample uses numpy's global RNG, so seeding + sampling must not
# interleave across threads (the API runs several fits concurrently).
_SAMPLE_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def load_target_points(path, n_points, seed=0):
    """Load a mesh (glb/obj/ply/...) and sample points + normals on its surface.

    Returns (points (N,3) float32, normals (N,3) float32, mesh).
    """
    scene = trimesh.load(path, process=False)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    try:
        trimesh.repair.fix_normals(mesh)
    except BaseException:
        pass  # winding repair is best-effort; normals weighting degrades gracefully
    with _SAMPLE_LOCK:
        np.random.seed(seed)
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    return np.asarray(pts, dtype=np.float32), normals, mesh


def chamfer_distance(a, b, chunk_size=2000):
    """Symmetric Chamfer distance between point sets a (N,3) and b (M,3).

    Computed in chunks of `a` instead of one torch.cdist(a, b) call: an
    N=10475, M=12000 pair would otherwise materialize a ~500MB (N, M) float32
    matrix just to reduce it down to N+M numbers. Chunking keeps the same
    mathematical result (this is an exact min-reduction, not an approximation)
    at a fraction of the peak memory — relevant since a few of these can be
    in flight at once (see api MAX_WORKERS).
    """
    a_to_b_mins = []
    b_to_a_min = None
    for start in range(0, a.shape[0], chunk_size):
        d_chunk = torch.cdist(a[start:start + chunk_size], b)   # (chunk, M)
        a_to_b_mins.append(d_chunk.min(dim=1).values)
        chunk_b_to_a_min = d_chunk.min(dim=0).values
        b_to_a_min = chunk_b_to_a_min if b_to_a_min is None \
            else torch.minimum(b_to_a_min, chunk_b_to_a_min)
    return torch.cat(a_to_b_mins).mean() + b_to_a_min.mean()


def _huber(d, delta):
    """Elementwise Huber penalty on (positive) distances."""
    return torch.where(d < delta, 0.5 * d * d / delta, d - 0.5 * delta)


def _vertex_normals(v, faces):
    """Area-weighted vertex normals of a triangle mesh, differentiable."""
    fn = torch.linalg.cross(v[faces[:, 1]] - v[faces[:, 0]],
                            v[faces[:, 2]] - v[faces[:, 0]])
    vn = torch.zeros_like(v)
    vn = vn.index_add(0, faces[:, 0], fn)
    vn = vn.index_add(0, faces[:, 1], fn)
    vn = vn.index_add(0, faces[:, 2], fn)
    return torch.nn.functional.normalize(vn, dim=1, eps=1e-8)


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def build_model(model_type, gender):
    """Build a body model, reusing measure.py's cache.

    Deliberately shares one cache with MeasureBody.from_body_model: a fit and the
    measurement that follows it use the same (model_type, gender), and each SMPLX
    model costs a few hundred MB, so two private caches would hold two copies.

    Pinned to device="cpu" regardless of TORCH_DEVICE (measure.py's opt-in GPU
    setting): fit() below creates all of its own tensors — target points,
    pose/shape/scale params, the vertex-normal computation, the sampling
    generator — with no device placement. A GPU-resident model here would
    crash on the first op in forward_verts() that mixes a CUDA model with
    CPU tensors, not just run slower. Migrating fit() to GPU is a real,
    worthwhile speedup for later (that's the whole loop this repo spends
    minutes in), but it means threading a device through every tensor
    created in this function, including the `generator` argument passed in
    from measurement_engine/fit_worker.py — do that as its own change, with
    real GPU hardware to validate the result against, not blindly here.
    """
    return create_model(model_type, "data", gender, num_betas=10, device="cpu")


def forward_verts(model, p):
    """SMPL/SMPLX vertices with an outer similarity transform (scale+rot+transl)."""
    out = model(betas=p["betas"], global_orient=p["global_orient"],
                body_pose=p["body_pose"], return_verts=True)
    v = out.vertices[0]                          # (V, 3)
    return v * torch.exp(p["log_scale"]) + p["transl"]


def fit(model, target_pts, target_normals=None, n_smpl_sample=6000,
        iters_rigid=200, iters_pose=300, iters_full=500, iters_polish=250,
        betas_reg=5e-3, pose_reg=2e-2, verbose=True, generator=None):
    """
    Fit SMPL/SMPLX params to target points in stages:
      0) rigid+scale init (two facing directions),
      1) rigid+scale refine,
      2) + body pose (align limbs; normal-aware matching keeps arms off torso),
      3) + shape betas,
      4) dense low-lr polish of everything.
    `target_normals` (same length as `target_pts`) enables normal-compatible
    matching; pass None to fall back to pure nearest-point matching.
    `generator` is an optional torch.Generator for point subsampling; pass a
    per-call seeded one for thread-safe, reproducible fits (default: global RNG).
    Returns (betas, posed_verts_in_target_space, faces, final_chamfer).
    """
    target = torch.tensor(np.asarray(target_pts), dtype=torch.float32)
    tgt_normals = None
    if target_normals is not None:
        tgt_normals = torch.nn.functional.normalize(
            torch.tensor(np.asarray(target_normals), dtype=torch.float32),
            dim=1, eps=1e-8)
    n_body = model.NUM_BODY_JOINTS  # 23 (SMPL) or 21 (SMPLX)
    faces_t = torch.as_tensor(np.asarray(model.faces, dtype=np.int64))

    # --- initialization: match height + centroid of the template to the target
    with torch.no_grad():
        base = model(betas=torch.zeros(1, 10), return_verts=True).vertices[0]
    smpl_h = (base[:, 1].max() - base[:, 1].min()).item()
    tgt_h = (target[:, 1].max() - target[:, 1].min()).item()
    init_scale = tgt_h / smpl_h
    init_transl = target.mean(0) - base.mean(0) * init_scale
    delta = 0.02 * tgt_h  # Huber threshold: ~2% of body height

    # --- detect arms-down (A-pose) vs arms-out (T-pose) from target proportions.
    tgt_w = (target[:, 0].max() - target[:, 0].min()).item()
    arms_down = tgt_w < 0.6 * tgt_h
    pose_init = torch.zeros(1, n_body * 3)
    if arms_down:
        pose_init[0, 45:48] = torch.tensor([0.0, 0.0, -1.2])  # left shoulder down
        pose_init[0, 48:51] = torch.tensor([0.0, 0.0, 1.2])   # right shoulder down
    if verbose:
        print(f"  target pose: {'arms-down (A-pose)' if arms_down else 'arms-out (T-pose)'}")

    def new_params(yaw):
        return {
            "betas": torch.zeros(1, 10, requires_grad=True),
            "body_pose": pose_init.clone().detach().requires_grad_(True),
            "global_orient": torch.tensor([[0.0, yaw, 0.0]], requires_grad=True),
            "transl": init_transl.clone().detach().requires_grad_(True),
            "log_scale": torch.log(torch.tensor([init_scale])).requires_grad_(True),
        }

    n_verts, n_tgt = base.shape[0], target.shape[0]

    def rand_idx(n, total):
        return torch.randperm(total, generator=generator)[:min(n, total)]

    def loss_fn(p, smpl_idx, tgt_idx, use_normals=False, trim_frac=0.05):
        v = forward_verts(model, p)
        vs = v[smpl_idx]
        d = torch.cdist(vs, target[tgt_idx])
        d_st, nn_st = d.min(dim=1)            # body -> target
        d_ts = d.min(dim=0).values            # target -> body
        l_st = _huber(d_st, delta)
        if use_normals and tgt_normals is not None:
            vn = _vertex_normals(v, faces_t)[smpl_idx]
            tn = tgt_normals[tgt_idx][nn_st]
            # weight in [0.3, 1]: opposing normals (arm matched to torso wall)
            # are down-weighted but never zeroed, so gradients stay alive
            w = 0.3 + 0.7 * (vn * tn).sum(-1).clamp(min=0.0)
            l_st = l_st * w
        if trim_frac > 0.0:
            k = max(1, int(round((1.0 - trim_frac) * d_ts.numel())))
            d_ts = torch.topk(d_ts, k, largest=False).values
        return l_st.mean() + _huber(d_ts, delta).mean()

    # --- try two facing directions (front/back) and keep the better one
    best = None
    si0, ti0 = rand_idx(3000, n_verts), rand_idx(4000, n_tgt)
    for yaw in (0.0, np.pi):
        p = new_params(yaw)
        opt = torch.optim.Adam([p["global_orient"], p["transl"], p["log_scale"]], lr=0.02)
        for _ in range(80):
            opt.zero_grad(); l = loss_fn(p, si0, ti0); l.backward(); opt.step()
        if best is None or l.item() < best[0]:
            best = (l.item(), p)
    p = best[1]

    # --- stage 1: rigid + scale
    si, ti = rand_idx(4000, n_verts), rand_idx(6000, n_tgt)
    opt = torch.optim.Adam([p["global_orient"], p["transl"], p["log_scale"]], lr=0.02)
    for i in range(iters_rigid):
        opt.zero_grad(); l = loss_fn(p, si, ti); l.backward(); opt.step()
        if verbose and i % 50 == 0:
            print(f"  [rigid  {i:4d}] loss={l.item():.5f}")

    # --- stage 2: + body pose (regularized toward T-pose so it only bends limbs)
    si, ti = rand_idx(n_smpl_sample, n_verts), rand_idx(8000, n_tgt)
    opt = torch.optim.Adam([p["body_pose"], p["global_orient"], p["transl"], p["log_scale"]], lr=0.01)
    for i in range(iters_pose):
        opt.zero_grad()
        l = loss_fn(p, si, ti, use_normals=True) \
            + pose_reg * (p["body_pose"] - pose_init).pow(2).mean()
        l.backward(); opt.step()
        if verbose and i % 50 == 0:
            print(f"  [pose   {i:4d}] loss={l.item():.5f}")

    # --- stage 3: + shape betas (regularized so shape stays plausible)
    si, ti = rand_idx(n_smpl_sample, n_verts), rand_idx(10000, n_tgt)
    opt = torch.optim.Adam(list(p.values()), lr=0.01)
    for i in range(iters_full):
        opt.zero_grad()
        l = loss_fn(p, si, ti, use_normals=True) \
            + betas_reg * p["betas"].pow(2).mean() \
            + pose_reg * (p["body_pose"] - pose_init).pow(2).mean()
        l.backward(); opt.step()
        if verbose and i % 50 == 0:
            print(f"  [shape  {i:4d}] loss={l.item():.5f}")

    # --- stage 4: dense low-lr polish (looser pose prior lets limbs settle)
    si, ti = rand_idx(8000, n_verts), rand_idx(n_tgt, n_tgt)
    opt = torch.optim.Adam(list(p.values()), lr=0.003)
    for i in range(iters_polish):
        opt.zero_grad()
        l = loss_fn(p, si, ti, use_normals=True) \
            + 0.5 * betas_reg * p["betas"].pow(2).mean() \
            + 0.25 * pose_reg * (p["body_pose"] - pose_init).pow(2).mean()
        l.backward(); opt.step()
        if verbose and i % 50 == 0:
            print(f"  [polish {i:4d}] loss={l.item():.5f}")

    with torch.no_grad():
        posed_t = forward_verts(model, p)
        posed = posed_t.cpu().numpy()
        # report the plain (non-robust) Chamfer on dense samples so the number
        # stays comparable across runs and with the previous fitter
        final = chamfer_distance(posed_t, target).item()
    return p["betas"].detach(), posed, model.faces, final


def write_overlay(path, target_pts, posed_verts, faces, target_mesh=None, auto_open=False):
    """Write an HTML overlay (fitted body vs target) via plotly. Shared by
    the CLI (below, target_mesh given: plots the full target surface) and
    measurement_engine/overlay.py (target_mesh=None: plots the sampled
    target points only, since the API never keeps the full mesh around).

    Returns `path` (as str) on success; raises on failure — callers that
    want overlay generation to be best-effort catch around this themselves
    (see measurement_engine/overlay.py).
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    if target_mesh is not None:
        tv = np.asarray(target_mesh.vertices)
        tf = np.asarray(target_mesh.faces)
        fig.add_trace(go.Mesh3d(x=tv[:, 0], y=tv[:, 1], z=tv[:, 2],
                                i=tf[:, 0], j=tf[:, 1], k=tf[:, 2],
                                color="lightgray", opacity=0.35, name="target (mesh)"))
    else:
        fig.add_trace(go.Scatter3d(
            x=target_pts[:, 0], y=target_pts[:, 1], z=target_pts[:, 2],
            mode="markers", marker=dict(size=1.5, color="gray"),
            name="target (scan)"))
    fig.add_trace(go.Mesh3d(
        x=posed_verts[:, 0], y=posed_verts[:, 1], z=posed_verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="crimson", opacity=0.55, name="fitted body"))
    fig.update_layout(title="Fit overlay: fitted body (red) vs target (gray)",
                      scene_aspectmode="data")
    fig.write_html(path, include_plotlyjs="cdn", auto_open=auto_open)
    return str(path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fit SMPL/SMPLX to a mesh, then measure it.")
    ap.add_argument("--input", required=True, help="Path to target mesh (.glb/.obj/.ply/...).")
    ap.add_argument("--model_type", default="smplx", choices=["smpl", "smplx"])
    ap.add_argument("--gender", default="NEUTRAL", choices=["MALE", "FEMALE", "NEUTRAL"])
    ap.add_argument("--height", type=float, default=None,
                    help="Real-world height in cm to normalize measurements to "
                         "(recommended: recon meshes have arbitrary scale).")
    ap.add_argument("--n_target", type=int, default=12000, help="Target surface samples.")
    ap.add_argument("--visualize", action="store_true", help="Open 3D measurement viz.")
    ap.add_argument("--export_fit", default=None, help="Save fitted SMPL mesh to this path.")
    ap.add_argument("--overlay", default=None,
                    help="Write an HTML overlay (fitted SMPL vs target) and open it.")
    ap.add_argument("--save_measurements", default=None,
                    help="Write predicted measurements to this CSV (with a blank 'real_cm' column).")
    args = ap.parse_args()

    import pandas as pd  # CLI-only (tables/CSV); keeps the API import path lean

    torch.manual_seed(0)

    print(f"Loading target mesh: {args.input}")
    target_pts, target_normals, target_mesh = load_target_points(args.input, args.n_target)
    print(f"  sampled {len(target_pts)} surface points from "
          f"{len(target_mesh.vertices)} verts / {len(target_mesh.faces)} faces")

    print(f"Fitting {args.model_type.upper()} ({args.gender}) to the mesh...")
    model = build_model(args.model_type, args.gender)
    betas, posed_verts, fit_faces, final_loss = fit(model, target_pts,
                                                    target_normals=target_normals)
    tgt_h = target_pts[:, 1].max() - target_pts[:, 1].min()
    print(f"Fit done. Final Chamfer: {final_loss:.5f} (~{final_loss/2/tgt_h*100:.1f}% of height per side)")
    print(f"Fitted betas: {np.round(betas.numpy().ravel(), 3)}")

    # --- optional overlay of the POSED fit against the target (visual QA) ---
    if args.overlay:
        write_overlay(args.overlay, target_pts, posed_verts, fit_faces,
                      target_mesh=target_mesh, auto_open=True)
        print(f"Wrote fit overlay to {args.overlay}")

    # --- measure the fitted shape in clean T-pose ---
    measurer = MeasureBody(args.model_type)
    measurer.from_body_model(gender=args.gender, shape=betas)
    measurer.measure(measurer.all_possible_measurements)
    measurer.label_measurements(STANDARD_LABELS)

    def as_table(d, col):
        return pd.DataFrame({"Measurement": list(d.keys()),
                             col: [round(v, 2) for v in d.values()]})

    print("\n=== Measurements of fitted body (SMPL-native scale, cm) ===")
    print(as_table(measurer.measurements, "Value (cm)").to_string(index=False))

    normalized = None
    if args.height is not None:
        measurer.height_normalize_measurements(args.height)
        normalized = measurer.height_normalized_measurements
        print(f"\n=== Measurements normalized to height = {args.height} cm ===")
        print(as_table(normalized, "Value (cm)").to_string(index=False))
    else:
        print("\n(!) No --height given: values above are at SMPL's arbitrary scale. "
              "Pass e.g. --height 170 for real-world cm.")

    # --- save a CSV for building a real-vs-predicted comparison report ---
    if args.save_measurements:
        names = list(measurer.measurements.keys())
        df = pd.DataFrame({
            "measurement": names,
            "predicted_native_cm": [round(measurer.measurements[n], 2) for n in names],
        })
        if normalized is not None:
            df[f"predicted_norm_{int(args.height)}cm"] = [round(normalized[n], 2) for n in names]
        df["real_cm"] = ""  # fill in your ground-truth values here
        df.to_csv(args.save_measurements, index=False)
        print(f"\nSaved measurements CSV to {args.save_measurements} "
              f"(add your ground truth in the 'real_cm' column).")

    # --- optional: export the fitted SMPL mesh for visual QA overlay ---
    if args.export_fit:
        verts = measurer.verts
        fitted = trimesh.Trimesh(vertices=verts, faces=measurer.faces, process=False)
        fitted.export(args.export_fit)
        print(f"\nSaved fitted (T-pose) mesh to {args.export_fit}")

    if args.visualize:
        print("\nOpening 3D measurement visualization in your browser...")
        measurer.visualize()


if __name__ == "__main__":
    main()
